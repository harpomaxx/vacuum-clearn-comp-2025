import os
import sys
from typing import List, Tuple, Callable, Optional, Dict

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from base_agent import BaseAgent

ActionFn = Callable[[], bool]


class ExhaustiveOptimalAgent(BaseAgent):
    """Agent that pursues (near) optimal cleaning order using exhaustive planning.

    Strategy hierarchy (inefficient by design for completeness):
    1. Fetch full global state each time a plan finishes or becomes invalid.
    2. If number of remaining dirt cells N <= HELD_KARP_LIMIT (default 14), compute exact
       shortest Hamiltonian path starting from current agent position visiting all dirt
       (Held-Karp dynamic programming for path TSP). Distances are Manhattan (optimal in open grid).
    3. If N > HELD_KARP_LIMIT, choose the K nearest dirt cells (K=HELD_KARP_LIMIT) to current
       position, solve exact path for them, execute it, then replan for remaining cells.
    4. If global state retrieval fails, fallback to serpentine full coverage sweep.

    This maximizes cleaned cells early and is intentionally heavy in CPU and memory use.
    """

    HELD_KARP_LIMIT = 17  # 2^16 states manageable; increase for more exhaustiveness

    def __init__(self, server_url: str = "http://localhost:5000", **kwargs):
        super().__init__(server_url, "ExhaustiveOptimalAgent", **kwargs)
        self.current_actions: List[ActionFn] = []
        self.last_dirty_signature: Optional[Tuple[Tuple[int, int], ...]] = None
        self.use_global = True

    # ------------------------------------------------------------------
    def get_strategy_description(self) -> str:
        return ("Exact Held-Karp TSP path planning on dirt set (clustered when large); "
                "replans exhaustively; serpentine fallback.")

    # ------------------------------------------------------------------
    def _serpentine_step(self) -> bool:
        state = self.get_environment_state()
        if not state or 'grid' not in state or 'agent_position' not in state:
            return self.idle()
        grid = state['grid']
        w, h = len(grid[0]), len(grid)
        x, y = state['agent_position']
        left_to_right = (y % 2 == 0)
        if left_to_right:
            if x < w - 1: return self.right()
            if y < h - 1: return self.down()
            return self.idle()
        else:
            if x > 0: return self.left()
            if y < h - 1: return self.down()
            return self.idle()

    # ------------------------------------------------------------------
    @staticmethod
    def _manhattan(a: Tuple[int, int], b: Tuple[int, int]) -> int:
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    def _held_karp_path(self, start: Tuple[int, int], targets: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
        """Compute optimal visitation order (excluding start) using Held-Karp path TSP.
        Returns list of target coordinates in visiting order.
        """
        if not targets:
            return []
        n = len(targets)
        # Precompute distance matrix from start and between targets (Manhattan)
        start_d = [self._manhattan(start, t) for t in targets]
        dist = [[0] * n for _ in range(n)]
        for i in range(n):
            xi = targets[i]
            for j in range(n):
                if i == j: continue
                dist[i][j] = self._manhattan(xi, targets[j])
        # dp[mask][i] = (cost, prev_index)
        dp: List[Dict[int, Tuple[int, int]]] = [dict() for _ in range(1 << n)]
        for i in range(n):
            dp[1 << i][i] = (start_d[i], -1)
        full_mask = (1 << n) - 1
        for mask in range(1 << n):
            for last in list(dp[mask].keys()):
                cost, _ = dp[mask][last]
                if mask == full_mask:
                    continue
                for nxt in range(n):
                    if mask & (1 << nxt):
                        continue
                    new_mask = mask | (1 << nxt)
                    new_cost = cost + dist[last][nxt]
                    prev_entry = dp[new_mask].get(nxt)
                    if prev_entry is None or new_cost < prev_entry[0]:
                        dp[new_mask][nxt] = (new_cost, last)
        # Find best terminal
        best_cost = float('inf')
        best_last = -1
        for last, (c, _) in dp[full_mask].items():
            if c < best_cost:
                best_cost = c
                best_last = last
        # Reconstruct path
        order_rev: List[int] = []
        mask = full_mask
        cur = best_last
        while cur != -1:
            order_rev.append(cur)
            entry = dp[mask][cur]
            prev = entry[1]
            mask ^= (1 << cur)
            cur = prev
        order_rev.reverse()
        return [targets[i] for i in order_rev]

    def _build_actions_between(self, a: Tuple[int, int], b: Tuple[int, int]) -> List[ActionFn]:
        actions: List[ActionFn] = []
        dx = b[0] - a[0]
        dy = b[1] - a[1]
        # Move in x then y (deterministic)
        if dx > 0:
            actions.extend([self.right] * dx)
        elif dx < 0:
            actions.extend([self.left] * (-dx))
        if dy > 0:
            actions.extend([self.down] * dy)
        elif dy < 0:
            actions.extend([self.up] * (-dy))
        return actions

    def _compute_plan(self) -> Optional[List[ActionFn]]:
        try:
            state = self.get_environment_state()
        except Exception:
            return None
        if not state:
            return None
        grid = state.get('grid')
        pos = state.get('agent_position')
        if grid is None or pos is None:
            return None
        dirt = [(x, y) for y, row in enumerate(grid) for x, c in enumerate(row) if c == 1]
        if not dirt:
            return []
        start = tuple(pos)
        # Determine planning set
        if len(dirt) <= self.HELD_KARP_LIMIT:
            subset = dirt
        else:
            # Pick nearest K dirt to start
            dirt.sort(key=lambda d: self._manhattan(start, d))
            subset = dirt[:self.HELD_KARP_LIMIT]
        order = self._held_karp_path(start, subset)
        # Append actions
        actions: List[ActionFn] = []
        cur = start
        for target in order:
            actions.extend(self._build_actions_between(cur, target))
            cur = target
        # After subset cleaned, we'll replan for remaining dirt automatically
        return actions

    # ------------------------------------------------------------------
    def think(self) -> bool:
        if not self.is_connected():
            return False
        perception = self.get_perception()
        if not perception or perception.get('is_finished', True):
            return False
        if perception.get('is_dirty', False):
            return self.suck()
        if self.current_actions:
            act = self.current_actions.pop(0)
            return act()
        if self.use_global:
            plan = self._compute_plan()
            if plan is None:
                self.use_global = False
            elif not plan:
                return self.idle()
            else:
                self.current_actions = plan
                act = self.current_actions.pop(0)
                return act()
        return self._serpentine_step()


def run_agent_simulation(size_x: int = 8, size_y: int = 8, dirt_rate: float = 0.3,
                         server_url: str = "http://localhost:5000", verbose: bool = True) -> int:
    agent = ExhaustiveOptimalAgent(server_url)
    try:
        if not agent.connect_to_environment(size_x, size_y, dirt_rate):
            return 0
        performance = agent.run_simulation(verbose)
        return performance
    finally:
        agent.disconnect()


if __name__ == "__main__":
    print("ExhaustiveOptimalAgent (Held-Karp exact planning)")
    print("Ensure the environment server is running on localhost:5000")
    perf = run_agent_simulation(verbose=True)
    print(f"\nFinal performance: {perf}")
