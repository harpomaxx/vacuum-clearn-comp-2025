import sys
import os
import heapq
from itertools import combinations
from typing import List, Tuple, Optional, Dict, Callable

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from base_agent import BaseAgent


class CompetitiveAgentV2(BaseAgent):
    """Competitive agent.

    - Multi-goal optimal tour (Held-Karp DP) when number of dirt cells <= MAX_OPT_TOUR.
    - Greedy nearest-neighbor ordering when above threshold, re-planning after each clean.
    - Local A* path reconstruction between successive targets (even though grid has no obstacles;
      retained for extensibility and overhead tolerance per requirement).
    - Serpentine sweep fallback when global state unavailable.
    - Aggressive re-planning triggers on: dirt cleaned, path blocked (should not happen here),
      or dirt set change.

    Intentionally not optimized for performance; emphasizes exhaustive planning quality.
    """

    MAX_OPT_TOUR = 14  # Maximum dirt cells for Held-Karp exact tour

    def __init__(
        self,
        server_url: str = "http://localhost:5000",
        enable_ui: bool = False,
        record_game: bool = False,
        replay_file: Optional[str] = None,
        cell_size: int = 60,
        fps: int = 10,
        auto_exit_on_finish: bool = True,
        live_stats: bool = False,
    ) -> None:
        super().__init__(
            server_url,
            "CompetitiveAgentV2",
            enable_ui,
            record_game,
            replay_file,
            cell_size,
            fps,
            auto_exit_on_finish,
            live_stats,
        )
        self.action_queue: List[Callable[[], bool]] = []
        self.current_tour: List[Tuple[int, int]] = []
        self.last_dirty_signature: Optional[Tuple[Tuple[int, int], ...]] = None
        self.fallback_serpentine_dir: str = "right"
        self.forced_greedy: bool = False

    # ------------------------------------------------------------------
    def get_strategy_description(self) -> str:
        return (
            "Multi-goal optimal (Held-Karp) tour planning with A* segment stitching, "
            "greedy fallback when large, and serpentine sweep without global state."
        )

    # ------------------------------------------------------------------
    def think(self) -> bool:
        if not self.is_connected():
            return False

        perception = self.get_perception()
        if not perception or perception.get("is_finished", True):
            return False

        # Clean immediately if standing on dirt.
        if perception.get("is_dirty", False):
            self._invalidate_plan()
            return self.suck()

        # If we have an action sequence ready, execute next.
        if self.action_queue:
            action = self.action_queue.pop(0)
            return action()

        # Need to plan or re-plan.
        state = self._safe_state()
        if state is None:
            return self._serpentine_step()

        grid = state["grid"]
        agent_pos = state["agent_position"]
        dirt_cells = [(x, y) for y, row in enumerate(grid) for x, v in enumerate(row) if v == 1]
        if not dirt_cells:
            return self.idle()

        dirty_signature = tuple(sorted(dirt_cells))
        if dirty_signature != self.last_dirty_signature:
            self._invalidate_plan(update_signature=False)
            self.last_dirty_signature = dirty_signature

        if not self.current_tour:
            self.current_tour = self._plan_tour(agent_pos, dirt_cells)
            # Build initial action queue from full tour.
            self.action_queue = self._build_action_queue(agent_pos, self.current_tour)

        if not self.action_queue:  # Fallback safety
            return self._serpentine_step()
        action = self.action_queue.pop(0)
        return action()

    # ------------------------------------------------------------------
    def _invalidate_plan(self, update_signature: bool = True):
        self.action_queue.clear()
        self.current_tour.clear()
        if update_signature:
            self.last_dirty_signature = None

    # ------------------------------------------------------------------
    def _safe_state(self) -> Optional[Dict]:
        try:
            state = self.get_environment_state()
        except Exception:
            return None
        if not state:
            return None
        grid = state.get("grid")
        pos = state.get("agent_position")
        if grid is None or pos is None:
            return None
        return {"grid": grid, "agent_position": tuple(pos)}

    # ------------------------------------------------------------------
    def _plan_tour(self, start: Tuple[int, int], dirt: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
        n = len(dirt)
        if n <= self.MAX_OPT_TOUR and not self.forced_greedy:
            tour = self._held_karp(start, dirt)
            if tour:
                return tour
            # If optimal fails, fallback to greedy but keep exhaustive flag off for next re-try
        # Greedy fallback
        self.forced_greedy = n > self.MAX_OPT_TOUR
        return self._greedy_order(start, dirt)

    # ------------------------------------------------------------------
    def _held_karp(self, start: Tuple[int, int], targets: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
        n = len(targets)
        if n == 0:
            return []
        # Precompute distances
        dist_start = [self._manhattan(start, t) for t in targets]
        dist = [[0] * n for _ in range(n)]
        for i in range(n):
            for j in range(n):
                if i != j:
                    dist[i][j] = self._manhattan(targets[i], targets[j])
        # DP table: key (mask, i) -> (cost, prev_index)
        dp: Dict[Tuple[int, int], Tuple[int, int]] = {}
        for i in range(n):
            dp[(1 << i, i)] = (dist_start[i], -1)
        full_mask = (1 << n) - 1
        for r in range(2, n + 1):
            for subset in combinations(range(n), r):
                mask = 0
                for idx in subset:
                    mask |= 1 << idx
                for i in subset:
                    prev_mask = mask & ~(1 << i)
                    if prev_mask == 0:
                        continue
                    best_cost = None
                    best_prev = -1
                    for j in subset:
                        if j == i:
                            continue
                        prev_entry = dp.get((prev_mask, j))
                        if prev_entry is None:
                            continue
                        candidate = prev_entry[0] + dist[j][i]
                        if best_cost is None or candidate < best_cost:
                            best_cost = candidate
                            best_prev = j
                    if best_cost is not None:
                        dp[(mask, i)] = (best_cost, best_prev)
        # Find best end
        end_cost = None
        end_idx = -1
        for i in range(n):
            entry = dp.get((full_mask, i))
            if entry and (end_cost is None or entry[0] < end_cost):
                end_cost = entry[0]
                end_idx = i
        if end_idx == -1:
            return []
        # Reconstruct
        order: List[int] = []
        mask = full_mask
        cur = end_idx
        while cur != -1:
            order.append(cur)
            entry = dp[(mask, cur)]
            prev = entry[1]
            if prev == -1:
                break
            mask &= ~(1 << cur)
            cur = prev
        order.reverse()
        return [targets[i] for i in order]

    # ------------------------------------------------------------------
    def _greedy_order(self, start: Tuple[int, int], targets: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
        remaining = set(targets)
        cur = start
        sequence: List[Tuple[int, int]] = []
        while remaining:
            nxt = min(remaining, key=lambda p: self._manhattan(cur, p))
            sequence.append(nxt)
            remaining.remove(nxt)
            cur = nxt
        return sequence

    # ------------------------------------------------------------------
    def _build_action_queue(self, start: Tuple[int, int], tour: List[Tuple[int, int]]) -> List[Callable[[], bool]]:
        actions: List[Callable[[], bool]] = []
        cur = start
        for target in tour:
            path_actions = self._a_star_path(cur, target)
            actions.extend(path_actions)
            actions.append(self.suck)
            cur = target
        return actions

    # ------------------------------------------------------------------
    def _a_star_path(self, start: Tuple[int, int], goal: Tuple[int, int]) -> List[Callable[[], bool]]:
        # Even without obstacles, keep full A* for overhead and future extensibility.
        if start == goal:
            return []
        open_heap: List[Tuple[int, Tuple[int, int]]] = []
        heapq.heappush(open_heap, (0, start))
        came_from: Dict[Tuple[int, int], Tuple[int, int]] = {}
        g: Dict[Tuple[int, int], int] = {start: 0}

        moves = [(0, -1), (0, 1), (-1, 0), (1, 0)]

        while open_heap:
            _, current = heapq.heappop(open_heap)
            if current == goal:
                break
            cx, cy = current
            for dx, dy in moves:
                nx, ny = cx + dx, cy + dy
                if nx < 0 or ny < 0:
                    continue  # Infinity grid assumption trimmed to quadrant (no bounds known here)
                tentative = g[current] + 1
                if tentative < g.get((nx, ny), 10**9):
                    g[(nx, ny)] = tentative
                    came_from[(nx, ny)] = current
                    f = tentative + self._manhattan((nx, ny), goal)
                    heapq.heappush(open_heap, (f, (nx, ny)))

        # Reconstruct path
        if goal not in g:
            return []  # Should not happen in open grid
        rev: List[Tuple[int, int]] = []
        cur = goal
        while cur != start:
            rev.append(cur)
            cur = came_from[cur]
        rev.reverse()

        # Convert steps to action callables
        action_list: List[Callable[[], bool]] = []
        x, y = start
        for nx, ny in rev:
            if nx == x + 1:
                action_list.append(self.right)
            elif nx == x - 1:
                action_list.append(self.left)
            elif ny == y + 1:
                action_list.append(self.down)
            elif ny == y - 1:
                action_list.append(self.up)
            x, y = nx, ny
        return action_list

    # ------------------------------------------------------------------
    def _serpentine_step(self) -> bool:
        # Basic serpentine fallback when global state can't be fetched.
        if self.fallback_serpentine_dir == "right":
            if self.right():
                return True
            if self.down():
                self.fallback_serpentine_dir = "left"
                return True
            return self.idle()
        else:
            if self.left():
                return True
            if self.down():
                self.fallback_serpentine_dir = "right"
                return True
            return self.idle()

    # ------------------------------------------------------------------
    @staticmethod
    def _manhattan(a: Tuple[int, int], b: Tuple[int, int]) -> int:
        return abs(a[0] - b[0]) + abs(a[1] - b[1])
