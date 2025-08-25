import sys
import os
import heapq
from typing import Optional, List, Tuple, Dict, Callable

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from base_agent import BaseAgent


class StudentChatGPTAgent(BaseAgent):
    """Agent that uses A* to find the nearest dirt when global state is available.

    If the global state cannot be retrieved, the agent performs a serpentine
    sweep across the grid as a fallback strategy.
    """

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
            "StudentChatGPTAgent",
            enable_ui,
            record_game,
            replay_file,
            cell_size,
            fps,
            auto_exit_on_finish,
            live_stats,
        )
        self.current_path: List[Callable[[], bool]] = []
        self.use_global: bool = True
        self.direction: str = "right"

    def get_strategy_description(self) -> str:
        return (
            "Plans paths to the nearest dirt using A* search over the global grid. "
            "If the global state cannot be accessed, falls back to a serpentine sweep"
            " of the environment."
        )

    # ------------------------------------------------------------------
    def _a_star_to_nearest_dirt(self) -> Optional[List[Callable[[], bool]]]:
        """Return a sequence of actions to the closest dirt using A* search.

        Returns None if the global state is unavailable. Returns an empty list if
        there is no dirt left in the grid.
        """
        try:
            state = self.get_environment_state()
        except Exception:
            return None
        if not state:
            return None

        grid: List[List[int]] = state.get("grid")
        start_pos: Tuple[int, int] = state.get("agent_position")
        if grid is None or start_pos is None:
            return None

        dirty: List[Tuple[int, int]] = [
            (x, y)
            for y, row in enumerate(grid)
            for x, cell in enumerate(row)
            if cell == 1
        ]
        if not dirty:
            return []

        width, height = len(grid[0]), len(grid)
        start = tuple(start_pos)

        def h_cost(pos: Tuple[int, int]) -> int:
            x, y = pos
            return min(abs(x - dx) + abs(y - dy) for dx, dy in dirty)

        pq: List[Tuple[int, Tuple[int, int], List[Callable[[], bool]]]] = []
        heapq.heappush(pq, (h_cost(start), start, []))
        best_g: Dict[Tuple[int, int], int] = {start: 0}
        parents: Dict[Tuple[int, int], Tuple[Tuple[int, int], Callable[[], bool]]] = {
            start: (None, None)
        }

        moves: List[Tuple[Callable[[], bool], Tuple[int, int]]] = [
            (self.up, (0, -1)),
            (self.down, (0, 1)),
            (self.left, (-1, 0)),
            (self.right, (1, 0)),
        ]

        while pq:
            f_cost, (x, y), path = heapq.heappop(pq)
            g_cost = len(path)
            if g_cost > best_g.get((x, y), float("inf")):
                continue

            if (x, y) in dirty:
                actions: List[Callable[[], bool]] = []
                cur = (x, y)
                while parents[cur][0] is not None:
                    parent_pos, act = parents[cur]
                    actions.append(act)
                    cur = parent_pos
                actions.reverse()
                return actions

            for action, (dx, dy) in moves:
                nx, ny = x + dx, y + dy
                if not (0 <= nx < width and 0 <= ny < height):
                    continue
                new_g = g_cost + 1
                if new_g >= best_g.get((nx, ny), float("inf")):
                    continue
                best_g[(nx, ny)] = new_g
                parents[(nx, ny)] = ((x, y), action)
                new_path = path + [action]
                heapq.heappush(
                    pq,
                    (new_g + h_cost((nx, ny)), (nx, ny), new_path),
                )
        return []

    # ------------------------------------------------------------------
    def _serpentine_step(self) -> bool:
        if self.direction == "right":
            if self.right():
                return True
            if self.down():
                self.direction = "left"
                return True
            return self.idle()
        if self.left():
            return True
        if self.down():
            self.direction = "right"
            return True
        return self.idle()

    # ------------------------------------------------------------------
    def think(self) -> bool:
        if not self.is_connected():
            return False

        perception = self.get_perception()
        if not perception or perception.get("is_finished", True):
            return False

        if perception.get("is_dirty", False):
            return self.suck()

        if self.current_path:
            action = self.current_path.pop(0)
            return action()

        if self.use_global:
            path = self._a_star_to_nearest_dirt()
            if path is None:
                self.use_global = False
            elif not path:
                return self.idle()
            else:
                self.current_path = path
                action = self.current_path.pop(0)
                return action()

        return self._serpentine_step()


# ----------------------------------------------------------------------
def run_agent_simulation(
    size_x: int = 8,
    size_y: int = 8,
    dirt_rate: float = 0.3,
    server_url: str = "http://localhost:5000",
    verbose: bool = True,
) -> int:
    agent = StudentChatGPTAgent(server_url)
    try:
        if not agent.connect_to_environment(size_x, size_y, dirt_rate):
            return 0
        performance = agent.run_simulation(verbose)
        return performance
    finally:
        agent.disconnect()


if __name__ == "__main__":
    print("StudentChatGPTAgent using A* search")
    print("Ensure the environment server is running on localhost:5000")
    performance = run_agent_simulation(verbose=True)
    print(f"\nFinal performance: {performance}")
