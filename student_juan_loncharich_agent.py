import sys
import os
from collections import deque
from typing import List, Optional, Tuple

# Add parent directory to import base_agent from main project
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from base_agent import BaseAgent


class CompetitiveAgent(BaseAgent):
    """An agent that uses global planning to clean optimally.

    The agent keeps a full map of the environment (when available) and
    computes the shortest path to the nearest dirty cell using BFS that takes
    walls/obstacles into account.  If the global state cannot be accessed, it
    falls back to a serpentine sweep similar to how a human would vacuum a
    room, ensuring full coverage.
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
            "CompetitiveAgent",
            enable_ui,
            record_game,
            replay_file,
            cell_size,
            fps,
            auto_exit_on_finish,
            live_stats,
        )

        self._planned_path: List[str] = []
        self._use_global: bool = True
        self._direction: str = "right"  # used in serpentine fallback

    # ------------------------------------------------------------------
    def get_strategy_description(self) -> str:  # pragma: no cover - simple string
        return (
            "BFS global planner that always moves along the shortest path to the"
            " nearest dirt. Falls back to a serpentine sweep if global state is"
            " unavailable."
        )

    # ------------------------------------------------------------------
    def _bfs_to_nearest_dirt(
        self, start: Tuple[int, int], grid: List[List[int]]
    ) -> List[str]:
        """Return list of actions to reach the closest dirty cell.

        Obstacles are considered impassable if the grid contains -1 or None.
        The returned list contains strings representing movement methods
        ('up', 'down', 'left', 'right').  An empty list means no dirt found.
        """

        h = len(grid)
        w = len(grid[0]) if h else 0
        visited = set([start])
        queue = deque([(start, [])])
        while queue:
            (x, y), path = queue.popleft()
            # Check if current cell has dirt
            if grid[y][x] in (1, "1"):
                return path
            for dx, dy, action in (
                (0, -1, "up"),
                (0, 1, "down"),
                (-1, 0, "left"),
                (1, 0, "right"),
            ):
                nx, ny = x + dx, y + dy
                if 0 <= nx < w and 0 <= ny < h and (nx, ny) not in visited:
                    cell = grid[ny][nx]
                    if cell not in (-1, None, "X"):
                        visited.add((nx, ny))
                        queue.append(((nx, ny), path + [action]))
        return []

    # ------------------------------------------------------------------
    def _serpentine_step(self) -> bool:
        """Perform one step of a serpentine sweep pattern."""
        if self._direction == "right":
            move_horizontal = self.right
            reverse = "left"
        else:
            move_horizontal = self.left
            reverse = "right"

        if move_horizontal():
            return True
        if self.down():
            self._direction = reverse
            return True
        return self.idle()

    # ------------------------------------------------------------------
    def think(self) -> bool:
        """Main decision loop executed at each time step."""
        if not self.is_connected():
            return False

        perception = self.get_perception()
        if not perception or perception.get("is_finished", False):
            return False

        if perception.get("is_dirty", False):
            self._planned_path.clear()
            return self.suck()

        # Follow planned path if available
        if self._planned_path:
            action = self._planned_path.pop(0)
            return getattr(self, action)()

        # Try to use global state to plan a path to the nearest dirt
        if self._use_global:
            try:
                state = self.get_environment_state()
            except Exception:
                state = None
            if state and state.get("grid") and state.get("agent_position"):
                grid = state["grid"]
                start = tuple(state["agent_position"])
                path = self._bfs_to_nearest_dirt(start, grid)
                if path:
                    self._planned_path = path
                    action = self._planned_path.pop(0)
                    return getattr(self, action)()
                else:
                    # No dirt left; remain idle
                    return self.idle()
            else:
                self._use_global = False

        # Fallback: serpentine sweep
        return self._serpentine_step()


# ----------------------------------------------------------------------
def run_agent_simulation(
    size_x: int = 8,
    size_y: int = 8,
    dirt_rate: float = 0.3,
    server_url: str = "http://localhost:5000",
    verbose: bool = True,
) -> int:
    """Utility to run the agent in a simulation for manual testing."""
    agent = CompetitiveAgent(server_url, enable_ui=verbose, live_stats=verbose)
    try:
        if not agent.connect_to_environment(size_x, size_y, dirt_rate):
            return 0
        return agent.run_simulation(verbose=verbose)
    finally:
        agent.disconnect()


if __name__ == "__main__":
    print("CompetitiveAgent - BFS Optimal Cleaner")
    print("Make sure the environment server is running on localhost:5000")
    print("Strategy: Shortest-path planning using global state with serpentine fallback")
    performance = run_agent_simulation(verbose=True)
    print(f"\nFinal performance: {performance}")