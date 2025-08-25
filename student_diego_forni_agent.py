import sys
import os
import heapq
import random
from typing import Optional, List, Tuple, Dict, Callable

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from base_agent import BaseAgent


class StudentChatGPTAgent(BaseAgent):
    """Agent that uses a genetic algorithm to plan a cleaning tour.

    The agent retrieves the global state of the environment and models the
    cleaning task as a Travelling Salesperson Problem (TSP) where each dirt
    cell must be visited once. A genetic algorithm optimizes the order of
    visitation, and A* search generates the movement actions between points.
    If the global state cannot be retrieved the agent falls back to a
    serpentine sweep across the grid.
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

    # ------------------------------------------------------------------
    def get_strategy_description(self) -> str:
        return (
            "Plans a full cleaning route using a genetic algorithm that solves a TSP "
            "over the global grid. If the global state is unavailable, performs a "
            "serpentine sweep as a fallback strategy."
        )

    # ------------------------------------------------------------------
    # GA helpers
    def _ordered_crossover(self, p1: List[int], p2: List[int]) -> List[int]:
        a, b = sorted(random.sample(range(len(p1)), 2))
        child = [None] * len(p1)
        child[a:b] = p1[a:b]
        pos = b
        for gene in p2:
            if gene not in child:
                if pos >= len(p1):
                    pos = 0
                child[pos] = gene
                pos += 1
        return child

    def _mutate(self, tour: List[int]) -> None:
        i, j = random.sample(range(len(tour)), 2)
        tour[i], tour[j] = tour[j], tour[i]

    def _evaluate(self, order: List[int], dist: List[List[int]]) -> int:
        cost = 0
        prev = 0  # start node index
        for idx in order:
            cost += dist[prev][idx + 1]
            prev = idx + 1
        return cost

    # ------------------------------------------------------------------
    def _a_star_path(
        self, grid: List[List[int]], start: Tuple[int, int], goal: Tuple[int, int]
    ) -> List[Callable[[], bool]]:
        """Return actions for a shortest path between two cells using A*."""
        if start == goal:
            return []

        width, height = len(grid[0]), len(grid)
        moves: List[Tuple[Callable[[], bool], Tuple[int, int]]] = [
            (self.up, (0, -1)),
            (self.down, (0, 1)),
            (self.left, (-1, 0)),
            (self.right, (1, 0)),
        ]

        def h(pos: Tuple[int, int]) -> int:
            return abs(pos[0] - goal[0]) + abs(pos[1] - goal[1])

        pq: List[Tuple[int, Tuple[int, int], List[Callable[[], bool]]]] = []
        heapq.heappush(pq, (h(start), start, []))
        best_g: Dict[Tuple[int, int], int] = {start: 0}

        while pq:
            f_cost, pos, path = heapq.heappop(pq)
            g_cost = len(path)
            if g_cost > best_g.get(pos, float("inf")):
                continue
            if pos == goal:
                return path

            x, y = pos
            for action, (dx, dy) in moves:
                nx, ny = x + dx, y + dy
                if not (0 <= nx < width and 0 <= ny < height):
                    continue
                if grid[ny][nx] == -1:  # treat -1 as obstacle if present
                    continue
                new_g = g_cost + 1
                if new_g >= best_g.get((nx, ny), float("inf")):
                    continue
                best_g[(nx, ny)] = new_g
                new_path = path + [action]
                heapq.heappush(
                    pq, (new_g + h((nx, ny)), (nx, ny), new_path)
                )
        return []

    # ------------------------------------------------------------------
    def _ga_plan_to_clean(self) -> Optional[List[Callable[[], bool]]]:
        """Plan a full route to clean all dirt using a GA-based TSP solver."""
        try:
            state = self.get_environment_state()
        except Exception:
            return None
        if not state:
            return None

        grid: List[List[int]] = state.get("grid")
        start_pos_raw = state.get("agent_position")
        if grid is None or start_pos_raw is None:
            return None

        # "agent_position" may be provided as a mutable list coming from JSON.
        # Convert it to a tuple so it can safely be used as a dictionary key
        # in the A* search (which stores visited cells in a hash map).
        start_pos: Tuple[int, int] = tuple(start_pos_raw)

        dirty: List[Tuple[int, int]] = [
            (x, y)
            for y, row in enumerate(grid)
            for x, cell in enumerate(row)
            if cell == 1
        ]
        if not dirty:
            return []

        # Pre-compute Manhattan distances between all nodes (start + dirt)
        nodes = [start_pos] + dirty
        dist = [[0] * len(nodes) for _ in nodes]
        for i, (x1, y1) in enumerate(nodes):
            for j, (x2, y2) in enumerate(nodes):
                dist[i][j] = abs(x1 - x2) + abs(y1 - y2)

        # Genetic algorithm setup
        population_size = 80
        generations = 120
        mutation_rate = 0.2
        num_targets = len(dirty)

        # Population: permutations of dirt indices
        population: List[List[int]] = [
            random.sample(range(num_targets), num_targets)
            for _ in range(population_size)
        ]
        best = min(population, key=lambda o: self._evaluate(o, dist))

        for _ in range(generations):
            def select() -> List[int]:
                contenders = random.sample(population, 3)
                contenders.sort(key=lambda o: self._evaluate(o, dist))
                return contenders[0]

            new_population: List[List[int]] = []
            while len(new_population) < population_size:
                p1, p2 = select(), select()
                child = self._ordered_crossover(p1, p2)
                if random.random() < mutation_rate:
                    self._mutate(child)
                new_population.append(child)
            population = new_population
            best = min(population + [best], key=lambda o: self._evaluate(o, dist))

        # Build path of actions using A*
        actions: List[Callable[[], bool]] = []
        current = start_pos
        for idx in best:
            goal = dirty[idx]
            segment = self._a_star_path(grid, current, goal)
            actions.extend(segment)
            current = goal
        return actions

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
            path = self._ga_plan_to_clean()
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
    performance = run_agent_simulation(verbose=True)