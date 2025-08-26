import sys
import os
import random
from collections import deque
from itertools import combinations

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from base_agent import BaseAgent


class GeneticTSPAgent(BaseAgent):
    """Agent that plans a near‑optimal tour over dirty cells.

    Key improvements vs the original:
    - Safer fallback: never plan a TSP over *all* cells if dirt is unknown. Uses a serpentine sweep in that case.
    - Robustness: on a blocked move, mark a wall and re‑plan from current position.
    - Exact solver when small: Held‑Karp DP for <= ``held_karp_threshold`` targets.
    - Better GA: tournament selection, seeded population (nearest‑neighbor), 2‑opt local search (memetic GA), early‑stopping, adaptive mutation.
    - Parameterized: tune GA size/rates from __init__.
    - Caching: on‑demand BFS caching; re-used across replans.
    """

    def __init__(
        self,
        server_url="http://localhost:5000",
        *,
        population_size=600,
        generations=80,
        elite=8,
        mutation_rate=0.15,
        tournament_k=5,
        two_opt_rate=0.25,
        held_karp_threshold=11,
        replan_on_block=True,
        rng_seed=None,
        **kwargs,
    ):
        super().__init__(server_url, "GeneticTSPAgent", **kwargs)
        # Execution state
        self.path = []  # list[(x,y)] of *unit* steps to follow
        self.path_index = 0
        self.dimensions = None
        self.grid = None  # optional full grid if environment exposes it
        self.initialized = False
        # Caches
        self._dist_cache = {}  # {src: {dst: dist}}
        self._parent_cache = {}  # {src: {cell: parent}}
        self._known_walls = set()
        # GA params
        self.population_size = int(population_size)
        self.generations = int(generations)
        self.elite = int(elite)
        self.mutation_rate = float(mutation_rate)
        self.tournament_k = int(tournament_k)
        self.two_opt_rate = float(two_opt_rate)
        self.held_karp_threshold = int(held_karp_threshold)
        self.replan_on_block = bool(replan_on_block)
        # RNG
        if rng_seed is not None:
            random.seed(rng_seed)

    # ------------------------------------------------------------------
    def get_strategy_description(self):
        return (
            "Plans a tour over dirty cells using (1) exact Held‑Karp DP for small sets "
            "and (2) a memetic Genetic Algorithm (tournament selection, OX crossover, "
            "swap/inversion mutation, and 2‑opt refinement) for larger sets. Distances "
            "are true shortest‑path lengths from BFS over obstacles. Falls back to a "
            "serpentine sweep if the environment doesn't expose dirt locations. "
            "Learns about new walls at runtime and replans when blocked."
        )

    # ------------------------------------------------------------------
    def think(self):
        if not self.is_connected():
            return False

        p = self.get_perception()
        if not p or p.get("is_finished", True):
            return False

        pos = tuple(p.get("position", (0, 0)))
        is_dirty = p.get("is_dirty", False)

        # Clean current tile if dirty
        if is_dirty:
            return self.suck()

        # First-time setup
        if not self.initialized:
            self._initialize_world(p, start=pos)
            self._plan_route_from(pos)
            self.initialized = True

        # No route? idle
        if self.path_index >= len(self.path):
            return self.idle()

        next_cell = self.path[self.path_index]
        dx = next_cell[0] - pos[0]
        dy = next_cell[1] - pos[1]

        if dx == 0 and dy == 0:
            self.path_index += 1
            return self.idle()

        move_fn = None
        if dx > 0:
            move_fn = self.right
        elif dx < 0:
            move_fn = self.left
        elif dy > 0:
            move_fn = self.down
        elif dy < 0:
            move_fn = self.up

        # Try to move
        moved = move_fn() if move_fn else False
        if moved:
            return True

        # Movement failed → treat as wall, replan
        if self.replan_on_block:
            bx, by = next_cell
            self._known_walls.add((bx, by))
            if self.grid and 0 <= by < len(self.grid) and 0 <= bx < len(self.grid[0]):
                # Mark as wall if grid is known
                self.grid[by][bx] = -1
            # Invalidate caches impacted by the new wall
            self._dist_cache.clear()
            self._parent_cache.clear()
            self._plan_route_from(pos)
            return self.idle()

        # Fallback: skip this step
        self.path_index += 1
        return self.idle()

    # ------------------------------------------------------------------
    # Initialization helpers
    def _initialize_world(self, p, start):
        """Try to fetch full grid + size; otherwise set dimensions and keep grid None."""
        try:
            state = self.get_environment_state()
        except Exception:
            state = None

        if state and state.get("grid") and state.get("agent_position"):
            grid = state["grid"]
            _start = tuple(state["agent_position"])  # prefer env origin if provided
            self.dimensions = (len(grid[0]), len(grid))
            self.grid = grid
            # Keep the actual runtime start (where we are now)
        else:
            # Use explicit perception if available
            sx, sy = p.get("max_x"), p.get("max_y")
            if sx is not None and sy is not None:
                self.dimensions = (int(sx), int(sy))
                self.grid = None  # unknown dirt; don't assume contents
            else:
                # Conservative fallback: 8x8 dimensions, unknown contents
                self.dimensions = (8, 8)
                self.grid = None

    # ------------------------------------------------------------------
    # Planning
    def _plan_route_from(self, start):
        """Plan a step-by-step route from current position to visit all *known* dirty cells.
        If we don't know dirt locations (no grid), plan a serpentine sweep.
        """
        width, height = self.dimensions

        if self.grid is None:
            # We don't know where dirt is → sweep deterministically
            self.path = self._serpentine_sweep(width, height, start)
            self.path_index = 0
            if getattr(self, "debug", False):
                print("[GeneticTSPAgent] Using serpentine sweep (unknown dirt).")
            return

        # Extract reachable dirt nodes
        nodes = [
            (x, y)
            for y, row in enumerate(self.grid)
            for x, cell in enumerate(row)
            if cell == 1 and (x, y) != start
        ]

        # Nothing to do
        if not nodes:
            self.path = [start]
            self.path_index = 0
            return

        # Precompute BFS (distances/parents) from all relevant nodes + start
        all_sources = nodes + [start]
        self._precompute_shortest_paths(all_sources)

        # If any pair is unreachable, drop unreachable nodes and warn
        nodes = [n for n in nodes if n in self._dist_cache[start]]
        if not nodes:
            self.path = [start]
            self.path_index = 0
            return

        # Solve order: exact for small, GA for large
        if len(nodes) <= self.held_karp_threshold:
            order = self._solve_order_held_karp(start, nodes)
        else:
            order = self._solve_order_ga(start, nodes)

        # Stitch full step-by-step path using parent chains
        full = [start]
        cur = start
        for node in order:
            seg = self._reconstruct_path(cur, node)
            if not seg:
                # If something became unreachable, replan entirely.
                if getattr(self, "debug", False):
                    print(f"[GeneticTSPAgent] Segment {cur}->{node} unreachable; replanning.")
                self._dist_cache.clear(); self._parent_cache.clear()
                return self._plan_route_from(start)
            full.extend(seg)
            cur = node

        self.path = full
        self.path_index = 0

    # ------------------------------------------------------------------
    # Exact TSP path (start→visit all, no return) using Held‑Karp DP
    def _solve_order_held_karp(self, start, nodes):
        idx = {v: i for i, v in enumerate(nodes)}
        # dp[(mask,last_idx)] = (cost, prev_last_idx)
        dp = {(1 << i, i): (self._dist(start, nodes[i]), -1) for i in range(len(nodes))}
        for r in range(2, len(nodes) + 1):
            for subset in combinations(range(len(nodes)), r):
                mask = 0
                for i in subset:
                    mask |= (1 << i)
                for j in subset:
                    pmask = mask ^ (1 << j)
                    best = (float("inf"), -1)
                    for i in subset:
                        if i == j:
                            continue
                        prev_cost, _ = dp[(pmask, i)]
                        cand = prev_cost + self._dist(nodes[i], nodes[j])
                        if cand < best[0]:
                            best = (cand, i)
                    dp[(mask, j)] = best
        # Recover best end
        full_mask = (1 << len(nodes)) - 1
        end_j, best_cost = None, float("inf")
        for j in range(len(nodes)):
            c, _ = dp[(full_mask, j)]
            if c < best_cost:
                best_cost, end_j = c, j
        # Reconstruct order
        order_idx = []
        mask, j = full_mask, end_j
        while j != -1:
            order_idx.append(j)
            c, i = dp[(mask, j)]
            if i == -1:
                break
            mask ^= (1 << j)
            j = i
        order_idx.reverse()
        return [nodes[i] for i in order_idx]

    # ------------------------------------------------------------------
    # Genetic Algorithm order solver (memetic GA)
    def _solve_order_ga(self, start, nodes):
        def path_length(order):
            total, prev = 0, start
            for n in order:
                total += self._dist(prev, n)
                prev = n
            return total

        # Seed: nearest‑neighbor tours from multiple random starts
        seeds = []
        for _ in range(min(16, len(nodes))):
            seeds.append(self._nearest_neighbor_order(start, nodes))
        # Random permutations to fill
        while len(seeds) < self.elite:
            seeds.append(random.sample(nodes, len(nodes)))

        population = list(seeds) + [random.sample(nodes, len(nodes)) for _ in range(max(0, self.population_size - len(seeds)))]

        best_order = min(population, key=path_length)
        best_cost = path_length(best_order)
        no_improve = 0
        mut = self.mutation_rate

        for gen in range(self.generations):
            # Score current population
            scored = sorted(((path_length(p), p) for p in population), key=lambda x: x[0])
            gen_best = scored[0][0]
            gen_worst = scored[-1][0]
            gen_mean = sum(d for d, _ in scored) / len(scored)
            gen_median = scored[len(scored)//2][0]

            # Track global best
            prev_global = best_cost
            if gen_best < best_cost:
                best_cost, best_order = scored[0]
                no_improve = 0
            else:
                no_improve += 1
            improve = prev_global - best_cost

            # Build next generation
            new_pop = [p for _, p in scored[: self.elite]]  # elitism

            # Tournament selection
            def select():
                cand = random.sample(scored[: max(64, self.population_size)], k=min(self.tournament_k, len(scored)))
                return min(cand, key=lambda x: x[0])[1]

            mutated = 0
            two_opt_applied = 0
            while len(new_pop) < self.population_size:
                p1, p2 = select(), select()
                child = self._ordered_crossover(p1, p2)
                if random.random() < mut:
                    self._mutate_swap_or_invert(child)
                    mutated += 1
                if random.random() < self.two_opt_rate:
                    child = self._two_opt(child, path_length)
                    two_opt_applied += 1
                new_pop.append(child)

            # Per‑generation print (always on)
            print(
                f"[GA] gen={gen:03d} "
                f"best={gen_best:.2f} mean={gen_mean:.2f} median={gen_median:.2f} worst={gen_worst:.2f} "
                f"global_best={best_cost:.2f} Δ={improve:.2f} mut={mut:.2f} "
                f"offspring_mutated={mutated} two_opt={two_opt_applied}"
            )

            # Update population
            population = new_pop

            # Early stopping if stagnated
            if no_improve >= 10:
                if mut < 0.6:
                    mut = min(0.6, mut * 1.5)  # adaptive mutation bump
                    no_improve = 0
                else:
                    break

        return self._two_opt(best_order, path_length)

    # ------------------------------------------------------------------
    # Local helpers
    def _nearest_neighbor_order(self, start, nodes):
        remaining = set(nodes)
        cur = start
        order = []
        while remaining:
            nxt = min(remaining, key=lambda n: self._dist(cur, n))
            order.append(nxt)
            remaining.remove(nxt)
            cur = nxt
        return order

    def _ordered_crossover(self, p1, p2):
        a, b = sorted(random.sample(range(len(p1)), 2))
        child = [None] * len(p1)
        child[a:b] = p1[a:b]
        fill = [node for node in p2 if node not in child]
        it = iter(fill)
        for i in range(len(child)):
            if child[i] is None:
                child[i] = next(it)
        return child

    def _mutate_swap_or_invert(self, path):
        if len(path) < 2:
            return
        if random.random() < 0.5:
            i, j = sorted(random.sample(range(len(path)), 2))
            path[i:j] = reversed(path[i:j])  # inversion
        else:
            i, j = random.sample(range(len(path)), 2)
            path[i], path[j] = path[j], path[i]  # swap

    # ------------------------------------------------------------------
    # Graph distances via BFS
    def _dist(self, a, b):
        if a not in self._dist_cache:
            dist, parent = self._bfs(a)
            self._dist_cache[a] = dist
            self._parent_cache[a] = parent
        return self._dist_cache[a].get(b, float("inf"))

    def _bfs(self, start):
        width, height = self.dimensions
        q = deque([start])
        dist = {start: 0}
        parent = {start: None}
        sx, sy = start
        while q:
            x, y = q.popleft()
            for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nx, ny = x + dx, y + dy
                if not (0 <= nx < width and 0 <= ny < height):
                    continue
                if (nx, ny) in dist:
                    continue
                # If a known grid exists, respect walls/blocked
                if self.grid is not None:
                    cell = self.grid[ny][nx]
                    if cell in (-1, 2):
                        continue
                # Also respect dynamically learned walls
                if (nx, ny) in self._known_walls:
                    continue
                dist[(nx, ny)] = dist[(x, y)] + 1
                parent[(nx, ny)] = (x, y)
                q.append((nx, ny))
        return dist, parent

    def _precompute_shortest_paths(self, nodes):
        for node in nodes:
            if node not in self._dist_cache:
                dist, parent = self._bfs(node)
                self._dist_cache[node] = dist
                self._parent_cache[node] = parent

    def _reconstruct_path(self, start, goal):
        if start not in self._parent_cache:
            dist, parent = self._bfs(start)
            self._dist_cache[start] = dist
            self._parent_cache[start] = parent
        parent = self._parent_cache[start]
        if goal not in parent:
            return []
        path = []
        cur = goal
        while cur != start:
            path.append(cur)
            cur = parent[cur]
        path.reverse()
        return path

    # ------------------------------------------------------------------
    # 2‑opt local search on an *order* over nodes
    def _two_opt(self, route, length_fn):
        if len(route) < 4:
            return route
        best = route
        best_len = length_fn(best)
        improved = True
        while improved:
            improved = False
            for i in range(1, len(best) - 2):
                for j in range(i + 1, len(best)):
                    if j - i == 1:
                        continue
                    new_route = best[:i] + best[i:j][::-1] + best[j:]
                    new_len = length_fn(new_route)
                    if new_len < best_len:
                        best = new_route
                        best_len = new_len
                        improved = True
        return best

    # ------------------------------------------------------------------
    # Fallback serpentine sweep if dirt unknown
    def _serpentine_sweep(self, width, height, start):
        # Build full serpentine path over the grid, then BFS from start to the first cell
        sweep = []
        for y in range(height):
            xs = range(width) if y % 2 == 0 else range(width - 1, -1, -1)
            for x in xs:
                sweep.append((x, y))
        # Find the first target and stitch
        if start in sweep:
            start_idx = sweep.index(start)
        else:
            # BFS to the closest sweep cell
            dist, parent = self._bfs(start)
            target = min(sweep, key=lambda c: dist.get(c, float("inf")))
            prefix = self._reconstruct_path(start, target)
            return [start] + prefix + sweep[sweep.index(target) + 1 :]
        return sweep[start_idx:]


if __name__ == "__main__":
    agent = GeneticTSPAgent(
        enable_ui=True,
        live_stats=True,
        debug=True,
        # You can tune these:
        population_size=300,
        generations=20,
        elite=8,
        mutation_rate=0.15,
        tournament_k=5,
        two_opt_rate=0.25,
        held_karp_threshold=11,
        replan_on_block=True,
        rng_seed=42,
    )
    if agent.connect_to_environment():
        perf = agent.run_simulation(verbose=True)
        print(f"Final performance: {perf}")
        agent.disconnect()
