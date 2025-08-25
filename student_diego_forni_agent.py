# student_agents/max_dist.py
import sys
import os
import random
from collections import deque, defaultdict
from typing import Dict, Set, Tuple, Optional, Callable, List

# Make base_agent importable like in your template
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from base_agent import BaseAgent  # noqa: E402

Coord = Tuple[int, int]
MoveFn = Callable[[], bool]


class max_distAgent(BaseAgent):
    """
    Vacuum World agent with:
      - Strict UNVISITED preference + information-gain (frontier) scoring
      - Learned walls (when a move doesn't change position)
      - Loop avoidance (short-term history + light pheromone penalty)
      - Inertia (keep direction if not harmful)
      - BFS recovery through known space to the nearest frontier

    Complies with your environment:
      - implements think() and get_strategy_description()
      - uses BaseAgent's up/down/left/right, suck(), idle()
      - does NOT assume internal position; always reads from perception
    """

    # -------------------------------------------------------------------------
    # Init
    # -------------------------------------------------------------------------
    def __init__(self, server_url="http://localhost:5000", debug=True, **kwargs):
        super().__init__(server_url, "max_distAgent", **kwargs)
        self.debug: bool = bool(debug)

        # Canonical bound movement functions captured ONCE to allow reliable equality checks
        self._UP, self._DOWN, self._LEFT, self._RIGHT = self.up, self.down, self.left, self.right
        self._moves: List[MoveFn] = [self._UP, self._DOWN, self._LEFT, self._RIGHT]

        # Deltas keyed by the canonical move fns (x,y) with y increasing downwards
        self.delta_map: Dict[MoveFn, Tuple[int, int]] = {
            self._UP:    (0, -1),
            self._DOWN:  (0,  1),
            self._LEFT:  (-1, 0),
            self._RIGHT: (1,  0),
        }
        self._delta_to_move: Dict[Tuple[int, int], MoveFn] = {d: m for m, d in self.delta_map.items()}

        # Names for logging
        self._move_name_map = {
            self._UP: "up",
            self._DOWN: "down",
            self._LEFT: "left",
            self._RIGHT: "right",
        }

        # (Optional) action map if your framework needs ids; we only pick functions
        self.action_map = {
            0: self._UP,
            1: self._DOWN,
            2: self._LEFT,
            3: self._RIGHT,
        }

        # Memory
        self.visited: Set[Coord] = set()
        self.visit_counts: Dict[Coord, int] = defaultdict(int)
        self.blocked: Dict[Coord, Set[MoveFn]] = defaultdict(set)  # learned walls per origin

        self.prev_pos: Optional[Coord] = None
        self.last_action: Optional[MoveFn] = None
        self.grid_size: Optional[Tuple[int, int]] = kwargs.get("grid_size")  # (W,H) or None
        self.step: int = 0
        self.history: deque[Coord] = deque(maxlen=12)  # recent positions to detect oscillations

        # Oscillation damping
        self._pheromone: Dict[Coord, float] = defaultdict(float)

        # Scoring weights (tunable)
        self._w = {
            "unvisited_bonus": 3.0,
            "frontier_gain": 1.0,
            "inertia": 0.25,
            "loop_penalty": 1.5,
            "visit_penalty": 0.2,
            "pheromone_penalty": 0.1,
        }

        # Opening: head straight until we hit a wall, then explore
        self._phase: str = "seek_wall"  # "seek_wall" | "explore"
        self._seek_heading: Optional[MoveFn] = None

    # -------------------------------------------------------------------------
    # Required by the environment
    # -------------------------------------------------------------------------
    def get_strategy_description(self):
        return ("Clean if dirty. Otherwise, explore by preferring unvisited tiles with high "
                "frontier exposure (information gain), avoid short loops with a light penalty, "
                "keep inertia when not harmful, learn walls from failed moves, and when stuck, "
                "pathfind through known space to the nearest frontier.")

    def think(self):
        if not self.is_connected():
            return False

        perception = self.get_perception()
        if not perception or perception.get('is_finished', True):
            return False

        pos = perception.get('position', None)  # (x, y)
        is_dirty = perception.get('is_dirty', False)

        # Normalize pos
        if isinstance(pos, (tuple, list)) and len(pos) == 2:
            pos = (int(pos[0]), int(pos[1]))

            # Learn walls from previous step if we failed to move
            if self.prev_pos is not None and self.last_action is not None:
                if pos == self.prev_pos:
                    self._mark_wall(self.prev_pos, self.last_action)
                    if self._phase == "seek_wall":
                        self._phase = "explore"
                        if self.debug:
                            print("[PHASE] Reached wall -> switch to 'explore'")

            # Update loop history & pheromones
            self.history.append(pos)
            for k in list(self._pheromone.keys()):
                self._pheromone[k] *= 0.95
                if self._pheromone[k] < 1e-3:
                    del self._pheromone[k]
            self._pheromone[pos] += 1.0

            # Visited bookkeeping
            self.visited.add(pos)
            self.visit_counts[pos] += 1

        self.step += 1

        if self.debug:
            print(f"\n[STEP {self.step}] pos={pos} dirty={is_dirty}")
            print(f"visited_count={len(self.visited)} here_times={self.visit_counts.get(pos, 0)}")

        # Clean first
        if is_dirty:
            if self.debug:
                print("[DECISION] Dirty: suck()")
            self.last_action = None
            return self.suck()

        # Choose a move
        move = self._choose_move(pos)

        # Execute
        if move is None:
            if self.debug:
                print("[DECISION] No valid move -> idle()")
            self.prev_pos = pos
            self.last_action = None
            return self.idle()

        self.prev_pos = pos
        self.last_action = move
        ok = move()
        if self.debug:
            print(f"[ACTION] {self._move_name(move)}() -> {ok}")
        return ok

    # -------------------------------------------------------------------------
    # Move selection
    # -------------------------------------------------------------------------
    def _choose_move(self, pos: Optional[Coord]) -> Optional[MoveFn]:
        if pos is None or len(self.visited) == 0:
            if self.debug:
                print("[SCORING] bootstrap -> random non-backtracking")
            return self._random_non_backtracking()

        # Opening: go straight until wall
        if self._phase == "seek_wall":
            if self._seek_heading is None:
                open_dirs = [m for m in self._moves if m not in self.blocked.get(pos, set())]
                self._seek_heading = random.choice(open_dirs) if open_dirs else self._random_non_backtracking()
                if self.debug:
                    print(f"[PHASE] seek_wall -> heading {self._move_name(self._seek_heading)}")
            return self._seek_heading

        # Candidates respecting bounds + learned walls
        candidates: List[Tuple[MoveFn, Coord]] = []
        for move_fn, (dx, dy) in self.delta_map.items():
            if move_fn in self.blocked.get(pos, set()):
                if self.debug:
                    print(f"  - {self._move_name(move_fn):>5} REJECT blocked_by_wall")
                continue
            nxt = (pos[0] + dx, pos[1] + dy)
            if self.grid_size is not None and not self._in_bounds(nxt):
                if self.debug:
                    print(f"  - {self._move_name(move_fn):>5} -> nxt={nxt} REJECT out_of_bounds")
                continue
            candidates.append((move_fn, nxt))

        if not candidates:
            if self.debug:
                print("[SCORING] No legal moves from here")
            return None

        # Strictly prefer UNVISITED
        unvisited = [(m, n) for (m, n) in candidates if n not in self.visited]
        pool = unvisited if unvisited else candidates
        if self.debug:
            print(f"[FILTER] {'UNVISITED only' if unvisited else 'no UNVISITED -> any viable'} "
                  f"({len(pool)}/{len(candidates)})")

        baseline_dir = self.last_action  # inertia reference

        def in_short_loop(coord: Coord) -> bool:
            return coord in self.history

        # Score with information gain & penalties
        scored: List[Tuple[float, MoveFn, Coord]] = []
        for move_fn, nxt in pool:
            unknown_local = self._unknown_neighbors_count(nxt)   # ring-1
            frontier2 = self._frontier_score(nxt, radius=2)      # ring-2 discounted
            inertia = 1.0 if (baseline_dir is not None and move_fn is baseline_dir) else 0.0
            loop_pen = 1.0 if in_short_loop(nxt) else 0.0
            visits = self.visit_counts.get(nxt, 0)
            pher = self._pheromone.get(nxt, 0.0)

            score = (
                self._w["unvisited_bonus"] * (1.0 if nxt not in self.visited else 0.0) +
                self._w["frontier_gain"]   * (unknown_local + frontier2) +
                self._w["inertia"]         * inertia -
                self._w["loop_penalty"]    * loop_pen -
                self._w["visit_penalty"]   * visits -
                self._w["pheromone_penalty"] * pher
            )
            if self.debug:
                nm = self._move_name(move_fn)
                print(f"  - {nm:>5} nxt={nxt} "
                      f"unvisited={nxt not in self.visited} "
                      f"unknown_local={unknown_local} frontier2={frontier2:.1f} "
                      f"inertia={inertia} loop_pen={loop_pen} visits={visits} pher={pher:.2f} "
                      f"=> score={score:.2f}")
            scored.append((score, move_fn, nxt))

        scored.sort(key=lambda t: t[0], reverse=True)
        best_score = scored[0][0]

        # If no local info gain, BFS to nearest frontier via visited tiles
        if all(self._unknown_neighbors_count(n) == 0 for _, _, n in scored):
            step = self._bfs_first_step_to_nearest_frontier(pos)
            if step is not None:
                if self.debug:
                    print(f"[RECOVERY] No local gain; BFS toward frontier -> {self._move_name(step)}")
                return step

        # Resolve ties, avoid immediate backtrack when possible
        top = [(s, m, n) for (s, m, n) in scored if abs(s - best_score) < 1e-9]
        if self.last_action:
            opposite = self._opposite_of(self.last_action)
            noback = [(s, m, n) for (s, m, n) in top if m is not opposite]
            if noback:
                top = noback

        move_fn = random.choice(top)[1]
        if self.debug:
            names = ", ".join(self._move_name(m) for _, m, _ in top)
            print(f"[CHOICE] best={best_score:.2f}, tie among {{{names}}} -> picked {self._move_name(move_fn)}")
        return move_fn

    # -------------------------------------------------------------------------
    # Neighborhood & scoring helpers
    # -------------------------------------------------------------------------
    def _neighbors4(self, cell: Coord):
        """Yield (move_fn, nxt) respecting bounds and learned walls."""
        for move_fn, (dx, dy) in self.delta_map.items():
            if move_fn in self.blocked.get(cell, set()):
                continue
            nxt = (cell[0] + dx, cell[1] + dy)
            if self.grid_size is not None and not self._in_bounds(nxt):
                continue
            yield move_fn, nxt

    def _unknown_neighbors_count(self, cell: Coord) -> int:
        """Immediate frontier exposure: neighbors not yet visited."""
        cnt = 0
        for _, nxt in self._neighbors4(cell):
            if nxt not in self.visited:
                cnt += 1
        return cnt

    def _frontier_score(self, cell: Coord, radius: int = 2) -> float:
        """
        Discounted count of frontier tiles in rings 1..radius, expanding ONLY through visited tiles.
        Frontier = visited tile with at least one unvisited neighbor.
        """
        if radius <= 0 or cell not in self.visited:
            return 0.0

        seen = {cell}
        q = deque([(cell, 0)])
        ring1 = ring2 = 0

        def is_frontier(t: Coord) -> bool:
            return t in self.visited and self._unknown_neighbors_count(t) > 0

        while q:
            cur, dist = q.popleft()
            if dist >= radius:
                continue

            for _, nxt in self._neighbors4(cur):
                if nxt in seen:
                    continue
                seen.add(nxt)

                # Expand BFS only through known (visited) space
                if nxt in self.visited:
                    q.append((nxt, dist + 1))

                if is_frontier(nxt):
                    if dist + 1 == 1:
                        ring1 += 1
                    elif dist + 1 == 2:
                        ring2 += 1

        return ring1 + 0.5 * ring2

    def _bfs_first_step_to_nearest_frontier(self, start: Coord) -> Optional[MoveFn]:
        """
        BFS over visited tiles to the nearest frontier (visited tile with an unvisited neighbor).
        Returns the first move_fn from `start` toward that frontier, or None.
        """
        if start not in self.visited:
            return None

        def is_frontier(t: Coord) -> bool:
            return t in self.visited and self._unknown_neighbors_count(t) > 0

        if is_frontier(start):
            return None  # already at a frontier; local scoring will handle

        parent: Dict[Coord, Optional[Coord]] = {start: None}
        q = deque([start])

        while q:
            cur = q.popleft()
            for mfn, nxt in self._neighbors4(cur):
                if nxt in parent:
                    continue
                parent[nxt] = cur

                if is_frontier(nxt):
                    # Reconstruct first step from start -> nxt
                    step = nxt
                    while parent[step] is not None and parent[step] != start:
                        step = parent[step]
                    dx = step[0] - start[0]
                    dy = step[1] - start[1]
                    return self._delta_to_move.get((dx, dy))
                if nxt in self.visited:
                    q.append(nxt)

        return None

    # -------------------------------------------------------------------------
    # Learning walls & utilities
    # -------------------------------------------------------------------------
    def _mark_wall(self, origin: Coord, move_fn: MoveFn):
        """Record a wall in the attempted direction; also mark the reverse on the far side."""
        self.blocked[origin].add(move_fn)
        dx, dy = self.delta_map[move_fn]
        nxt = (origin[0] + dx, origin[1] + dy)
        opp = self._opposite_of(move_fn)
        self.blocked[nxt].add(opp)  # symmetric (even if nxt not yet visited)
        if self.debug:
            print(f"[LEARN] Wall between {origin} --{self._move_name(move_fn)}|{self._move_name(opp)}--> {nxt}")

    def _random_non_backtracking(self) -> MoveFn:
        if self.last_action is None:
            return random.choice(self._moves)
        opp = self._opposite_of(self.last_action)
        options = [m for m in self._moves if m is not opp]
        return random.choice(options) if options else opp

    def _in_bounds(self, pos: Coord) -> bool:
        if self.grid_size is None:
            return True
        x, y = pos
        W, H = self.grid_size
        return 0 <= x < W and 0 <= y < H

    def _opposite_of(self, move_fn: MoveFn) -> MoveFn:
        if move_fn is self._UP:    return self._DOWN
        if move_fn is self._DOWN:  return self._UP
        if move_fn is self._LEFT:  return self._RIGHT
        if move_fn is self._RIGHT: return self._LEFT
        return move_fn

    def _move_name(self, move_fn: MoveFn) -> str:
        return self._move_name_map.get(move_fn, "?")
