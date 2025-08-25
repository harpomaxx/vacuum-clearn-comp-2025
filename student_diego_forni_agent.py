import sys
import os
import random
from collections import deque, defaultdict

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from base_agent import BaseAgent

class reflexiveAgent(BaseAgent):
    def __init__(self, server_url="http://localhost:5000", debug=True, **kwargs):
        super().__init__(server_url, "reflexiveAgent", **kwargs)

        # Movement funcs from BaseAgent
        self.action_map = {
            0: self.up,
            1: self.down,
            2: self.left,
            3: self.right,
        }
        # Deltas for scoring/logging (x,y) with y increasing downwards
        self.delta_map = {
            self.up:    (0, -1),
            self.down:  (0,  1),
            self.left:  (-1, 0),
            self.right: (1,  0),
        }

        # Memory
        self.visited = set()
        self.visit_counts = defaultdict(int)
        self.blocked = defaultdict(set)  # {(x,y): {move_fn, ...}}
        self.last_action = None          # last movement function (for anti-backtrack)
        self.grid_size = None
        self.step = 0
        self.history = deque(maxlen=10)
        self.debug = debug

        # Track last attempted movement to detect “bump”
        self._prev_pos_before_move = None
        self._prev_move_attempt = None

    def get_strategy_description(self):
        return ("Clean if dirty. Otherwise, move randomly but prefer moves that "
                "do NOT reduce distance to explored tiles; among those, pick the farthest. "
                "If a move doesn’t change position, remember that direction as a wall and avoid it.")

    def think(self):
        if not self.is_connected():
            return False

        perception = self.get_perception()
        if not perception or perception.get('is_finished', True):
            return False

        # --- Read percepts from this env ---
        pos = perception.get('position', None)  # (x,y) tuple
        is_dirty = perception.get('is_dirty', False)
        actions_remaining = perception.get('actions_remaining', None)

        # Update memory
        self.step += 1
        if isinstance(pos, (tuple, list)) and len(pos) == 2:
            pos = (int(pos[0]), int(pos[1]))
            self.visited.add(pos)
            self.visit_counts[pos] += 1
            self.history.append(pos)

        # Detect a wall from last attempted move (if any)
        self._update_blocked_from_last_attempt(current_pos=pos)

        # --- Debug header ---
        if self.debug:
            print(f"\n[STEP {self.step}] pos={pos} dirty={is_dirty} actions_left={actions_remaining}")
            print(f"visited_count={len(self.visited)} here_visited_times={self.visit_counts.get(pos, 0)}")
            if pos in self.blocked and self.blocked[pos]:
                names = ", ".join(self._move_name(m) for m in self.blocked[pos])
                print(f"[WALLS] From {pos} blocked: {{{names}}}")

        # Clean first
        if is_dirty:
            if self.debug: print("[DECISION] Dirty: Suck")
            self.last_action = None
            self._clear_prev_move_attempt()  # not a movement
            ok = self.suck()
            if self.debug: print(f"[ACTION] suck() -> {ok}")
            return ok

        # Choose a move (with detailed tracing)
        move = self._choose_move(pos)

        # Execute
        if move is None:
            if self.debug: print("[DECISION] No valid move found -> idle()")
            self._clear_prev_move_attempt()
            return self.idle()

        self.last_action = move
        # Mark that we’re about to try to move from this position
        self._prev_pos_before_move = pos
        self._prev_move_attempt = move

        ok = move()
        if self.debug:
            name = self._move_name(move)
            print(f"[ACTION] {name}() -> {ok}")

        # If the environment already tells us the move failed, mark the wall immediately.
        if ok is False and self._prev_pos_before_move is not None and self._prev_move_attempt is not None:
            if self.debug:
                print(f"[WALL-LEARNED] Immediate failure reported for {self._move_name(self._prev_move_attempt)} "
                      f"from {self._prev_pos_before_move}")
            self._mark_blocked(self._prev_pos_before_move, self._prev_move_attempt)
            self._clear_prev_move_attempt()

        return ok

    # ------------------ Move selection with tracing ------------------

    def _choose_move(self, pos):
        if pos is None or len(self.visited) == 0:
            if self.debug: print("[SCORING] No position/visited info -> random among all moves (avoid immediate backtrack)")
            return self._random_non_backtracking()

        # Build candidate list (respect bounds if known)
        raw_candidates = []
        for move_fn, d in self.delta_map.items():
            nxt = (pos[0] + d[0], pos[1] + d[1])
            inb = self._in_bounds(nxt)
            raw_candidates.append((move_fn, nxt, inb))

        # Filter: out-of-bounds (if known) and blocked directions
        candidates = []
        blocked_here = self.blocked.get(pos, set())
        for move_fn, nxt, inb in raw_candidates:
            if self.grid_size is not None and not inb:
                if self.debug:
                    print(f"  - candidate {self._move_name(move_fn):>5} -> nxt={nxt} REJECT out_of_bounds")
                continue
            if move_fn in blocked_here:
                if self.debug:
                    print(f"  - candidate {self._move_name(move_fn):>5} -> nxt={nxt} REJECT blocked_by_wall")
                continue
            candidates.append((move_fn, nxt, inb))

        if not candidates:
            if self.debug: print("[SCORING] No candidates after bounds/blocked filter")
            return None

        # ---- NEW: Prefer UNVISITED strictly ----
        unvisited = [c for c in candidates if c[1] not in self.visited]
        pool = unvisited if unvisited else candidates
        if self.debug:
            print(f"[FILTER] {'prefer UNVISITED' if unvisited else 'no unvisited available -> allow VISITED'}: "
                f"{len(pool)}/{len(candidates)} candidates kept")

        # Distance to nearest visited from current pos
        baseline = self._min_dist_to_visited(pos, exclude=pos)
        if self.debug:
            print(f"[SCORING] baseline_min_dist_from_current={baseline}")

        # Score pool
        scored = []
        for move_fn, nxt, inb in pool:
            dmin = self._min_dist_to_visited(nxt, exclude=None)
            not_closer = (dmin >= baseline)
            visited_times = self.visit_counts.get(nxt, 0)
            name = self._move_name(move_fn)
            if self.debug:
                print(f"  - candidate {name:>5} -> nxt={nxt} in_bounds={inb} "
                    f"minDistToVisited={dmin} not_closer={not_closer} visited_times={visited_times} "
                    f"{'(UNVISITED)' if nxt not in self.visited else '(VISITED)'}")
            scored.append((move_fn, nxt, dmin, not_closer, visited_times))

        # Prefer not getting closer to explored area within the (unvisited-or-fallback) pool
        not_closer_scored = [t for t in scored if t[3]]
        pool2 = not_closer_scored if not_closer_scored else scored
        if self.debug:
            print(f"[FILTER] {'not_closer' if not_closer_scored else 'all'} within "
                f"{'UNVISITED' if unvisited else 'ANY'}: kept {len(pool2)}")

        # Maximize distance; then prefer never-visited; avoid immediate backtrack; then random
        max_d = max(t[2] for t in pool2)
        best = [t for t in pool2 if t[2] == max_d]

        # If fallback to visited happened, this keeps pref for tiles with fewer visits
        never = [t for t in best if t[4] == 0]
        if never:
            best = never

        if self.last_action:
            opposite = self._opposite_of(self.last_action)
            noback = [t for t in best if t[0] != opposite]
            if noback:
                best = noback

        choice = random.choice(best)[0]
        if self.debug:
            names = ", ".join(self._move_name(t[0]) for t in best)
            print(f"[CHOICE] tie among {{{names}}} -> picked {self._move_name(choice)}")
        return choice

    # ------------------ Wall learning ------------------

    def _update_blocked_from_last_attempt(self, current_pos):
        """
        If last step we tried to move but our position did not change, learn a wall.
        This runs at the *start* of each think() with the current position.
        """
        if self._prev_pos_before_move is None or self._prev_move_attempt is None:
            return
        # If we’re still at the same position after attempting a move, it was blocked
        if current_pos == self._prev_pos_before_move:
            if self.debug:
                print(f"[WALL-LEARNED] Stayed at {current_pos} after trying "
                      f"{self._move_name(self._prev_move_attempt)} -> marking as blocked")
            self._mark_blocked(self._prev_pos_before_move, self._prev_move_attempt)
        # Clear attempt tracking regardless (we evaluated the outcome)
        self._clear_prev_move_attempt()

    def _mark_blocked(self, origin_pos, move_fn):
        """Mark a direction from a tile as blocked; also mark the symmetric block on the other side."""
        self.blocked[origin_pos].add(move_fn)
        # Symmetric block (optional but harmless): the other side can’t be entered from its opposite direction
        dx, dy = self.delta_map[move_fn]
        neighbor = (origin_pos[0] + dx, origin_pos[1] + dy)
        opp = self._opposite_of(move_fn)
        if opp is not None:
            self.blocked[neighbor].add(opp)
        if self.debug:
            print(f"[WALL-MARK] From {origin_pos} block {self._move_name(move_fn)} "
                  f"(and from {neighbor} block {self._move_name(opp) if opp else 'None'})")

    def _clear_prev_move_attempt(self):
        self._prev_pos_before_move = None
        self._prev_move_attempt = None

    # ------------------ Helpers & utilities ------------------

    def _random_non_backtracking(self):
        moves = list(self.delta_map.keys())
        if self.last_action:
            opposite = self._opposite_of(self.last_action)
            non_back = [m for m in moves if m != opposite]
            if non_back:
                return random.choice(non_back)
        return random.choice(moves)

    def _min_dist_to_visited(self, pos, exclude=None):
        vx = [v for v in self.visited if v != exclude]
        if not vx:
            return 10**9
        x, y = pos
        return min(abs(x - a) + abs(y - b) for (a, b) in vx)

    def _in_bounds(self, pos):
        if self.grid_size is None:
            return True
        x, y = pos
        W, H = self.grid_size
        return 0 <= x < W and 0 <= y < H

    def _opposite_of(self, move_fn):
        if move_fn == self.up:    return self.down
        if move_fn == self.down:  return self.up
        if move_fn == self.left:  return self.right
        if move_fn == self.right: return self.left
        return None

    def _move_name(self, move_fn):
        if move_fn == self.up: return "up"
        if move_fn == self.down: return "down"
        if move_fn == self.left: return "left"
        if move_fn == self.right: return "right"
        return "?"

if __name__ == "__main__":
    agent = reflexiveAgent(enable_ui=True, live_stats=True, debug=True)
    if agent.connect_to_environment(sizeX=8, sizeY=8, dirt_rate=0.3):
        performance = agent.run_simulation(verbose=True)
        print(f"Final performance: {performance}")
        agent.disconnect()
