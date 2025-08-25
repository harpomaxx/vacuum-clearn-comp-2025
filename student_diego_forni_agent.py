import sys
import os
from collections import defaultdict, deque

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from base_agent import BaseAgent

class WallSweepAgent(BaseAgent):
    """
    Unknown-size matrix traversal:
      Phase A: go UP until wall, then LEFT until wall -> reach top-left corner.
      Phase B: serpentine sweep (RIGHT to wall, DOWN 1, LEFT to wall, DOWN 1, ...).
    Cleans whenever on dirt. Learns walls when a move fails or position doesn't change.
    """

    def __init__(self, server_url="http://localhost:5000", debug=True, **kwargs):
        super().__init__(server_url, "WallSweepAgent", **kwargs)
        self.debug = debug

        # Movement functions and deltas (x, y) with y increasing downward
        self.delta_map = {
            self.up:    (0, -1),
            self.down:  (0,  1),
            self.left:  (-1, 0),
            self.right: (1,  0),
        }

        # Memory
        self.visited = set()
        self.visit_counts = defaultdict(int)
        self.blocked = defaultdict(set)   # {(x,y): {move_fn, ...}}
        self.history = deque(maxlen=10)

        # Control state
        self.mode = "go_top"  # go_top -> go_left -> sweep
        self.row_dir = 1      # +1 = moving right, -1 = moving left (during sweep)
        self.last_action = None

        # Track last movement attempt to detect "bump"
        self._prev_pos_before_move = None
        self._prev_move_attempt = None

        self.step = 0

    # ---------- main loop ----------
    def get_strategy_description(self):
        return ("Reach a corner (top-left) by going to the nearest walls, "
                "then traverse the whole grid in a serpentine pattern. "
                "Clean first, and remember walls when bumps occur.")

    def think(self):
        if not self.is_connected():
            return False

        p = self.get_perception()
        if not p or p.get("is_finished", True):
            return False

        pos = p.get("position", None)
        is_dirty = p.get("is_dirty", False)
        if isinstance(pos, (tuple, list)) and len(pos) == 2:
            pos = (int(pos[0]), int(pos[1]))
            self.visited.add(pos)
            self.visit_counts[pos] += 1
            self.history.append(pos)

        self.step += 1
        self._update_blocked_from_last_attempt(current_pos=pos)

        if self.debug:
            print(f"\n[STEP {self.step}] mode={self.mode} pos={pos} dirty={is_dirty}")
            if pos in self.blocked and self.blocked[pos]:
                bl = ", ".join(self._move_name(m) for m in self.blocked[pos])
                print(f"[WALLS] From {pos} blocked: {{{bl}}}")

        # Clean first
        if is_dirty:
            if self.debug: print("[DECISION] Dirty -> suck()")
            self._clear_prev_move_attempt()
            ok = self.suck()
            if self.debug: print(f"[ACTION] suck() -> {ok}")
            self.last_action = None
            return ok

        # Control-state behavior
        if self.mode == "go_top":
            # If UP is blocked here, we're at the top; switch to go_left
            if self._is_blocked_here(pos, self.up):
                if self.debug: print("[MODE] Reached top wall -> go_left")
                self.mode = "go_left"
                return self._idle_or_progress()
            return self._attempt_move(pos, self.up)

        elif self.mode == "go_left":
            if self._is_blocked_here(pos, self.left):
                if self.debug: print("[MODE] Reached left wall -> sweep mode")
                self.mode = "sweep"
                self.row_dir = 1  # start sweeping to the right
                return self._idle_or_progress()
            return self._attempt_move(pos, self.left)

        elif self.mode == "sweep":
            # Determine horizontal direction function
            horiz = self.right if self.row_dir == 1 else self.left
            # 1) Try to continue horizontally
            if not self._is_blocked_here(pos, horiz):
                return self._attempt_move(pos, horiz)
            # 2) If horizontal blocked, try to go DOWN one row
            if not self._is_blocked_here(pos, self.down):
                # Flip row direction after moving down
                moved = self._attempt_move(pos, self.down)
                if moved:
                    self.row_dir *= -1
                return moved
            # 3) Nowhere to go (bottom reached and at wall) -> done; idle
            if self.debug: print("[MODE] Sweep complete or stuck at bottom/corner -> idle()")
            self._clear_prev_move_attempt()
            return self.idle()

        # Fallback
        if self.debug: print("[MODE] Unknown mode; idling")
        self._clear_prev_move_attempt()
        return self.idle()

    # ---------- move & wall learning ----------
    def _attempt_move(self, pos, move_fn):
        """Try a move, learn immediate failure as wall if move() returns False."""
        self._prev_pos_before_move = pos
        self._prev_move_attempt = move_fn
        ok = move_fn()
        if self.debug:
            print(f"[ACTION] {self._move_name(move_fn)}() -> {ok}")
        self.last_action = move_fn

        # Immediate failure: learn wall now
        if ok is False and pos is not None:
            if self.debug:
                print(f"[WALL-LEARNED] Immediate failure for {self._move_name(move_fn)} from {pos}")
            self._mark_blocked(pos, move_fn)
            self._clear_prev_move_attempt()
        return ok

    def _update_blocked_from_last_attempt(self, current_pos):
        """If last step tried to move but pos didn't change, mark that direction as blocked."""
        if self._prev_pos_before_move is None or self._prev_move_attempt is None:
            return
        if current_pos == self._prev_pos_before_move:
            if self.debug:
                print(f"[WALL-LEARNED] Stayed at {current_pos} after trying "
                      f"{self._move_name(self._prev_move_attempt)} -> marking wall")
            self._mark_blocked(self._prev_pos_before_move, self._prev_move_attempt)
        self._clear_prev_move_attempt()

    def _mark_blocked(self, origin_pos, move_fn):
        self.blocked[origin_pos].add(move_fn)
        dx, dy = self.delta_map[move_fn]
        neighbor = (origin_pos[0] + dx, origin_pos[1] + dy)
        opp = self._opposite_of(move_fn)
        if opp:
            self.blocked[neighbor].add(opp)
        if self.debug:
            print(f"[WALL-MARK] From {origin_pos} block {self._move_name(move_fn)} "
                  f"(and from {neighbor} block {self._move_name(opp) if opp else 'None'})")

    def _is_blocked_here(self, pos, move_fn):
        if pos is None:  # play it safe
            return False
        return move_fn in self.blocked.get(pos, set())

    def _clear_prev_move_attempt(self):
        self._prev_pos_before_move = None
        self._prev_move_attempt = None

    # ---------- utilities ----------
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

    def _idle_or_progress(self):
        """Small helper so mode switches still 'do something' in the same step if no dirt."""
        return self.idle()


if __name__ == "__main__":
    # Run directly, or via repo's run_agent.py
    agent = WallSweepAgent(enable_ui=True, live_stats=True, debug=True)
    if agent.connect_to_environment(sizeX=8, sizeY=8, dirt_rate=0.3):
        perf = agent.run_simulation(verbose=True)
        print(f"Final performance: {perf}")
        agent.disconnect()
