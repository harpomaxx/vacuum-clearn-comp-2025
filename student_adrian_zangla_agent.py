import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from base_agent import BaseAgent
from collections import deque


class StudentAdrianZanglaAgent(BaseAgent):
    directions = {"up": (0, -1), "left": (-1, 0), "down": (0, 1), "right": (1, 0)}

    def __init__(self, server_url="http://localhost:5000", **kwargs):
        super().__init__(server_url, "StudentAdrianZanglaAgent", **kwargs)
        self.last_position = None
        self.visited = set()
        self.size = None
        self.target = None

    def get_strategy_description(self):
        return """This agent assumes a square, obstacle-free grid. 
        It first moves downward to determine the grid’s size. During exploration, 
        it always prioritizes moving to adjacent unvisited cells
        in a fixed counterclockwise order (up, left, down, right). 
        If all adjacent cells have been visited, it uses breadth-first search (BFS) 
        to find and move toward the nearest unexplored cell. 
        The agent cleans any dirty cell it encounters before moving."""

    def search(self, x, y):
        # BFS
        queue = deque([(x, y)])
        visited = set()
        while queue:
            cx, cy = queue.popleft()
            for dx, dy in self.directions.values():
                nx, ny = cx + dx, cy + dy
                if (
                    0 <= nx < self.size
                    and 0 <= ny < self.size
                    and (nx, ny) not in visited
                ):
                    if (nx, ny) not in self.visited:
                        return (nx, ny)

                    visited.add((nx, ny))
                    queue.append((nx, ny))
        return None

    def think(self):
        if not self.is_connected():
            return False

        perception = self.get_perception()
        if not perception or perception.get("is_finished", True):
            return False

        # Your logic here
        if perception.get("is_dirty", False):
            return self.suck()

        x, y = perception.get("position")
        self.visited.add((x, y))

        if not self.size:
            if self.last_position == (x, y):
                self.size = y + 1  # Square grid assumption
                self.last_position = None
            else:
                self.last_position = (x, y)
                return self.down()

        for move, (dx, dy) in self.directions.items():
            nx, ny = x + dx, y + dy
            if (
                0 <= nx < self.size
                and 0 <= ny < self.size
                and (nx, ny) not in self.visited
            ):
                self.target = None
                return getattr(self, move)()

        if (x, y) == self.target:
            self.target = None

        self.target = self.search(x, y)
        if not self.target:
            return False

        tx, ty = self.target
        if ty < y:
            return self.up()
        elif tx < x:
            return self.left()
        elif ty > y:
            return self.down()
        elif tx > x:
            return self.right()
