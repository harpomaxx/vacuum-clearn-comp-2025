import sys
import os
from typing import Optional
import random

# Add the parent directory to the path to import base_agent
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from base_agent import BaseAgent


class CompetitiveAgent(BaseAgent):
    """
    An agent that prioritizes cleaning and explores unvisited cells systematically.
    Avoids immediate reversals when possible.
    """

    def __init__(self,
                 server_url: str = "http://localhost:5000",
                 enable_ui: bool = False,
                 record_game: bool = False,
                 replay_file: Optional[str] = None,
                 cell_size: int = 60,
                 fps: int = 10,
                 auto_exit_on_finish: bool = True,
                 live_stats: bool = False):
        super().__init__(server_url, "CompetitiveAgent", enable_ui, record_game,
                         replay_file, cell_size, fps, auto_exit_on_finish, live_stats)

        self.visited_cells = set()
        self.grid_width = 0
        self.grid_height = 0
        self.last_action = None

    def get_strategy_description(self) -> str:
        return "CompetitiveAgent: Prioritizes cleaning, then explores unvisited cells. Avoids immediate reversals."

    def think(self) -> bool:
        """
        Implements the decision logic of the agent.

        Returns:
            bool: True if the action was executed successfully, False if simulation should stop.
        """
        if not self.is_connected():
            return False

        perception = self.get_perception()
        if not perception or perception.get('is_finished', False):
            return False

        # Initialize grid dimensions if not set
        if self.grid_width == 0 or self.grid_height == 0:
            env_state = self.get_environment_state()
            if env_state and 'grid' in env_state:
                grid = env_state['grid']
                self.grid_height = len(grid)
                if self.grid_height > 0:
                    self.grid_width = len(grid[0])
                print(f"[{self.agent_name}] Initialized with grid dimensions: {self.grid_width}x{self.grid_height}")

        current_x, current_y = perception.get('position', (0, 0))
        self.visited_cells.add((current_x, current_y))

        # Priority 1: Clean if dirty
        if perception.get('is_dirty', False):
            self.last_action = 'SUCK'
            return self.suck()

        # Priority 2: Move to unvisited cells
        possible_moves = []

        # Check all directions for unvisited cells
        if current_y > 0 and (current_x, current_y - 1) not in self.visited_cells:
            possible_moves.append('UP')
        if current_y < self.grid_height - 1 and (current_x, current_y + 1) not in self.visited_cells:
            possible_moves.append('DOWN')
        if current_x > 0 and (current_x - 1, current_y) not in self.visited_cells:
            possible_moves.append('LEFT')
        if current_x < self.grid_width - 1 and (current_x + 1, current_y) not in self.visited_cells:
            possible_moves.append('RIGHT')

        chosen_action = None

        if possible_moves:
            # Prefer unvisited cells
            chosen_action = random.choice(possible_moves)
        else:
            # All adjacent cells visited, try valid moves avoiding immediate reversal
            all_valid_moves = []

            if current_y > 0 and self.last_action != 'DOWN':
                all_valid_moves.append('UP')
            if current_y < self.grid_height - 1 and self.last_action != 'UP':
                all_valid_moves.append('DOWN')
            if current_x > 0 and self.last_action != 'RIGHT':
                all_valid_moves.append('LEFT')
            if current_x < self.grid_width - 1 and self.last_action != 'LEFT':
                all_valid_moves.append('RIGHT')

            if all_valid_moves:
                chosen_action = random.choice(all_valid_moves)
            else:
                # Last resort: any valid move
                fallback_moves = []
                if current_y > 0:
                    fallback_moves.append('UP')
                if current_y < self.grid_height - 1:
                    fallback_moves.append('DOWN')
                if current_x > 0:
                    fallback_moves.append('LEFT')
                if current_x < self.grid_width - 1:
                    fallback_moves.append('RIGHT')

                if fallback_moves:
                    chosen_action = random.choice(fallback_moves)
                else:
                    chosen_action = 'IDLE'

        # Execute chosen action
        self.last_action = chosen_action

        if chosen_action == 'UP':
            return self.up()
        elif chosen_action == 'DOWN':
            return self.down()
        elif chosen_action == 'LEFT':
            return self.left()
        elif chosen_action == 'RIGHT':
            return self.right()
        else:
            return self.idle()


def run_agent_simulation(size_x: int = 8, size_y: int = 8,
                             dirt_rate: float = 0.3,
                             server_url: str = "http://localhost:5000",
                             verbose: bool = True) -> int:
    """
    Function to run a simulation with the CompetitiveAgent.
    """
    agent = CompetitiveAgent(server_url, enable_ui=True, live_stats=verbose, record_game=False)

    try:
        if not agent.connect_to_environment(size_x, size_y, dirt_rate):
            print("Failed to connect to environment.")
            return 0

        performance = agent.run_simulation(verbose=verbose)
        return performance

    finally:
        agent.disconnect()


if __name__ == "__main__":
    print("New Agent - Smart Exploration Strategy")
    print("Make sure the environment server is running on localhost:5000")
    print("Strategy: Prioritizes cleaning, then explores unvisited cells. Avoids immediate reversals.")
    print()

    performance = run_agent_simulation(verbose=True)
    print(f"\nFinal performance: {performance}")

    print("\nThis agent can be registered in run_agent.py like:")
    print('    "new": CompetitiveAgent')