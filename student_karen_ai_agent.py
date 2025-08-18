import sys
import os
from typing import Optional
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from base_agent import BaseAgent

class StudentKarenAiAgent(BaseAgent):
    """
    The most stupid agent ever created.
    
    This agent simply moves back and forth between two cells:
    - Move right
    - Move left
    - Repeat forever
    
    It never cleans dirt, never stops, just mindlessly oscillates.
    """
    
    def __init__(self, server_url: str = "http://localhost:5000", 
                 enable_ui: bool = False,
                 record_game: bool = False, 
                 replay_file: Optional[str] = None,
                 cell_size: int = 60,
                 fps: int = 10,
                 auto_exit_on_finish: bool = True,
                 live_stats: bool = False):
        super().__init__(server_url, "StudentKarenAiAgent", enable_ui, record_game, 
                        replay_file, cell_size, fps, auto_exit_on_finish, live_stats)
        
        # Simple state: True = move right, False = move left
        self.move_right = True
    
    def get_strategy_description(self) -> str:
        return "Move right, then left, repeat forever - never clean anything"
    
    def think(self) -> bool:
        """
        The simplest possible logic: just alternate between moving right and left.
        """
        if not self.is_connected():
            return False
        
        perception = self.get_perception()
        if not perception or perception.get('is_finished', True):
            return False
        
        # Move right or left based on current state
        if self.move_right:
            success = self.right()
        else:
            success = self.left()
        
        # Flip direction for next move
        self.move_right = not self.move_right
        
        return success

def run_stupid_agent_simulation(size_x: int = 8, size_y: int = 8, 
                               dirt_rate: float = 0.3, 
                               server_url: str = "http://localhost:5000",
                               verbose: bool = True) -> int:
    """
    Function to run a simulation with the stupid agent.
    """
    agent = StudentSillyCleanerAgent(server_url)
    
    try:
        if not agent.connect_to_environment(size_x, size_y, dirt_rate):
            return 0
        
        performance = agent.run_simulation(verbose)
        return performance
    
    finally:
        agent.disconnect()

if __name__ == "__main__":
    print("Silly Cleaner Agent - Back and Forth Movement")
    print("Make sure the environment server is running on localhost:5000")
    print("Strategy: Move right, then left, repeat forever - never clean")
    print()
    
    performance = run_stupid_agent_simulation(verbose=True)
    print(f"\nFinal performance: {performance}")
    print("As expected, performance is 0 because this agent never cleans anything!")