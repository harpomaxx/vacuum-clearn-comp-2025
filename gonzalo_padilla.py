import sys
import os
import random
from typing import Optional
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from base_agent import BaseAgent

class RandomAgent(BaseAgent):
    
    def __init__(self, server_url: str = "http://localhost:5000", 
                 enable_ui: bool = False,
                 record_game: bool = False, 
                 replay_file: Optional[str] = None,
                 cell_size: int = 60,
                 fps: int = 10,
                 auto_exit_on_finish: bool = True,
                 live_stats: bool = False):
        super().__init__(server_url, "RandomAgent", enable_ui, record_game, 
                        replay_file, cell_size, fps, auto_exit_on_finish, live_stats)
        
        self.started_raster = False
        
    def get_strategy_description(self) -> str:
        return "Toma acciones al azar"
    
    def think(self) -> bool:
        if not self.is_connected():
            return False
        
        perception = self.get_perception()
        if not perception or perception.get('is_finished', True):
            return False

        pos = perception.get('position')
        
        state = self.get_environment_state()
        if not state:
            return False
        
        grid = state.get('grid')
        performance = state.get('performance')
        grid_size = len(grid)

        if perception.get('is_dirty'):
            return self.suck()

        if not self.started_raster:
            if pos[0] > 0:
                return self.left()
            if pos[1] > 0:
                return self.up()
            
            self.started_raster = True
        
        if pos[0] < grid_size - 1:
            if pos[1] % 2 == 0:
                return self.right()
            else:
                return self.left()
        else:
            return self.down()
        

def run_example_agent_simulation(size_x: int = 8, size_y: int = 8, 
                                dirt_rate: float = 0.3, 
                                server_url: str = "http://localhost:5000",
                                verbose: bool = True) -> int:
    agent = RandomAgent(server_url)
    
    try:
        if not agent.connect_to_environment(size_x, size_y, dirt_rate):
            return 0
        
        performance = agent.run_simulation(verbose)
        return performance
    
    finally:
        agent.disconnect()

if __name__ == "__main__":
    print("Random Agent")
    print("Make sure the environment server is running on localhost:5000")
    print("Strategy: Clean if dirty, move in circular pattern when clean")
    print()
    
    performance = run_example_agent_simulation(verbose=True)
    print(f"\nFinal performance: {performance}")
    
    print("\nTo create your own agent:")
    print("1. Copy this file and rename it")
    print("2. Change the class name")  
    print("3. Implement your logic in the think() method")
    print("4. Register it in run_agent.py AVAILABLE_AGENTS dictionary")