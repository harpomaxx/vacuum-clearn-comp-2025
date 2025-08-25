import sys
import os
import random
from typing import Optional
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from base_agent import BaseAgent

class ExampleAgent(BaseAgent):
    """
    Agente de ejemplo que demuestra cómo crear un nuevo tipo de agente.
    
    Este agente implementa una estrategia simple como ejemplo:
    - Limpia si hay suciedad
    - Se mueve en un patrón circular cuando no hay suciedad
    
    Para usar este agente como plantilla:
    1. Copia este archivo y renómbralo
    2. Cambia el nombre de la clase
    3. Implementa tu lógica en el método think()
    4. Registra el agente en run_agent.py
    """
    
    def __init__(self, server_url: str = "http://localhost:5000", 
                 enable_ui: bool = False,
                 record_game: bool = False, 
                 replay_file: Optional[str] = None,
                 cell_size: int = 60,
                 fps: int = 10,
                 auto_exit_on_finish: bool = True,
                 live_stats: bool = False):
        super().__init__(server_url, "ExampleAgent", enable_ui, record_game, 
                        replay_file, cell_size, fps, auto_exit_on_finish, live_stats)
        
        # Estado interno para movimiento circular
        self.star_serpent = False
        self.reach_side = False
        self.reach_up = False
        self.last_position = (-1,-1)
        self.movement_sequence = [self.up, self.right, self.down, self.left]
        self.current_move_index = 0
        
    
    def get_strategy_description(self) -> str:
        return "Clean if dirty, move in circular pattern when clean"
    
    def think(self) -> bool:

        


        if not self.is_connected():
            return False
        
        perception = self.get_perception()
        if not perception or perception.get('is_finished', True):
            return False
        
        pos = perception.get('position')

        if self.star_serpent == False:
            if self.reach_side==False:
                if pos!=self.last_position:
                    self.current_move_index = 1
                else:
                    self.reach_side=True
                    self.current_move_index = 0
            else:
                if pos!=self.last_position:
                    self.current_move_index = 0
                else:
                    self.star_serpent=True
                    self.current_move_index = 3
        else:
            if perception.get('is_dirty',False):
                return self.suck()
            
            if self.reach_side==True:

                if pos==self.last_position:
                    self.reach_side= False
                    self.current_move_index = 2
                else:
                    self.current_move_index = 3
            else:
                if pos==self.last_position:
                    self.reach_side= True
                    self.current_move_index = 2
                else:
                    self.current_move_index = 1
        move_function = self.movement_sequence[self.current_move_index]
        success = move_function()
        self.last_position = pos
        return success
                    




def run_example_agent_simulation(size_x: int = 8, size_y: int = 8, 
                                dirt_rate: float = 0.3, 
                                server_url: str = "http://localhost:5000",
                                verbose: bool = True) -> int:
    """
    Función de conveniencia para ejecutar una simulación con ExampleAgent.
    """
    agent = ExampleAgent(server_url)
    
    try:
        if not agent.connect_to_environment(size_x, size_y, dirt_rate):
            return 0
        
        performance = agent.run_simulation(verbose)
        return performance
    
    finally:
        agent.disconnect()

if __name__ == "__main__":
    print("Example Agent - Circular Movement Pattern")
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