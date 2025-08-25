import sys
import os
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
    last_position = None
    
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
        self.step=0
        self.steppedown = False
    
    def get_strategy_description(self) -> str:
        return "Clean if dirty, move to the corner and snake side to side"
    
    def think(self) -> bool:
        """
        Implementa la lógica de decisión del agente de ejemplo.
        
        Estrategia:
        1. Si hay suciedad → Limpiar
        2. Si no hay suciedad → Moverse en patrón circular
        """
        if not self.is_connected():
            return False
        
        perception = self.get_perception()
        if not perception or perception.get('is_finished', True):
            return False
        
        #print(f"Current Perception: {perception}")
        current_pos = perception['position']
        
        # REGLA 1: Si hay suciedad, limpiar
        if perception.get('is_dirty', False):
            return self.suck()
        
        if not hasattr(self, "_last_position"):
            self._last_position = current_pos
            
        last_pos = self._last_position
        self._last_position = current_pos
        
        def safe_move(move_func):
            prev_pos = self._last_position
            success = move_func()
            new_pos = self.get_perception().get('position', prev_pos)
            return new_pos != prev_pos
            if moved:
                self._last_position = new_pos
            return moved
        
        row, col = current_pos
        moved = False
        
        if row>0 and self.step==0:
            moved = safe_move(self.up)
            if moved:
                return True
            
        if not moved and self.step==0:
            moved = safe_move(self.right)
            if moved:
                return True
            else:
                self.step = 1  
                      
        #print(self.step, self.steppedown)
        if self.step==1:
            if self.steppedown:
                #print("Moving Left")
                self.steppedown = False
                return safe_move(self.left)
            if safe_move(self.left):
                #print("Moving Left")
                self.steppedown = False
                return True
            else:
                #print("Moving Down")
                self.steppedown = True
                self.step=2
                return safe_move(self.down)
        else:
            if self.steppedown:
                #print("Moving Right")
                self.steppedown = False
                return safe_move(self.right)
            if safe_move(self.right):
                #print("Moving Left")
                self.steppedown = False
                return True
            else:
                #print("Moving Down")
                self.steppedown = True
                self.step=1
                return safe_move(self.down)

            
        return False
        

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