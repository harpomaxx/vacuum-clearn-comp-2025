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
        self.movement_sequence = [self.up, self.right, self.down, self.left]
        self.current_move_index = 0
        self.last_position = (-1,-1)
        self.reach_side = False
        self.reach_corner = False
        self.change_side = True
        self.size = -1
        self.alredy_up = False
    
    def get_strategy_description(self) -> str:
        return "Clean if dirty, move in circular pattern when clean"
    
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

        print(self.size)
        print(perception.get('position'))

        if perception.get('is_dirty', False):
            return self.suck()
        

        if not self.reach_side and perception.get('position') != self.last_position:
            self.last_position = perception.get('position')
            return self.left()
        elif not self.reach_side and perception.get('position') == self.last_position:
            self.reach_side = True
            self.last_position = perception.get('position')
            return self.down()

        if self.reach_side and not self.reach_corner and perception.get('position') != self.last_position:
            self.last_position = perception.get('position')
            return self.down()
        elif not self.reach_corner and perception.get('position') == self.last_position:
            self.reach_corner = True
            self.last_position = (-1,-1)
            return self.suck()

        if self.reach_corner:

            print(perception.get('position')[0])

            if self.size == perception.get('position')[0]:
                self.change_side = not self.change_side
                if not self.alredy_up:
                    self.alredy_up = not self.alredy_up
                    self.change_side = not self.change_side
                    self.last_position = perception.get('position')
                    return self.up() 
            elif self.size != -1 and perception.get('position')[0] == 0:
                if self.alredy_up:
                    self.alredy_up = not self.alredy_up
                    self.change_side = not self.change_side
                    self.last_position = perception.get('position')
                    return self.up()
            

            if self.change_side and self.last_position != perception.get('position'):
                self.last_position = perception.get('position')
                return self.right()
            elif not self.change_side:
                self.last_position = perception.get('position')
                return self.left()
            else:
                self.size = perception.get('position')[0]
                self.alredy_up = not self.alredy_up
                return self.up()

    

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