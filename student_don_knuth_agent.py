import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from base_agent import BaseAgent

class ReflexAgent(BaseAgent):
    """
    Agente reflexivo simple con comportamiento determinístico.
    
    Estrategia:
    1. Si hay suciedad en posición actual → Limpiar
    2. Si no hay suciedad → Seguir patrón de barrido sistemático
    
    Patrón de barrido: izquierda-derecha, fila por fila, de arriba hacia abajo.
    Comportamiento completamente determinístico y predecible.
    """
    
    def __init__(self, server_url: str = "http://localhost:5000", **kwargs):
        super().__init__(server_url, "ReflexAgent", **kwargs)
        
        # Estado interno para patrón de barrido
        self.current_row = 0
        self.current_col = 0
        self.direction = 1  # 1 = derecha, -1 = izquierda
        self.grid_width = 0
        self.grid_height = 0
        self.initialized = False
    
    def get_strategy_description(self) -> str:
        return "Deterministic systematic sweep with immediate cleaning"
    
    def _initialize_sweep_pattern(self):
        """
        Inicializa el patrón de barrido basado en el estado actual.
        """
        if self.initialized:
            return
        
        state = self.get_environment_state()
        if state and 'grid' in state:
            grid = state['grid']
            self.grid_height = len(grid)
            self.grid_width = len(grid[0]) if self.grid_height > 0 else 0
            
            if 'agent_position' in state:
                self.current_col, self.current_row = state['agent_position']
            
            self.initialized = True
    
    def _get_next_sweep_position(self):
        """
        Calcula la siguiente posición en el patrón de barrido sistemático.
        Patrón: izquierda-derecha, fila por fila.
        """
        next_col = self.current_col + self.direction
        next_row = self.current_row
        
        # Si llegamos al final de la fila
        if next_col < 0 or next_col >= self.grid_width:
            # Cambiar de fila
            next_row = self.current_row + 1
            
            # Si llegamos al final del grid, volver al inicio
            if next_row >= self.grid_height:
                next_row = 0
                next_col = 0
                self.direction = 1  # Resetear dirección
            else:
                # Cambiar dirección para la nueva fila (zigzag)
                self.direction *= -1
                if self.direction == 1:
                    next_col = 0
                else:
                    next_col = self.grid_width - 1
        
        return next_col, next_row
    
    def _move_towards_target(self, target_col: int, target_row: int) -> bool:
        """
        Mueve el agente un paso hacia la posición objetivo.
        Usa el camino más corto (Manhattan distance).
        """
        # Primero mover horizontalmente, luego verticalmente
        if self.current_col < target_col:
            return self.right()
        elif self.current_col > target_col:
            return self.left()
        elif self.current_row < target_row:
            return self.down()
        elif self.current_row > target_row:
            return self.up()
        else:
            # Ya estamos en la posición objetivo
            return True
    
    def _update_position(self):
        """
        Actualiza la posición interna basada en la percepción real.
        """
        perception = self.get_perception()
        if perception and 'position' in perception:
            self.current_col, self.current_row = perception['position']
    
    def think(self) -> bool:
        """
        Implementa la lógica de decisión del agente reflexivo simple.
        
        Returns:
            True si se ejecutó una acción, False si debe terminar
        """
        if not self.is_connected():
            return False
        
        perception = self.get_perception()
        if not perception or perception.get('is_finished', True):
            return False
        
        # Inicializar patrón de barrido si es necesario
        self._initialize_sweep_pattern()
        if not self.initialized:
            return False
        
        # Actualizar posición interna
        self._update_position()
        
        # REGLA 1: Si hay suciedad, limpiar inmediatamente
        if perception.get('is_dirty', False):
            return self.suck()
        
        # REGLA 2: Si no hay suciedad, seguir patrón de barrido
        target_col, target_row = self._get_next_sweep_position()
        
        # Moverse hacia la siguiente posición del patrón
        return self._move_towards_target(target_col, target_row)

if __name__ == "__main__":
    print("Reflex Agent - Deterministic Systematic Sweep")
    print("Make sure the environment server is running on localhost:5000")
    print("Strategy: Clean if dirty, otherwise follow systematic sweep pattern")
    print()
    
    agent = ReflexAgent()
    
    try:
        if agent.connect_to_environment(8, 8, 0.3):
            performance = agent.run_simulation(verbose=True)
            print(f"\nFinal performance: {performance}")
    finally:
        agent.disconnect()