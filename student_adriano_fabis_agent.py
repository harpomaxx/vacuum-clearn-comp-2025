import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from base_agent import BaseAgent


def calculate_grid_dirtiness(grid):
    total_cells = sum(len(row) for row in grid)
    dirty_cells = sum(cell == '1' for row in grid for cell in row)
    return dirty_cells / total_cells if total_cells > 0 else 0


class AdrianoAgent(BaseAgent):

    def __init__(self, server_url="http://localhost:5000", **kwargs):
        super().__init__(server_url, "AdrianoAgent", **kwargs)
        env = self.get_environment_state()
        self.grid_dirtiness = calculate_grid_dirtiness(env.get('grid', []))
        # Estado interno para saber si seguimos intentando normalizar a (0,0)
        self.normalizing = True

    def get_strategy_description(self):
        return (
            "Normaliza a (0,0) si hay acciones suficientes comparado con el coste del recorrido; "
            "luego recorre en serpentina."
        )

    def think(self):
        if not self.is_connected():
            return False

        perception = self.get_perception()
        if not perception or perception.get('is_finished', True):
            return False

        if perception.get('is_dirty', False):
            return self.suck()

        x, y = perception.get('position')
        grid = self.get_environment_state().get('grid')
        if not grid:
            return self.idle()

        remaining_actions = perception.get('actions_remaining', 0)
        height, width = len(grid), len(grid[0])

        # Estimación del coste restante del recorrido completo en serpentina desde (x,y)
        map_remaining = (height - y - 1) * width
        if y % 2 == 0:  # fila par → vamos de izquierda a derecha
            map_remaining += (width - 1 - x)
        else:  # fila impar → derecha a izquierda
            map_remaining += x

        # Criterio para decidir si vale la pena seguir normalizando
        # Si no quedan suficientes acciones para gastar "de más" normalizando, detiene normalización.
        if remaining_actions <= map_remaining * (1 + self.grid_dirtiness):
            self.normalizing = False

        # Si ya estamos en (0,0), dejamos de normalizar.
        if (x, y) == (0, 0):
            self.normalizing = False

        # Fase de normalización: mover hacia arriba y luego a la izquierda.
        if self.normalizing:
            if y > 0:
                return self.up()
            if x > 0:
                return self.left()
            # Si no podemos movernos más, desactivar para evitar ciclo
            self.normalizing = False

        # Fase de serpentina
        return self._serpentine_movement(x, y, height, width)

    def _serpentine_movement(self, x, y, height, width):
        """
        Movimiento en serpentina:
        - Filas pares: izquierda → derecha
        - Filas impares: derecha → izquierda
        - Al final de cada fila, bajar
        """
        # Si estamos en la última fila y al final del patrón, nada más que hacer
        if y == height - 1 and ((y % 2 == 0 and x == width - 1) or (y % 2 == 1 and x == 0)):
            return self.idle()

        if y % 2 == 0:  # fila par: mover derecha
            if x < width - 1:
                return self.right()
            if y < height - 1:
                return self.down()
            return self.idle()
        else:  # fila impar: mover izquierda
            if x > 0:
                return self.left()
            if y < height - 1:
                return self.down()
            return self.idle()
