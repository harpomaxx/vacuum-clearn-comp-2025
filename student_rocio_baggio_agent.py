"""
Efficient Vacuum Cleaner Agent
=============================

Este módulo define una clase `EfficientAgent` que combina dos estrategias
de limpieza: búsqueda voraz de la suciedad más cercana utilizando el estado
global (si está disponible) y un barrido serpenteante como mecanismo de
exploración cuando dicho estado global no se puede consultar.
"""

from __future__ import annotations
from typing import Callable, List, Optional, Tuple

try:
    from base_agent import BaseAgent  # Importa BaseAgent del proyecto original
except ImportError as exc:
    raise ImportError(
        "No se pudo importar BaseAgent. Asegúrate de ejecutar este archivo "
        "a través de run_agent.py dentro del repositorio vacuum-cleaner-world."
    ) from exc


class EfficientAgent(BaseAgent):
    """
    Agente aspiradora que combina búsqueda voraz y barrido serpenteante.

    - current_path: secuencia de acciones pendiente para llegar a la siguiente celda sucia.
    - use_global: indica si se seguirá intentando usar get_environment_state().
    - direction: dirección horizontal actual para el barrido serpenteante ("right" o "left").
    """

    def __init__(self, server_url: str = "http://localhost:5000", **kwargs) -> None:
        super().__init__(server_url, "EfficientAgent", **kwargs)
        self.current_path: List[Callable[[], bool]] = []
        self.use_global: bool = True
        self.direction: str = "right"

    def get_strategy_description(self) -> str:
        return (
            "Combina búsqueda voraz de la suciedad más cercana con un barrido "
            "serpenteante de respaldo. Si el estado global es accesible, el agente "
            "planifica un camino directo hasta la celda sucia más cercana; "
            "si no, recorre el entorno fila por fila."
        )

    def _compute_path_to_nearest_dirt(self) -> Optional[List[Callable[[], bool]]]:
        """
        Planifica un camino hacia la celda sucia más cercana usando el estado global.
        Devuelve una lista de métodos de movimiento (bound methods) o None si no
        puede planificar porque no hay acceso global.
        """
        try:
            state = self.get_environment_state()
        except Exception:
            return None
        if not state:
            return None

        grid = state.get("grid")
        agent_pos = state.get("agent_position")
        if grid is None or agent_pos is None:
            return None

        # Lista de coordenadas sucias
        dirty_positions: List[Tuple[int, int]] = [
            (x, y)
            for y, row in enumerate(grid)
            for x, cell in enumerate(row)
            if cell == 1
        ]
        if not dirty_positions:
            return []  # No hay suciedad

        ax, ay = agent_pos
        # Escoge la celda sucia más cercana
        target_x, target_y = min(
            dirty_positions,
            key=lambda pos: abs(pos[0] - ax) + abs(pos[1] - ay)
        )

        path: List[Callable[[], bool]] = []
        dx = target_x - ax
        dy = target_y - ay
        # Movimientos horizontales
        step = self.right if dx > 0 else self.left
        for _ in range(abs(dx)):
            path.append(step)
        # Movimientos verticales
        step = self.down if dy > 0 else self.up
        for _ in range(abs(dy)):
            path.append(step)
        return path

    def _serpentine_step(self) -> bool:
        """
        Ejecuta un paso del barrido serpenteante. Intenta avanzar en la dirección
        actual; si choca con un borde, desciende una fila y cambia de dirección.
        Si tampoco puede bajar, ejecuta 'idle' para consumir la acción.
        """
        if self.direction == "right":
            move_horizontal = self.right
            reverse_direction = "left"
        else:
            move_horizontal = self.left
            reverse_direction = "right"

        if move_horizontal():
            return True
        if self.down():
            self.direction = reverse_direction
            return True
        return self.idle()

    def think(self) -> bool:
        """
        Método principal de decisión. Sigue la estrategia:
        1. Si no está conectado, termina.
        2. Si la celda actual está sucia, límpiala.
        3. Si hay un camino planificado, ejecuta el siguiente paso.
        4. Si se puede usar el estado global, planifica hacia la celda sucia
           más cercana; si falla, desactiva el acceso global.
        5. Si no hay acceso global, recorre en serpentina.
        """
        if not self.is_connected():
            return False

        perception = self.get_perception()
        if perception.get("is_dirty", False):
            return self.suck()

        # Ejecuta el camino planificado (si existe)
        if self.current_path:
            next_action = self.current_path.pop(0)
            return next_action()

        # Intenta planificar con estado global
        if self.use_global:
            path = self._compute_path_to_nearest_dirt()
            if path is None:
                # Falla al acceder al estado global
                self.use_global = False
            elif not path:
                # No hay suciedad; permanece inactivo
                return self.idle()
            else:
                # Almacena el camino y ejecuta el primer paso
                self.current_path = path
                next_action = self.current_path.pop(0)
                return next_action()

        # Barrido serpenteante como último recurso
        return self._serpentine_step()
