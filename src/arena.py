"""
Arena definition with 3D boundaries.

The Arena represents a bounded volume of airspace where the simulation takes place.
"""

from dataclasses import dataclass
from typing import List, Tuple, Optional
import numpy as np
from .entities import Entity


@dataclass
class GeographicReference:
    """Geographic reference point for the arena."""
    latitude: float  # degrees
    longitude: float  # degrees
    altitude: float = 0.0  # meters above sea level


class Arena:
    """
    A 3D bounded arena for drone simulation.

    The arena is defined as a rectangular prism with:
    - Origin at (0, 0, 0) representing the southwest corner at ground level
    - X-axis pointing East
    - Y-axis pointing North
    - Z-axis pointing Up

    Attributes:
        bounds: (x_max, y_max, z_max) in meters
        geo_reference: Optional geographic location
        entities: List of all entities in the arena
    """

    def __init__(
        self,
        bounds: Tuple[float, float, float],
        geo_reference: Optional[GeographicReference] = None
    ):
        """
        Initialize the arena.

        Args:
            bounds: (x_max, y_max, z_max) dimensions in meters
            geo_reference: Optional geographic reference point
        """
        self.bounds = np.array(bounds, dtype=float)
        self.geo_reference = geo_reference
        self.entities: List[Entity] = []
        self.time = 0.0  # Simulation time in seconds

    def add_entity(self, entity: Entity) -> None:
        """Add an entity to the arena."""
        # Ensure entity starts within bounds
        entity.position = self.enforce_bounds(entity.position)
        self.entities.append(entity)

    def remove_entity(self, entity: Entity) -> None:
        """Remove an entity from the arena."""
        if entity in self.entities:
            self.entities.remove(entity)

    def enforce_bounds(self, position: np.ndarray) -> np.ndarray:
        """
        Enforce arena boundaries by clamping position.

        Args:
            position: 3D position array

        Returns:
            Clamped position within bounds
        """
        return np.clip(position, [0, 0, 0], self.bounds)

    def is_within_bounds(self, position: np.ndarray) -> bool:
        """Check if a position is within arena bounds."""
        return np.all(position >= 0) and np.all(position <= self.bounds)

    def get_active_entities(self) -> List[Entity]:
        """Get list of all active entities."""
        return [e for e in self.entities if e.active]

    def find_entities_near(
        self,
        position: np.ndarray,
        radius: float
    ) -> List[Tuple[Entity, float]]:
        """
        Find all entities within a radius of a position.

        Args:
            position: Center position
            radius: Search radius in meters

        Returns:
            List of (entity, distance) tuples
        """
        nearby = []
        for entity in self.get_active_entities():
            dist = np.linalg.norm(entity.position - position)
            if dist <= radius:
                nearby.append((entity, dist))
        return nearby

    def update(self, dt: float) -> None:
        """
        Update arena state.

        Args:
            dt: Time step in seconds
        """
        self.time += dt

    def reset(self) -> None:
        """Reset the arena to initial state."""
        self.entities.clear()
        self.time = 0.0

    def get_bounds_center(self) -> np.ndarray:
        """Get the center point of the arena."""
        return self.bounds / 2.0

    def __repr__(self) -> str:
        return (
            f"Arena(bounds={self.bounds}, "
            f"entities={len(self.entities)}, "
            f"time={self.time:.1f}s)"
        )
