"""
Physics engine for entity motion.

Implements simple kinematic updates with boundary enforcement.
"""

import numpy as np
from typing import List
from .entities import Entity
from .arena import Arena


class PhysicsEngine:
    """
    Simple kinematic physics engine.

    Updates entity positions based on velocity with boundary enforcement.
    Future extensions: gravity, drag, thrust models, collisions.
    """

    def __init__(self, arena: Arena):
        """
        Initialize physics engine.

        Args:
            arena: Arena instance for boundary checking
        """
        self.arena = arena

    def update(self, entities: List[Entity], dt: float) -> None:
        """
        Update all entity positions.

        Args:
            entities: List of entities to update
            dt: Time step in seconds
        """
        for entity in entities:
            if not entity.active:
                continue

            # Simple kinematic update: position += velocity * dt
            new_position = entity.position + entity.velocity * dt

            # Enforce boundaries (bounce or clamp)
            new_position, new_velocity = self._handle_boundaries(
                new_position,
                entity.velocity
            )

            entity.position = new_position
            entity.velocity = new_velocity

    def _handle_boundaries(
        self,
        position: np.ndarray,
        velocity: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Handle boundary collisions.

        Currently implements clamping (entity stops at boundary).
        Future: Could implement bounce, destruction, or other behaviors.

        Args:
            position: Proposed new position
            velocity: Current velocity

        Returns:
            (corrected_position, corrected_velocity)
        """
        corrected_position = position.copy()
        corrected_velocity = velocity.copy()

        # Check each axis
        for i in range(3):
            if position[i] < 0:
                corrected_position[i] = 0
                corrected_velocity[i] = 0  # Stop at boundary
            elif position[i] > self.arena.bounds[i]:
                corrected_position[i] = self.arena.bounds[i]
                corrected_velocity[i] = 0  # Stop at boundary

        return corrected_position, corrected_velocity

    def set_velocity_towards(
        self,
        entity: Entity,
        target_position: np.ndarray,
        speed: float
    ) -> None:
        """
        Set entity velocity to move towards a target position.

        Args:
            entity: Entity to update
            target_position: Target position
            speed: Desired speed in m/s
        """
        direction = target_position - entity.position
        distance = np.linalg.norm(direction)

        if distance > 0:
            # Unit direction vector * desired speed
            entity.velocity = (direction / distance) * speed
        else:
            entity.velocity = np.array([0.0, 0.0, 0.0])

    def set_circular_path(
        self,
        entity: Entity,
        center: np.ndarray,
        radius: float,
        angular_velocity: float,
        current_time: float
    ) -> None:
        """
        Set entity on a circular path around a center point.

        Args:
            entity: Entity to update
            center: Center of circular path
            radius: Radius of circle in meters
            angular_velocity: Angular velocity in radians/second
            current_time: Current simulation time
        """
        # Calculate angle based on time
        angle = angular_velocity * current_time

        # Calculate position on circle (in XY plane)
        x = center[0] + radius * np.cos(angle)
        y = center[1] + radius * np.sin(angle)
        z = entity.position[2]  # Maintain altitude

        # Calculate tangential velocity
        vx = -radius * angular_velocity * np.sin(angle)
        vy = radius * angular_velocity * np.cos(angle)
        vz = 0.0

        entity.position = np.array([x, y, z])
        entity.velocity = np.array([vx, vy, vz])
