"""
Detection and sensor systems.

Implements range-based detection with friend/foe identification.
"""

from dataclasses import dataclass
from typing import List
import numpy as np
from .entities import Entity, Drone


@dataclass
class Detection:
    """
    Represents a detected entity.

    Attributes:
        entity: The detected entity
        distance: Distance to entity in meters
        bearing: Unit vector pointing to entity
        is_friendly: Whether entity is friendly
    """
    entity: Entity
    distance: float
    bearing: np.ndarray
    is_friendly: bool


class Detector:
    """
    Base detector class for sensor systems.

    Implements simple range-based detection.
    Future extensions: cone-based (camera), radar, audio sensors.
    """

    def __init__(self, max_range: float = 200.0):
        """
        Initialize detector.

        Args:
            max_range: Maximum detection range in meters
        """
        self.max_range = max_range

    def detect(self, observer: Entity, entities: List[Entity]) -> List[Detection]:
        """
        Detect entities within range.

        Args:
            observer: Entity doing the detecting
            entities: List of all entities in arena

        Returns:
            List of Detection objects
        """
        detections = []

        for entity in entities:
            # Don't detect self or inactive entities
            if entity is observer or not entity.active:
                continue

            # Calculate distance
            distance = observer.distance_to(entity)

            # Check if within range
            if distance <= self.max_range:
                bearing = observer.bearing_to(entity)

                # Determine if friendly (if both are Drones)
                is_friendly = False
                if isinstance(observer, Drone) and isinstance(entity, Drone):
                    is_friendly = observer.is_friend(entity)

                detections.append(Detection(
                    entity=entity,
                    distance=distance,
                    bearing=bearing,
                    is_friendly=is_friendly
                ))

        # Sort by distance (closest first)
        detections.sort(key=lambda d: d.distance)
        return detections

    def filter_foes(self, detections: List[Detection]) -> List[Detection]:
        """Filter detections to only hostile entities."""
        return [d for d in detections if not d.is_friendly]

    def filter_friends(self, detections: List[Detection]) -> List[Detection]:
        """Filter detections to only friendly entities."""
        return [d for d in detections if d.is_friendly]


class RadarDetector(Detector):
    """
    Radar-based detector (future extension).

    Longer range, 360° coverage, but lower resolution.
    """
    def __init__(self, max_range: float = 500.0):
        super().__init__(max_range)


class CameraDetector(Detector):
    """
    Camera-based detector (future extension).

    Cone-based field of view, requires line of sight.
    """
    def __init__(self, max_range: float = 300.0, fov_degrees: float = 90.0):
        super().__init__(max_range)
        self.fov = np.radians(fov_degrees)

    # Future: implement cone-based detection with LOS checks


class AudioDetector(Detector):
    """
    Audio-based detector (future extension).

    Short range, 360° coverage.
    """
    def __init__(self, max_range: float = 100.0):
        super().__init__(max_range)
