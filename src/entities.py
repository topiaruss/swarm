"""
Entity definitions for the simulation.

Defines the base Entity class and Drone subclass with team/role designations.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional
import numpy as np


class Team(Enum):
    """Team designation for entities."""
    FRIENDLY = "friendly"
    ENEMY = "enemy"
    NEUTRAL = "neutral"


class Role(Enum):
    """Role designation for entities."""
    PATROL = "patrol"
    ATTACK = "attack"
    SUPPORT = "support"
    INTRUDER = "intruder"
    CHARGING_STATION = "charging_station"


@dataclass
class Entity:
    """
    Base class for all entities in the arena.

    Attributes:
        position: 3D position [x, y, z] in meters
        velocity: 3D velocity [vx, vy, vz] in m/s
        team: Team designation (FRIENDLY, ENEMY, NEUTRAL)
        role: Role in the scenario
        id: Unique identifier
        active: Whether entity is active in simulation
    """
    position: np.ndarray
    velocity: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, 0.0]))
    team: Team = Team.NEUTRAL
    role: Role = Role.PATROL
    id: Optional[str] = None
    active: bool = True

    def __post_init__(self):
        """Ensure position and velocity are numpy arrays."""
        self.position = np.array(self.position, dtype=float)
        self.velocity = np.array(self.velocity, dtype=float)

        if self.id is None:
            # Generate unique ID based on team and role
            self.id = f"{self.team.value}_{self.role.value}_{id(self):x}"

    def distance_to(self, other: 'Entity') -> float:
        """Calculate Euclidean distance to another entity."""
        return np.linalg.norm(self.position - other.position)

    def bearing_to(self, other: 'Entity') -> np.ndarray:
        """Calculate unit vector pointing to another entity."""
        diff = other.position - self.position
        dist = np.linalg.norm(diff)
        return diff / dist if dist > 0 else np.array([0.0, 0.0, 0.0])


@dataclass
class Drone(Entity):
    """
    Drone entity with flight-specific properties.

    Attributes:
        max_speed: Maximum speed in m/s
        detection_range: Detection radius in meters
        battery_capacity: Battery capacity in mAh (future use)
        battery_level: Current battery level in mAh (future use)
    """
    max_speed: float = 20.0  # m/s (~45 mph)
    detection_range: float = 200.0  # meters
    battery_capacity: float = 5000.0  # mAh (future use)
    battery_level: float = 5000.0  # mAh (future use)

    def __post_init__(self):
        super().__post_init__()
        # Ensure velocity doesn't exceed max speed
        self.limit_velocity()

    def limit_velocity(self):
        """Limit velocity to max_speed."""
        speed = np.linalg.norm(self.velocity)
        if speed > self.max_speed:
            self.velocity = (self.velocity / speed) * self.max_speed

    def set_velocity(self, new_velocity: np.ndarray):
        """Set velocity with automatic speed limiting."""
        self.velocity = np.array(new_velocity, dtype=float)
        self.limit_velocity()

    def is_friend(self, other: Entity) -> bool:
        """Check if another entity is on the same team."""
        return self.team == other.team

    def is_foe(self, other: Entity) -> bool:
        """Check if another entity is on an opposing team."""
        return (self.team == Team.FRIENDLY and other.team == Team.ENEMY) or \
               (self.team == Team.ENEMY and other.team == Team.FRIENDLY)


@dataclass
class Transponder(Entity):
    """
    Navigation transponder for GPS-denied positioning.

    Broadcasts position and encrypted signals for drone navigation.
    Can relocate periodically to avoid targeting.

    Attributes:
        transmit_power: Transmission power in dBm
        relocate_interval: Time between relocations in seconds
        last_relocate_time: Time of last relocation
        encrypted: Whether transmission is encrypted
        next_position: Next position to move to (if scheduled)
    """
    transmit_power: float = -40.0  # dBm (RSSI at 1m)
    relocate_interval: float = 600.0  # 10 minutes
    last_relocate_time: float = 0.0
    encrypted: bool = True
    next_position: Optional[np.ndarray] = None

    def __post_init__(self):
        super().__post_init__()
        # Transponders are stationary ground units by default
        if self.role == Role.PATROL:
            self.role = Role.SUPPORT
        self.velocity = np.array([0.0, 0.0, 0.0])

    def should_relocate(self, current_time: float) -> bool:
        """Check if it's time to relocate."""
        return (current_time - self.last_relocate_time) >= self.relocate_interval

    def schedule_relocation(self, new_position: np.ndarray):
        """Schedule relocation to a new position."""
        self.next_position = np.array(new_position, dtype=float)

    def execute_relocation(self, current_time: float):
        """Execute scheduled relocation."""
        if self.next_position is not None:
            self.position = self.next_position.copy()
            self.last_relocate_time = current_time
            self.next_position = None
