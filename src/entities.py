"""
Entity definitions for the simulation.

Defines the base Entity class and Drone subclass with team/role designations.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Dict, List
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
class Battery:
    """
    Battery model for drones.

    Tracks charge level, discharge rates, and provides utility functions
    for battery management.

    Attributes:
        capacity: Maximum charge capacity in mAh
        current_charge: Current charge level in mAh
        discharge_rate_base: Base discharge rate in mAh/s (patrol/cruise)
        critical_level: Critical battery threshold (0.0-1.0)
        optimal_level: Optimal battery threshold (0.0-1.0)
        full_level: Consider fully charged threshold (0.0-1.0)
    """
    capacity: float = 5000.0  # mAh
    current_charge: float = 5000.0  # mAh
    discharge_rate_base: float = 50.0  # mAh/s (100s flight time at full)
    critical_level: float = 0.15  # 15%
    optimal_level: float = 0.80  # 80%
    full_level: float = 0.95  # 95%

    def level(self) -> float:
        """
        Get battery level as percentage (0.0 to 1.0).

        Returns:
            Battery percentage
        """
        return self.current_charge / self.capacity

    def discharge(self, dt: float, rate_multiplier: float = 1.0):
        """
        Discharge battery over time interval.

        Args:
            dt: Time interval in seconds
            rate_multiplier: Multiplier for discharge rate
                           (1.0 = patrol, 1.6 = tracking, 0.8 = hovering)
        """
        discharge_amount = self.discharge_rate_base * rate_multiplier * dt
        self.current_charge = max(0.0, self.current_charge - discharge_amount)

    def charge(self, dt: float, charge_rate: float):
        """
        Charge battery over time interval.

        Args:
            dt: Time interval in seconds
            charge_rate: Charge rate in % per second (e.g., 0.02 = 2%/s)
        """
        charge_amount = self.capacity * charge_rate * dt
        self.current_charge = min(self.capacity, self.current_charge + charge_amount)

    def time_to_empty(self, rate_multiplier: float = 1.0) -> float:
        """
        Calculate time until battery empty at given discharge rate.

        Args:
            rate_multiplier: Discharge rate multiplier

        Returns:
            Time in seconds until empty
        """
        rate = self.discharge_rate_base * rate_multiplier
        if rate <= 0:
            return float('inf')
        return self.current_charge / rate

    def time_to_full(self, charge_rate: float) -> float:
        """
        Calculate time until battery full at given charge rate.

        Args:
            charge_rate: Charge rate in % per second

        Returns:
            Time in seconds until full
        """
        if charge_rate <= 0:
            return float('inf')
        charge_needed = self.capacity - self.current_charge
        charge_per_second = self.capacity * charge_rate
        return charge_needed / charge_per_second

    def is_critical(self) -> bool:
        """Check if battery is at critical level."""
        return self.level() < self.critical_level

    def is_below_optimal(self) -> bool:
        """Check if battery is below optimal level."""
        return self.level() < self.optimal_level

    def is_full(self) -> bool:
        """Check if battery is considered fully charged."""
        return self.level() >= self.full_level


@dataclass
class Drone(Entity):
    """
    Drone entity with flight-specific properties.

    Attributes:
        max_speed: Maximum speed in m/s
        detection_range: Detection radius in meters
        battery: Battery instance for power management
        battery_capacity: Legacy attribute (use battery.capacity instead)
        battery_level: Legacy attribute (use battery.current_charge instead)
        charging_slot: Slot number if currently charging, None otherwise
    """
    max_speed: float = 20.0  # m/s (~45 mph)
    detection_range: float = 200.0  # meters
    battery: Optional[Battery] = None
    battery_capacity: float = 5000.0  # Legacy (for compatibility)
    battery_level: float = 5000.0  # Legacy (for compatibility)
    charging_slot: Optional[int] = None  # Slot at charging station

    def __post_init__(self):
        super().__post_init__()

        # Create Battery instance if not provided
        if self.battery is None:
            self.battery = Battery(
                capacity=self.battery_capacity,
                current_charge=self.battery_level
            )

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


@dataclass
class ChargingStation(Entity):
    """
    Charging station for drones.

    Provides charging slots for multiple drones simultaneously.
    Drones on charger have radio transmitter disabled (receive-only).

    Attributes:
        capacity: Number of simultaneous charging slots
        charge_rate: Charge rate in % per second (e.g., 0.02 = 2%/s = 50s for full)
        occupied_slots: Dict mapping slot number to drone ID
        queue: List of drone IDs waiting for available slot
        slot_positions: Dict mapping slot number to 3D position offset from station center
    """
    capacity: int = 4  # Number of slots
    charge_rate: float = 0.02  # 2% per second = 50s for full charge
    occupied_slots: Dict[int, str] = field(default_factory=dict)
    queue: List[str] = field(default_factory=list)
    slot_positions: Dict[int, np.ndarray] = field(default_factory=dict)

    def __post_init__(self):
        super().__post_init__()
        self.role = Role.CHARGING_STATION
        self.team = Team.FRIENDLY

        # Initialize slot positions in a circle around station
        if not self.slot_positions:
            radius = 5.0  # 5m radius around center
            for i in range(self.capacity):
                angle = (i / self.capacity) * 2 * np.pi
                offset = np.array([
                    radius * np.cos(angle),
                    radius * np.sin(angle),
                    0.0  # Ground level
                ])
                self.slot_positions[i] = offset

    def has_available_slot(self) -> bool:
        """Check if station has an available charging slot."""
        return len(self.occupied_slots) < self.capacity

    def get_available_slot(self) -> Optional[int]:
        """
        Get next available slot number.

        Returns:
            Slot number if available, None otherwise
        """
        if not self.has_available_slot():
            return None

        for slot in range(self.capacity):
            if slot not in self.occupied_slots:
                return slot
        return None

    def request_slot(self, drone_id: str) -> Optional[int]:
        """
        Request a charging slot for a drone.

        Args:
            drone_id: ID of drone requesting slot

        Returns:
            Slot number if assigned, None if no slots available
        """
        if not self.has_available_slot():
            # Add to queue if not already there
            if drone_id not in self.queue:
                self.queue.append(drone_id)
            return None

        # Assign available slot
        slot = self.get_available_slot()
        if slot is not None:
            self.occupied_slots[slot] = drone_id
            # Remove from queue if was waiting
            if drone_id in self.queue:
                self.queue.remove(drone_id)
        return slot

    def release_slot(self, slot: int) -> bool:
        """
        Release a charging slot.

        Args:
            slot: Slot number to release

        Returns:
            True if slot was occupied and released, False otherwise
        """
        if slot in self.occupied_slots:
            del self.occupied_slots[slot]
            return True
        return False

    def get_drone_slot(self, drone_id: str) -> Optional[int]:
        """
        Get slot number for a specific drone.

        Args:
            drone_id: ID of drone

        Returns:
            Slot number if drone is charging, None otherwise
        """
        for slot, did in self.occupied_slots.items():
            if did == drone_id:
                return slot
        return None

    def get_slot_position(self, slot: int) -> np.ndarray:
        """
        Get absolute position of a charging slot.

        Args:
            slot: Slot number

        Returns:
            3D position of slot
        """
        offset = self.slot_positions.get(slot, np.array([0.0, 0.0, 0.0]))
        return self.position + offset

    def charge_drone(self, drone: 'Drone', dt: float):
        """
        Charge a drone's battery.

        Args:
            drone: Drone to charge
            dt: Time interval in seconds
        """
        if hasattr(drone, 'battery') and isinstance(drone.battery, Battery):
            drone.battery.charge(dt, self.charge_rate)

    def get_queue_position(self, drone_id: str) -> Optional[int]:
        """
        Get position of drone in queue.

        Args:
            drone_id: ID of drone

        Returns:
            Queue position (0 = next), None if not in queue
        """
        if drone_id in self.queue:
            return self.queue.index(drone_id)
        return None
