"""
Autonomous strategy and decision-making for drone swarm.

Handles patrol assignments, threat response, and autonomous operation
during network partitions.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Set, Optional, Tuple
from enum import Enum

from src.entities import Drone, Team
from src.network import NetworkState, NetworkMode


class ThreatLevel(Enum):
    """Threat assessment levels."""
    NONE = 0
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4


@dataclass
class ThreatDetection:
    """
    Record of a detected threat.

    Attributes:
        threat_id: Unique threat identifier
        position: Last known position
        timestamp: When detected
        detector_id: ID of drone that detected it
        confidence: Detection confidence (0.0-1.0)
        threat_level: Assessed threat level
        velocity: Estimated velocity (if tracked)
        last_updated: Last update timestamp
    """
    threat_id: str
    position: np.ndarray
    timestamp: float
    detector_id: str
    confidence: float = 1.0
    threat_level: ThreatLevel = ThreatLevel.MEDIUM
    velocity: Optional[np.ndarray] = None
    last_updated: float = 0.0

    def __post_init__(self):
        self.position = np.array(self.position, dtype=float)
        if self.velocity is not None:
            self.velocity = np.array(self.velocity, dtype=float)
        self.last_updated = self.timestamp


@dataclass
class PatrolSector:
    """
    Patrol sector definition.

    Attributes:
        sector_id: Unique sector identifier
        center: Center point of sector
        radius: Sector radius
        assigned_drone: ID of drone assigned (if any)
        priority: Patrol priority (higher = more important)
    """
    sector_id: str
    center: np.ndarray
    radius: float
    assigned_drone: Optional[str] = None
    priority: int = 1

    def __post_init__(self):
        self.center = np.array(self.center, dtype=float)

    def contains_point(self, point: np.ndarray) -> bool:
        """Check if point is within sector."""
        return np.linalg.norm(point - self.center) <= self.radius

    def distance_to(self, point: np.ndarray) -> float:
        """Distance from point to sector center."""
        return np.linalg.norm(point - self.center)


class AutonomousStrategy:
    """
    Autonomous decision-making for a single drone.

    Handles patrol assignment, threat response, and operation during
    network partitions.
    """

    def __init__(
        self,
        drone: Drone,
        network_state: NetworkState,
        patrol_sectors: Optional[List[PatrolSector]] = None
    ):
        """
        Initialize autonomous strategy.

        Args:
            drone: The drone this strategy controls
            network_state: Network state tracker
            patrol_sectors: List of patrol sectors (optional)
        """
        self.drone = drone
        self.network_state = network_state
        self.patrol_sectors = patrol_sectors or []

        # Assigned sector
        self.assigned_sector: Optional[PatrolSector] = None

        # Threat tracking
        self.known_threats: Dict[str, ThreatDetection] = {}
        self.threat_alert_sent = set()  # IDs of threats we've alerted about

        # Last known peer positions (for coordination when disconnected)
        self.last_known_positions: Dict[str, Tuple[np.ndarray, float]] = {}

        # Autonomous operation state
        self.autonomous_mode = False
        self.autonomy_started_at: Optional[float] = None

    def update_peer_position(self, peer_id: str, position: np.ndarray, timestamp: float):
        """Update last known position of a peer."""
        self.last_known_positions[peer_id] = (position.copy(), timestamp)

    def add_threat(
        self,
        threat_id: str,
        position: np.ndarray,
        timestamp: float,
        detector_id: str,
        confidence: float = 1.0,
        threat_level: ThreatLevel = ThreatLevel.MEDIUM
    ):
        """
        Add or update a threat detection.

        Args:
            threat_id: Unique threat identifier
            position: Threat position
            timestamp: Detection timestamp
            detector_id: ID of detecting drone
            confidence: Detection confidence
            threat_level: Assessed threat level
        """
        if threat_id in self.known_threats:
            # Update existing threat
            threat = self.known_threats[threat_id]
            threat.position = np.array(position, dtype=float)
            threat.last_updated = timestamp
            threat.confidence = max(threat.confidence, confidence)  # Increase confidence
        else:
            # New threat
            threat = ThreatDetection(
                threat_id=threat_id,
                position=position,
                timestamp=timestamp,
                detector_id=detector_id,
                confidence=confidence,
                threat_level=threat_level
            )
            self.known_threats[threat_id] = threat

    def get_active_threats(self, current_time: float, max_age: float = 30.0) -> List[ThreatDetection]:
        """
        Get list of active threats (recent detections).

        Args:
            current_time: Current simulation time
            max_age: Maximum age of threat to consider active (seconds)

        Returns:
            List of active threats
        """
        active = []
        for threat in self.known_threats.values():
            age = current_time - threat.last_updated
            if age <= max_age:
                active.append(threat)
        return active

    def should_alert_threat(self, threat_id: str) -> bool:
        """
        Check if we should send alert for this threat.

        Only alert once per threat, unless in autonomous mode where
        we alert independently.
        """
        if threat_id in self.threat_alert_sent:
            # Already alerted
            return False

        if self.network_state.is_partitioned():
            # In partition - always alert (can't assume others got it)
            return True

        # First time seeing this threat
        return True

    def mark_threat_alerted(self, threat_id: str):
        """Mark that we've alerted about this threat."""
        self.threat_alert_sent.add(threat_id)

    def assign_patrol_sector(self, sector: PatrolSector):
        """Assign this drone to a patrol sector."""
        # Unassign from previous sector
        if self.assigned_sector is not None:
            self.assigned_sector.assigned_drone = None

        # Assign new sector
        self.assigned_sector = sector
        sector.assigned_drone = self.drone.id

    def get_patrol_target(self, current_time: float) -> Optional[np.ndarray]:
        """
        Get target position for patrol.

        Returns:
            Target position, or None if no patrol assigned
        """
        if self.assigned_sector is None:
            return None

        # Simple circular patrol around sector center
        # Use time-based circular motion
        angular_velocity = 0.1  # rad/s
        angle = angular_velocity * current_time

        # Patrol at 80% of sector radius
        patrol_radius = self.assigned_sector.radius * 0.8

        offset = np.array([
            patrol_radius * np.cos(angle),
            patrol_radius * np.sin(angle),
            0  # Maintain current altitude
        ])

        target = self.assigned_sector.center + offset
        target[2] = self.drone.position[2]  # Keep same Z

        return target

    def expand_patrol_sector(self, additional_sectors: List[PatrolSector]):
        """
        Expand patrol to cover additional sectors (e.g., when peer goes offline).

        Args:
            additional_sectors: Sectors to add to patrol
        """
        # For now, just patrol between centers
        # More sophisticated implementation would optimize coverage path
        pass

    def enter_autonomous_mode(self, current_time: float):
        """Enter autonomous operation mode (network partitioned)."""
        if not self.autonomous_mode:
            self.autonomous_mode = True
            self.autonomy_started_at = current_time
            print(f"[{self.drone.id}] Entering AUTONOMOUS mode - operating independently")

    def exit_autonomous_mode(self):
        """Exit autonomous mode (network restored)."""
        if self.autonomous_mode:
            self.autonomous_mode = False
            print(f"[{self.drone.id}] Exiting autonomous mode - network restored")

    def get_autonomous_patrol_expansion(self, current_time: float) -> float:
        """
        Calculate patrol radius expansion during autonomy.

        When disconnected, expand patrol slightly to cover potential gaps
        from missing peers.

        Returns:
            Expansion factor (1.0 = normal, 1.2 = 20% larger)
        """
        if not self.autonomous_mode or self.assigned_sector is None:
            return 1.0

        # Expand by 20% to cover potential gaps
        # More sophisticated: expand towards last known positions of disconnected peers
        disconnected_count = len(self.network_state.get_disconnected_peers())
        total_peers = len(self.network_state.expected_peers)

        if total_peers == 0:
            return 1.0

        # Expand proportionally to disconnected fraction
        disconnected_fraction = disconnected_count / total_peers
        expansion = 1.0 + (0.5 * disconnected_fraction)  # Up to 50% expansion

        return expansion

    def get_nearest_threat(self, current_time: float) -> Optional[ThreatDetection]:
        """
        Get nearest active threat.

        Args:
            current_time: Current simulation time

        Returns:
            Nearest threat, or None
        """
        active_threats = self.get_active_threats(current_time)
        if not active_threats:
            return None

        # Find nearest
        nearest = None
        nearest_dist = float('inf')

        for threat in active_threats:
            dist = np.linalg.norm(threat.position - self.drone.position)
            if dist < nearest_dist:
                nearest_dist = dist
                nearest = threat

        return nearest

    def should_track_threat(
        self,
        threat: ThreatDetection,
        current_time: float
    ) -> bool:
        """
        Decide if this drone should track a threat.

        During normal operation, coordinate with peers.
        During partition, track independently.

        Args:
            threat: Threat to evaluate
            current_time: Current simulation time

        Returns:
            True if should track
        """
        # If partitioned, track independently
        if self.network_state.is_partitioned():
            return True

        # If I detected it, I track it (unless someone closer volunteers)
        if threat.detector_id == self.drone.id:
            return True

        # Otherwise, let closer drones handle it
        return False

    def update(self, current_time: float):
        """
        Update autonomous strategy state.

        Args:
            current_time: Current simulation time
        """
        # Check if we should enter/exit autonomous mode
        if self.network_state.is_partitioned():
            if not self.autonomous_mode:
                self.enter_autonomous_mode(current_time)
        else:
            if self.autonomous_mode:
                self.exit_autonomous_mode()

        # Update last known positions from network state
        for peer_id in self.network_state.expected_peers:
            pos = self.network_state.get_last_known_position(peer_id)
            if pos is not None:
                peer_state = self.network_state.peers.get(peer_id)
                if peer_state and peer_state.last_heartbeat:
                    self.update_peer_position(
                        peer_id,
                        pos,
                        peer_state.last_heartbeat.timestamp
                    )

    def get_status_string(self) -> str:
        """Get human-readable status string."""
        mode = "AUTONOMOUS" if self.autonomous_mode else "CONNECTED"
        sector = self.assigned_sector.sector_id if self.assigned_sector else "NONE"
        threats = len(self.get_active_threats(0))  # Pass dummy time

        return f"{mode} | Sector: {sector} | Threats: {threats}"


def create_perimeter_sectors(
    center: np.ndarray,
    radius: float,
    num_sectors: int = 8
) -> List[PatrolSector]:
    """
    Create evenly-spaced patrol sectors around a perimeter.

    Args:
        center: Center point of perimeter
        radius: Radius of perimeter
        num_sectors: Number of sectors to create

    Returns:
        List of patrol sectors
    """
    sectors = []

    for i in range(num_sectors):
        angle = (i / num_sectors) * 2 * np.pi

        # Sector center is on the perimeter circle
        sector_center = center + np.array([
            radius * np.cos(angle),
            radius * np.sin(angle),
            0  # Ground level sectors
        ])

        # Each sector covers an arc
        sector_radius = (2 * np.pi * radius) / (2 * num_sectors)  # Half arc length

        sector = PatrolSector(
            sector_id=f"SECTOR-{i+1}",
            center=sector_center,
            radius=sector_radius,
            priority=1
        )

        sectors.append(sector)

    return sectors
