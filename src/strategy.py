"""
Autonomous strategy and decision-making for drone swarm.

Handles patrol assignments, threat response, and autonomous operation
during network partitions.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Set, Optional, Tuple
from enum import Enum

from src.entities import Drone, Team, ChargingStation, Battery
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

        # State sync tracking
        self.peers_to_sync: Set[str] = set()  # Peers we need to sync with

        # Patrol handoff tracking
        self.covering_sectors: List[PatrolSector] = []  # Additional sectors we're covering
        self.handoff_pending: bool = False  # Waiting for handoff before charging
        self.handoff_volunteer: Optional[str] = None  # Peer who volunteered to cover

    def update_peer_position(self, peer_id: str, position: np.ndarray, timestamp: float):
        """Update last known position of a peer."""
        self.last_known_positions[peer_id] = (position.copy(), timestamp)

    def request_state_sync(self, peer_id: str, current_time: float) -> Dict:
        """
        Create state sync request for a newly reconnected peer.

        Args:
            peer_id: ID of peer to sync with
            current_time: Current simulation time

        Returns:
            Dictionary containing state sync request data
        """
        return {
            'requester_id': self.drone.id,
            'timestamp': current_time,
            'threat_count': len(self.known_threats),
            'threats': [
                {
                    'threat_id': threat.threat_id,
                    'position': threat.position.tolist(),
                    'timestamp': threat.timestamp,
                    'detector_id': threat.detector_id,
                    'confidence': threat.confidence,
                    'threat_level': threat.threat_level.value,
                    'last_updated': threat.last_updated
                }
                for threat in self.known_threats.values()
            ]
        }

    def merge_threat_from_sync(
        self,
        threat_data: Dict,
        current_time: float
    ):
        """
        Merge a threat from state sync response.

        Args:
            threat_data: Threat data dictionary
            current_time: Current simulation time
        """
        threat_id = threat_data['threat_id']

        if threat_id in self.known_threats:
            # Update existing threat if this is more recent
            existing = self.known_threats[threat_id]
            if threat_data['last_updated'] > existing.last_updated:
                existing.position = np.array(threat_data['position'])
                existing.last_updated = threat_data['last_updated']
                existing.confidence = max(existing.confidence, threat_data['confidence'])
                print(f"[{self.drone.id}] Updated threat {threat_id} from state sync")
        else:
            # New threat we didn't know about
            self.add_threat(
                threat_id=threat_id,
                position=np.array(threat_data['position']),
                timestamp=threat_data['timestamp'],
                detector_id=threat_data['detector_id'],
                confidence=threat_data['confidence'],
                threat_level=ThreatLevel(threat_data['threat_level'])
            )
            print(f"[{self.drone.id}] Learned about threat {threat_id} from state sync")

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

    def on_peer_reconnected(self, peer_id: str):
        """
        Called when a peer reconnects after partition.

        Args:
            peer_id: ID of reconnected peer
        """
        # Add to sync list if not already there
        if peer_id not in self.peers_to_sync:
            self.peers_to_sync.add(peer_id)
            print(f"[{self.drone.id}] Will sync state with {peer_id}")

    def get_peers_needing_sync(self) -> Set[str]:
        """Get set of peers that need state sync."""
        return self.peers_to_sync.copy()

    def mark_peer_synced(self, peer_id: str):
        """Mark peer as synced."""
        if peer_id in self.peers_to_sync:
            self.peers_to_sync.remove(peer_id)

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

        # Detect reconnections and trigger state sync
        for peer_id in self.network_state.expected_peers:
            peer_state = self.network_state.peers.get(peer_id)
            if peer_state:
                # Check if this peer just reconnected (rejoin_count changed)
                # We track this in the network_state, check if connected but wasn't before
                if peer_state.connected and peer_id not in self.last_known_positions:
                    # First time seeing this peer since we started, or it reconnected
                    self.on_peer_reconnected(peer_id)

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

    def calculate_return_energy(self, station: ChargingStation) -> float:
        """
        Calculate energy required to return to charging station.

        Args:
            station: Charging station to return to

        Returns:
            Energy required in mAh
        """
        distance = np.linalg.norm(self.drone.position - station.position)

        # Estimate flight time at cruise speed (half max speed)
        cruise_speed = self.drone.max_speed * 0.5
        flight_time = distance / cruise_speed if cruise_speed > 0 else 0

        # Energy consumption at patrol rate
        if hasattr(self.drone, 'battery') and isinstance(self.drone.battery, Battery):
            energy_required = self.drone.battery.discharge_rate_base * flight_time
            return energy_required

        return 0.0

    def should_return_to_charge(self, station: ChargingStation) -> bool:
        """
        Determine if drone should return to charging station.

        Args:
            station: Charging station

        Returns:
            True if should return to charge
        """
        if not hasattr(self.drone, 'battery') or not isinstance(self.drone.battery, Battery):
            return False

        battery = self.drone.battery
        battery_pct = battery.level()

        # Critical: MUST return immediately
        if battery.is_critical():
            return True

        # Calculate return energy cost
        energy_to_return = self.calculate_return_energy(station)
        energy_margin = battery.current_charge - energy_to_return

        # Not enough to return safely + 10% margin
        margin_threshold = 0.1 * battery.capacity
        if energy_margin < margin_threshold:
            return True

        # Below optimal AND charging slot available
        if battery.is_below_optimal() and station.has_available_slot():
            return True

        # Below optimal AND we're lower than anyone in queue
        if battery.is_below_optimal() and len(station.queue) > 0:
            # Check if we should jump the queue (we're more critical)
            # This would require access to other drones' battery levels
            # For now, just return True if below optimal
            pass

        return False

    def should_leave_charger(
        self,
        station: ChargingStation,
        other_drones: Optional[List[Drone]] = None
    ) -> bool:
        """
        Determine if drone should leave charging station.

        Args:
            station: Charging station
            other_drones: List of other drones (to check for critical batteries)

        Returns:
            True if should leave charger
        """
        if not hasattr(self.drone, 'battery') or not isinstance(self.drone.battery, Battery):
            return False

        battery = self.drone.battery
        battery_pct = battery.level()

        # Fully charged
        if battery.is_full():
            return True

        # Someone waiting with critical battery
        if other_drones:
            for other in other_drones:
                if (hasattr(other, 'battery') and
                    isinstance(other.battery, Battery) and
                    other.battery.is_critical()):
                    # If I'm at optimal level, let critical drone charge
                    if battery_pct >= battery.optimal_level:
                        return True

        # Good enough (80%) and someone is waiting
        if battery_pct >= battery.optimal_level and len(station.queue) > 0:
            return True

        return False

    def announce_charge_intent(self, current_time: float, station: ChargingStation) -> Dict:
        """
        Announce intention to charge and request patrol handoff.

        Args:
            current_time: Current simulation time
            station: Charging station

        Returns:
            Dictionary containing charge request data
        """
        if not hasattr(self.drone, 'battery') or not isinstance(self.drone.battery, Battery):
            return {}

        energy_to_return = self.calculate_return_energy(station)
        estimated_departure = current_time + (energy_to_return / self.drone.battery.discharge_rate_base)

        return {
            'drone_id': self.drone.id,
            'battery_level': self.drone.battery.level(),
            'estimated_departure_time': estimated_departure,
            'patrol_sector': self.assigned_sector.sector_id if self.assigned_sector else None,
            'sector_center': self.assigned_sector.center.tolist() if self.assigned_sector else None,
            'handoff_needed': self.assigned_sector is not None
        }

    def can_cover_handoff(self, request_data: Dict) -> bool:
        """
        Evaluate if this drone can cover another drone's patrol sector.

        Args:
            request_data: Charge request from peer

        Returns:
            True if can cover the handoff
        """
        if not hasattr(self.drone, 'battery') or not isinstance(self.drone.battery, Battery):
            return False

        # Already covering too many sectors
        if len(self.covering_sectors) >= 2:
            return False

        # Battery too low to take on extra work
        if self.drone.battery.is_below_optimal():
            return False

        # Not assigned to a sector ourselves
        if self.assigned_sector is None:
            return False

        # Check if request sector is adjacent to our assigned sector
        if request_data.get('sector_center') and self.assigned_sector:
            request_center = np.array(request_data['sector_center'])
            distance = np.linalg.norm(self.assigned_sector.center - request_center)

            # Adjacent if within ~2x sector radius
            max_adjacent_distance = self.assigned_sector.radius * 3
            if distance > max_adjacent_distance:
                return False

        return True

    def accept_handoff(self, request_data: Dict, current_time: float) -> Dict:
        """
        Accept patrol handoff from another drone.

        Args:
            request_data: Charge request from peer
            current_time: Current simulation time

        Returns:
            Dictionary containing handoff accept data
        """
        # Find the sector being handed off
        sector_id = request_data.get('patrol_sector')
        handoff_sector = None

        for sector in self.patrol_sectors:
            if sector.sector_id == sector_id:
                handoff_sector = sector
                break

        if handoff_sector:
            self.covering_sectors.append(handoff_sector)

        return {
            'drone_id': self.drone.id,
            'battery_level': self.drone.battery.level() if hasattr(self.drone, 'battery') else 1.0,
            'sector_accepted': sector_id,
            'timestamp': current_time
        }

    def release_handoff_coverage(self, sector_id: str):
        """
        Release coverage of a handed-off sector.

        Args:
            sector_id: Sector ID to release
        """
        self.covering_sectors = [s for s in self.covering_sectors if s.sector_id != sector_id]

    def get_all_patrol_sectors(self) -> List[PatrolSector]:
        """
        Get all sectors this drone is currently patrolling.

        Returns:
            List of sectors (assigned + covering)
        """
        sectors = []
        if self.assigned_sector:
            sectors.append(self.assigned_sector)
        sectors.extend(self.covering_sectors)
        return sectors


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
