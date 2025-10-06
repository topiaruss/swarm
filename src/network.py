"""
Network state tracking and partition detection.

Manages connectivity monitoring, heartbeat protocol, and partition detection
for mesh network reliability.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, Set, List, Optional, Tuple
from enum import Enum


class NetworkMode(Enum):
    """Network operation modes."""
    CONNECTED = "connected"      # Normal operation, full connectivity
    PARTITIONED = "partitioned"  # Network split detected
    ISOLATED = "isolated"        # Completely alone


@dataclass
class HeartbeatInfo:
    """
    Heartbeat information from a peer.

    Attributes:
        drone_id: ID of the drone
        timestamp: When heartbeat was sent
        position: Drone position
        battery_level: Battery charge (0.0-1.0)
        role_status: Current role (PATROL, CHARGING, etc.)
        sequence_number: Monotonic counter for message ordering
    """
    drone_id: str
    timestamp: float
    position: np.ndarray
    battery_level: float
    role_status: str
    sequence_number: int


@dataclass
class PeerState:
    """
    State tracking for a single peer.

    Attributes:
        peer_id: ID of the peer
        last_heartbeat: Most recent heartbeat info
        last_seen: Timestamp of last heartbeat
        connected: Whether currently connected
        sequence_number: Last sequence number seen
        missed_heartbeats: Consecutive missed heartbeats
    """
    peer_id: str
    last_heartbeat: Optional[HeartbeatInfo] = None
    last_seen: float = 0.0
    connected: bool = False
    sequence_number: int = 0
    missed_heartbeats: int = 0


class NetworkState:
    """
    Network state tracker for a single drone.

    Monitors connectivity to peers via heartbeat protocol and detects
    network partitions.
    """

    # Configuration constants
    HEARTBEAT_INTERVAL = 1.0      # Send heartbeat every 1 second
    HEARTBEAT_TIMEOUT = 5.0       # Consider disconnected after 5 seconds
    MAX_MISSED_HEARTBEATS = 5     # 5 consecutive misses = disconnected

    def __init__(self, drone_id: str, expected_peers: Set[str]):
        """
        Initialize network state tracker.

        Args:
            drone_id: ID of this drone
            expected_peers: Set of peer IDs we expect to see
        """
        self.drone_id = drone_id
        self.expected_peers = expected_peers.copy()

        # Peer tracking
        self.peers: Dict[str, PeerState] = {}
        for peer_id in expected_peers:
            self.peers[peer_id] = PeerState(peer_id=peer_id)

        # Network state
        self.mode = NetworkMode.CONNECTED
        self.partition_detected_at: Optional[float] = None
        self.partition_id = 0  # Which partition am I in?

        # Heartbeat management
        self.last_heartbeat_sent = 0.0
        self.own_sequence_number = 0

        # Statistics
        self.partition_count = 0
        self.rejoin_count = 0
        self.total_disconnects = 0

    def get_connected_peers(self) -> Set[str]:
        """Get set of currently connected peer IDs."""
        return {peer_id for peer_id, peer in self.peers.items() if peer.connected}

    def get_disconnected_peers(self) -> Set[str]:
        """Get set of currently disconnected peer IDs."""
        return {peer_id for peer_id, peer in self.peers.items() if not peer.connected}

    def is_connected_to(self, peer_id: str) -> bool:
        """Check if connected to specific peer."""
        return peer_id in self.peers and self.peers[peer_id].connected

    def is_partitioned(self) -> bool:
        """Check if network is currently partitioned."""
        return self.mode == NetworkMode.PARTITIONED or self.mode == NetworkMode.ISOLATED

    def update_heartbeat(
        self,
        peer_id: str,
        heartbeat: HeartbeatInfo,
        current_time: float
    ):
        """
        Process a heartbeat from a peer.

        Args:
            peer_id: ID of peer sending heartbeat
            heartbeat: Heartbeat information
            current_time: Current simulation time
        """
        if peer_id not in self.peers:
            # Unknown peer - add dynamically
            self.peers[peer_id] = PeerState(peer_id=peer_id)
            self.expected_peers.add(peer_id)

        peer = self.peers[peer_id]

        # Check if this is a reconnection
        was_disconnected = not peer.connected

        # Update peer state
        peer.last_heartbeat = heartbeat
        peer.last_seen = current_time
        peer.sequence_number = heartbeat.sequence_number
        peer.missed_heartbeats = 0

        if not peer.connected:
            peer.connected = True

            if was_disconnected and peer.last_seen > 0:
                # This is a rejoin
                self.rejoin_count += 1
                print(f"[{self.drone_id}] Peer {peer_id} reconnected (seq={heartbeat.sequence_number})")

    def should_send_heartbeat(self, current_time: float) -> bool:
        """Check if it's time to send a heartbeat."""
        return (current_time - self.last_heartbeat_sent) >= self.HEARTBEAT_INTERVAL

    def create_heartbeat(
        self,
        current_time: float,
        position: np.ndarray,
        battery_level: float,
        role_status: str
    ) -> HeartbeatInfo:
        """
        Create a heartbeat to broadcast.

        Args:
            current_time: Current simulation time
            position: Current position
            battery_level: Current battery level (0.0-1.0)
            role_status: Current role status string

        Returns:
            HeartbeatInfo ready to send
        """
        self.own_sequence_number += 1
        self.last_heartbeat_sent = current_time

        return HeartbeatInfo(
            drone_id=self.drone_id,
            timestamp=current_time,
            position=position.copy(),
            battery_level=battery_level,
            role_status=role_status,
            sequence_number=self.own_sequence_number
        )

    def detect_partitions(self, current_time: float) -> bool:
        """
        Detect network partitions based on missed heartbeats.

        Args:
            current_time: Current simulation time

        Returns:
            True if partition state changed
        """
        old_mode = self.mode
        connected_count = 0

        # Check each peer for timeout
        for peer_id, peer in self.peers.items():
            time_since_seen = current_time - peer.last_seen

            if peer.connected and time_since_seen > self.HEARTBEAT_TIMEOUT:
                # Peer has timed out
                peer.connected = False
                peer.missed_heartbeats += 1
                self.total_disconnects += 1
                print(f"[{self.drone_id}] Lost connection to {peer_id} (timeout={time_since_seen:.1f}s)")

            if peer.connected:
                connected_count += 1

        # Determine network mode
        total_peers = len(self.expected_peers)

        if connected_count == total_peers:
            # Full connectivity
            if old_mode != NetworkMode.CONNECTED and self.partition_detected_at is not None:
                print(f"[{self.drone_id}] Network fully restored - all {total_peers} peers connected")
            self.mode = NetworkMode.CONNECTED
            self.partition_detected_at = None

        elif connected_count == 0:
            # Completely isolated
            if old_mode != NetworkMode.ISOLATED:
                print(f"[{self.drone_id}] ISOLATED - no peers reachable!")
                self.partition_detected_at = current_time
                self.partition_count += 1
            self.mode = NetworkMode.ISOLATED

        else:
            # Partial connectivity = partition
            if old_mode == NetworkMode.CONNECTED:
                print(f"[{self.drone_id}] PARTITION detected - {connected_count}/{total_peers} peers reachable")
                self.partition_detected_at = current_time
                self.partition_count += 1
            self.mode = NetworkMode.PARTITIONED

        # Return True if mode changed
        return old_mode != self.mode

    def get_partition_info(self) -> Dict:
        """Get information about current partition state."""
        connected = self.get_connected_peers()
        disconnected = self.get_disconnected_peers()

        partition_duration = None
        if self.partition_detected_at is not None:
            partition_duration = self.partition_detected_at

        return {
            'mode': self.mode.value,
            'connected_peers': list(connected),
            'disconnected_peers': list(disconnected),
            'connected_count': len(connected),
            'total_peers': len(self.expected_peers),
            'partition_detected_at': self.partition_detected_at,
            'partition_duration': partition_duration,
            'partition_count': self.partition_count,
            'rejoin_count': self.rejoin_count
        }

    def get_last_known_position(self, peer_id: str) -> Optional[np.ndarray]:
        """
        Get last known position of a peer.

        Args:
            peer_id: ID of peer

        Returns:
            Last known position, or None if never seen
        """
        if peer_id in self.peers:
            peer = self.peers[peer_id]
            if peer.last_heartbeat is not None:
                return peer.last_heartbeat.position.copy()
        return None

    def get_position_uncertainty(
        self,
        peer_id: str,
        current_time: float,
        drift_rate: float = 5.0
    ) -> float:
        """
        Estimate position uncertainty for a peer based on time since last seen.

        Args:
            peer_id: ID of peer
            current_time: Current simulation time
            drift_rate: Uncertainty growth rate (m/s)

        Returns:
            Estimated position uncertainty in meters
        """
        if peer_id not in self.peers:
            return float('inf')

        peer = self.peers[peer_id]

        if not peer.connected:
            # Disconnected - uncertainty grows with time
            time_disconnected = current_time - peer.last_seen
            return 10.0 + drift_rate * time_disconnected  # Base 10m + drift
        else:
            # Connected - low uncertainty
            return 5.0  # Base uncertainty from sensor noise

    def get_statistics(self) -> Dict:
        """Get network statistics."""
        return {
            'mode': self.mode.value,
            'connected_peers': len(self.get_connected_peers()),
            'total_peers': len(self.expected_peers),
            'partition_count': self.partition_count,
            'rejoin_count': self.rejoin_count,
            'total_disconnects': self.total_disconnects,
            'sequence_number': self.own_sequence_number
        }
