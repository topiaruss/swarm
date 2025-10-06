"""
Mesh communications network for drone swarm.

Simulates realistic radio mesh networking with:
- Range-based connectivity (ESP-NOW, LoRa characteristics)
- Latency and packet loss
- Multi-hop routing
- Network splits and healing
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Set, Optional, Tuple
from enum import Enum
import time


class MessageType(Enum):
    """Types of messages in the mesh network."""
    HEARTBEAT = "heartbeat"         # Connectivity heartbeat
    POSITION = "position"           # Position broadcast
    THREAT_ALERT = "threat_alert"   # Enemy detected
    STATE_SYNC_REQUEST = "state_sync_request"  # Request state sync on rejoin
    STATE_SYNC_RESPONSE = "state_sync_response"  # Respond with state data
    STATUS = "status"               # Battery, health status
    COMMAND = "command"             # Coordination commands
    ACK = "ack"                     # Acknowledgment


class RadioType(Enum):
    """Radio module characteristics."""
    ESP_NOW = "esp_now"    # ESP32 ESP-NOW: 200-400m, fast
    LORA = "lora"          # LoRa: 2-10km, slow
    NRF24 = "nrf24"        # nRF24L01+: 100m-1km, cheap
    XBEE = "xbee"          # XBee Pro: 1-3km, reliable


@dataclass
class Message:
    """
    Message in the mesh network.

    Attributes:
        msg_id: Unique message ID
        msg_type: Type of message
        sender_id: ID of sender
        data: Message payload (dict)
        timestamp: Creation time
        ttl: Time-to-live (hops remaining)
        path: IDs of nodes this message has traversed
    """
    msg_id: str
    msg_type: MessageType
    sender_id: str
    data: dict
    timestamp: float
    ttl: int = 5  # Max hops
    path: List[str] = field(default_factory=list)

    def __post_init__(self):
        if self.sender_id not in self.path:
            self.path.append(self.sender_id)


@dataclass
class RadioConfig:
    """Radio module configuration."""
    radio_type: RadioType
    max_range: float         # meters
    bandwidth: float         # bits/second
    base_latency: float      # seconds
    packet_loss_rate: float  # 0.0-1.0 at max range
    power_tx: float          # mA when transmitting
    power_rx: float          # mA when receiving


# Preset radio configurations
RADIO_PRESETS = {
    RadioType.ESP_NOW: RadioConfig(
        radio_type=RadioType.ESP_NOW,
        max_range=300.0,        # 300m typical
        bandwidth=250000,       # 250 kbps
        base_latency=0.010,     # 10ms
        packet_loss_rate=0.05,  # 5% at max range
        power_tx=80.0,          # 80mA
        power_rx=30.0           # 30mA
    ),
    RadioType.LORA: RadioConfig(
        radio_type=RadioType.LORA,
        max_range=3000.0,       # 3km typical
        bandwidth=5000,         # 5 kbps
        base_latency=0.100,     # 100ms
        packet_loss_rate=0.10,  # 10% at max range
        power_tx=120.0,         # 120mA
        power_rx=15.0           # 15mA
    ),
    RadioType.NRF24: RadioConfig(
        radio_type=RadioType.NRF24,
        max_range=500.0,        # 500m with PA/LNA
        bandwidth=250000,       # 250 kbps
        base_latency=0.005,     # 5ms
        packet_loss_rate=0.08,  # 8% at max range
        power_tx=12.0,          # 12mA
        power_rx=13.0           # 13mA
    ),
    RadioType.XBEE: RadioConfig(
        radio_type=RadioType.XBEE,
        max_range=2000.0,       # 2km
        bandwidth=10000,        # 10 kbps
        base_latency=0.030,     # 30ms
        packet_loss_rate=0.02,  # 2% at max range (very reliable)
        power_tx=45.0,          # 45mA
        power_rx=45.0           # 45mA
    )
}


class MeshNode:
    """
    Node in the mesh network (attached to a drone/entity).

    Handles message transmission, reception, and routing.
    """

    def __init__(
        self,
        node_id: str,
        radio_config: RadioConfig,
        position: np.ndarray
    ):
        """
        Initialize mesh node.

        Args:
            node_id: Unique node identifier
            radio_config: Radio module configuration
            position: 3D position of node
        """
        self.node_id = node_id
        self.radio = radio_config
        self.position = position

        # Message handling
        self.outbox: List[Message] = []
        self.inbox: List[Message] = []
        self.seen_messages: Set[str] = set()  # For duplicate detection

        # Routing
        self.neighbors: Set[str] = set()  # Direct neighbors in range
        self.routing_table: Dict[str, str] = {}  # dest -> next_hop

        # Statistics
        self.messages_sent = 0
        self.messages_received = 0
        self.messages_dropped = 0

    def send_message(
        self,
        msg_type: MessageType,
        data: dict,
        timestamp: float
    ) -> Message:
        """
        Create and queue a message for transmission.

        Args:
            msg_type: Type of message
            data: Message payload
            timestamp: Current simulation time

        Returns:
            Created message
        """
        msg_id = f"{self.node_id}_{timestamp}_{np.random.randint(10000)}"

        msg = Message(
            msg_id=msg_id,
            msg_type=msg_type,
            sender_id=self.node_id,
            data=data,
            timestamp=timestamp,
            ttl=5
        )

        self.outbox.append(msg)
        self.messages_sent += 1
        return msg

    def receive_message(self, msg: Message) -> bool:
        """
        Receive a message (if not duplicate).

        Args:
            msg: Incoming message

        Returns:
            True if accepted, False if duplicate
        """
        if msg.msg_id in self.seen_messages:
            return False  # Duplicate

        self.seen_messages.add(msg.msg_id)
        self.inbox.append(msg)
        self.messages_received += 1
        return True

    def should_forward(self, msg: Message) -> bool:
        """Check if message should be forwarded."""
        return (
            msg.sender_id != self.node_id and  # Not our own message
            msg.ttl > 0 and                     # Still has hops left
            self.node_id not in msg.path        # We haven't forwarded it
        )

    def forward_message(self, msg: Message) -> Message:
        """
        Create forwarded copy of message.

        Args:
            msg: Message to forward

        Returns:
            Forwarded message with decremented TTL
        """
        forwarded = Message(
            msg_id=msg.msg_id,
            msg_type=msg.msg_type,
            sender_id=msg.sender_id,
            data=msg.data,
            timestamp=msg.timestamp,
            ttl=msg.ttl - 1,
            path=msg.path.copy()
        )
        forwarded.path.append(self.node_id)
        self.outbox.append(forwarded)
        return forwarded


class MeshNetwork:
    """
    Mesh network managing all nodes and message routing.

    Handles connectivity, latency, packet loss, and multi-hop routing.
    """

    def __init__(self, radio_type: RadioType = RadioType.ESP_NOW):
        """
        Initialize mesh network.

        Args:
            radio_type: Type of radio to use
        """
        self.radio_config = RADIO_PRESETS[radio_type]
        self.nodes: Dict[str, MeshNode] = {}

        # Pending messages (in-flight)
        self.pending_messages: List[Tuple[Message, str, float]] = []  # (msg, dest_id, arrival_time)

        # Network statistics
        self.total_messages = 0
        self.total_dropped = 0
        self.total_hops = 0

    def add_node(self, node_id: str, position: np.ndarray) -> MeshNode:
        """Add a node to the mesh network."""
        node = MeshNode(node_id, self.radio_config, position)
        self.nodes[node_id] = node
        return node

    def remove_node(self, node_id: str):
        """Remove a node from the mesh network."""
        if node_id in self.nodes:
            del self.nodes[node_id]

    def update_node_position(self, node_id: str, position: np.ndarray):
        """Update a node's position."""
        if node_id in self.nodes:
            self.nodes[node_id].position = position

    def calculate_connectivity(self):
        """Update neighbor lists based on current positions."""
        for node_id, node in self.nodes.items():
            node.neighbors.clear()

            for other_id, other in self.nodes.items():
                if node_id == other_id:
                    continue

                distance = np.linalg.norm(node.position - other.position)

                if distance <= self.radio_config.max_range:
                    node.neighbors.add(other_id)

    def calculate_latency(self, distance: float) -> float:
        """
        Calculate message latency based on distance.

        Args:
            distance: Distance in meters

        Returns:
            Latency in seconds
        """
        # Base latency + distance-proportional delay
        propagation_delay = distance / 300000000  # Speed of light (negligible)
        processing_delay = self.radio_config.base_latency
        queue_delay = np.random.uniform(0, 0.005)  # 0-5ms random queue delay

        return propagation_delay + processing_delay + queue_delay

    def calculate_packet_loss(self, distance: float) -> bool:
        """
        Determine if packet is lost based on distance.

        Args:
            distance: Distance in meters

        Returns:
            True if packet is lost
        """
        if distance > self.radio_config.max_range:
            return True  # Out of range

        # Linear packet loss model
        loss_rate = (distance / self.radio_config.max_range) * self.radio_config.packet_loss_rate

        return np.random.random() < loss_rate

    def transmit_messages(self, current_time: float):
        """
        Process outgoing messages from all nodes.

        Args:
            current_time: Current simulation time
        """
        for node_id, node in self.nodes.items():
            while node.outbox:
                msg = node.outbox.pop(0)

                # Broadcast to all neighbors
                for neighbor_id in node.neighbors:
                    distance = np.linalg.norm(
                        node.position - self.nodes[neighbor_id].position
                    )

                    # Check packet loss
                    if self.calculate_packet_loss(distance):
                        node.messages_dropped += 1
                        self.total_dropped += 1
                        continue

                    # Calculate arrival time
                    latency = self.calculate_latency(distance)
                    arrival_time = current_time + latency

                    # Queue for delivery
                    self.pending_messages.append((msg, neighbor_id, arrival_time))
                    self.total_messages += 1

    def deliver_messages(self, current_time: float):
        """
        Deliver messages that have arrived.

        Args:
            current_time: Current simulation time
        """
        # Deliver messages whose time has come
        remaining = []

        for msg, dest_id, arrival_time in self.pending_messages:
            if arrival_time <= current_time:
                # Deliver message
                if dest_id in self.nodes:
                    dest_node = self.nodes[dest_id]

                    if dest_node.receive_message(msg):
                        # Forward if needed (flooding)
                        if dest_node.should_forward(msg):
                            dest_node.forward_message(msg)
                            self.total_hops += 1
            else:
                remaining.append((msg, dest_id, arrival_time))

        self.pending_messages = remaining

    def update(self, current_time: float):
        """
        Update mesh network.

        Args:
            current_time: Current simulation time
        """
        # Update connectivity based on positions
        self.calculate_connectivity()

        # Transmit queued messages
        self.transmit_messages(current_time)

        # Deliver messages
        self.deliver_messages(current_time)

    def get_network_stats(self) -> dict:
        """Get network statistics."""
        return {
            'total_nodes': len(self.nodes),
            'total_messages': self.total_messages,
            'total_dropped': self.total_dropped,
            'total_hops': self.total_hops,
            'drop_rate': self.total_dropped / max(self.total_messages, 1)
        }

    def get_connectivity_graph(self) -> List[Tuple[str, str]]:
        """Get list of all active connections (for visualization)."""
        edges = []
        for node_id, node in self.nodes.items():
            for neighbor_id in node.neighbors:
                # Only add each edge once (avoid duplicates)
                if node_id < neighbor_id:
                    edges.append((node_id, neighbor_id))
        return edges
