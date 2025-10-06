#!/usr/bin/env python3
"""
Autonomous Operation During Network Partition

Demonstrates:
- Continued patrol during network partition
- Independent threat detection and tracking
- Autonomous decision-making when disconnected
- Threat information merge on rejoin
"""

import sys
import os
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.arena import Arena, GeographicReference
from src.entities import Drone, Team, Role
from src.physics import PhysicsEngine
from src.detection import Detector
from src.mesh import MeshNetwork, RadioType, MessageType
from src.network import NetworkState, HeartbeatInfo
from src.strategy import AutonomousStrategy, create_perimeter_sectors, ThreatLevel
from src.visualization import ArenaVisualizer


class AutonomousOperationSimulation:
    """Simulation demonstrating autonomous operation during partition."""

    def __init__(self):
        # Create arena
        self.arena = Arena(
            bounds=(1000, 1000, 500),
            geo_reference=GeographicReference(
                latitude=37.7749,
                longitude=-122.4194
            )
        )

        # Physics and detection
        self.physics = PhysicsEngine(self.arena)
        self.detector = Detector(max_range=200)

        # Mesh network (ESP-NOW)
        self.mesh = MeshNetwork(radio_type=RadioType.ESP_NOW)

        # Network states and strategies for each drone
        self.network_states = {}
        self.strategies = {}

        # Simulation parameters
        self.dt = 0.1  # 100ms timestep

        # Partition control
        self.partition_active = False
        self.partition_start_time = 15.0  # Partition at t=15s
        self.partition_end_time = 40.0    # Restore at t=40s

        # Threat spawn
        self.threat_spawn_time = 20.0  # Spawn threat during partition
        self.threat_spawned = False

        # Setup scenario
        self._setup_scenario()

    def _setup_scenario(self):
        """Create scenario with patrol drones and sectors."""

        # Create perimeter patrol sectors (8 sectors around center)
        center = self.arena.get_bounds_center()
        patrol_radius = 300  # 300m from center
        sectors = create_perimeter_sectors(center, patrol_radius, num_sectors=8)

        # Create 4 patrol drones
        drone_positions = [
            [300, 300, 100],  # SW
            [700, 300, 100],  # SE
            [700, 700, 100],  # NE
            [300, 700, 100],  # NW
        ]

        drone_ids = []

        for i, pos in enumerate(drone_positions):
            drone = Drone(
                position=pos,
                team=Team.FRIENDLY,
                role=Role.PATROL,
                max_speed=15.0,
                detection_range=200.0
            )
            drone.id = f"PATROL-{i+1}"
            drone_ids.append(drone.id)
            self.arena.add_entity(drone)

            # Add to mesh network
            self.mesh.add_node(drone.id, np.array(pos))

        # Create network state and strategy for each drone
        for i, drone_id in enumerate(drone_ids):
            # Each drone expects to see all other drones
            expected_peers = set(drone_ids) - {drone_id}
            net_state = NetworkState(drone_id, expected_peers)
            self.network_states[drone_id] = net_state

            # Get drone entity
            drone = self.arena.get_entity_by_id(drone_id)

            # Create strategy with patrol sectors
            strategy = AutonomousStrategy(drone, net_state, sectors)
            self.strategies[drone_id] = strategy

            # Assign each drone to 2 sectors initially
            # Drone 1: sectors 0,1
            # Drone 2: sectors 2,3
            # Drone 3: sectors 4,5
            # Drone 4: sectors 6,7
            sector_idx = i * 2
            strategy.assign_patrol_sector(sectors[sector_idx])

            print(f"[SETUP] {drone_id} assigned to {sectors[sector_idx].sector_id}")

    def apply_partition(self):
        """Partition network: Group A (1,2) vs Group B (3,4)."""
        self.partition_active = True
        print(f"\n{'='*70}")
        print(f"[{self.arena.time:.1f}s] NETWORK PARTITION ACTIVE")
        print(f"  Group A: PATROL-1, PATROL-2 (SW, SE)")
        print(f"  Group B: PATROL-3, PATROL-4 (NE, NW)")
        print(f"  Drones will operate autonomously")
        print(f"{'='*70}\n")

    def remove_partition(self):
        """Restore network connectivity."""
        self.partition_active = False
        print(f"\n{'='*70}")
        print(f"[{self.arena.time:.1f}s] NETWORK RESTORED")
        print(f"  All drones reconnecting")
        print(f"  Synchronizing threat information")
        print(f"{'='*70}\n")

    def spawn_threat(self):
        """Spawn an intruder during partition."""
        self.threat_spawned = True

        # Spawn intruder near PATROL-3's sector (NE)
        intruder = Drone(
            position=[650, 650, 80],
            velocity=[5, 5, 0],  # Moving
            team=Team.ENEMY,
            role=Role.INTRUDER,
            max_speed=10.0
        )
        intruder.id = "INTRUDER-1"
        self.arena.add_entity(intruder)

        print(f"\n{'='*70}")
        print(f"[{self.arena.time:.1f}s] THREAT DETECTED")
        print(f"  Intruder spawned at NE sector")
        print(f"  Only PATROL-3, PATROL-4 can detect (partition active)")
        print(f"  PATROL-1, PATROL-2 will learn on rejoin")
        print(f"{'='*70}\n")

    def is_message_blocked(self, sender_id: str, receiver_id: str) -> bool:
        """Check if message blocked by partition."""
        if not self.partition_active:
            return False

        group_a = {"PATROL-1", "PATROL-2"}
        group_b = {"PATROL-3", "PATROL-4"}

        sender_in_a = sender_id in group_a
        sender_in_b = sender_id in group_b
        receiver_in_a = receiver_id in group_a
        receiver_in_b = receiver_id in group_b

        return (sender_in_a and receiver_in_b) or (sender_in_b and receiver_in_a)

    def update(self):
        """Update simulation step."""

        # Partition events (only trigger once)
        if (not self.partition_active and
            self.arena.time >= self.partition_start_time and
            self.arena.time < self.partition_start_time + self.dt):
            self.apply_partition()

        if (self.partition_active and
            self.arena.time >= self.partition_end_time and
            self.arena.time < self.partition_end_time + self.dt):
            self.remove_partition()

        # Threat spawn (only once)
        if (not self.threat_spawned and
            self.arena.time >= self.threat_spawn_time and
            self.arena.time < self.threat_spawn_time + self.dt):
            self.spawn_threat()

        # Update strategies
        for drone_id, strategy in self.strategies.items():
            strategy.update(self.arena.time)

        # Update patrol drone movements (move toward patrol targets)
        for entity in self.arena.get_active_entities():
            if entity.role == Role.PATROL and isinstance(entity, Drone):
                strategy = self.strategies.get(entity.id)
                if strategy:
                    # Get patrol target from strategy
                    target = strategy.get_patrol_target(self.arena.time)

                    if target is not None:
                        # Apply autonomous patrol expansion if partitioned
                        if strategy.autonomous_mode:
                            expansion = strategy.get_autonomous_patrol_expansion(self.arena.time)
                            # Expand patrol radius
                            if strategy.assigned_sector:
                                direction = target - strategy.assigned_sector.center
                                target = strategy.assigned_sector.center + direction * expansion

                        # Move toward target
                        direction = target - entity.position
                        distance = np.linalg.norm(direction)

                        if distance > 1.0:
                            direction = direction / distance
                            entity.velocity = direction * entity.max_speed * 0.5  # Half speed for patrol
                        else:
                            entity.velocity = np.array([0.0, 0.0, 0.0])

                # Update mesh position
                self.mesh.update_node_position(entity.id, entity.position)

        # Update physics for all entities
        self.physics.update(self.arena.get_active_entities(), self.dt)

        # Detect threats
        for entity in self.arena.get_active_entities():
            if entity.role == Role.PATROL and isinstance(entity, Drone):
                strategy = self.strategies.get(entity.id)
                if strategy:
                    # Detect enemies
                    detections = self.detector.detect(
                        entity,
                        self.arena.get_active_entities()
                    )

                    # Filter to foes only
                    foes = self.detector.filter_foes(detections)

                    for detection in foes:
                        detected_entity = detection.entity
                        # Add to strategy's threat list
                        strategy.add_threat(
                            threat_id=detected_entity.id,
                            position=detected_entity.position,
                            timestamp=self.arena.time,
                            detector_id=entity.id,
                            confidence=1.0,  # Full confidence for direct detection
                            threat_level=ThreatLevel.HIGH
                        )

                        # Check if should alert
                        if strategy.should_alert_threat(detected_entity.id):
                            print(f"[{entity.id}] THREAT ALERT: {detected_entity.id} detected at {detected_entity.position[:2]}")
                            strategy.mark_threat_alerted(detected_entity.id)

                            # Send threat alert via mesh (if connected)
                            mesh_node = self.mesh.nodes.get(entity.id)
                            if mesh_node:
                                mesh_node.send_message(
                                    msg_type=MessageType.THREAT_ALERT,
                                    data={
                                        'threat_id': detected_entity.id,
                                        'position': detected_entity.position.tolist(),
                                        'detector_id': entity.id,
                                        'threat_level': ThreatLevel.HIGH.value
                                    },
                                    timestamp=self.arena.time
                                )

        # Process heartbeats
        for entity in self.arena.get_active_entities():
            if isinstance(entity, Drone) and entity.id in self.network_states:
                net_state = self.network_states[entity.id]
                mesh_node = self.mesh.nodes.get(entity.id)

                if mesh_node is None:
                    continue

                # Send heartbeat?
                if net_state.should_send_heartbeat(self.arena.time):
                    battery_pct = entity.battery_level / entity.battery_capacity

                    heartbeat = net_state.create_heartbeat(
                        current_time=self.arena.time,
                        position=entity.position,
                        battery_level=battery_pct,
                        role_status=entity.role.value
                    )

                    mesh_node.send_message(
                        msg_type=MessageType.HEARTBEAT,
                        data={'heartbeat': heartbeat},
                        timestamp=self.arena.time
                    )

        # Update mesh with partition filtering
        self._update_mesh_with_partition()

        # Process received messages
        self._process_received_messages()

        # Update arena
        self.arena.update(self.dt)

    def _update_mesh_with_partition(self):
        """Update mesh network with partition filtering."""
        # Custom transmit with partition blocking
        for node_id, node in self.mesh.nodes.items():
            while node.outbox:
                msg = node.outbox.pop(0)

                for neighbor_id in node.neighbors:
                    # Check partition
                    if self.is_message_blocked(node_id, neighbor_id):
                        continue

                    distance = np.linalg.norm(
                        node.position - self.mesh.nodes[neighbor_id].position
                    )

                    if self.mesh.calculate_packet_loss(distance):
                        node.messages_dropped += 1
                        self.mesh.total_dropped += 1
                        continue

                    latency = self.mesh.calculate_latency(distance)
                    arrival_time = self.arena.time + latency

                    self.mesh.pending_messages.append((msg, neighbor_id, arrival_time))
                    self.mesh.total_messages += 1

        # Deliver messages
        self.mesh.deliver_messages(self.arena.time)

        # Update connectivity
        self.mesh.calculate_connectivity()

    def _process_received_messages(self):
        """Process received mesh messages."""
        for entity in self.arena.get_active_entities():
            if isinstance(entity, Drone) and entity.id in self.network_states:
                net_state = self.network_states[entity.id]
                strategy = self.strategies[entity.id]
                mesh_node = self.mesh.nodes.get(entity.id)

                if mesh_node is None:
                    continue

                for msg in mesh_node.inbox:
                    if msg.msg_type == MessageType.HEARTBEAT:
                        heartbeat_data = msg.data.get('heartbeat')
                        if heartbeat_data:
                            net_state.update_heartbeat(
                                peer_id=msg.sender_id,
                                heartbeat=heartbeat_data,
                                current_time=self.arena.time
                            )

                    elif msg.msg_type == MessageType.THREAT_ALERT:
                        # Received threat info from peer
                        threat_id = msg.data.get('threat_id')
                        position = np.array(msg.data.get('position'))
                        detector_id = msg.data.get('detector_id')
                        threat_level_val = msg.data.get('threat_level', ThreatLevel.MEDIUM.value)

                        if threat_id:
                            # Add to our threat list
                            strategy.add_threat(
                                threat_id=threat_id,
                                position=position,
                                timestamp=self.arena.time,
                                detector_id=detector_id,
                                threat_level=ThreatLevel(threat_level_val)
                            )

                            print(f"[{entity.id}] Received threat alert for {threat_id} from {msg.sender_id}")

                mesh_node.inbox.clear()

                # Detect partitions
                net_state.detect_partitions(self.arena.time)

    def run_visual(self):
        """Run simulation with visualization."""
        viz = ArenaVisualizer(
            self.arena,
            show_detection_range=True,
            show_trails=True,
            trail_length=200,
            mesh_network=self.mesh
        )

        print("=" * 70)
        print("Autonomous Operation During Network Partition")
        print("=" * 70)
        print()
        print("Scenario:")
        print("  - 4 patrol drones covering 8 perimeter sectors")
        print("  - Each drone covers 2 adjacent sectors")
        print()
        print("Timeline:")
        print(f"  t=0s    - Normal coordinated patrol")
        print(f"  t={self.partition_start_time:.0f}s   - PARTITION: Drones 1,2 isolated from 3,4")
        print(f"  t={self.threat_spawn_time:.0f}s   - THREAT: Intruder appears in NE sector")
        print(f"  t={self.partition_end_time:.0f}s   - RESTORE: Network reconnects, info syncs")
        print(f"  t=60s   - End simulation")
        print()
        print("Watch for:")
        print("  - Autonomous mode activation on partition")
        print("  - Patrol expansion when disconnected (cover gaps)")
        print("  - Independent threat detection in each partition")
        print("  - Threat information sharing on rejoin")
        print()

        def update_with_info():
            self.update()
            viz.mesh_network = self.mesh

        anim = viz.animate(
            update_func=update_with_info,
            interval=33,
            frames=600
        )

        viz.show()

        # Print final statistics
        print()
        print("=" * 70)
        print("Final Statistics")
        print("=" * 70)

        for drone_id, strategy in self.strategies.items():
            net_state = self.network_states[drone_id]
            print(f"\n{drone_id}:")
            print(f"  Status: {strategy.get_status_string()}")
            print(f"  Known threats: {len(strategy.known_threats)}")
            print(f"  Network stats: {net_state.get_statistics()}")


def main():
    """Main entry point."""
    sim = AutonomousOperationSimulation()
    sim.run_visual()


if __name__ == "__main__":
    main()
