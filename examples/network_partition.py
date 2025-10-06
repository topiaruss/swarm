#!/usr/bin/env python3
"""
Network Partition Example

Demonstrates:
- Heartbeat protocol for connectivity monitoring
- Network partition detection (5 second timeout)
- Autonomous operation during partition
- Reconnection and state sync
"""

import sys
import os
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.arena import Arena, GeographicReference
from src.entities import Drone, Team, Role
from src.physics import PhysicsEngine
from src.mesh import MeshNetwork, RadioType, MessageType
from src.network import NetworkState, HeartbeatInfo
from src.visualization import ArenaVisualizer


class NetworkPartitionSimulation:
    """Simulation demonstrating network partition detection and recovery."""

    def __init__(self):
        # Create arena
        self.arena = Arena(
            bounds=(1000, 1000, 500),
            geo_reference=GeographicReference(
                latitude=37.7749,
                longitude=-122.4194
            )
        )

        # Physics
        self.physics = PhysicsEngine(self.arena)

        # Mesh network (ESP-NOW)
        self.mesh = MeshNetwork(radio_type=RadioType.ESP_NOW)

        # Network states for each drone
        self.network_states = {}

        # Simulation parameters
        self.dt = 0.1  # 100ms timestep

        # Partition control
        self.partition_active = False
        self.partition_start_time = 10.0  # Partition at t=10s
        self.partition_end_time = 30.0    # Restore at t=30s

        # Setup scenario
        self._setup_scenario()

    def _setup_scenario(self):
        """Create scenario with 4 patrol drones."""

        # Create 4 patrol drones in different positions
        drone_positions = [
            [300, 300, 100],  # SW quadrant
            [700, 300, 100],  # SE quadrant
            [700, 700, 100],  # NE quadrant
            [300, 700, 100],  # NW quadrant
        ]

        drone_ids = []

        for i, pos in enumerate(drone_positions):
            drone = Drone(
                position=pos,
                team=Team.FRIENDLY,
                role=Role.PATROL,
                max_speed=15.0
            )
            drone.id = f"PATROL-{i+1}"
            drone_ids.append(drone.id)
            self.arena.add_entity(drone)

            # Add to mesh network
            self.mesh.add_node(drone.id, np.array(pos))

            # Give each drone a circular patrol path
            center = self.arena.get_bounds_center()
            angle = (i / 4) * 2 * np.pi
            drone.phase_offset = angle
            drone.patrol_radius = 150

        # Create network state tracker for each drone
        for drone_id in drone_ids:
            # Each drone expects to see all other drones
            expected_peers = set(drone_ids) - {drone_id}
            self.network_states[drone_id] = NetworkState(drone_id, expected_peers)

    def apply_partition(self):
        """
        Simulate network partition by preventing communication
        between drones 1,2 and drones 3,4.
        """
        # Manually break mesh connections by setting positions far apart temporarily
        # (We'll do this by marking certain messages as dropped)
        self.partition_active = True
        print(f"\n{'='*70}")
        print(f"[{self.arena.time:.1f}s] SIMULATING NETWORK PARTITION")
        print(f"  Group A: PATROL-1, PATROL-2")
        print(f"  Group B: PATROL-3, PATROL-4")
        print(f"  (No communication between groups)")
        print(f"{'='*70}\n")

    def remove_partition(self):
        """Restore network connectivity."""
        self.partition_active = False
        print(f"\n{'='*70}")
        print(f"[{self.arena.time:.1f}s] RESTORING NETWORK CONNECTIVITY")
        print(f"  All drones should rejoin")
        print(f"{'='*70}\n")

    def is_message_blocked(self, sender_id: str, receiver_id: str) -> bool:
        """
        Check if message should be blocked due to partition.

        Partition configuration:
        - Group A: PATROL-1, PATROL-2
        - Group B: PATROL-3, PATROL-4
        - No communication between groups
        """
        if not self.partition_active:
            return False

        group_a = {"PATROL-1", "PATROL-2"}
        group_b = {"PATROL-3", "PATROL-4"}

        sender_in_a = sender_id in group_a
        sender_in_b = sender_id in group_b
        receiver_in_a = receiver_id in group_a
        receiver_in_b = receiver_id in group_b

        # Block if sender and receiver in different groups
        return (sender_in_a and receiver_in_b) or (sender_in_b and receiver_in_a)

    def update(self):
        """Update simulation step."""

        # Check for partition events
        if not self.partition_active and self.arena.time >= self.partition_start_time:
            self.apply_partition()

        if self.partition_active and self.arena.time >= self.partition_end_time:
            self.remove_partition()

        # Update patrol drone movements (circular paths)
        for entity in self.arena.get_active_entities():
            if entity.role == Role.PATROL and isinstance(entity, Drone):
                center = self.arena.get_bounds_center()
                angular_velocity = 0.1
                angle = angular_velocity * self.arena.time + entity.phase_offset
                radius = entity.patrol_radius

                x = center[0] + radius * np.cos(angle)
                y = center[1] + radius * np.sin(angle)
                z = 100

                entity.position = np.array([x, y, z])

                # Update mesh position
                self.mesh.update_node_position(entity.id, entity.position)

        # Process heartbeats from each drone
        for entity in self.arena.get_active_entities():
            if isinstance(entity, Drone) and entity.id in self.network_states:
                net_state = self.network_states[entity.id]
                mesh_node = self.mesh.nodes.get(entity.id)

                if mesh_node is None:
                    continue

                # Should send heartbeat?
                if net_state.should_send_heartbeat(self.arena.time):
                    battery_pct = entity.battery_level / entity.battery_capacity

                    heartbeat = net_state.create_heartbeat(
                        current_time=self.arena.time,
                        position=entity.position,
                        battery_level=battery_pct,
                        role_status=entity.role.value
                    )

                    # Send as mesh message
                    mesh_node.send_message(
                        msg_type=MessageType.HEARTBEAT,
                        data={
                            'heartbeat': heartbeat
                        },
                        timestamp=self.arena.time
                    )

        # Update mesh network (with partition filtering)
        original_transmit = self.mesh.transmit_messages

        def filtered_transmit(current_time):
            """Transmit with partition filtering."""
            for node_id, node in self.mesh.nodes.items():
                while node.outbox:
                    msg = node.outbox.pop(0)

                    # Broadcast to neighbors
                    for neighbor_id in node.neighbors:
                        # Check if partition blocks this
                        if self.is_message_blocked(node_id, neighbor_id):
                            # Silently drop
                            continue

                        distance = np.linalg.norm(
                            node.position - self.mesh.nodes[neighbor_id].position
                        )

                        # Check packet loss
                        if self.mesh.calculate_packet_loss(distance):
                            node.messages_dropped += 1
                            self.mesh.total_dropped += 1
                            continue

                        # Calculate arrival time
                        latency = self.mesh.calculate_latency(distance)
                        arrival_time = current_time + latency

                        # Queue for delivery
                        self.mesh.pending_messages.append((msg, neighbor_id, arrival_time))
                        self.mesh.total_messages += 1

        # Temporarily replace transmit method
        self.mesh.transmit_messages = filtered_transmit
        self.mesh.update(self.arena.time)
        self.mesh.transmit_messages = original_transmit

        # Process received heartbeats
        for entity in self.arena.get_active_entities():
            if isinstance(entity, Drone) and entity.id in self.network_states:
                net_state = self.network_states[entity.id]
                mesh_node = self.mesh.nodes.get(entity.id)

                if mesh_node is None:
                    continue

                # Process inbox
                for msg in mesh_node.inbox:
                    if msg.msg_type == MessageType.HEARTBEAT:
                        heartbeat_data = msg.data.get('heartbeat')
                        if heartbeat_data:
                            net_state.update_heartbeat(
                                peer_id=msg.sender_id,
                                heartbeat=heartbeat_data,
                                current_time=self.arena.time
                            )

                # Clear inbox
                mesh_node.inbox.clear()

                # Detect partitions
                net_state.detect_partitions(self.arena.time)

        # Update arena
        self.arena.update(self.dt)

    def run_visual(self):
        """Run simulation with visualization."""
        viz = ArenaVisualizer(
            self.arena,
            show_detection_range=False,
            show_trails=True,
            trail_length=200,
            mesh_network=self.mesh
        )

        print("=" * 70)
        print("Network Partition Simulation")
        print("=" * 70)
        print()
        print("Scenario:")
        print("  - 4 patrol drones (PATROL-1 through PATROL-4)")
        print("  - Heartbeat interval: 1 second")
        print("  - Timeout threshold: 5 seconds")
        print()
        print("Timeline:")
        print(f"  t=0s    - Normal operation, full connectivity")
        print(f"  t={self.partition_start_time:.0f}s   - PARTITION: Drones 1,2 isolated from 3,4")
        print(f"  t={self.partition_end_time:.0f}s   - RESTORE: Full connectivity resumed")
        print(f"  t=60s   - End simulation")
        print()
        print("Watch for:")
        print("  - Partition detection messages")
        print("  - Mesh connections disappear during partition")
        print("  - Reconnection messages when restored")
        print()

        def update_with_info():
            self.update()

            # Update mesh visualization
            viz.mesh_network = self.mesh

        # Animate (60 seconds at 10 Hz = 600 frames, but display at 30 fps)
        anim = viz.animate(
            update_func=update_with_info,
            interval=33,  # 30 fps display
            frames=600    # 60 seconds worth
        )

        viz.show()

        # Print final statistics
        print()
        print("=" * 70)
        print("Final Network Statistics")
        print("=" * 70)
        for drone_id, net_state in self.network_states.items():
            stats = net_state.get_statistics()
            print(f"\n{drone_id}:")
            print(f"  Mode: {stats['mode']}")
            print(f"  Connected peers: {stats['connected_peers']}/{stats['total_peers']}")
            print(f"  Partition count: {stats['partition_count']}")
            print(f"  Rejoin count: {stats['rejoin_count']}")
            print(f"  Total disconnects: {stats['total_disconnects']}")

        mesh_stats = self.mesh.get_network_stats()
        print(f"\nMesh Network:")
        print(f"  Total messages: {mesh_stats['total_messages']}")
        print(f"  Dropped: {mesh_stats['total_dropped']}")
        print(f"  Drop rate: {mesh_stats['drop_rate']:.1%}")


def main():
    """Main entry point."""
    sim = NetworkPartitionSimulation()
    sim.run_visual()


if __name__ == "__main__":
    main()
