#!/usr/bin/env python3
"""
Mesh Communications Example

Demonstrates:
- ESP-NOW mesh network (300m range)
- Position broadcasts from all drones
- Threat alert propagation through mesh
- Multi-hop routing (messages relay through swarm)
- Network visualization (cyan lines = connections)
- Packet loss and latency simulation
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
from src.visualization import ArenaVisualizer


class MeshCommsSimulation:
    """Simulation demonstrating mesh communications."""

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

        # Mesh network (ESP-NOW: 300m range)
        self.mesh = MeshNetwork(radio_type=RadioType.ESP_NOW)

        # Simulation parameters
        self.dt = 0.1  # 10 Hz
        self.position_broadcast_interval = 1.0  # Broadcast position every 1s
        self.last_position_broadcast = {}

        # Threat detection
        self.threat_detected = False
        self.threat_detection_time = None

        # Setup scenario
        self._setup_scenario()

    def _setup_scenario(self):
        """Create scenario: 6 patrol drones in mesh network."""

        # Create 6 patrol drones spread across arena
        patrol_positions = [
            [200, 200, 100],   # SW
            [500, 200, 100],   # S
            [800, 200, 100],   # SE
            [200, 500, 100],   # W
            [500, 500, 100],   # Center
            [800, 500, 100],   # E
        ]

        for i, pos in enumerate(patrol_positions):
            drone = Drone(
                position=pos,
                team=Team.FRIENDLY,
                role=Role.PATROL,
                max_speed=10.0,
                detection_range=200.0
            )
            drone.id = f"PATROL-{i+1}"
            drone.mesh_node = self.mesh.add_node(drone.id, np.array(pos))
            self.last_position_broadcast[drone.id] = 0.0
            self.arena.add_entity(drone)

        # Create intruder (will be detected)
        intruder = Drone(
            position=[100, 800, 80],  # Far from most drones
            velocity=[5, -5, 0],
            team=Team.ENEMY,
            role=Role.INTRUDER,
            max_speed=15.0
        )
        intruder.id = "INTRUDER-1"
        self.arena.add_entity(intruder)

    def broadcast_positions(self):
        """Drones broadcast their positions to mesh."""
        for entity in self.arena.get_active_entities():
            if isinstance(entity, Drone) and entity.team == Team.FRIENDLY:
                # Check if it's time to broadcast
                if (self.arena.time - self.last_position_broadcast[entity.id]) >= self.position_broadcast_interval:
                    # Send position message
                    entity.mesh_node.send_message(
                        msg_type=MessageType.POSITION,
                        data={
                            'position': entity.position.tolist(),
                            'velocity': entity.velocity.tolist(),
                            'battery': 100.0  # Placeholder
                        },
                        timestamp=self.arena.time
                    )
                    self.last_position_broadcast[entity.id] = self.arena.time

    def check_threat_detection(self):
        """Check if any drone detects the intruder."""
        for entity in self.arena.get_active_entities():
            if entity.team == Team.FRIENDLY and isinstance(entity, Drone):
                detections = self.detector.detect(entity, self.arena.get_active_entities())
                foes = self.detector.filter_foes(detections)

                if foes and not self.threat_detected:
                    # First detection - broadcast threat alert!
                    self.threat_detected = True
                    self.threat_detection_time = self.arena.time

                    threat = foes[0]
                    entity.mesh_node.send_message(
                        msg_type=MessageType.THREAT_ALERT,
                        data={
                            'threat_id': threat.entity.id,
                            'threat_position': threat.entity.position.tolist(),
                            'threat_distance': threat.distance,
                            'detector_id': entity.id
                        },
                        timestamp=self.arena.time
                    )

                    print(f"\n{'='*70}")
                    print(f"[{self.arena.time:6.1f}s] ⚠️  THREAT DETECTED!")
                    print(f"{'='*70}")
                    print(f"Detector: {entity.id}")
                    print(f"Threat: {threat.entity.id} at {threat.distance:.1f}m")
                    print(f"Broadcasting alert to mesh network...")
                    print()

    def process_mesh_messages(self):
        """Process received mesh messages."""
        for node_id, node in self.mesh.nodes.items():
            while node.inbox:
                msg = node.inbox.pop(0)

                if msg.msg_type == MessageType.THREAT_ALERT:
                    # Received threat alert
                    hops = len(msg.path) - 1
                    latency = self.arena.time - msg.timestamp

                    print(
                        f"[{self.arena.time:6.1f}s] {node_id} received THREAT ALERT "
                        f"from {msg.sender_id} via {hops} hop(s), latency: {latency*1000:.0f}ms"
                    )

                elif msg.msg_type == MessageType.POSITION:
                    # Received position update (too verbose to print all)
                    pass

    def update(self):
        """Update simulation step."""

        # Update drone positions (simple waypoint patrol)
        for entity in self.arena.get_active_entities():
            if entity.role == Role.PATROL and isinstance(entity, Drone):
                # Patrol: slow circular motion
                center = self.arena.get_bounds_center()
                angle = 0.05 * self.arena.time
                radius = 200

                target_x = center[0] + radius * np.cos(angle)
                target_y = center[1] + radius * np.sin(angle)

                # Move toward target (simple)
                direction = np.array([target_x - entity.position[0],
                                     target_y - entity.position[1], 0])
                dist = np.linalg.norm(direction)
                if dist > 0:
                    entity.velocity = (direction / dist) * 5.0

        # Update physics
        self.physics.update(self.arena.get_active_entities(), self.dt)

        # Update mesh node positions
        for entity in self.arena.get_active_entities():
            if isinstance(entity, Drone) and entity.team == Team.FRIENDLY:
                self.mesh.update_node_position(entity.id, entity.position)

        # Broadcast positions periodically
        self.broadcast_positions()

        # Check for threat detection
        self.check_threat_detection()

        # Update mesh network (transmit, route, deliver messages)
        self.mesh.update(self.arena.time)

        # Process received messages
        self.process_mesh_messages()

        # Update arena
        self.arena.update(self.dt)

    def run_visual(self):
        """Run simulation with visualization."""
        viz = ArenaVisualizer(
            self.arena,
            show_detection_range=True,
            show_trails=True,
            trail_length=100,
            mesh_network=self.mesh  # Show mesh connections!
        )

        print("=" * 70)
        print("Mesh Communications Simulation")
        print("=" * 70)
        print()
        print("Configuration:")
        print("  - Radio: ESP-NOW (300m range, 10ms latency)")
        print("  - Network: 6 patrol drones in mesh")
        print("  - Messages: Position broadcasts (1 Hz)")
        print("  - Threat alerts: Broadcast on detection")
        print()
        print("Visualization:")
        print("  - BRIGHT GREEN lines = mesh connections (300m max)")
        print("  - Blue dots = friendly patrol drones")
        print("  - Red dot = intruder")
        print()
        print("Watch for:")
        print("  - Mesh topology changes as drones move")
        print("  - Threat alert propagating through network")
        print("  - Multi-hop routing to distant drones")
        print()

        def update_all():
            self.update()

        # Animate
        anim = viz.animate(
            update_func=update_all,
            interval=100,  # 100ms between frames
            frames=600     # 60 seconds
        )

        viz.show()

        # Print statistics
        stats = self.mesh.get_network_stats()
        print()
        print("=" * 70)
        print("Network Statistics")
        print("=" * 70)
        print(f"Total nodes: {stats['total_nodes']}")
        print(f"Total messages: {stats['total_messages']}")
        print(f"Messages dropped: {stats['total_dropped']}")
        print(f"Total hops: {stats['total_hops']}")
        print(f"Packet loss rate: {stats['drop_rate']*100:.1f}%")
        print(f"Avg hops per message: {stats['total_hops']/max(stats['total_messages'], 1):.1f}")


def main():
    """Main entry point."""
    sim = MeshCommsSimulation()
    sim.run_visual()


if __name__ == "__main__":
    main()
