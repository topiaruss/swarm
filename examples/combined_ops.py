#!/usr/bin/env python3
"""
Combined Operations Demo

Demonstrates the complete defensive swarm system:
- Network partition detection and autonomous operation
- Battery discharge and charging management
- Patrol handoff coordination
- Threat detection and state synchronization
- Coverage maintenance throughout

Timeline:
- t=0-20s: Normal patrol, batteries draining
- t=20s: Network partition (Group A vs Group B)
- t=25s: Threat spawns in Group B's area
- t=30s: First drone needs charging (with handoff)
- t=50s: Network restores, state sync
- t=60-120s: Continued operations with charging cycles
"""

import sys
import os
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.arena import Arena, GeographicReference
from src.entities import Drone, Team, Role, ChargingStation
from src.physics import PhysicsEngine
from src.detection import Detector
from src.strategy import AutonomousStrategy, create_perimeter_sectors, ThreatLevel
from src.network import NetworkState
from src.mesh import MeshNetwork, RadioType, MessageType
from src.visualization import ArenaVisualizer


class CombinedOpsSimulation:
    """Combined operations simulation."""

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

        # Network states and strategies
        self.network_states = {}
        self.strategies = {}

        # Charging management
        self.charging_drones = {}  # drone_id -> slot
        self.returning_drones = set()  # drone_ids returning to charge

        # Simulation parameters
        self.dt = 0.033  # 33ms timestep (real-time at 30 fps)

        # Event timing
        self.partition_start_time = 20.0
        self.partition_end_time = 50.0
        self.threat_spawn_time = 25.0

        self.partition_active = False
        self.threat_spawned = False

        # Setup scenario
        self._setup_scenario()

    def _setup_scenario(self):
        """Create scenario with patrol drones, charging station, and sectors."""

        center = self.arena.get_bounds_center()

        # Create charging station at center
        self.station = ChargingStation(
            position=center,
            capacity=2,  # 2 charging slots
            charge_rate=0.02  # 2%/s = 50s for full charge
        )
        self.station.id = "CHARGING-STATION"
        self.arena.add_entity(self.station)

        # Create perimeter patrol sectors (8 sectors)
        patrol_radius = 300  # 300m from center
        self.sectors = create_perimeter_sectors(center, patrol_radius, num_sectors=8)

        # Create 4 patrol drones starting close together
        drone_positions = [
            center + np.array([-50, -50, 100]),  # PATROL-1 (Group A)
            center + np.array([50, -50, 100]),   # PATROL-2 (Group A)
            center + np.array([50, 50, 100]),    # PATROL-3 (Group B)
            center + np.array([-50, 50, 100]),   # PATROL-4 (Group B)
        ]

        # Start with moderate battery levels to trigger charging
        initial_batteries = [
            0.50,  # 50% - will need charging during demo
            0.60,  # 60%
            0.55,  # 55%
            0.65,  # 65%
        ]

        drone_ids = []

        for i, (pos, battery_pct) in enumerate(zip(drone_positions, initial_batteries)):
            drone = Drone(
                position=pos,
                team=Team.FRIENDLY,
                role=Role.PATROL,
                max_speed=15.0,
                detection_range=200.0,
                battery_level=5000.0 * battery_pct
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
            strategy = AutonomousStrategy(drone, net_state, self.sectors)
            self.strategies[drone_id] = strategy

            # Assign each drone to 2 sectors initially
            sector_idx = i * 2
            strategy.assign_patrol_sector(self.sectors[sector_idx])

            print(f"[SETUP] {drone_id} assigned to {self.sectors[sector_idx].sector_id}, battery: {drone.battery.level()*100:.0f}%")

    def apply_partition(self):
        """Partition network: Group A (1,2) vs Group B (3,4)."""
        self.partition_active = True
        print(f"\n{'='*70}")
        print(f"[{self.arena.time:.1f}s] NETWORK PARTITION ACTIVE")
        print(f"  Group A: PATROL-1, PATROL-2 (west)")
        print(f"  Group B: PATROL-3, PATROL-4 (east)")
        print(f"{'='*70}\n")

    def remove_partition(self):
        """Restore network connectivity."""
        self.partition_active = False
        print(f"\n{'='*70}")
        print(f"[{self.arena.time:.1f}s] NETWORK RESTORED")
        print(f"  All drones reconnecting")
        print(f"  Synchronizing state")
        print(f"{'='*70}\n")

    def spawn_threat(self):
        """Spawn an intruder during partition."""
        self.threat_spawned = True

        # Spawn intruder near PATROL-3's sector (Group B, east side)
        center = self.arena.get_bounds_center()
        intruder = Drone(
            position=center + np.array([350, 0, 80]),
            velocity=[5, 0, 0],  # Moving
            team=Team.ENEMY,
            role=Role.INTRUDER,
            max_speed=10.0
        )
        intruder.id = "INTRUDER-1"
        self.arena.add_entity(intruder)

        print(f"\n{'='*70}")
        print(f"[{self.arena.time:.1f}s] THREAT DETECTED")
        print(f"  Intruder spawned near Group B")
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

        # Event triggers
        if (not self.partition_active and
            self.arena.time >= self.partition_start_time and
            self.arena.time < self.partition_start_time + self.dt):
            self.apply_partition()

        if (self.partition_active and
            self.arena.time >= self.partition_end_time and
            self.arena.time < self.partition_end_time + self.dt):
            self.remove_partition()

        if (not self.threat_spawned and
            self.arena.time >= self.threat_spawn_time and
            self.arena.time < self.threat_spawn_time + self.dt):
            self.spawn_threat()

        # Update strategies
        for drone_id, strategy in self.strategies.items():
            strategy.update(self.arena.time)

        # Get all patrol drones
        patrol_drones = [
            e for e in self.arena.get_active_entities()
            if e.role == Role.PATROL and isinstance(e, Drone)
        ]

        # Handle charging decisions
        for drone in patrol_drones:
            strategy = self.strategies.get(drone.id)
            if not strategy:
                continue

            # Currently charging?
            if drone.id in self.charging_drones:
                slot = self.charging_drones[drone.id]
                self.station.charge_drone(drone, self.dt)

                # Check if should leave
                if strategy.should_leave_charger(self.station, patrol_drones):
                    self.station.release_slot(slot)
                    del self.charging_drones[drone.id]
                    drone.charging_slot = None

                    print(f"[{self.arena.time:.1f}s] {drone.id} finished charging (battery: {drone.battery.level()*100:.0f}%)")

                    # Resume patrol
                    direction = drone.position - self.station.position
                    if np.linalg.norm(direction) > 0:
                        direction = direction / np.linalg.norm(direction)
                        drone.velocity = direction * drone.max_speed * 0.5

                    self.returning_drones.discard(drone.id)

            # Returning to charge?
            elif drone.id in self.returning_drones:
                distance_to_station = np.linalg.norm(drone.position - self.station.position)

                if distance_to_station < 10.0:
                    slot = self.station.request_slot(drone.id)

                    if slot is not None:
                        self.charging_drones[drone.id] = slot
                        drone.charging_slot = slot
                        drone.position = self.station.get_slot_position(slot)
                        drone.velocity = np.array([0.0, 0.0, 0.0])

                        print(f"[{self.arena.time:.1f}s] {drone.id} started charging on slot {slot} (battery: {drone.battery.level()*100:.0f}%)")
                    else:
                        drone.velocity = np.array([0.0, 0.0, 0.0])
                else:
                    direction = self.station.position - drone.position
                    direction = direction / np.linalg.norm(direction)
                    drone.velocity = direction * drone.max_speed

            # Normal patrol
            else:
                # Check if should return to charge
                if strategy.should_return_to_charge(self.station):
                    self.returning_drones.add(drone.id)
                    print(f"[{self.arena.time:.1f}s] {drone.id} returning to charge (battery: {drone.battery.level()*100:.0f}%)")

                    # Announce charge intent (request handoff)
                    request = strategy.announce_charge_intent(self.arena.time, self.station)
                    mesh_node = self.mesh.nodes.get(drone.id)
                    if mesh_node:
                        mesh_node.send_message(
                            msg_type=MessageType.CHARGE_REQUEST,
                            data=request,
                            timestamp=self.arena.time
                        )
                    continue

                # Get patrol target
                target = strategy.get_patrol_target(self.arena.time)

                if target is not None:
                    # Move toward target
                    direction = target - drone.position
                    distance = np.linalg.norm(direction)

                    if distance > 1.0:
                        direction = direction / distance
                        drone.velocity = direction * drone.max_speed * 0.5
                    else:
                        drone.velocity = np.array([0.0, 0.0, 0.0])

                # Discharge battery during patrol
                drone.battery.discharge(self.dt, rate_multiplier=1.0)

        # Update physics
        self.physics.update(self.arena.get_active_entities(), self.dt)

        # Update mesh positions for all drones (after physics update)
        for drone in patrol_drones:
            self.mesh.update_node_position(drone.id, drone.position)

        # Detect threats
        for entity in self.arena.get_active_entities():
            if entity.role == Role.PATROL and isinstance(entity, Drone):
                strategy = self.strategies.get(entity.id)
                if strategy:
                    detections = self.detector.detect(entity, self.arena.get_active_entities())
                    foes = self.detector.filter_foes(detections)

                    for detection in foes:
                        detected_entity = detection.entity
                        strategy.add_threat(
                            threat_id=detected_entity.id,
                            position=detected_entity.position,
                            timestamp=self.arena.time,
                            detector_id=entity.id,
                            confidence=1.0,
                            threat_level=ThreatLevel.HIGH
                        )

                        if strategy.should_alert_threat(detected_entity.id):
                            print(f"[{entity.id}] THREAT ALERT: {detected_entity.id} detected")
                            strategy.mark_threat_alerted(detected_entity.id)

        # Process heartbeats
        for entity in self.arena.get_active_entities():
            if isinstance(entity, Drone) and entity.id in self.network_states:
                net_state = self.network_states[entity.id]
                mesh_node = self.mesh.nodes.get(entity.id)

                if mesh_node is None:
                    continue

                # Send heartbeat?
                if net_state.should_send_heartbeat(self.arena.time):
                    battery_pct = entity.battery.level()

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

                    elif msg.msg_type == MessageType.CHARGE_REQUEST:
                        # Received charge request - evaluate if can cover
                        if strategy.can_cover_handoff(msg.data):
                            response = strategy.accept_handoff(msg.data, self.arena.time)
                            mesh_node.send_message(
                                msg_type=MessageType.HANDOFF_ACCEPT,
                                data=response,
                                timestamp=self.arena.time
                            )
                            print(f"[{entity.id}] Accepted handoff for {msg.data['patrol_sector']}")

                    elif msg.msg_type == MessageType.HANDOFF_ACCEPT:
                        # Handoff accepted
                        strategy.handoff_volunteer = msg.data['drone_id']
                        strategy.handoff_pending = False

                mesh_node.inbox.clear()

                # Detect partitions
                net_state.detect_partitions(self.arena.time)

    def run_visual(self):
        """Run simulation with visualization."""
        viz = ArenaVisualizer(
            self.arena,
            show_detection_range=True,
            show_trails=True,
            trail_length=100,
            mesh_network=self.mesh
        )

        print("=" * 70)
        print("Combined Operations Demo")
        print("=" * 70)
        print()
        print("Timeline:")
        print(f"  t=0-20s   : Normal patrol, batteries draining")
        print(f"  t={self.partition_start_time:.0f}s     : PARTITION activated")
        print(f"  t={self.threat_spawn_time:.0f}s     : THREAT spawns (Group B area)")
        print(f"  t=30s     : First drones need charging (with handoff)")
        print(f"  t={self.partition_end_time:.0f}s     : Network RESTORED (state sync)")
        print(f"  t=60-120s : Continued ops with charging cycles")
        print()
        print("Watch for:")
        print("  - Network partition (mesh connections break)")
        print("  - Autonomous mode activation")
        print("  - Threat detection (Group B only during partition)")
        print("  - Charging returns with patrol handoff")
        print("  - State sync on reconnection")
        print("  - Coverage maintained throughout")
        print()

        def update_with_status():
            self.update()

            # Update mesh visualization
            viz.mesh_network = self.mesh

            # Print status every 10 seconds
            if int(self.arena.time) % 10 == 0 and self.arena.time > 0 and abs(self.arena.time - int(self.arena.time)) < self.dt:
                print(f"\n[{self.arena.time:.0f}s] Status:")
                for drone in self.arena.get_active_entities():
                    if drone.role == Role.PATROL and isinstance(drone, Drone):
                        status = "CHARGING" if drone.id in self.charging_drones else \
                                "RETURNING" if drone.id in self.returning_drones else \
                                "PATROL"
                        battery_pct = drone.battery.level() * 100
                        battery_bar = "█" * int(battery_pct / 5) + "░" * (20 - int(battery_pct / 5))

                        strategy = self.strategies[drone.id]
                        sectors = strategy.get_all_patrol_sectors()
                        sector_str = f"{sectors[0].sector_id}" + (f"+{len(sectors)-1}" if len(sectors) > 1 else "")

                        print(f"  {drone.id}: {battery_bar} {battery_pct:5.1f}% | {status:10s} | {sector_str}")

        anim = viz.animate(
            update_func=update_with_status,
            interval=33,
            frames=1200  # 2 minutes
        )

        viz.show()


def main():
    """Main entry point."""
    sim = CombinedOpsSimulation()
    sim.run_visual()


if __name__ == "__main__":
    main()
