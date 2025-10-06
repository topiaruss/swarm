#!/usr/bin/env python3
"""
Integration test for autonomous operation during partition.

Tests threat detection and autonomous operation without GUI.
"""

import sys
import os
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.arena import Arena, GeographicReference
from src.entities import Drone, Team, Role
from src.detection import Detector
from src.mesh import MeshNetwork, RadioType, MessageType
from src.network import NetworkState
from src.strategy import AutonomousStrategy, create_perimeter_sectors, ThreatLevel


def test_autonomous_operation():
    """Test autonomous operation with threat detection during partition."""

    print("=" * 70)
    print("Testing Autonomous Operation During Partition")
    print("=" * 70)

    # Create arena
    arena = Arena(
        bounds=(1000, 1000, 500),
        geo_reference=GeographicReference(latitude=37.7749, longitude=-122.4194)
    )

    # Detector and mesh
    detector = Detector(max_range=200)
    mesh = MeshNetwork(radio_type=RadioType.ESP_NOW)

    # Create patrol sectors
    center = arena.get_bounds_center()
    sectors = create_perimeter_sectors(center, 300, num_sectors=8)

    # Create 4 patrol drones (close enough for ESP-NOW)
    drone_positions = [
        [400, 400, 100],
        [600, 400, 100],
        [600, 600, 100],
        [400, 600, 100],
    ]

    drone_ids = []
    network_states = {}
    strategies = {}

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
        arena.add_entity(drone)
        mesh.add_node(drone.id, np.array(pos))

    # Create network states and strategies
    for i, drone_id in enumerate(drone_ids):
        expected_peers = set(drone_ids) - {drone_id}
        net_state = NetworkState(drone_id, expected_peers)
        network_states[drone_id] = net_state

        drone = arena.get_entity_by_id(drone_id)
        strategy = AutonomousStrategy(drone, net_state, sectors)
        strategies[drone_id] = strategy

        # Assign sectors (each drone gets 2)
        sector_idx = i * 2
        strategy.assign_patrol_sector(sectors[sector_idx])

    dt = 0.1
    current_time = 0.0

    print("\nPhase 1: Establish connectivity (0-5s)")
    print("-" * 70)

    # Establish connectivity
    for step in range(50):
        current_time = step * dt

        # Send heartbeats
        for drone_id in drone_ids:
            net_state = network_states[drone_id]
            strategy = strategies[drone_id]
            mesh_node = mesh.nodes[drone_id]

            strategy.update(current_time)

            if net_state.should_send_heartbeat(current_time):
                drone = arena.get_entity_by_id(drone_id)
                heartbeat = net_state.create_heartbeat(
                    current_time=current_time,
                    position=drone.position,
                    battery_level=1.0,
                    role_status=drone.role.value
                )

                mesh_node.send_message(
                    msg_type=MessageType.HEARTBEAT,
                    data={'heartbeat': heartbeat},
                    timestamp=current_time
                )

        mesh.update(current_time)

        # Process heartbeats
        for drone_id in drone_ids:
            net_state = network_states[drone_id]
            mesh_node = mesh.nodes[drone_id]

            for msg in mesh_node.inbox:
                if msg.msg_type == MessageType.HEARTBEAT:
                    heartbeat = msg.data.get('heartbeat')
                    if heartbeat:
                        net_state.update_heartbeat(
                            peer_id=msg.sender_id,
                            heartbeat=heartbeat,
                            current_time=current_time
                        )

            mesh_node.inbox.clear()
            net_state.detect_partitions(current_time)

    print(f"\nAt t={current_time:.1f}s:")
    for drone_id, strategy in strategies.items():
        stats = network_states[drone_id].get_statistics()
        print(f"  {drone_id}: mode={stats['mode']}, autonomous={strategy.autonomous_mode}")
        assert stats['mode'] == 'connected'
        assert not strategy.autonomous_mode

    print("\n✓ All drones connected, not in autonomous mode")

    # Phase 2: Partition and spawn threat
    print("\nPhase 2: Partition network and spawn threat (5-10s)")
    print("-" * 70)

    group_a = {"PATROL-1", "PATROL-2"}
    group_b = {"PATROL-3", "PATROL-4"}

    partition_start = current_time

    # Spawn intruder near PATROL-3
    intruder = Drone(
        position=[580, 580, 80],  # Near PATROL-3
        team=Team.ENEMY,
        role=Role.INTRUDER,
        max_speed=10.0
    )
    intruder.id = "INTRUDER-1"
    arena.add_entity(intruder)

    print(f"  Spawned {intruder.id} at {intruder.position[:2]}")

    # Simulate with partition
    for step in range(100):  # 10 seconds
        current_time = partition_start + step * dt

        # Send heartbeats
        for drone_id in drone_ids:
            net_state = network_states[drone_id]
            strategy = strategies[drone_id]
            mesh_node = mesh.nodes[drone_id]

            strategy.update(current_time)

            if net_state.should_send_heartbeat(current_time):
                drone = arena.get_entity_by_id(drone_id)
                heartbeat = net_state.create_heartbeat(
                    current_time=current_time,
                    position=drone.position,
                    battery_level=1.0,
                    role_status=drone.role.value
                )

                mesh_node.send_message(
                    msg_type=MessageType.HEARTBEAT,
                    data={'heartbeat': heartbeat},
                    timestamp=current_time
                )

        # Detect threats
        for drone_id in drone_ids:
            drone = arena.get_entity_by_id(drone_id)
            strategy = strategies[drone_id]

            detections = detector.detect(drone, arena.get_active_entities())
            foes = detector.filter_foes(detections)

            for detection in foes:
                detected_entity = detection.entity
                strategy.add_threat(
                    threat_id=detected_entity.id,
                    position=detected_entity.position,
                    timestamp=current_time,
                    detector_id=drone_id,
                    confidence=1.0,
                    threat_level=ThreatLevel.HIGH
                )

        # Mesh update with partition
        mesh.calculate_connectivity()

        for node_id, node in mesh.nodes.items():
            while node.outbox:
                msg = node.outbox.pop(0)

                for neighbor_id in node.neighbors:
                    # Block cross-group
                    sender_in_a = node_id in group_a
                    sender_in_b = node_id in group_b
                    receiver_in_a = neighbor_id in group_a
                    receiver_in_b = neighbor_id in group_b

                    if (sender_in_a and receiver_in_b) or (sender_in_b and receiver_in_a):
                        continue

                    distance = np.linalg.norm(
                        node.position - mesh.nodes[neighbor_id].position
                    )

                    if not mesh.calculate_packet_loss(distance):
                        latency = mesh.calculate_latency(distance)
                        arrival_time = current_time + latency
                        mesh.pending_messages.append((msg, neighbor_id, arrival_time))

        mesh.deliver_messages(current_time)

        # Process messages
        for drone_id in drone_ids:
            net_state = network_states[drone_id]
            mesh_node = mesh.nodes[drone_id]

            for msg in mesh_node.inbox:
                if msg.msg_type == MessageType.HEARTBEAT:
                    heartbeat = msg.data.get('heartbeat')
                    if heartbeat:
                        net_state.update_heartbeat(
                            peer_id=msg.sender_id,
                            heartbeat=heartbeat,
                            current_time=current_time
                        )

            mesh_node.inbox.clear()
            net_state.detect_partitions(current_time)

    # Check partition detected and autonomous mode activated
    print(f"\nAt t={current_time:.1f}s (after partition):")
    for drone_id, strategy in strategies.items():
        stats = network_states[drone_id].get_statistics()
        print(f"  {drone_id}: mode={stats['mode']}, autonomous={strategy.autonomous_mode}, threats={len(strategy.known_threats)}")

        assert stats['mode'] in ['partitioned', 'isolated']
        assert strategy.autonomous_mode

    # Check threats detected
    patrol3_threats = len(strategies["PATROL-3"].known_threats)
    patrol4_threats = len(strategies["PATROL-4"].known_threats)
    print(f"\n  PATROL-3 detected {patrol3_threats} threats")
    print(f"  PATROL-4 detected {patrol4_threats} threats")

    # At least one of partition B should detect the intruder (within 200m)
    assert (patrol3_threats > 0 or patrol4_threats > 0), "Partition B should detect intruder"

    print("\n✓ Partition detected, autonomous mode activated")
    print("✓ Threat detected by partition B")

    print("\n" + "=" * 70)
    print("TEST PASSED: Autonomous operation working correctly")
    print("=" * 70)

    return True


if __name__ == "__main__":
    try:
        test_autonomous_operation()
        sys.exit(0)
    except AssertionError as e:
        print(f"\n❌ TEST FAILED: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ TEST ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
