#!/usr/bin/env python3
"""
Integration test for state synchronization on network rejoin.

Tests that threat information is shared when partitioned drones reconnect.
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


def test_state_synchronization():
    """Test state sync when drones reconnect after partition."""

    print("=" * 70)
    print("Testing State Synchronization on Rejoin")
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

    # Create 4 patrol drones starting close together
    drone_positions = [
        center + np.array([-50, -50, 0]),  # PATROL-1 (Group A)
        center + np.array([50, -50, 0]),   # PATROL-2 (Group A)
        center + np.array([50, 50, 0]),    # PATROL-3 (Group B)
        center + np.array([-50, 50, 0]),   # PATROL-4 (Group B)
    ]

    drone_ids = []
    network_states = {}
    strategies = {}
    group_a = {"PATROL-1", "PATROL-2"}
    group_b = {"PATROL-3", "PATROL-4"}

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

        sector_idx = i * 2
        strategy.assign_patrol_sector(sectors[sector_idx])

    dt = 0.1
    current_time = 0.0

    print("\nPhase 1: Establish connectivity (0-5s)")
    print("-" * 70)

    # Establish connectivity
    for step in range(50):
        current_time = step * dt

        # Send heartbeats and update
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

    print(f"At t={current_time:.1f}s: All drones connected")

    # Phase 2: Partition and spawn threat in Group B's area
    print("\nPhase 2: Partition and spawn threat (5-15s)")
    print("-" * 70)

    partition_start = current_time

    # Spawn intruder near Group B (PATROL-3, PATROL-4 will detect it)
    intruder = Drone(
        position=center + np.array([350, 0, 80]),  # East side
        team=Team.ENEMY,
        role=Role.INTRUDER,
        max_speed=10.0
    )
    intruder.id = "INTRUDER-1"
    arena.add_entity(intruder)
    print(f"  Spawned {intruder.id} near Group B")

    # Simulate partition with physical separation
    for step in range(100):  # 10 seconds
        current_time = partition_start + step * dt

        # Move groups apart
        for drone_id in drone_ids:
            drone = arena.get_entity_by_id(drone_id)

            if drone_id in group_a:
                target = center + np.array([-300, 0, 0])  # West
            else:
                target = center + np.array([300, 0, 0])   # East

            direction = target - drone.position
            distance = np.linalg.norm(direction)

            if distance > 5.0:
                direction = direction / distance
                drone.position = drone.position + direction * 50.0 * dt

            mesh.update_node_position(drone_id, drone.position)

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

    # Check partition and threat detection
    print(f"\nAt t={current_time:.1f}s (after partition):")
    for drone_id, strategy in strategies.items():
        threats = len(strategy.known_threats)
        print(f"  {drone_id}: {threats} threat(s) known")

    # Group B should know about the threat
    group_b_threats = sum(len(strategies[d].known_threats) for d in group_b)
    assert group_b_threats > 0, "Group B should have detected the intruder"

    # Group A should NOT know about the threat yet (partitioned)
    group_a_threats = sum(len(strategies[d].known_threats) for d in group_a)
    assert group_a_threats == 0, "Group A should not know about threat during partition"

    print("\n✓ Threat detected by Group B only (partition active)")

    # Phase 3: Restore connectivity and sync state
    print("\nPhase 3: Restore connectivity and sync state (15-25s)")
    print("-" * 70)

    restore_start = current_time

    # Move groups back together
    for step in range(100):  # 10 seconds
        current_time = restore_start + step * dt

        # Move back to center
        for drone_id in drone_ids:
            drone = arena.get_entity_by_id(drone_id)

            target = center
            direction = target - drone.position
            distance = np.linalg.norm(direction)

            if distance > 5.0:
                direction = direction / distance
                drone.position = drone.position + direction * 50.0 * dt

            mesh.update_node_position(drone_id, drone.position)

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

            # Send state sync requests to peers that need it
            peers_to_sync = strategy.get_peers_needing_sync()
            for peer_id in peers_to_sync:
                if net_state.is_connected_to(peer_id):
                    # Send state sync request
                    sync_data = strategy.request_state_sync(peer_id, current_time)
                    mesh_node.send_message(
                        msg_type=MessageType.STATE_SYNC_REQUEST,
                        data=sync_data,
                        timestamp=current_time
                    )
                    strategy.mark_peer_synced(peer_id)

        mesh.update(current_time)

        # Process messages
        for drone_id in drone_ids:
            net_state = network_states[drone_id]
            strategy = strategies[drone_id]
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

                elif msg.msg_type == MessageType.STATE_SYNC_REQUEST:
                    # Received sync request - respond with our threats
                    requester_id = msg.sender_id
                    response_data = strategy.request_state_sync(requester_id, current_time)
                    mesh_node.send_message(
                        msg_type=MessageType.STATE_SYNC_RESPONSE,
                        data=response_data,
                        timestamp=current_time
                    )

                    # Also merge threats from the request
                    for threat_data in msg.data.get('threats', []):
                        strategy.merge_threat_from_sync(threat_data, current_time)

                elif msg.msg_type == MessageType.STATE_SYNC_RESPONSE:
                    # Merge threats from response
                    for threat_data in msg.data.get('threats', []):
                        strategy.merge_threat_from_sync(threat_data, current_time)

            mesh_node.inbox.clear()
            net_state.detect_partitions(current_time)

    # Check that all drones now know about the threat
    print(f"\nAt t={current_time:.1f}s (after rejoin and sync):")
    for drone_id, strategy in strategies.items():
        threats = len(strategy.known_threats)
        print(f"  {drone_id}: {threats} threat(s) known")
        assert threats > 0, f"{drone_id} should know about threat after sync"

    print("\n✓ All drones now aware of threat via state sync")

    print("\n" + "=" * 70)
    print("TEST PASSED: State synchronization working correctly")
    print("=" * 70)

    return True


if __name__ == "__main__":
    try:
        test_state_synchronization()
        sys.exit(0)
    except AssertionError as e:
        print(f"\n❌ TEST FAILED: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ TEST ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
