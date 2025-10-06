#!/usr/bin/env python3
"""
Integration test for network partition scenario.

Tests that the network partition detection works correctly without GUI.
"""

import sys
import os
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.arena import Arena, GeographicReference
from src.entities import Drone, Team, Role
from src.mesh import MeshNetwork, RadioType, MessageType
from src.network import NetworkState, NetworkMode


def test_network_partition():
    """Test network partition detection and recovery."""

    print("=" * 70)
    print("Testing Network Partition Detection")
    print("=" * 70)

    # Create arena
    arena = Arena(
        bounds=(1000, 1000, 500),
        geo_reference=GeographicReference(latitude=37.7749, longitude=-122.4194)
    )

    # Create mesh network
    mesh = MeshNetwork(radio_type=RadioType.ESP_NOW)

    # Create 4 drones starting close together
    center = arena.get_bounds_center()
    drone_positions = [
        center + np.array([-50, -50, 0]),  # PATROL-1 (Group A)
        center + np.array([50, -50, 0]),   # PATROL-2 (Group A)
        center + np.array([50, 50, 0]),    # PATROL-3 (Group B)
        center + np.array([-50, 50, 0]),   # PATROL-4 (Group B)
    ]

    drone_ids = []
    network_states = {}
    group_a = {"PATROL-1", "PATROL-2"}
    group_b = {"PATROL-3", "PATROL-4"}

    for i, pos in enumerate(drone_positions):
        drone = Drone(
            position=pos,
            team=Team.FRIENDLY,
            role=Role.PATROL,
            max_speed=15.0
        )
        drone.id = f"PATROL-{i+1}"
        drone_ids.append(drone.id)
        arena.add_entity(drone)
        mesh.add_node(drone.id, np.array(pos))

        # Create network state
        expected_peers = set(drone_ids[:-1]) if i > 0 else set()
        if i == len(drone_positions) - 1:
            expected_peers = set(drone_ids) - {drone.id}
        network_states[drone.id] = NetworkState(drone.id, expected_peers)

    # Update all network states to have correct expected peers after all drones created
    for drone_id in drone_ids:
        expected_peers = set(drone_ids) - {drone_id}
        network_states[drone_id] = NetworkState(drone_id, expected_peers)

    dt = 0.1
    current_time = 0.0

    print("\nPhase 1: Normal operation (0-5s)")
    print("-" * 70)

    # Simulate 5 seconds of normal operation
    for step in range(50):  # 5 seconds at 0.1s steps
        current_time = step * dt

        # Send heartbeats
        for drone_id in drone_ids:
            net_state = network_states[drone_id]
            mesh_node = mesh.nodes[drone_id]

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

        # Update mesh
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

    # Check all drones are connected
    print(f"\nAt t={current_time:.1f}s:")
    for drone_id, net_state in network_states.items():
        stats = net_state.get_statistics()
        print(f"  {drone_id}: {stats['connected_peers']}/{stats['total_peers']} connected, mode={stats['mode']}")
        assert stats['mode'] == 'connected', f"{drone_id} should be connected"
        # Note: total_peers includes self apparently, so connected should be total-1 or total
        assert stats['connected_peers'] >= 3, f"{drone_id} should see at least 3 peers"

    print("\n✓ All drones connected")

    # Phase 2: Simulate partition by physical separation
    print("\nPhase 2: Partition (physically separate groups)")
    print("-" * 70)

    partition_start = current_time

    # Move groups apart over 10 seconds
    for step in range(100):  # 10 seconds
        current_time = partition_start + step * dt

        # Move drones to separated positions (600m apart)
        for drone_id in drone_ids:
            drone = arena.get_entity_by_id(drone_id)

            # Target: Group A goes west, Group B goes east
            if drone_id in group_a:
                target = center + np.array([-300, 0, 0])
            else:
                target = center + np.array([300, 0, 0])

            # Move toward target
            direction = target - drone.position
            distance = np.linalg.norm(direction)

            if distance > 5.0:
                direction = direction / distance
                drone.position = drone.position + direction * 50.0 * dt  # 50 m/s

            # Update mesh position
            mesh.update_node_position(drone_id, drone.position)

        # Send heartbeats
        for drone_id in drone_ids:
            net_state = network_states[drone_id]
            mesh_node = mesh.nodes[drone_id]

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

        # Normal mesh update (physical separation breaks connections naturally)
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

    # Check partitions detected
    print(f"\nAt t={current_time:.1f}s (after partition):")
    for drone_id, net_state in network_states.items():
        stats = net_state.get_statistics()
        print(f"  {drone_id}: {stats['connected_peers']}/{stats['total_peers']} connected, mode={stats['mode']}")

        # Should see 1 or 2 peers (same group) instead of 3-4
        # Groups of 2 drones each, so each sees 1 peer + maybe self
        assert stats['connected_peers'] <= 2, f"{drone_id} should see <=2 peers in partition, saw {stats['connected_peers']}"
        assert stats['mode'] in ['partitioned', 'isolated'], f"{drone_id} should detect partition"
        assert stats['partition_count'] >= 1, f"{drone_id} should have detected partition"

    print("\n✓ Partition detected by all drones")

    # Phase 3: Restore connectivity by bringing groups back together
    print("\nPhase 3: Restore connectivity (groups return to center)")
    print("-" * 70)

    restore_start = current_time

    # Move groups back together over 10 seconds
    for step in range(100):
        current_time = restore_start + step * dt

        # Move drones back to center
        for drone_id in drone_ids:
            drone = arena.get_entity_by_id(drone_id)

            # Target: back to initial area near center
            target = center + np.array([0, 0, 0])

            # Move toward target
            direction = target - drone.position
            distance = np.linalg.norm(direction)

            if distance > 5.0:
                direction = direction / distance
                drone.position = drone.position + direction * 50.0 * dt  # 50 m/s

            # Update mesh position
            mesh.update_node_position(drone_id, drone.position)

        # Send heartbeats
        for drone_id in drone_ids:
            net_state = network_states[drone_id]
            mesh_node = mesh.nodes[drone_id]

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

        # Normal mesh update
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

    # Check all reconnected
    print(f"\nAt t={current_time:.1f}s (after restore):")
    for drone_id, net_state in network_states.items():
        stats = net_state.get_statistics()
        print(f"  {drone_id}: {stats['connected_peers']}/{stats['total_peers']} connected, mode={stats['mode']}")

        assert stats['mode'] == 'connected', f"{drone_id} should be reconnected"
        assert stats['connected_peers'] >= 3, f"{drone_id} should see at least 3 peers (saw {stats['connected_peers']})"
        assert stats['rejoin_count'] >= 2, f"{drone_id} should have rejoined peers"

    print("\n✓ All drones reconnected")

    print("\n" + "=" * 70)
    print("TEST PASSED: Network partition detection working correctly")
    print("=" * 70)

    return True


if __name__ == "__main__":
    try:
        test_network_partition()
        sys.exit(0)
    except AssertionError as e:
        print(f"\n❌ TEST FAILED: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ TEST ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
