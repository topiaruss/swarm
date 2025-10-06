#!/usr/bin/env python3
"""
Integration test for patrol handoff coordination.

Tests handoff protocol when drones leave for charging.
"""

import sys
import os
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.arena import Arena, GeographicReference
from src.entities import Drone, ChargingStation, Team, Role
from src.strategy import AutonomousStrategy, create_perimeter_sectors
from src.network import NetworkState
from src.mesh import MeshNetwork, RadioType, MessageType


def test_charge_request_announcement():
    """Test charge request announcement."""
    print("=" * 70)
    print("Testing Charge Request Announcement")
    print("=" * 70)

    arena = Arena(bounds=(1000, 1000, 500))
    center = arena.get_bounds_center()

    station = ChargingStation(position=center, capacity=2)

    sectors = create_perimeter_sectors(center, 300, num_sectors=8)

    # Create drone needing charge
    drone = Drone(
        position=center + np.array([300, 0, 100]),
        team=Team.FRIENDLY,
        battery_level=2000.0  # 40%
    )
    drone.id = "PATROL-1"

    net_state = NetworkState(drone.id, set())
    strategy = AutonomousStrategy(drone, net_state, sectors)
    strategy.assign_patrol_sector(sectors[0])

    current_time = 10.0

    # Announce charge intent
    request = strategy.announce_charge_intent(current_time, station)

    print(f"\nCharge Request:")
    print(f"  Drone: {request['drone_id']}")
    print(f"  Battery: {request['battery_level']*100:.0f}%")
    print(f"  Patrol sector: {request['patrol_sector']}")
    print(f"  Handoff needed: {request['handoff_needed']}")
    print(f"  Est. departure: {request['estimated_departure_time']:.1f}s")

    assert request['drone_id'] == drone.id
    assert request['patrol_sector'] == sectors[0].sector_id
    assert request['handoff_needed'] == True

    print("\n✓ Charge request announcement working correctly\n")


def test_can_cover_handoff():
    """Test evaluation of handoff coverage capability."""
    print("=" * 70)
    print("Testing Can Cover Handoff Logic")
    print("=" * 70)

    arena = Arena(bounds=(1000, 1000, 500))
    center = arena.get_bounds_center()

    sectors = create_perimeter_sectors(center, 300, num_sectors=8)

    # Test 1: Drone with good battery, adjacent sector - can cover
    print("\n1. Good battery, adjacent sector")
    drone = Drone(
        position=center + np.array([300, 0, 100]),
        team=Team.FRIENDLY,
        battery_level=4500.0  # 90%
    )
    strategy = AutonomousStrategy(drone, NetworkState(drone.id, set()), sectors)
    strategy.assign_patrol_sector(sectors[0])

    # Request from adjacent sector (sector 1)
    request = {
        'drone_id': 'PATROL-2',
        'patrol_sector': sectors[1].sector_id,
        'sector_center': sectors[1].center.tolist(),
        'handoff_needed': True
    }

    can_cover = strategy.can_cover_handoff(request)
    print(f"  Battery: {drone.battery.level()*100:.0f}%")
    print(f"  Assigned sector: {strategy.assigned_sector.sector_id}")
    print(f"  Request sector: {request['patrol_sector']}")
    print(f"  Can cover: {can_cover}")
    assert can_cover, "Should be able to cover adjacent sector"

    # Test 2: Low battery - cannot cover
    print("\n2. Low battery")
    drone2 = Drone(
        position=center + np.array([300, 0, 100]),
        team=Team.FRIENDLY,
        battery_level=2000.0  # 40% - below optimal
    )
    strategy2 = AutonomousStrategy(drone2, NetworkState(drone2.id, set()), sectors)
    strategy2.assign_patrol_sector(sectors[0])

    can_cover2 = strategy2.can_cover_handoff(request)
    print(f"  Battery: {drone2.battery.level()*100:.0f}%")
    print(f"  Can cover: {can_cover2}")
    assert not can_cover2, "Should not cover with low battery"

    # Test 3: Already covering too many sectors
    print("\n3. Already covering 2 sectors")
    drone3 = Drone(
        position=center + np.array([300, 0, 100]),
        team=Team.FRIENDLY,
        battery_level=4500.0
    )
    strategy3 = AutonomousStrategy(drone3, NetworkState(drone3.id, set()), sectors)
    strategy3.assign_patrol_sector(sectors[0])
    strategy3.covering_sectors.append(sectors[2])
    strategy3.covering_sectors.append(sectors[3])

    can_cover3 = strategy3.can_cover_handoff(request)
    print(f"  Assigned sector: {strategy3.assigned_sector.sector_id}")
    print(f"  Covering sectors: {len(strategy3.covering_sectors)}")
    print(f"  Can cover: {can_cover3}")
    assert not can_cover3, "Should not cover when already covering 2+ sectors"

    print("\n✓ Can cover handoff logic working correctly\n")


def test_handoff_accept():
    """Test handoff acceptance."""
    print("=" * 70)
    print("Testing Handoff Accept")
    print("=" * 70)

    arena = Arena(bounds=(1000, 1000, 500))
    center = arena.get_bounds_center()

    sectors = create_perimeter_sectors(center, 300, num_sectors=8)

    # Create drone that will accept handoff
    drone = Drone(
        position=center + np.array([300, 0, 100]),
        team=Team.FRIENDLY,
        battery_level=4500.0  # 90%
    )
    drone.id = "PATROL-2"

    net_state = NetworkState(drone.id, set())
    strategy = AutonomousStrategy(drone, net_state, sectors)
    strategy.assign_patrol_sector(sectors[1])

    # Request from PATROL-1 for sector 0
    request = {
        'drone_id': 'PATROL-1',
        'battery_level': 0.40,
        'patrol_sector': sectors[0].sector_id,
        'sector_center': sectors[0].center.tolist(),
        'handoff_needed': True
    }

    current_time = 10.0

    # Accept handoff
    response = strategy.accept_handoff(request, current_time)

    print(f"\nHandoff acceptance:")
    print(f"  Accepting drone: {response['drone_id']}")
    print(f"  Battery: {response['battery_level']*100:.0f}%")
    print(f"  Sector accepted: {response['sector_accepted']}")
    print(f"  Assigned sector: {strategy.assigned_sector.sector_id}")
    print(f"  Covering sectors: {[s.sector_id for s in strategy.covering_sectors]}")

    assert response['drone_id'] == drone.id
    assert response['sector_accepted'] == sectors[0].sector_id
    assert len(strategy.covering_sectors) == 1
    assert strategy.covering_sectors[0].sector_id == sectors[0].sector_id

    # Get all patrol sectors
    all_sectors = strategy.get_all_patrol_sectors()
    print(f"  All patrol sectors: {[s.sector_id for s in all_sectors]}")
    assert len(all_sectors) == 2  # Assigned + covering

    print("\n✓ Handoff accept working correctly\n")


def test_handoff_release():
    """Test releasing handoff coverage when drone returns."""
    print("=" * 70)
    print("Testing Handoff Release")
    print("=" * 70)

    arena = Arena(bounds=(1000, 1000, 500))
    center = arena.get_bounds_center()

    sectors = create_perimeter_sectors(center, 300, num_sectors=8)

    # Create drone covering extra sectors
    drone = Drone(
        position=center + np.array([300, 0, 100]),
        team=Team.FRIENDLY,
        battery_level=4500.0
    )
    drone.id = "PATROL-2"

    strategy = AutonomousStrategy(drone, NetworkState(drone.id, set()), sectors)
    strategy.assign_patrol_sector(sectors[1])
    strategy.covering_sectors.append(sectors[0])
    strategy.covering_sectors.append(sectors[2])

    print(f"\nBefore release:")
    print(f"  Assigned: {strategy.assigned_sector.sector_id}")
    print(f"  Covering: {[s.sector_id for s in strategy.covering_sectors]}")

    # Release sector 0
    strategy.release_handoff_coverage(sectors[0].sector_id)

    print(f"\nAfter releasing {sectors[0].sector_id}:")
    print(f"  Assigned: {strategy.assigned_sector.sector_id}")
    print(f"  Covering: {[s.sector_id for s in strategy.covering_sectors]}")

    assert len(strategy.covering_sectors) == 1
    assert strategy.covering_sectors[0].sector_id == sectors[2].sector_id

    print("\n✓ Handoff release working correctly\n")


def test_complete_handoff_scenario():
    """Test complete handoff scenario with messaging."""
    print("=" * 70)
    print("Testing Complete Handoff Scenario")
    print("=" * 70)

    arena = Arena(bounds=(1000, 1000, 500))
    center = arena.get_bounds_center()

    station = ChargingStation(position=center, capacity=2)
    arena.add_entity(station)

    mesh = MeshNetwork(radio_type=RadioType.ESP_NOW)

    sectors = create_perimeter_sectors(center, 300, num_sectors=8)

    # Create 2 drones with adjacent sectors
    drones = []
    strategies = {}
    network_states = {}

    for i in range(2):
        # Position drones close together (within ESP-NOW range)
        angle = i * (2 * np.pi / 8)  # Adjacent sectors
        drone = Drone(
            position=center + np.array([200 * np.cos(angle), 200 * np.sin(angle), 100]),
            team=Team.FRIENDLY,
            battery_level=4500.0 if i == 1 else 2000.0  # Drone 0 needs charge
        )
        drone.id = f"PATROL-{i+1}"
        drones.append(drone)
        arena.add_entity(drone)
        mesh.add_node(drone.id, drone.position)

        net_state = NetworkState(drone.id, {drones[j].id for j in range(len(drones)) if j != i})
        network_states[drone.id] = net_state

        strategy = AutonomousStrategy(drone, net_state, sectors)
        strategy.assign_patrol_sector(sectors[i])
        strategies[drone.id] = strategy

    print(f"\nInitial setup:")
    for drone in drones:
        strategy = strategies[drone.id]
        print(f"  {drone.id}: battery {drone.battery.level()*100:.0f}%, sector {strategy.assigned_sector.sector_id}")

    current_time = 10.0

    # PATROL-1 announces charge intent
    print(f"\n[{current_time:.1f}s] PATROL-1 announces charge intent")
    request = strategies[drones[0].id].announce_charge_intent(current_time, station)

    # Broadcast via mesh
    mesh_node = mesh.nodes[drones[0].id]
    mesh_node.send_message(
        msg_type=MessageType.CHARGE_REQUEST,
        data=request,
        timestamp=current_time
    )

    # Update mesh
    mesh.update(current_time)

    # Wait for message delivery (advance time slightly)
    current_time += 0.1
    mesh.update(current_time)

    # Debug: Check connectivity
    distance = np.linalg.norm(drones[0].position - drones[1].position)
    print(f"  Distance between drones: {distance:.1f}m")
    print(f"  PATROL-1 neighbors: {mesh.nodes[drones[0].id].neighbors}")
    print(f"  PATROL-2 neighbors: {mesh.nodes[drones[1].id].neighbors}")

    # PATROL-2 receives request
    mesh_node_2 = mesh.nodes[drones[1].id]
    print(f"  PATROL-2 inbox: {len(mesh_node_2.inbox)} messages")
    for msg in mesh_node_2.inbox:
        if msg.msg_type == MessageType.CHARGE_REQUEST:
            print(f"\n[{current_time:.1f}s] PATROL-2 received charge request")

            # Evaluate if can cover
            can_cover = strategies[drones[1].id].can_cover_handoff(msg.data)
            print(f"  Can cover: {can_cover}")

            if can_cover:
                # Accept handoff
                response = strategies[drones[1].id].accept_handoff(msg.data, current_time)
                print(f"  Accepted handoff for sector {response['sector_accepted']}")

                # Send response
                mesh_node_2.send_message(
                    msg_type=MessageType.HANDOFF_ACCEPT,
                    data=response,
                    timestamp=current_time
                )

                assert can_cover, "PATROL-2 should be able to cover"
                assert len(strategies[drones[1].id].covering_sectors) == 1, "Should be covering 1 sector"

    mesh_node_2.inbox.clear()

    # Update mesh again
    mesh.update(current_time)

    # PATROL-1 receives handoff accept
    for msg in mesh_node.inbox:
        if msg.msg_type == MessageType.HANDOFF_ACCEPT:
            print(f"\n[{current_time:.1f}s] PATROL-1 received handoff accept from {msg.data['drone_id']}")
            strategies[drones[0].id].handoff_volunteer = msg.data['drone_id']
            strategies[drones[0].id].handoff_pending = False

    print(f"\nFinal state:")
    for drone in drones:
        strategy = strategies[drone.id]
        all_sectors = strategy.get_all_patrol_sectors()
        print(f"  {drone.id}: covering {len(all_sectors)} sectors - {[s.sector_id for s in all_sectors]}")

    # Verify coverage maintained
    assert len(strategies[drones[1].id].get_all_patrol_sectors()) == 2, "PATROL-2 should cover 2 sectors"

    print("\n✓ Complete handoff scenario working correctly\n")

    print("=" * 70)
    print("ALL TESTS PASSED: Patrol handoff coordination working")
    print("=" * 70)


if __name__ == "__main__":
    try:
        test_charge_request_announcement()
        test_can_cover_handoff()
        test_handoff_accept()
        test_handoff_release()
        test_complete_handoff_scenario()
        sys.exit(0)
    except AssertionError as e:
        print(f"\n❌ TEST FAILED: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ TEST ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
