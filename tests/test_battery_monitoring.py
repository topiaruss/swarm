#!/usr/bin/env python3
"""
Integration test for battery monitoring and return-to-charge logic.

Tests the decision logic for when drones should return to charge and leave the charger.
"""

import sys
import os
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.arena import Arena, GeographicReference
from src.entities import Drone, ChargingStation, Battery, Team, Role
from src.strategy import AutonomousStrategy, create_perimeter_sectors
from src.network import NetworkState


def test_return_energy_calculation():
    """Test calculation of energy required to return to base."""
    print("=" * 70)
    print("Testing Return Energy Calculation")
    print("=" * 70)

    # Create arena and station
    arena = Arena(bounds=(1000, 1000, 500))
    center = arena.get_bounds_center()

    station = ChargingStation(
        position=center,
        capacity=2
    )

    # Create drone at varying distances
    distances = [100, 200, 300]  # meters

    for distance in distances:
        drone = Drone(
            position=center + np.array([distance, 0, 0]),
            team=Team.FRIENDLY,
            max_speed=20.0
        )

        net_state = NetworkState(drone.id, set())
        strategy = AutonomousStrategy(drone, net_state, [])

        energy_required = strategy.calculate_return_energy(station)

        # Calculate expected
        cruise_speed = drone.max_speed * 0.5  # 10 m/s
        flight_time = distance / cruise_speed
        expected_energy = drone.battery.discharge_rate_base * flight_time

        print(f"\nDistance: {distance}m")
        print(f"  Flight time: {flight_time:.1f}s")
        print(f"  Energy required: {energy_required:.1f} mAh")
        print(f"  Expected: {expected_energy:.1f} mAh")

        assert abs(energy_required - expected_energy) < 0.1, "Energy calculation incorrect"

    print("\n✓ Return energy calculation working correctly\n")


def test_should_return_to_charge():
    """Test decision logic for returning to charge."""
    print("=" * 70)
    print("Testing Should Return to Charge Logic")
    print("=" * 70)

    arena = Arena(bounds=(1000, 1000, 500))
    center = arena.get_bounds_center()

    station = ChargingStation(
        position=center,
        capacity=2
    )

    # Test 1: Critical battery - always return
    print("\n1. Critical battery (10%)")
    drone = Drone(
        position=center + np.array([100, 0, 0]),
        team=Team.FRIENDLY,
        battery_level=500.0  # 10%
    )
    net_state = NetworkState(drone.id, set())
    strategy = AutonomousStrategy(drone, net_state, [])

    should_return = strategy.should_return_to_charge(station)
    print(f"  Battery: {drone.battery.level()*100:.0f}%")
    print(f"  Should return: {should_return}")
    assert should_return, "Should return at critical battery"

    # Test 2: Below optimal with slot available
    print("\n2. Below optimal (50%) with slot available")
    drone = Drone(
        position=center + np.array([100, 0, 0]),
        team=Team.FRIENDLY,
        battery_level=2500.0  # 50%
    )
    strategy = AutonomousStrategy(drone, NetworkState(drone.id, set()), [])

    should_return = strategy.should_return_to_charge(station)
    print(f"  Battery: {drone.battery.level()*100:.0f}%")
    print(f"  Slot available: {station.has_available_slot()}")
    print(f"  Should return: {should_return}")
    assert should_return, "Should return when below optimal with slot available"

    # Test 3: Above optimal - don't return
    print("\n3. Above optimal (90%)")
    drone = Drone(
        position=center + np.array([100, 0, 0]),
        team=Team.FRIENDLY,
        battery_level=4500.0  # 90%
    )
    strategy = AutonomousStrategy(drone, NetworkState(drone.id, set()), [])

    should_return = strategy.should_return_to_charge(station)
    print(f"  Battery: {drone.battery.level()*100:.0f}%")
    print(f"  Should return: {should_return}")
    assert not should_return, "Should not return when well charged"

    # Test 4: Insufficient energy to return
    print("\n4. Insufficient energy to return (far away, low battery)")
    drone = Drone(
        position=center + np.array([400, 0, 0]),  # Very far
        team=Team.FRIENDLY,
        battery_level=1000.0,  # 20%
        max_speed=20.0
    )
    strategy = AutonomousStrategy(drone, NetworkState(drone.id, set()), [])

    energy_required = strategy.calculate_return_energy(station)
    energy_margin = drone.battery.current_charge - energy_required
    margin_threshold = 0.1 * drone.battery.capacity

    print(f"  Battery: {drone.battery.level()*100:.0f}% ({drone.battery.current_charge:.0f} mAh)")
    print(f"  Distance: 400m")
    print(f"  Energy required: {energy_required:.0f} mAh")
    print(f"  Energy margin: {energy_margin:.0f} mAh")
    print(f"  Margin threshold: {margin_threshold:.0f} mAh")

    should_return = strategy.should_return_to_charge(station)
    print(f"  Should return: {should_return}")
    assert should_return, "Should return when energy margin is insufficient"

    print("\n✓ Should return to charge logic working correctly\n")


def test_should_leave_charger():
    """Test decision logic for leaving charger."""
    print("=" * 70)
    print("Testing Should Leave Charger Logic")
    print("=" * 70)

    arena = Arena(bounds=(1000, 1000, 500))
    center = arena.get_bounds_center()

    station = ChargingStation(
        position=center,
        capacity=2
    )

    # Test 1: Fully charged - leave
    print("\n1. Fully charged (95%)")
    drone = Drone(
        position=center,
        team=Team.FRIENDLY,
        battery_level=4750.0  # 95%
    )
    strategy = AutonomousStrategy(drone, NetworkState(drone.id, set()), [])

    should_leave = strategy.should_leave_charger(station)
    print(f"  Battery: {drone.battery.level()*100:.0f}%")
    print(f"  Should leave: {should_leave}")
    assert should_leave, "Should leave when fully charged"

    # Test 2: At optimal with queue - leave
    print("\n2. At optimal (80%) with queue")
    drone = Drone(
        position=center,
        team=Team.FRIENDLY,
        battery_level=4000.0  # 80%
    )
    strategy = AutonomousStrategy(drone, NetworkState(drone.id, set()), [])

    # Add drone to queue
    station.queue.append("WAITING-DRONE")

    should_leave = strategy.should_leave_charger(station)
    print(f"  Battery: {drone.battery.level()*100:.0f}%")
    print(f"  Queue length: {len(station.queue)}")
    print(f"  Should leave: {should_leave}")
    assert should_leave, "Should leave at optimal level with queue"

    # Test 3: Below optimal, no queue - stay
    print("\n3. Below optimal (60%), no queue")
    drone = Drone(
        position=center,
        team=Team.FRIENDLY,
        battery_level=3000.0  # 60%
    )
    strategy = AutonomousStrategy(drone, NetworkState(drone.id, set()), [])
    station.queue.clear()

    should_leave = strategy.should_leave_charger(station)
    print(f"  Battery: {drone.battery.level()*100:.0f}%")
    print(f"  Queue length: {len(station.queue)}")
    print(f"  Should leave: {should_leave}")
    assert not should_leave, "Should stay when below optimal with no queue"

    # Test 4: At optimal, critical drone waiting - leave
    print("\n4. At optimal (80%), critical drone waiting")
    drone = Drone(
        position=center,
        team=Team.FRIENDLY,
        battery_level=4000.0  # 80%
    )
    strategy = AutonomousStrategy(drone, NetworkState(drone.id, set()), [])

    critical_drone = Drone(
        position=center + np.array([50, 0, 0]),
        team=Team.FRIENDLY,
        battery_level=600.0  # 12% - critical
    )

    should_leave = strategy.should_leave_charger(station, [critical_drone])
    print(f"  Battery: {drone.battery.level()*100:.0f}%")
    print(f"  Critical drone waiting: {critical_drone.battery.level()*100:.0f}%")
    print(f"  Should leave: {should_leave}")
    assert should_leave, "Should leave to let critical drone charge"

    print("\n✓ Should leave charger logic working correctly\n")


def test_battery_discharge_during_patrol():
    """Test battery discharge during patrol."""
    print("=" * 70)
    print("Testing Battery Discharge During Patrol")
    print("=" * 70)

    arena = Arena(bounds=(1000, 1000, 500))
    center = arena.get_bounds_center()

    station = ChargingStation(position=center, capacity=2)

    # Create drone
    drone = Drone(
        position=center + np.array([200, 0, 100]),
        team=Team.FRIENDLY,
        battery_level=5000.0  # 100%
    )

    print(f"\nInitial battery: {drone.battery.level()*100:.0f}%")

    # Simulate 30 seconds of patrol
    dt = 0.1
    patrol_time = 30.0
    steps = int(patrol_time / dt)

    for step in range(steps):
        drone.battery.discharge(dt, rate_multiplier=1.0)  # Patrol rate

    print(f"After {patrol_time}s patrol: {drone.battery.level()*100:.0f}%")

    # Expected: 30s * 50 mAh/s = 1500 mAh consumed
    # Remaining: 5000 - 1500 = 3500 mAh = 70%
    expected_level = 0.70

    assert abs(drone.battery.level() - expected_level) < 0.01, "Battery discharge incorrect"

    print(f"Expected: {expected_level*100:.0f}%")
    print(f"Time to empty: {drone.battery.time_to_empty():.1f}s")

    print("\n✓ Battery discharge working correctly\n")


def test_complete_charge_cycle():
    """Test complete charge cycle: patrol -> low battery -> return -> charge -> resume."""
    print("=" * 70)
    print("Testing Complete Charge Cycle")
    print("=" * 70)

    arena = Arena(bounds=(1000, 1000, 500))
    center = arena.get_bounds_center()

    station = ChargingStation(
        position=center,
        capacity=2,
        charge_rate=0.02
    )
    arena.add_entity(station)

    # Create drone on patrol
    drone = Drone(
        position=center + np.array([300, 0, 100]),  # 300m away
        team=Team.FRIENDLY,
        battery_level=2000.0  # 40% - below optimal
    )
    arena.add_entity(drone)

    net_state = NetworkState(drone.id, set())
    sectors = create_perimeter_sectors(center, 300, num_sectors=8)
    strategy = AutonomousStrategy(drone, net_state, sectors)

    print(f"\nInitial state:")
    print(f"  Position: {drone.position}")
    print(f"  Battery: {drone.battery.level()*100:.0f}%")
    print(f"  Distance to station: {np.linalg.norm(drone.position - station.position):.1f}m")

    # Check if should return
    should_return = strategy.should_return_to_charge(station)
    print(f"  Should return: {should_return}")
    assert should_return, "Should return at 40% battery"

    # Request slot
    slot = station.request_slot(drone.id)
    print(f"\nRequested slot: {slot}")
    assert slot is not None, "Should get a slot"

    # Move to station (simulate flight)
    print(f"\nFlying to station...")
    target = station.get_slot_position(slot)
    distance = np.linalg.norm(target - drone.position)
    cruise_speed = drone.max_speed * 0.5
    flight_time = distance / cruise_speed

    # Discharge during flight
    drone.battery.discharge(flight_time, rate_multiplier=1.0)
    drone.position = target

    print(f"  Arrived after {flight_time:.1f}s")
    print(f"  Battery after flight: {drone.battery.level()*100:.0f}%")

    # Charge until full
    print(f"\nCharging...")
    dt = 0.1
    charge_start_level = drone.battery.level()

    while not drone.battery.is_full():
        station.charge_drone(drone, dt)

    charge_time = drone.battery.time_to_full(station.charge_rate)
    print(f"  Charged from {charge_start_level*100:.0f}% to {drone.battery.level()*100:.0f}%")
    print(f"  Charge time: ~{charge_time:.1f}s")

    # Check if should leave
    should_leave = strategy.should_leave_charger(station)
    print(f"  Should leave: {should_leave}")
    assert should_leave, "Should leave when fully charged"

    # Release slot
    station.release_slot(slot)
    print(f"\nReleased slot {slot}")
    print(f"Final battery: {drone.battery.level()*100:.0f}%")

    assert drone.battery.is_full(), "Should be fully charged"

    print("\n✓ Complete charge cycle working correctly\n")

    print("=" * 70)
    print("ALL TESTS PASSED: Battery monitoring and return-to-charge working")
    print("=" * 70)


if __name__ == "__main__":
    try:
        test_return_energy_calculation()
        test_should_return_to_charge()
        test_should_leave_charger()
        test_battery_discharge_during_patrol()
        test_complete_charge_cycle()
        sys.exit(0)
    except AssertionError as e:
        print(f"\n❌ TEST FAILED: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ TEST ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
