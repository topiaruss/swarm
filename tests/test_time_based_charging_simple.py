#!/usr/bin/env python3
"""
Simple test for time-based charging scheduling.

Demonstrates that drones with 70% battery stay on patrol when station is busy,
rather than hovering around the station.
"""

import sys
import os
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.entities import Drone, Team, Role, ChargingStation, Battery
from src.strategy import AutonomousStrategy
from src.network import NetworkState


def test_dont_hover_when_busy():
    """Test that drones don't hover around busy station with 70% battery."""

    print("=" * 70)
    print("Test: Drones Don't Hover Around Busy Charging Station")
    print("=" * 70)

    # Create charging station
    station = ChargingStation(
        position=np.array([500, 500, 0]),
        capacity=2,  # 2 slots
        charge_rate=0.02
    )

    # Create 4 drones
    drones = []
    strategies = []

    # DRONE-1 and DRONE-2: Currently charging (both slots occupied)
    for i in range(2):
        drone = Drone(
            position=np.array([500 + i*10, 500, 10]),  # At station
            team=Team.FRIENDLY,
            role=Role.PATROL,
            battery_level=5000.0 * 0.40,  # 40% battery, charging
            max_speed=15.0
        )
        drone.id = f"DRONE-{i+1}"
        drone.charging_slot = i  # Currently charging
        drones.append(drone)

        net_state = NetworkState(drone.id, set())
        strategy = AutonomousStrategy(drone, net_state, [])
        strategies.append(strategy)

    # DRONE-3: On patrol, 70% battery, 200m away
    drone3 = Drone(
        position=np.array([700, 500, 100]),  # 200m from station
        team=Team.FRIENDLY,
        role=Role.PATROL,
        battery_level=5000.0 * 0.70,  # 70% battery
        max_speed=15.0
    )
    drone3.id = "DRONE-3"
    drones.append(drone3)

    net_state3 = NetworkState(drone3.id, set())
    strategy3 = AutonomousStrategy(drone3, net_state3, [])
    strategies.append(strategy3)

    # DRONE-4: On patrol, 75% battery, 200m away
    drone4 = Drone(
        position=np.array([300, 500, 100]),  # 200m from station
        team=Team.FRIENDLY,
        role=Role.PATROL,
        battery_level=5000.0 * 0.75,  # 75% battery
        max_speed=15.0
    )
    drone4.id = "DRONE-4"
    drones.append(drone4)

    net_state4 = NetworkState(drone4.id, set())
    strategy4 = AutonomousStrategy(drone4, net_state4, [])
    strategies.append(strategy4)

    # Mark both slots as occupied
    station.occupied_slots[0] = "DRONE-1"
    station.occupied_slots[1] = "DRONE-2"

    current_time = 0.0

    print("\nScenario:")
    print("  DRONE-1: Charging (40% battery)")
    print("  DRONE-2: Charging (40% battery)")
    print("  DRONE-3: On patrol 200m away (70% battery)")
    print("  DRONE-4: On patrol 200m away (75% battery)")
    print("  Charging station: 2 slots, BOTH occupied")
    print()

    # Calculate when charging drones will finish
    drone1_remaining = drones[0].battery.full_level - drones[0].battery.level()
    time_to_finish_1 = drone1_remaining / station.charge_rate

    print(f"DRONE-1 will finish charging in ~{time_to_finish_1:.0f}s")
    print(f"DRONE-3 flight time to station: {200 / 7.5:.1f}s (200m at 7.5 m/s)")
    print()

    print("=" * 70)
    print("Test 1: At t=0s (station busy)")
    print("=" * 70)

    # At t=0, with station busy, DRONE-3 and DRONE-4 should NOT return
    # They should stay on patrol despite being below 80%
    should_return_3 = strategies[2].should_return_to_charge(station, drones, current_time)
    should_return_4 = strategies[3].should_return_to_charge(station, drones, current_time)

    print(f"DRONE-3 (70%, 200m away) should_return: {should_return_3}")
    print(f"DRONE-4 (75%, 200m away) should_return: {should_return_4}")

    if not should_return_3:
        print("✓ DRONE-3 stays on patrol (station busy, too early to return)")
    else:
        print("✗ DRONE-3 incorrectly wants to return (would hover at station)")

    if not should_return_4:
        print("✓ DRONE-4 stays on patrol (station busy, too early to return)")
    else:
        print("✗ DRONE-4 incorrectly wants to return (would hover at station)")

    # Main assertion: drones should NOT hover when station is busy
    assert not should_return_3, "DRONE-3 should stay on patrol, not hover at busy station"
    assert not should_return_4, "DRONE-4 should stay on patrol, not hover at busy station"

    print("\n" + "=" * 70)
    print("Test 2: Advance time closer to slot availability")
    print("=" * 70)

    # Advance time to ~20s before slot is ready
    # At this point, it's still too early for the drones to return
    current_time = time_to_finish_1 - 40.0

    if current_time > 0:
        should_return_3 = strategies[2].should_return_to_charge(station, drones, current_time)
        should_return_4 = strategies[3].should_return_to_charge(station, drones, current_time)

        print(f"At t={current_time:.1f}s:")
        print(f"  DRONE-3 should_return: {should_return_3}")
        print(f"  DRONE-4 should_return: {should_return_4}")

        assert not should_return_3, "Still too early to return"
        assert not should_return_4, "Still too early to return"

        print("✓ Drones still on patrol (still too early)")

    print("\n" + "=" * 70)
    print("Test 3: Slot becomes available")
    print("=" * 70)

    # Now free up a slot
    current_time = time_to_finish_1 + 1.0
    drones[0].charging_slot = None
    del station.occupied_slots[0]

    should_return_3 = strategies[2].should_return_to_charge(station, drones, current_time)
    should_return_4 = strategies[3].should_return_to_charge(station, drones, current_time)

    print(f"At t={current_time:.1f}s (slot just freed up):")
    print(f"  DRONE-3 (70%) should_return: {should_return_3}")
    print(f"  DRONE-4 (75%) should_return: {should_return_4}")

    # At least one should return now (slot is available and they're below optimal)
    assert should_return_3 or should_return_4, "At least one drone should return when slot is free"

    if should_return_3:
        print("✓ DRONE-3 returns now that slot is available")
    if should_return_4:
        print("✓ DRONE-4 returns now that slot is available")

    print("\n" + "=" * 70)
    print("All tests passed! ✓")
    print("=" * 70)
    print("\nKey behavior verified:")
    print("  ✓ Drones with 70-75% battery don't hover around busy station")
    print("  ✓ Drones stay on patrol when it's too early to return")
    print("  ✓ Drones return promptly when slots become available")


if __name__ == "__main__":
    test_dont_hover_when_busy()
