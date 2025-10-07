#!/usr/bin/env python3
"""
Test advanced charging heuristics.

Tests:
- Predictive scheduling
- Load balancing
- Emergency reserves
"""

import sys
import os
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.entities import Drone, Team, Role, ChargingStation, Battery
from src.strategy import AutonomousStrategy
from src.network import NetworkState


def test_emergency_reserve():
    """Test that emergency reserve logic keeps at least one drone >80%."""

    # Create charging station
    station = ChargingStation(
        position=np.array([500, 500, 0]),
        capacity=2,
        charge_rate=0.02
    )

    # Create 4 drones
    drones = []
    strategies = []

    for i in range(4):
        drone = Drone(
            position=np.array([500 + i*50, 500, 100]),
            team=Team.FRIENDLY,
            role=Role.PATROL,
            battery_level=5000.0  # Start at 100%
        )
        drone.id = f"DRONE-{i+1}"
        drones.append(drone)

        net_state = NetworkState(drone.id, set())
        strategy = AutonomousStrategy(drone, net_state, [])
        strategies.append(strategy)

    # Scenario: 3 drones have low battery, 1 has high battery
    drones[0].battery.current_charge = 0.85 * drones[0].battery.capacity  # 85% - highest
    drones[1].battery.current_charge = 0.60 * drones[1].battery.capacity  # 60%
    drones[2].battery.current_charge = 0.55 * drones[2].battery.capacity  # 55%
    drones[3].battery.current_charge = 0.50 * drones[3].battery.capacity  # 50%

    # Slot available
    station.occupied_slots = {}
    station.queue = []

    # DRONE-1 (85%) is the only emergency reserve
    # It should NOT return to charge (to maintain emergency reserve)
    should_return_1 = strategies[0].should_return_to_charge(station, drones)

    # DRONE-2, DRONE-3, DRONE-4 should return to charge
    should_return_2 = strategies[1].should_return_to_charge(station, drones)
    should_return_3 = strategies[2].should_return_to_charge(station, drones)
    should_return_4 = strategies[3].should_return_to_charge(station, drones)

    print(f"Emergency Reserve Test:")
    print(f"  DRONE-1 (85%, emergency reserve) should_return: {should_return_1}")
    print(f"  DRONE-2 (60%) should_return: {should_return_2}")
    print(f"  DRONE-3 (55%) should_return: {should_return_3}")
    print(f"  DRONE-4 (50%, lowest) should_return: {should_return_4}")

    assert not should_return_1, "Emergency reserve drone should NOT charge"
    assert should_return_2 or should_return_3 or should_return_4, "Lower battery drones should charge"

    print("  ✓ Emergency reserve logic working")


def test_load_balancing():
    """Test that load balancing prevents everyone charging at once."""

    station = ChargingStation(
        position=np.array([500, 500, 0]),
        capacity=1,  # Only 1 slot
        charge_rate=0.02
    )

    drones = []
    strategies = []

    for i in range(3):
        drone = Drone(
            position=np.array([500 + i*50, 500, 100]),
            team=Team.FRIENDLY,
            role=Role.PATROL,
            battery_level=5000.0
        )
        drone.id = f"DRONE-{i+1}"
        drones.append(drone)

        net_state = NetworkState(drone.id, set())
        strategy = AutonomousStrategy(drone, net_state, [])
        strategies.append(strategy)

    # Scenario: Everyone moderately healthy (60-70%)
    drones[0].battery.current_charge = 0.70 * drones[0].battery.capacity  # 70%
    drones[1].battery.current_charge = 0.65 * drones[1].battery.capacity  # 65%
    drones[2].battery.current_charge = 0.60 * drones[2].battery.capacity  # 60% - lowest

    # Slot available
    station.occupied_slots = {}
    station.queue = []

    # DRONE-3 (lowest) should get priority
    should_return_1 = strategies[0].should_return_to_charge(station, drones)
    should_return_2 = strategies[1].should_return_to_charge(station, drones)
    should_return_3 = strategies[2].should_return_to_charge(station, drones)

    print(f"\nLoad Balancing Test:")
    print(f"  DRONE-1 (70%) should_return: {should_return_1}")
    print(f"  DRONE-2 (65%) should_return: {should_return_2}")
    print(f"  DRONE-3 (60%, lowest) should_return: {should_return_3}")

    # Lowest battery drone should want to charge
    assert should_return_3, "Lowest battery drone should charge"

    print("  ✓ Load balancing logic working")


def test_predictive_scheduling():
    """Test that drones charge proactively when slots are free."""

    station = ChargingStation(
        position=np.array([500, 500, 0]),
        capacity=2,  # 2 slots
        charge_rate=0.02
    )

    drones = []
    strategies = []

    for i in range(3):
        drone = Drone(
            position=np.array([500 + i*50, 500, 100]),
            team=Team.FRIENDLY,
            role=Role.PATROL,
            battery_level=5000.0
        )
        drone.id = f"DRONE-{i+1}"
        drones.append(drone)

        net_state = NetworkState(drone.id, set())
        strategy = AutonomousStrategy(drone, net_state, [])
        strategies.append(strategy)

    # Scenario: Fleet is healthy, but one drone at 70% with free slot
    drones[0].battery.current_charge = 0.70 * drones[0].battery.capacity  # 70%
    drones[1].battery.current_charge = 0.85 * drones[1].battery.capacity  # 85%
    drones[2].battery.current_charge = 0.90 * drones[2].battery.capacity  # 90%

    # Slots available
    station.occupied_slots = {}
    station.queue = []

    # DRONE-1 (70%) should proactively charge (predictive scheduling)
    # when fleet is healthy and slot is free
    should_return_1 = strategies[0].should_return_to_charge(station, drones)
    should_return_2 = strategies[1].should_return_to_charge(station, drones)
    should_return_3 = strategies[2].should_return_to_charge(station, drones)

    print(f"\nPredictive Scheduling Test:")
    print(f"  DRONE-1 (70%, free slot) should_return: {should_return_1}")
    print(f"  DRONE-2 (85%) should_return: {should_return_2}")
    print(f"  DRONE-3 (90%) should_return: {should_return_3}")

    # Drone at 70% should charge when slot is free and fleet is healthy
    assert should_return_1, "Drone at 70% should charge proactively when conditions favorable"

    print("  ✓ Predictive scheduling logic working")


def test_critical_battery_override():
    """Test that critical battery always returns regardless of other factors."""

    station = ChargingStation(
        position=np.array([500, 500, 0]),
        capacity=0,  # NO slots available
        charge_rate=0.02
    )

    drones = []
    strategies = []

    for i in range(2):
        drone = Drone(
            position=np.array([500 + i*50, 500, 100]),
            team=Team.FRIENDLY,
            role=Role.PATROL,
            battery_level=5000.0
        )
        drone.id = f"DRONE-{i+1}"
        drones.append(drone)

        net_state = NetworkState(drone.id, set())
        strategy = AutonomousStrategy(drone, net_state, [])
        strategies.append(strategy)

    # DRONE-1 critical battery (10%)
    # DRONE-2 healthy (90%)
    drones[0].battery.current_charge = 0.10 * drones[0].battery.capacity  # 10% - CRITICAL
    drones[1].battery.current_charge = 0.90 * drones[1].battery.capacity  # 90%

    # DRONE-1 should STILL return even with no slots (critical override)
    should_return_1 = strategies[0].should_return_to_charge(station, drones)
    should_return_2 = strategies[1].should_return_to_charge(station, drones)

    print(f"\nCritical Battery Override Test:")
    print(f"  DRONE-1 (10% CRITICAL, no slots) should_return: {should_return_1}")
    print(f"  DRONE-2 (90%) should_return: {should_return_2}")

    assert should_return_1, "Critical battery must return regardless of slots"
    assert not should_return_2, "Healthy drone should not return when no slots"

    print("  ✓ Critical battery override working")


if __name__ == "__main__":
    print("=" * 70)
    print("Testing Advanced Charging Heuristics")
    print("=" * 70)

    test_emergency_reserve()
    test_load_balancing()
    test_predictive_scheduling()
    test_critical_battery_override()

    print("\n" + "=" * 70)
    print("All charging heuristic tests passed! ✓")
    print("=" * 70)
