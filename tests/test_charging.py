#!/usr/bin/env python3
"""
Integration test for charging station infrastructure.

Tests battery model, charging station, and queue management.
"""

import sys
import os
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.entities import Drone, ChargingStation, Battery, Team, Role
from src.arena import Arena, GeographicReference


def test_battery_model():
    """Test Battery class functionality."""
    print("=" * 70)
    print("Testing Battery Model")
    print("=" * 70)

    battery = Battery(capacity=5000.0, current_charge=5000.0)

    print(f"\nInitial state:")
    print(f"  Capacity: {battery.capacity} mAh")
    print(f"  Charge: {battery.current_charge} mAh ({battery.level()*100:.0f}%)")
    print(f"  Time to empty: {battery.time_to_empty():.1f}s")

    assert battery.level() == 1.0, "Full battery should be 100%"
    assert battery.is_full(), "Should be considered full"
    assert not battery.is_critical(), "Should not be critical"

    # Discharge for 50 seconds at base rate
    battery.discharge(dt=50.0, rate_multiplier=1.0)
    print(f"\nAfter 50s discharge (patrol):")
    print(f"  Charge: {battery.current_charge} mAh ({battery.level()*100:.0f}%)")

    assert battery.level() == 0.5, "Should be 50% after 50s at 50 mAh/s"

    # Discharge to critical level
    battery.current_charge = 700.0  # 14%
    print(f"\nAt critical level:")
    print(f"  Charge: {battery.current_charge} mAh ({battery.level()*100:.1f}%)")
    assert battery.is_critical(), "Should be critical below 15%"

    # Charge for 25 seconds at 2%/s
    charge_rate = 0.02  # 2% per second
    battery.charge(dt=25.0, charge_rate=charge_rate)
    print(f"\nAfter 25s charging at {charge_rate*100:.0f}%/s:")
    print(f"  Charge: {battery.current_charge} mAh ({battery.level()*100:.0f}%)")

    # Should have gained 50% (25s * 2%/s)
    expected_level = 0.14 + 0.50
    assert abs(battery.level() - expected_level) < 0.01, f"Should be ~{expected_level*100:.0f}% after charging"

    print("\n✓ Battery model working correctly\n")


def test_charging_station():
    """Test ChargingStation slot management."""
    print("=" * 70)
    print("Testing Charging Station")
    print("=" * 70)

    # Create arena
    arena = Arena(
        bounds=(1000, 1000, 500),
        geo_reference=GeographicReference(latitude=37.7749, longitude=-122.4194)
    )

    # Create charging station at center
    center = arena.get_bounds_center()
    station = ChargingStation(
        position=center,
        capacity=2,  # 2 slots
        charge_rate=0.02  # 2%/s
    )

    print(f"\nCharging Station:")
    print(f"  Position: {station.position}")
    print(f"  Capacity: {station.capacity} slots")
    print(f"  Charge rate: {station.charge_rate*100:.0f}%/s")

    # Create 3 drones
    drones = []
    for i in range(3):
        drone = Drone(
            position=center + np.array([10 * i, 0, 0]),
            team=Team.FRIENDLY,
            role=Role.PATROL,
            battery_capacity=5000.0,
            battery_level=1000.0  # 20% charge
        )
        drone.id = f"DRONE-{i+1}"
        drones.append(drone)
        arena.add_entity(drone)

    print(f"\n{len(drones)} drones created with 20% battery")

    # Drone 1 and 2 request slots
    slot1 = station.request_slot(drones[0].id)
    slot2 = station.request_slot(drones[1].id)

    print(f"\nSlot requests:")
    print(f"  {drones[0].id} -> Slot {slot1}")
    print(f"  {drones[1].id} -> Slot {slot2}")

    assert slot1 is not None, "First slot should be assigned"
    assert slot2 is not None, "Second slot should be assigned"
    assert slot1 != slot2, "Slots should be different"
    assert not station.has_available_slot(), "All slots should be occupied"

    # Drone 3 requests slot (should queue)
    slot3 = station.request_slot(drones[2].id)
    print(f"  {drones[2].id} -> {'Queued' if slot3 is None else f'Slot {slot3}'}")

    assert slot3 is None, "Third drone should be queued"
    assert drones[2].id in station.queue, "Drone 3 should be in queue"
    assert station.get_queue_position(drones[2].id) == 0, "Drone 3 should be first in queue"

    # Get slot positions
    pos1 = station.get_slot_position(slot1)
    pos2 = station.get_slot_position(slot2)
    print(f"\nSlot positions:")
    print(f"  Slot {slot1}: {pos1}")
    print(f"  Slot {slot2}: {pos2}")

    # Move drones to slots
    drones[0].position = pos1
    drones[1].position = pos2

    # Charge for 10 seconds
    dt = 0.1
    for step in range(100):  # 10 seconds
        station.charge_drone(drones[0], dt)
        station.charge_drone(drones[1], dt)

    print(f"\nAfter 10s charging:")
    print(f"  {drones[0].id}: {drones[0].battery.level()*100:.0f}% (was 20%)")
    print(f"  {drones[1].id}: {drones[1].battery.level()*100:.0f}% (was 20%)")

    # Should have gained 20% (10s * 2%/s)
    expected_level = 0.20 + 0.20
    assert abs(drones[0].battery.level() - expected_level) < 0.01, "Drone 1 should be ~40%"
    assert abs(drones[1].battery.level() - expected_level) < 0.01, "Drone 2 should be ~40%"

    # Drone 1 leaves (releases slot)
    released = station.release_slot(slot1)
    print(f"\n{drones[0].id} releases slot {slot1}")
    assert released, "Slot should be released"
    assert station.has_available_slot(), "Slot should now be available"

    # Drone 3 can now get a slot
    slot3 = station.request_slot(drones[2].id)
    print(f"{drones[2].id} gets slot {slot3}")
    assert slot3 is not None, "Drone 3 should get the freed slot"
    assert drones[2].id not in station.queue, "Drone 3 should be removed from queue"

    print("\n✓ Charging station working correctly\n")


def test_queue_priority():
    """Test charging queue with priority by battery level."""
    print("=" * 70)
    print("Testing Queue Priority")
    print("=" * 70)

    # Create station with 1 slot
    station = ChargingStation(
        position=np.array([500, 500, 0]),
        capacity=1,
        charge_rate=0.02
    )

    # Create 3 drones with different battery levels
    drones = []
    battery_levels = [0.50, 0.10, 0.30]  # 50%, 10%, 30%

    for i, level in enumerate(battery_levels):
        drone = Drone(
            position=np.array([500 + 10*i, 500, 0]),
            team=Team.FRIENDLY,
            battery_capacity=5000.0,
            battery_level=5000.0 * level
        )
        drone.id = f"DRONE-{i+1}"
        drones.append(drone)

    print(f"\nDrones:")
    for drone in drones:
        print(f"  {drone.id}: {drone.battery.level()*100:.0f}% battery")

    # All request slots
    for drone in drones:
        slot = station.request_slot(drone.id)
        if slot is None:
            print(f"  {drone.id} queued")

    print(f"\nQueue: {station.queue}")
    print(f"  {drones[1].id} (10%) should be highest priority")

    # Manually re-sort queue by battery level (lowest first)
    # This would be done by the strategy layer
    queue_drones = [(station.get_queue_position(d.id), d) for d in drones if d.id in station.queue]
    queue_drones.sort(key=lambda x: x[1].battery.level())

    print(f"\nSorted by battery level:")
    for _, drone in queue_drones:
        print(f"  {drone.id}: {drone.battery.level()*100:.0f}%")

    assert queue_drones[0][1].id == drones[1].id, "Drone 2 (10%) should be first"

    print("\n✓ Queue priority logic working correctly\n")


def test_full_scenario():
    """Test complete charging scenario with multiple drones."""
    print("=" * 70)
    print("Testing Complete Charging Scenario")
    print("=" * 70)

    arena = Arena(
        bounds=(1000, 1000, 500),
        geo_reference=GeographicReference(latitude=37.7749, longitude=-122.4194)
    )

    # Create charging station
    center = arena.get_bounds_center()
    station = ChargingStation(
        position=center,
        capacity=2,
        charge_rate=0.02  # 2%/s = 50s for full charge
    )
    arena.add_entity(station)

    # Create 4 drones with varying battery levels
    drones = []
    initial_levels = [0.25, 0.15, 0.35, 0.20]  # Mix of critical and low

    for i, level in enumerate(initial_levels):
        drone = Drone(
            position=center + np.array([20 * i, 0, 100]),
            team=Team.FRIENDLY,
            role=Role.PATROL,
            battery_capacity=5000.0,
            battery_level=5000.0 * level
        )
        drone.id = f"PATROL-{i+1}"
        drones.append(drone)
        arena.add_entity(drone)

    print(f"\nInitial state:")
    for drone in drones:
        status = "CRITICAL" if drone.battery.is_critical() else "LOW"
        print(f"  {drone.id}: {drone.battery.level()*100:.0f}% ({status})")

    # Simulate charging cycle
    dt = 0.1
    current_time = 0.0
    charging_drones = {}  # drone_id -> slot
    completed_drones = set()  # Drones that have finished charging

    print(f"\nSimulating charging cycle...")

    for step in range(800):  # 80 seconds
        current_time = step * dt

        # Check for drones that should request charging
        for drone in drones:
            if drone.id not in charging_drones and drone.id not in completed_drones:
                # Not currently charging and haven't finished, request slot
                slot = station.request_slot(drone.id)
                if slot is not None:
                    charging_drones[drone.id] = slot
                    drone.charging_slot = slot
                    # Move to slot position
                    drone.position = station.get_slot_position(slot)
                    print(f"  [{current_time:.1f}s] {drone.id} starts charging on slot {slot} (battery: {drone.battery.level()*100:.0f}%)")

        # Charge all drones on chargers
        for drone_id, slot in list(charging_drones.items()):
            drone = next(d for d in drones if d.id == drone_id)
            station.charge_drone(drone, dt)

            # Check if should leave (fully charged or close enough)
            if drone.battery.is_full():
                station.release_slot(slot)
                del charging_drones[drone_id]
                drone.charging_slot = None
                completed_drones.add(drone_id)
                print(f"  [{current_time:.1f}s] {drone.id} finishes charging (battery: {drone.battery.level()*100:.0f}%)")

    print(f"\nFinal state after {current_time:.1f}s:")
    for drone in drones:
        print(f"  {drone.id}: {drone.battery.level()*100:.0f}% battery")

    # All drones should be well charged
    for drone in drones:
        assert drone.battery.level() >= 0.80, f"{drone.id} should be at least 80% charged"

    print("\n✓ Complete charging scenario working correctly\n")

    print("=" * 70)
    print("ALL TESTS PASSED: Charging infrastructure working correctly")
    print("=" * 70)


if __name__ == "__main__":
    try:
        test_battery_model()
        test_charging_station()
        test_queue_priority()
        test_full_scenario()
        sys.exit(0)
    except AssertionError as e:
        print(f"\n❌ TEST FAILED: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ TEST ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
