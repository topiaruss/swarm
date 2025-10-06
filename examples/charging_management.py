#!/usr/bin/env python3
"""
Charging Management Demo

Demonstrates:
- Battery discharge during patrol
- Automatic return-to-charge when battery low
- Multi-slot charging station with queue
- Drones resuming patrol after charging
- Battery level visualization
"""

import sys
import os
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.arena import Arena, GeographicReference
from src.entities import Drone, Team, Role, ChargingStation
from src.physics import PhysicsEngine
from src.strategy import AutonomousStrategy, create_perimeter_sectors
from src.network import NetworkState
from src.visualization import ArenaVisualizer


class ChargingManagementSimulation:
    """Simulation demonstrating charging management."""

    def __init__(self):
        # Create arena
        self.arena = Arena(
            bounds=(1000, 1000, 500),
            geo_reference=GeographicReference(
                latitude=37.7749,
                longitude=-122.4194
            )
        )

        # Physics
        self.physics = PhysicsEngine(self.arena)

        # Network states and strategies
        self.network_states = {}
        self.strategies = {}

        # Charging management
        self.charging_drones = {}  # drone_id -> slot
        self.returning_drones = set()  # drone_ids returning to charge

        # Simulation parameters
        self.dt = 0.033  # 33ms timestep (real-time at 30 fps)

        # Setup scenario
        self._setup_scenario()

    def _setup_scenario(self):
        """Create scenario with patrol drones and charging station."""

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
        sectors = create_perimeter_sectors(center, patrol_radius, num_sectors=8)

        # Create 4 patrol drones with varying initial battery levels
        drone_positions = [
            [300, 300, 100],  # SW
            [700, 300, 100],  # SE
            [700, 700, 100],  # NE
            [300, 700, 100],  # NW
        ]

        # Start with different battery levels to stagger charging
        initial_batteries = [
            0.60,  # 60% - will need charging soon
            0.40,  # 40% - will need charging first
            0.70,  # 70% - moderate
            0.50,  # 50% - will need charging
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

        # Create network state and strategy for each drone
        for i, drone_id in enumerate(drone_ids):
            # Each drone expects to see all other drones
            expected_peers = set(drone_ids) - {drone_id}
            net_state = NetworkState(drone_id, expected_peers)
            self.network_states[drone_id] = net_state

            # Get drone entity
            drone = self.arena.get_entity_by_id(drone_id)

            # Create strategy with patrol sectors
            strategy = AutonomousStrategy(drone, net_state, sectors)
            self.strategies[drone_id] = strategy

            # Assign each drone to 2 sectors initially
            sector_idx = i * 2
            strategy.assign_patrol_sector(sectors[sector_idx])

            print(f"[SETUP] {drone_id} assigned to {sectors[sector_idx].sector_id}, battery: {drone.battery.level()*100:.0f}%")

    def update(self):
        """Update simulation step."""

        # Get all patrol drones
        patrol_drones = [
            e for e in self.arena.get_active_entities()
            if e.role == Role.PATROL and isinstance(e, Drone)
        ]

        # Update strategies
        for drone_id, strategy in self.strategies.items():
            strategy.update(self.arena.time)

        # Check charging decisions for each drone
        for drone in patrol_drones:
            strategy = self.strategies.get(drone.id)
            if not strategy:
                continue

            # Currently charging?
            if drone.id in self.charging_drones:
                slot = self.charging_drones[drone.id]

                # Charge the battery
                self.station.charge_drone(drone, self.dt)

                # Check if should leave
                if strategy.should_leave_charger(self.station, patrol_drones):
                    # Release slot
                    self.station.release_slot(slot)
                    del self.charging_drones[drone.id]
                    drone.charging_slot = None

                    print(f"[{self.arena.time:.1f}s] {drone.id} finished charging (battery: {drone.battery.level()*100:.0f}%)")

                    # Resume patrol - move away from station
                    direction = drone.position - self.station.position
                    if np.linalg.norm(direction) > 0:
                        direction = direction / np.linalg.norm(direction)
                        drone.velocity = direction * drone.max_speed * 0.5

                    self.returning_drones.discard(drone.id)

            # Returning to charge?
            elif drone.id in self.returning_drones:
                # Check if arrived at station (within 10m)
                distance_to_station = np.linalg.norm(drone.position - self.station.position)

                if distance_to_station < 10.0:
                    # Request slot
                    slot = self.station.request_slot(drone.id)

                    if slot is not None:
                        # Got a slot - start charging
                        self.charging_drones[drone.id] = slot
                        drone.charging_slot = slot

                        # Move to exact slot position
                        drone.position = self.station.get_slot_position(slot)
                        drone.velocity = np.array([0.0, 0.0, 0.0])

                        print(f"[{self.arena.time:.1f}s] {drone.id} started charging on slot {slot} (battery: {drone.battery.level()*100:.0f}%)")
                    else:
                        # No slot available - hover near station
                        drone.velocity = np.array([0.0, 0.0, 0.0])
                        print(f"[{self.arena.time:.1f}s] {drone.id} waiting for charging slot (battery: {drone.battery.level()*100:.0f}%)")
                else:
                    # Fly toward station
                    direction = self.station.position - drone.position
                    direction = direction / np.linalg.norm(direction)
                    drone.velocity = direction * drone.max_speed

            # Normal patrol
            else:
                # Check if should return to charge (with fleet awareness and timing)
                if strategy.should_return_to_charge(self.station, patrol_drones, self.arena.time):
                    self.returning_drones.add(drone.id)
                    print(f"[{self.arena.time:.1f}s] {drone.id} returning to charge (battery: {drone.battery.level()*100:.0f}%)")
                    continue

                # Get patrol target from strategy
                target = strategy.get_patrol_target(self.arena.time)

                if target is not None:
                    # Move toward target
                    direction = target - drone.position
                    distance = np.linalg.norm(direction)

                    if distance > 1.0:
                        direction = direction / distance
                        drone.velocity = direction * drone.max_speed * 0.5  # Half speed for patrol
                    else:
                        drone.velocity = np.array([0.0, 0.0, 0.0])

                # Discharge battery during patrol (not when charging)
                drone.battery.discharge(self.dt, rate_multiplier=1.0)

        # Update physics for all entities
        self.physics.update(self.arena.get_active_entities(), self.dt)

        # Update arena
        self.arena.update(self.dt)

    def run_visual(self):
        """Run simulation with visualization."""
        viz = ArenaVisualizer(
            self.arena,
            show_detection_range=False,
            show_trails=True,
            trail_length=100
        )

        print("=" * 70)
        print("Charging Management Demo")
        print("=" * 70)
        print()
        print("Scenario:")
        print("  - 4 patrol drones with varying battery levels")
        print("  - 1 charging station with 2 slots at arena center")
        print("  - Battery discharge during patrol (50 mAh/s)")
        print("  - Charge rate: 2%/s (50s for full charge)")
        print()
        print("Battery thresholds:")
        print("  - Critical: <15% (must return immediately)")
        print("  - Optimal: 80% (should return if slot available)")
        print("  - Full: 95% (leave charger)")
        print()
        print("Watch for:")
        print("  - Drones returning when battery low")
        print("  - Queue management (2 slots, 4 drones)")
        print("  - Automatic resumption of patrol after charging")
        print("  - Battery levels in visualization")
        print()

        def update_with_battery_display():
            self.update()

            # Print battery status every 5 seconds
            if int(self.arena.time) % 5 == 0 and self.arena.time > 0 and abs(self.arena.time - int(self.arena.time)) < self.dt:
                print(f"\n[{self.arena.time:.0f}s] Battery Status:")
                for drone in self.arena.get_active_entities():
                    if drone.role == Role.PATROL and isinstance(drone, Drone):
                        status = "CHARGING" if drone.id in self.charging_drones else \
                                "RETURNING" if drone.id in self.returning_drones else \
                                "PATROL"
                        battery_pct = drone.battery.level() * 100
                        battery_bar = "█" * int(battery_pct / 5) + "░" * (20 - int(battery_pct / 5))
                        print(f"  {drone.id}: {battery_bar} {battery_pct:5.1f}% ({status})")

        anim = viz.animate(
            update_func=update_with_battery_display,
            interval=33,
            frames=1200  # 2 minutes at 30 fps
        )

        viz.show()

        # Print final statistics
        print()
        print("=" * 70)
        print("Final Statistics")
        print("=" * 70)

        total_charges = 0
        for drone_id, strategy in self.strategies.items():
            drone = self.arena.get_entity_by_id(drone_id)
            print(f"\n{drone_id}:")
            print(f"  Final battery: {drone.battery.level()*100:.1f}%")
            print(f"  Status: {strategy.get_status_string()}")

        print(f"\nCharging station:")
        print(f"  Occupied slots: {len(self.station.occupied_slots)}/{self.station.capacity}")
        print(f"  Queue length: {len(self.station.queue)}")


def main():
    """Main entry point."""
    sim = ChargingManagementSimulation()
    sim.run_visual()


if __name__ == "__main__":
    main()
