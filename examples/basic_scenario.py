#!/usr/bin/env python3
"""
Basic scenario: Patrol drones detect and track intruder.

Demonstrates:
- Arena setup with boundaries
- Friendly patrol drones circling an area
- Enemy intruder entering the arena
- Detection events logged
- Real-time 3D visualization
"""

import sys
import os
import numpy as np

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.arena import Arena, GeographicReference
from src.entities import Drone, Team, Role
from src.physics import PhysicsEngine
from src.detection import Detector
from src.visualization import ArenaVisualizer


class Simulation:
    """Main simulation controller."""

    def __init__(self):
        # Create arena: 1km x 1km x 500m
        self.arena = Arena(
            bounds=(1000, 1000, 500),
            geo_reference=GeographicReference(
                latitude=37.7749,  # Example: San Francisco
                longitude=-122.4194
            )
        )

        # Physics and detection
        self.physics = PhysicsEngine(self.arena)
        self.detector = Detector(max_range=200)

        # Simulation parameters
        self.dt = 0.1  # Time step: 100ms
        self.detection_log = []

        # Setup scenario
        self._setup_scenario()

    def _setup_scenario(self):
        """Create the scenario: 3 patrol drones + 1 intruder."""

        # Patrol area center
        center = self.arena.get_bounds_center()
        patrol_altitude = 100  # meters

        # Create 3 friendly patrol drones in circular formation with phase offsets
        for i in range(3):
            angle = (i / 3) * 2 * np.pi
            radius = 200  # meters from center

            x = center[0] + radius * np.cos(angle)
            y = center[1] + radius * np.sin(angle)

            drone = Drone(
                position=[x, y, patrol_altitude],
                team=Team.FRIENDLY,
                role=Role.PATROL,
                max_speed=15.0,
                detection_range=200.0
            )
            drone.id = f"PATROL-{i+1}"
            # Store phase offset so drones are separated
            drone.phase_offset = angle
            self.arena.add_entity(drone)

        # Create enemy intruder entering from corner on clear diagonal path
        intruder = Drone(
            position=[100, 100, 80],  # Start near SW corner
            velocity=[8, 8, 1],      # Clear diagonal toward center
            team=Team.ENEMY,
            role=Role.INTRUDER,
            max_speed=20.0
        )
        intruder.id = "INTRUDER-1"
        self.arena.add_entity(intruder)

    def update(self):
        """Update simulation by one time step."""

        # Update patrol drones to circle with phase offsets
        center = self.arena.get_bounds_center()
        angular_velocity = 0.1  # radians per second

        for entity in self.arena.get_active_entities():
            if entity.role == Role.PATROL:
                # Use phase offset if available
                phase = getattr(entity, 'phase_offset', 0)
                angle = angular_velocity * self.arena.time + phase
                radius = 200

                x = center[0] + radius * np.cos(angle)
                y = center[1] + radius * np.sin(angle)
                z = 100

                vx = -radius * angular_velocity * np.sin(angle)
                vy = radius * angular_velocity * np.cos(angle)
                vz = 0

                entity.position = np.array([x, y, z])
                entity.velocity = np.array([vx, vy, vz])

        # Update physics
        self.physics.update(self.arena.get_active_entities(), self.dt)

        # Run detection for each friendly drone
        for entity in self.arena.get_active_entities():
            if entity.team == Team.FRIENDLY and isinstance(entity, Drone):
                detections = self.detector.detect(entity, self.arena.get_active_entities())
                foes = self.detector.filter_foes(detections)

                # Log new detections
                for detection in foes:
                    log_entry = {
                        'time': self.arena.time,
                        'detector': entity.id,
                        'target': detection.entity.id,
                        'distance': detection.distance,
                        'position': detection.entity.position.copy()
                    }
                    self.detection_log.append(log_entry)

                    # Print detection event
                    print(
                        f"[{self.arena.time:6.1f}s] {entity.id} detected {detection.entity.id} "
                        f"at {detection.distance:.1f}m"
                    )

        # Update arena time
        self.arena.update(self.dt)

    def run_headless(self, duration: float):
        """
        Run simulation without visualization.

        Args:
            duration: Simulation duration in seconds
        """
        steps = int(duration / self.dt)
        print(f"Running simulation for {duration}s ({steps} steps)...")
        print()

        for _ in range(steps):
            self.update()

        print()
        print(f"Simulation complete. Total detections: {len(self.detection_log)}")

    def run_visual(self):
        """Run simulation with 3D visualization."""
        viz = ArenaVisualizer(
            self.arena,
            show_detection_range=True,
            show_trails=True,
            trail_length=200  # Longer trails for longer sim
        )

        print("Starting visual simulation...")
        print("Close the plot window to end.")
        print()

        # Animate - 4x longer (120 seconds)
        anim = viz.animate(
            update_func=self.update,
            interval=100,  # 100ms between frames
            frames=1200    # 120 seconds of simulation
        )

        viz.show()


def main():
    """Main entry point."""
    print("=" * 60)
    print("Drone Swarm Defense Simulation - Basic Scenario")
    print("=" * 60)
    print()
    print("Scenario: 3 patrol drones detect an intruder")
    print()

    sim = Simulation()

    # Run visual simulation
    sim.run_visual()

    # Print summary
    print()
    print("=" * 60)
    print("Simulation Summary")
    print("=" * 60)
    print(f"Total simulation time: {sim.arena.time:.1f}s")
    print(f"Total entities: {len(sim.arena.entities)}")
    print(f"Detection events: {len(sim.detection_log)}")

    if sim.detection_log:
        print()
        print("First detection:")
        first = sim.detection_log[0]
        print(f"  Time: {first['time']:.1f}s")
        print(f"  Detector: {first['detector']}")
        print(f"  Target: {first['target']}")
        print(f"  Distance: {first['distance']:.1f}m")


if __name__ == "__main__":
    main()
