#!/usr/bin/env python3
"""
GPS-Denied Navigation Scenario

Demonstrates:
- Transponder-based positioning (RSSI + bearing)
- Sensor fusion for robust navigation
- Transponder relocation to avoid targeting
- Position uncertainty visualization
- Drones navigating without GPS
"""

import sys
import os
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.arena import Arena, GeographicReference
from src.entities import Drone, Transponder, Team, Role
from src.physics import PhysicsEngine
from src.detection import Detector
from src.navigation import TransponderNavigator, TransponderSignal, SignalType
from src.visualization import ArenaVisualizer


class GPSDeniedSimulation:
    """Simulation with GPS-denied navigation using transponders."""

    def __init__(self):
        # Create arena
        self.arena = Arena(
            bounds=(1000, 1000, 500),
            geo_reference=GeographicReference(
                latitude=37.7749,
                longitude=-122.4194
            )
        )

        # Physics and detection
        self.physics = PhysicsEngine(self.arena)
        self.detector = Detector(max_range=200)

        # Navigation system (sensor fusion: RSSI + bearing)
        self.navigator = TransponderNavigator(method=SignalType.BOTH)

        # Simulation parameters
        self.dt = 0.1
        self.gps_jammed = True  # GPS is denied

        # Setup scenario
        self._setup_scenario()

    def _setup_scenario(self):
        """Create scenario: transponders + drones navigating without GPS."""

        # Deploy 4 transponders around the arena (hidden under vegetation)
        transponder_positions = [
            [200, 200, 0],   # SW
            [800, 200, 0],   # SE
            [800, 800, 0],   # NE
            [200, 800, 0],   # NW
        ]

        for i, pos in enumerate(transponder_positions):
            transponder = Transponder(
                position=pos,
                team=Team.FRIENDLY,
                role=Role.SUPPORT,
                relocate_interval=30.0  # Relocate every 30 seconds for demo
            )
            transponder.id = f"TRANSPONDER-{i+1}"
            self.arena.add_entity(transponder)

        # Create patrol drone (knows transponder locations via encrypted signal)
        patrol = Drone(
            position=[500, 500, 100],  # Start at center
            team=Team.FRIENDLY,
            role=Role.PATROL,
            max_speed=15.0
        )
        patrol.id = "PATROL-GPS-DENIED"
        patrol.true_position = patrol.position.copy()  # For comparison
        patrol.estimated_position = patrol.position.copy()
        patrol.filtered_position = patrol.position.copy()  # EWMA filtered
        patrol.position_uncertainty = 0.0
        patrol.ewma_alpha = 0.35  # Decay over ~5 samples (2/(5+1) ≈ 0.33)
        self.arena.add_entity(patrol)

        # Create intruder (also using transponders)
        intruder = Drone(
            position=[100, 100, 80],
            velocity=[8, 8, 1],
            team=Team.ENEMY,
            role=Role.INTRUDER,
            max_speed=20.0
        )
        intruder.id = "INTRUDER-GPS-DENIED"
        intruder.true_position = intruder.position.copy()
        intruder.estimated_position = intruder.position.copy()
        intruder.filtered_position = intruder.position.copy()  # EWMA filtered
        intruder.position_uncertainty = 0.0
        intruder.ewma_alpha = 0.35  # Decay over ~5 samples
        self.arena.add_entity(intruder)

    def get_transponder_signals(self, drone_position: np.ndarray) -> list:
        """Simulate receiving signals from transponders."""
        signals = []

        transponders = [e for e in self.arena.get_active_entities()
                       if isinstance(e, Transponder)]

        for transponder in transponders:
            # Calculate real distance
            distance = np.linalg.norm(drone_position - transponder.position)

            # Simulate RSSI (with noise)
            rssi = self.navigator.distance_to_rssi(distance, add_noise=True)

            # Simulate bearing (with noise)
            bearing = self.navigator.calculate_bearing(
                drone_position,
                transponder.position,
                add_noise=True
            )

            # Signal quality degrades with distance
            signal_quality = max(0.3, 1.0 - distance / 1000.0)

            signal = TransponderSignal(
                transponder_id=transponder.id,
                transponder_position=transponder.position,
                rssi=rssi,
                bearing=bearing,
                distance_estimate=self.navigator.rssi_to_distance(rssi),
                signal_quality=signal_quality
            )
            signals.append(signal)

        return signals

    def update_drone_navigation(self, drone: Drone):
        """Update drone position estimate using transponder signals with EWMA filtering."""
        # Get signals from transponders
        signals = self.get_transponder_signals(drone.true_position)

        if len(signals) >= 2:
            try:
                # Calculate raw position using sensor fusion
                estimated_pos, uncertainty = self.navigator.calculate_position(signals)

                # Apply EWMA filter: filtered = alpha * new + (1-alpha) * old
                # This smooths out noise, with ~5 sample decay
                alpha = drone.ewma_alpha
                drone.filtered_position = (
                    alpha * estimated_pos + (1 - alpha) * drone.filtered_position
                )

                # Store both raw and filtered
                drone.estimated_position = estimated_pos
                drone.position_uncertainty = uncertainty

                # In GPS-denied mode, drone uses FILTERED position
                if self.gps_jammed:
                    drone.position = drone.filtered_position

            except ValueError as e:
                print(f"[{self.arena.time:.1f}s] {drone.id} navigation failed: {e}")

    def relocate_transponders(self):
        """Relocate transponders if needed."""
        transponders = [e for e in self.arena.get_active_entities()
                       if isinstance(e, Transponder)]

        for transponder in transponders:
            if transponder.should_relocate(self.arena.time):
                # Move to random nearby location (within 100m)
                offset = np.random.uniform(-100, 100, 2)
                new_pos = transponder.position[:2] + offset

                # Keep within arena bounds
                new_pos = np.clip(new_pos, [50, 50], self.arena.bounds[:2] - 50)

                transponder.schedule_relocation([new_pos[0], new_pos[1], 0])
                transponder.execute_relocation(self.arena.time)

                print(f"[{self.arena.time:.1f}s] {transponder.id} relocated to ({new_pos[0]:.0f}, {new_pos[1]:.0f})")

    def update(self):
        """Update simulation by one time step."""
        center = self.arena.get_bounds_center()
        angular_velocity = 0.1

        # Update patrol drone circular path
        for entity in self.arena.get_active_entities():
            if entity.role == Role.PATROL and isinstance(entity, Drone):
                angle = angular_velocity * self.arena.time
                radius = 200

                x = center[0] + radius * np.cos(angle)
                y = center[1] + radius * np.sin(angle)
                z = 100

                vx = -radius * angular_velocity * np.sin(angle)
                vy = radius * angular_velocity * np.cos(angle)
                vz = 0

                # Update true position
                entity.true_position = np.array([x, y, z])
                entity.velocity = np.array([vx, vy, vz])

        # Update physics (for intruder)
        for entity in self.arena.get_active_entities():
            if entity.role == Role.INTRUDER and isinstance(entity, Drone):
                entity.true_position = entity.true_position + entity.velocity * self.dt
                entity.true_position = self.arena.enforce_bounds(entity.true_position)

        # Update navigation for all drones
        for entity in self.arena.get_active_entities():
            if isinstance(entity, Drone):
                self.update_drone_navigation(entity)

        # Relocate transponders periodically
        self.relocate_transponders()

        # Run detection
        for entity in self.arena.get_active_entities():
            if entity.team == Team.FRIENDLY and isinstance(entity, Drone):
                detections = self.detector.detect(entity, self.arena.get_active_entities())
                foes = self.detector.filter_foes(detections)

                if foes:
                    for det in foes:
                        # Only print occasionally
                        if int(self.arena.time * 10) % 10 == 0:
                            print(
                                f"[{self.arena.time:6.1f}s] {entity.id} detected {det.entity.id} "
                                f"at {det.distance:.1f}m (uncertainty: ±{entity.position_uncertainty:.1f}m)"
                            )

        self.arena.update(self.dt)

    def run_visual(self):
        """Run simulation with 3D visualization."""
        viz = ArenaVisualizer(
            self.arena,
            show_detection_range=True,
            show_trails=True,
            trail_length=100,
            show_uncertainty=True  # Show position uncertainty
        )

        print("Starting GPS-denied navigation simulation...")
        print("Transponders provide positioning via RSSI + bearing")
        print("Blue triangles = transponders (relocate every 30s)")
        print("Dotted circles = position uncertainty")
        print()

        # Custom update function
        def update_with_uncertainty():
            self.update()
            # Update uncertainty visualization
            for entity in self.arena.get_active_entities():
                if isinstance(entity, Drone) and hasattr(entity, 'position_uncertainty'):
                    viz.update_uncertainty(entity.id, entity.position_uncertainty)

        # Animate
        anim = viz.animate(
            update_func=update_with_uncertainty,
            interval=100,  # 100ms between frames
            frames=600     # 60 seconds
        )

        viz.show()


def main():
    """Main entry point."""
    print("=" * 70)
    print("GPS-Denied Navigation Simulation")
    print("=" * 70)
    print()
    print("Scenario: Drones navigate using transponder signals")
    print("Methods: RSSI-based trilateration + bearing triangulation")
    print("         Sensor fusion combines both for best accuracy")
    print()

    sim = GPSDeniedSimulation()
    sim.run_visual()

    print()
    print("=" * 70)
    print("Simulation Complete")
    print("=" * 70)


if __name__ == "__main__":
    main()
