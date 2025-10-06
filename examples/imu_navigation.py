#!/usr/bin/env python3
"""
IMU-Assisted Navigation Example

Demonstrates:
- High-frequency IMU dead reckoning (100 Hz)
- Low-frequency transponder corrections (when drift > threshold)
- Realistic IMU sensor noise and drift
- Position estimation between transponder updates
- Drift accumulation and correction cycles
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
from src.imu import IMUSensor, IMUAssistedNavigator
from src.visualization import ArenaVisualizer


class IMUNavigationSimulation:
    """Simulation demonstrating IMU-assisted navigation."""

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

        # Transponder navigation (for corrections)
        self.transponder_nav = TransponderNavigator(method=SignalType.BOTH)

        # Simulation parameters
        self.dt = 0.01  # 100 Hz for IMU (10ms updates)
        self.transponder_update_interval = 0.1  # Check transponders every 100ms (10 Hz)
        self.last_transponder_update = 0.0

        # Statistics
        self.correction_count = 0
        self.max_drift_seen = 0.0

        # Setup scenario
        self._setup_scenario()

    def _setup_scenario(self):
        """Create scenario with IMU-equipped drone."""

        # Deploy 4 transponders
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
                relocate_interval=60.0
            )
            transponder.id = f"TRANSPONDER-{i+1}"
            self.arena.add_entity(transponder)

        # Create patrol drone with IMU
        patrol = Drone(
            position=[500, 500, 100],
            team=Team.FRIENDLY,
            role=Role.PATROL,
            max_speed=15.0
        )
        patrol.id = "PATROL-IMU"
        patrol.true_position = patrol.position.copy()
        patrol.true_velocity = np.array([0.0, 0.0, 0.0])

        # Initialize IMU sensor (low-cost consumer grade)
        patrol.imu_sensor = IMUSensor(
            accel_noise_std=0.02,      # m/s² (MPU6050 typical)
            accel_bias_drift=0.005,    # m/s²/s
            gyro_noise_std=0.01,       # rad/s
            gyro_bias_drift=0.001,     # rad/s/s
            update_rate=100.0          # 100 Hz
        )

        # Initialize IMU-assisted navigator
        patrol.imu_nav = IMUAssistedNavigator(
            imu_sensor=patrol.imu_sensor,
            initial_position=patrol.position.copy(),
            initial_velocity=patrol.true_velocity.copy(),
            correction_threshold=2.0   # Correct when drift > 2m
        )

        # EWMA filter for transponder corrections
        patrol.ewma_alpha = 0.15  # Low alpha: trust smooth IMU more than noisy transponders
        patrol.filtered_position = patrol.position.copy()
        patrol.position_uncertainty = 0.0
        self.arena.add_entity(patrol)

    def get_transponder_signals(self, drone_position: np.ndarray) -> list:
        """Get signals from transponders."""
        signals = []

        transponders = [e for e in self.arena.get_active_entities()
                       if isinstance(e, Transponder)]

        for transponder in transponders:
            distance = np.linalg.norm(drone_position - transponder.position)
            rssi = self.transponder_nav.distance_to_rssi(distance, add_noise=True)
            bearing = self.transponder_nav.calculate_bearing(
                drone_position,
                transponder.position,
                add_noise=True
            )

            signal_quality = max(0.3, 1.0 - distance / 1000.0)

            signal = TransponderSignal(
                transponder_id=transponder.id,
                transponder_position=transponder.position,
                rssi=rssi,
                bearing=bearing,
                distance_estimate=self.transponder_nav.rssi_to_distance(rssi),
                signal_quality=signal_quality
            )
            signals.append(signal)

        return signals

    def update(self):
        """Update simulation step."""

        # Update patrol drone circular path (true motion)
        for entity in self.arena.get_active_entities():
            if entity.role == Role.PATROL and isinstance(entity, Drone):
                center = self.arena.get_bounds_center()
                angular_velocity = 0.1
                angle = angular_velocity * self.arena.time
                radius = 200

                x = center[0] + radius * np.cos(angle)
                y = center[1] + radius * np.sin(angle)
                z = 100

                vx = -radius * angular_velocity * np.sin(angle)
                vy = radius * angular_velocity * np.cos(angle)
                vz = 0

                entity.true_position = np.array([x, y, z])
                entity.true_velocity = np.array([vx, vy, vz])

                # Calculate true acceleration (derivative of velocity)
                # For circular motion: a = -ω²r (centripetal)
                omega_squared = angular_velocity ** 2
                ax = -omega_squared * radius * np.cos(angle)
                ay = -omega_squared * radius * np.sin(angle)
                az = 0
                true_acceleration = np.array([ax, ay, az])

                # Update IMU navigation (high frequency - every step)
                pos_est, vel_est, drift = entity.imu_nav.update_imu(
                    true_acceleration=true_acceleration,
                    true_angular_velocity=np.array([0, 0, angular_velocity]),
                    timestamp=self.arena.time,
                    dt=self.dt
                )

                # Track max drift
                self.max_drift_seen = max(self.max_drift_seen, drift)

                # Check if we should correct with transponders
                should_correct = entity.imu_nav.should_correct()

                # Also do periodic transponder updates (even if below threshold)
                time_for_update = (self.arena.time - self.last_transponder_update) >= self.transponder_update_interval

                if should_correct or time_for_update:
                    # Get transponder position fix
                    signals = self.get_transponder_signals(entity.true_position)

                    if len(signals) >= 2:
                        try:
                            transponder_pos, uncertainty = self.transponder_nav.calculate_position(signals)

                            # Apply EWMA filter to transponder correction
                            # Low alpha (0.15) = trust smooth IMU more than noisy transponders
                            current_imu_pos = entity.imu_nav.get_position()
                            blended_pos = (
                                entity.ewma_alpha * transponder_pos +
                                (1 - entity.ewma_alpha) * current_imu_pos
                            )

                            # Correct IMU drift with SMOOTHED position
                            entity.imu_nav.correct_with_transponder(
                                blended_pos,
                                entity.true_velocity  # Use true velocity for now
                            )

                            # Update filtered position for display
                            entity.filtered_position = blended_pos
                            entity.position_uncertainty = uncertainty

                            if should_correct:
                                self.correction_count += 1
                                print(
                                    f"[{self.arena.time:6.2f}s] IMU drift corrected: "
                                    f"{drift:.2f}m → transponder fix (±{uncertainty:.1f}m) "
                                    f"[alpha={entity.ewma_alpha}]"
                                )

                            self.last_transponder_update = self.arena.time

                        except ValueError as e:
                            pass

                # Use IMU position estimate
                entity.position = entity.imu_nav.get_position()
                entity.velocity = entity.imu_nav.get_velocity()

                # Calculate actual error (for display)
                actual_error = np.linalg.norm(entity.position - entity.true_position)

                # Periodic status
                if int(self.arena.time * 10) % 10 == 0:  # Every 1 second
                    print(
                        f"[{self.arena.time:6.1f}s] IMU drift: {drift:.2f}m, "
                        f"Actual error: {actual_error:.2f}m, "
                        f"Corrections: {self.correction_count}"
                    )

        self.arena.update(self.dt)

    def run_visual(self):
        """Run simulation with visualization."""
        viz = ArenaVisualizer(
            self.arena,
            show_detection_range=False,
            show_trails=True,
            trail_length=300,
            show_uncertainty=True
        )

        print("=" * 70)
        print("IMU-Assisted Navigation Simulation")
        print("=" * 70)
        print()
        print("Configuration:")
        print("  - IMU update rate: 100 Hz (high frequency)")
        print("  - Transponder check: 10 Hz (when needed)")
        print("  - Correction threshold: 2.0m drift")
        print("  - EWMA alpha: 0.15 (trust IMU 85%, transponders 15%)")
        print("  - IMU sensor: Low-cost consumer grade (MPU6050)")
        print()
        print("Watch for:")
        print("  - Smooth position from IMU dead reckoning")
        print("  - Periodic corrections when drift accumulates")
        print("  - Position uncertainty visualization")
        print()

        def update_with_uncertainty():
            self.update()
            # Update uncertainty visualization
            for entity in self.arena.get_active_entities():
                if isinstance(entity, Drone) and hasattr(entity, 'imu_nav'):
                    drift = entity.imu_nav.get_drift()
                    viz.update_uncertainty(entity.id, drift)

        # Animate (60 seconds at 100 Hz = 6000 frames, but display at 30 fps)
        anim = viz.animate(
            update_func=update_with_uncertainty,
            interval=33,  # 30 fps display
            frames=1800   # 60 seconds worth
        )

        viz.show()

        print()
        print("=" * 70)
        print("Simulation Statistics")
        print("=" * 70)
        print(f"Total corrections: {self.correction_count}")
        print(f"Max drift seen: {self.max_drift_seen:.2f}m")
        print(f"Avg time between corrections: {60.0/max(self.correction_count, 1):.1f}s")


def main():
    """Main entry point."""
    sim = IMUNavigationSimulation()
    sim.run_visual()


if __name__ == "__main__":
    main()
