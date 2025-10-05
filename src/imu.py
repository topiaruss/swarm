"""
IMU (Inertial Measurement Unit) sensor model for dead reckoning.

Simulates low-cost IMU (accelerometer + gyroscope) with realistic noise
and drift characteristics for GPS-denied navigation.
"""

import numpy as np
from dataclasses import dataclass
from typing import Tuple


@dataclass
class IMUReading:
    """
    Raw IMU sensor reading.

    Attributes:
        acceleration: 3D acceleration [ax, ay, az] in m/s²
        angular_velocity: 3D angular velocity [wx, wy, wz] in rad/s
        timestamp: Time of reading in seconds
    """
    acceleration: np.ndarray
    angular_velocity: np.ndarray
    timestamp: float


class IMUSensor:
    """
    Low-cost IMU sensor model.

    Simulates realistic sensor characteristics:
    - Accelerometer noise and bias drift
    - Gyroscope noise and bias drift
    - Integration errors that accumulate over time

    Typical for consumer-grade IMUs (MPU6050, BMI160, etc.)
    """

    def __init__(
        self,
        accel_noise_std: float = 0.02,      # m/s² (realistic for MPU6050)
        accel_bias_drift: float = 0.005,    # m/s² per second
        gyro_noise_std: float = 0.01,       # rad/s (realistic for MPU6050)
        gyro_bias_drift: float = 0.001,     # rad/s per second
        update_rate: float = 100.0          # Hz (typical: 50-200 Hz)
    ):
        """
        Initialize IMU sensor.

        Args:
            accel_noise_std: Accelerometer noise std deviation
            accel_bias_drift: Accelerometer bias drift rate
            gyro_noise_std: Gyroscope noise std deviation
            gyro_bias_drift: Gyroscope bias drift rate
            update_rate: Sensor update frequency in Hz
        """
        self.accel_noise_std = accel_noise_std
        self.accel_bias_drift = accel_bias_drift
        self.gyro_noise_std = gyro_noise_std
        self.gyro_bias_drift = gyro_bias_drift
        self.update_rate = update_rate
        self.dt = 1.0 / update_rate

        # Current bias (starts small, drifts over time)
        self.accel_bias = np.random.normal(0, 0.01, 3)
        self.gyro_bias = np.random.normal(0, 0.001, 3)

        # Last update time
        self.last_update = 0.0

    def read(
        self,
        true_acceleration: np.ndarray,
        true_angular_velocity: np.ndarray,
        timestamp: float
    ) -> IMUReading:
        """
        Simulate IMU reading with realistic noise and drift.

        Args:
            true_acceleration: True acceleration in body frame
            true_angular_velocity: True angular velocity in body frame
            timestamp: Current time in seconds

        Returns:
            IMUReading with noisy sensor data
        """
        # Update bias drift
        dt_drift = timestamp - self.last_update
        if dt_drift > 0:
            self.accel_bias += np.random.normal(0, self.accel_bias_drift * dt_drift, 3)
            self.gyro_bias += np.random.normal(0, self.gyro_bias_drift * dt_drift, 3)

        self.last_update = timestamp

        # Add noise and bias to measurements
        accel_noise = np.random.normal(0, self.accel_noise_std, 3)
        gyro_noise = np.random.normal(0, self.gyro_noise_std, 3)

        measured_accel = true_acceleration + self.accel_bias + accel_noise
        measured_gyro = true_angular_velocity + self.gyro_bias + gyro_noise

        return IMUReading(
            acceleration=measured_accel,
            angular_velocity=measured_gyro,
            timestamp=timestamp
        )


class DeadReckoning:
    """
    Dead reckoning using IMU integration.

    Integrates acceleration to estimate velocity and position.
    Drift accumulates over time - needs periodic correction from
    external position sources (GPS, transponders, etc.)
    """

    def __init__(self, initial_position: np.ndarray, initial_velocity: np.ndarray):
        """
        Initialize dead reckoning.

        Args:
            initial_position: Starting position [x, y, z]
            initial_velocity: Starting velocity [vx, vy, vz]
        """
        self.position = np.array(initial_position, dtype=float)
        self.velocity = np.array(initial_velocity, dtype=float)
        self.last_update_time = 0.0

        # Track drift error (for analysis)
        self.drift_error = 0.0
        self.time_since_correction = 0.0

    def update(self, imu_reading: IMUReading, dt: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        Update position estimate using IMU data.

        Args:
            imu_reading: IMU sensor reading
            dt: Time step in seconds

        Returns:
            (estimated_position, estimated_velocity) tuple
        """
        # Simple integration: v = v + a*dt, p = p + v*dt
        # Note: This assumes acceleration is in world frame (not body frame)
        # Real implementation would need attitude estimation and frame transformation

        self.velocity += imu_reading.acceleration * dt
        self.position += self.velocity * dt

        # Track drift (grows with sqrt(time))
        self.time_since_correction += dt
        # Drift error grows as sqrt(t) for random walk
        self.drift_error = 0.5 * self.time_since_correction ** 1.5  # meters

        return self.position.copy(), self.velocity.copy()

    def correct(
        self,
        corrected_position: np.ndarray,
        corrected_velocity: np.ndarray = None
    ):
        """
        Correct dead reckoning with external position fix.

        This resets accumulated drift errors.

        Args:
            corrected_position: Accurate position from external source
            corrected_velocity: Optional accurate velocity
        """
        self.position = np.array(corrected_position, dtype=float)

        if corrected_velocity is not None:
            self.velocity = np.array(corrected_velocity, dtype=float)

        # Reset drift tracking
        self.drift_error = 0.0
        self.time_since_correction = 0.0

    def get_drift_error(self) -> float:
        """Get estimated drift error in meters."""
        return self.drift_error


class IMUAssistedNavigator:
    """
    Combines IMU dead reckoning with periodic transponder corrections.

    High-frequency IMU updates provide smooth position estimates.
    Low-frequency transponder fixes correct accumulated drift.
    """

    def __init__(
        self,
        imu_sensor: IMUSensor,
        initial_position: np.ndarray,
        initial_velocity: np.ndarray,
        correction_threshold: float = 2.0  # Correct when drift > 2m
    ):
        """
        Initialize IMU-assisted navigator.

        Args:
            imu_sensor: IMU sensor model
            initial_position: Starting position
            initial_velocity: Starting velocity
            correction_threshold: Drift threshold for correction in meters
        """
        self.imu = imu_sensor
        self.dead_reckoning = DeadReckoning(initial_position, initial_velocity)
        self.correction_threshold = correction_threshold

        # Track true vs estimated for analysis
        self.position_estimate = initial_position.copy()
        self.velocity_estimate = initial_velocity.copy()

    def update_imu(
        self,
        true_acceleration: np.ndarray,
        true_angular_velocity: np.ndarray,
        timestamp: float,
        dt: float
    ) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        Update position using IMU reading.

        Args:
            true_acceleration: True acceleration (for simulation)
            true_angular_velocity: True angular velocity (for simulation)
            timestamp: Current time
            dt: Time step

        Returns:
            (position, velocity, drift_error) tuple
        """
        # Get noisy IMU reading
        imu_reading = self.imu.read(
            true_acceleration,
            true_angular_velocity,
            timestamp
        )

        # Update dead reckoning
        self.position_estimate, self.velocity_estimate = \
            self.dead_reckoning.update(imu_reading, dt)

        drift = self.dead_reckoning.get_drift_error()

        return self.position_estimate, self.velocity_estimate, drift

    def correct_with_transponder(
        self,
        transponder_position: np.ndarray,
        transponder_velocity: np.ndarray = None
    ):
        """
        Correct IMU drift using transponder position fix.

        Args:
            transponder_position: Position from transponder navigation
            transponder_velocity: Optional velocity estimate
        """
        self.dead_reckoning.correct(transponder_position, transponder_velocity)
        self.position_estimate = transponder_position.copy()

        if transponder_velocity is not None:
            self.velocity_estimate = transponder_velocity.copy()

    def should_correct(self) -> bool:
        """Check if drift has exceeded threshold and correction is needed."""
        return self.dead_reckoning.get_drift_error() > self.correction_threshold

    def get_position(self) -> np.ndarray:
        """Get current position estimate."""
        return self.position_estimate.copy()

    def get_velocity(self) -> np.ndarray:
        """Get current velocity estimate."""
        return self.velocity_estimate.copy()

    def get_drift(self) -> float:
        """Get current drift error estimate."""
        return self.dead_reckoning.get_drift_error()
