"""
GPS-denied navigation using transponders.

Implements RSSI-based trilateration and bearing-based triangulation
with sensor fusion for robust positioning.
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Optional, Tuple
from enum import Enum


class SignalType(Enum):
    """Type of positioning signal."""
    RSSI = "rssi"  # Received Signal Strength Indicator
    BEARING = "bearing"  # Direction finding
    BOTH = "both"  # Sensor fusion


@dataclass
class TransponderSignal:
    """
    Signal received from a transponder.

    Attributes:
        transponder_id: ID of transmitting transponder
        transponder_position: Known position of transponder
        rssi: Received signal strength in dBm (more negative = weaker)
        bearing: Bearing from receiver to transponder in radians
        distance_estimate: Estimated distance based on RSSI
        signal_quality: Quality metric 0-1 (1 = perfect)
    """
    transponder_id: str
    transponder_position: np.ndarray
    rssi: Optional[float] = None
    bearing: Optional[float] = None
    distance_estimate: Optional[float] = None
    signal_quality: float = 1.0


class TransponderNavigator:
    """
    GPS-denied navigation using transponder signals.

    Supports:
    - RSSI-based trilateration (signal strength)
    - Bearing-based triangulation (direction finding)
    - Sensor fusion (combines both methods)
    """

    def __init__(
        self,
        method: SignalType = SignalType.BOTH,
        rssi_model_params: Optional[dict] = None
    ):
        """
        Initialize navigator.

        Args:
            method: Positioning method to use
            rssi_model_params: Parameters for RSSI distance model
        """
        self.method = method

        # Default RSSI path loss model: RSSI = A - 10*n*log10(d)
        # A = RSSI at 1m, n = path loss exponent
        self.rssi_params = rssi_model_params or {
            'A': -40,  # RSSI at 1 meter (dBm)
            'n': 2.5,  # Path loss exponent (2=free space, 3-4=urban)
            'noise_std': 3.0  # Signal noise standard deviation (dBm)
        }

    def rssi_to_distance(self, rssi: float) -> float:
        """
        Convert RSSI to distance estimate using path loss model.

        Args:
            rssi: Received signal strength in dBm

        Returns:
            Estimated distance in meters
        """
        A = self.rssi_params['A']
        n = self.rssi_params['n']

        # Solve: RSSI = A - 10*n*log10(d) for d
        # d = 10^((A - RSSI) / (10*n))
        distance = 10 ** ((A - rssi) / (10 * n))
        return max(distance, 0.1)  # Minimum 0.1m

    def distance_to_rssi(self, distance: float, add_noise: bool = False) -> float:
        """
        Convert distance to RSSI (for simulation).

        Args:
            distance: Distance in meters
            add_noise: Whether to add realistic noise

        Returns:
            RSSI in dBm
        """
        A = self.rssi_params['A']
        n = self.rssi_params['n']

        rssi = A - 10 * n * np.log10(max(distance, 0.1))

        if add_noise:
            noise = np.random.normal(0, self.rssi_params['noise_std'])
            rssi += noise

        return rssi

    def calculate_bearing(
        self,
        from_position: np.ndarray,
        to_position: np.ndarray,
        add_noise: bool = False
    ) -> float:
        """
        Calculate bearing from one position to another.

        Args:
            from_position: Observer position [x, y, z]
            to_position: Target position [x, y, z]
            add_noise: Whether to add realistic noise

        Returns:
            Bearing in radians (0 = North/+Y, increases clockwise)
        """
        diff = to_position - from_position
        # Use atan2 with Y (North) and X (East) to get bearing
        bearing = np.arctan2(diff[0], diff[1])  # Note: X, Y order for clockwise from North

        if add_noise:
            # Bearing accuracy typically ±5 degrees
            noise = np.radians(np.random.normal(0, 5))
            bearing += noise

        return bearing

    def trilaterate_rssi(self, signals: List[TransponderSignal]) -> Tuple[np.ndarray, float]:
        """
        Calculate position using RSSI trilateration.

        Requires at least 3 signals. Uses least-squares optimization.

        Args:
            signals: List of transponder signals with RSSI

        Returns:
            (estimated_position, uncertainty) tuple
        """
        if len(signals) < 3:
            raise ValueError("RSSI trilateration requires at least 3 signals")

        # Convert RSSI to distance estimates
        positions = []
        distances = []
        qualities = []

        for sig in signals:
            if sig.rssi is None:
                continue
            positions.append(sig.transponder_position[:2])  # Use XY only
            distances.append(self.rssi_to_distance(sig.rssi))
            qualities.append(sig.signal_quality)

        if len(positions) < 3:
            raise ValueError("Not enough RSSI signals")

        positions = np.array(positions)
        distances = np.array(distances)
        qualities = np.array(qualities)

        # Weighted least squares trilateration
        # Solve for position that minimizes error to all circles
        def error_function(pos):
            errors = np.abs(np.linalg.norm(positions - pos, axis=1) - distances)
            return np.sum(qualities * errors**2)

        # Initial guess: centroid of transponders
        initial_guess = np.mean(positions, axis=0)

        # Simple gradient descent (could use scipy.optimize for production)
        position = initial_guess.copy()
        learning_rate = 0.1

        for _ in range(100):
            grad = np.zeros(2)
            for i in range(len(positions)):
                diff = position - positions[i]
                dist = np.linalg.norm(diff)
                if dist > 0:
                    grad += 2 * qualities[i] * (dist - distances[i]) * diff / dist

            position -= learning_rate * grad

            if np.linalg.norm(grad) < 0.01:
                break

        # Estimate uncertainty based on signal quality
        avg_quality = np.mean(qualities)
        uncertainty = 10.0 * (1.0 - avg_quality)  # 0-10m uncertainty

        # Return 3D position (keep original Z)
        z = np.mean([sig.transponder_position[2] for sig in signals])
        position_3d = np.array([position[0], position[1], z])

        return position_3d, uncertainty

    def triangulate_bearing(self, signals: List[TransponderSignal]) -> Tuple[np.ndarray, float]:
        """
        Calculate position using bearing triangulation.

        Requires at least 2 signals with known bearings.

        Args:
            signals: List of transponder signals with bearings

        Returns:
            (estimated_position, uncertainty) tuple
        """
        if len(signals) < 2:
            raise ValueError("Bearing triangulation requires at least 2 signals")

        # Extract signals with bearing data
        valid_signals = [s for s in signals if s.bearing is not None]

        if len(valid_signals) < 2:
            raise ValueError("Not enough bearing signals")

        # For each pair of bearings, find intersection point
        # This is a simplified 2D solution
        positions = []

        for i in range(len(valid_signals) - 1):
            sig1 = valid_signals[i]
            sig2 = valid_signals[i + 1]

            # Transponder positions (2D)
            p1 = sig1.transponder_position[:2]
            p2 = sig2.transponder_position[:2]

            # Bearing vectors (direction from transponder to receiver)
            # Note: bearing is FROM receiver TO transponder, so reverse it
            bearing1 = sig1.bearing + np.pi  # Reverse direction
            bearing2 = sig2.bearing + np.pi

            # Direction vectors
            d1 = np.array([np.sin(bearing1), np.cos(bearing1)])
            d2 = np.array([np.sin(bearing2), np.cos(bearing2)])

            # Line intersection: p1 + t1*d1 = p2 + t2*d2
            # Solve for t1
            det = d1[0] * d2[1] - d1[1] * d2[0]

            if abs(det) > 0.01:  # Lines not parallel
                dp = p2 - p1
                t1 = (dp[0] * d2[1] - dp[1] * d2[0]) / det
                intersection = p1 + t1 * d1
                positions.append(intersection)

        if len(positions) == 0:
            raise ValueError("Could not triangulate position (parallel bearings)")

        # Average all intersection points
        position_2d = np.mean(positions, axis=0)

        # Estimate uncertainty based on spread of intersection points
        if len(positions) > 1:
            uncertainty = np.std([np.linalg.norm(p - position_2d) for p in positions])
        else:
            uncertainty = 5.0  # Default 5m for 2-signal triangulation

        # Return 3D position
        z = np.mean([sig.transponder_position[2] for sig in signals])
        position_3d = np.array([position_2d[0], position_2d[1], z])

        return position_3d, uncertainty

    def calculate_position(
        self,
        signals: List[TransponderSignal],
        method: Optional[SignalType] = None
    ) -> Tuple[np.ndarray, float]:
        """
        Calculate position using specified method or sensor fusion.

        Args:
            signals: List of transponder signals
            method: Override default positioning method

        Returns:
            (estimated_position, uncertainty) tuple
        """
        method = method or self.method

        if method == SignalType.RSSI:
            return self.trilaterate_rssi(signals)

        elif method == SignalType.BEARING:
            return self.triangulate_bearing(signals)

        elif method == SignalType.BOTH:
            # Sensor fusion: combine both methods
            try:
                pos_rssi, unc_rssi = self.trilaterate_rssi(signals)
                weight_rssi = 1.0 / (unc_rssi + 0.1)
            except ValueError:
                pos_rssi = None
                weight_rssi = 0

            try:
                pos_bearing, unc_bearing = self.triangulate_bearing(signals)
                weight_bearing = 1.0 / (unc_bearing + 0.1)
            except ValueError:
                pos_bearing = None
                weight_bearing = 0

            if pos_rssi is None and pos_bearing is None:
                raise ValueError("Insufficient signals for positioning")

            if pos_rssi is None:
                return pos_bearing, unc_bearing

            if pos_bearing is None:
                return pos_rssi, unc_rssi

            # Weighted average
            total_weight = weight_rssi + weight_bearing
            fused_position = (
                weight_rssi * pos_rssi + weight_bearing * pos_bearing
            ) / total_weight

            # Combined uncertainty
            fused_uncertainty = np.sqrt(
                (weight_rssi * unc_rssi**2 + weight_bearing * unc_bearing**2) / total_weight
            )

            return fused_position, fused_uncertainty
