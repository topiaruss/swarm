"""
Metrics collection for swarm operations.

Tracks charging cycles, battery statistics, and operational efficiency.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional
import numpy as np


@dataclass
class ChargingCycleMetrics:
    """Metrics for a single charging cycle."""
    drone_id: str
    start_time: float
    end_time: float
    start_battery: float  # Percentage 0.0-1.0
    end_battery: float  # Percentage 0.0-1.0
    wait_time: float  # Time spent waiting for slot
    charge_time: float  # Time spent actually charging
    slot_id: int


@dataclass
class FleetMetrics:
    """Aggregate metrics for the entire fleet."""

    # Charging cycles
    charging_cycles: List[ChargingCycleMetrics] = field(default_factory=list)

    # Battery statistics (time-series)
    battery_history: Dict[str, List[tuple]] = field(default_factory=dict)  # drone_id -> [(time, battery), ...]

    # Operational coverage
    coverage_gaps: List[tuple] = field(default_factory=list)  # [(start_time, end_time, sector_id), ...]

    # Emergency events
    critical_battery_events: List[tuple] = field(default_factory=list)  # [(time, drone_id, battery), ...]

    def record_charging_cycle(self, cycle: ChargingCycleMetrics):
        """Record a completed charging cycle."""
        self.charging_cycles.append(cycle)

    def record_battery_level(self, drone_id: str, time: float, battery_level: float):
        """Record battery level at a point in time."""
        if drone_id not in self.battery_history:
            self.battery_history[drone_id] = []
        self.battery_history[drone_id].append((time, battery_level))

    def record_critical_battery(self, time: float, drone_id: str, battery_level: float):
        """Record critical battery event."""
        self.critical_battery_events.append((time, drone_id, battery_level))

    def record_coverage_gap(self, start_time: float, end_time: float, sector_id: str):
        """Record period when a sector had no coverage."""
        self.coverage_gaps.append((start_time, end_time, sector_id))

    def get_charging_statistics(self) -> Dict:
        """Calculate charging statistics."""
        if not self.charging_cycles:
            return {
                'total_cycles': 0,
                'avg_charge_time': 0.0,
                'avg_wait_time': 0.0,
                'avg_battery_gain': 0.0
            }

        total_cycles = len(self.charging_cycles)
        avg_charge_time = np.mean([c.charge_time for c in self.charging_cycles])
        avg_wait_time = np.mean([c.wait_time for c in self.charging_cycles])
        avg_battery_gain = np.mean([c.end_battery - c.start_battery for c in self.charging_cycles])

        return {
            'total_cycles': total_cycles,
            'avg_charge_time': avg_charge_time,
            'avg_wait_time': avg_wait_time,
            'avg_battery_gain': avg_battery_gain * 100  # Convert to percentage
        }

    def get_battery_statistics(self) -> Dict:
        """Calculate battery statistics across fleet."""
        if not self.battery_history:
            return {
                'min_battery_ever': 1.0,
                'avg_battery': 1.0,
                'battery_variance': 0.0,
                'critical_events': 0
            }

        # Get all battery readings
        all_readings = []
        for readings in self.battery_history.values():
            all_readings.extend([r[1] for r in readings])

        if not all_readings:
            return {
                'min_battery_ever': 1.0,
                'avg_battery': 1.0,
                'battery_variance': 0.0,
                'critical_events': len(self.critical_battery_events)
            }

        return {
            'min_battery_ever': min(all_readings),
            'avg_battery': np.mean(all_readings),
            'battery_variance': np.var(all_readings),
            'critical_events': len(self.critical_battery_events)
        }

    def get_operational_statistics(self) -> Dict:
        """Calculate operational efficiency statistics."""
        return {
            'coverage_gaps': len(self.coverage_gaps),
            'total_gap_time': sum(end - start for start, end, _ in self.coverage_gaps),
            'critical_battery_events': len(self.critical_battery_events)
        }

    def get_fleet_health_score(self) -> float:
        """
        Calculate overall fleet health score (0.0 to 1.0).

        Considers:
        - Battery levels (higher is better)
        - Coverage gaps (fewer is better)
        - Critical events (fewer is better)

        Returns:
            Score from 0.0 (poor) to 1.0 (excellent)
        """
        battery_stats = self.get_battery_statistics()
        ops_stats = self.get_operational_statistics()

        # Battery component (40% weight)
        battery_score = battery_stats['avg_battery'] * 0.4

        # Coverage component (30% weight)
        # Penalize for gaps (assume 100s simulation, 1 gap = -0.1)
        gap_penalty = min(0.3, ops_stats['coverage_gaps'] * 0.1)
        coverage_score = max(0.0, 0.3 - gap_penalty)

        # Reliability component (30% weight)
        # Penalize for critical events
        critical_penalty = min(0.3, ops_stats['critical_battery_events'] * 0.1)
        reliability_score = max(0.0, 0.3 - critical_penalty)

        total_score = battery_score + coverage_score + reliability_score

        return total_score

    def print_summary(self):
        """Print human-readable metrics summary."""
        print("\n" + "=" * 70)
        print("Fleet Metrics Summary")
        print("=" * 70)

        charging_stats = self.get_charging_statistics()
        print("\nCharging Statistics:")
        print(f"  Total cycles: {charging_stats['total_cycles']}")
        print(f"  Avg charge time: {charging_stats['avg_charge_time']:.1f}s")
        print(f"  Avg wait time: {charging_stats['avg_wait_time']:.1f}s")
        print(f"  Avg battery gain: {charging_stats['avg_battery_gain']:.1f}%")

        battery_stats = self.get_battery_statistics()
        print("\nBattery Statistics:")
        print(f"  Minimum battery (fleet): {battery_stats['min_battery_ever']*100:.1f}%")
        print(f"  Average battery: {battery_stats['avg_battery']*100:.1f}%")
        print(f"  Battery variance: {battery_stats['battery_variance']:.3f}")
        print(f"  Critical events: {battery_stats['critical_events']}")

        ops_stats = self.get_operational_statistics()
        print("\nOperational Statistics:")
        print(f"  Coverage gaps: {ops_stats['coverage_gaps']}")
        print(f"  Total gap time: {ops_stats['total_gap_time']:.1f}s")

        health_score = self.get_fleet_health_score()
        print(f"\nFleet Health Score: {health_score:.2f} / 1.00")

        if health_score >= 0.8:
            print("  Status: EXCELLENT ✓")
        elif health_score >= 0.6:
            print("  Status: GOOD")
        elif health_score >= 0.4:
            print("  Status: FAIR")
        else:
            print("  Status: POOR ⚠")

        print("=" * 70)
