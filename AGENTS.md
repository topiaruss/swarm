# Agent Architecture

**Multi-agent system design for autonomous drone swarm defense**

## Overview

This simulation implements an **entity-based agent architecture** where each drone, transponder, and future entity operates as an autonomous agent within a shared 3D arena. Agents perceive their environment through sensors, communicate via mesh networks, and execute behaviors to accomplish team objectives.

### Design Philosophy

- **Modularity**: Agents are composable entities with pluggable behaviors
- **Extensibility**: Easy to add new agent types, sensors, and strategies
- **Testability**: Simulation-first enables comprehensive testing before hardware deployment
- **Reality-aligned**: Agent models map directly to real hardware capabilities

---

## Agent Types

### 1. Drone Agents

**Primary autonomous agents with flight capabilities.**

**Properties:**
- `position`: 3D coordinates [x, y, z] in meters
- `velocity`: 3D velocity vector [vx, vy, vz] in m/s
- `team`: FRIENDLY, ENEMY, or NEUTRAL
- `role`: PATROL, ATTACK, SUPPORT, or INTRUDER
- `max_speed`: Speed limit (default 20 m/s ≈ 45 mph)
- `detection_range`: Sensor radius (default 200m)
- `battery_capacity`: Energy capacity in mAh
- `battery_level`: Current energy remaining

**Capabilities:**
- Kinematic motion with boundary enforcement (src/physics.py:20)
- Range-based detection of other entities (src/detection.py:15)
- Friend/foe identification (src/entities.py:101)
- Mesh network communication (src/mesh.py:50)
- GPS-denied navigation via transponders (src/navigation.py:40)

**Defined in:** `src/entities.py:70`

### 2. Transponder Agents

**Stationary navigation beacons for GPS-denied positioning.**

**Properties:**
- `position`: Fixed ground location
- `transmit_power`: RSSI at 1m reference (default -40 dBm)
- `relocate_interval`: Time between relocations (default 10 min)
- `encrypted`: Whether signal is encrypted (default True)

**Capabilities:**
- Broadcast position signals via RSSI
- Periodic relocation to avoid targeting
- Differential positioning support (src/navigation.py:120)

**Defined in:** `src/entities.py:112`

### 3. Future Agent Types

**Planned extensions:**
- **Charging Stations**: Ground-based energy replenishment
- **Ground Vehicles**: Mobile transponders or sensor platforms
- **Interceptors**: High-speed kinetic drones
- **Decoys**: Fake targets for distraction

---

## Agent Perception

### Detection System

Agents perceive their environment through a **range-based detection system** with extensible sensor architecture.

**Current Implementation:**
```python
# src/detection.py:15
class Detector:
    def detect(self, observer, entities):
        """Returns entities within detection range with distance/bearing."""
```

**Detection Data:**
- List of detected entities
- Distance to each entity (Euclidean)
- Bearing vector (unit direction)
- Friend/foe classification

**Example:**
```python
detector = Detector(max_range=200)  # 200m radius
contacts = detector.detect(my_drone, all_entities)

for entity, distance, bearing in contacts:
    if my_drone.is_foe(entity):
        print(f"Threat detected at {distance:.1f}m")
```

### Sensor Extensions

**Planned sensor types:**

**Camera Detector** (cone-based, line-of-sight):
```python
class CameraDetector(Detector):
    def __init__(self, max_range=300, fov_degrees=90):
        super().__init__(max_range)
        self.fov = np.radians(fov_degrees)

    def detect(self, observer, entities):
        # Only detect within cone + LOS check
        pass
```

**Radar Detector** (360°, long-range):
```python
class RadarDetector(Detector):
    def __init__(self, max_range=1000, resolution=5.0):
        super().__init__(max_range)
        self.resolution = resolution  # meters

    def detect(self, observer, entities):
        # Long range, omnidirectional, lower resolution
        pass
```

**Audio Detector** (short-range, 360°):
```python
class AudioDetector(Detector):
    def __init__(self, max_range=50):
        super().__init__(max_range)

    def detect(self, observer, entities):
        # Short range, omnidirectional sound detection
        pass
```

---

## Agent Communication

Agents communicate via **mesh networking** with multiple radio technologies simulated.

### Mesh Network Architecture

**Implemented in:** `src/mesh.py`

**Supported Radio Types:**
- **ESP-NOW**: 250 Kbps, 300m range, 10ms latency
- **LoRa**: 50 Kbps, 2000m range, 100ms latency
- **nRF24L01**: 1 Mbps, 100m range, 5ms latency

**Features:**
- Multi-hop routing with path discovery
- Network healing on topology changes
- Message priority queuing
- Packet loss simulation based on distance

**Example:**
```python
# Create mesh network
network = MeshNetwork(radio_type=RadioType.ESP_NOW)
network.add_node(drone1)
network.add_node(drone2)

# Send detection alert
network.send_message(
    sender=drone1,
    recipient=drone2,
    message={"type": "threat_detected", "position": [100, 200, 50]}
)
```

**Network Topology:**
- Dynamic routing adapts to drone movement
- Multi-hop extends effective range beyond radio limits
- Encrypted channels for friendly-only communication

---

## Agent Behaviors

### Current Behavior System

Agents currently use **simple kinematic motion** with manual velocity control.

**Example - Circular Patrol:**
```python
# examples/basic_scenario.py:25
def circular_patrol(drone, center, radius, angular_speed, dt):
    """Simple circular patrol behavior."""
    angle = angular_speed * current_time
    target_x = center[0] + radius * np.cos(angle)
    target_y = center[1] + radius * np.sin(angle)

    velocity = calculate_velocity_to_target(drone.position, [target_x, target_y])
    drone.set_velocity(velocity)
```

### Strategy Pattern for AI

**Recommended architecture for complex behaviors:**

```python
class Strategy:
    """Base class for agent strategies."""

    def update(self, agent, perception, dt):
        """
        Update agent behavior based on perception.

        Args:
            agent: The drone/entity to control
            perception: Dict with detected_entities, network_messages, etc.
            dt: Time step in seconds
        """
        raise NotImplementedError


class PatrolStrategy(Strategy):
    def __init__(self, waypoints):
        self.waypoints = waypoints
        self.current_waypoint = 0

    def update(self, agent, perception, dt):
        target = self.waypoints[self.current_waypoint]

        if np.linalg.norm(agent.position - target) < 10:  # 10m threshold
            self.current_waypoint = (self.current_waypoint + 1) % len(self.waypoints)

        velocity = calculate_velocity_to_target(agent.position, target)
        agent.set_velocity(velocity)


class InterceptStrategy(Strategy):
    def update(self, agent, perception, dt):
        threats = [e for e in perception['detected_entities'] if agent.is_foe(e)]

        if threats:
            # Target closest threat
            target = min(threats, key=lambda e: agent.distance_to(e))

            # Calculate intercept course
            intercept_point = self._calculate_intercept(agent, target)
            velocity = calculate_velocity_to_target(agent.position, intercept_point)
            agent.set_velocity(velocity)

    def _calculate_intercept(self, agent, target):
        # Lead target based on velocity
        time_to_intercept = agent.distance_to(target) / agent.max_speed
        return target.position + target.velocity * time_to_intercept
```

**Usage:**
```python
# Assign strategies to drones
patrol_drone.strategy = PatrolStrategy(waypoints=[...])
attack_drone.strategy = InterceptStrategy()

# Simulation loop
for drone in drones:
    perception = {
        'detected_entities': detector.detect(drone, all_entities),
        'network_messages': network.get_messages(drone)
    }
    drone.strategy.update(drone, perception, dt)
```

---

## Creating Custom Agents

### Extending Entity Classes

**Example - Charging Station:**
```python
from src.entities import Entity, Team, Role
import numpy as np

class ChargingStation(Entity):
    """Ground-based charging station for drones."""

    def __init__(self, position, charge_rate=100):
        super().__init__(
            position=position,
            velocity=np.array([0.0, 0.0, 0.0]),  # Stationary
            team=Team.FRIENDLY,
            role=Role.SUPPORT
        )
        self.charge_rate = charge_rate  # mAh/second
        self.occupied = False
        self.queue = []

    def charge_drone(self, drone, dt):
        """Charge a docked drone."""
        if drone.battery_level < drone.battery_capacity:
            charge_amount = self.charge_rate * dt
            drone.battery_level = min(
                drone.battery_level + charge_amount,
                drone.battery_capacity
            )
            return True
        return False  # Fully charged

    def is_available(self):
        """Check if station is available."""
        return not self.occupied and len(self.queue) == 0
```

**Example - Kamikaze Interceptor:**
```python
from src.entities import Drone, Team, Role
import numpy as np

class KamikazeDrone(Drone):
    """High-speed interceptor that destroys on contact."""

    def __init__(self, position, team=Team.FRIENDLY):
        super().__init__(
            position=position,
            team=team,
            role=Role.ATTACK,
            max_speed=40.0,  # 2x normal speed
            detection_range=300.0
        )
        self.armed = True
        self.detonation_range = 5.0  # meters

    def check_detonation(self, entities):
        """Check if close enough to target to detonate."""
        for entity in entities:
            if self.is_foe(entity):
                if self.distance_to(entity) <= self.detonation_range:
                    self.active = False
                    entity.active = False
                    return True
        return False
```

### Custom Strategy Example

**Stealth/Hiding Behavior:**
```python
class StealthStrategy(Strategy):
    """Hide from threats by moving to low altitude and reducing speed."""

    def __init__(self, hide_altitude=10.0, cruise_altitude=50.0):
        self.hide_altitude = hide_altitude
        self.cruise_altitude = cruise_altitude
        self.hiding = False

    def update(self, agent, perception, dt):
        threats = [e for e in perception['detected_entities'] if agent.is_foe(e)]

        if threats:
            # Threats detected - hide
            if not self.hiding:
                print(f"{agent.id} entering stealth mode")
                self.hiding = True

            # Descend to low altitude
            target_z = self.hide_altitude
            agent.set_velocity(np.array([0, 0, (target_z - agent.position[2]) * 0.5]))

        else:
            # No threats - resume normal patrol
            if self.hiding:
                print(f"{agent.id} exiting stealth mode")
                self.hiding = False

            # Return to cruise altitude
            target_z = self.cruise_altitude
            agent.set_velocity(np.array([0, 0, (target_z - agent.position[2]) * 0.5]))
```

---

## Multi-Agent Coordination

### Swarm Behaviors

**Formation Flying:**
```python
class FormationStrategy(Strategy):
    """Maintain position relative to leader."""

    def __init__(self, leader, offset):
        self.leader = leader
        self.offset = np.array(offset)  # Relative position [x, y, z]

    def update(self, agent, perception, dt):
        target_position = self.leader.position + self.offset
        error = target_position - agent.position

        # Proportional control
        velocity = error * 0.5  # Gain factor
        agent.set_velocity(velocity)


# Create formation
leader = drones[0]
leader.strategy = PatrolStrategy(waypoints)

wingman_left = drones[1]
wingman_left.strategy = FormationStrategy(leader, offset=[-20, 0, 0])

wingman_right = drones[2]
wingman_right.strategy = FormationStrategy(leader, offset=[20, 0, 0])
```

**Distributed Threat Response:**
```python
class CoordinatedInterceptStrategy(Strategy):
    """Coordinate with teammates to intercept threats efficiently."""

    def __init__(self, network):
        self.network = network
        self.assigned_target = None

    def update(self, agent, perception, dt):
        threats = [e for e in perception['detected_entities'] if agent.is_foe(e)]

        if threats:
            # Share threat detection with team
            for threat in threats:
                self.network.broadcast(
                    sender=agent,
                    message={
                        'type': 'threat_detected',
                        'threat_id': threat.id,
                        'position': threat.position.tolist(),
                        'velocity': threat.velocity.tolist()
                    }
                )

            # Process team messages to avoid duplicate targeting
            team_messages = [m for m in perception['network_messages']
                           if m['type'] == 'threat_detected']

            # Claim closest unassigned threat
            if self.assigned_target is None:
                self.assigned_target = self._claim_target(agent, threats, team_messages)

            # Intercept assigned target
            if self.assigned_target:
                velocity = calculate_intercept(agent, self.assigned_target)
                agent.set_velocity(velocity)

    def _claim_target(self, agent, threats, team_messages):
        # Logic to avoid multiple drones targeting same threat
        # Returns closest threat not claimed by closer teammate
        pass
```

### Handoff Protocols

**Battery Management with Charging Handoff:**
```python
class BatteryManagedStrategy(Strategy):
    """Return to charge when battery low, coordinate handoff."""

    def __init__(self, base_strategy, charging_station, low_battery_threshold=0.2):
        self.base_strategy = base_strategy
        self.charging_station = charging_station
        self.low_battery_threshold = low_battery_threshold
        self.returning = False

    def update(self, agent, perception, dt):
        battery_percent = agent.battery_level / agent.battery_capacity

        if battery_percent < self.low_battery_threshold and not self.returning:
            # Request handoff from teammate
            self._request_handoff(agent, perception)
            self.returning = True

        if self.returning:
            # Navigate to charging station
            if agent.distance_to(self.charging_station) < 5:
                # Dock and charge
                self.charging_station.charge_drone(agent, dt)

                if battery_percent > 0.9:  # 90% charged
                    self.returning = False
            else:
                velocity = calculate_velocity_to_target(
                    agent.position,
                    self.charging_station.position
                )
                agent.set_velocity(velocity)
        else:
            # Execute normal strategy
            self.base_strategy.update(agent, perception, dt)

    def _request_handoff(self, agent, perception):
        # Broadcast request for teammate to take over patrol
        pass
```

---

## State Machine Architecture

**Recommended for complex agent behaviors:**

```python
from enum import Enum

class DroneState(Enum):
    IDLE = "idle"
    PATROL = "patrol"
    INVESTIGATE = "investigate"
    TRACK = "track"
    INTERCEPT = "intercept"
    RTL = "rtl"  # Return to launch
    CHARGING = "charging"


class StateMachineStrategy(Strategy):
    """State machine for tactical drone behavior."""

    def __init__(self):
        self.state = DroneState.PATROL
        self.target = None
        self.patrol_waypoints = []
        self.home_position = None

    def update(self, agent, perception, dt):
        # State transition logic
        new_state = self._evaluate_transitions(agent, perception)
        if new_state != self.state:
            print(f"{agent.id}: {self.state.value} -> {new_state.value}")
            self.state = new_state

        # Execute state behavior
        if self.state == DroneState.PATROL:
            self._patrol(agent, dt)
        elif self.state == DroneState.INVESTIGATE:
            self._investigate(agent, perception, dt)
        elif self.state == DroneState.TRACK:
            self._track(agent, perception, dt)
        elif self.state == DroneState.INTERCEPT:
            self._intercept(agent, perception, dt)
        elif self.state == DroneState.RTL:
            self._return_to_launch(agent, dt)
        elif self.state == DroneState.CHARGING:
            self._charge(agent, dt)

    def _evaluate_transitions(self, agent, perception):
        """Determine state transitions based on perception."""
        threats = [e for e in perception['detected_entities'] if agent.is_foe(e)]
        battery_low = agent.battery_level < agent.battery_capacity * 0.2

        # Priority order: safety > mission > patrol
        if battery_low and self.state != DroneState.CHARGING:
            return DroneState.RTL

        if threats:
            closest_threat = min(threats, key=lambda e: agent.distance_to(e))
            distance = agent.distance_to(closest_threat)

            if distance < 50:  # Within intercept range
                self.target = closest_threat
                return DroneState.INTERCEPT
            elif distance < 100:  # Within tracking range
                self.target = closest_threat
                return DroneState.TRACK
            else:  # Detected but far
                self.target = closest_threat
                return DroneState.INVESTIGATE

        # No threats, continue patrol
        return DroneState.PATROL

    def _patrol(self, agent, dt):
        # Circular or waypoint patrol
        pass

    def _investigate(self, agent, perception, dt):
        # Move toward distant detection
        pass

    def _track(self, agent, perception, dt):
        # Follow target, maintain distance
        pass

    def _intercept(self, agent, perception, dt):
        # Aggressive pursuit
        pass

    def _return_to_launch(self, agent, dt):
        # Navigate to home/charging station
        pass

    def _charge(self, agent, dt):
        # Stationary charging
        pass
```

---

## Future Agent Capabilities

### Reinforcement Learning Agents

**Planned integration:**
```python
class RLAgent(Strategy):
    """Reinforcement learning agent using trained policy."""

    def __init__(self, policy_network):
        self.policy = policy_network
        self.observation_buffer = []

    def update(self, agent, perception, dt):
        # Construct observation vector
        obs = self._build_observation(agent, perception)

        # Get action from policy network
        action = self.policy.predict(obs)

        # Apply action (velocity commands)
        agent.set_velocity(action)

    def _build_observation(self, agent, perception):
        """Convert perception to observation vector for NN input."""
        # Example: [own_pos, own_vel, nearest_threat_pos, nearest_threat_vel, ...]
        obs = np.concatenate([
            agent.position,
            agent.velocity,
            # Add detected entities, battery state, etc.
        ])
        return obs
```

**Training environment:**
- Simulation provides perfect environment for RL training
- Reward shaping for desired behaviors (detect threats, avoid crashes, conserve battery)
- Policy transfer to real hardware via domain randomization

### Swarm Intelligence

**Planned behaviors:**
- **Flocking**: Cohesion, alignment, separation rules
- **Consensus**: Distributed agreement on threat priority
- **Task Allocation**: Auction-based job assignment
- **Emergent Behaviors**: Complex patterns from simple rules

**Example - Boid Flocking:**
```python
class FlockingStrategy(Strategy):
    """Emergent swarm behavior using boid rules."""

    def __init__(self, cohesion_weight=1.0, alignment_weight=1.0, separation_weight=1.5):
        self.cohesion_weight = cohesion_weight
        self.alignment_weight = alignment_weight
        self.separation_weight = separation_weight

    def update(self, agent, perception, dt):
        teammates = [e for e in perception['detected_entities']
                    if agent.is_friend(e) and isinstance(e, Drone)]

        if not teammates:
            return

        # Cohesion: steer toward average position
        cohesion = self._cohesion(agent, teammates)

        # Alignment: match average velocity
        alignment = self._alignment(agent, teammates)

        # Separation: avoid crowding
        separation = self._separation(agent, teammates)

        # Combine behaviors
        velocity = (
            cohesion * self.cohesion_weight +
            alignment * self.alignment_weight +
            separation * self.separation_weight
        )

        agent.set_velocity(velocity)

    def _cohesion(self, agent, teammates):
        center = np.mean([t.position for t in teammates], axis=0)
        return (center - agent.position) * 0.01

    def _alignment(self, agent, teammates):
        avg_velocity = np.mean([t.velocity for t in teammates], axis=0)
        return (avg_velocity - agent.velocity) * 0.1

    def _separation(self, agent, teammates):
        separate = np.array([0.0, 0.0, 0.0])
        for t in teammates:
            diff = agent.position - t.position
            dist = np.linalg.norm(diff)
            if dist < 20:  # Separation threshold
                separate += diff / (dist ** 2)
        return separate
```

---

## Testing Agents

### Unit Testing

**Test agent behaviors in isolation:**
```python
# tests/test_strategies.py
import numpy as np
from src.entities import Drone, Team, Role
from strategies import InterceptStrategy

def test_intercept_strategy():
    # Create test agents
    friendly = Drone(
        position=np.array([0, 0, 50]),
        velocity=np.array([0, 0, 0]),
        team=Team.FRIENDLY,
        role=Role.ATTACK
    )

    enemy = Drone(
        position=np.array([100, 0, 50]),
        velocity=np.array([10, 0, 0]),
        team=Team.ENEMY,
        role=Role.INTRUDER
    )

    # Apply strategy
    strategy = InterceptStrategy()
    perception = {'detected_entities': [enemy]}

    strategy.update(friendly, perception, dt=0.1)

    # Verify drone moves toward intercept point
    assert np.linalg.norm(friendly.velocity) > 0
    # Should lead target, not just chase
    assert friendly.velocity[0] > 0  # Moving in x direction
```

### Scenario Testing

**Test multi-agent interactions:**
```python
# examples/test_coordination.py
from src.arena import Arena
from src.entities import Drone, Team, Role
from strategies import CoordinatedInterceptStrategy
import numpy as np

def test_coordinated_response():
    arena = Arena(bounds=(1000, 1000, 500))

    # Create friendly patrol
    patrol = [
        Drone(position=np.array([100, 100, 50]), team=Team.FRIENDLY, role=Role.PATROL),
        Drone(position=np.array([200, 200, 50]), team=Team.FRIENDLY, role=Role.PATROL),
        Drone(position=np.array([300, 300, 50]), team=Team.FRIENDLY, role=Role.PATROL),
    ]

    # Create intruders
    intruders = [
        Drone(position=np.array([500, 500, 50]), team=Team.ENEMY, role=Role.INTRUDER),
        Drone(position=np.array([600, 600, 50]), team=Team.ENEMY, role=Role.INTRUDER),
    ]

    # Run simulation
    for t in range(1000):  # 100 seconds at 0.1s timestep
        # Update strategies
        # Verify coordination (no duplicate targeting)
        pass
```

### Hardware Validation

**Bridge simulation to real drones:**
```python
class HardwareDrone(Drone):
    """Drone controlled by real hardware (Tello, Crazyflie, etc.)."""

    def __init__(self, hw_connection, **kwargs):
        super().__init__(**kwargs)
        self.hw = hw_connection

    def update_from_hardware(self):
        """Sync simulation state with real drone telemetry."""
        self.position = self.hw.get_position()
        self.velocity = self.hw.get_velocity()
        self.battery_level = self.hw.get_battery()

    def apply_to_hardware(self):
        """Send velocity commands to real drone."""
        self.hw.set_velocity(self.velocity)

# Test strategy on real hardware
real_drone = HardwareDrone(tello_connection)
real_drone.strategy = PatrolStrategy(waypoints)

while True:
    real_drone.update_from_hardware()
    perception = get_perception(real_drone)  # From real sensors
    real_drone.strategy.update(real_drone, perception, dt=0.1)
    real_drone.apply_to_hardware()
```

---

## Agent Development Workflow

### 1. Design in Simulation
- Prototype behavior in pure Python
- Test with simple scenarios
- Validate edge cases

### 2. Unit Test
- Test individual strategies
- Verify state transitions
- Check coordination logic

### 3. Integration Test
- Multi-agent scenarios
- Network communication
- Resource management

### 4. Hardware Validation
- Start with desk test (ESP32 modules)
- Graduate to safe platforms (Tello EDU)
- Deploy to custom builds

### 5. Iterate
- Refine based on real-world performance
- Update simulation models from hardware data
- Close the simulation-reality gap

---

## Best Practices

### Agent Design

✅ **Keep strategies stateless when possible**
- Easier to test and debug
- Better composability

✅ **Use dependency injection**
- Pass network, detector as parameters
- Enables mocking for tests

✅ **Fail gracefully**
- Handle missing detections
- Degrade performance, don't crash

✅ **Log decisions**
- Record why agent chose action
- Essential for debugging swarm behaviors

### Performance

✅ **Cache expensive calculations**
- Distance, bearing computations
- Pre-compute waypoint paths

✅ **Limit perception range**
- Don't process all entities
- Use spatial partitioning for large swarms

✅ **Vectorize when possible**
- NumPy operations for multiple agents
- Significant speedup for 50+ drones

### Extensibility

✅ **Use composition over inheritance**
- Strategy pattern > deep class hierarchies
- Mix and match behaviors

✅ **Design for modularity**
- Sensors, communication, physics separate
- Easy to swap implementations

✅ **Document assumptions**
- What coordinate system?
- What units (m/s vs km/h)?
- What range limits?

---

## Resources

**Code Examples:**
- `examples/basic_scenario.py` - Simple patrol and detection
- `examples/mesh_comms.py` - Network communication patterns
- `examples/gps_denied_navigation.py` - Transponder-based positioning

**Core Modules:**
- `src/entities.py` - Agent base classes
- `src/detection.py` - Perception system
- `src/mesh.py` - Communication layer
- `src/physics.py` - Motion and dynamics

**External References:**
- [Boids Algorithm](https://en.wikipedia.org/wiki/Boids) - Flocking behaviors
- [Multi-Agent RL](https://arxiv.org/abs/1911.10635) - Training strategies
- [Swarm Robotics](https://mitpress.mit.edu/books/swarm-robotics) - Coordination theory

---

## Roadmap

**Current (Phase 0):**
- ✅ Entity-based agent architecture
- ✅ Range-based detection
- ✅ Mesh networking
- ✅ GPS-denied navigation

**Next (Phase 1):**
- [ ] Strategy pattern implementation
- [ ] State machine behaviors
- [ ] Fault tolerance (agent failures)
- [ ] Coordinated threat response

**Future (Phase 2+):**
- [ ] Reinforcement learning integration
- [ ] Swarm intelligence algorithms
- [ ] Multi-modal sensor fusion
- [ ] Hardware abstraction layer
- [ ] ROS 2 migration path

---

## Contributing

Agent development areas needing contribution:

1. **Strategy Implementations**: More behavioral patterns (search, escort, etc.)
2. **Coordination Algorithms**: Improved multi-agent task allocation
3. **Learning Systems**: RL training pipelines, reward shaping
4. **Hardware Bridges**: Real drone integration (Tello, Crazyflie, PX4)
5. **Testing Utilities**: Scenario generators, benchmark suites

See `PLAN.md` for detailed extension roadmap.

---

## Questions?

- **Architecture**: See this document (AGENTS.md)
- **Installation**: See README.md
- **Development Plan**: See PLAN.md
- **Hardware**: See NEXT.md
