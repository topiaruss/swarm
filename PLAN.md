# Drone Swarm Defense Simulation - Build Plan

## Phase 0: Minimal Prototype (2-3 hours)

### Goal
Build a working 3D simulation sandbox to test counter-drone defense strategies with clean, extensible architecture.

### Core Components

**1. Arena**
- 3D bounded volume (e.g., 1km x 1km x 500m)
- Geographic reference point (lat/lon)
- Boundary enforcement

**2. Entity System**
- Base `Entity` class: position, velocity, team, role
- `Drone` subclass: simple flight physics
- Friend/foe designation
- Role types: PATROL, ATTACK, SUPPORT, INTRUDER

**3. Physics Engine**
- Simple kinematic model (position += velocity * dt)
- Boundary collision handling
- Configurable time step (0.1s default)

**4. Detection System**
- Range-based detection (e.g., 200m radius)
- Returns: detected entities, distance, bearing
- No sensor fusion yet (future extension)

**5. Visualization**
- Matplotlib 3D real-time plot
- Color coding: blue=friendly, red=enemy
- Detection range indicators
- Trajectory trails (optional)

**6. Simulation Loop**
- Fixed time step updates
- Event logging
- State snapshots for replay

### Tech Stack
- Python 3.10+
- NumPy (vectors, math)
- Matplotlib (3D visualization)
- dataclasses (clean entity definitions)

### File Structure
```
swarm/
├── PLAN.md (this file)
├── README.md (usage instructions)
├── src/
│   ├── __init__.py
│   ├── arena.py          # Arena class, boundaries
│   ├── entities.py       # Entity, Drone classes
│   ├── physics.py        # Physics engine
│   ├── detection.py      # Sensor/detection systems
│   └── visualization.py  # 3D plotting
├── examples/
│   └── basic_scenario.py # Demo: patrol drones detect intruder
└── tests/
    └── test_arena.py     # Basic unit tests
```

### Example Scenario (what you'll see working)
- 3 friendly patrol drones circling an area
- 1 enemy drone enters from boundary
- Friendly drones detect intruder when in range
- Console logs detection events
- 3D visualization shows real-time movement

---

## Future Extensions (Phase 1+)

### Realistic Physics
- Thrust/drag model
- Wind simulation
- Collision detection between drones
- Terrain following

### Resource Management
- Battery capacity (mAh)
- Power consumption model
- Charging stations (ground entities)
- Low-battery RTL (return to launch)
- Handoff protocols

### Advanced Sensors
- `CameraDetector` (cone-based, requires line of sight)
- `RadarDetector` (longer range, lower resolution)
- `AudioDetector` (short range, 360°)
- Sensor fusion (combine multiple inputs)
- False positive/negative rates

### GPS-Denied Navigation
- Transponder entities (broadcast position)
- Differential positioning
- Moving transponder support
- Position uncertainty modeling

### AI/Strategy Layer
- State machines (OBSERVE → TRACK → INTERCEPT)
- Reinforcement learning agents
- Swarm coordination algorithms
- Threat assessment (trajectory prediction)
- Multi-wave attack detection

### Networking
- Communication range limits
- Message latency/dropout
- Mesh networking simulation
- Ground station support

### Advanced Visualization
- Web dashboard (Flask + Plotly)
- Multi-camera views
- Sensor cone overlays
- Heat maps (threat zones)
- Time-series plots (battery, detections)

### Scenarios
- Capture the flag
- Defend critical infrastructure
- Search and track
- Swarm vs swarm
- Coordinated multi-wave attacks

---

## Extension Architecture

The minimal prototype is designed for easy extension:

**Adding new entity types:**
```python
class ChargingStation(Entity):
    def __init__(self, position):
        super().__init__(position, team=Team.FRIENDLY, role=Role.SUPPORT)
        self.charge_rate = 100  # mAh per second
```

**Adding new sensors:**
```python
class CameraDetector(Detector):
    def __init__(self, fov_degrees=90, max_range=300):
        self.fov = fov_degrees
        self.max_range = max_range

    def detect(self, observer, entities):
        # Cone-based detection with LOS check
        pass
```

**Adding AI strategies:**
```python
class InterceptStrategy(Strategy):
    def update(self, drone, detected_entities, dt):
        if detected_entities:
            target = self.select_target(detected_entities)
            drone.velocity = self.intercept_vector(drone, target)
```

---

## Migration to ROS 2 (Future)

When ready for real hardware:
1. Entity classes → ROS 2 nodes
2. Detection → sensor_msgs topics
3. Physics → PX4/ArduPilot SITL
4. Visualization → RViz2 + Gazebo

The strategy/AI code ports directly since it operates on abstract entities.

---

## Success Criteria (Minimal Prototype)

✅ Arena with 3D boundaries enforced
✅ Multiple drones moving simultaneously
✅ Friend/foe designation working
✅ Detection system reports contacts
✅ 3D visualization updates in real-time
✅ Clean, documented code
✅ Extensible architecture
✅ Working example scenario

**Time estimate: 2-3 hours**
**Lines of code: ~250-300**
