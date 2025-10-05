# Drone Swarm Defense Simulation

A Python-based simulation framework for testing counter-drone defense strategies using AI and multi-agent systems.

## Features

- **3D Arena**: Bounded airspace with configurable dimensions
- **Entity System**: Drones with team (friend/foe) and role designations
- **Physics Engine**: Simple kinematic motion with boundary enforcement
- **Detection System**: Range-based sensor detection with extensible architecture
- **Real-time Visualization**: 3D Matplotlib plotting with detection ranges and trails
- **Extensible Architecture**: Clean design for adding AI strategies, sensors, and behaviors

## Quick Start

### Installation

```bash
# Clone or navigate to the repository
cd /Users/russ/dev/int/swarm

# Install dependencies
pip install numpy matplotlib

# Or use requirements.txt
pip install -r requirements.txt
```

### Run the Example

```bash
python examples/basic_scenario.py
```

This runs a scenario with 3 friendly patrol drones circling an area. An enemy intruder enters the arena, and detection events are logged when friendly drones detect the threat.

### What You'll See

- **3D visualization** showing the arena boundaries
- **Blue drones**: Friendly patrol units
- **Red drone**: Enemy intruder
- **Dotted circles**: Detection ranges
- **Lines**: Trajectory trails (if enabled)
- **Arrows**: Velocity vectors
- **Console output**: Detection events in real-time

## Architecture

```
swarm/
├── src/
│   ├── arena.py         # Arena with 3D boundaries
│   ├── entities.py      # Entity, Drone classes with Team/Role
│   ├── physics.py       # Kinematic physics engine
│   ├── detection.py     # Sensor/detection systems
│   └── visualization.py # 3D Matplotlib plotting
├── examples/
│   └── basic_scenario.py # Demo scenario
└── tests/
    └── (unit tests)
```

## Core Concepts

### Arena

A bounded 3D volume representing airspace:
- Configurable dimensions (e.g., 1km × 1km × 500m)
- Optional geographic reference point
- Boundary enforcement for entities

### Entities

Base class for all objects in the simulation:
- **Position**: 3D coordinates (x, y, z) in meters
- **Velocity**: 3D velocity vector in m/s
- **Team**: FRIENDLY, ENEMY, or NEUTRAL
- **Role**: PATROL, ATTACK, SUPPORT, INTRUDER, etc.

### Drones

Specialized entities with flight properties:
- Max speed limiting
- Detection range
- Battery capacity (for future use)
- Friend/foe identification

### Detection

Range-based sensor system:
- Detects entities within radius
- Returns distance and bearing
- Filters by friend/foe

### Physics

Simple kinematic updates:
- Position += velocity × dt
- Boundary collision handling
- Helper methods for circular paths, targeting

## Extending the Simulation

### Adding New Entity Types

```python
from src.entities import Entity, Team, Role

class ChargingStation(Entity):
    def __init__(self, position):
        super().__init__(
            position=position,
            team=Team.FRIENDLY,
            role=Role.SUPPORT
        )
        self.charge_rate = 100  # mAh/s
```

### Adding New Sensors

```python
from src.detection import Detector
import numpy as np

class CameraDetector(Detector):
    def __init__(self, max_range=300, fov_degrees=90):
        super().__init__(max_range)
        self.fov = np.radians(fov_degrees)

    def detect(self, observer, entities):
        # Custom cone-based detection logic
        pass
```

### Adding AI Strategies

```python
class InterceptStrategy:
    def update(self, drone, detected_entities, dt):
        if detected_entities:
            target = detected_entities[0]  # Closest
            # Calculate intercept course
            drone.set_velocity(intercept_vector)
```

## Roadmap

See `PLAN.md` for detailed extension plan.

### Phase 1: Enhanced Physics
- [ ] Realistic flight dynamics (thrust, drag)
- [ ] Wind simulation
- [ ] Drone-to-drone collision detection
- [ ] Terrain following

### Phase 2: Resource Management
- [ ] Battery consumption model
- [ ] Charging stations
- [ ] Low-battery return-to-launch
- [ ] Handoff protocols

### Phase 3: Advanced Sensors
- [ ] Camera (cone-based, LOS required)
- [ ] Radar (long range, 360°)
- [ ] Audio (short range)
- [ ] Sensor fusion

### Phase 4: GPS-Denied Navigation
- [ ] Mobile transponders
- [ ] Differential positioning
- [ ] Position uncertainty

### Phase 5: AI & Strategy
- [ ] State machines (OBSERVE → TRACK → INTERCEPT)
- [ ] Reinforcement learning agents
- [ ] Swarm coordination
- [ ] Threat prediction

### Phase 6: Advanced Scenarios
- [ ] Capture the flag
- [ ] Defend infrastructure
- [ ] Multi-wave attacks
- [ ] Swarm vs swarm

## Example Use Cases

1. **Algorithm Testing**: Test swarm coordination algorithms
2. **AI Training**: Train RL agents for counter-drone tactics
3. **Strategy Development**: Develop and evaluate defense strategies
4. **Education**: Learn multi-agent systems and robotics
5. **Research**: Prototype for academic research

## Migration to ROS 2

When ready for real hardware:
1. Entities → ROS 2 nodes
2. Detection → `sensor_msgs` topics
3. Physics → PX4/ArduPilot SITL
4. Visualization → RViz2 + Gazebo

The AI strategy code ports directly.

## Requirements

- Python 3.10+
- NumPy
- Matplotlib

## License

MIT (or your choice)

## Contributing

This is a minimal prototype designed for extension. Key areas for contribution:
- Additional sensors
- AI strategies
- Realistic physics
- Performance optimization
- Unit tests

## Questions?

See `PLAN.md` for the complete build plan and architecture decisions.
