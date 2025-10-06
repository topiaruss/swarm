# Network Partitioning and Charging Management

## Overview

This document describes the design for handling mesh network partitions and battery/charging management in the defensive drone swarm. These features are critical for real-world deployment where:
- Network connectivity is unreliable (jamming, interference, range limits)
- Drones must operate autonomously when disconnected
- Battery limitations require rotation through charging stations
- Coverage must be maintained during charging cycles

---

## Part 1: Network Partition Handling

### Problem Statement

Mesh networks can partition (split) due to:
- **Distance**: Drones fly beyond radio range
- **Interference**: Radio jamming or environmental noise
- **Node failure**: Drone crashes or loses radio
- **Intentional**: Drones charging at base station (radio off)

When partitioned, drones must:
1. **Detect** they've lost connectivity to some/all peers
2. **Operate autonomously** using last known information
3. **Rejoin gracefully** when connectivity restored
4. **Synchronize state** after rejoining (catch up on missed info)

### Detection Mechanisms

#### Heartbeat Protocol
Each drone broadcasts periodic heartbeat messages:
```python
Heartbeat {
    drone_id: str
    timestamp: float
    position: [x, y, z]
    battery_level: float
    role_status: str  # "PATROL", "CHARGING", "TRACKING", etc.
    sequence_number: int  # Monotonic counter
}
```

**Timing:**
- Broadcast interval: 1 Hz (every second)
- Timeout threshold: 5 seconds (5 missed heartbeats)
- If no heartbeat from peer for >5s → assume disconnected

#### Network Topology Tracking
Each drone maintains:
```python
NetworkState {
    connected_peers: Set[drone_id]
    last_seen: Dict[drone_id, timestamp]
    partition_detected_at: Optional[float]
    partition_id: int  # Which partition am I in?
}
```

#### Partition Detection Algorithm
```
FOR each peer in expected_peers:
    IF (current_time - last_seen[peer]) > TIMEOUT:
        Mark peer as disconnected

IF len(connected_peers) < (total_swarm_size - 1):
    partition_detected = True
    Switch to AUTONOMOUS mode
```

### Autonomous Operation

When partitioned, each drone/sub-swarm operates independently:

#### Information to Preserve
- **Last known positions** of all drones (even disconnected ones)
- **Threat detections** (with timestamps and uncertainty)
- **Patrol assignments** (continue your assigned sector)
- **Battery states** (when peers last reported charge level)

#### Autonomous Behaviors

**1. Continue Patrol**
- Maintain assigned patrol sector
- Use last known positions to avoid overlapping with (possibly still active) peers
- Increase patrol radius slightly to cover potential gaps

**2. Threat Response**
- Track detected threats independently
- **DO NOT** assume other drones are coordinating (they might not see your messages)
- Alert owner even if uncertain whether others have also alerted

**3. Conservative Battery Management**
- Assume charging station may be occupied by disconnected peer
- Return to base earlier (e.g., 30% instead of 20% battery)
- Hover near base if occupied, wait for opening

**4. Position Uncertainty Growth**
- Disconnected drones' positions become increasingly uncertain
- Model as growing uncertainty sphere: uncertainty(t) = base + drift_rate * time_disconnected
- After 60s disconnected, treat peer position as "unknown"

### Rejoin Protocol

When connectivity restored:

#### 1. Detect Reconnection
```
IF receive_heartbeat(peer) AND peer in disconnected_peers:
    Mark peer as reconnected
    Initiate state sync with peer
```

#### 2. State Synchronization

**Information Exchange (bidirectional):**
```python
StateSyncRequest {
    drone_id: str
    last_sequence_seen: Dict[drone_id, int]  # What seq# did I last see from each peer?
    threat_log: List[ThreatDetection]  # Threats I saw while disconnected
    battery_history: List[(timestamp, level)]
    partition_duration: float
}
```

**Sync Process:**
1. Each drone sends StateSyncRequest to newly reconnected peer
2. Peer responds with any messages/events missed (based on sequence numbers)
3. Merge threat detections (deduplicate by position + timestamp)
4. Update patrol assignments if needed (resolve any conflicts)

#### 3. Conflict Resolution

**Patrol Overlap:**
- If two drones now patrolling same sector → higher battery level keeps sector, lower battery relocates

**Charging Station Conflict:**
- If multiple drones converged on charger while disconnected → queue by battery level (lowest charges first)

**Threat Tracking:**
- If same threat detected by multiple partitions → merge tracks, use most recent high-confidence position

### Edge Cases

**1. Solo Drone (Completely Isolated)**
- Continues patrol alone
- Alerts owner of network failure (user needs to know system degraded)
- Increases alert sensitivity (no backup confirmation available)

**2. Multiple Partitions**
- Partition A: Drones 1, 2
- Partition B: Drones 3, 4
- Each partition operates as mini-swarm
- When A & B rejoin, synchronize as if two "peers" reconnecting

**3. Drone Returns from Charging**
- Charging drone missed N seconds of mesh traffic (radio off)
- Upon powering radio back on, requests state sync
- Other drones send condensed update (recent threats, positions, assignments)

**4. Permanent Node Loss**
- If drone never reconnects after timeout (e.g., 300s), assume permanent failure
- Remaining drones redistribute patrol coverage
- Alert owner of reduced swarm capacity

---

## Part 2: Charging Management

### Problem Statement

Drones have limited battery capacity and must rotate through charging stations to maintain continuous coverage. Challenges:
- **Coverage gaps**: Don't want all drones charging simultaneously
- **Battery optimization**: Keep drones highly charged (ready for extended tracking mission)
- **Coordination**: Avoid contention for limited charging slots
- **Network impact**: Charging drones don't transmit (stealth + wiring constraints)

### Charging Station Design

#### Physical Model
```python
ChargingStation(Entity):
    position: [x, y, z]  # Near arena center
    capacity: int  # Number of simultaneous charge slots (e.g., 4)
    charge_rate: float  # % per second (e.g., 2%/s = 50s for full charge)
    occupied_slots: Dict[slot_id, drone_id]
    queue: List[drone_id]  # Drones waiting for slot
```

**Placement:**
- Center of arena (minimizes max return distance)
- Ground level (z=0) for safety
- Visible/advertised position (all drones know location)

**Charging Process:**
1. Drone lands on available slot
2. Radio transmitter **DISABLED** (stealth + wiring)
3. Radio receiver **ENABLED** (can hear mesh, can't talk)
4. Battery charges at fixed rate
5. When charged/needed, drone departs and re-enables transmitter

### Battery Model

#### Drone Battery Parameters
```python
Battery:
    capacity: float = 5000.0  # mAh (milliamp-hours)
    current_charge: float  # 0.0 to capacity
    discharge_rate: float = 50.0  # mAh/second (10min flight time)
    critical_level: float = 0.15  # 15% = must return NOW
    optimal_level: float = 0.80  # 80% = should return soon
    full_level: float = 0.95  # 95% = consider fully charged
```

**Discharge Rates:**
- **Patrol (cruising)**: 50 mAh/s (base rate)
- **Tracking (high speed)**: 80 mAh/s (60% more power)
- **Hovering**: 40 mAh/s (20% less power)

**Flight Time:**
- Full battery (5000 mAh) @ 50 mAh/s = 100 seconds = ~1.7 minutes
- This is realistic for micro quads (scale up for larger drones in real deployment)

### Charging Heuristic

**Goal:** Maximize operational uptime while keeping fleet highly charged.

#### When to Return to Charge

**Decision Function:**
```python
def should_return_to_charge(drone) -> bool:
    battery_pct = drone.battery.level()

    # Critical: MUST return immediately
    if battery_pct < CRITICAL_LEVEL:
        return True

    # Calculate return energy cost
    distance_to_base = distance(drone.position, charging_station.position)
    energy_to_return = estimate_energy(distance_to_base, current_velocity)
    energy_margin = drone.battery.current_charge - energy_to_return

    # Not enough to return safely + margin
    if energy_margin < 0.1 * capacity:
        return True

    # Below optimal AND charging slot available
    if battery_pct < OPTIMAL_LEVEL and charging_station.has_available_slot():
        return True

    # Below optimal AND no one in queue with lower battery
    if battery_pct < OPTIMAL_LEVEL:
        lowest_in_queue = min([d.battery.level() for d in queue], default=1.0)
        if battery_pct < lowest_in_queue:
            return True

    return False
```

**Priority Queue:**
When multiple drones want to charge, order by:
1. **Critical level** (< 15%) → absolute priority
2. **Battery level** (lowest first) → fairness
3. **Time since last charge** (longest first) → prevent starvation

#### When to Leave Charger

**Decision Function:**
```python
def should_leave_charger(drone) -> bool:
    battery_pct = drone.battery.level()

    # Fully charged
    if battery_pct >= FULL_LEVEL:
        return True

    # Someone waiting with critical battery
    if any(d.battery.level() < CRITICAL_LEVEL for d in queue):
        if battery_pct >= OPTIMAL_LEVEL:  # I'm good enough
            return True

    # Coverage gap detected (hear on radio receiver)
    if detect_coverage_gap() and battery_pct >= 0.50:
        return True  # Emergency exit

    return False
```

### Patrol Handoff Coordination

When a drone leaves for charging, coverage must be maintained.

#### Handoff Protocol

**1. Announce Intent (while still operational)**
```python
ChargeRequest {
    drone_id: str
    current_battery: float
    estimated_departure_time: float
    patrol_sector: Sector  # Which area I'm covering
    handoff_needed: bool
}
```

**2. Peer Response**
```python
# Other drones evaluate: can I cover that sector?
def can_cover_handoff(request) -> bool:
    my_battery = self.battery.level()

    # I'm also low, can't help
    if my_battery < OPTIMAL_LEVEL:
        return False

    # I'm already covering adjacent sector, can expand
    if self.patrol_sector.adjacent_to(request.patrol_sector):
        return True

    return False

# Highest battery volunteer responds
HandoffAccept {
    drone_id: str
    my_battery: float
}
```

**3. Transition**
- Departing drone continues patrol until handoff drone arrives at sector boundary
- Handoff drone expands patrol to cover both sectors
- Departing drone returns to base

**4. Return from Charging**
- Charged drone announces availability
- If original sector still needs coverage, reclaim it
- Otherwise, take on different role (roaming reserve, extended range patrol)

### Radio Silence During Charging

**Impact on Mesh:**
- Charging drone **does not relay** messages (can't transmit)
- Other drones must route around it
- Network topology recalculates without charging node

**Benefits:**
1. **Stealth**: Charging station doesn't become radio beacon (could be targeted)
2. **Wiring**: Difficult to share power lines with RF transmission (noise)
3. **Power**: Radio draws power, charging wants maximum rate

**Receive-Only Mode:**
- Charging drone **listens** to mesh traffic
- Builds up current state (threat positions, peer locations)
- When departing, already synchronized (no need for full state sync)
- Only announces own position once radio re-enabled

### Coverage Maintenance

**Goal:** Always maintain minimum patrol coverage even during charging cycles.

#### Coverage Rules

1. **Minimum Active Drones:**
   - If swarm size N, require at least ⌈N/2⌉ active (e.g., 5 drones → 3 minimum active)
   - Block additional charging requests if would violate minimum

2. **Sector Coverage:**
   - Arena divided into sectors (e.g., 8 sectors around perimeter)
   - Each sector must have at least 1 drone within detection range
   - If sector uncovered, highest-battery available drone assigned

3. **Emergency Protocol:**
   - If threat detected, ALL drones (even charging) notified
   - Charging drones can emergency-depart (even at low charge) if needed for response
   - Owner alert if forced to pull charging drone (indicates insufficient swarm size)

### Metrics and Monitoring

Track these metrics to evaluate charging heuristic effectiveness:

```python
ChargeMetrics:
    total_charge_cycles: int
    average_battery_level: float  # Fleet average over time
    coverage_gaps: int  # Times when sector had no coverage
    emergency_departures: int  # Forced to leave charger early
    queue_wait_time: List[float]  # How long drones wait for slot
    energy_wasted: float  # Energy spent traveling to/from charger
```

**Optimization Goals:**
- Average fleet battery > 60% (well charged)
- Coverage gaps = 0 (perfect coverage)
- Emergency departures < 5% of cycles (good planning)
- Queue wait time < 10s (sufficient capacity)

---

## Implementation Plan

### Phase 1: Network Partition Detection (Feature #3)

**Files to Create/Modify:**
- `src/network.py` - Network state tracking, partition detection
- `src/entities.py` - Add network state to Drone class
- `src/mesh.py` - Add heartbeat protocol, connectivity tracking

**Core Classes:**
```python
class NetworkState:
    def __init__(self, drone_id: str, expected_peers: Set[str])
    def update_heartbeat(self, peer_id: str, timestamp: float)
    def detect_partitions(self, current_time: float) -> List[Set[str]]
    def is_connected_to(self, peer_id: str) -> bool

class HeartbeatMessage(Message):
    drone_id: str
    position: np.ndarray
    battery_level: float
    sequence_number: int
```

**Test Scenario:**
- 4 drones patrolling
- At t=10s, manually partition network (drones 1,2 can't see 3,4)
- Verify each sub-swarm continues independently
- At t=30s, restore connectivity
- Verify state synchronization

### Phase 2: Autonomous Operation (Feature #3 continued)

**Files to Modify:**
- `src/strategy.py` - Autonomous decision making
- `src/detection.py` - Independent threat handling

**Autonomous Behaviors:**
- Continue patrol with last known peer positions
- Track threats independently
- Conservative battery management
- Alert owner of degraded network

**Test Scenario:**
- Partition network during active threat tracking
- Verify both partitions continue tracking
- Verify alerts sent from both (even if duplicate)
- Verify threat tracks merge after rejoin

### Phase 3: State Synchronization (Feature #3 continued)

**Files to Create/Modify:**
- `src/sync.py` - State sync protocol
- `src/mesh.py` - Message history, sequence numbers

**Core Classes:**
```python
class StateSyncManager:
    def request_sync(self, peer_id: str, last_seen_seq: int)
    def respond_to_sync(self, request: StateSyncRequest) -> StateSyncResponse
    def merge_threat_tracks(self, local: List, remote: List) -> List
    def resolve_conflicts(self, local_state: State, remote_state: State)
```

**Test Scenario:**
- Partition network
- Both partitions detect different threats
- Rejoin network
- Verify both threats now known to all drones
- Verify no duplicate/conflicting information

### Phase 4: Charging Station Infrastructure (Feature #4)

**Files to Create/Modify:**
- `src/entities.py` - Add ChargingStation class, Battery model
- `src/arena.py` - Support for charging stations
- `src/visualization.py` - Render charging station, battery indicators

**Core Classes:**
```python
class ChargingStation(Entity):
    capacity: int
    charge_rate: float
    occupied_slots: Dict[int, str]  # slot -> drone_id
    queue: List[str]  # waiting drone_ids

    def request_slot(self, drone_id: str) -> Optional[int]
    def release_slot(self, slot: int)
    def charge_drone(self, drone: Drone, dt: float)

class Battery:
    capacity: float
    current_charge: float
    discharge_rate: float

    def level(self) -> float  # 0.0 to 1.0
    def discharge(self, dt: float, rate_multiplier: float = 1.0)
    def charge(self, dt: float, charge_rate: float)
    def time_to_empty(self, rate: float) -> float
    def time_to_full(self, charge_rate: float) -> float
```

**Test Scenario:**
- 3 drones, 2-slot charging station
- Manually drain all batteries to 20%
- Verify drones queue properly
- Verify 2 charge simultaneously, 1 waits
- Verify queue ordering (lowest battery first)

### Phase 5: Battery Management & Return Logic (Feature #4 continued)

**Files to Create/Modify:**
- `src/strategy.py` - Add charging decision logic
- `src/navigation.py` - Return-to-base pathfinding

**Core Functions:**
```python
def should_return_to_charge(drone: Drone, station: ChargingStation) -> bool
def calculate_return_energy(drone: Drone, station: ChargingStation) -> float
def should_leave_charger(drone: Drone, station: ChargingStation) -> bool
def prioritize_charging_queue(queue: List[Drone]) -> List[Drone]
```

**Test Scenario:**
- 5 drones continuous patrol
- Batteries drain naturally
- Verify drones return automatically at thresholds
- Verify no energy-exhaustion failures (ran out before reaching base)
- Measure coverage gaps (should be none)

### Phase 6: Patrol Handoff Coordination (Feature #4 continued)

**Files to Modify:**
- `src/strategy.py` - Handoff protocol
- `src/mesh.py` - Add ChargeRequest, HandoffAccept messages

**Core Protocol:**
```python
def announce_charge_intent(drone: Drone) -> ChargeRequest
def find_handoff_volunteer(request: ChargeRequest, peers: List[Drone]) -> Optional[Drone]
def execute_handoff(departing: Drone, covering: Drone, sector: Sector)
def reclaim_sector(returning: Drone, current_coverage: List[Drone])
```

**Test Scenario:**
- 4 drones, 4 sectors (perimeter divided in quadrants)
- Drain drone 1 battery, trigger return-to-charge
- Verify handoff request sent
- Verify drone 2 (adjacent sector) volunteers and expands coverage
- Verify no gap in drone 1's former sector during transition

### Phase 7: Charging Heuristic & Optimization (Feature #4 continued)

**Files to Modify:**
- `src/strategy.py` - Advanced charging decision logic
- `src/metrics.py` - Charging metrics collection

**Heuristic Improvements:**
- Predictive scheduling (charge before needed, when slots available)
- Load balancing (avoid all drones low simultaneously)
- Emergency reserves (always keep 1 drone at >80%)

**Test Scenario:**
- 6 drones, 3-slot station, 60-minute simulation
- Verify average fleet battery > 60%
- Verify zero coverage gaps
- Measure total energy wasted on charging trips
- Compare to naive heuristic (charge only at critical)

### Phase 8: Example Scenario (Both Features)

**Files to Create:**
- `examples/network_partition.py` - Demo partition detection & recovery
- `examples/charging_management.py` - Demo charging cycles & coverage
- `examples/combined_ops.py` - Demo both (partition while charging)

**Combined Scenario:**
- 8 drones, 4-slot charging station
- Active threat tracking
- Manual network partition at t=30s (jamming simulation)
- Battery drain during partition
- Drones in each partition must manage charging independently
- Rejoin at t=60s
- Verify state sync, threat info merged, charging queue reconciled

---

## Success Criteria

### Network Partitioning (Feature #3)
- ✅ Detects partition within 5 seconds
- ✅ Continues patrol autonomously during partition
- ✅ Rejoins gracefully when connectivity restored
- ✅ Synchronizes state correctly (no lost threats)
- ✅ Handles multi-partition scenarios (A+B, C+D)
- ✅ Solo drone operates correctly (complete isolation)

### Charging Management (Feature #4)
- ✅ Drones return before battery exhausted (0% failures)
- ✅ Coverage maintained during charging (0 gaps)
- ✅ Queue prioritizes critical battery correctly
- ✅ Average fleet battery > 60%
- ✅ Charging drones silent on mesh (receive-only)
- ✅ Handoff coordination works (no dropped sectors)
- ✅ Charged drones rejoin smoothly

### Combined
- ✅ Charging drone can rejoin mesh after partition heals
- ✅ Network partition doesn't cause charging conflicts
- ✅ System handles worst case (partition + multiple low batteries)

---

## Future Enhancements

**Dynamic Charging Station:**
- Mobile charging station (truck/trailer)
- Station relocation based on threat zones

**Smart Scheduling:**
- Machine learning to predict optimal charging times
- Anticipate threat patterns (charge during quiet periods)

**Multi-Station:**
- Multiple charging stations for large areas
- Load balancing across stations

**Fast Charging:**
- Hot-swap batteries (land, swap, relaunch in 10s)
- Partial charges (just enough to continue mission)

**Emergency Power Sharing:**
- Drone-to-drone wireless charging (future tech)
- Sacrifice one drone to charge another in critical situation
