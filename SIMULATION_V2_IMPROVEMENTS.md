# BIM-Agent Studio — Simulation Engine v2.0 Enhancement Summary

## Overview

The simulation logic has been significantly upgraded, taking heavy inspiration from **JuPedSim** (the open-source pedestrian dynamics framework). The core improvements focus on **realistic microscopic pedestrian movement**, **complex journey routing**, **wall/obstacle avoidance**, **batch simulation for statistical robustness**, and **standardized benchmark scenarios**.

---

## Files Added / Modified

| File | Status | Description |
|------|--------|-------------|
| `engine/pedestrian_models.py` | **NEW** | 5 pedestrian dynamics models (SFM, CFSM, AVM, GCFM, Basic) |
| `engine/journey_system.py` | **NEW** | Journey/Stage system for waypoints, waiting, flow limits, direct steering |
| `engine/benchmark_scenarios.py` | **NEW** | RiMEA & ISO 20414-inspired benchmark test cases |
| `engine/simulation_engine.py` | **UPDATED** | Integrated new models, wall cache, flow limits, batch runs, queuing |
| `engine/__init__.py` | **UPDATED** | Exports all new components |

---

## 1. Pedestrian Dynamics Models (JuPedSim-Inspired)

Previously, agents simply moved toward waypoints with a basic repulsion vector. Now you can select from **five distinct microscopic models**, each suited to different research questions:

### `basic` — Original BIM-Agent Studio Model
- Simple velocity-steering toward the next waypoint.
- Kept for backward compatibility.

### `social_force` — Social Force Model (Helbing et al. 2000)
- **Force-based** model.
- Agents experience: a **driving force** toward their goal, **exponential repulsion** from other agents, and **repulsion from walls**.
- Best for: **high-density crowds, evacuation scenarios, pushing behavior**.

### `collision_free_speed` — Collision-Free Speed Model (Tordeux et al. 2015)
- **Velocity-based** model.
- Direction is adjusted by isotropic repulsion from neighbors.
- Speed is limited by the nearest headway distance: `v = min(v0, clearance / T)`.
- Best for: **computationally efficient large-scale simulations, normal walking**.

### `anticipation_velocity` — Anticipation Velocity Model (AVM)
- **Velocity-based** with anticipation.
- Three phases: **perception** → **prediction** of future neighbor positions → **strategy selection** (choosing the best direction).
- Best for: **superior collision avoidance, lane formation, bidirectional flow**.

### `generalized_centrifugal_force` — GCFM (Chraibi et al. 2010)
- **Enhanced force-based** model.
- Adds a **centrifugal force** component based on tangential relative velocity to the Social Force Model.
- Best for: **improved lane formation and cornering behavior**.

### How to use

```python
from engine.simulation_engine import SimulationEngine, AgentType, HumanRole

engine = SimulationEngine()
scenario = engine.create_scenario(
    name="Evacuation Test",
    description="Test social force model",
    duration=300,
    time_step=0.1,
    default_movement_model="social_force"  # <-- set scenario default
)

engine.add_agent_profile(
    scenario=scenario,
    name="Occupant",
    agent_type=AgentType.HUMAN,
    role=HumanRole.VISITOR,
    count=100,
    base_speed=1.34,
    movement_model="anticipation_velocity"  # <-- or per-agent override
)
```

### Per-Agent Parameters

Each agent now carries a full `AgentMovementParams` object:
- `mass` (kg), `radius` (m)
- `desired_speed`, `max_speed`
- `relaxation_time` (tau)
- `neighbor_repulsion_strength`, `neighbor_repulsion_range`
- `geometry_repulsion_strength`, `geometry_repulsion_range`
- `social_force_A`, `social_force_B` (SFM constants)
- `centrifugal_force_factor` (GCFM)
- `reaction_time` (AVM)

These are **automatically synced** from legacy `AgentProfile` fields (`base_speed`, `size`, etc.) so old code still works.

---

## 2. Journey / Stage System

Inspired by JuPedSim's **Journey** system, agents can now follow complex multi-step behaviors instead of just having a single destination.

### Available Stages

| Stage | Purpose |
|-------|---------|
| `WaypointStage` | Navigate to a specific space |
| `WaitStage` | Wait in a space for a duration |
| `FlowLimitStage` | Pass through a door/area with a max flow rate (creates queues) |
| `EvacuateStage` | Navigate to nearest exit |
| `DirectSteeringStage` | Go to a precise 3D coordinate |
| `RepeatStage` | Repeat a sub-journey N times |

### Example: Lobby → Wait → Queue through Door → Exit

```python
from engine.journey_system import Journey, WaypointStage, WaitStage, FlowLimitStage
from engine.simulation_engine import SimulationEngine, AgentType, HumanRole

engine = SimulationEngine()
scenario = engine.create_scenario("Queue Test", "Test queueing", 600)

journey = Journey(
    id="lobby-exit",
    name="Lobby to Exit",
    stages=[
        WaitStage("lobby", wait_duration=5.0, name="Wait in lobby"),
        FlowLimitStage("exit", max_flow_rate=1.5, name="Pass through door"),  # 1.5 agents/sec
        WaypointStage("outside", name="Exit building"),
    ]
)

engine.add_agent_profile(
    scenario=scenario,
    name="Visitor",
    agent_type=AgentType.HUMAN,
    role=HumanRole.VISITOR,
    count=30,
    journey=journey,
    patience=0.8
)
```

### Flow Limitation & Queueing

`FlowLimitStage` uses a **token-bucket-like mechanism** inside `BIMSimulationModel.claim_flow_time()`:
- Each space with a flow limit tracks the next available time.
- Agents entering the stage request a time slot.
- If no slot is available, the agent enters the `QUEUING` state and waits.
- This naturally produces **queue formation** at narrow doors without explicit queue logic.

---

## 3. Wall & Obstacle Avoidance

Previously, agents had no concept of walls. They could walk through them. Now:

### Wall Segment Cache
- On simulation init, all `IfcWall` elements are converted into **2D line segments** (`WallSegment`).
- These are cached in `BIMSimulationModel._wall_segments` for O(N) fast lookup.
- Nearby segments are retrieved via `_get_wall_segments_near(position, radius)`.

### Geometry Repulsion
- Every pedestrian model now receives a list of nearby `WallSegment`s.
- Each model applies its own **wall repulsion**:
  - **SFM/GCFM**: exponential repulsion from the nearest point on the wall segment.
  - **CFSM**: isotropic repulsion + speed reduction near walls.
  - **AVM**: predicted position scoring penalizes future wall proximity.

### Soft Collision Constraint
- After integration, `_apply_wall_constraint()` pushes the agent outward if it overlaps a wall (radius < distance).

---

## 4. Batch Simulation

For **statistical robustness** (a key JuPedSim best-practice), you can now run the same scenario multiple times with different random seeds:

```python
results = engine.run_batch(
    bim_model=bim_model,
    spatial_engine=spatial_engine,
    scenario=scenario,
    runs=10,
    seeds=[42, 43, 44, 45, 46, 47, 48, 49, 50, 51]
)

# Get aggregated statistics
summary = engine.get_batch_summary()
print(summary)
# {
#   "runs": 10,
#   "avg_speed_mean": 1.12,
#   "avg_speed_std": 0.08,
#   "evacuation_time_mean": 87.5,
#   "evacuation_time_std": 5.2,
#   ...
# }
```

This is essential for **scientific reproducibility** and comparing models.

---

## 5. Benchmark Scenarios (RiMEA-Inspired)

Standardized test cases allow you to **validate** that a model produces physically plausible results:

| Scenario | What It Tests |
|----------|---------------|
| **RiMEA-1: Straight Corridor** | Single pedestrian walks at desired speed. |
| **RiMEA-2: Room Evacuation** | 50 agents find a 1m exit. |
| **RiMEA-3: Bottleneck Flow** | Fundamental diagram (flow vs. density). |
| **RiMEA-4: Counterflow** | Lane formation in bidirectional flow. |
| **ISO-20414: Density-Speed** | Speed reduction at different densities. |
| **High Density Stability** | No overlaps or explosions at >4 p/m². |
| **Queue Formation** | Natural queueing at narrow door. |

### Usage

```python
from engine.benchmark_scenarios import BenchmarkScenarios

scenario = BenchmarkScenarios.get("RiMEA-3: Bottleneck Flow")
# or
scenario = BenchmarkScenarios.rimea_3_bottleneck()
```

---

## 6. New Simulation Events

Two new event types were added to the engine:

### `set_journey`
Assign a journey to matching agents at a specific time.

```python
engine.add_event(
    scenario, time=60, event_type="set_journey",
    filter={"profile.role": HumanRole.VISITOR},
    journey=journey
)
```

### `switch_model`
Dynamically switch an agent's movement model mid-simulation.

```python
engine.add_event(
    scenario, time=120, event_type="switch_model",
    filter={"profile.agent_type": AgentType.HUMAN},
    model="social_force"
)
```

---

## 7. Improved Metrics

The `SimulationMetrics` dataclass now includes:

- `agents_queuing` — count of agents waiting in flow-limited stages.
- `avg_density` — average persons/m² across occupied spaces.
- `flow_rate` — cumulative evacuation rate (agents/sec).

The DataCollector also tracks the **movement model** used by each agent.

---

## 8. Backward Compatibility

All existing code continues to work:

- `AgentProfile`, `SimulationScenario`, `BIMSimulationModel`, `SimulationEngine` signatures are unchanged.
- Legacy `base_speed`, `max_speed`, `size` fields are automatically copied into `movement_params`.
- If no `movement_model` is specified, it defaults to `"basic"` (the original behavior).
- All existing presets (`office_scenario`, `evacuation_scenario`, `hospital_scenario`, `university_scenario`) are preserved and updated with sensible default models.

---

## Quick Start: Running a Model Comparison

```python
from engine.simulation_engine import SimulationEngine
from engine.benchmark_scenarios import BenchmarkScenarios

# Load your BIM model and spatial engine first...

for model_name in ["basic", "social_force", "collision_free_speed", "anticipation_velocity"]:
    scenario = BenchmarkScenarios.get("RiMEA-3: Bottleneck Flow")
    scenario.default_movement_model = model_name
    
    model = engine.initialize_simulation(bim_model, spatial_engine, scenario)
    engine.run()
    results = engine.get_results()
    
    print(f"{model_name}: evacuated={results['evacuated_agents']}, "
          f"congestions={len(results['congestion_events'])}")
```

---

## Dependencies

No new runtime dependencies were added. The improvements use only:
- `numpy` (already required)
- `mesa` (already required)
- `networkx` (already required)
- Standard library (`abc`, `copy`, `math`, `uuid`)

---

## Future Enhancements (Next Version Ideas)

Based on the JuPedSim roadmap, the following could be added next:

1. **Fire Dynamics Simulator (FDS) coupling** — read actual FDS output files (smoke concentration, temperature) to drive FED accumulation and visibility calculation instead of the current scalar approximation.
2. **AI-assisted scenario copilot** (MCP-based) — natural language scenario generation.
3. **Spatial index for space-occupancy lookup** — `_update_current_space()` is O(M) per agent; a spatial hash of space bounding boxes would accelerate it.
4. **Agent mobility impairments** — wheelchair users, elderly with reduced max speed; tie into `needs_accessible` flag and accessible pathfinding.
5. **Multi-floor evacuation** — stairwell flow limits + elevator queuing integrated with the Journey system.

---

## v1.3.0 Feature Summary

| Feature | Status | Files |
|---------|--------|-------|
| WarpDriver Model (gradient nav field) | ✅ Done | `pedestrian_models.py` |
| UniformGridSpatialIndex (O(1) wall queries) | ✅ Done | `pedestrian_models.py`, `simulation_engine.py` |
| FED tracking per agent (ISO 13571) | ✅ Done | `simulation_engine.py` |
| Smoke/visibility speed effects (Jin 1978) | ✅ Done | `simulation_engine.py` |
| Runtime geometry switching (block/unblock) | ✅ Done | `simulation_engine.py` |
| Group behavior (cohesion + leader-follower) | ✅ Done | `simulation_engine.py` |
| BlockedPathStage (journey-level path waiting) | ✅ Done | `journey_system.py` |
| RiMEA-5 T-Junction benchmark | ✅ Done | `benchmark_scenarios.py` |
| FED Evacuation benchmark | ✅ Done | `benchmark_scenarios.py` |
| fire_evacuation_scenario preset | ✅ Done | `simulation_engine.py` |

---

## v2.0 → v1.3.0 Cumulative Summary

| Feature | Before v2.0 | After v2.0 | After v1.3.0 |
|---------|-------------|------------|--------------|
| Movement models | 1 simple | 5 physics-based | **6 (+ WarpDriver)** |
| Wall query cost | O(N) brute-force | O(N) brute-force | **O(1) avg (grid index)** |
| Fire safety | Node flag only | Node flag only | **FED + smoke + speed effects** |
| Path blocking | Attribute-only | Attribute-only | **Graph edge removal + rerouting** |
| Group dynamics | group_id field unused | group_id field unused | **Cohesion force + leader-follower** |
| Benchmark tests | 0 | 7 RiMEA/ISO | **9 (+ T-junction + FED evac)** |
| Journey stages | — | 6 stages | **7 (+ BlockedPathStage)** |
| Scenario presets | 4 | 5 | **6 (+ fire_evacuation)** |
| Metrics | speed, density | + queuing, flow rate | **+ FED, incapacitated, smoke** |

