"""
Agent-Based Simulation Engine
Powered by Mesa for agent-based modeling and simulation.

v2.0 Enhancements (inspired by JuPedSim):
- Multiple pedestrian dynamics models (Social Force, Collision-Free Speed,
  Anticipation Velocity, Generalized Centrifugal Force)
- Journey / Stage system for complex routing (waypoints, waiting, flow limits)
- Proper wall/obstacle avoidance with geometry-based repulsion
- Per-agent movement parameters
- Batch simulation support for statistical robustness
- Benchmark scenarios (RiMEA-inspired)
- Flow limitation and queueing support

v1.3.0 Enhancements:
- WarpDriver pedestrian model (gradient navigation field)
- UniformGridSpatialIndex for O(1) average wall queries
- Fractional Effective Dose (FED) tracking per agent (ISO 13571)
- Smoke/visibility effects on agent speed and pathfinding
- Runtime geometry switching: block_path now removes graph edges;
  new unblock_path event restores them; affected agents are rerouted
- Group behavior: cohesion force + leader-follower dynamics
- New benchmark scenarios: RiMEA-5 (T-junction), FED evacuation
"""

import uuid
import random
import logging
import copy
import math
from typing import Dict, List, Optional, Tuple, Any, Callable
from dataclasses import dataclass, field
from enum import Enum

import numpy as np
import mesa
from mesa import Model, Agent
from mesa.datacollection import DataCollector

# Mesa 2.x / 3.x scheduler compatibility
try:
    from mesa.time import SimultaneousActivation
except ImportError:
    class SimultaneousActivation:
        def __init__(self, model):
            self.model = model
            self._agents: dict = {}
            self.steps = 0
            self.time = 0

        def add(self, agent):
            self._agents[agent.unique_id] = agent

        def remove(self, agent):
            self._agents.pop(agent.unique_id, None)

        @property
        def agents(self):
            return list(self._agents.values())

        def step(self):
            for agent in list(self._agents.values()):
                agent.step()
            self.steps += 1
            self.time += 1

try:
    from mesa.space import ContinuousSpace
except ImportError:
    ContinuousSpace = None

from core.bim_processor import BIMModel, BIMSpace, BIMElement, ElementCategory
from core.spatial_engine import SpatialGraph, SpatialIntelligenceEngine
from engine.pedestrian_models import (
    PedestrianModel, AgentMovementParams, WallSegment,
    UniformGridSpatialIndex, get_model, list_models
)
from engine.journey_system import (
    Journey, Stage, WaypointStage, WaitStage, FlowLimitStage, EvacuateStage,
    DirectSteeringStage, BlockedPathStage, StageResult, StageState
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Enums (kept for backward compatibility)
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Group Behavior (NEW v1.3.0)
# ---------------------------------------------------------------------------

@dataclass
class GroupBehavior:
    """
    Parameters controlling group cohesion and leader-follower dynamics.

    Agents sharing the same `group_id` in their AgentProfile are linked at
    simulation init. The first agent created in the group becomes the leader;
    all others are followers. The leader navigates normally; followers are
    attracted toward the leader's position (cohesion spring) while still
    applying the standard pedestrian model for collision avoidance.
    """
    cohesion_strength: float = 0.8       # N — spring pull toward group centroid
    separation_distance: float = 1.5    # m — preferred spacing within group
    leader_id: Optional[int] = None     # unique_id of the group leader agent
    follow_weight: float = 0.6          # blend between leader target and own target


# ---------------------------------------------------------------------------
# FED Constants (NEW v1.3.0) — simplified ISO 13571 CO + O₂ deficit model
# ---------------------------------------------------------------------------

# Time-to-incapacitation at different CO concentrations (ppm) and
# O₂ deficit fractions, combined into a single hazard rate per time-step.
# We use a simplified scalar "hazard_intensity" per fire zone (0–1).
# FED increment per second = hazard_intensity * FED_RATE_COEFFICIENT
# Agent is incapacitated when FED >= 1.0
FED_RATE_COEFFICIENT: float = 0.02      # 1/s at hazard_intensity=1 → 50s to incap
FED_SPEED_FACTOR: float = 0.3           # max speed reduction at FED=1 (30% of v0)

# Smoke visibility constants (Jin 1978 / SFPE Handbook)
# visibility = K / (smoke_OD_per_m)   — we use a dimensionless 0–1 smoke_level
# speed_factor = (visibility / max_visibility)^0.5 clamped [0.3, 1.0]
SMOKE_MAX_VISIBILITY: float = 10.0     # m — free-smoke visibility
SMOKE_MIN_SPEED_RATIO: float = 0.3     # minimum speed ratio in dense smoke


class AgentType(Enum):
    HUMAN = "human"
    VEHICLE = "vehicle"
    SERVICE = "service"
    AUTONOMOUS = "autonomous"


class HumanRole(Enum):
    OFFICE_WORKER = "office_worker"
    STUDENT = "student"
    PATIENT = "patient"
    VISITOR = "visitor"
    RESIDENT = "resident"
    STAFF = "staff"


class AgentState(Enum):
    IDLE = "idle"
    MOVING = "moving"
    WORKING = "working"
    WAITING = "waiting"
    RESTING = "resting"
    EVACUATING = "evacuating"
    INTERACTING = "interacting"
    DISABLED = "disabled"
    QUEUING = "queuing"


# ---------------------------------------------------------------------------
# Agent Profile
# ---------------------------------------------------------------------------

@dataclass
class AgentProfile:
    """Profile defining agent behavior characteristics."""
    id: str
    name: str
    agent_type: AgentType
    role: Optional[HumanRole] = None

    # Movement properties
    base_speed: float = 1.2  # m/s
    max_speed: float = 1.5
    acceleration: float = 0.5
    turn_rate: float = 90.0  # degrees

    # Behavior properties
    patience: float = 0.5  # 0-1
    sociability: float = 0.5  # 0-1
    risk_tolerance: float = 0.5  # 0-1
    destination_preference: List[str] = field(default_factory=list)

    # Physical properties
    size: float = 0.5  # meters radius
    vision_range: float = 10.0  # meters

    # Schedule
    schedule: Dict[float, str] = field(default_factory=dict)  # time -> activity

    # Special attributes
    needs_accessible: bool = False
    can_use_stairs: bool = True
    can_use_elevator: bool = True
    group_id: Optional[str] = None

    # NEW v2.0: Movement model selection
    movement_model: str = "basic"  # basic, social_force, collision_free_speed, anticipation_velocity, generalized_centrifugal_force, warp_driver
    movement_params: AgentMovementParams = field(default_factory=AgentMovementParams)

    # NEW v2.0: Journey for complex routing
    journey: Optional[Journey] = None

    # NEW v1.3.0: Group behavior config
    group_behavior: Optional[GroupBehavior] = None

    def __post_init__(self):
        # Sync legacy fields into movement_params
        if self.movement_params.desired_speed == 1.34 and self.base_speed != 1.34:
            self.movement_params.desired_speed = self.base_speed
        if self.movement_params.max_speed == 2.0 and self.max_speed != 2.0:
            self.movement_params.max_speed = self.max_speed
        if self.movement_params.radius == 0.25 and self.size != 0.25:
            self.movement_params.radius = self.size


# ---------------------------------------------------------------------------
# Simulation Scenario
# ---------------------------------------------------------------------------

@dataclass
class SimulationScenario:
    """A simulation scenario defining agents and events."""
    id: str
    name: str
    description: str
    duration: int  # steps
    time_step: float = 1.0  # seconds per step

    # Agent definitions
    agent_profiles: List[AgentProfile] = field(default_factory=list)
    agent_counts: Dict[str, int] = field(default_factory=dict)

    # Events
    events: List[Dict] = field(default_factory=list)

    # Environment
    environmental_conditions: Dict[str, Any] = field(default_factory=dict)

    # Goals
    goals: Dict[str, Any] = field(default_factory=dict)

    # NEW v2.0: Model selection for the whole scenario
    default_movement_model: str = "basic"


# ---------------------------------------------------------------------------
# Simulation Metrics
# ---------------------------------------------------------------------------

@dataclass
class SimulationMetrics:
    """Metrics collected during simulation."""
    timestamp: int
    agent_count: int
    agents_moving: int
    agents_waiting: int
    avg_speed: float
    avg_travel_time: float
    density_map: Dict[str, float] = field(default_factory=dict)
    congestion_points: List[Tuple] = field(default_factory=list)
    social_interactions: int = 0
    evacuation_progress: float = 0.0
    # NEW v2.0
    agents_queuing: int = 0
    avg_density: float = 0.0
    flow_rate: float = 0.0  # agents per second through exits
    # NEW v1.3.0
    agents_incapacitated: int = 0     # FED >= 1.0
    max_fed: float = 0.0              # highest FED value among all agents
    avg_smoke_exposure: float = 0.0   # average smoke_level experienced


# ---------------------------------------------------------------------------
# BIMAgent
# ---------------------------------------------------------------------------

class BIMAgent(Agent):
    """Base agent class for BIM simulation with JuPedSim-inspired movement."""

    def __init__(self, unique_id: int, model: 'BIMSimulationModel', profile: AgentProfile):
        super().__init__(unique_id, model)
        self.profile = profile
        self.state = AgentState.IDLE
        self.position = np.array([0.0, 0.0, 0.0])
        self.velocity = np.array([0.0, 0.0, 0.0])
        self.destination = None
        self.path = []
        self.current_space = None
        self.traveled_distance = 0.0
        self.travel_time = 0.0
        self.waiting_time = 0.0
        self.interactions = 0
        self.state_history = []
        self.position_history = []
        self.speed_history = []

        # Current speed (can vary)
        self.current_speed = profile.base_speed

        # Group behavior (NEW v1.3.0)
        self.group_members: List['BIMAgent'] = []   # filled by model._link_groups()
        self.is_group_leader: bool = False

        # NEW v2.0: Movement model instance
        self._movement_model: PedestrianModel = get_model(profile.movement_model)
        self._movement_params: AgentMovementParams = copy.deepcopy(profile.movement_params)

        # NEW v2.0: Journey tracking
        self._journey_index: int = 0
        if profile.journey:
            profile.journey.reset()

        # NEW v2.0: Waiting state
        self._wait_timer: float = 0.0
        self._flow_wait: bool = False

        # NEW v1.3.0: FED / smoke tracking
        self.fed: float = 0.0              # Fractional Effective Dose (0 = safe, >=1 = incapacitated)
        self.is_incapacitated: bool = False
        self.smoke_exposure: float = 0.0   # cumulative smoke exposure (0–1 per step)

    def step(self):
        """Execute one step of the agent."""
        current_step = getattr(getattr(self, 'model', None), '_step_count', 0)
        if hasattr(self.model, 'schedule') and hasattr(self.model.schedule, 'steps'):
            current_step = self.model.schedule.steps

        self.state_history.append({
            "step": current_step,
            "state": self.state.value,
            "position": tuple(self.position),
            "speed": float(self.current_speed)
        })
        self.position_history.append(tuple(self.position))
        self.speed_history.append(float(self.current_speed))

        # NEW v2.0: Journey takes precedence over simple state machine
        if self.profile.journey and self._journey_index < len(self.profile.journey.stages):
            self._process_journey()
            return

        # Legacy behavior based on state
        if self.state == AgentState.IDLE:
            self._behavior_idle()
        elif self.state == AgentState.MOVING:
            self._behavior_moving()
        elif self.state == AgentState.WAITING:
            self._behavior_waiting()
        elif self.state == AgentState.WORKING:
            self._behavior_working()
        elif self.state == AgentState.EVACUATING:
            self._behavior_evacuating()
        elif self.state == AgentState.INTERACTING:
            self._behavior_interacting()
        elif self.state == AgentState.QUEUING:
            self._behavior_queuing()

    # ------------------------------------------------------------------
    # Journey processing (NEW v2.0)
    # ------------------------------------------------------------------

    def _process_journey(self):
        """Process the current journey stage."""
        journey = self.profile.journey
        stage = journey.stages[self._journey_index]
        result = stage.update(self, self.model.time_step)

        if result.state == StageState.COMPLETED and result.next_stage:
            self._journey_index += 1
            if self._journey_index >= len(journey.stages):
                # Journey complete
                self.state = AgentState.IDLE
                self.velocity = np.zeros(3)
            else:
                # Reset next stage and continue
                journey.stages[self._journey_index].reset()
                self._process_journey()
            return

        if result.state == StageState.ACTIVE:
            if result.wait_duration > 0:
                self.state = AgentState.WAITING
                self._wait_timer = result.wait_duration
                self.velocity = np.zeros(3)
            elif result.flow_limit < float('inf'):
                self.state = AgentState.QUEUING
            elif result.destination:
                self.destination = result.destination
                self.state = AgentState.MOVING
                self._behavior_moving()
            elif result.metadata.get("direct_target") is not None:
                target = np.array(result.metadata["direct_target"])
                self._move_toward_point(target)
            return

    # ------------------------------------------------------------------
    # Behaviors
    # ------------------------------------------------------------------

    def _behavior_idle(self):
        """Behavior when idle - decide next action."""
        current_time = self.model.current_time
        if current_time in self.profile.schedule:
            activity = self.profile.schedule[current_time]
            if activity == "move" and self.destination:
                self.state = AgentState.MOVING
                self._plan_path()
            elif activity == "work":
                self.state = AgentState.WORKING
            elif activity == "evacuate":
                self.state = AgentState.EVACUATING
                self._set_evacuation_destination()
        else:
            if random.random() < 0.1 and self.destination:
                self.state = AgentState.MOVING
                self._plan_path()

    def _behavior_moving(self):
        """Behavior when moving along path using pedestrian model."""
        if not self.path and not self.destination:
            self.state = AgentState.IDLE
            return

        # If no path, plan one
        if not self.path and self.destination:
            self._plan_path()
            if not self.path:
                return

        # Get target waypoint
        target = np.array(self.path[0]) if self.path else np.array([0.0, 0.0, 0.0])
        direction = target - self.position
        distance = np.linalg.norm(direction)

        if distance < 0.5:  # Reached waypoint
            self.path.pop(0)
            if not self.path:
                self.state = AgentState.IDLE
                self.velocity = np.zeros(3)
                return
            target = np.array(self.path[0])
            direction = target - self.position
            distance = np.linalg.norm(direction)

        # Normalize desired direction
        desired_direction = direction / (distance + EPS)

        # NEW v2.0: Use pedestrian dynamics model for acceleration
        acceleration = self._compute_acceleration(desired_direction)

        # NEW v1.3.0: Group cohesion — add spring force toward group leader/centroid
        if self.group_members and not self.is_group_leader and self.profile.group_behavior:
            gb = self.profile.group_behavior
            if gb.leader_id is not None:
                leader = self.model._agent_registry.get(gb.leader_id)
                if leader and leader != self:
                    cohesion_dir = leader.position - self.position
                    cohesion_dist = np.linalg.norm(cohesion_dir)
                    if cohesion_dist > gb.separation_distance:
                        cohesion_force = gb.cohesion_strength * (cohesion_dir / (cohesion_dist + EPS))
                        acceleration = acceleration + cohesion_force

        # Integrate velocity
        self.velocity += acceleration * self.model.time_step

        # Clamp speed
        speed = np.linalg.norm(self.velocity)
        max_speed = self._movement_params.max_speed

        # Density-dependent speed reduction (Weidmann-like)
        nearby = self._count_nearby_agents(2.0)
        if nearby > 0:
            # Simple density-speed relationship: v = v0 * exp(-alpha * density)
            # Approximate local density from nearby count
            local_density = nearby / (np.pi * 2.0 * 2.0)  # agents per m² in 2m radius
            max_speed *= max(0.1, math.exp(-0.5 * local_density))

        # NEW v1.3.0: Smoke visibility — reduce speed based on local smoke level (Jin 1978)
        if self.current_space:
            fire_zone = self.model._fire_zones.get(self.current_space)
            if fire_zone:
                smoke_level = fire_zone.get("smoke_level", 0.0)
                self.smoke_exposure = smoke_level
                if smoke_level > 0:
                    # visibility drops proportionally to smoke_level
                    visibility = SMOKE_MAX_VISIBILITY * (1.0 - smoke_level)
                    speed_ratio = max(SMOKE_MIN_SPEED_RATIO,
                                      min(1.0, (visibility / SMOKE_MAX_VISIBILITY) ** 0.5))
                    max_speed *= speed_ratio

        # NEW v1.3.0: FED-based speed reduction (weakened agent due to CO exposure)
        if self.fed > 0:
            fed_factor = max(FED_SPEED_FACTOR, 1.0 - self.fed * (1.0 - FED_SPEED_FACTOR))
            max_speed *= fed_factor

        if speed > max_speed:
            self.velocity = (self.velocity / speed) * max_speed

        # Update position
        new_position = self.position + self.velocity * self.model.time_step

        # Wall collision constraint (soft)
        new_position = self._apply_wall_constraint(new_position)

        self.position = new_position
        self.current_speed = float(np.linalg.norm(self.velocity))
        self.traveled_distance += self.current_speed * self.model.time_step
        self.travel_time += self.model.time_step

        self._update_current_space()

        # NEW v1.3.0: FED accumulation in fire zones
        if self.current_space and not self.is_incapacitated:
            fire_zone = self.model._fire_zones.get(self.current_space)
            if fire_zone:
                hazard = fire_zone.get("hazard_intensity", 0.0)
                self.fed += hazard * FED_RATE_COEFFICIENT * self.model.time_step
                if self.fed >= 1.0:
                    self.is_incapacitated = True
                    self.state = AgentState.DISABLED
                    self.velocity = np.zeros(3)
                    logger.debug(f"Agent {self.unique_id} incapacitated (FED={self.fed:.2f})")

    def _behavior_waiting(self):
        """Behavior when waiting."""
        self.waiting_time += self.model.time_step
        self.velocity = np.zeros(3)
        self._wait_timer -= self.model.time_step

        if self._wait_timer <= 0:
            self.state = AgentState.IDLE

    def _behavior_queuing(self):
        """Behavior when queuing (NEW v2.0)."""
        self.velocity = np.zeros(3)
        # Agent remains in place, waiting for flow permission
        # The FlowLimitStage handles timing via the model's claim_flow_time

    def _behavior_working(self):
        """Behavior when working."""
        self.velocity = np.zeros(3)
        if random.random() < 0.01:
            self.state = AgentState.IDLE

    def _behavior_evacuating(self):
        """Behavior during evacuation."""
        if not self.destination:
            self._set_evacuation_destination()
        if not self.path:
            self._plan_path()
        if self.path:
            # Move faster during evacuation
            self._movement_params.desired_speed = min(self.profile.max_speed * 1.2, 2.0)
            self._behavior_moving()
        else:
            self.state = AgentState.DISABLED
            self.model.evacuated_agents += 1

    def _behavior_interacting(self):
        """Behavior when interacting with other agents."""
        self.velocity = np.zeros(3)
        self.interactions += 1
        if random.random() < 0.1:
            self.state = AgentState.IDLE

    # ------------------------------------------------------------------
    # NEW v2.0: Physics-based movement
    # ------------------------------------------------------------------

    def _compute_acceleration(self, desired_direction: np.ndarray) -> np.ndarray:
        """Compute acceleration using the selected pedestrian dynamics model."""
        neighbors = self._get_neighbor_info()
        walls = self._get_nearby_walls()

        return self._movement_model.compute_velocity_change(
            agent_position=self.position,
            agent_velocity=self.velocity,
            params=self._movement_params,
            desired_direction=desired_direction,
            neighbors=neighbors,
            walls=walls,
            time_step=self.model.time_step
        )

    def _get_neighbor_info(self) -> List[Tuple[np.ndarray, np.ndarray, AgentMovementParams]]:
        """Get nearby agents as (position, velocity, params) tuples."""
        neighbors = []
        vision = self._movement_params.radius * 4 + 2.0  # neighbor search radius
        for agent in self.model._get_all_agents():
            if agent != self:
                dist = np.linalg.norm(self.position - agent.position)
                if dist < vision:
                    # Use agent's own params if it has them, else default
                    other_params = getattr(agent, '_movement_params', AgentMovementParams(radius=agent.profile.size))
                    neighbors.append((agent.position.copy(), agent.velocity.copy(), other_params))
        return neighbors

    def _get_nearby_walls(self) -> List[WallSegment]:
        """Get wall segments near the agent."""
        return self.model._get_wall_segments_near(self.position, radius=3.0)

    def _apply_wall_constraint(self, position: np.ndarray) -> np.ndarray:
        """Soft constraint: push agent away from walls if it overlaps."""
        walls = self.model._get_wall_segments_near(position, radius=1.0)
        correction = np.zeros(3)
        for wall in walls:
            nearest = self._movement_model._nearest_point_on_segment(position, wall)
            diff = position - nearest
            dist = np.linalg.norm(diff) + EPS
            if dist < self._movement_params.radius:
                # Push out along normal
                push = (self._movement_params.radius - dist) * (diff / dist)
                correction += push
        return position + correction

    def _move_toward_point(self, target: np.ndarray):
        """Direct steering toward a point (used by DirectSteeringStage)."""
        direction = target - self.position
        distance = np.linalg.norm(direction)
        if distance < 0.5:
            return
        desired_direction = direction / distance
        acceleration = self._compute_acceleration(desired_direction)
        self.velocity += acceleration * self.model.time_step
        speed = np.linalg.norm(self.velocity)
        if speed > self._movement_params.max_speed:
            self.velocity = (self.velocity / speed) * self._movement_params.max_speed
        self.position = self.position + self.velocity * self.model.time_step
        self._update_current_space()

    # ------------------------------------------------------------------
    # Path planning
    # ------------------------------------------------------------------

    def _plan_path(self):
        """Plan path to destination using spatial graph."""
        if not self.destination or not self.model.spatial_engine:
            return
        current_space = self.current_space
        if not current_space:
            return
        path = self.model.spatial_engine.find_shortest_path(current_space, self.destination)
        if path:
            self.path = []
            for node_id in path[1:]:
                space_id = node_id.replace("space_", "")
                if space_id in self.model.bim_model.spaces:
                    space = self.model.bim_model.spaces[space_id]
                    if space.center:
                        self.path.append(space.center)

    def _set_evacuation_destination(self):
        """Set nearest exit as destination during evacuation."""
        if not self.model.bim_model:
            return
        exits = [
            s for s in self.model.bim_model.spaces.values()
            if s.category in ["entrance", "exit", "lobby", "public"]
        ]
        if exits:
            closest = min(exits, key=lambda e: self._distance_to(e.center))
            self.destination = closest.id

    def _update_current_space(self):
        """Update which space the agent is currently in."""
        if not self.model.bim_model:
            return
        closest_space = None
        closest_dist = float('inf')
        for space in self.model.bim_model.spaces.values():
            if space.center:
                dist = self._distance_to(space.center)
                if dist < closest_dist:
                    closest_dist = dist
                    closest_space = space.id
        if closest_space:
            self.current_space = closest_space

    def _distance_to(self, point: Optional[Tuple]) -> float:
        """Calculate distance to a point."""
        if not point:
            return float('inf')
        return float(np.linalg.norm(self.position - np.array(point)))

    def _count_nearby_agents(self, radius: float) -> int:
        """Count agents within radius."""
        count = 0
        for agent in self.model._get_all_agents():
            if agent != self:
                dist = np.linalg.norm(self.position - agent.position)
                if dist < radius:
                    count += 1
        return count

    def get_metrics(self) -> Dict:
        """Get agent metrics."""
        return {
            "id": self.unique_id,
            "type": self.profile.agent_type.value,
            "role": self.profile.role.value if self.profile.role else None,
            "state": self.state.value,
            "position": tuple(self.position),
            "current_space": self.current_space,
            "traveled_distance": self.traveled_distance,
            "travel_time": self.travel_time,
            "waiting_time": self.waiting_time,
            "interactions": self.interactions,
            "current_speed": self.current_speed,
            "movement_model": self.profile.movement_model,
            # NEW v1.3.0
            "fed": round(self.fed, 4),
            "is_incapacitated": self.is_incapacitated,
            "smoke_exposure": round(self.smoke_exposure, 4),
            "group_id": self.profile.group_id,
            "is_group_leader": self.is_group_leader,
        }


# ---------------------------------------------------------------------------
# BIMSimulationModel
# ---------------------------------------------------------------------------

class BIMSimulationModel(Model):
    """Mesa model for BIM agent-based simulation with JuPedSim-inspired features."""

    def __init__(
        self,
        bim_model: BIMModel,
        spatial_engine: SpatialIntelligenceEngine,
        scenario: SimulationScenario,
        width: float = 200.0,
        height: float = 200.0
    ):
        super().__init__()
        self.bim_model = bim_model
        self.spatial_engine = spatial_engine
        self.scenario = scenario
        self.width = width
        self.height = height

        # Time tracking
        self.current_time = 0.0
        self.time_step = scenario.time_step
        self.max_steps = scenario.duration

        # Spatial setup
        self.space = ContinuousSpace(width, height, torus=False)

        # Schedule
        self.schedule = SimultaneousActivation(self)

        # Agent tracking
        self._agent_registry: Dict[int, BIMAgent] = {}
        self.evacuated_agents = 0
        self.next_agent_id = 0

        # Events
        self.pending_events = sorted(scenario.events, key=lambda e: e.get("time", 0))
        self.completed_events = []

        # Metrics collection
        self.metrics_history: List[SimulationMetrics] = []
        self.social_interactions = 0
        self.congestion_events = []

        # Density map
        self.density_map: Dict[str, int] = {}

        # NEW v2.0: Wall segment cache + NEW v1.3.0: uniform grid spatial index
        self._wall_segments: List[WallSegment] = []
        self._wall_grid_index: UniformGridSpatialIndex = UniformGridSpatialIndex(cell_size=5.0)
        self._build_wall_cache()

        # NEW v2.0: Flow limitation tracking (space_id -> next available time)
        self._flow_limits: Dict[str, float] = {}

        # NEW v1.3.0: Fire zone tracking (space_id -> {hazard_intensity, smoke_level, visibility})
        self._fire_zones: Dict[str, Dict[str, float]] = {}

        # NEW v1.3.0: Blocked edges cache for unblock_path (edge_data per removed edge)
        self._blocked_edges: Dict[str, List[Tuple]] = {}  # space_id -> [(u, v, data), ...]

        # Data collector
        self.datacollector = DataCollector(
            model_reporters={
                "Agent Count": lambda m: len(m._get_all_agents()),
                "Moving": self._count_moving_agents,
                "Waiting": self._count_waiting_agents,
                "Evacuated": lambda m: m.evacuated_agents,
                "Queuing": self._count_queuing_agents,
                "Avg Speed": self._get_avg_speed,
                # NEW v1.3.0
                "Incapacitated": lambda m: sum(1 for a in m._get_all_agents() if a.is_incapacitated),
                "Max FED": lambda m: max((a.fed for a in m._get_all_agents()), default=0.0),
            },
            agent_reporters={
                "State": lambda a: a.state.value,
                "Speed": lambda a: a.current_speed,
                "Distance": lambda a: a.traveled_distance,
                "Space": lambda a: a.current_space,
                "Model": lambda a: a.profile.movement_model,
                # NEW v1.3.0
                "FED": lambda a: a.fed,
                "GroupLeader": lambda a: a.is_group_leader,
            }
        )

        # Initialize agents
        self._initialize_agents()

        # NEW v1.3.0: Link group members after all agents are created
        self._link_groups()

        logger.info(f"Simulation model initialized: {len(self._get_all_agents())} agents, model={scenario.default_movement_model}")

    # ------------------------------------------------------------------
    # Wall cache (NEW v2.0)
    # ------------------------------------------------------------------

    def _build_wall_cache(self):
        """Precompute wall segments from BIM elements."""
        if not self.bim_model:
            return

        walls = []
        for elem in self.bim_model.elements.values():
            if elem.category == ElementCategory.WALL and elem.bounds:
                (min_x, min_y, min_z), (max_x, max_y, max_z) = elem.bounds
                # Create edge segments from the bounding box (2D approximation)
                edges = [
                    (np.array([min_x, min_y, min_z]), np.array([max_x, min_y, min_z])),
                    (np.array([max_x, min_y, min_z]), np.array([max_x, max_y, min_z])),
                    (np.array([max_x, max_y, min_z]), np.array([min_x, max_y, min_z])),
                    (np.array([min_x, max_y, min_z]), np.array([min_x, min_y, min_z])),
                ]
                for p1, p2 in edges:
                    walls.append(WallSegment(p1, p2))

        self._wall_segments = walls
        # NEW v1.3.0: populate uniform grid spatial index for fast queries
        self._wall_grid_index.build(walls)
        logger.info(f"Wall cache built: {len(walls)} segments (spatial index ready)")

    def _get_wall_segments_near(self, position: np.ndarray, radius: float = 3.0) -> List[WallSegment]:
        """Return wall segments within radius of position.

        NEW v1.3.0: Uses UniformGridSpatialIndex for O(1) average cost instead
        of the previous O(N) linear scan. Falls back to brute-force if the index
        is empty (e.g. no BIM walls loaded).
        """
        if self._wall_segments:
            # Candidate set from grid (conservative — may contain segments slightly beyond radius)
            candidates = self._wall_grid_index.query(position, radius)
            # Exact filter
            nearby = []
            for wall in candidates:
                nearest = self._nearest_point_on_segment(position, wall)
                if np.linalg.norm(position - nearest) < radius + 0.5:
                    nearby.append(wall)
            return nearby
        return []

    @staticmethod
    def _nearest_point_on_segment(point: np.ndarray, wall: WallSegment) -> np.ndarray:
        t = np.dot(point - wall.p1, wall.unit)
        t = max(0.0, min(wall.length, t))
        return wall.p1 + t * wall.unit

    # ------------------------------------------------------------------
    # Flow limitation (NEW v2.0)
    # ------------------------------------------------------------------

    def claim_flow_time(self, space_id: str, max_flow_rate: float) -> Optional[float]:
        """
        Claim a time slot to pass through a flow-limited space.
        Returns the earliest allowed time, or None if cannot claim.
        """
        now = self.current_time
        interval = 1.0 / max_flow_rate if max_flow_rate > 0 else 0.0
        last_time = self._flow_limits.get(space_id, now - interval)
        next_time = max(now, last_time + interval)
        self._flow_limits[space_id] = next_time
        return next_time

    # ------------------------------------------------------------------
    # Agent initialization
    # ------------------------------------------------------------------

    def _initialize_agents(self):
        """Create agents based on scenario."""
        for profile in self.scenario.agent_profiles:
            count = self.scenario.agent_counts.get(profile.id, 1)
            for _ in range(count):
                self._create_agent(profile)

    def _link_groups(self):
        """
        NEW v1.3.0: Link agents that share the same group_id.
        The first agent created in the group becomes the leader.
        All group members get references to each other and share a GroupBehavior.
        """
        # Gather agents by group_id
        group_map: Dict[str, List['BIMAgent']] = {}
        for agent in self._get_all_agents():
            gid = agent.profile.group_id
            if gid:
                if gid not in group_map:
                    group_map[gid] = []
                group_map[gid].append(agent)

        for gid, members in group_map.items():
            if len(members) < 2:
                continue
            # First member is the leader
            leader = members[0]
            leader.is_group_leader = True

            # Create a shared GroupBehavior if one is not specified on the profile
            gb = members[0].profile.group_behavior or GroupBehavior(leader_id=leader.unique_id)
            gb.leader_id = leader.unique_id

            for member in members:
                member.group_members = members
                # Assign shared GroupBehavior to each member's profile
                member.profile.group_behavior = gb

            logger.info(f"Group '{gid}': {len(members)} agents linked, leader={leader.unique_id}")

    def _create_agent(self, profile: AgentProfile, position: Optional[Tuple] = None) -> BIMAgent:
        """Create a new agent in the simulation."""
        agent_id = self.next_agent_id
        self.next_agent_id += 1

        agent = BIMAgent(agent_id, self, profile)

        # Set initial position
        if position:
            agent.position = np.array(position, dtype=float)
        else:
            if self.bim_model and self.bim_model.spaces:
                space = random.choice(list(self.bim_model.spaces.values()))
                if space.center:
                    agent.position = np.array([
                        space.center[0] + random.uniform(-2, 2),
                        space.center[1] + random.uniform(-2, 2),
                        space.center[2] if len(space.center) > 2 else 0.0
                    ], dtype=float)
                    agent.current_space = space.id

        self.schedule.add(agent)
        self._agent_registry[agent_id] = agent

        try:
            self.space.place_agent(agent, (agent.position[0], agent.position[1]))
        except Exception:
            pass

        return agent

    # ------------------------------------------------------------------
    # Mesa-version-safe agent iterator
    # ------------------------------------------------------------------

    def _get_all_agents(self) -> list:
        """Return all active agents, compatible with Mesa 2.x and 3.x."""
        sched = self.schedule
        if hasattr(sched, 'agents'):
            try:
                return list(sched.agents)
            except Exception:
                pass
        if hasattr(sched, '_agents'):
            if isinstance(sched._agents, dict):
                return list(sched._agents.values())
            try:
                return list(sched._agents)
            except Exception:
                pass
        return list(self._agent_registry.values())

    # ------------------------------------------------------------------
    # Step
    # ------------------------------------------------------------------

    def step(self):
        """Execute one simulation step."""
        self.current_time += self.time_step
        self._process_events()
        self.schedule.step()
        self._update_density_map()
        self._detect_congestion()
        self._detect_interactions()
        self._collect_metrics()
        self.datacollector.collect(self)

    def _process_events(self):
        """Process scheduled events."""
        events_to_trigger = [
            e for e in self.pending_events
            if e.get("time", 0) <= self.current_time
        ]
        for event in events_to_trigger:
            self._trigger_event(event)
            self.pending_events.remove(event)
            self.completed_events.append(event)

    def _trigger_event(self, event: Dict):
        """Trigger a simulation event."""
        event_type = event.get("type", "")
        logger.info(f"Triggering event: {event_type} at time {self.current_time}")

        if event_type == "spawn_agents":
            self._event_spawn_agents(event)
        elif event_type == "evacuate":
            self._event_evacuate(event)
        elif event_type == "fire":
            self._event_fire(event)
        elif event_type == "block_path":
            self._event_block_path(event)
        elif event_type == "unblock_path":                # NEW v1.3.0
            self._event_unblock_path(event)
        elif event_type == "set_destination":
            self._event_set_destination(event)
        elif event_type == "set_journey":
            self._event_set_journey(event)
        elif event_type == "switch_model":
            self._event_switch_model(event)

    def _event_spawn_agents(self, event: Dict):
        profile_id = event.get("profile_id", "")
        count = event.get("count", 1)
        location = event.get("location")
        profile = next(
            (p for p in self.scenario.agent_profiles if p.id == profile_id),
            None
        )
        if profile:
            for _ in range(count):
                self._create_agent(profile, location)

    def _event_evacuate(self, event: Dict):
        for agent in self._get_all_agents():
            agent.state = AgentState.EVACUATING
            agent._set_evacuation_destination()

    def _event_fire(self, event: Dict):
        """
        NEW v1.3.0: Enhanced fire event — now also populates _fire_zones with
        hazard_intensity and smoke_level per affected space, enabling FED
        accumulation and visibility-based speed reduction.
        """
        location = event.get("location")
        spread_rate = event.get("spread_rate", 1.0)
        hazard_intensity = event.get("hazard_intensity", 0.8)  # 0–1
        smoke_level = event.get("smoke_level", 0.7)            # 0–1

        if location and self.spatial_engine and self.spatial_engine.spatial_graph:
            for node_id, node in self.spatial_engine.spatial_graph.nodes.items():
                dist = np.linalg.norm(np.array(node.center) - np.array(location))
                if dist < 10.0 * spread_rate:
                    node.attributes["on_fire"] = True
                    # Map node -> space_id (strip "space_" prefix)
                    space_id = node_id.replace("space_", "")
                    # Hazard decays with distance from fire origin
                    decay = max(0.1, 1.0 - dist / (10.0 * spread_rate + 1e-6))
                    self._fire_zones[space_id] = {
                        "hazard_intensity": hazard_intensity * decay,
                        "smoke_level": smoke_level * decay,
                        "visibility": SMOKE_MAX_VISIBILITY * (1.0 - smoke_level * decay),
                    }

        # Also penalize pathfinding through fire zones by increasing edge weight
        if self.spatial_engine and self.spatial_engine.spatial_graph:
            for space_id, zone in self._fire_zones.items():
                node_id = f"space_{space_id}"
                graph = self.spatial_engine.spatial_graph.network
                if graph.has_node(node_id):
                    # Increase weight of all edges to this node (smoke penalty)
                    smoke_penalty = 100.0 * zone["smoke_level"]
                    for u, v, data in list(graph.edges(node_id, data=True)):
                        data["weight"] = data.get("weight", 1.0) + smoke_penalty

    def _event_block_path(self, event: Dict):
        """
        NEW v1.3.0: Actually removes graph edges to/from the blocked node so
        pathfinding genuinely avoids the space. Edges are cached for later
        restoration via unblock_path. Also forces affected agents to re-plan.
        """
        space_id = event.get("space_id")
        if not space_id or not self.spatial_engine or not self.spatial_engine.spatial_graph:
            return
        node_id = f"space_{space_id}"
        graph = self.spatial_engine.spatial_graph.network
        if node_id not in graph:
            return

        # Mark as blocked in node attributes
        if node_id in self.spatial_engine.spatial_graph.nodes:
            self.spatial_engine.spatial_graph.nodes[node_id].attributes["blocked"] = True

        # Cache and remove all edges incident to this node
        removed_edges = []
        for u, v, data in list(graph.edges(node_id, data=True)):
            removed_edges.append((u, v, data))
        for u, v, _ in removed_edges:
            graph.remove_edge(u, v)
        self._blocked_edges[space_id] = removed_edges
        logger.info(f"Path blocked: {space_id} ({len(removed_edges)} edges removed)")

        # Force agents currently routing through this space to re-plan
        self._reroute_agents_avoiding(space_id)

    def _event_unblock_path(self, event: Dict):
        """
        NEW v1.3.0: Restore previously removed graph edges when a blocked space
        is cleared (door opened, fire suppressed, etc.).
        """
        space_id = event.get("space_id")
        if not space_id or not self.spatial_engine or not self.spatial_engine.spatial_graph:
            return
        node_id = f"space_{space_id}"

        # Unmark
        if node_id in self.spatial_engine.spatial_graph.nodes:
            self.spatial_engine.spatial_graph.nodes[node_id].attributes["blocked"] = False

        # Restore edges
        graph = self.spatial_engine.spatial_graph.network
        restored = 0
        for u, v, data in self._blocked_edges.pop(space_id, []):
            if not graph.has_edge(u, v):
                graph.add_edge(u, v, **data)
                restored += 1
        logger.info(f"Path unblocked: {space_id} ({restored} edges restored)")

        # Also remove from fire zones if present
        self._fire_zones.pop(space_id, None)

    def _reroute_agents_avoiding(self, blocked_space_id: str):
        """
        NEW v1.3.0: Clear the cached path for any agent that was planning to
        pass through a now-blocked space. Agents will re-plan on the next step.
        """
        for agent in self._get_all_agents():
            if agent.path:
                # Check if any path waypoint belongs to this space
                # Waypoints are space center tuples; we check if the closest
                # space to any waypoint is the blocked one.
                for waypoint in agent.path:
                    if self.bim_model and self.bim_model.spaces:
                        wp = np.array(waypoint)
                        for sid, space in self.bim_model.spaces.items():
                            if space.center:
                                dist = np.linalg.norm(wp - np.array(space.center))
                                if dist < 1.0 and sid == blocked_space_id:
                                    agent.path = []  # Force re-plan
                                    break

    def _event_set_destination(self, event: Dict):
        agent_filter = event.get("filter", {})
        destination = event.get("destination")
        for agent in self._get_all_agents():
            match = True
            for key, value in agent_filter.items():
                if getattr(agent.profile, key, None) != value:
                    match = False
                    break
            if match:
                agent.destination = destination
                agent.state = AgentState.IDLE

    # NEW v2.0
    def _event_set_journey(self, event: Dict):
        """Assign a journey to matching agents."""
        agent_filter = event.get("filter", {})
        journey = event.get("journey")
        if not journey:
            return
        for agent in self._get_all_agents():
            match = True
            for key, value in agent_filter.items():
                if getattr(agent.profile, key, None) != value:
                    match = False
                    break
            if match:
                agent.profile.journey = journey
                agent._journey_index = 0
                journey.reset()

    # NEW v2.0
    def _event_switch_model(self, event: Dict):
        """Switch movement model for agents."""
        agent_filter = event.get("filter", {})
        model_name = event.get("model", "basic")
        for agent in self._get_all_agents():
            match = True
            for key, value in agent_filter.items():
                if getattr(agent.profile, key, None) != value:
                    match = False
                    break
            if match:
                agent.profile.movement_model = model_name
                agent._movement_model = get_model(model_name)

    # ------------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------------

    def _update_density_map(self):
        self.density_map = {}
        for agent in self._get_all_agents():
            if agent.current_space:
                self.density_map[agent.current_space] = self.density_map.get(agent.current_space, 0) + 1

    def _detect_congestion(self):
        congestion_threshold = 3.0
        for space_id, count in self.density_map.items():
            if space_id in self.bim_model.spaces:
                space = self.bim_model.spaces[space_id]
                if space.area > 0:
                    density = count / space.area
                    if density > congestion_threshold:
                        self.congestion_events.append({
                            "step": self.schedule.steps,
                            "space_id": space_id,
                            "space_name": space.name,
                            "density": density,
                            "agent_count": count
                        })

    def _detect_interactions(self):
        interaction_distance = 2.0
        agents_list = self._get_all_agents()
        for i in range(len(agents_list)):
            for j in range(i + 1, len(agents_list)):
                a1 = agents_list[i]
                a2 = agents_list[j]
                dist = np.linalg.norm(a1.position - a2.position)
                if dist < interaction_distance:
                    if a1.profile.sociability > 0.3 and a2.profile.sociability > 0.3:
                        self.social_interactions += 1
                        if random.random() < 0.1:
                            a1.state = AgentState.INTERACTING
                            a2.state = AgentState.INTERACTING

    def _collect_metrics(self) -> SimulationMetrics:
        agents = self._get_all_agents()
        total_agents = len(agents)

        # Compute average density across all occupied spaces
        avg_density = 0.0
        if self.density_map:
            densities = []
            for sid, count in self.density_map.items():
                if sid in self.bim_model.spaces:
                    area = self.bim_model.spaces[sid].area
                    if area > 0:
                        densities.append(count / area)
            if densities:
                avg_density = sum(densities) / len(densities)

        # Compute flow rate (evacuated agents per second over last step)
        flow_rate = self.evacuated_agents / max(self.current_time, 1.0)

        # NEW v1.3.0: FED metrics
        incapacitated = sum(1 for a in agents if a.is_incapacitated)
        max_fed = max((a.fed for a in agents), default=0.0)
        avg_smoke = sum(a.smoke_exposure for a in agents) / total_agents if total_agents else 0.0

        metrics = SimulationMetrics(
            timestamp=getattr(self.schedule, 'steps', int(self.current_time)),
            agent_count=total_agents,
            agents_moving=sum(1 for a in agents if a.state == AgentState.MOVING),
            agents_waiting=sum(1 for a in agents if a.state == AgentState.WAITING),
            agents_queuing=sum(1 for a in agents if a.state == AgentState.QUEUING),
            avg_speed=sum(a.current_speed for a in agents) / total_agents if total_agents else 0,
            avg_travel_time=sum(a.travel_time for a in agents) / total_agents if total_agents else 0,
            density_map={
                sid: count / self.bim_model.spaces[sid].area
                for sid, count in self.density_map.items()
                if sid in self.bim_model.spaces and self.bim_model.spaces[sid].area > 0
            },
            social_interactions=self.social_interactions,
            evacuation_progress=(self.evacuated_agents / total_agents * 100) if total_agents else 0,
            avg_density=avg_density,
            flow_rate=flow_rate,
            # NEW v1.3.0
            agents_incapacitated=incapacitated,
            max_fed=round(max_fed, 4),
            avg_smoke_exposure=round(avg_smoke, 4),
        )
        self.metrics_history.append(metrics)
        return metrics

    def _count_moving_agents(self) -> int:
        return sum(1 for a in self._get_all_agents() if a.state == AgentState.MOVING)

    def _count_waiting_agents(self) -> int:
        return sum(1 for a in self._get_all_agents() if a.state == AgentState.WAITING)

    def _count_queuing_agents(self) -> int:
        return sum(1 for a in self._get_all_agents() if a.state == AgentState.QUEUING)

    def _get_avg_speed(self) -> float:
        agents = self._get_all_agents()
        return sum(a.current_speed for a in agents) / len(agents) if agents else 0

    def get_agent_metrics(self) -> List[Dict]:
        return [agent.get_metrics() for agent in self._get_all_agents()]

    def get_space_occupancy(self) -> Dict[str, Dict]:
        occupancy = {}
        for space_id, space in self.bim_model.spaces.items():
            agent_count = self.density_map.get(space_id, 0)
            capacity = getattr(space, 'capacity', max(1, space.area / 2.0))
            density = agent_count / space.area if space.area > 0 else 0
            capacity_ratio = agent_count / capacity if capacity > 0 else 0
            occupancy[space_id] = {
                "space_name": space.name,
                "agent_count": agent_count,
                "density": density,
                "capacity": capacity,
                "capacity_ratio": capacity_ratio,
                "is_overcrowded": capacity_ratio > 1.0
            }
        return occupancy

    def get_current_metrics(self) -> SimulationMetrics:
        if self.metrics_history:
            return self.metrics_history[-1]
        return SimulationMetrics(timestamp=0, agent_count=0, agents_moving=0,
                                 agents_waiting=0, avg_speed=0, avg_travel_time=0)


# ---------------------------------------------------------------------------
# SimulationEngine
# ---------------------------------------------------------------------------

class SimulationEngine:
    """Main simulation engine managing simulation runs."""

    def __init__(self):
        self.current_model: Optional[BIMSimulationModel] = None
        self.is_running = False
        self.current_step = 0
        self.scenarios: Dict[str, SimulationScenario] = {}
        self.on_step_callbacks: List[Callable] = []
        self.on_complete_callbacks: List[Callable] = []

        # NEW v2.0: Batch results
        self.batch_results: List[Dict] = []

    def create_scenario(self, name: str, description: str, duration: int, **kwargs) -> SimulationScenario:
        scenario = SimulationScenario(
            id=str(uuid.uuid4()),
            name=name,
            description=description,
            duration=duration,
            **kwargs
        )
        self.scenarios[scenario.id] = scenario
        return scenario

    def add_agent_profile(
        self,
        scenario: SimulationScenario,
        name: str,
        agent_type: AgentType,
        role: Optional[HumanRole] = None,
        count: int = 1,
        **kwargs
    ) -> AgentProfile:
        """Add an agent profile to a scenario."""
        # NEW v2.0: allow movement_model and journey in kwargs
        movement_model = kwargs.pop("movement_model", scenario.default_movement_model)
        journey = kwargs.pop("journey", None)

        profile = AgentProfile(
            id=str(uuid.uuid4()),
            name=name,
            agent_type=agent_type,
            role=role,
            movement_model=movement_model,
            journey=journey,
            **kwargs
        )
        scenario.agent_profiles.append(profile)
        scenario.agent_counts[profile.id] = count
        return profile

    def add_event(self, scenario: SimulationScenario, time: float, event_type: str, **kwargs):
        event = {
            "time": time,
            "type": event_type,
            **kwargs
        }
        scenario.events.append(event)

    def initialize_simulation(
        self,
        bim_model: BIMModel,
        spatial_engine: SpatialIntelligenceEngine,
        scenario: SimulationScenario
    ) -> BIMSimulationModel:
        logger.info(f"Initializing simulation: {scenario.name}")
        self.current_model = BIMSimulationModel(
            bim_model=bim_model,
            spatial_engine=spatial_engine,
            scenario=scenario
        )
        self.current_step = 0
        self.is_running = False
        return self.current_model

    def step(self):
        if not self.current_model or not self.is_running:
            return
        self.current_model.step()
        self.current_step += 1
        for callback in self.on_step_callbacks:
            callback(self.current_step, self.current_model)
        if self.current_step >= self.current_model.max_steps:
            self.stop()

    def run(self, steps: Optional[int] = None):
        if not self.current_model:
            logger.error("No simulation model initialized")
            return
        self.is_running = True
        target_steps = steps or self.current_model.max_steps
        while self.is_running and self.current_step < target_steps:
            self.step()
        if not self.is_running:
            for callback in self.on_complete_callbacks:
                callback(self.current_model)

    def start(self):
        self.is_running = True
        logger.info("Simulation started")

    def pause(self):
        self.is_running = False
        logger.info("Simulation paused")

    def stop(self):
        self.is_running = False
        self.current_step = 0
        logger.info("Simulation stopped")

    def reset(self):
        self.is_running = False
        self.current_step = 0
        if self.current_model:
            scenario = self.current_model.scenario
            bim_model = self.current_model.bim_model
            spatial_engine = self.current_model.spatial_engine
            self.current_model = BIMSimulationModel(
                bim_model=bim_model,
                spatial_engine=spatial_engine,
                scenario=scenario
            )
        logger.info("Simulation reset")

    # NEW v2.0: Batch simulation
    def run_batch(
        self,
        bim_model: BIMModel,
        spatial_engine: SpatialIntelligenceEngine,
        scenario: SimulationScenario,
        runs: int = 10,
        seeds: Optional[List[int]] = None
    ) -> List[Dict]:
        """
        Run the same scenario multiple times with different seeds for statistical robustness.
        Returns a list of result dicts.
        """
        if seeds is None:
            seeds = list(range(42, 42 + runs))
        if len(seeds) < runs:
            seeds = seeds + list(range(seeds[-1] + 1, seeds[-1] + 1 + runs - len(seeds))) if seeds else list(range(42, 42 + runs))

        self.batch_results = []
        for i, seed in enumerate(seeds[:runs]):
            logger.info(f"Batch run {i+1}/{runs} with seed {seed}")
            random.seed(seed)
            np.random.seed(seed)
            model = self.initialize_simulation(bim_model, spatial_engine, scenario)
            self.run()
            results = self.get_results()
            results["seed"] = seed
            results["run_index"] = i + 1
            self.batch_results.append(results)
            self.reset()

        logger.info(f"Batch simulation complete: {len(self.batch_results)} runs")
        return self.batch_results

    def get_batch_summary(self) -> Dict:
        """Compute statistics across batch runs."""
        if not self.batch_results:
            return {}

        evacuation_times = []
        avg_speeds = []
        total_interactions = []
        congestion_counts = []

        for r in self.batch_results:
            final = r.get("final_metrics", {})
            avg_speeds.append(final.get("avg_speed", 0))
            total_interactions.append(r.get("total_interactions", 0))
            congestion_counts.append(len(r.get("congestion_events", [])))
            # Approximate evacuation time from last evacuated agent
            hist = r.get("metrics_history", [])
            if hist:
                evacuation_times.append(len(hist) * r["scenario"].get("duration", 0) / max(len(hist), 1))

        import statistics
        summary = {
            "runs": len(self.batch_results),
            "avg_speed_mean": statistics.mean(avg_speeds) if avg_speeds else 0,
            "avg_speed_std": statistics.stdev(avg_speeds) if len(avg_speeds) > 1 else 0,
            "total_interactions_mean": statistics.mean(total_interactions) if total_interactions else 0,
            "congestion_events_mean": statistics.mean(congestion_counts) if congestion_counts else 0,
            "evacuation_time_mean": statistics.mean(evacuation_times) if evacuation_times else 0,
            "evacuation_time_std": statistics.stdev(evacuation_times) if len(evacuation_times) > 1 else 0,
        }
        return summary

    def get_results(self) -> Dict:
        if not self.current_model:
            return {}
        model = self.current_model
        return {
            "scenario": {
                "name": model.scenario.name,
                "description": model.scenario.description,
                "duration": model.scenario.duration
            },
            "final_metrics": model.get_current_metrics().__dict__,
            "metrics_history": [m.__dict__ for m in model.metrics_history],
            "agent_metrics": model.get_agent_metrics(),
            "space_occupancy": model.get_space_occupancy(),
            "congestion_events": model.congestion_events,
            "total_interactions": model.social_interactions,
            "evacuated_agents": model.evacuated_agents,
            "completed_events": model.completed_events,
            "movement_models_used": list(set(a.profile.movement_model for a in model._get_all_agents())),
        }

    def on_step(self, callback: Callable):
        self.on_step_callbacks.append(callback)

    def on_complete(self, callback: Callable):
        self.on_complete_callbacks.append(callback)

    def list_available_models(self) -> List[str]:
        """Return available pedestrian dynamics model names."""
        return list_models()


# ---------------------------------------------------------------------------
# Preset scenarios (updated for v2.0)
# ---------------------------------------------------------------------------

class ScenarioPresets:
    """Predefined simulation scenarios."""

    @staticmethod
    def office_scenario() -> SimulationScenario:
        engine = SimulationEngine()
        scenario = engine.create_scenario(
            name="Office Daily Operations",
            description="Simulate a typical day in an office building",
            duration=28800,
            default_movement_model="collision_free_speed"
        )
        profile = engine.add_agent_profile(
            scenario=scenario,
            name="Office Worker",
            agent_type=AgentType.HUMAN,
            role=HumanRole.OFFICE_WORKER,
            count=50,
            base_speed=1.2,
            sociability=0.4,
            schedule={0: "move", 14400: "move", 18000: "move", 25200: "move"}
        )
        engine.add_event(scenario, time=0, event_type="spawn_agents", profile_id=profile.id, count=50)
        engine.add_event(scenario, time=14400, event_type="set_destination", filter={"profile.agent_type": AgentType.HUMAN})
        engine.add_event(scenario, time=20000, event_type="evacuate")
        return scenario

    @staticmethod
    def evacuation_scenario() -> SimulationScenario:
        engine = SimulationEngine()
        scenario = engine.create_scenario(
            name="Emergency Evacuation",
            description="Simulate emergency evacuation of the building",
            duration=1800,
            default_movement_model="social_force"
        )
        occupants = engine.add_agent_profile(
            scenario=scenario,
            name="Occupant",
            agent_type=AgentType.HUMAN,
            role=HumanRole.VISITOR,
            count=100,
            base_speed=1.4,
            risk_tolerance=0.3,
            schedule={0: "evacuate"}
        )
        engine.add_event(scenario, time=0, event_type="evacuate")
        engine.add_event(scenario, time=120, event_type="fire", location=[0, 0, 0], spread_rate=0.5)
        return scenario

    @staticmethod
    def hospital_scenario() -> SimulationScenario:
        engine = SimulationEngine()
        scenario = engine.create_scenario(
            name="Hospital Operations",
            description="Simulate hospital patient and staff movement",
            duration=86400,
            default_movement_model="anticipation_velocity"
        )
        staff = engine.add_agent_profile(
            scenario=scenario,
            name="Medical Staff",
            agent_type=AgentType.HUMAN,
            role=HumanRole.STAFF,
            count=30,
            base_speed=1.5,
            sociability=0.6
        )
        patients = engine.add_agent_profile(
            scenario=scenario,
            name="Patient",
            agent_type=AgentType.HUMAN,
            role=HumanRole.PATIENT,
            count=40,
            base_speed=0.8,
            needs_accessible=True,
            can_use_stairs=False
        )
        robots = engine.add_agent_profile(
            scenario=scenario,
            name="Service Robot",
            agent_type=AgentType.AUTONOMOUS,
            count=5,
            base_speed=1.0,
            size=0.3
        )
        return scenario

    @staticmethod
    def university_scenario() -> SimulationScenario:
        engine = SimulationEngine()
        scenario = engine.create_scenario(
            name="University Class Transitions",
            description="Simulate student movement between classes",
            duration=10800,
            default_movement_model="collision_free_speed"
        )
        students = engine.add_agent_profile(
            scenario=scenario,
            name="Student",
            agent_type=AgentType.HUMAN,
            role=HumanRole.STUDENT,
            count=200,
            base_speed=1.3,
            sociability=0.7,
            schedule={0: "move", 2700: "move", 5400: "move", 8100: "move"}
        )
        return scenario

    @staticmethod
    def queueing_scenario() -> SimulationScenario:
        """NEW v2.0: Scenario demonstrating flow limits and queueing."""
        engine = SimulationEngine()
        scenario = engine.create_scenario(
            name="Queue Formation at Narrow Door",
            description="Demonstrate queueing behavior with flow-limited door",
            duration=600,
            default_movement_model="social_force"
        )
        # Create a journey: wait in lobby -> flow through door -> exit
        from engine.journey_system import Journey, WaypointStage, FlowLimitStage, WaitStage
        journey = Journey(
            id=str(uuid.uuid4()),
            name="Lobby to Exit",
            stages=[
                WaitStage("lobby", 5.0, "Wait in lobby"),
                FlowLimitStage("exit", max_flow_rate=1.5, name="Pass through door"),
                WaypointStage("outside", "Exit building"),
            ]
        )
        engine.add_agent_profile(
            scenario=scenario,
            name="Visitor",
            agent_type=AgentType.HUMAN,
            role=HumanRole.VISITOR,
            count=30,
            base_speed=1.2,
            journey=journey,
            patience=0.8
        )
        return scenario

    @staticmethod
    def fire_evacuation_scenario() -> SimulationScenario:
        """
        NEW v1.3.0: Fire evacuation with FED tracking + smoke effects.

        Demonstrates the full fire-safety simulation stack:
        - Social Force Model for realistic crowd pushing during evacuation
        - Fire breaks out at t=60s, generating smoke and hazard
        - Agents in fire zones slow down (Jin 1978 visibility model)
        - FED accumulates; incapacitated agents can no longer move
        - Corridor is blocked at t=60s; agents must find alternate routes
        - Corridor unblocked at t=180s (fire suppressed)
        - Metrics: evacuation_progress, agents_incapacitated, max_fed
        """
        engine = SimulationEngine()
        scenario = engine.create_scenario(
            name="Fire Evacuation with FED Tracking",
            description=(
                "Realistic fire evacuation scenario with smoke, visibility effects, "
                "FED tracking, and dynamic path blocking. Uses Social Force Model."
            ),
            duration=600,
            time_step=0.1,
            default_movement_model="social_force"
        )

        # Staff: trained, move faster, higher risk tolerance
        engine.add_agent_profile(
            scenario=scenario,
            name="Staff",
            agent_type=AgentType.HUMAN,
            role=HumanRole.STAFF,
            count=20,
            base_speed=1.6,
            max_speed=2.0,
            risk_tolerance=0.7,
            schedule={0: "evacuate"}
        )

        # Visitors: untrained, slower, more panicked
        engine.add_agent_profile(
            scenario=scenario,
            name="Visitor",
            agent_type=AgentType.HUMAN,
            role=HumanRole.VISITOR,
            count=80,
            base_speed=1.2,
            max_speed=1.8,
            risk_tolerance=0.3,
            schedule={0: "evacuate"}
        )

        # Start evacuation immediately
        engine.add_event(scenario, time=0, event_type="evacuate")

        # Fire breaks out at t=60s
        engine.add_event(
            scenario, time=60,
            event_type="fire",
            location=[5.0, 5.0, 0.0],
            spread_rate=1.0,
            hazard_intensity=0.9,
            smoke_level=0.8
        )

        # Primary corridor blocked by fire at t=60s
        engine.add_event(scenario, time=60, event_type="block_path",
                        space_id="corridor")

        # Fire suppressed at t=180s; corridor re-opens
        engine.add_event(scenario, time=180, event_type="unblock_path",
                        space_id="corridor")

        return scenario
