"""
Agent-Based Simulation Engine
Powered by Mesa for agent-based modeling and simulation.
"""

import uuid
import random
import logging
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
    # Mesa 3.x removed mesa.time — use a simple list-based scheduler shim
    class SimultaneousActivation:
        """Minimal drop-in for Mesa 3.x where mesa.time was removed."""
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
    ContinuousSpace = None  # not critical — we skip placement if unavailable

from core.bim_processor import BIMModel, BIMSpace, BIMElement, ElementCategory
from core.spatial_engine import SpatialGraph, SpatialIntelligenceEngine

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


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


class BIMAgent(Agent):
    """Base agent class for BIM simulation."""
    
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
        
        # Group behavior
        self.group_members = []
        
    def step(self):
        """Execute one step of the agent."""
        # Record state — schedule.steps attr differs between Mesa versions
        current_step = getattr(getattr(self, 'model', None), '_step_count', 0)
        if hasattr(self.model, 'schedule') and hasattr(self.model.schedule, 'steps'):
            current_step = self.model.schedule.steps

        self.state_history.append({
            "step": current_step,
            "state": self.state.value,
            "position": tuple(self.position),
            "speed": self.current_speed
        })
        self.position_history.append(tuple(self.position))
        self.speed_history.append(self.current_speed)
        
        # Execute behavior based on state
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
            
    def _behavior_idle(self):
        """Behavior when idle - decide next action."""
        # Check schedule
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
            # Random behavior
            if random.random() < 0.1 and self.destination:
                self.state = AgentState.MOVING
                self._plan_path()
                
    def _behavior_moving(self):
        """Behavior when moving along path."""
        if not self.path:
            self.state = AgentState.IDLE
            return
            
        # Move towards next waypoint
        target = np.array(self.path[0])
        direction = target - self.position
        distance = np.linalg.norm(direction)
        
        if distance < 0.5:  # Reached waypoint
            self.path.pop(0)
            if not self.path:
                self.state = AgentState.IDLE
                self.velocity = np.array([0.0, 0.0, 0.0])
                return
            target = np.array(self.path[0])
            direction = target - self.position
            distance = np.linalg.norm(direction)
        
        # Normalize and apply speed
        if distance > 0:
            direction = direction / distance
            
            # Avoid other agents (simple collision avoidance)
            avoidance = self._calculate_avoidance()
            direction = direction + avoidance
            direction = direction / (np.linalg.norm(direction) + 1e-6)
            
            # Apply speed with variation
            self.current_speed = min(
                self.profile.base_speed * (0.8 + random.random() * 0.4),
                self.profile.max_speed
            )
            
            # Reduce speed in crowds
            nearby_agents = self._count_nearby_agents(2.0)
            if nearby_agents > 3:
                self.current_speed *= max(0.3, 1.0 - nearby_agents * 0.15)
            
            self.velocity = direction * self.current_speed
            new_position = self.position + self.velocity * self.model.time_step
            
            # Update position
            self.position = new_position
            self.traveled_distance += np.linalg.norm(self.velocity) * self.model.time_step
            self.travel_time += self.model.time_step
            
            # Update current space
            self._update_current_space()
            
    def _behavior_waiting(self):
        """Behavior when waiting."""
        self.waiting_time += self.model.time_step
        self.velocity = np.array([0.0, 0.0, 0.0])
        
        # Check if wait is over
        if self.waiting_time > 30:  # Max 30 seconds wait
            self.state = AgentState.IDLE
            
    def _behavior_working(self):
        """Behavior when working."""
        self.velocity = np.array([0.0, 0.0, 0.0])
        # Work for a duration then become idle
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
            self.profile.base_speed = min(self.profile.max_speed * 1.2, 2.0)
            self._behavior_moving()
        else:
            # Reached exit
            self.state = AgentState.DISABLED
            self.model.evacuated_agents += 1
            
    def _behavior_interacting(self):
        """Behavior when interacting with other agents."""
        self.velocity = np.array([0.0, 0.0, 0.0])
        self.interactions += 1
        
        # Interaction lasts for a few steps
        if random.random() < 0.1:
            self.state = AgentState.IDLE
            
    def _plan_path(self):
        """Plan path to destination using spatial graph."""
        if not self.destination or not self.model.spatial_engine:
            return
            
        current_space = self.current_space
        if not current_space:
            return
            
        # Find path through spaces
        path = self.model.spatial_engine.find_shortest_path(
            current_space,
            self.destination
        )
        
        if path:
            # Convert space path to waypoints
            self.path = []
            for node_id in path[1:]:  # Skip current space
                space_id = node_id.replace("space_", "")
                if space_id in self.model.bim_model.spaces:
                    space = self.model.bim_model.spaces[space_id]
                    if space.center:
                        self.path.append(space.center)
                        
    def _set_evacuation_destination(self):
        """Set nearest exit as destination during evacuation."""
        if not self.model.bim_model:
            return
            
        # Find nearest exit (space with category "entrance" or "exit")
        exits = [
            s for s in self.model.bim_model.spaces.values()
            if s.category in ["entrance", "exit", "lobby", "public"]
        ]
        
        if exits:
            # Find closest exit
            closest = min(exits, key=lambda e: self._distance_to(e.center))
            self.destination = closest.id
            
    def _calculate_avoidance(self) -> np.ndarray:
        """Calculate avoidance vector from nearby agents."""
        avoidance = np.array([0.0, 0.0, 0.0])
        
        for agent in self.model.schedule.agents:
            if agent != self:
                diff = self.position - agent.position
                dist = np.linalg.norm(diff)
                if 0 < dist < self.profile.size * 3:
                    force = (self.profile.size * 3 - dist) / (self.profile.size * 3)
                    avoidance += (diff / (dist + 1e-6)) * force
                    
        return avoidance * 0.5
        
    def _count_nearby_agents(self, radius: float) -> int:
        """Count agents within radius."""
        count = 0
        for agent in self.model.schedule.agents:
            if agent != self:
                dist = np.linalg.norm(self.position - agent.position)
                if dist < radius:
                    count += 1
        return count
        
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
        return np.linalg.norm(self.position - np.array(point))
        
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
            "current_speed": self.current_speed
        }


class BIMSimulationModel(Model):
    """Mesa model for BIM agent-based simulation."""
    
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
        
        # Agent tracking — use _agent_registry to avoid conflict with
        # Mesa 2.4+ which reserves model.agents as an AgentSet
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
        
        # Density map (space_id -> agent count)
        self.density_map: Dict[str, int] = {}
        
        # Data collector
        self.datacollector = DataCollector(
            model_reporters={
                "Agent Count": lambda m: len(m.schedule.agents),
                "Moving": self._count_moving_agents,
                "Waiting": self._count_waiting_agents,
                "Avg Speed": self._get_avg_speed,
                "Evacuated": lambda m: m.evacuated_agents
            },
            agent_reporters={
                "State": lambda a: a.state.value,
                "Speed": lambda a: a.current_speed,
                "Distance": lambda a: a.traveled_distance,
                "Space": lambda a: a.current_space
            }
        )
        
        # Initialize agents
        self._initialize_agents()
        
        logger.info(f"Simulation model initialized: {len(self.schedule.agents)} agents")
        
    def _initialize_agents(self):
        """Create agents based on scenario."""
        for profile in self.scenario.agent_profiles:
            count = self.scenario.agent_counts.get(profile.id, 1)
            for _ in range(count):
                self._create_agent(profile)
                
    def _create_agent(self, profile: AgentProfile, position: Optional[Tuple] = None) -> BIMAgent:
        """Create a new agent in the simulation."""
        agent_id = self.next_agent_id
        self.next_agent_id += 1
        
        agent = BIMAgent(agent_id, self, profile)
        
        # Set initial position
        if position:
            agent.position = np.array(position)
        else:
            # Random position in a random space
            if self.bim_model and self.bim_model.spaces:
                space = random.choice(list(self.bim_model.spaces.values()))
                if space.center:
                    agent.position = np.array([
                        space.center[0] + random.uniform(-2, 2),
                        space.center[1] + random.uniform(-2, 2),
                        space.center[2] if len(space.center) > 2 else 0
                    ])
                    agent.current_space = space.id
                    
        self.schedule.add(agent)
        self._agent_registry[agent_id] = agent
        
        # Place on continuous space
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
        # Mesa 2.4+ AgentSet via .agents property or our shim
        if hasattr(sched, 'agents'):
            try:
                return list(sched.agents)
            except Exception:
                pass
        # Older Mesa 2.x SimultaneousActivation stores in ._agents dict
        if hasattr(sched, '_agents'):
            if isinstance(sched._agents, dict):
                return list(sched._agents.values())
            try:
                return list(sched._agents)
            except Exception:
                pass
        # Last fallback: our own registry dict
        return list(self._agent_registry.values())

    def step(self):
        """Execute one simulation step."""
        # Update time
        self.current_time += self.time_step
        
        # Process events
        self._process_events()
        
        # Execute agent steps
        self.schedule.step()
        
        # Update density map
        self._update_density_map()
        
        # Detect congestion
        self._detect_congestion()
        
        # Detect social interactions
        self._detect_interactions()
        
        # Collect metrics
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
        elif event_type == "set_destination":
            self._event_set_destination(event)
            
    def _event_spawn_agents(self, event: Dict):
        """Spawn new agents at specified location."""
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
        """Trigger evacuation for all agents."""
        for agent in self._get_all_agents():
            agent.state = AgentState.EVACUATING
            agent._set_evacuation_destination()
            
    def _event_fire(self, event: Dict):
        """Simulate fire in specified location."""
        location = event.get("location")
        spread_rate = event.get("spread_rate", 1.0)
        
        # Block affected areas
        if location and self.spatial_engine and self.spatial_engine.spatial_graph:
            for node_id, node in self.spatial_engine.spatial_graph.nodes.items():
                dist = np.linalg.norm(
                    np.array(node.center) - np.array(location)
                )
                if dist < 10.0 * spread_rate:
                    node.attributes["on_fire"] = True
                    
    def _event_block_path(self, event: Dict):
        """Block a path or space."""
        space_id = event.get("space_id")
        if space_id and self.spatial_engine and self.spatial_engine.spatial_graph:
            node_id = f"space_{space_id}"
            if node_id in self.spatial_engine.spatial_graph.nodes:
                self.spatial_engine.spatial_graph.nodes[node_id].attributes["blocked"] = True
                
    def _event_set_destination(self, event: Dict):
        """Set destination for specific agents."""
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
                
    def _update_density_map(self):
        """Update the density map of spaces."""
        self.density_map = {}
        for agent in self._get_all_agents():
            if agent.current_space:
                self.density_map[agent.current_space] = \
                    self.density_map.get(agent.current_space, 0) + 1
                    
    def _detect_congestion(self):
        """Detect congestion points in the building."""
        congestion_threshold = 3  # agents per square meter
        
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
        """Detect social interactions between agents."""
        interaction_distance = 2.0  # meters
        
        agents_list = self._get_all_agents()
        for i in range(len(agents_list)):
            for j in range(i + 1, len(agents_list)):
                a1 = agents_list[i]
                a2 = agents_list[j]
                
                dist = np.linalg.norm(a1.position - a2.position)
                if dist < interaction_distance:
                    # Check if both agents are sociable enough
                    if (a1.profile.sociability > 0.3 and 
                        a2.profile.sociability > 0.3):
                        self.social_interactions += 1
                        if random.random() < 0.1:  # 10% chance to start interacting
                            a1.state = AgentState.INTERACTING
                            a2.state = AgentState.INTERACTING
                            
    def _collect_metrics(self) -> SimulationMetrics:
        """Collect simulation metrics for current step."""
        agents = self._get_all_agents()
        
        metrics = SimulationMetrics(
            timestamp=getattr(self.schedule, 'steps', int(self.current_time)),
            agent_count=len(agents),
            agents_moving=sum(1 for a in agents if a.state == AgentState.MOVING),
            agents_waiting=sum(1 for a in agents if a.state == AgentState.WAITING),
            avg_speed=sum(a.current_speed for a in agents) / len(agents) if agents else 0,
            avg_travel_time=sum(a.travel_time for a in agents) / len(agents) if agents else 0,
            density_map={
                sid: count / self.bim_model.spaces[sid].area
                for sid, count in self.density_map.items()
                if sid in self.bim_model.spaces and self.bim_model.spaces[sid].area > 0
            },
            social_interactions=self.social_interactions,
            evacuation_progress=(self.evacuated_agents / len(agents) * 100) if agents else 0
        )
        
        self.metrics_history.append(metrics)
        return metrics
        
    def _count_moving_agents(self) -> int:
        """Count agents that are currently moving."""
        return sum(1 for a in self._get_all_agents() if a.state == AgentState.MOVING)
        
    def _count_waiting_agents(self) -> int:
        """Count agents that are waiting."""
        return sum(1 for a in self._get_all_agents() if a.state == AgentState.WAITING)
        
    def _get_avg_speed(self) -> float:
        """Get average speed of all agents."""
        agents = self._get_all_agents()
        return sum(a.current_speed for a in agents) / len(agents) if agents else 0
        
    def get_agent_metrics(self) -> List[Dict]:
        """Get metrics for all agents."""
        return [agent.get_metrics() for agent in self._get_all_agents()]
        
    def get_space_occupancy(self) -> Dict[str, Dict]:
        """Get occupancy information for all spaces."""
        occupancy = {}
        for space_id, space in self.bim_model.spaces.items():
            agent_count = self.density_map.get(space_id, 0)
            # BIMSpace doesn't have a capacity attribute by default, assume 2 sq m per person
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
        """Get current simulation metrics."""
        if self.metrics_history:
            return self.metrics_history[-1]
        return SimulationMetrics(timestamp=0, agent_count=0, agents_moving=0, 
                               agents_waiting=0, avg_speed=0, avg_travel_time=0)


class SimulationEngine:
    """Main simulation engine managing simulation runs."""
    
    def __init__(self):
        self.current_model: Optional[BIMSimulationModel] = None
        self.is_running = False
        self.current_step = 0
        self.scenarios: Dict[str, SimulationScenario] = {}
        self.on_step_callbacks: List[Callable] = []
        self.on_complete_callbacks: List[Callable] = []
        
    def create_scenario(self, name: str, description: str, duration: int, **kwargs) -> SimulationScenario:
        """Create a new simulation scenario."""
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
        profile = AgentProfile(
            id=str(uuid.uuid4()),
            name=name,
            agent_type=agent_type,
            role=role,
            **kwargs
        )
        scenario.agent_profiles.append(profile)
        scenario.agent_counts[profile.id] = count
        return profile
        
    def add_event(self, scenario: SimulationScenario, time: float, event_type: str, **kwargs):
        """Add an event to a scenario."""
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
        """Initialize a simulation run."""
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
        """Execute one simulation step."""
        if not self.current_model or not self.is_running:
            return
            
        self.current_model.step()
        self.current_step += 1
        
        # Notify callbacks
        for callback in self.on_step_callbacks:
            callback(self.current_step, self.current_model)
            
        # Check if simulation is complete
        if self.current_step >= self.current_model.max_steps:
            self.stop()
            
    def run(self, steps: Optional[int] = None):
        """Run the simulation for specified steps."""
        if not self.current_model:
            logger.error("No simulation model initialized")
            return
            
        self.is_running = True
        target_steps = steps or self.current_model.max_steps
        
        while self.is_running and self.current_step < target_steps:
            self.step()
            
        # Notify completion callbacks
        if not self.is_running:
            for callback in self.on_complete_callbacks:
                callback(self.current_model)
                
    def start(self):
        """Start or resume the simulation."""
        self.is_running = True
        logger.info("Simulation started")
        
    def pause(self):
        """Pause the simulation."""
        self.is_running = False
        logger.info("Simulation paused")
        
    def stop(self):
        """Stop the simulation."""
        self.is_running = False
        self.current_step = 0
        logger.info("Simulation stopped")
        
    def reset(self):
        """Reset the simulation to initial state."""
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
        
    def get_results(self) -> Dict:
        """Get comprehensive simulation results."""
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
            "completed_events": model.completed_events
        }
        
    def on_step(self, callback: Callable):
        """Register a callback for simulation steps."""
        self.on_step_callbacks.append(callback)
        
    def on_complete(self, callback: Callable):
        """Register a callback for simulation completion."""
        self.on_complete_callbacks.append(callback)


# Preset scenarios
class ScenarioPresets:
    """Predefined simulation scenarios."""
    
    @staticmethod
    def office_scenario() -> SimulationScenario:
        """Create a typical office scenario."""
        engine = SimulationEngine()
        scenario = engine.create_scenario(
            name="Office Daily Operations",
            description="Simulate a typical day in an office building",
            duration=28800  # 8 hours in seconds
        )
        
        # Add office workers
        profile = engine.add_agent_profile(
            scenario=scenario,
            name="Office Worker",
            agent_type=AgentType.HUMAN,
            role=HumanRole.OFFICE_WORKER,
            count=50,
            base_speed=1.2,
            sociability=0.4,
            schedule={
                0: "move",  # Arrival
                14400: "move",  # Lunch
                18000: "move",  # Return
                25200: "move"  # Departure
            }
        )
        
        # Add morning arrival event
        engine.add_event(scenario, time=0, event_type="spawn_agents", 
                        profile_id=profile.id, count=50)
        
        # Add lunch event
        engine.add_event(scenario, time=14400, event_type="set_destination",
                        filter={"profile.agent_type": AgentType.HUMAN})
        
        # Add evacuation drill event
        engine.add_event(scenario, time=20000, event_type="evacuate")
        
        return scenario
    
    @staticmethod
    def evacuation_scenario() -> SimulationScenario:
        """Create an evacuation scenario."""
        engine = SimulationEngine()
        scenario = engine.create_scenario(
            name="Emergency Evacuation",
            description="Simulate emergency evacuation of the building",
            duration=1800  # 30 minutes
        )
        
        # Mix of occupants
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
        
        # Immediate evacuation
        engine.add_event(scenario, time=0, event_type="evacuate")
        
        # Fire event after 2 minutes
        engine.add_event(scenario, time=120, event_type="fire",
                        location=[0, 0, 0], spread_rate=0.5)
        
        return scenario
    
    @staticmethod
    def hospital_scenario() -> SimulationScenario:
        """Create a hospital scenario."""
        engine = SimulationEngine()
        scenario = engine.create_scenario(
            name="Hospital Operations",
            description="Simulate hospital patient and staff movement",
            duration=86400  # 24 hours
        )
        
        # Staff
        staff = engine.add_agent_profile(
            scenario=scenario,
            name="Medical Staff",
            agent_type=AgentType.HUMAN,
            role=HumanRole.STAFF,
            count=30,
            base_speed=1.5,
            sociability=0.6
        )
        
        # Patients
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
        
        # Service robots
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
        """Create a university campus scenario."""
        engine = SimulationEngine()
        scenario = engine.create_scenario(
            name="University Class Transitions",
            description="Simulate student movement between classes",
            duration=10800  # 3 hours (multiple class periods)
        )
        
        # Students
        students = engine.add_agent_profile(
            scenario=scenario,
            name="Student",
            agent_type=AgentType.HUMAN,
            role=HumanRole.STUDENT,
            count=200,
            base_speed=1.3,
            sociability=0.7,
            schedule={
                0: "move",      # First class transition
                2700: "move",   # Second transition
                5400: "move",   # Third transition
                8100: "move"    # End of day
            }
        )
        
        return scenario
