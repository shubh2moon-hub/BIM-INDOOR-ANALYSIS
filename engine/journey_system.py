from __future__ import annotations

"""
Journey / Stage System

Inspired by JuPedSim's journey system, agents follow a sequence of stages.
Each stage represents a part of the agent's task: moving to a waypoint,
waiting in an area, passing through a flow-limited door, etc.
"""

import logging
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Stage base classes
# ---------------------------------------------------------------------------

class StageState:
    """State of a stage execution."""
    PENDING = "pending"
    ACTIVE = "active"
    COMPLETED = "completed"
    SKIPPED = "skipped"


@dataclass
class StageResult:
    """Result of a stage update."""
    state: str
    destination: Optional[str] = None          # space_id to navigate to
    wait_duration: float = 0.0                    # seconds to wait
    flow_limit: float = float('inf')             # max agents per second
    next_stage: bool = False                      # true if stage is finished
    metadata: Dict[str, Any] = field(default_factory=dict)


class Stage(ABC):
    """Abstract base for a journey stage."""

    def __init__(self, name: str = "", stage_id: Optional[str] = None):
        self.id = stage_id or str(uuid.uuid4())
        self.name = name or self.__class__.__name__
        self.state = StageState.PENDING

    @abstractmethod
    def update(self, agent, dt: float) -> StageResult:
        """Called each step. Returns instructions for the agent."""
        ...

    def reset(self):
        self.state = StageState.PENDING


# ---------------------------------------------------------------------------
# Concrete stages
# ---------------------------------------------------------------------------

class WaypointStage(Stage):
    """Navigate to a specific space/waypoint."""

    def __init__(self, target_space_id: str, tolerance: float = 1.0, name: str = ""):
        super().__init__(name=name or f"Waypoint({target_space_id})")
        self.target_space_id = target_space_id
        self.tolerance = tolerance
        self._arrived = False

    def update(self, agent, dt: float) -> StageResult:
        if self.state == StageState.PENDING:
            self.state = StageState.ACTIVE

        if self.state == StageState.ACTIVE:
            # Check if arrived
            if agent.current_space == self.target_space_id:
                self.state = StageState.COMPLETED
                return StageResult(state=StageState.COMPLETED, next_stage=True)

            # If we are physically close enough
            target = _get_space_center(agent.model, self.target_space_id)
            if target is not None:
                dist = np.linalg.norm(agent.position - np.array(target))
                if dist < self.tolerance:
                    self.state = StageState.COMPLETED
                    return StageResult(state=StageState.COMPLETED, next_stage=True)

            return StageResult(
                state=StageState.ACTIVE,
                destination=self.target_space_id,
                next_stage=False
            )

        return StageResult(state=self.state, next_stage=True)

    def reset(self):
        super().reset()
        self._arrived = False


class WaitStage(Stage):
    """Wait in a specific area for a duration."""

    def __init__(self, wait_space_id: str, duration: float, name: str = ""):
        super().__init__(name=name or f"Wait({wait_space_id}, {duration}s)")
        self.wait_space_id = wait_space_id
        self.duration = duration
        self._elapsed = 0.0

    def update(self, agent, dt: float) -> StageResult:
        if self.state == StageState.PENDING:
            self.state = StageState.ACTIVE

        if self.state == StageState.ACTIVE:
            # First make sure we are in the right space
            if agent.current_space != self.wait_space_id:
                return StageResult(
                    state=StageState.ACTIVE,
                    destination=self.wait_space_id,
                    next_stage=False
                )

            self._elapsed += dt
            if self._elapsed >= self.duration:
                self.state = StageState.COMPLETED
                return StageResult(state=StageState.COMPLETED, next_stage=True)

            return StageResult(
                state=StageState.ACTIVE,
                wait_duration=self.duration - self._elapsed,
                next_stage=False
            )

        return StageResult(state=self.state, next_stage=True)

    def reset(self):
        super().reset()
        self._elapsed = 0.0


class FlowLimitStage(Stage):
    """
    Pass through a door or area with a flow limitation.
    Agents may queue if capacity is exceeded.
    """

    def __init__(self, target_space_id: str, max_flow_rate: float = 1.0, name: str = ""):
        super().__init__(name=name or f"FlowLimit({target_space_id})")
        self.target_space_id = target_space_id
        self.max_flow_rate = max_flow_rate  # agents per second
        self._pass_time: Optional[float] = None

    def update(self, agent, dt: float) -> StageResult:
        if self.state == StageState.PENDING:
            self.state = StageState.ACTIVE

        if self.state == StageState.ACTIVE:
            # Check if already in target space
            if agent.current_space == self.target_space_id:
                self.state = StageState.COMPLETED
                return StageResult(state=StageState.COMPLETED, next_stage=True)

            # If we haven't claimed a pass time, request one from the model
            if self._pass_time is None:
                self._pass_time = agent.model.claim_flow_time(self.target_space_id, self.max_flow_rate)
                if self._pass_time is None:
                    # Cannot pass yet -> wait
                    return StageResult(
                        state=StageState.ACTIVE,
                        destination=self.target_space_id,
                        wait_duration=dt,
                        flow_limit=self.max_flow_rate,
                        next_stage=False
                    )

            # If current time >= pass_time, we can proceed
            if agent.model.current_time >= self._pass_time:
                self.state = StageState.COMPLETED
                return StageResult(state=StageState.COMPLETED, next_stage=True)

            # Still waiting
            return StageResult(
                state=StageState.ACTIVE,
                destination=self.target_space_id,
                wait_duration=self._pass_time - agent.model.current_time,
                flow_limit=self.max_flow_rate,
                next_stage=False
            )

        return StageResult(state=self.state, next_stage=True)

    def reset(self):
        super().reset()
        self._pass_time = None


class EvacuateStage(Stage):
    """Navigate to nearest exit and mark as evacuated."""

    def __init__(self, name: str = ""):
        super().__init__(name=name or "Evacuate")

    def update(self, agent, dt: float) -> StageResult:
        if self.state == StageState.PENDING:
            self.state = StageState.ACTIVE
            agent._set_evacuation_destination()

        if self.state == StageState.ACTIVE:
            if agent.state.value == "disabled":
                self.state = StageState.COMPLETED
                return StageResult(state=StageState.COMPLETED, next_stage=True)

            return StageResult(
                state=StageState.ACTIVE,
                destination=agent.destination,
                next_stage=False
            )

        return StageResult(state=self.state, next_stage=True)


class DirectSteeringStage(Stage):
    """Go to a specific 3D point (direct steering)."""

    def __init__(self, target_point: Tuple[float, float, float], tolerance: float = 1.0, name: str = ""):
        super().__init__(name=name or f"DirectSteer({target_point})")
        self.target_point = np.array(target_point)
        self.tolerance = tolerance

    def update(self, agent, dt: float) -> StageResult:
        if self.state == StageState.PENDING:
            self.state = StageState.ACTIVE

        if self.state == StageState.ACTIVE:
            dist = np.linalg.norm(agent.position - self.target_point)
            if dist < self.tolerance:
                self.state = StageState.COMPLETED
                return StageResult(state=StageState.COMPLETED, next_stage=True)

            return StageResult(
                state=StageState.ACTIVE,
                next_stage=False,
                metadata={"direct_target": self.target_point}
            )

        return StageResult(state=self.state, next_stage=True)


class RepeatStage(Stage):
    """Repeat a sub-journey N times (or forever if N=-1)."""

    def __init__(self, stages: List[Stage], repeats: int = 1, name: str = ""):
        super().__init__(name=name or f"Repeat(x{repeats})")
        self.stages = stages
        self.repeats = repeats
        self._current_repeat = 0
        self._current_index = 0

    def update(self, agent, dt: float) -> StageResult:
        if self.state == StageState.PENDING:
            self.state = StageState.ACTIVE

        while self.state == StageState.ACTIVE:
            if self.repeats >= 0 and self._current_repeat >= self.repeats:
                self.state = StageState.COMPLETED
                return StageResult(state=StageState.COMPLETED, next_stage=True)

            if self._current_index >= len(self.stages):
                self._current_repeat += 1
                self._current_index = 0
                if self.repeats >= 0 and self._current_repeat >= self.repeats:
                    self.state = StageState.COMPLETED
                    return StageResult(state=StageState.COMPLETED, next_stage=True)
                for s in self.stages:
                    s.reset()
                continue

            sub = self.stages[self._current_index]
            res = sub.update(agent, dt)
            if res.next_stage:
                self._current_index += 1
                # continue the while loop so we can immediately start next stage
                continue
            return res

        return StageResult(state=self.state, next_stage=True)

    def reset(self):
        super().reset()
        self._current_repeat = 0
        self._current_index = 0
        for s in self.stages:
            s.reset()


# ---------------------------------------------------------------------------
# Journey
# ---------------------------------------------------------------------------

@dataclass
class Journey:
    """A sequence of stages that an agent follows."""
    id: str
    name: str
    stages: List[Stage] = field(default_factory=list)
    description: str = ""

    def reset(self):
        for s in self.stages:
            s.reset()

    def get_current_stage(self, agent) -> Optional[Stage]:
        """Return the current stage for the agent based on its journey progress."""
        idx = getattr(agent, '_journey_index', 0)
        if 0 <= idx < len(self.stages):
            return self.stages[idx]
        return None

    def advance(self, agent):
        """Advance agent to the next stage."""
        idx = getattr(agent, '_journey_index', 0)
        agent._journey_index = idx + 1
        if agent._journey_index < len(self.stages):
            self.stages[agent._journey_index].reset()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_space_center(model, space_id: str) -> Optional[Tuple[float, float, float]]:
    """Get the center of a space by id."""
    if not model or not model.bim_model:
        return None
    space = model.bim_model.spaces.get(space_id)
    if space and space.center:
        return space.center
    return None
