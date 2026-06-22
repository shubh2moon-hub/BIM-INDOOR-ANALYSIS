from __future__ import annotations

"""
Pedestrian Dynamics Models

Inspired by JuPedSim, implements multiple microscopic pedestrian movement models:
- Social Force Model (SFM)          : Helbing et al. (2000)
- Collision-Free Speed Model (CFSM)  : Tordeux et al. (2015)
- Anticipation Velocity Model (AVM)    : Seitz & Köster (2012)
- Generalized Centrifugal Force Model (GCFM): Chraibi et al. (2010)

Each model returns a velocity change (acceleration) for a given agent based on
its desired direction, nearby agents, and wall geometry.
"""

import logging
import math
import random
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helper constants
# ---------------------------------------------------------------------------
EPS = 1e-6


# ---------------------------------------------------------------------------
# Model Parameters
# ---------------------------------------------------------------------------

@dataclass
class AgentMovementParams:
    """Per-agent parameters for pedestrian models."""
    mass: float = 80.0                    # kg
    radius: float = 0.25                  # meters (shoulder width / 2)
    desired_speed: float = 1.34           # m/s (free-flow walking speed)
    max_speed: float = 2.0                # m/s (running speed)
    relaxation_time: float = 0.5          # seconds (tau in SFM)
    # CFSM / AVM specific
    neighbor_repulsion_strength: float = 1.0
    neighbor_repulsion_range: float = 0.3
    geometry_repulsion_strength: float = 2.0
    geometry_repulsion_range: float = 0.2
    # SFM specific
    social_force_A: float = 2.0           # N
    social_force_B: float = 0.3           # m
    # GCFM specific
    centrifugal_force_factor: float = 3.0
    # AVM specific
    reaction_time: float = 0.4          # seconds


@dataclass
class WallSegment:
    """A line segment representing a wall or obstacle edge."""
    p1: np.ndarray
    p2: np.ndarray
    # Precomputed for speed
    length: float = field(init=False)
    unit: np.ndarray = field(init=False)

    def __post_init__(self):
        diff = self.p2 - self.p1
        self.length = np.linalg.norm(diff) + EPS
        self.unit = diff / self.length


# ---------------------------------------------------------------------------
# Base Model
# ---------------------------------------------------------------------------

class PedestrianModel(ABC):
    """Abstract base class for pedestrian dynamics models."""

    @abstractmethod
    def compute_velocity_change(
        self,
        agent_position: np.ndarray,
        agent_velocity: np.ndarray,
        params: AgentMovementParams,
        desired_direction: np.ndarray,
        neighbors: List[Tuple[np.ndarray, np.ndarray, AgentMovementParams]],
        walls: List[WallSegment],
        time_step: float
    ) -> np.ndarray:
        """
        Return the acceleration vector (m/s²) for the agent.

        Parameters
        ----------
        agent_position : current position
        agent_velocity : current velocity
        params         : per-agent movement parameters
        desired_direction : unit vector toward the goal
        neighbors      : list of (position, velocity, params) for each nearby agent
        walls          : list of WallSegment near the agent
        time_step      : simulation time step

        Returns
        -------
        np.ndarray : acceleration vector
        """
        ...

    @staticmethod
    def _nearest_point_on_segment(point: np.ndarray, wall: WallSegment) -> np.ndarray:
        """Project point onto the wall segment (clamped)."""
        t = np.dot(point - wall.p1, wall.unit)
        t = max(0.0, min(wall.length, t))
        return wall.p1 + t * wall.unit


# ---------------------------------------------------------------------------
# 1. Social Force Model (SFM)
# ---------------------------------------------------------------------------

class SocialForceModel(PedestrianModel):
    """
    Helbing et al. Social Force Model.

    Forces:
    - driving force:  m * (v0 * e0 - v) / tau
    - agent repulsion:  A * exp((r_ij - d_ij) / B) * n_ij
    - wall repulsion:  A * exp((r_i - d_iW) / B) * n_iW
    """

    def compute_velocity_change(
        self,
        agent_position: np.ndarray,
        agent_velocity: np.ndarray,
        params: AgentMovementParams,
        desired_direction: np.ndarray,
        neighbors: List[Tuple[np.ndarray, np.ndarray, AgentMovementParams]],
        walls: List[WallSegment],
        time_step: float
    ) -> np.ndarray:
        # Driving force
        desired_velocity = params.desired_speed * desired_direction
        driving = params.mass * (desired_velocity - agent_velocity) / params.relaxation_time

        # Agent repulsion
        repulsion = np.zeros(3)
        for other_pos, other_vel, other_params in neighbors:
            diff = agent_position - other_pos
            dist = np.linalg.norm(diff) + EPS
            n_ij = diff / dist
            r_sum = params.radius + other_params.radius
            # Exponential repulsion
            force_mag = params.social_force_A * math.exp((r_sum - dist) / params.social_force_B)
            repulsion += force_mag * n_ij

        # Wall repulsion
        wall_force = np.zeros(3)
        for wall in walls:
            nearest = self._nearest_point_on_segment(agent_position, wall)
            diff = agent_position - nearest
            dist = np.linalg.norm(diff) + EPS
            n_iW = diff / dist
            force_mag = params.social_force_A * math.exp((params.radius - dist) / params.social_force_B)
            wall_force += force_mag * n_iW

        # Sum of forces -> acceleration (F = m*a)
        acceleration = (driving + repulsion + wall_force) / params.mass
        return acceleration


# ---------------------------------------------------------------------------
# 2. Collision-Free Speed Model (CFSM)
# ---------------------------------------------------------------------------

class CollisionFreeSpeedModel(PedestrianModel):
    """
    Tordeux et al. Collision-Free Speed Model.

    Direction = normalized(desired_direction + sum of repulsion from neighbors)
    Speed   = min(desired_speed, max(0, (nearest_neighbor_distance - r_sum) / T))
    where T is a time-to-collision-like parameter.
    """

    def __init__(self, time_to_collision: float = 0.5):
        self.time_to_collision = time_to_collision

    def compute_velocity_change(
        self,
        agent_position: np.ndarray,
        agent_velocity: np.ndarray,
        params: AgentMovementParams,
        desired_direction: np.ndarray,
        neighbors: List[Tuple[np.ndarray, np.ndarray, AgentMovementParams]],
        walls: List[WallSegment],
        time_step: float
    ) -> np.ndarray:
        # Repulsion from neighbors
        repulsion = np.zeros(3)
        min_clearance = float('inf')

        for other_pos, other_vel, other_params in neighbors:
            diff = agent_position - other_pos
            dist = np.linalg.norm(diff) + EPS
            r_sum = params.radius + other_params.radius
            clearance = dist - r_sum
            if clearance < min_clearance:
                min_clearance = clearance

            # Isotropic exponential repulsion
            strength = params.neighbor_repulsion_strength
            range_ = params.neighbor_repulsion_range
            if range_ > 0:
                repulsion += strength * math.exp(-dist / range_) * (diff / dist)

        # Repulsion from walls
        wall_clearance = float('inf')
        for wall in walls:
            nearest = self._nearest_point_on_segment(agent_position, wall)
            dist = np.linalg.norm(agent_position - nearest) + EPS
            if dist < wall_clearance:
                wall_clearance = dist
            diff = agent_position - nearest
            strength = params.geometry_repulsion_strength
            range_ = params.geometry_repulsion_range
            if range_ > 0:
                repulsion += strength * math.exp(-dist / range_) * (diff / dist)

        # Direction
        new_direction = desired_direction + repulsion
        dir_norm = np.linalg.norm(new_direction) + EPS
        new_direction = new_direction / dir_norm

        # Speed
        if min_clearance == float('inf'):
            speed = params.desired_speed
        else:
            speed = min(params.desired_speed, max(0.0, min_clearance / self.time_to_collision))

        # Wall also limits speed
        if wall_clearance != float('inf'):
            speed = min(speed, max(0.0, wall_clearance / self.time_to_collision))

        target_velocity = new_direction * speed
        # Simple first-order acceleration toward target velocity
        acceleration = (target_velocity - agent_velocity) / max(time_step, params.relaxation_time)
        return acceleration


# ---------------------------------------------------------------------------
# 3. Anticipation Velocity Model (AVM)
# ---------------------------------------------------------------------------

class AnticipationVelocityModel(PedestrianModel):
    """
    AVM with three phases: perception, prediction, strategy selection.

    Agents anticipate future positions of neighbors and choose a direction
    that avoids predicted collisions. Speed is adjusted based on headway.
    """

    def __init__(self, prediction_horizon: float = 2.0):
        self.prediction_horizon = prediction_horizon

    def compute_velocity_change(
        self,
        agent_position: np.ndarray,
        agent_velocity: np.ndarray,
        params: AgentMovementParams,
        desired_direction: np.ndarray,
        neighbors: List[Tuple[np.ndarray, np.ndarray, AgentMovementParams]],
        walls: List[WallSegment],
        time_step: float
    ) -> np.ndarray:
        # Perceive current situation and predict future positions
        # Strategy: choose direction that maximizes clearance in predicted future

        # Candidate directions: fan around desired_direction
        candidate_angles = np.linspace(-60, 60, 13) * (np.pi / 180.0)
        best_dir = desired_direction.copy()
        best_score = -1e9

        for angle in candidate_angles:
            # Rotate desired_direction around Z axis
            cos_a = math.cos(angle)
            sin_a = math.sin(angle)
            # 2D rotation (XY plane)
            dx = desired_direction[0]
            dy = desired_direction[1]
            cand = np.array([
                dx * cos_a - dy * sin_a,
                dx * sin_a + dy * cos_a,
                desired_direction[2]
            ])
            cand = cand / (np.linalg.norm(cand) + EPS)

            # Predict agent position after reaction_time
            pred_pos = agent_position + cand * params.desired_speed * params.reaction_time

            score = 0.0
            # Penalize closeness to predicted neighbor positions
            for other_pos, other_vel, other_params in neighbors:
                other_pred = other_pos + other_vel * params.reaction_time
                dist = np.linalg.norm(pred_pos - other_pred) + EPS
                r_sum = params.radius + other_params.radius
                score += math.log(dist - r_sum + 1.0)

            # Penalize closeness to walls
            for wall in walls:
                nearest = self._nearest_point_on_segment(pred_pos, wall)
                dist = np.linalg.norm(pred_pos - nearest) + EPS
                score += math.log(dist + 1.0)

            if score > best_score:
                best_score = score
                best_dir = cand

        # Speed based on nearest headway in chosen direction
        min_headway = float('inf')
        for other_pos, other_vel, other_params in neighbors:
            diff = other_pos - agent_position
            proj = np.dot(diff, best_dir)
            if proj > 0:  # only ahead
                perp = np.linalg.norm(diff - proj * best_dir)
                if perp < (params.radius + other_params.radius) * 2.0:
                    min_headway = min(min_headway, proj)

        if min_headway == float('inf'):
            speed = params.desired_speed
        else:
            speed = min(params.desired_speed, max(0.0, min_headway / params.reaction_time))

        target_velocity = best_dir * speed
        acceleration = (target_velocity - agent_velocity) / max(time_step, params.relaxation_time)
        return acceleration


# ---------------------------------------------------------------------------
# 4. Generalized Centrifugal Force Model (GCFM)
# ---------------------------------------------------------------------------

class GeneralizedCentrifugalForceModel(PedestrianModel):
    """
    GCFM: adds centrifugal forces to the Social Force Model.

    The tangential component of relative velocity creates a centrifugal-like
    repulsion that improves lane formation and cornering behavior.
    """

    def compute_velocity_change(
        self,
        agent_position: np.ndarray,
        agent_velocity: np.ndarray,
        params: AgentMovementParams,
        desired_direction: np.ndarray,
        neighbors: List[Tuple[np.ndarray, np.ndarray, AgentMovementParams]],
        walls: List[WallSegment],
        time_step: float
    ) -> np.ndarray:
        # Driving force (same as SFM)
        desired_velocity = params.desired_speed * desired_direction
        driving = params.mass * (desired_velocity - agent_velocity) / params.relaxation_time

        repulsion = np.zeros(3)
        for other_pos, other_vel, other_params in neighbors:
            diff = agent_position - other_pos
            dist = np.linalg.norm(diff) + EPS
            n_ij = diff / dist
            r_sum = params.radius + other_params.radius

            # Normal social force
            force_mag = params.social_force_A * math.exp((r_sum - dist) / params.social_force_B)
            repulsion += force_mag * n_ij

            # Centrifugal component: tangential relative velocity
            rel_vel = agent_velocity - other_vel
            tangent = rel_vel - np.dot(rel_vel, n_ij) * n_ij
            tangent_norm = np.linalg.norm(tangent)
            if tangent_norm > EPS:
                # Centrifugal force increases with tangential speed
                cf = params.centrifugal_force_factor * tangent_norm / (dist + EPS)
                repulsion += cf * (tangent / tangent_norm)

        # Wall repulsion (same as SFM)
        wall_force = np.zeros(3)
        for wall in walls:
            nearest = self._nearest_point_on_segment(agent_position, wall)
            diff = agent_position - nearest
            dist = np.linalg.norm(diff) + EPS
            n_iW = diff / dist
            force_mag = params.social_force_A * math.exp((params.radius - dist) / params.social_force_B)
            wall_force += force_mag * n_iW

        acceleration = (driving + repulsion + wall_force) / params.mass
        return acceleration


# ---------------------------------------------------------------------------
# 5. Basic Model (original, kept for backward compatibility)
# ---------------------------------------------------------------------------

class BasicModel(PedestrianModel):
    """The original simple model from BIM-Agent Studio."""

    def compute_velocity_change(
        self,
        agent_position: np.ndarray,
        agent_velocity: np.ndarray,
        params: AgentMovementParams,
        desired_direction: np.ndarray,
        neighbors: List[Tuple[np.ndarray, np.ndarray, AgentMovementParams]],
        walls: List[WallSegment],
        time_step: float
    ) -> np.ndarray:
        target_velocity = desired_direction * params.desired_speed
        acceleration = (target_velocity - agent_velocity) / max(time_step, 0.5)
        return acceleration


# ---------------------------------------------------------------------------
# Model Registry
# ---------------------------------------------------------------------------

MODEL_REGISTRY: Dict[str, type] = {
    "basic": BasicModel,
    "social_force": SocialForceModel,
    "collision_free_speed": CollisionFreeSpeedModel,
    "anticipation_velocity": AnticipationVelocityModel,
    "generalized_centrifugal_force": GeneralizedCentrifugalForceModel,
}


def get_model(name: str) -> PedestrianModel:
    """Factory: get a model instance by name."""
    model_cls = MODEL_REGISTRY.get(name.lower(), BasicModel)
    return model_cls()


def list_models() -> List[str]:
    """Return available model names."""
    return list(MODEL_REGISTRY.keys())
