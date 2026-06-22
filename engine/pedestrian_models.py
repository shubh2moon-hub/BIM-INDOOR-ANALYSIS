from __future__ import annotations

"""
Pedestrian Dynamics Models  (v1.3.0)

Inspired by JuPedSim, implements multiple microscopic pedestrian movement models:
- Social Force Model (SFM)                    : Helbing et al. (2000)
- Collision-Free Speed Model (CFSM)           : Tordeux et al. (2015)
- Anticipation Velocity Model (AVM)           : Seitz & Köster (2012)
- Generalized Centrifugal Force Model (GCFM)  : Chraibi et al. (2010)
- WarpDriver Model (WD)                       : Gradient navigation field (JuPedSim 2024)

NEW v1.3.0:
- UniformGridSpatialIndex for O(1) average wall segment lookup
- WarpDriverModel: probabilistic collision-field / gradient navigation model

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
    reaction_time: float = 0.4            # seconds
    # WarpDriver specific
    warp_sigma: float = 0.5              # Gaussian width of each agent's repulsion field


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
# Spatial Index (NEW v1.3.0)
# ---------------------------------------------------------------------------

class UniformGridSpatialIndex:
    """
    Uniform grid spatial index for fast O(1)-average wall segment lookup.

    The building footprint is partitioned into a grid of cells. Each wall
    segment is inserted into all cells it overlaps. Queries return the union
    of all segments in cells within the query radius — dramatically faster
    than O(N) brute-force for large buildings.
    """

    def __init__(self, cell_size: float = 5.0):
        self.cell_size = cell_size
        # cell (ix, iy) -> list of WallSegment
        self._cells: Dict[Tuple[int, int], List[WallSegment]] = {}

    def _cell_key(self, x: float, y: float) -> Tuple[int, int]:
        return (int(math.floor(x / self.cell_size)),
                int(math.floor(y / self.cell_size)))

    def insert(self, wall: WallSegment):
        """Insert a wall segment into all overlapping grid cells."""
        # Compute bounding box of the segment (2D: XY plane)
        min_x = min(wall.p1[0], wall.p2[0])
        max_x = max(wall.p1[0], wall.p2[0])
        min_y = min(wall.p1[1], wall.p2[1])
        max_y = max(wall.p1[1], wall.p2[1])

        ix_min, iy_min = self._cell_key(min_x, min_y)
        ix_max, iy_max = self._cell_key(max_x, max_y)

        for ix in range(ix_min, ix_max + 1):
            for iy in range(iy_min, iy_max + 1):
                key = (ix, iy)
                if key not in self._cells:
                    self._cells[key] = []
                self._cells[key].append(wall)

    def query(self, position: np.ndarray, radius: float) -> List[WallSegment]:
        """
        Return all wall segments that could be within `radius` of `position`.
        This is conservative (may include segments slightly beyond radius),
        exact filtering is done by the caller.
        """
        x, y = float(position[0]), float(position[1])
        ix_min, iy_min = self._cell_key(x - radius, y - radius)
        ix_max, iy_max = self._cell_key(x + radius, y + radius)

        seen = set()
        result = []
        for ix in range(ix_min, ix_max + 1):
            for iy in range(iy_min, iy_max + 1):
                key = (ix, iy)
                for wall in self._cells.get(key, []):
                    wid = id(wall)
                    if wid not in seen:
                        seen.add(wid)
                        result.append(wall)
        return result

    def build(self, walls: List[WallSegment]):
        """Bulk-insert a list of wall segments."""
        self._cells.clear()
        for wall in walls:
            self.insert(wall)
        logger.debug(f"SpatialIndex built: {len(walls)} walls into {len(self._cells)} cells "
                     f"(cell_size={self.cell_size}m)")


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
# 6. WarpDriver Model — Gradient Navigation Field (NEW v1.3.0)
# ---------------------------------------------------------------------------

class WarpDriverModel(PedestrianModel):
    """
    WarpDriver: Probabilistic Collision-Field / Gradient Navigation Field Model.

    Inspired by the newest JuPedSim model (2024). Each agent creates a Gaussian
    repulsion "charge cloud". The navigation field is the sum of all repulsive
    fields plus a goal-attraction potential. The agent steers by descending the
    gradient of the combined potential field.

    This produces smoother, more natural large-crowd behavior without explicit
    force calculations — particularly good for dense environments where SFM
    tends to produce oscillatory behavior.

    Algorithm:
    1. Compute attraction potential gradient toward goal
    2. Compute repulsion gradient from all neighbors (Gaussian fields)
    3. Compute repulsion gradient from walls (exponential fields)
    4. Combine gradients; derive target velocity
    5. Clamp speed and apply relaxation
    """

    def __init__(self, attraction_strength: float = 3.0, num_samples: int = 8):
        """
        Parameters
        ----------
        attraction_strength : strength of the goal-attraction field
        num_samples         : number of candidate directions sampled for gradient descent
        """
        self.attraction_strength = attraction_strength
        self.num_samples = num_samples

    def _gaussian_repulsion_gradient(
        self,
        agent_pos: np.ndarray,
        source_pos: np.ndarray,
        sigma: float,
        amplitude: float
    ) -> np.ndarray:
        """
        Gradient of a Gaussian repulsion field at agent_pos due to source at source_pos.
        Field: U(r) = amplitude * exp(-|r|^2 / (2*sigma^2))
        Gradient: dU/dx = U(r) * (-r / sigma^2)  [points away from source]
        """
        diff = agent_pos - source_pos
        dist_sq = np.dot(diff, diff)
        sigma_sq = sigma * sigma
        field_val = amplitude * math.exp(-dist_sq / (2.0 * sigma_sq + EPS))
        # Negative of gradient (ascent of repulsion = movement away from source)
        gradient = field_val * (diff / (sigma_sq + EPS))
        return gradient

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
        sigma = params.warp_sigma

        # --- 1. Goal attraction: strong pull in desired direction ---
        attraction = self.attraction_strength * params.desired_speed * desired_direction

        # --- 2. Neighbor repulsion (Gaussian fields) ---
        neighbor_repulsion = np.zeros(3)
        for other_pos, other_vel, other_params in neighbors:
            combined_sigma = sigma + other_params.warp_sigma
            # Amplitude scales with combined agent sizes
            amplitude = params.neighbor_repulsion_strength * 2.0
            grad = self._gaussian_repulsion_gradient(
                agent_position, other_pos, combined_sigma, amplitude
            )
            # Also anticipate future position (velocity-weighted)
            future_pos = other_pos + other_vel * params.reaction_time
            future_grad = self._gaussian_repulsion_gradient(
                agent_position, future_pos, combined_sigma * 1.5, amplitude * 0.5
            )
            neighbor_repulsion += grad + future_grad

        # --- 3. Wall repulsion (exponential decay, stronger than neighbor) ---
        wall_repulsion = np.zeros(3)
        for wall in walls:
            nearest = self._nearest_point_on_segment(agent_position, wall)
            diff = agent_position - nearest
            dist = np.linalg.norm(diff) + EPS
            # Exponential wall field: very strong up close
            amplitude = params.geometry_repulsion_strength * 3.0
            decay = params.geometry_repulsion_range
            field_val = amplitude * math.exp(-dist / (decay + EPS))
            wall_repulsion += field_val * (diff / dist)

        # --- 4. Combine into navigation gradient ---
        nav_gradient = attraction + neighbor_repulsion + wall_repulsion
        nav_norm = np.linalg.norm(nav_gradient) + EPS
        nav_direction = nav_gradient / nav_norm

        # --- 5. Speed: modulated by nearest obstacle clearance ---
        min_clearance = float('inf')
        for other_pos, _, other_params in neighbors:
            dist = np.linalg.norm(agent_position - other_pos) + EPS
            clearance = dist - (params.radius + other_params.radius)
            if clearance < min_clearance:
                min_clearance = clearance

        for wall in walls:
            nearest = self._nearest_point_on_segment(agent_position, wall)
            dist = np.linalg.norm(agent_position - nearest) + EPS
            clearance = dist - params.radius
            if clearance < min_clearance:
                min_clearance = clearance

        if min_clearance == float('inf') or min_clearance > 2.0:
            speed = params.desired_speed
        else:
            # Smoothly reduce speed as clearance decreases
            speed = params.desired_speed * max(0.0, math.tanh(min_clearance / params.radius))

        target_velocity = nav_direction * speed
        acceleration = (target_velocity - agent_velocity) / max(time_step, params.relaxation_time)
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
    "warp_driver": WarpDriverModel,  # NEW v1.3.0
}


def get_model(name: str) -> PedestrianModel:
    """Factory: get a model instance by name."""
    model_cls = MODEL_REGISTRY.get(name.lower(), BasicModel)
    return model_cls()


def list_models() -> List[str]:
    """Return available model names."""
    return list(MODEL_REGISTRY.keys())
