"""
3D Visualization Engine
Uses PyVista/VTK for real-time 3D visualization of BIM models and agents.
"""

import logging
import uuid
from typing import Dict, List, Optional, Tuple, Any, Callable
from dataclasses import dataclass, field
from enum import Enum

import numpy as np
import pyvista as pv
from pyvistaqt import QtInteractor
from PySide6.QtCore import QObject, Signal, QTimer

from core.bim_processor import BIMModel, BIMSpace, BIMElement, ElementCategory
from core.spatial_engine import SpatialGraph, SpaceNode, Connection
from engine.simulation_engine import BIMSimulationModel, BIMAgent, AgentState

logger = logging.getLogger(__name__)


class VisualizationMode(Enum):
    BIM_ONLY = "bim_only"
    AGENTS = "agents"
    DENSITY = "density"
    HEAT_MAP = "heat_map"
    EVACUATION = "evacuation"
    NAVIGATION = "navigation"


@dataclass
class VisualizationSettings:
    """Settings for 3D visualization."""
    show_walls: bool = True
    show_floors: bool = True
    show_doors: bool = True
    show_windows: bool = True
    show_furniture: bool = True
    show_stairs: bool = True
    show_spaces: bool = True
    show_labels: bool = True
    show_navigation: bool = False
    show_spatial_graph: bool = True
    show_evacuation_paths: bool = False
    level_filter: Optional[str] = None
    
    # Agent visualization
    show_agents: bool = True
    agent_size: float = 0.3
    show_trails: bool = False
    agent_trail_length: int = 50
    show_interactions: bool = False
    
    # Color settings
    wall_color: Tuple[float, float, float] = (0.85, 0.85, 0.85)
    wall_opacity: float = 1.0
    floor_color: Tuple[float, float, float] = (0.7, 0.7, 0.7)
    door_color: Tuple[float, float, float] = (0.6, 0.4, 0.2)
    window_color: Tuple[float, float, float] = (0.5, 0.7, 0.9)
    space_colors: Dict[str, Tuple[float, float, float]] = field(default_factory=lambda: {
        "office": (0.9, 0.9, 0.8),
        "corridor": (0.95, 0.95, 0.95),
        "staircase": (0.8, 0.8, 0.9),
        "elevator": (0.7, 0.7, 0.8),
        "restroom": (0.9, 0.8, 0.8),
        "meeting": (0.8, 0.9, 0.8),
        "public": (0.85, 0.85, 0.9),
        "space": (0.9, 0.9, 0.9)
    })
    
    # Agent state colors
    agent_state_colors: Dict[str, Tuple[float, float, float]] = field(default_factory=lambda: {
        "idle": (0.2, 0.6, 1.0),
        "moving": (0.2, 0.8, 0.2),
        "waiting": (1.0, 0.8, 0.2),
        "working": (0.8, 0.4, 1.0),
        "evacuating": (1.0, 0.2, 0.2),
        "interacting": (1.0, 0.6, 0.2),
        "resting": (0.6, 0.6, 0.6),
        "disabled": (0.3, 0.3, 0.3)
    })
    
    # Display settings
    background_color: Tuple[float, float, float] = (0.95, 0.95, 0.97)
    opacity: float = 0.85


class Visualization3D(QObject):
    """3D visualization engine for BIM models and simulations."""
    
    # Signals
    agent_selected = Signal(int)  # agent_id
    space_selected = Signal(str)  # space_id
    element_selected = Signal(str)  # element_id
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.settings = VisualizationSettings()
        self.plotter: Optional[QtInteractor] = None
        self.current_model: Optional[BIMModel] = None
        self.current_simulation: Optional[BIMSimulationModel] = None
        self.spatial_graph: Optional[SpatialGraph] = None
        
        # Actor storage
        self.actors: Dict[str, Any] = {}
        self.agent_actors: Dict[int, Any] = {}
        self.space_actors: Dict[str, Any] = {}
        self.element_actors: Dict[str, Any] = {}
        self.label_actors: Dict[str, Any] = {}
        self.path_actors: Dict[str, Any] = {}       # evacuation path actors
        self.exit_actors: Dict[str, Any] = {}       # virtual exit marker actors
        self.trail_actors: Dict[int, Any] = {}      # agent trail/trace actors
        
        # Update timer
        self.update_timer: Optional[QTimer] = None
        self.update_interval = 100  # ms
        
        # Callbacks
        self.on_selection_callbacks: List[Callable] = []
        
    def create_plotter(self, parent=None, show=True) -> QtInteractor:
        """Create and configure the PyVista plotter.

        Uses QtInteractor (embedded widget) instead of BackgroundPlotter
        (separate window) so it can be placed inside the main window layout
        without triggering the 'multiple values for parent' error.
        """
        self.plotter = QtInteractor(parent)

        # Configure plotter
        self.plotter.set_background(self.settings.background_color)
        self.plotter.add_axes()
        self.plotter.add_bounding_box()

        # Enable picking
        try:
            self.plotter.enable_mesh_picking(
                callback=self._on_element_picked,
                left_clicking=True,
                show=False
            )
        except Exception as e:
            logger.warning(f"Element picking not available: {e}")

        # Setup update timer
        self.update_timer = QTimer(self)
        self.update_timer.timeout.connect(self._update_simulation_visualization)

        logger.info("3D plotter created (QtInteractor)")
        return self.plotter
        
    def load_bim_model(self, model: BIMModel):
        """Load and display a BIM model."""
        self.current_model = model
        self.spatial_graph = model.spatial_graph
        
        logger.info(f"Loading BIM model into 3D view: {model.name}")
        
        # Clear existing
        self.clear()
        
        # Display elements
        self._display_elements()
        
        # Display spaces
        self._display_spaces()
        
        # Display spatial graph if available
        if self.spatial_graph and self.settings.show_spatial_graph:
            self._display_spatial_graph()
        
        # Display evacuation paths if available
        if self.spatial_graph and self.settings.show_evacuation_paths:
            self.display_evacuation_paths()
        
        # Display exit markers
        self.display_exits()
            
        # Reset camera
        if self.plotter:
            self.plotter.reset_camera()
            
    # ------------------------------------------------------------------
    # Evacuation Path Visualization
    # ------------------------------------------------------------------
    
    def display_evacuation_paths(self):
        """Display multicolored evacuation paths from every space to the nearest exit."""
        if not self.plotter or not self.current_model or not self.spatial_graph:
            return
        
        self.clear_evacuation_paths()
        
        if not self.spatial_graph.evacuation_paths:
            logger.info("No evacuation paths to display")
            return
        
        # Generate distinct colors using HSV colormap
        n_paths = len(self.spatial_graph.evacuation_paths)
        colors = self._generate_path_colors(n_paths)
        
        color_idx = 0
        for space_node_id, path in self.spatial_graph.evacuation_paths.items():
            if len(path) < 2:
                continue
            
            # Get 3D points along the path
            points = []
            for node_id in path:
                space_id = node_id.replace("space_", "")
                if space_id in self.current_model.spaces:
                    space = self.current_model.spaces[space_id]
                    if space.center:
                        points.append(space.center)
            
            if len(points) < 2:
                continue
            
            # Create a line from the points
            line_points = np.array(points)
            
            # Create tube for visibility
            try:
                # Use PolyData with lines
                poly = pv.PolyData(line_points)
                # Create line cells
                cells = np.full((len(points) - 1, 3), 2, dtype=np.int_)
                cells[:, 1] = np.arange(0, len(points) - 1)
                cells[:, 2] = np.arange(1, len(points))
                poly.lines = cells.flatten()
                
                # Tube for thickness
                tube = poly.tube(radius=0.15)
                
                color = colors[color_idx % len(colors)]
                color_idx += 1
                
                actor = self.plotter.add_mesh(
                    tube,
                    color=color,
                    opacity=0.7,
                    smooth_shading=True,
                    name=f"path_{space_node_id}"
                )
                self.path_actors[space_node_id] = actor
            except Exception as e:
                logger.warning(f"Could not render path for {space_node_id}: {e}")
        
        logger.info(f"Displayed {len(self.path_actors)} evacuation paths")
    
    def _generate_path_colors(self, n: int) -> List[Tuple[float, float, float]]:
        """Generate a list of distinct colors for paths."""
        import colorsys
        colors = []
        for i in range(max(n, 1)):
            h = (i * 0.618033988749895) % 1.0  # golden ratio for even distribution
            s = 0.7 + (i % 3) * 0.1
            v = 0.8 + (i % 2) * 0.1
            r, g, b = colorsys.hsv_to_rgb(h, min(s, 1.0), min(v, 1.0))
            colors.append((r, g, b))
        return colors
    
    def highlight_evacuation_path(self, space_id: str):
        """Highlight a single evacuation path and dim others."""
        if not self.plotter or not self.spatial_graph:
            return
        
        target_node_id = f"space_{space_id}"
        
        for path_id, actor in self.path_actors.items():
            if path_id == target_node_id:
                try:
                    actor.GetProperty().SetOpacity(1.0)
                    actor.GetProperty().SetLineWidth(5.0)
                except Exception:
                    pass
            else:
                try:
                    actor.GetProperty().SetOpacity(0.15)
                except Exception:
                    pass
        
        self.plotter.render()
    
    def clear_evacuation_paths(self):
        """Remove all evacuation path actors from the view."""
        if not self.plotter:
            return
        for actor in self.path_actors.values():
            try:
                self.plotter.remove_actor(actor)
            except Exception:
                pass
        self.path_actors.clear()
    
    def display_exits(self):
        """Display markers for annotated exits and virtual exits."""
        if not self.plotter or not self.current_model:
            return
        
        # Clear old exit markers
        for actor in self.exit_actors.values():
            try:
                self.plotter.remove_actor(actor)
            except Exception:
                pass
        self.exit_actors.clear()
        
        # Exit marker color: bright green
        exit_color = (0.1, 0.9, 0.2)
        
        # 1. Show annotated exit spaces
        if self.current_model.annotations:
            for space_id, ann in self.current_model.annotations.space_annotations.items():
                if ann.is_exit and space_id in self.current_model.spaces:
                    space = self.current_model.spaces[space_id]
                    if space.center:
                        sphere = pv.Sphere(radius=0.6, center=space.center, theta_resolution=12, phi_resolution=12)
                        actor = self.plotter.add_mesh(sphere, color=exit_color, opacity=0.9, name=f"exit_{space_id}")
                        self.exit_actors[f"exit_{space_id}"] = actor
            
            # 2. Show virtual exits
            for v_exit in self.current_model.annotations.exits:
                sphere = pv.Sphere(radius=0.6, center=v_exit.position, theta_resolution=12, phi_resolution=12)
                actor = self.plotter.add_mesh(sphere, color=exit_color, opacity=0.9, name=f"v_exit_{v_exit.id}")
                
                # Add label
                label_actor = self.plotter.add_point_labels(
                    [v_exit.position],
                    [v_exit.name or "Exit"],
                    font_size=10,
                    name=f"v_exit_label_{v_exit.id}"
                )
                self.exit_actors[f"v_exit_{v_exit.id}"] = actor
                self.exit_actors[f"v_exit_label_{v_exit.id}"] = label_actor
    
    def set_show_evacuation_paths(self, show: bool):
        """Toggle evacuation path visibility."""
        self.settings.show_evacuation_paths = show
        if show:
            self.display_evacuation_paths()
        else:
            self.clear_evacuation_paths()
            
    def _display_elements(self):
        """Display BIM building elements."""
        if not self.current_model or not self.plotter:
            return
            
        category_groups = {
            ElementCategory.WALL: [],
            ElementCategory.FLOOR: [],
            ElementCategory.DOOR: [],
            ElementCategory.WINDOW: [],
            ElementCategory.STAIR: [],
            ElementCategory.COLUMN: [],
            ElementCategory.BEAM: [],
            ElementCategory.FURNITURE: []
        }
        
        # Group elements by category
        for elem_id, elem in self.current_model.elements.items():
            if self.settings.level_filter and elem.level != self.settings.level_filter:
                continue
            if elem.category in category_groups:
                category_groups[elem.category].append(elem)
                
        # Display walls
        if self.settings.show_walls and ElementCategory.WALL in category_groups:
            self._display_element_category(
                category_groups[ElementCategory.WALL],
                "walls",
                self.settings.wall_color,
                opacity=self.settings.wall_opacity
            )
            
        # Display floors
        if self.settings.show_floors and category_groups[ElementCategory.FLOOR]:
            self._display_element_category(
                category_groups[ElementCategory.FLOOR],
                "floors",
                self.settings.floor_color,
                opacity=0.8
            )
            
        # Display doors
        if self.settings.show_doors and category_groups[ElementCategory.DOOR]:
            self._display_element_category(
                category_groups[ElementCategory.DOOR],
                "doors",
                self.settings.door_color,
                opacity=1.0
            )
            
        # Display windows
        if self.settings.show_windows and category_groups[ElementCategory.WINDOW]:
            self._display_element_category(
                category_groups[ElementCategory.WINDOW],
                "windows",
                self.settings.window_color,
                opacity=0.4
            )
            
        # Display stairs
        if self.settings.show_stairs and category_groups[ElementCategory.STAIR]:
            self._display_element_category(
                category_groups[ElementCategory.STAIR],
                "stairs",
                (0.6, 0.6, 0.7),
                opacity=0.9
            )
            
        # Display furniture
        if self.settings.show_furniture and category_groups[ElementCategory.FURNITURE]:
            self._display_element_category(
                category_groups[ElementCategory.FURNITURE],
                "furniture",
                (0.7, 0.6, 0.5),
                opacity=0.8
            )
                
    def _display_element_category(
        self,
        elements: List[BIMElement],
        category_name: str,
        color: Tuple[float, float, float],
        opacity: float = 1.0
    ):
        """Display a category of elements as combined mesh."""
        if not self.plotter:
            return
            
        combined_points = []
        combined_faces = []
        vertex_offset = 0
        
        for elem in elements:
            if getattr(elem, 'geometry', None):
                verts = elem.geometry["vertices"]
                faces = elem.geometry["faces"]
                
                combined_points.extend(verts)
                # Adjust face indices by vertex_offset
                for i in range(0, len(faces), 4):
                    combined_faces.extend([3, faces[i+1] + vertex_offset, faces[i+2] + vertex_offset, faces[i+3] + vertex_offset])
                    
                vertex_offset += len(verts)
            elif elem.bounds:
                # Create a simple box from bounds
                (min_x, min_y, min_z), (max_x, max_y, max_z) = elem.bounds
                
                # Create 8 vertices for the box
                verts = [
                    [min_x, min_y, min_z],
                    [max_x, min_y, min_z],
                    [max_x, max_y, min_z],
                    [min_x, max_y, min_z],
                    [min_x, min_y, max_z],
                    [max_x, min_y, max_z],
                    [max_x, max_y, max_z],
                    [min_x, max_y, max_z]
                ]
                
                # Create 12 triangular faces
                faces = [
                    # Bottom
                    3, 0 + vertex_offset, 1 + vertex_offset, 2 + vertex_offset,
                    3, 0 + vertex_offset, 2 + vertex_offset, 3 + vertex_offset,
                    # Top
                    3, 4 + vertex_offset, 5 + vertex_offset, 6 + vertex_offset,
                    3, 4 + vertex_offset, 6 + vertex_offset, 7 + vertex_offset,
                    # Front
                    3, 0 + vertex_offset, 1 + vertex_offset, 5 + vertex_offset,
                    3, 0 + vertex_offset, 5 + vertex_offset, 4 + vertex_offset,
                    # Back
                    3, 2 + vertex_offset, 3 + vertex_offset, 7 + vertex_offset,
                    3, 2 + vertex_offset, 7 + vertex_offset, 6 + vertex_offset,
                    # Left
                    3, 0 + vertex_offset, 3 + vertex_offset, 7 + vertex_offset,
                    3, 0 + vertex_offset, 7 + vertex_offset, 4 + vertex_offset,
                    # Right
                    3, 1 + vertex_offset, 2 + vertex_offset, 6 + vertex_offset,
                    3, 1 + vertex_offset, 6 + vertex_offset, 5 + vertex_offset
                ]
                
                combined_points.extend(verts)
                combined_faces.extend(faces)
                vertex_offset += 8
                    
        if combined_points:
            mesh = pv.PolyData(
                np.array(combined_points),
                np.array(combined_faces) if combined_faces else None
            )
            
            actor = self.plotter.add_mesh(
                mesh,
                color=color,
                opacity=opacity,
                smooth_shading=True,
                name=f"elements_{category_name}"
            )
            self.actors[f"elements_{category_name}"] = actor
            
    def _display_spaces(self):
        """Display spaces as colored regions."""
        if not self.current_model or not self.settings.show_spaces or not self.plotter:
            return
            
        for space_id, space in self.current_model.spaces.items():
            if self.settings.level_filter and space.level != self.settings.level_filter:
                continue
            if not space.center:
                continue
                
            # Get color based on category
            color = self.settings.space_colors.get(
                space.category,
                (0.9, 0.9, 0.9)
            )
            
            # Create space representation
            center = space.center
            radius = max(0.5, (space.area / np.pi) ** 0.5 * 0.3) if space.area > 0 else 1.0
            
            mesh = None
            if getattr(space, 'geometry', None) and isinstance(space.geometry, dict) and "boundary" in space.geometry:
                boundary = space.geometry["boundary"]
                if boundary and len(boundary) >= 3:
                    pts = np.array([[pt[0], pt[1], center[2] + 0.05] for pt in boundary], dtype=np.float32)
                    faces = np.hstack([[len(pts)], np.arange(len(pts))])
                    mesh = pv.PolyData(pts, faces)
                    
            if mesh is None:
                mesh = pv.Sphere(
                    radius=radius,
                    center=center,
                    theta_resolution=16,
                    phi_resolution=16
                )
            
            actor = self.plotter.add_mesh(
                mesh,
                color=color,
                opacity=0.6,
                name=f"space_{space_id}"
            )
            self.space_actors[space_id] = actor
            
            # Add label if enabled
            if self.settings.show_labels:
                # Offset label slightly upward to prevent z-fighting with the sphere/floor
                label_pos = [center[0], center[1], center[2] + radius + 0.2]
                label_actor = self.plotter.add_point_labels(
                    [label_pos],
                    [space.name],
                    font_size=12,
                    name=f"label_{space_id}",
                    text_color="black",
                    shape_color="white",
                    shape_opacity=0.5
                )
                self.label_actors[space_id] = label_actor
                
    def _display_spatial_graph(self):
        """Display the spatial graph connections."""
        if not self.spatial_graph or not self.plotter:
            return
            
        # Color map for connection types
        conn_colors = {
            "door": (0.2, 0.8, 0.2),
            "stair": (0.8, 0.2, 0.2),
            "elevator": (0.2, 0.2, 0.8),
            "corridor": (0.9, 0.6, 0.2),
            "open": (0.6, 0.6, 0.6),
            "exit": (0.1, 0.9, 0.2),
        }
        
        # Draw connections as lines
        i = 0
        for conn in self.spatial_graph.connections.values():
            source = self.spatial_graph.nodes.get(conn.source_id)
            target = self.spatial_graph.nodes.get(conn.target_id)
            
            if source and target:
                line = pv.Line(source.center, target.center)
                color = conn_colors.get(conn.connection_type, (0.5, 0.5, 0.5))
                        
                self.plotter.add_mesh(
                    line,
                    color=color,
                    line_width=2,
                    name=f"graph_line_{i}"
                )
                i += 1
                
        # Draw nodes
        for node_id, node in self.spatial_graph.nodes.items():
            sphere = pv.Sphere(
                radius=0.5,
                center=node.center,
                theta_resolution=8,
                phi_resolution=8
            )
            
            actor = self.plotter.add_mesh(
                sphere,
                color=(0.2, 0.4, 0.8),
                opacity=0.6,
                name=f"graph_node_{node_id}"
            )
            
    # ------------------------------------------------------------------
    # Mesa-version-safe agent iterator
    # ------------------------------------------------------------------
    @staticmethod
    def _get_agents(simulation: 'BIMSimulationModel') -> list:
        """Return a list of agents compatible with Mesa 2.x and 3.x."""
        # Prefer the model's own version-safe method
        if hasattr(simulation, '_get_all_agents'):
            return simulation._get_all_agents()
        sched = simulation.schedule
        # Mesa 2.4 AgentSet via .agents property
        if hasattr(sched, 'agents'):
            try:
                return list(sched.agents)
            except Exception:
                pass
        # Mesa 2.x SimultaneousActivation stores in ._agents dict
        if hasattr(sched, '_agents'):
            if isinstance(sched._agents, dict):
                return list(sched._agents.values())
            try:
                return list(sched._agents)
            except Exception:
                pass
        # Last resort: our own registry dict
        if hasattr(simulation, '_agent_registry'):
            return list(simulation._agent_registry.values())
        return []

    def load_simulation(self, simulation: 'BIMSimulationModel'):
        """Load a simulation for visualization."""
        self.current_simulation = simulation

        # If BIM model not yet loaded, load it first
        if not self.current_model and simulation.bim_model:
            self.load_bim_model(simulation.bim_model)

        # Draw agents immediately so they appear before the user presses Start
        if self.plotter:
            self._update_agents()

        logger.info(f"Simulation loaded into 3D view ({len(self._get_agents(simulation))} agents)")
        
    def start_simulation_visualization(self):
        """Start real-time simulation visualization updates."""
        if self.update_timer:
            self.update_timer.start(self.update_interval)
            logger.info("Simulation visualization started")
            
    def stop_simulation_visualization(self):
        """Stop simulation visualization updates."""
        if self.update_timer:
            self.update_timer.stop()
            logger.info("Simulation visualization stopped")
            
    def _update_simulation_visualization(self):
        """Update visualization for current simulation step."""
        if not self.current_simulation or not self.plotter:
            return
            
        # Update agent positions
        self._update_agents()
        
        # Update density visualization if needed
        if self.settings.show_agents:
            self._update_density_map()
            
    def _update_agents(self):
        """Update agent representations in 3D view."""
        if not self.current_simulation or not self.plotter:
            return

        agents = self._get_agents(self.current_simulation)

        # Remove actors for agents that no longer exist
        live_ids = {a.unique_id for a in agents}
        dead_ids = set(self.agent_actors.keys()) - live_ids
        for agent_id in dead_ids:
            try:
                self.plotter.remove_actor(self.agent_actors[agent_id])
            except Exception:
                pass
            del self.agent_actors[agent_id]

        # Update or create actor for each live agent
        for agent in agents:
            # Remove stale actor (position may have changed)
            if agent.unique_id in self.agent_actors:
                try:
                    self.plotter.remove_actor(self.agent_actors[agent.unique_id])
                except Exception:
                    pass

            position = agent.position
            color = self.settings.agent_state_colors.get(
                agent.state.value, (0.5, 0.5, 0.5)
            )

            # Scale by agent type
            size = self.settings.agent_size
            if agent.profile.agent_type.value == "vehicle":
                size *= 2.0
            elif agent.profile.agent_type.value == "autonomous":
                size *= 0.6

            # Use a taller, more visible cylinder-on-sphere body
            body_height = size * 3.0
            body = pv.Cylinder(
                center=(
                    position[0],
                    position[1],
                    position[2] + size + body_height / 2.0,
                ),
                direction=(0, 0, 1),
                radius=size,
                height=body_height,
                resolution=12,
            )
            head = pv.Sphere(
                radius=size * 0.8,
                center=(
                    position[0],
                    position[1],
                    position[2] + size + body_height + size * 0.8,
                ),
                theta_resolution=10,
                phi_resolution=10,
            )
            mesh = body.merge(head)

            actor = self.plotter.add_mesh(
                mesh,
                color=color,
                opacity=0.95,
                name=f"agent_{agent.unique_id}",
            )
            self.agent_actors[agent.unique_id] = actor

        # Draw trail lines if enabled
        if self.settings.show_trails:
            self._update_trails(agents)

        # Render immediately so agents appear without waiting for the timer
        try:
            self.plotter.render()
        except Exception:
            pass
            
    def _update_trails(self, agents):
        """Draw trail lines from each agent's position_history."""
        if not self.plotter:
            return

        # Remove stale trail actors for agents that no longer exist
        live_ids = {a.unique_id for a in agents}
        dead_trail_ids = set(self.trail_actors.keys()) - live_ids
        for agent_id in dead_trail_ids:
            try:
                self.plotter.remove_actor(self.trail_actors[agent_id])
            except Exception:
                pass
            del self.trail_actors[agent_id]

        trail_length = self.settings.agent_trail_length

        for agent in agents:
            # Remove previous trail actor for this agent
            if agent.unique_id in self.trail_actors:
                try:
                    self.plotter.remove_actor(self.trail_actors[agent.unique_id])
                except Exception:
                    pass

            history = list(agent.position_history)[-trail_length:]
            if len(history) < 2:
                continue

            # Build 3D point array from position history
            points = np.array(history, dtype=np.float64)
            # Ensure 3D (some histories may be 2-tuples)
            if points.shape[1] < 3:
                z_col = np.zeros((points.shape[0], 1))
                points = np.hstack([points, z_col])

            # Create polyline
            n_pts = len(points)
            poly = pv.PolyData(points)
            cells = np.full((n_pts - 1, 3), 2, dtype=np.int_)
            cells[:, 1] = np.arange(0, n_pts - 1)
            cells[:, 2] = np.arange(1, n_pts)
            poly.lines = cells.flatten()

            try:
                tube = poly.tube(radius=0.05)
                # Use the agent's state color with reduced opacity
                color = self.settings.agent_state_colors.get(
                    agent.state.value, (0.5, 0.5, 0.5)
                )
                actor = self.plotter.add_mesh(
                    tube,
                    color=color,
                    opacity=0.4,
                    smooth_shading=True,
                    name=f"trail_{agent.unique_id}",
                )
                self.trail_actors[agent.unique_id] = actor
            except Exception as e:
                logger.debug(f"Could not render trail for agent {agent.unique_id}: {e}")

    def _update_density_map(self):
        """Update density heat map visualization."""
        if not self.current_simulation:
            return
            
        density = self.current_simulation.density_map
        
        # Update space colors based on density
        for space_id, actor in self.space_actors.items():
            count = density.get(space_id, 0)
            if count > 0 and space_id in self.current_model.spaces:
                space = self.current_model.spaces[space_id]
                if space.area > 0:
                    density_value = count / space.area
                    
                    # Color from green (low) to red (high)
                    if density_value < 0.5:
                        color = (0.2, 1.0, 0.2)
                    elif density_value < 1.0:
                        color = (1.0, 1.0, 0.2)
                    elif density_value < 2.0:
                        color = (1.0, 0.6, 0.2)
                    else:
                        color = (1.0, 0.2, 0.2)
                        
                    # Update actor color
                    try:
                        actor.GetProperty().SetColor(color)
                    except Exception:
                        pass
                        
    def set_visualization_mode(self, mode: VisualizationMode):
        """Set the visualization mode."""
        if mode == VisualizationMode.BIM_ONLY:
            self.settings.show_agents = False
            self.settings.show_navigation = False
        elif mode == VisualizationMode.AGENTS:
            self.settings.show_agents = True
            self.settings.show_trails = True
        elif mode == VisualizationMode.DENSITY:
            self.settings.show_agents = True
        elif mode == VisualizationMode.EVACUATION:
            self.settings.show_agents = True
            self.settings.show_trails = True
            self.settings.show_evacuation_paths = True
        elif mode == VisualizationMode.NAVIGATION:
            self.settings.show_navigation = True
            self.settings.show_spatial_graph = True
            self.settings.show_evacuation_paths = True
            
        self.refresh()
        
    def set_settings(self, settings: VisualizationSettings):
        """Update visualization settings."""
        self.settings = settings
        self.refresh()
        
    def refresh(self):
        """Refresh the 3D view."""
        if self.current_model:
            self.load_bim_model(self.current_model)
        if self.current_simulation:
            self._update_agents()
            
    def clear(self):
        """Clear all visualization."""
        if not self.plotter:
            return
            
        self.plotter.clear()
        self.actors.clear()
        self.agent_actors.clear()
        self.trail_actors.clear()
        self.space_actors.clear()
        self.element_actors.clear()
        self.label_actors.clear()
        self.path_actors.clear()
        self.exit_actors.clear()
        
    def reset_camera(self):
        """Reset camera to default view."""
        if self.plotter:
            self.plotter.reset_camera()
            
    def set_view(self, view: str):
        """Set camera view (top, front, side, isometric)."""
        if not self.plotter:
            return
            
        views = {
            "top": (0, 0, 1),
            "front": (0, -1, 0),
            "side": (1, 0, 0),
            "isometric": (1, 1, 1)
        }
        
        if view in views:
            self.plotter.view_vector(views[view])
            
    def take_screenshot(self, filename: str):
        """Save a screenshot of the current view."""
        if self.plotter:
            self.plotter.screenshot(filename)
            logger.info(f"Screenshot saved: {filename}")
            
    def export_scene(self, filename: str):
        """Export the current 3D scene."""
        if self.plotter:
            self.plotter.export_obj(filename)
            logger.info(f"Scene exported: {filename}")
            
    def _on_element_picked(self, picked):
        """Handle element picking in 3D view."""
        if picked is None:
            return
            
        logger.info(f"Picked element: {picked}")
        
        # Try to identify what was picked
        def is_match(actor, picked_item):
            if actor == picked_item:
                return True
            try:
                if hasattr(actor, 'mapper') and actor.mapper and actor.mapper.dataset == picked_item:
                    return True
                if hasattr(actor, 'GetMapper') and actor.GetMapper() and actor.GetMapper().GetInput() == picked_item:
                    return True
            except Exception:
                pass
            return False
            
        for agent_id, actor in self.agent_actors.items():
            if is_match(actor, picked):
                self.agent_selected.emit(agent_id)
                return
                
        for space_id, actor in self.space_actors.items():
            if is_match(actor, picked):
                self.space_selected.emit(space_id)
                return
                

    def toggle_element_category(self, category: ElementCategory, visible: bool):
        """Toggle visibility of an element category."""
        category_names = {
            ElementCategory.WALL: "walls",
            ElementCategory.FLOOR: "floors",
            ElementCategory.DOOR: "doors",
            ElementCategory.WINDOW: "windows",
            ElementCategory.STAIR: "stairs",
            ElementCategory.FURNITURE: "furniture"
        }
        
        name = category_names.get(category)
        if name and f"elements_{name}" in self.actors:
            actor = self.actors[f"elements_{name}"]
            actor.SetVisibility(visible)
            
    def highlight_space(self, space_id: str, color: Tuple[float, float, float] = (1.0, 0.0, 0.0)):
        """Highlight a specific space."""
        if space_id in self.space_actors:
            try:
                self.space_actors[space_id].GetProperty().SetColor(color)
            except Exception:
                pass
                
    def highlight_agent(self, agent_id: int, color: Tuple[float, float, float] = (1.0, 1.0, 0.0)):
        """Highlight a specific agent."""
        if agent_id in self.agent_actors:
            try:
                self.agent_actors[agent_id].GetProperty().SetColor(color)
            except Exception:
                pass
                
    def focus_on_space(self, space_id: str):
        """Focus camera on a specific space."""
        if not self.current_model or not self.plotter:
            return
            
        if space_id in self.current_model.spaces:
            space = self.current_model.spaces[space_id]
            if space.center:
                self.plotter.camera.SetFocalPoint(space.center)
                self.plotter.reset_camera()
                
    def focus_on_agent(self, agent_id: int):
        """Focus camera on a specific agent."""
        if not self.current_simulation or not self.plotter:
            return
            
        for agent in self._get_agents(self.current_simulation):
            if agent.unique_id == agent_id:
                self.plotter.camera.SetFocalPoint(
                    agent.position[0],
                    agent.position[1],
                    agent.position[2]
                )
                break
