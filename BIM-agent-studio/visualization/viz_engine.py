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

logging.basicConfig(level=logging.INFO)
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
    show_labels: bool = False
    show_navigation: bool = False
    show_spatial_graph: bool = False
    level_filter: Optional[str] = None
    
    # Agent visualization
    show_agents: bool = True
    agent_size: float = 0.3
    show_trails: bool = False
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
            self.plotter.enable_element_picking(
                callback=self._on_element_picked,
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
            
        # Reset camera
        if self.plotter:
            self.plotter.reset_camera()
            
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
        if self.settings.show_walls and category_groups[ElementCategory.WALL]:
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
            if elem.geometry:
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
                
                # Box vertices
                vertices = [
                    [min_x, min_y, min_z],
                    [max_x, min_y, min_z],
                    [max_x, max_y, min_z],
                    [min_x, max_y, min_z],
                    [min_x, min_y, max_z],
                    [max_x, min_y, max_z],
                    [max_x, max_y, max_z],
                    [min_x, max_y, max_z]
                ]
                
                # Box faces (quads)
                faces = [
                    [0, 1, 2, 3],  # bottom
                    [4, 5, 6, 7],  # top
                    [0, 1, 5, 4],  # front
                    [2, 3, 7, 6],  # back
                    [1, 2, 6, 5],  # right
                    [0, 3, 7, 4]   # left
                ]
                
                combined_points.extend(vertices)
                for face in faces:
                    combined_faces.append([4] + [v + vertex_offset for v in face])
                vertex_offset += 8
                    
        if combined_points:
            mesh = pv.PolyData(
                np.array(combined_points),
                np.hstack(combined_faces) if combined_faces else None
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
        if not self.current_model or not self.plotter:
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
            
            # Create space representation (sphere at center for now)
            center = space.center
            radius = max(0.5, (space.area / np.pi) ** 0.5 * 0.3) if space.area > 0 else 1.0
            
            sphere = pv.Sphere(
                radius=radius,
                center=center,
                theta_resolution=16,
                phi_resolution=16
            )
            
            actor = self.plotter.add_mesh(
                sphere,
                color=color,
                opacity=0.3,
                name=f"space_{space_id}"
            )
            self.space_actors[space_id] = actor
            
            # Add label if enabled
            if self.settings.show_labels:
                label_actor = self.plotter.add_point_labels(
                    [center],
                    [space.name],
                    font_size=10,
                    name=f"label_{space_id}"
                )
                self.label_actors[space_id] = label_actor
                
    def _display_spatial_graph(self):
        """Display the spatial graph connections."""
        if not self.spatial_graph or not self.plotter:
            return
            
        # Draw connections as lines
        lines = []
        for conn in self.spatial_graph.connections.values():
            source = self.spatial_graph.nodes.get(conn.source_id)
            target = self.spatial_graph.nodes.get(conn.target_id)
            
            if source and target:
                lines.append([source.center, target.center])
                
        if lines:
            for i, line_points in enumerate(lines):
                line = pv.Line(*line_points)
                
                # Color by connection type
                color = (0.5, 0.5, 0.5)  # default gray
                if hasattr(line_points, 'connection_type'):
                    if line_points.connection_type == "door":
                        color = (0.2, 0.8, 0.2)
                    elif line_points.connection_type == "stair":
                        color = (0.8, 0.2, 0.2)
                    elif line_points.connection_type == "elevator":
                        color = (0.2, 0.2, 0.8)
                        
                actor = self.plotter.add_mesh(
                    line,
                    color=color,
                    line_width=2,
                    name=f"graph_line_{i}"
                )
                
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

        # Render immediately so agents appear without waiting for the timer
        try:
            self.plotter.render()
        except Exception:
            pass
            
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
            self.settings.show_trails = False
        elif mode == VisualizationMode.DENSITY:
            self.settings.show_agents = True
        elif mode == VisualizationMode.EVACUATION:
            self.settings.show_agents = True
        elif mode == VisualizationMode.NAVIGATION:
            self.settings.show_navigation = True
            self.settings.show_spatial_graph = True
            
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
        self.space_actors.clear()
        self.element_actors.clear()
        self.label_actors.clear()
        
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
        for agent_id, actor in self.agent_actors.items():
            if actor == picked:
                self.agent_selected.emit(agent_id)
                return
                
        for space_id, actor in self.space_actors.items():
            if actor == picked:
                self.space_selected.emit(space_id)
                return
                
    def show_legend(self):
        """Show legend for current visualization."""
        if not self.plotter:
            return
            
        legend_entries = []
        
        # Agent states
        for state, color in self.settings.agent_state_colors.items():
            legend_entries.append([state.capitalize(), color])
            
        # Space types
        for space_type, color in self.settings.space_colors.items():
            legend_entries.append([space_type.capitalize(), color])
            
        self.plotter.add_legend(legend_entries)
        
    def show_scale_bar(self):
        """Show scale bar in the 3D view."""
        if self.plotter:
            self.plotter.add_scalar_bar("Density")
            
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
            
        for agent in self.current_simulation.schedule.agents:
            if agent.unique_id == agent_id:
                self.plotter.camera.SetFocalPoint(
                    agent.position[0],
                    agent.position[1],
                    agent.position[2]
                )
                break
