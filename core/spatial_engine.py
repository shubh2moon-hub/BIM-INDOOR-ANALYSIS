"""
Spatial Intelligence Engine
Handles space recognition, connectivity detection, and navigation network creation.
"""

import logging
import uuid
from typing import Dict, List, Optional, Set, Tuple, Any
from dataclasses import dataclass, field
from collections import defaultdict

import numpy as np
import networkx as nx
from scipy.spatial import distance
from shapely.geometry import Polygon, Point, LineString
from shapely.ops import unary_union

from core.bim_processor import BIMModel, BIMSpace, BIMElement, ElementCategory

logger = logging.getLogger(__name__)


@dataclass
class SpaceNode:
    """A node in the spatial graph representing a space."""
    id: str
    space_id: str
    name: str
    category: str
    level: str
    center: Tuple[float, float, float]
    area: float
    capacity: int = 0
    attributes: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Connection:
    """A connection between two spaces."""
    id: str
    source_id: str
    target_id: str
    connection_type: str  # door, corridor, stair, elevator, open
    weight: float
    accessible: bool = True
    width: float = 1.0
    attributes: Dict[str, Any] = field(default_factory=dict)


@dataclass
class NavigationNode:
    """A node in the navigation network."""
    id: str
    position: Tuple[float, float, float]
    space_id: str
    node_type: str  # waypoint, door, stair, elevator, entrance
    connections: List[str] = field(default_factory=list)


@dataclass
class SpatialGraph:
    """Complete spatial graph of the building."""
    id: str
    name: str
    nodes: Dict[str, SpaceNode] = field(default_factory=dict)
    connections: Dict[str, Connection] = field(default_factory=dict)
    navigation_nodes: Dict[str, NavigationNode] = field(default_factory=dict)
    network: nx.Graph = field(default_factory=nx.Graph)
    nav_network: nx.Graph = field(default_factory=nx.Graph)
    level_graphs: Dict[str, nx.Graph] = field(default_factory=dict)
    # NEW: precomputed evacuation paths (space_id -> list of space_ids)
    evacuation_paths: Dict[str, List[str]] = field(default_factory=dict)
    exit_nodes: List[str] = field(default_factory=list)  # space node IDs that are exits


class SpatialIntelligenceEngine:
    """Engine for spatial analysis and navigation graph generation."""

    def __init__(self):
        self.spatial_graph: Optional[SpatialGraph] = None
        self.current_model: Optional[BIMModel] = None

    def process_model(self, model: BIMModel, annotations=None) -> SpatialGraph:
        """Process a BIM model and generate spatial intelligence.
        
        Args:
            model: The BIM model to process.
            annotations: Optional ModelAnnotations with user overrides.
        """
        logger.info("Starting spatial intelligence processing...")
        self.current_model = model

        graph = SpatialGraph(
            id=str(uuid.uuid4()),
            name=f"Spatial Graph - {model.name}"
        )
            
        self.spatial_graph = graph

        # Step 1: Create space nodes (with category overrides from annotations)
        logger.info("Creating space nodes...")
        self._create_space_nodes(model, graph, annotations)

        # Step 2: Detect connectivity between spaces
        logger.info("Detecting space connectivity...")
        self._detect_connectivity(model, graph, annotations)

        # Step 3: Build navigation network
        logger.info("Building navigation network...")
        self._build_navigation_network(model, graph, annotations)

        # Step 4: Create NetworkX graph
        logger.info("Creating NetworkX graphs...")
        self._create_networkx_graphs(graph, annotations)

        # Step 5: Analyze spatial properties
        logger.info("Analyzing spatial properties...")
        self._analyze_spatial_properties(graph)

        # Step 6: Compute evacuation paths
        logger.info("Computing evacuation paths...")
        self._compute_evacuation_paths(graph, annotations)

        self.spatial_graph = graph
        model.spatial_graph = graph

        logger.info(f"Spatial processing complete: {len(graph.nodes)} spaces, {len(graph.connections)} connections, {len(graph.evacuation_paths)} evacuation paths")
        return graph

    def _create_space_nodes(self, model: BIMModel, graph: SpatialGraph, annotations=None):
        """Create nodes for each space in the building."""
        for space_id, space in model.spaces.items():
            # Determine effective category (annotation overrides IFC default)
            category = space.category
            custom_name = space.name
            is_exit = False
            if annotations and space_id in annotations.space_annotations:
                ann = annotations.space_annotations[space_id]
                if ann.category_override:
                    category = ann.category_override
                if ann.custom_name:
                    custom_name = ann.custom_name
                is_exit = ann.is_exit

            # Estimate capacity based on area (1 person per 10 sqm default)
            capacity = int(space.area / 10) if space.area > 0 else 5

            # Determine if space is navigable
            navigable = category not in ["wall", "void", "exterior"]

            node = SpaceNode(
                id=f"space_{space_id}",
                space_id=space_id,
                name=custom_name,
                category=category,
                level=space.level,
                center=space.center or (0, 0, 0),
                area=space.area,
                capacity=capacity,
                attributes={
                    "navigable": navigable,
                    "volume": space.volume,
                    "long_name": space.long_name,
                    "is_exit": is_exit,
                }
            )
            graph.nodes[node.id] = node

            # Track exit nodes
            if is_exit or category in ["exit", "entrance", "lobby"]:
                if node.id not in graph.exit_nodes:
                    graph.exit_nodes.append(node.id)

        logger.info(f"  Created {len(graph.nodes)} space nodes, {len(graph.exit_nodes)} marked as exits")

    def _detect_connectivity(self, model: BIMModel, graph: SpatialGraph, annotations=None):
        """Detect connectivity between spaces using doors, corridors, and stairs."""
        # Method 1: Door-based connectivity
        self._detect_door_connectivity(model, graph)

        # Method 2: Proximity-based connectivity
        self._detect_proximity_connectivity(model, graph)

        # Method 3: Vertical connectivity (stairs, elevators)
        self._detect_vertical_connectivity(model, graph)

        # Method 4: Corridor-based connectivity
        self._detect_corridor_connectivity(model, graph)

        # NEW: Method 5: Virtual connections from annotations (missing doors)
        self._add_virtual_connections(graph, annotations)

    def _detect_door_connectivity(self, model: BIMModel, graph: SpatialGraph):
        """Detect connections through doors."""
        doors = [e for e in model.elements.values() if e.category == ElementCategory.DOOR]

        for door in doors:
            # Find spaces adjacent to this door
            adjacent_spaces = self._find_adjacent_spaces_to_door(model, door)

            if len(adjacent_spaces) == 1:
                # Exterior door (only connects to 1 space).
                # Only treat it as an EXIT if it opens from a corridor.
                space = adjacent_spaces[0]
                if space.category == "corridor":
                    exit_id = f"exit_{door.id}"
                    
                    # Add virtual exit spatial node
                    graph.nodes[exit_id] = SpaceNode(
                        id=exit_id,
                        space_id=exit_id,
                        name=f"{door.name} (Exterior Exit)",
                        category="exit",
                        level=door.level,
                        center=door.center or (0,0,0),
                        area=5.0,
                        capacity=100
                    )
                    graph.exit_nodes.append(exit_id)
                    
                    # Add connection
                    connection = Connection(
                        id=f"conn_{door.id}_exit",
                        source_id=f"space_{space.id}",
                        target_id=exit_id,
                        connection_type="door",
                        weight=1.0,
                        width=door.properties.get("Width", 1.0),
                        attributes={
                            "door_id": door.id,
                            "door_name": door.name,
                            "is_exterior": True
                        }
                    )
                    graph.connections[connection.id] = connection

            elif len(adjacent_spaces) >= 2:
                # Create connections between all pairs of adjacent spaces
                for i in range(len(adjacent_spaces)):
                    for j in range(i + 1, len(adjacent_spaces)):
                        space_a = adjacent_spaces[i]
                        space_b = adjacent_spaces[j]

                        # Calculate distance between space centers
                        dist = self._distance_between(
                            space_a.center or (0, 0, 0),
                            space_b.center or (0, 0, 0)
                        )

                        connection = Connection(
                            id=f"conn_{door.id}_{i}_{j}",
                            source_id=f"space_{space_a.id}",
                            target_id=f"space_{space_b.id}",
                            connection_type="door",
                            weight=dist,
                            width=door.properties.get("Width", 0.9),
                            attributes={
                                "door_id": door.id,
                                "door_name": door.name,
                                "is_external": False
                            }
                        )
                        graph.connections[connection.id] = connection

    def _add_virtual_connections(self, graph: SpatialGraph, annotations=None):
        """Add user-defined virtual connections (e.g. missing doors)."""
        if not annotations or not annotations.virtual_connections:
            return
        for conn in annotations.virtual_connections:
            source_id = f"space_{conn.source_space_id}"
            target_id = f"space_{conn.target_space_id}"
            if source_id not in graph.nodes or target_id not in graph.nodes:
                logger.warning(f"Virtual connection references unknown space(s): {conn.source_space_id} -> {conn.target_space_id}")
                continue
            connection = Connection(
                id=f"conn_virt_{conn.id}",
                source_id=source_id,
                target_id=target_id,
                connection_type=conn.connection_type,
                weight=1.0,
                width=conn.width,
                attributes={"virtual": True, "connection_id": conn.id}
            )
            graph.connections[connection.id] = connection
            logger.info(f"  Added virtual connection: {conn.source_space_id} -> {conn.target_space_id}")

    def _build_navigation_network(self, model: BIMModel, graph: SpatialGraph, annotations=None):
        """Build a detailed navigation network with waypoints."""
        # Create navigation nodes at space centers
        for node_id, space_node in graph.nodes.items():
            nav_node = NavigationNode(
                id=f"nav_{space_node.space_id}",
                position=space_node.center,
                space_id=space_node.space_id,
                node_type="waypoint"
            )
            graph.navigation_nodes[nav_node.id] = nav_node

        # Create navigation nodes at door positions
        doors = [e for e in model.elements.values() if e.category == ElementCategory.DOOR]
        for door in doors:
            if door.center:
                nav_node = NavigationNode(
                    id=f"nav_door_{door.id}",
                    position=door.center,
                    space_id="",
                    node_type="door"
                )
                graph.navigation_nodes[nav_node.id] = nav_node

        # Create navigation nodes at stair positions
        stairs = [e for e in model.elements.values() if e.category == ElementCategory.STAIR]
        for stair in stairs:
            if stair.center:
                nav_node = NavigationNode(
                    id=f"nav_stair_{stair.id}",
                    position=stair.center,
                    space_id="",
                    node_type="stair"
                )
                graph.navigation_nodes[nav_node.id] = nav_node

        # NEW: Create navigation nodes for virtual exits
        if annotations:
            for v_exit in annotations.exits:
                nav_node = NavigationNode(
                    id=f"nav_exit_{v_exit.id}",
                    position=v_exit.position,
                    space_id="",
                    node_type="exit"
                )
                graph.navigation_nodes[nav_node.id] = nav_node
                # Connect to nearby spaces on same level
                for space_id, space in model.spaces.items():
                    if space.center and space.level == v_exit.level_id:
                        dist = self._distance_between(space.center, v_exit.position)
                        # Ensure exit connects back to the space
                        if dist < 15.0:
                            conn_id = f"conn_exit_{v_exit.id}_{space_id}"
                            graph.connections[conn_id] = Connection(
                                id=conn_id,
                                source_id=f"space_{space_id}",
                                target_id="__exit__",
                                connection_type="exit",
                                weight=dist,
                                width=v_exit.width,
                                accessible=v_exit.accessible,
                                attributes={"virtual_exit_id": v_exit.id, "position": list(v_exit.position)}
                            )
                            # Add exit to exit_nodes if not already there
                            sid = f"space_{space_id}"
                            if sid not in graph.exit_nodes:
                                graph.exit_nodes.append(sid)

        # Connect navigation nodes
        self._connect_navigation_nodes(graph)

    def _create_networkx_graphs(self, graph: SpatialGraph, annotations=None):
        """Create NetworkX graphs for pathfinding."""
        # Main space graph
        G = nx.Graph()

        # Add nodes
        for node_id, space_node in graph.nodes.items():
            # Check if blocked by annotation
            blocked = False
            if annotations and space_node.space_id in annotations.space_annotations:
                blocked = annotations.space_annotations[space_node.space_id].block_path

            G.add_node(
                node_id,
                name=space_node.name,
                category=space_node.category,
                level=space_node.level,
                center=space_node.center,
                area=space_node.area,
                capacity=space_node.capacity,
                blocked=blocked,
            )

        # Add edges
        for conn_id, conn in graph.connections.items():
            # Skip edges involving blocked nodes
            if G.has_node(conn.source_id) and G.nodes[conn.source_id].get("blocked", False):
                continue
            if G.has_node(conn.target_id) and G.nodes[conn.target_id].get("blocked", False):
                continue
            G.add_edge(
                conn.source_id,
                conn.target_id,
                weight=conn.weight,
                connection_type=conn.connection_type,
                width=conn.width,
                accessible=conn.accessible
            )

        graph.network = G

        # Create per-level graphs
        levels = set(node.level for node in graph.nodes.values())
        for level in levels:
            level_nodes = [n for n in graph.nodes.values() if n.level == level]
            level_G = nx.Graph()

            for node in level_nodes:
                level_G.add_node(
                    node.id,
                    name=node.name,
                    category=node.category,
                    center=node.center,
                    area=node.area
                )

            # Add edges for this level
            for conn in graph.connections.values():
                source_node = graph.nodes.get(conn.source_id)
                target_node = graph.nodes.get(conn.target_id)
                if source_node and target_node and source_node.level == level and target_node.level == level:
                    level_G.add_edge(
                        conn.source_id,
                        conn.target_id,
                        weight=conn.weight,
                        connection_type=conn.connection_type
                    )

            graph.level_graphs[level] = level_G

        # Navigation graph
        nav_G = nx.Graph()
        for nav_id, nav_node in graph.navigation_nodes.items():
            nav_G.add_node(
                nav_id,
                position=nav_node.position,
                space_id=nav_node.space_id,
                node_type=nav_node.node_type
            )

        for nav_id, nav_node in graph.navigation_nodes.items():
            for connected_id in nav_node.connections:
                if nav_G.has_node(connected_id):
                    other = graph.navigation_nodes[connected_id]
                    dist = self._distance_between(nav_node.position, other.position)
                    nav_G.add_edge(nav_id, connected_id, weight=dist)

        graph.nav_network = nav_G

    def _find_adjacent_spaces_to_door(self, model: BIMModel, door: BIMElement) -> List[BIMSpace]:
        """Find spaces adjacent to a door element."""
        adjacent = []
        door_center = door.center or (0, 0, 0)

        # Find spaces on the same level that are close to the door
        for space in model.spaces.values():
            if space.level != door.level:
                continue
            space_center = space.center or (0, 0, 0)
            dist = self._distance_between(door_center, space_center)

            # If door is within reasonable distance of space center, consider adjacent
            # Threshold kept tight so it only picks up actual adjacent spaces
            if dist < 5.0:
                adjacent.append(space)

        return adjacent

    def _detect_proximity_connectivity(self, model: BIMModel, graph: SpatialGraph):
        """Detect connections based on spatial proximity."""
        spaces = list(model.spaces.values())
        threshold = 30.0  # Increased maximum distance for proximity connection

        # Build set of existing connections for O(1) lookup
        existing = set()
        for c in graph.connections.values():
            existing.add((c.source_id, c.target_id))
            existing.add((c.target_id, c.source_id))

        for i in range(len(spaces)):
            for j in range(i + 1, len(spaces)):
                space_a = spaces[i]
                space_b = spaces[j]

                # Only connect spaces on same level
                if space_a.level != space_b.level:
                    continue

                # Skip if already connected (O(1) set check)
                key = (f"space_{space_a.id}", f"space_{space_b.id}")
                if key in existing:
                    continue

                center_a = space_a.center or (0, 0, 0)
                center_b = space_b.center or (0, 0, 0)
                dist = ((center_a[0] - center_b[0]) ** 2 + (center_a[1] - center_b[1]) ** 2) ** 0.5

                # Use a huge threshold if one is a corridor to catch far centroids
                threshold = 30.0 if (space_a.category == "corridor" or space_b.category == "corridor") else 5.0
                
                if dist < threshold:
                    connection = Connection(
                        id=f"conn_prox_{space_a.id}_{space_b.id}",
                        source_id=f"space_{space_a.id}",
                        target_id=f"space_{space_b.id}",
                        connection_type="open",
                        weight=dist,
                        attributes={"connection_method": "proximity"}
                    )
                    graph.connections[connection.id] = connection
                    existing.add(key)
                    existing.add((key[1], key[0]))

    def _detect_vertical_connectivity(self, model: BIMModel, graph: SpatialGraph):
        """Detect vertical connections through stairs and elevators."""
        # Find stair elements
        stairs = [e for e in model.elements.values() if e.category == ElementCategory.STAIR]
        elevators = [e for e in model.elements.values() if "elevator" in e.name.lower() or "lift" in e.name.lower()]

        vertical_elements = stairs + elevators

        for vert_elem in vertical_elements:
            elem_type = "stair" if vert_elem.category == ElementCategory.STAIR else "elevator"

            # Find spaces at different levels that are near this vertical element
            vert_center = vert_elem.center or (0, 0, 0)

            # First pass: collect nearby spaces grouped by level (O(S) instead of O(S²))
            nearby_by_level = defaultdict(list)
            for space in model.spaces.values():
                center = space.center or (0, 0, 0)
                dist = ((center[0] - vert_center[0]) ** 2 + (center[1] - vert_center[1]) ** 2) ** 0.5
                # Huge margin of error for stair connectivity (50m) to catch long corridors
                if dist < 50.0:
                    nearby_by_level[space.level].append(space)

            # Second pass: connect spaces across different levels
            levels = list(nearby_by_level.keys())
            for li in range(len(levels)):
                for lj in range(li + 1, len(levels)):
                    for space_a in nearby_by_level[levels[li]]:
                        center_a = space_a.center or (0, 0, 0)
                        for space_b in nearby_by_level[levels[lj]]:
                            center_b = space_b.center or (0, 0, 0)
                            connection = Connection(
                                id=f"conn_vert_{vert_elem.id}_{space_a.id}_{space_b.id}",
                                source_id=f"space_{space_a.id}",
                                target_id=f"space_{space_b.id}",
                                connection_type=elem_type,
                                weight=abs(center_a[2] - center_b[2]),
                                attributes={
                                    "vertical_element_id": vert_elem.id,
                                    "levels": [space_a.level, space_b.level]
                                }
                            )
                            graph.connections[connection.id] = connection

    def _detect_corridor_connectivity(self, model: BIMModel, graph: SpatialGraph):
        """Connect rooms to adjacent corridors."""
        corridors = [s for s in model.spaces.values() if s.category == "corridor"]
        rooms = [s for s in model.spaces.values() if s.category not in ["corridor", "staircase", "elevator", "void"]]

        # Build set of existing connections for O(1) lookup
        existing = set()
        for c in graph.connections.values():
            existing.add((c.source_id, c.target_id))
            existing.add((c.target_id, c.source_id))

        for room in rooms:
            room_center = room.center or (0, 0, 0)

            for corridor in corridors:
                if corridor.level != room.level:
                    continue

                corridor_center = corridor.center or (0, 0, 0)
                dist = ((room_center[0] - corridor_center[0]) ** 2 + (room_center[1] - corridor_center[1]) ** 2) ** 0.5
                
                # Margin of error for room-to-corridor connections (40m)
                if dist < 40.0:
                    key = (f"space_{room.id}", f"space_{corridor.id}")
                    if key not in existing:
                        connection = Connection(
                            id=f"conn_corr_{room.id}_{corridor.id}",
                            source_id=f"space_{room.id}",
                            target_id=f"space_{corridor.id}",
                            connection_type="corridor",
                            weight=dist,
                            attributes={"connection_method": "corridor_access"}
                        )
                        graph.connections[connection.id] = connection
                        existing.add(key)
                        existing.add((key[1], key[0]))

    def _connect_navigation_nodes(self, graph: SpatialGraph):
        """Create connections between navigation nodes."""
        nav_nodes = list(graph.navigation_nodes.values())

        for i, node_a in enumerate(nav_nodes):
            for node_b in nav_nodes[i + 1:]:
                # Connect nodes in the same space
                if node_a.space_id and node_a.space_id == node_b.space_id:
                    dist = self._distance_between(node_a.position, node_b.position)
                    if dist > 0:
                        node_a.connections.append(node_b.id)
                        node_b.connections.append(node_a.id)

                # Connect nearby waypoints
                elif node_a.node_type == "waypoint" and node_b.node_type == "waypoint":
                    dist = self._distance_between(node_a.position, node_b.position)
                    if dist < 15.0:
                        node_a.connections.append(node_b.id)
                        node_b.connections.append(node_a.id)

    def _analyze_spatial_properties(self, graph: SpatialGraph):
        """Calculate spatial analysis metrics."""
        if not graph.network:
            return

        # Calculate centrality measures
        try:
            betweenness = nx.betweenness_centrality(graph.network, weight="weight")
            for node_id, value in betweenness.items():
                if node_id in graph.nodes:
                    graph.nodes[node_id].attributes["betweenness_centrality"] = value
        except Exception as e:
            logger.warning(f"Could not calculate betweenness centrality: {e}")

        # Calculate degree centrality
        try:
            degree = nx.degree_centrality(graph.network)
            for node_id, value in degree.items():
                if node_id in graph.nodes:
                    graph.nodes[node_id].attributes["degree_centrality"] = value
        except Exception as e:
            logger.warning(f"Could not calculate degree centrality: {e}")

        # Find critical paths and bottlenecks
        try:
            # Find nodes with high betweenness (potential bottlenecks)
            avg_betweenness = sum(betweenness.values()) / len(betweenness) if betweenness else 0
            for node_id, value in betweenness.items():
                if value > avg_betweenness * 2:
                    if node_id in graph.nodes:
                        graph.nodes[node_id].attributes["is_bottleneck"] = True
        except Exception:
            pass

        # Calculate clustering coefficient
        try:
            clustering = nx.clustering(graph.network)
            for node_id, value in clustering.items():
                if node_id in graph.nodes:
                    graph.nodes[node_id].attributes["clustering_coefficient"] = value
        except Exception:
            pass

    def _compute_evacuation_paths(self, graph: SpatialGraph, annotations=None):
        """Compute shortest path from every navigable space to the nearest exit."""
        graph.evacuation_paths.clear()

        if not graph.network or not graph.exit_nodes:
            logger.warning("No network or exit nodes available for evacuation path computation")
            return

        # Build a set of valid exit node IDs in the network
        valid_exit_nodes = [n for n in graph.exit_nodes if n in graph.network]
        if not valid_exit_nodes:
            logger.warning("No valid exit nodes found in the network")
            return

        for node_id in graph.nodes:
            if node_id not in graph.network:
                continue
            # Skip non-navigable or blocked
            if graph.nodes[node_id].attributes.get("navigable", True) is False:
                continue
            if graph.network.nodes[node_id].get("blocked", False):
                continue

            # Find shortest path to nearest exit
            best_path = None
            best_length = float('inf')
            for exit_node in valid_exit_nodes:
                if exit_node == node_id:
                    best_path = [node_id]
                    best_length = 0
                    break
                try:
                    path = nx.shortest_path(graph.network, source=node_id, target=exit_node, weight="weight")
                    length = nx.shortest_path_length(graph.network, source=node_id, target=exit_node, weight="weight")
                    if length < best_length:
                        best_length = length
                        best_path = path
                except (nx.NetworkXNoPath, nx.NodeNotFound):
                    continue

            if best_path:
                graph.evacuation_paths[node_id] = best_path

        logger.info(f"  Computed {len(graph.evacuation_paths)} evacuation paths")

    def get_evacuation_path(self, space_id: str) -> Optional[List[str]]:
        """Return the precomputed evacuation path for a space (space IDs, not node IDs)."""
        if not self.spatial_graph:
            return None
        node_id = f"space_{space_id}"
        path = self.spatial_graph.evacuation_paths.get(node_id)
        if path:
            # Strip "space_" prefix from each node ID
            return [n.replace("space_", "") for n in path]
        return None

    def find_shortest_path(self, source_space_id: str, target_space_id: str) -> Optional[List[str]]:
        """Find shortest path between two spaces."""
        if not self.spatial_graph or not self.spatial_graph.network:
            return None

        source = f"space_{source_space_id}"
        target = f"space_{target_space_id}"

        if source not in self.spatial_graph.network or target not in self.spatial_graph.network:
            return None

        try:
            path = nx.shortest_path(
                self.spatial_graph.network,
                source=source,
                target=target,
                weight="weight"
            )
            return path
        except nx.NetworkXNoPath:
            return None
        except Exception as e:
            logger.warning(f"Pathfinding error: {e}")
            return None

    def find_accessible_path(self, source_space_id: str, target_space_id: str) -> Optional[List[str]]:
        """Find an accessible path between two spaces."""
        if not self.spatial_graph or not self.spatial_graph.network:
            return None

        source = f"space_{source_space_id}"
        target = f"space_{target_space_id}"

        # Create subgraph with only accessible edges
        accessible_edges = [
            (u, v) for u, v, d in self.spatial_graph.network.edges(data=True)
            if d.get("accessible", True)
        ]
        accessible_graph = self.spatial_graph.network.edge_subgraph(accessible_edges).copy()

        if source not in accessible_graph or target not in accessible_graph:
            return None

        try:
            path = nx.shortest_path(accessible_graph, source=source, target=target, weight="weight")
            return path
        except nx.NetworkXNoPath:
            return None

    def get_space_centrality(self, space_id: str) -> Dict[str, float]:
        """Get centrality measures for a space."""
        node_id = f"space_{space_id}"
        if not self.spatial_graph or node_id not in self.spatial_graph.nodes:
            return {}

        node = self.spatial_graph.nodes[node_id]
        return {
            "betweenness": node.attributes.get("betweenness_centrality", 0),
            "degree": node.attributes.get("degree_centrality", 0),
            "clustering": node.attributes.get("clustering_coefficient", 0),
            "is_bottleneck": node.attributes.get("is_bottleneck", False)
        }

    def get_connected_spaces(self, space_id: str) -> List[str]:
        """Get all spaces directly connected to a given space."""
        node_id = f"space_{space_id}"
        if not self.spatial_graph or not self.spatial_graph.network:
            return []

        if node_id not in self.spatial_graph.network:
            return []

        neighbors = list(self.spatial_graph.network.neighbors(node_id))
        return [n.replace("space_", "") for n in neighbors]

    def get_level_connectivity(self, level_id: str) -> Dict:
        """Get connectivity statistics for a level."""
        if not self.spatial_graph or level_id not in self.spatial_graph.level_graphs:
            return {}

        level_graph = self.spatial_graph.level_graphs[level_id]

        return {
            "node_count": level_graph.number_of_nodes(),
            "edge_count": level_graph.number_of_edges(),
            "is_connected": nx.is_connected(level_graph) if level_graph.number_of_nodes() > 0 else False,
            "density": nx.density(level_graph),
            "avg_path_length": nx.average_shortest_path_length(level_graph) if nx.is_connected(level_graph) and level_graph.number_of_nodes() > 0 else None,
            "diameter": nx.diameter(level_graph) if nx.is_connected(level_graph) and level_graph.number_of_nodes() > 0 else None
        }

    @staticmethod
    def _distance_between(a: Tuple[float, ...], b: Tuple[float, ...]) -> float:
        """Calculate Euclidean distance between two points."""
        return sum((a[i] - b[i]) ** 2 for i in range(min(len(a), len(b)))) ** 0.5

    def get_navigation_waypoints(self, space_id: str) -> List[Tuple[float, float, float]]:
        """Get navigation waypoints for a space."""
        if not self.spatial_graph:
            return []

        waypoints = []
        for nav_id, nav_node in self.spatial_graph.navigation_nodes.items():
            if nav_node.space_id == space_id:
                waypoints.append(nav_node.position)

        return waypoints

    def get_critical_bottlenecks(self, top_n: int = 10) -> List[Dict]:
        """Get the most critical bottlenecks in the building."""
        if not self.spatial_graph:
            return []

        bottlenecks = []
        for node_id, node in self.spatial_graph.nodes.items():
            if node.attributes.get("is_bottleneck", False):
                bottlenecks.append({
                    "space_id": node.space_id,
                    "name": node.name,
                    "betweenness": node.attributes.get("betweenness_centrality", 0),
                    "category": node.category
                })

        bottlenecks.sort(key=lambda x: x["betweenness"], reverse=True)
        return bottlenecks[:top_n]

    def analyze_accessibility(self) -> Dict[str, Any]:
        """Analyze accessibility of the building."""
        if not self.spatial_graph:
            return {}

        results = {
            "total_spaces": len(self.spatial_graph.nodes),
            "total_connections": len(self.spatial_graph.connections),
            "connection_types": {},
            "avg_connections_per_space": 0,
            "isolated_spaces": [],
            "accessible_issues": []
        }

        # Count connection types
        for conn in self.spatial_graph.connections.values():
            conn_type = conn.connection_type
            results["connection_types"][conn_type] = results["connection_types"].get(conn_type, 0) + 1

        # Calculate average connections
        degrees = dict(self.spatial_graph.network.degree()) if self.spatial_graph.network else {}
        if degrees:
            results["avg_connections_per_space"] = sum(degrees.values()) / len(degrees)

        # Find isolated spaces
        for node_id, node in self.spatial_graph.nodes.items():
            degree = degrees.get(node_id, 0)
            if degree == 0:
                results["isolated_spaces"].append({
                    "space_id": node.space_id,
                    "name": node.name,
                    "category": node.category
                })

        # Check for potential accessibility issues
        for conn in self.spatial_graph.connections.values():
            if conn.connection_type == "stair" and conn.width < 1.2:
                results["accessible_issues"].append({
                    "type": "narrow_stair",
                    "connection_id": conn.id,
                    "width": conn.width
                })

        return results
