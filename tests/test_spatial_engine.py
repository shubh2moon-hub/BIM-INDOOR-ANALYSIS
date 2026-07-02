"""
Unit tests for Spatial Intelligence Engine
"""

import unittest
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.spatial_engine import SpatialIntelligenceEngine, SpatialGraph
from core.bim_processor import BIMModel


class TestSpatialIntelligenceEngine(unittest.TestCase):
    """Test cases for SpatialIntelligenceEngine."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.engine = SpatialIntelligenceEngine()
    
    def test_engine_initialization(self):
        """Test engine initialization."""
        self.assertIsNotNone(self.engine)
        self.assertIsNone(self.engine.spatial_graph)
    
    def test_distance_calculation(self):
        """Test distance between two points."""
        a = (0, 0, 0)
        b = (3, 4, 0)
        dist = self.engine._distance_between(a, b)
        self.assertEqual(dist, 5.0)
    
    def test_distance_3d(self):
        """Test 3D distance calculation."""
        a = (1, 2, 3)
        b = (4, 6, 8)
        dist = self.engine._distance_between(a, b)
        expected = ((4-1)**2 + (6-2)**2 + (8-3)**2) ** 0.5
        self.assertAlmostEqual(dist, expected)

    def test_detect_door_connectivity(self):
        """Test that door connectivity is correctly detected between spaces near a door."""
        from core.bim_processor import BIMModel, BIMSpace, BIMElement, ElementCategory
        from core.spatial_engine import SpatialGraph, SpaceNode
        
        # Create a mock model
        model = BIMModel(
            id="test_model",
            name="Test Model",
            file_path="",
            schema="IFC4"
        )
        
        # Create a door element
        door = BIMElement(
            id="door1",
            global_id="gid_door1",
            name="Test Door",
            element_type="IfcDoor",
            category=ElementCategory.DOOR,
            level="L1",
            center=(0.0, 0.0, 0.0),
            properties={"Width": 1.0}
        )
        model.elements["door1"] = door
        
        # Create two adjacent spaces
        space_a = BIMSpace(
            id="space1",
            global_id="gid_space1",
            name="Room A",
            long_name="Room A",
            level="L1",
            area=20.0,
            volume=60.0,
            center=(-2.0, 0.0, 0.0)
        )
        space_b = BIMSpace(
            id="space2",
            global_id="gid_space2",
            name="Room B",
            long_name="Room B",
            level="L1",
            area=20.0,
            volume=60.0,
            center=(2.0, 0.0, 0.0)
        )
        model.spaces["space1"] = space_a
        model.spaces["space2"] = space_b
        
        # Set up a spatial graph with the space nodes
        graph = SpatialGraph(id="graph1", name="Test Graph")
        self.engine._create_space_nodes(model, graph)
        
        # Detect door connectivity
        self.engine._detect_door_connectivity(model, graph)
        
        # We expect 1 connection between space1 and space2
        self.assertEqual(len(graph.connections), 1)
        connection = list(graph.connections.values())[0]
        self.assertEqual(connection.connection_type, "door")
        self.assertEqual(connection.source_id, "space_space1")
        self.assertEqual(connection.target_id, "space_space2")
        self.assertEqual(connection.weight, 4.0)  # Distance between (-2,0,0) and (2,0,0)


if __name__ == "__main__":
    unittest.main()
