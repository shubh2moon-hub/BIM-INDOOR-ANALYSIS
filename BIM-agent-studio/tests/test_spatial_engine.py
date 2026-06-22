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


if __name__ == "__main__":
    unittest.main()
