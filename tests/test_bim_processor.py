"""
Unit tests for BIM Processing Module
"""

import unittest
import os
import tempfile
from pathlib import Path

import numpy as np

# Add parent to path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.bim_processor import BIMProcessor, BIMModel, ElementCategory


class TestBIMProcessor(unittest.TestCase):
    """Test cases for BIMProcessor."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.processor = BIMProcessor()
    
    def test_processor_initialization(self):
        """Test processor initialization."""
        self.assertIsNotNone(self.processor)
        self.assertIsNone(self.processor.current_model)
    
    def test_load_nonexistent_file(self):
        """Test loading a non-existent file."""
        with self.assertRaises(FileNotFoundError):
            self.processor.load_ifc("nonexistent.ifc")
    
    def test_element_category_enum(self):
        """Test element category enum values."""
        self.assertEqual(ElementCategory.WALL.value, "Wall")
        self.assertEqual(ElementCategory.DOOR.value, "Door")
        self.assertEqual(ElementCategory.WINDOW.value, "Window")
        self.assertEqual(ElementCategory.FLOOR.value, "Floor")


class TestBIMModel(unittest.TestCase):
    """Test cases for BIMModel dataclass."""
    
    def test_model_creation(self):
        """Test creating a BIM model."""
        model = BIMModel(
            id="test-123",
            name="Test Building",
            file_path="/test/path.ifc",
            schema="IFC4"
        )
        
        self.assertEqual(model.id, "test-123")
        self.assertEqual(model.name, "Test Building")
        self.assertEqual(model.file_path, "/test/path.ifc")
        self.assertEqual(model.schema, "IFC4")
        self.assertEqual(len(model.elements), 0)
        self.assertEqual(len(model.spaces), 0)
        self.assertEqual(len(model.levels), 0)


if __name__ == "__main__":
    unittest.main()
