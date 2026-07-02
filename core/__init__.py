# BIM-CrowdSim Core Package
from .model_annotations import (
    ModelAnnotations,
    SpaceAnnotation,
    VirtualExit,
    VirtualConnection,
    VirtualSpace,
    VALID_SPACE_CATEGORIES,
)
from .bim_processor import BIMProcessor, BIMModel, BIMElement, BIMSpace, BIMLevel, ElementCategory
from .spatial_engine import SpatialIntelligenceEngine, SpatialGraph, SpaceNode, Connection, NavigationNode
