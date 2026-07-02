"""
Model Annotation Layer

Allows users to manually annotate BIM models to fix incomplete or incorrect
IFC data — marking space types, exits, fire origins, and adding virtual
doors/connections that don't exist in the original IFC file.
"""

import uuid
import json
import logging
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field, asdict
from pathlib import Path

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Valid user-overridable space categories
# ---------------------------------------------------------------------------

VALID_SPACE_CATEGORIES = [
    "room",
    "corridor",
    "staircase",
    "elevator",
    "restroom",
    "office",
    "meeting",
    "public",
    "entrance",
    "exit",
    "lobby",
    "void",
    "space",
]


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class VirtualExit:
    """A manually-placed exit point (e.g. when the IFC has no door there)."""
    id: str
    name: str
    position: Tuple[float, float, float]
    level_id: str = ""
    width: float = 1.2
    accessible: bool = True


@dataclass
class VirtualConnection:
    """A manually-created connection between two spaces (e.g. missing door)."""
    id: str
    source_space_id: str
    target_space_id: str
    connection_type: str = "virtual_door"
    width: float = 1.0


@dataclass
class VirtualSpace:
    """A manually-created virtual space for models that lack IfcSpace entities."""
    id: str
    name: str
    position: Tuple[float, float, float]
    level_id: str = ""
    radius: float = 3.0
    category: str = "space"
    boundary: Optional[List[Tuple[float, float]]] = None


@dataclass
class SpaceAnnotation:
    """User annotations for a single space."""
    space_id: str
    category_override: Optional[str] = None          # e.g. "office", "corridor"
    is_exit: bool = False
    is_fire_origin: bool = False
    agent_count: int = 0
    custom_name: Optional[str] = None
    block_path: bool = False

    def __post_init__(self):
        if self.category_override and self.category_override not in VALID_SPACE_CATEGORIES:
            logger.warning(
                f"Invalid category_override '{self.category_override}' for space "
                f"{self.space_id}. Valid: {VALID_SPACE_CATEGORIES}"
            )
            self.category_override = None


@dataclass
class ModelAnnotations:
    """Complete set of user annotations for a BIM model."""
    exits: List[VirtualExit] = field(default_factory=list)
    space_annotations: Dict[str, SpaceAnnotation] = field(default_factory=dict)
    virtual_connections: List[VirtualConnection] = field(default_factory=list)
    virtual_spaces: List[VirtualSpace] = field(default_factory=list)

    # ------------------------------------------------------------------
    # Helpers for the UI
    # ------------------------------------------------------------------

    def get_or_create_space(self, space_id: str) -> SpaceAnnotation:
        """Return existing annotation for a space, or create a blank one."""
        if space_id not in self.space_annotations:
            self.space_annotations[space_id] = SpaceAnnotation(space_id=space_id)
        return self.space_annotations[space_id]

    def set_category_override(self, space_id: str, category: Optional[str]):
        """Set (or clear) the category override for a space."""
        ann = self.get_or_create_space(space_id)
        ann.category_override = category

    def set_is_exit(self, space_id: str, value: bool):
        ann = self.get_or_create_space(space_id)
        ann.is_exit = value

    def set_is_fire_origin(self, space_id: str, value: bool):
        ann = self.get_or_create_space(space_id)
        ann.is_fire_origin = value

    def set_agent_count(self, space_id: str, count: int):
        ann = self.get_or_create_space(space_id)
        ann.agent_count = count

    def set_custom_name(self, space_id: str, name: Optional[str]):
        ann = self.get_or_create_space(space_id)
        ann.custom_name = name

    def set_block_path(self, space_id: str, value: bool):
        ann = self.get_or_create_space(space_id)
        ann.block_path = value

    def add_virtual_exit(self, name: str, position: Tuple[float, float, float],
                         level_id: str = "", width: float = 1.2) -> VirtualExit:
        """Add a new virtual exit and return it."""
        exit_obj = VirtualExit(
            id=str(uuid.uuid4()),
            name=name,
            position=position,
            level_id=level_id,
            width=width,
        )
        self.exits.append(exit_obj)
        return exit_obj

    def remove_virtual_exit(self, exit_id: str) -> bool:
        """Remove a virtual exit by ID. Returns True if found and removed."""
        for i, e in enumerate(self.exits):
            if e.id == exit_id:
                self.exits.pop(i)
                return True
        return False

    def add_virtual_connection(self, source_id: str, target_id: str,
                                 width: float = 1.0) -> VirtualConnection:
        """Add a virtual connection between two spaces."""
        conn = VirtualConnection(
            id=str(uuid.uuid4()),
            source_space_id=source_id,
            target_space_id=target_id,
            width=width,
        )
        self.virtual_connections.append(conn)
        return conn

    def remove_virtual_connection(self, conn_id: str) -> bool:
        for i, c in enumerate(self.virtual_connections):
            if c.id == conn_id:
                self.virtual_connections.pop(i)
                return True
        return False

    def add_virtual_space(self, name: str, position: Tuple[float, float, float],
                          level_id: str = "", radius: float = 3.0, 
                          category: str = "space", boundary: Optional[List[Tuple[float, float]]] = None) -> VirtualSpace:
        """Add a new virtual space."""
        space = VirtualSpace(
            id=f"v_space_{uuid.uuid4()}",
            name=name,
            position=position,
            level_id=level_id,
            radius=radius,
            category=category,
            boundary=boundary,
        )
        self.virtual_spaces.append(space)
        return space

    def remove_virtual_space(self, space_id: str) -> bool:
        for i, s in enumerate(self.virtual_spaces):
            if s.id == space_id:
                self.virtual_spaces.pop(i)
                return True
        return False

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def to_dict(self) -> dict:
        """Serialize annotations to a plain dict."""
        return {
            "exits": [
                {
                    "id": e.id,
                    "name": e.name,
                    "position": list(e.position),
                    "level_id": e.level_id,
                    "width": e.width,
                    "accessible": e.accessible,
                }
                for e in self.exits
            ],
            "space_annotations": {
                sid: {
                    "space_id": a.space_id,
                    "category_override": a.category_override,
                    "is_exit": a.is_exit,
                    "is_fire_origin": a.is_fire_origin,
                    "agent_count": a.agent_count,
                    "custom_name": a.custom_name,
                    "block_path": a.block_path,
                }
                for sid, a in self.space_annotations.items()
            },
            "virtual_connections": [
                {
                    "id": c.id,
                    "source_space_id": c.source_space_id,
                    "target_space_id": c.target_space_id,
                    "connection_type": c.connection_type,
                    "width": c.width,
                }
                for c in self.virtual_connections
            ],
            "virtual_spaces": [
                {
                    "id": s.id,
                    "name": s.name,
                    "position": list(s.position),
                    "level_id": s.level_id,
                    "radius": s.radius,
                    "category": s.category,
                    "boundary": s.boundary,
                }
                for s in self.virtual_spaces
            ],
        }

    @staticmethod
    def from_dict(data: dict) -> "ModelAnnotations":
        """Deserialize annotations from a plain dict."""
        ann = ModelAnnotations()

        for e in data.get("exits", []):
            ann.exits.append(VirtualExit(
                id=e["id"],
                name=e["name"],
                position=tuple(e["position"]),
                level_id=e.get("level_id", ""),
                width=e.get("width", 1.2),
                accessible=e.get("accessible", True),
            ))

        for sid, a in data.get("space_annotations", {}).items():
            ann.space_annotations[sid] = SpaceAnnotation(
                space_id=a["space_id"],
                category_override=a.get("category_override"),
                is_exit=a.get("is_exit", False),
                is_fire_origin=a.get("is_fire_origin", False),
                agent_count=a.get("agent_count", 0),
                custom_name=a.get("custom_name"),
                block_path=a.get("block_path", False),
            )

        for c in data.get("virtual_connections", []):
            ann.virtual_connections.append(VirtualConnection(
                id=c["id"],
                source_space_id=c["source_space_id"],
                target_space_id=c["target_space_id"],
                connection_type=c.get("connection_type", "virtual_door"),
                width=c.get("width", 1.0),
            ))

        for s in data.get("virtual_spaces", []):
            ann.virtual_spaces.append(VirtualSpace(
                id=s["id"],
                name=s["name"],
                position=tuple(s["position"]),
                level_id=s.get("level_id", ""),
                radius=s.get("radius", 3.0),
                category=s.get("category", "space"),
                boundary=s.get("boundary", None),
            ))

        return ann

    def save(self, file_path: str):
        """Save annotations to a JSON file."""
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)
        logger.info(f"Annotations saved to {file_path}")

    @staticmethod
    def load(file_path: str) -> "ModelAnnotations":
        """Load annotations from a JSON file."""
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        ann = ModelAnnotations.from_dict(data)
        logger.info(f"Annotations loaded from {file_path}")
        return ann
