"""
BIM Processing Module
Handles IFC file import, validation, and element extraction.
"""

import os
import uuid
import logging
import multiprocessing
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum

import numpy as np
import ifcopenshell
import ifcopenshell.geom
import ifcopenshell.util
import ifcopenshell.util.element
import ifcopenshell.util.shape
import ifcopenshell.util.placement
from ifcopenshell import entity_instance

from core.model_annotations import ModelAnnotations

logger = logging.getLogger(__name__)


class ElementCategory(Enum):
    WALL = "Wall"
    DOOR = "Door"
    WINDOW = "Window"
    FLOOR = "Floor"
    ROOF = "Roof"
    STAIR = "Stair"
    RAMP = "Ramp"
    ELEVATOR = "Elevator"
    ROOM = "Room"
    SPACE = "Space"
    CORRIDOR = "Corridor"
    FURNITURE = "Furniture"
    COLUMN = "Column"
    BEAM = "Beam"
    SLAB = "Slab"
    ROOFING = "Roofing"
    PROXYY = "Proxy"
    UNKNOWN = "Unknown"


@dataclass
class BIMElement:
    """Represents a BIM element with extracted properties."""
    id: str
    global_id: str
    name: str
    element_type: str
    category: ElementCategory
    level: str
    properties: Dict[str, Any] = field(default_factory=dict)
    geometry: Optional[Dict] = None
    bounds: Optional[Tuple] = None
    center: Optional[Tuple[float, float, float]] = None
    area: float = 0.0
    volume: float = 0.0


@dataclass
class BIMLevel:
    """Represents a building level/story."""
    id: str
    name: str
    elevation: float
    height: float
    elements: List[str] = field(default_factory=list)


@dataclass
class BIMSpace:
    """Represents a space/room in the building."""
    id: str
    global_id: str
    name: str
    long_name: str
    level: str
    area: float
    volume: float
    bounds: Optional[Tuple] = None
    center: Optional[Tuple[float, float, float]] = None
    geometry: Optional[Dict] = None
    category: str = "space"
    adjacent_spaces: List[str] = field(default_factory=list)
    connected_doors: List[str] = field(default_factory=list)
    occupancy_capacity: Optional[int] = None


@dataclass
class BIMModel:
    """Complete BIM model representation."""
    id: str
    name: str
    file_path: str
    schema: str
    elements: Dict[str, BIMElement] = field(default_factory=dict)
    spaces: Dict[str, BIMSpace] = field(default_factory=dict)
    levels: Dict[str, BIMLevel] = field(default_factory=dict)
    project_info: Dict[str, str] = field(default_factory=dict)
    validation_report: Dict = field(default_factory=dict)
    spatial_graph: Optional[Any] = None
    annotations: Any = field(default=None)  # ModelAnnotations, imported lazily to avoid circular refs


class BIMProcessor:
    """Main class for processing IFC BIM models."""

    def __init__(self):
        self.current_model: Optional[BIMModel] = None
        self.settings = ifcopenshell.geom.settings()
        self.settings.set(self.settings.USE_WORLD_COORDS, True)
        self._has_occ = False
        try:
            self.settings.set(self.settings.USE_PYTHON_OPENCASCADE, True)
            self._has_occ = True
        except AttributeError:
            logger.warning("Python OpenCASCADE not installed. Falling back to default geometry engine.")
            try:
                self.settings.set(self.settings.DISABLE_OPENING_SUBTRACTIONS, True)
                self.settings.set(self.settings.DISABLE_BOOLEAN_RESULT, True)
            except AttributeError:
                pass

    def load_ifc(self, file_path: str, progress_callback=None) -> BIMModel:
        """Load and process an IFC file."""
        logger.info(f"Loading IFC file: {file_path}")

        if progress_callback:
            progress_callback("Loading IFC file...", 5)

        if not os.path.exists(file_path):
            raise FileNotFoundError(f"IFC file not found: {file_path}")

        try:
            ifc_file = ifcopenshell.open(file_path)
        except Exception as e:
            raise RuntimeError(f"Failed to open IFC file: {e}")

        # Create model container
        project = ifc_file.by_type("IfcProject")[0] if ifc_file.by_type("IfcProject") else None
        model = BIMModel(
            id=str(uuid.uuid4()),
            name=project.Name if project else "Unnamed Project",
            file_path=file_path,
            schema=ifc_file.schema
        )

        # Initialize empty annotations so the UI can attach overrides immediately
        try:
            model.annotations = ModelAnnotations()
        except Exception:
            model.annotations = None

        # Extract project info
        logger.info("Extracting project metadata...")
        model.project_info = self._extract_project_info(ifc_file, project)
        
        # Pre-compute geometry for all elements
        logger.info("Pre-computing geometry skipped. Using element-by-element fallback to avoid Windows hangs.")
        self._geometry_cache = {}
        # Geometry will be extracted dynamically via self._get_element_geometry_and_bounds()

        # Extract levels
        logger.info("Extracting building levels...")
        if progress_callback: progress_callback("Extracting building levels...", 15)
        model.levels = self._extract_levels(ifc_file)

        # Extract spaces
        logger.info("Extracting spaces...")
        if progress_callback: progress_callback("Extracting spaces...", 20)
        model.spaces = self._extract_spaces(ifc_file, model.levels, progress_callback)

        # Extract building elements
        logger.info("Extracting building elements...")
        if progress_callback: progress_callback("Extracting building elements...", 40)
        model.elements = self._extract_elements(ifc_file, model.levels, progress_callback)

        # NEW: Automatically convert doors labeled "exit" into virtual exit spaces
        for element in list(model.elements.values()):
            if element.category == ElementCategory.DOOR:
                name_lower = element.name.lower()
                props_lower = " ".join([str(v).lower() for v in element.properties.values()])
                if "exit" in name_lower or "exit" in props_lower:
                    v_id = f"virt_exit_{element.id}"
                    model.spaces[v_id] = BIMSpace(
                        id=v_id,
                        global_id=f"virt_{element.global_id}",
                        name=f"{element.name} (Exit Door)",
                        level=element.level,
                        center=element.center,
                        geometry=element.geometry,
                        category="exit",
                        occupancy_capacity=20
                    )

        # Validate model
        logger.info("Validating model...")
        model.validation_report = self._validate_model(ifc_file, model)

        self.current_model = model
        logger.info(f"Loaded model: {model.name} with {len(model.elements)} elements, {len(model.spaces)} spaces")

        return model

    def _extract_project_info(self, ifc_file, project) -> Dict[str, str]:
        """Extract project metadata."""
        info = {}
        if project:
            info["name"] = project.Name or ""
            info["description"] = getattr(project, "Description", "") or ""
            info["phase"] = getattr(project, "Phase", "") or ""

            # Get units
            units = ifc_file.by_type("IfcUnitAssignment")
            if units:
                info["units"] = str(units[0].Units)

            # Get site info
            sites = ifc_file.by_type("IfcSite")
            if sites:
                site = sites[0]
                info["site_name"] = site.Name or ""
                info["site_address"] = getattr(site, "PostalAddress", "") or ""

            # Get building info
            buildings = ifc_file.by_type("IfcBuilding")
            if buildings:
                building = buildings[0]
                info["building_name"] = building.Name or ""
                info["building_type"] = getattr(building, "BuildingType", "") or ""

        return info

    def _extract_levels(self, ifc_file) -> Dict[str, BIMLevel]:
        """Extract building levels/stories."""
        levels = {}

        for storey in ifc_file.by_type("IfcBuildingStorey"):
            elevation = getattr(storey, "Elevation", 0.0) or 0.0
            level = BIMLevel(
                id=str(storey.id()),
                name=storey.Name or f"Level {storey.id()}",
                elevation=float(elevation),
                height=0.0  # Will be calculated
            )
            levels[str(storey.id())] = level

        # Calculate level heights
        sorted_levels = sorted(levels.values(), key=lambda l: l.elevation)
        for i in range(len(sorted_levels) - 1):
            sorted_levels[i].height = sorted_levels[i + 1].elevation - sorted_levels[i].elevation

        return levels

    def _extract_spaces(self, ifc_file, levels: Dict[str, BIMLevel], progress_callback=None) -> Dict[str, BIMSpace]:
        """Extract spaces/rooms from IFC."""
        spaces = {}
        
        ifc_spaces = ifc_file.by_type("IfcSpace")
        total = len(ifc_spaces)

        for i, space in enumerate(ifc_spaces):
            if progress_callback and i % 5 == 0:
                progress_callback(f"Extracting spaces ({i}/{total})...", 20 + int((i/total)*20))
            # Get level
            level_id = ""
            for rel in getattr(space, "ContainedInStructure", []) or []:
                if rel and hasattr(rel, "RelatingStructure"):
                    level_id = str(rel.RelatingStructure.id())
                    break
            
            # Spaces often decompose the storey instead of being contained in it
            if not level_id:
                for rel in getattr(space, "Decomposes", []) or []:
                    if rel and hasattr(rel, "RelatingObject"):
                        level_id = str(rel.RelatingObject.id())
                        break

            # Get area and volume from quantities
            area = 0.0
            volume = 0.0
            occupancy_capacity = None
            if hasattr(space, "IsDefinedBy"):
                for rel in space.IsDefinedBy:
                    if hasattr(rel, "RelatingPropertyDefinition"):
                        pset = rel.RelatingPropertyDefinition
                        if hasattr(pset, "Quantities"):
                            for q in pset.Quantities:
                                if q.Name == "NetFloorArea" and hasattr(q, "AreaValue"):
                                    area = q.AreaValue
                                elif q.Name == "NetVolume" and hasattr(q, "VolumeValue"):
                                    volume = q.VolumeValue
                        if hasattr(pset, "HasProperties"):
                            for prop in pset.HasProperties:
                                if prop.Name in ["OccupancyNumber", "Occupancy", "Capacity"] and hasattr(prop, "NominalValue"):
                                    try:
                                        val = prop.NominalValue
                                        if hasattr(val, "wrappedValue"):
                                            occupancy_capacity = int(float(val.wrappedValue))
                                        else:
                                            occupancy_capacity = int(float(val))
                                    except (ValueError, TypeError, AttributeError):
                                        pass

            # Get bounds
            geometry, bounds, center = self._get_element_geometry_and_bounds(space)

            # Categorize space
            category = self._categorize_space(space)

            bim_space = BIMSpace(
                id=str(space.id()),
                global_id=space.GlobalId,
                name=space.Name or f"Space {space.id()}",
                long_name=getattr(space, "LongName", "") or "",
                level=level_id,
                area=float(area) if area else 0.0,
                volume=float(volume) if volume else 0.0,
                bounds=bounds,
                center=center,
                geometry=geometry,
                category=category,
                occupancy_capacity=occupancy_capacity
            )
            spaces[str(space.id())] = bim_space

            # Add to level
            if level_id in levels:
                levels[level_id].elements.append(str(space.id()))

        return spaces

    def _categorize_space(self, space) -> str:
        """Categorize a space by its type and name."""
        name = (space.Name or "").lower()
        long_name = (getattr(space, "LongName", "") or "").lower()
        full_name = f"{name} {long_name}"

        # Check space type
        space_type = ""
        if hasattr(space, "PredefinedType") and space.PredefinedType:
            space_type = str(space.PredefinedType).lower()

        corridor_keywords = ["corridor", "hallway", "hall", "passage", "lobby", "foyer", "entrance"]
        stair_keywords = ["stair", "staircase", "stairwell", "escape"]
        elevator_keywords = ["elevator", "lift", "escalator"]
        restroom_keywords = ["restroom", "bathroom", "toilet", "wc", "lavatory"]
        office_keywords = ["office", "work", "workspace", "desk"]
        meeting_keywords = ["meeting", "conference", "boardroom"]
        public_keywords = ["lobby", "atrium", "reception", "waiting", "lounge"]

        for kw in corridor_keywords:
            if kw in full_name:
                return "corridor"
        for kw in stair_keywords:
            if kw in full_name:
                return "staircase"
        for kw in elevator_keywords:
            if kw in full_name:
                return "elevator"
        for kw in restroom_keywords:
            if kw in full_name:
                return "restroom"
        for kw in office_keywords:
            if kw in full_name:
                return "office"
        for kw in meeting_keywords:
            if kw in full_name:
                return "meeting"
        for kw in public_keywords:
            if kw in full_name:
                return "public"

        if "internal" in space_type or "external" in space_type:
            return space_type

        return "space"

    def _extract_elements(self, ifc_file, levels: Dict[str, BIMLevel], progress_callback=None) -> Dict[str, BIMElement]:
        """Extract building elements from IFC."""
        elements = {}

        relevant_types = [
            "IfcWall", "IfcWallStandardCase",
            "IfcDoor", "IfcWindow",
            "IfcSlab", "IfcRoof", "IfcCovering",
            "IfcStair", "IfcStairFlight", "IfcRamp", "IfcRampFlight",
            "IfcColumn", "IfcBeam", "IfcMember",
            "IfcFurnishingElement", "IfcFlowTerminal",
            "IfcTransportElement", "IfcBuildingElementProxy"
        ]

        all_elems = []
        for element_type in relevant_types:
            all_elems.extend(ifc_file.by_type(element_type))
            
        total = len(all_elems)

        for i, elem in enumerate(all_elems):
            if progress_callback and i % 10 == 0:
                progress_callback(f"Extracting elements ({i}/{total})...", 40 + int((i/total)*40))
            try:
                bim_element = self._process_element(elem, levels)
                if bim_element:
                    elements[bim_element.id] = bim_element
            except Exception as e:
                logger.warning(f"Error processing element {elem.id()}: {e}")

        return elements

    def _process_element(self, elem, levels: Dict[str, BIMLevel]) -> Optional[BIMElement]:
        """Process a single IFC element."""
        category = self._classify_element(elem)

        # Get level
        level_id = ""
        for rel in getattr(elem, "ContainedInStructure", []) or []:
            if rel and hasattr(rel, "RelatingStructure"):
                level_id = str(rel.RelatingStructure.id())
                break

        # Get geometry
        geometry, bounds, center = self._get_element_geometry_and_bounds(elem)

        # Get properties
        properties = self._get_element_properties(elem)

        # Calculate area and volume
        area = 0.0
        volume = 0.0
        if hasattr(elem, "IsDefinedBy"):
            for rel in elem.IsDefinedBy:
                if hasattr(rel, "RelatingPropertyDefinition"):
                    pset = rel.RelatingPropertyDefinition
                    if hasattr(pset, "Quantities"):
                        for q in pset.Quantities:
                            if q.Name in ["Area", "GrossArea", "NetFloorArea"] and hasattr(q, "AreaValue"):
                                area = q.AreaValue
                            elif q.Name in ["Volume", "GrossVolume", "NetVolume"] and hasattr(q, "VolumeValue"):
                                volume = q.VolumeValue

        element = BIMElement(
            id=str(elem.id()),
            global_id=elem.GlobalId,
            name=elem.Name or f"{elem.is_a()} {elem.id()}",
            element_type=elem.is_a(),
            category=category,
            level=level_id,
            properties=properties,
            geometry=geometry,
            bounds=bounds,
            center=center,
            area=float(area) if area else 0.0,
            volume=float(volume) if volume else 0.0
        )

        # Add to level
        if level_id in levels:
            levels[level_id].elements.append(str(elem.id()))

        return element

    def _classify_element(self, elem) -> ElementCategory:
        """Classify an IFC element into a category."""
        elem_type = elem.is_a()

        if "Wall" in elem_type:
            return ElementCategory.WALL
        elif "Door" in elem_type:
            return ElementCategory.DOOR
        elif "Window" in elem_type:
            return ElementCategory.WINDOW
        elif "Slab" in elem_type or "Floor" in elem_type:
            return ElementCategory.FLOOR
        elif "Roof" in elem_type or "Covering" in elem_type:
            return ElementCategory.ROOF
        elif "Stair" in elem_type:
            return ElementCategory.STAIR
        elif "Ramp" in elem_type:
            return ElementCategory.RAMP
        elif "Column" in elem_type:
            return ElementCategory.COLUMN
        elif "Beam" in elem_type:
            return ElementCategory.BEAM
        elif "Furnishing" in elem_type:
            return ElementCategory.FURNITURE
        elif "Transport" in elem_type:
            return ElementCategory.ELEVATOR
        elif "Proxy" in elem_type:
            return ElementCategory.PROXYY
        else:
            return ElementCategory.UNKNOWN

    def _get_element_geometry_and_bounds(self, elem) -> Tuple[Optional[Dict], Optional[Tuple], Optional[Tuple]]:
        """Get full geometry, bounding box and center of an element."""
        if hasattr(self, "_geometry_cache") and getattr(elem, "GlobalId", None) in self._geometry_cache:
            return self._geometry_cache[elem.GlobalId]
            


        try:
            shape = ifcopenshell.geom.create_shape(self.settings, elem)
            if shape:
                verts = np.array(shape.geometry.verts).reshape(-1, 3)
                faces = np.array(shape.geometry.faces)
                if len(verts) > 0:
                    min_bounds = tuple(verts.min(axis=0))
                    max_bounds = tuple(verts.max(axis=0))
                    center = tuple((verts.min(axis=0) + verts.max(axis=0)) / 2)
                    pv_faces = []
                    for i in range(0, len(faces), 3):
                        pv_faces.extend([3, faces[i], faces[i+1], faces[i+2]])
                    geometry = {"vertices": verts.tolist(), "faces": pv_faces}
                    return geometry, (min_bounds, max_bounds), center
        except Exception as e:
            import traceback
            print("INTERNAL GEOMETRY EXCEPTION:", type(e), e)
            # traceback.print_exc()
        return None, None, None

    def _get_element_properties(self, elem) -> Dict[str, Any]:
        """Extract properties from an IFC element."""
        properties = {}

        # Get property sets
        if hasattr(elem, "IsDefinedBy"):
            for rel in elem.IsDefinedBy:
                if hasattr(rel, "RelatingPropertyDefinition"):
                    pset = rel.RelatingPropertyDefinition
                    if hasattr(pset, "HasProperties"):
                        for prop in pset.HasProperties:
                            if hasattr(prop, "NominalValue"):
                                properties[prop.Name] = prop.NominalValue.wrappedValue if hasattr(prop.NominalValue, 'wrappedValue') else str(prop.NominalValue)

        # Get type properties
        if hasattr(elem, "IsTypedBy"):
            for rel in elem.IsTypedBy:
                if hasattr(rel, "RelatingType"):
                    elem_type = rel.RelatingType
                    if hasattr(elem_type, "HasPropertySets"):
                        for pset in elem_type.HasPropertySets:
                            if hasattr(pset, "HasProperties"):
                                for prop in pset.HasProperties:
                                    if hasattr(prop, "NominalValue"):
                                        properties[prop.Name] = prop.NominalValue.wrappedValue if hasattr(prop.NominalValue, 'wrappedValue') else str(prop.NominalValue)

        # Get material
        if hasattr(elem, "HasAssociations"):
            for assoc in elem.HasAssociations:
                if assoc.is_a("IfcRelAssociatesMaterial"):
                    material = assoc.RelatingMaterial
                    if hasattr(material, "Name"):
                        properties["Material"] = material.Name
                    elif hasattr(material, "ForLayerSet"):
                        properties["Material"] = material.ForLayerSet.MaterialLayers[0].Material.Name if material.ForLayerSet.MaterialLayers else ""

        return properties

    def _validate_model(self, ifc_file, model: BIMModel) -> Dict:
        """Validate the IFC model and generate a report."""
        report = {
            "file_valid": True,
            "warnings": [],
            "errors": [],
            "stats": {}
        }

        # Check for required elements
        required_types = ["IfcProject", "IfcBuilding", "IfcBuildingStorey"]
        for req_type in required_types:
            elements = ifc_file.by_type(req_type)
            if not elements:
                report["warnings"].append(f"Missing {req_type}")

        # Check geometry and duplicate GlobalIds using extracted elements
        elements_with_geom = 0
        elements_without_geom = 0
        global_ids = {}

        for elem in model.elements.values():
            if elem.geometry:
                elements_with_geom += 1
            else:
                elements_without_geom += 1
                
            gid = elem.global_id
            if gid in global_ids:
                report["errors"].append(f"Duplicate GlobalId: {gid}")
            global_ids[gid] = elem.id

        report["stats"]["elements_with_geometry"] = elements_with_geom
        report["stats"]["elements_without_geometry"] = elements_without_geom

        # Check for spaces
        if not model.spaces:
            report["warnings"].append("No spaces found in model")
        
        for space in model.spaces.values():
            gid = space.global_id
            if gid in global_ids:
                report["errors"].append(f"Duplicate GlobalId: {gid}")
            global_ids[gid] = space.id

        report["stats"]["total_elements"] = len(model.elements)
        report["stats"]["total_spaces"] = len(model.spaces)
        report["stats"]["total_levels"] = len(model.levels)
        report["stats"]["unique_global_ids"] = len(global_ids)

        return report

    def get_spaces_by_level(self, level_id: str) -> List[BIMSpace]:
        """Get all spaces on a specific level."""
        if not self.current_model:
            return []
        return [s for s in self.current_model.spaces.values() if s.level == level_id]

    def get_elements_by_category(self, category: ElementCategory) -> List[BIMElement]:
        """Get all elements of a specific category."""
        if not self.current_model:
            return []
        return [e for e in self.current_model.elements.values() if e.category == category]

    def export_summary(self, model: BIMModel) -> str:
        """Export a text summary of the BIM model."""
        summary = f"""
BIM Model Summary
================
Project: {model.name}
File: {model.file_path}
Schema: {model.schema}

Project Information:
"""
        for key, value in model.project_info.items():
            summary += f"  {key}: {value}\n"

        summary += f"""
Statistics:
  Total Elements: {len(model.elements)}
  Total Spaces: {len(model.spaces)}
  Total Levels: {len(model.levels)}

Levels:
"""
        for level in sorted(model.levels.values(), key=lambda l: l.elevation):
            summary += f"  {level.name}: Elevation {level.elevation:.2f}m, Height {level.height:.2f}m\n"

        summary += f"""
Element Categories:
"""
        from collections import Counter
        categories = Counter(e.category.value for e in model.elements.values())
        for cat, count in categories.most_common():
            summary += f"  {cat}: {count}\n"

        summary += f"""
Space Categories:
"""
        space_cats = Counter(s.category for s in model.spaces.values())
        for cat, count in space_cats.most_common():
            summary += f"  {cat}: {count}\n"

        return summary
