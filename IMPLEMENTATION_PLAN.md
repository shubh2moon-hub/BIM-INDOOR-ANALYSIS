# BIM-CrowdSim Implementation Plan: Practical Fixes & Wayfinding Paths

## Problem Statement

The current BIM-CrowdSim works **in theory** but not **in practice** because:

1. **Room Configurator is broken/too limited**: Only allows setting agent count and fire origin. No way to fix missing space types, missing doors, or mark exits.
2. **No visible paths**: Users cannot see where agents will go — no multicolored wayfinding/escape route lines from rooms to exits.
3. **Cannot handle incomplete IFC models**: If the model has unmarked spaces, missing doors, or no explicit exit elements, the spatial graph is broken and simulation fails.
4. **No manual override**: Users cannot manually mark a space as "corridor", "office", "stair", etc. They cannot manually add an exit when the model doesn't have one.

## Solution Overview

Build a **Model Annotation Layer** that sits between the raw IFC import and the spatial engine. This layer lets users:

- **Override space types** (room, corridor, stair, elevator, restroom, office, meeting, public)
- **Mark spaces as exits** (when the model lacks explicit exit doors)
- **Add virtual exits** at arbitrary positions (when no door exists at all)
- **Add virtual connections** between spaces (when doors are missing)
- **See evacuation paths** from every space to the nearest exit as multicolored 3D lines

---

## Phase 1: Core Annotation Model (`core/model_annotations.py`)

### 1.1 Data Structures

```python
@dataclass
class VirtualExit:
    id: str
    name: str
    position: Tuple[float, float, float]
    level_id: str
    width: float = 1.2
    accessible: bool = True

@dataclass
class SpaceAnnotation:
    space_id: str
    category_override: Optional[str] = None   # "room", "corridor", "stair", ...
    is_exit: bool = False
    is_fire_origin: bool = False
    agent_count: int = 0
    custom_name: Optional[str] = None

@dataclass
class VirtualConnection:
    id: str
    source_space_id: str
    target_space_id: str
    connection_type: str = "virtual_door"
    width: float = 1.0

@dataclass
class ModelAnnotations:
    exits: List[VirtualExit]
    space_annotations: Dict[str, SpaceAnnotation]
    virtual_connections: List[VirtualConnection]
```

### 1.2 Integration with BIMModel

- Add `annotations: ModelAnnotations` field to `BIMModel`
- Initialize with empty annotations on IFC load
- Provide `apply_annotations()` method that overrides space categories and adds virtual elements

---

## Phase 2: Spatial Engine Enhancement (`core/spatial_engine.py`)

### 2.1 Accept Annotations

Modify `process_model()` to accept optional `annotations: ModelAnnotations`.

### 2.2 Apply Space Type Overrides

- Before creating `SpaceNode`s, apply `category_override` from annotations
- If a space has an override, use it instead of the auto-detected category

### 2.3 Add Virtual Exits as Nodes

- Create `NavigationNode` for each `VirtualExit` with `node_type = "exit"`
- Connect nearby spaces (within 10m) to the exit via `Connection` edges
- Mark these connections with `accessible=True` and `connection_type="exit"`

### 2.4 Add Virtual Connections

- For each `VirtualConnection`, add an edge between the two spaces in the graph

### 2.5 Mark Exit Spaces

- If a space is annotated as `is_exit`, add it as a direct exit node in the navigation graph
- Compute shortest paths from every space to every exit
- Store these paths for visualization

### 2.6 Path Computation for Visualization

Add new method to `SpatialIntelligenceEngine`:

```python
def compute_evacuation_paths(self, to_nearest_exit: bool = True) -> Dict[str, List[str]]:
    """Compute shortest path from every space to the nearest exit."""
```

Returns a dict mapping `space_id -> [space_id, ...]` path to exit.

---

## Phase 3: Path Visualization (`visualization/viz_engine.py`)

### 3.1 New Method: `display_evacuation_paths()`

- Read precomputed paths from spatial graph
- For each path, create a 3D tube (or thick line) from space center → space center → exit
- Color each path uniquely using a colormap (e.g., one color per path, or rainbow per floor)
- Use PyVista `add_mesh` with `pv.Line` or `pv.Tube`
- Make paths toggleable via `show_evacuation_paths` setting

### 3.2 Color Scheme

- Each floor gets a distinct base hue
- Paths on the same floor vary slightly in shade
- Exit nodes highlighted as green spheres with labels

### 3.3 Add Path Actors to Actor Registry

- Store path actors in `self.path_actors` dict
- Clear/rebuild on refresh
- Toggle visibility with setting

---

## Phase 4: Room Configurator Overhaul (`gui/main_window.py`)

### 4.1 New Room Configurator Layout

```
Room Configurator Panel
├── Section: Selected Space Info
│   ├── Name (editable custom name)
│   ├── Current Category (read-only from IFC)
│   └── Override Category (dropdown)
├── Section: Exit & Safety
│   ├── [ ] Mark as Exit
│   ├── [ ] Set as Fire Origin
│   └── [ ] Block Path
├── Section: Agents
│   └── Number of Agents: [spinbox]
├── Section: Evacuation Path
│   ├── [Show Path from This Room] button
│   └── Path preview text (space names)
├── Section: Virtual Exits (global, not per-room)
│   ├── [Add Virtual Exit at Center] button
│   └── List of virtual exits with delete
├── Section: Global Actions
│   ├── [Show All Escape Routes]
│   ├── [Hide All Paths]
│   └── [Recompute Spatial Graph]
└── Section: Apply
    └── [Apply to Simulation] button
```

### 4.2 Category Override Dropdown

Options: `Use IFC Default`, `Room`, `Corridor`, `Stair`, `Elevator`, `Restroom`, `Office`, `Meeting`, `Public`, `Exit`

### 4.3 Add Virtual Exit

- When user clicks "Add Virtual Exit", place it at the current selected space's center
- Allow position editing (X, Y, Z spinboxes)
- Show list of all virtual exits with delete buttons

### 4.4 Show Path Button

- When clicked, compute path from selected space to nearest exit
- Call visualization to highlight that path
- Show text description in the panel (e.g., "Room 101 → Corridor A → Lobby → Exit")

### 4.5 Apply to Simulation

- Re-process spatial graph with annotations
- Re-initialize simulation with updated graph
- Update 3D view with new paths

---

## Phase 5: Integration (`gui/main_window.py`)

### 5.1 Data Flow

```
User opens IFC
    → BIMProcessor loads model
    → ModelAnnotations initialized (empty)
    → SpatialEngine.process_model(model, annotations)
    → 3D view shows building

User annotates spaces / adds exits
    → Annotations stored in model.annotations
    → "Recompute" triggers:
        → spatial_engine.process_model(model, model.annotations)
        → viz_engine.load_bim_model(model)  [with paths]
        → simulation re-initialized if needed

User runs simulation
    → Agents use updated spatial graph with virtual exits
    → Agents can actually find their way out
```

### 5.2 Menu Additions

- **Model → Annotations** submenu:
  - `Show All Paths`
  - `Hide All Paths`
  - `Clear All Annotations`
  - `Export Annotations`
  - `Import Annotations`

### 5.3 Project Tree Updates

- Show annotation indicators on tree items:
  - `*` if category overridden
  - `🚪` if marked as exit
  - `🔥` if fire origin
  - `#N` if agent count set

---

## Phase 6: Fallback & Robustness

### 6.1 Handle Models with No Spaces

If `IfcSpace` elements are missing:
- Detect in `_extract_spaces()`
- Log warning
- User must manually create spaces OR the system should auto-generate placeholder spaces from bounding boxes of rooms (advanced — out of scope for this pass, but warn user)

### 6.2 Handle Models with No Doors

If no `IfcDoor` elements found:
- Log warning
- Proximity-based connectivity becomes primary method
- User must add virtual exits/connections manually
- Automatically suggest nearby spaces as potential connections

### 6.3 Handle Models with No Explicit Exits

If no spaces are categorized as "exit" or "entrance":
- Scan for spaces near building perimeter (low neighbor count, high betweenness)
- Suggest these as potential exits to user
- Allow user to mark any space as exit

---

## Implementation Order

1. **Phase 1**: `core/model_annotations.py` (new file)
2. **Phase 2**: Modify `core/spatial_engine.py` to accept annotations
3. **Phase 3**: Modify `visualization/viz_engine.py` for path rendering
4. **Phase 4**: Rewrite room configurator in `gui/main_window.py`
5. **Phase 5**: Wire integration in `gui/main_window.py`
6. **Phase 6**: Test end-to-end

---

## Files to Modify

| File | Action | Description |
|------|--------|-------------|
| `core/model_annotations.py` | **CREATE** | Annotation data model |
| `core/bim_processor.py` | **MODIFY** | Add `annotations` field to `BIMModel` |
| `core/spatial_engine.py` | **MODIFY** | Accept annotations, apply overrides, compute paths |
| `visualization/viz_engine.py` | **MODIFY** | Add evacuation path rendering |
| `gui/main_window.py` | **MODIFY** | New room configurator, integration, menus |

---

## Expected Behavior After Implementation

1. User opens an IFC model (even a messy one)
2. Sees the building in 3D
3. Clicks on a space in the project tree or 3D view
4. In the **Room Configurator** panel:
   - Overrides the category if it was auto-detected wrong
   - Sets agent count for that room
   - Marks it as an exit if it's near the outside
   - Adds a virtual exit if no door exists
5. Clicks **"Show Escape Routes"** — sees multicolored 3D lines from every room to the nearest exit
6. Clicks **"Recompute Spatial Graph"** — the graph is rebuilt with annotations
7. Loads a simulation preset — agents now use the corrected graph and can actually reach exits
8. Runs simulation — agents follow visible paths, and the user can watch them evacuate
