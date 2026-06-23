# BIM-INDOOR-ANALYSIS

## BIM-native Agent-Based Modeling and Simulation Platform

BIM-Agent Studio transforms BIM models (IFC files) into intelligent simulation environments where people, vehicles, services, resources, and environmental phenomena can interact within a digital representation of a building. The platform combines BIM data processing, spatial graph generation, agent-based simulation, and real-time 3D visualization in a single workflow.

Unlike traditional BIM viewers that focus on geometry and documentation, BIM-Agent Studio focuses on behavior, movement, interaction, and performance within built environments.

## Features

### 1. IFC Import & BIM Processing
- Import IFC models from Revit, Archicad, Tekla, FreeCAD
- Automatic model validation and repair
- Extract walls, doors, windows, floors, stairs, spaces, furniture

### 2. Spatial Intelligence Engine
- Automatic space recognition (rooms, corridors, stairs, etc.)
- Connectivity detection between spaces
- Navigation network generation using NetworkX
- Accessibility analysis

### 3. Agent-Based Simulation Engine (Mesa)
- Human agents (office workers, students, patients, visitors)
- Vehicle agents (cars, ambulances, delivery vehicles)
- Service agents (cleaning, security, maintenance)
- Autonomous agents (robots, drones)

### 4. Environmental Simulation
- Fire & smoke spread
- Evacuation dynamics
- Crowd density analysis
- Indoor airflow patterns

### 5. Social & Behavioral Analysis
- Social interaction frequency
- Spatial clustering analysis
- Segregation pattern detection
- Space utilization metrics

### 6. Real-Time 3D Visualization
- Interactive 3D building geometry
- Moving agent visualization
- Heat maps and density overlays
- Movement trails
- Pause, rewind, fast-forward controls

### 7. Scenario Builder
- Visual scenario creation (no coding required)
- Preset scenarios (Office, Hospital, University, Evacuation)
- Custom agent types and behaviors
- Event scheduling system

### 8. Simulation Analytics
- Movement metrics (travel distance, time, speed)
- Occupancy metrics (utilization, peak times)
- Accessibility metrics (route difficulty, coverage)
- Social interaction analysis
- Export to CSV/JSON

## Technology Stack

| Layer | Technology |
|-------|-----------|
| GUI | PySide6 |
| BIM Processing | IfcOpenShell |
| Simulation | Mesa |
| Graph Engine | NetworkX |
| Visualization | PyVista + VTK |
| Data Analysis | Pandas |
| Storage | SQLite |
| Packaging | PyInstaller |

## Installation

### Prerequisites
- Python 3.10+
- pip

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Run Application

```bash
python app.py
```

### Open IFC File

```bash
python app.py --ifc path/to/model.ifc
```

## Building Executable

### Single File
```bash
python build.py --mode onefile
```

### Directory Mode (faster build)
```bash
python build.py --mode onedir
```

## Usage Guide

### Basic Workflow

1. **Open IFC Model**: File > Open IFC or click "Open IFC" button
2. **Process Spatial Intelligence**: Model > Process Spatial Intelligence
3. **Load Scenario**: Simulation > Presets > Select scenario
4. **Run Simulation**: Click Start button
5. **View Analytics**: Check Analytics panel for results

### Creating Custom Scenarios

1. Open Scenario Builder: Simulation > New Scenario
2. Add Agent Profiles with custom behaviors
3. Configure Events (spawn, evacuate, fire, etc.)
4. Save and run

### Visualization Controls

- **View Modes**: BIM Only, Agents, Density, Heat Map, Evacuation, Navigation
- **Camera Views**: Perspective, Top, Front, Side, Isometric
- **Speed Control**: Adjust simulation speed with slider

## Project Structure

```
bim-agent-studio/
|-- app.py                  # Main entry point
|-- build.py                # Build script
|-- requirements.txt        # Python dependencies
|-- core/                   # Core BIM processing
|   |-- bim_processor.py    # IFC import and processing
|   |-- spatial_engine.py   # Spatial intelligence engine
|-- engine/                 # Simulation engine
|   |-- simulation_engine.py # Mesa-based ABM simulation
|-- gui/                    # User interface
|   |-- main_window.py      # Main application window
|   |-- scenario_builder.py # Scenario creation UI
|-- visualization/          # 3D visualization
|   |-- viz_engine.py       # PyVista/VTK visualization
|-- data/                   # Data storage
|-- tests/                  # Unit tests
```

## Architecture

```
IFC Model
    |
    v
Model Validation & Repair
    |
    v
Spatial Understanding Engine (NetworkX)
    |
    v
Agent Generation (Mesa)
    |
    v
Simulation Engine (Mesa + Custom)
    |
    v
3D Visualization (PyVista + VTK)
    |
    v
Analysis & Reports (Pandas)
```

## License

This project is released under the MIT License.

## Contributing

Contributions are welcome! Please submit issues and pull requests.

## Acknowledgments

- Inspired by the GAMA Platform (gama-platform.org)
- Uses IfcOpenShell for BIM processing
- Built on Mesa framework for agent-based modeling
- Visualization powered by PyVista and VTK
