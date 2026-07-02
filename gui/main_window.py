"""
Main Window for BIM-CrowdSim
Provides the main application window with dockable panels, menus, and toolbars.
"""

import os
import sys
import logging
from typing import Optional, Dict, List, Callable
from pathlib import Path

from PySide6.QtCore import Qt, QTimer, Signal, Slot, QSize
from PySide6.QtGui import QAction, QIcon, QKeySequence, QFont, QColor
from PySide6.QtWidgets import (
    QMainWindow, QWidget, QDockWidget, QMenuBar, QToolBar,
    QStatusBar, QFileDialog, QMessageBox, QProgressDialog,
    QVBoxLayout, QHBoxLayout, QSplitter, QLabel, QPushButton,
    QComboBox, QSlider, QSpinBox, QDoubleSpinBox, QCheckBox,
    QGroupBox, QTabWidget, QTreeWidget, QTreeWidgetItem,
    QTableWidget, QTableWidgetItem, QHeaderView, QTextEdit,
    QProgressBar, QFrame, QApplication, QStyle, QSizePolicy,
    QMenu, QToolButton, QWidgetAction, QInputDialog
)

from core.bim_processor import BIMProcessor, BIMModel, BIMSpace, BIMElement, ElementCategory
from core.spatial_engine import SpatialIntelligenceEngine, SpatialGraph
from engine.simulation_engine import SimulationEngine, BIMSimulationModel, ScenarioPresets
from visualization.viz_engine import Visualization3D, VisualizationSettings, VisualizationMode

logger = logging.getLogger(__name__)


class MainWindow(QMainWindow):
    """Main application window for BIM-CrowdSim."""
    
    # Signals
    model_loaded = Signal(object)  # BIMModel
    simulation_started = Signal()
    simulation_paused = Signal()
    simulation_stopped = Signal()
    
    def __init__(self):
        super().__init__()
        
        self.setWindowTitle("BIM-CrowdSim")
        self.setMinimumSize(1400, 900)
        
        # Core components
        self.bim_processor = BIMProcessor()
        self.spatial_engine = SpatialIntelligenceEngine()
        self.simulation_engine = SimulationEngine()
        self.visualization = Visualization3D()
        
        # State
        self.current_model: Optional[BIMModel] = None
        self.current_simulation: Optional[BIMSimulationModel] = None
        self.is_simulation_running = False
        self.simulation_timer: Optional[QTimer] = None
        
        # Room config state
        self.custom_room_settings = {}
        self.currently_selected_space_id = None
        
        # UI Components storage
        self.panels: Dict[str, QDockWidget] = {}
        
        # Build UI
        self._create_menu_bar()
        self._create_tool_bars()
        self._create_status_bar()
        self._create_central_widget()
        self._create_dock_panels()
        self._create_room_config_panel()
        
        # Connect signals
        self.visualization.space_selected.connect(self._on_space_selected)
        
        # Setup simulation timer
        self.simulation_timer = QTimer(self)
        self.simulation_timer.timeout.connect(self._simulation_step)
        
        logger.info("Main window initialized")
        
    def _create_menu_bar(self):
        """Create the application menu bar."""
        menubar = self.menuBar()
        
        # File Menu
        file_menu = menubar.addMenu("&File")
        
        open_action = QAction("&Open IFC...", self)
        open_action.setShortcut(QKeySequence.Open)
        open_action.setStatusTip("Open an IFC BIM model")
        open_action.triggered.connect(self._on_open_ifc)
        file_menu.addAction(open_action)
        
        file_menu.addSeparator()
        
        recent_menu = file_menu.addMenu("Recent Files")
        recent_menu.addAction("No recent files")
        
        file_menu.addSeparator()
        
        save_action = QAction("&Save Project", self)
        save_action.setShortcut(QKeySequence.Save)
        save_action.triggered.connect(self._on_save_project)
        file_menu.addAction(save_action)
        
        export_action = QAction("&Export Results...", self)
        export_action.triggered.connect(self._on_export_results)
        file_menu.addAction(export_action)
        
        file_menu.addSeparator()
        
        exit_action = QAction("E&xit", self)
        exit_action.setShortcut(QKeySequence.Quit)
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)
        
        # View Menu
        view_menu = menubar.addMenu("&View")
        
        for panel_name in ["Project", "Properties", "Simulation", "Analytics", "Console"]:
            action = QAction(f"Show {panel_name} Panel", self)
            action.setCheckable(True)
            action.setChecked(True)
            action.triggered.connect(lambda checked, name=panel_name: self._toggle_panel(name, checked))
            view_menu.addAction(action)
            
        view_menu.addSeparator()
        
        # Visualization modes
        viz_menu = view_menu.addMenu("Visualization Mode")
        self.viz_mode_group = {}
        for mode in ["BIM Only", "Agents", "Density", "Heat Map", "Evacuation", "Navigation"]:
            action = QAction(mode, self)
            action.setCheckable(True)
            action.triggered.connect(lambda checked, m=mode: self._set_visualization_mode(m))
            viz_menu.addAction(action)
            self.viz_mode_group[mode] = action
            
        # Model Menu
        model_menu = menubar.addMenu("&Model")
        
        validate_action = QAction("&Validate Model", self)
        validate_action.triggered.connect(self._on_validate_model)
        model_menu.addAction(validate_action)
        
        process_spatial_action = QAction("&Process Spatial Intelligence", self)
        process_spatial_action.triggered.connect(self._on_process_spatial)
        model_menu.addAction(process_spatial_action)
        
        model_menu.addSeparator()
        
        # Annotation actions
        show_paths_action = QAction("Show &All Escape Routes", self)
        show_paths_action.triggered.connect(self._on_show_all_paths_clicked)
        model_menu.addAction(show_paths_action)
        
        hide_paths_action = QAction("&Hide All Paths", self)
        hide_paths_action.triggered.connect(self._on_hide_paths_clicked)
        model_menu.addAction(hide_paths_action)
        
        model_menu.addSeparator()
        
        model_info_action = QAction("Model &Information", self)
        model_info_action.triggered.connect(self._on_model_info)
        model_menu.addAction(model_info_action)
        
        # Simulation Menu
        sim_menu = menubar.addMenu("&Simulation")
        
        new_scenario_action = QAction("&New Scenario...", self)
        new_scenario_action.triggered.connect(self._on_new_scenario)
        sim_menu.addAction(new_scenario_action)
        
        presets_menu = sim_menu.addMenu("&Presets")
        
        office_preset = QAction("&Office Daily Operations", self)
        office_preset.triggered.connect(lambda: self._load_preset("office"))
        presets_menu.addAction(office_preset)
        
        evac_preset = QAction("&Emergency Evacuation", self)
        evac_preset.triggered.connect(lambda: self._load_preset("evacuation"))
        presets_menu.addAction(evac_preset)
        
        hospital_preset = QAction("&Hospital Operations", self)
        hospital_preset.triggered.connect(lambda: self._load_preset("hospital"))
        presets_menu.addAction(hospital_preset)
        
        uni_preset = QAction("&University Class Transitions", self)
        uni_preset.triggered.connect(lambda: self._load_preset("university"))
        presets_menu.addAction(uni_preset)
        
        fire_evac_preset = QAction("&Fire Evacuation", self)
        fire_evac_preset.triggered.connect(lambda: self._load_preset("fire_evacuation"))
        presets_menu.addAction(fire_evac_preset)
        
        sim_menu.addSeparator()
        
        start_action = QAction("&Start", self)
        start_action.setShortcut("Ctrl+R")
        start_action.triggered.connect(self._on_start_simulation)
        sim_menu.addAction(start_action)
        
        pause_action = QAction("&Pause", self)
        pause_action.triggered.connect(self._on_pause_simulation)
        sim_menu.addAction(pause_action)
        
        stop_action = QAction("S&top", self)
        stop_action.triggered.connect(self._on_stop_simulation)
        sim_menu.addAction(stop_action)
        
        step_action = QAction("Step &Forward", self)
        step_action.setShortcut("Ctrl+Right")
        step_action.triggered.connect(self._on_step_forward)
        sim_menu.addAction(step_action)
        
        reset_action = QAction("&Reset", self)
        reset_action.triggered.connect(self._on_reset_simulation)
        sim_menu.addAction(reset_action)
        
        # Tools Menu
        tools_menu = menubar.addMenu("&Tools")
        
        prefs_action = QAction("&Preferences...", self)
        prefs_action.triggered.connect(self._on_preferences)
        tools_menu.addAction(prefs_action)
        
        # Help Menu
        help_menu = menubar.addMenu("&Help")
        
        about_action = QAction("&About", self)
        about_action.triggered.connect(self._on_about)
        help_menu.addAction(about_action)
        
    def _create_tool_bars(self):
        """Create application toolbars."""
        # Main Toolbar
        main_toolbar = QToolBar("Main", self)
        main_toolbar.setMovable(False)
        self.addToolBar(main_toolbar)
        
        open_btn = QPushButton("Open IFC")
        open_btn.clicked.connect(self._on_open_ifc)
        main_toolbar.addWidget(open_btn)
        
        main_toolbar.addSeparator()
        
        # Simulation Controls
        self.btn_start = QPushButton("Start")
        self.btn_start.clicked.connect(self._on_start_simulation)
        main_toolbar.addWidget(self.btn_start)
        
        self.btn_pause = QPushButton("Pause")
        self.btn_pause.clicked.connect(self._on_pause_simulation)
        self.btn_pause.setEnabled(False)
        main_toolbar.addWidget(self.btn_pause)
        
        self.btn_stop = QPushButton("Stop")
        self.btn_stop.clicked.connect(self._on_stop_simulation)
        self.btn_stop.setEnabled(False)
        main_toolbar.addWidget(self.btn_stop)
        
        self.btn_step = QPushButton("Step")
        self.btn_step.clicked.connect(self._on_step_forward)
        main_toolbar.addWidget(self.btn_step)
        
        main_toolbar.addSeparator()
        
        # Speed control
        main_toolbar.addWidget(QLabel("Speed:"))
        self.speed_slider = QSlider(Qt.Horizontal)
        self.speed_slider.setRange(1, 100)
        self.speed_slider.setValue(10)
        self.speed_slider.setMaximumWidth(100)
        main_toolbar.addWidget(self.speed_slider)
        
        main_toolbar.addSeparator()
        
        # View controls
        view_label = QLabel("View:")
        main_toolbar.addWidget(view_label)
        
        self.view_combo = QComboBox()
        self.view_combo.addItems(["Perspective", "Top", "Front", "Side", "Isometric"])
        self.view_combo.currentTextChanged.connect(self._on_view_changed)
        main_toolbar.addWidget(self.view_combo)
        
        main_toolbar.addSeparator()
        
        # Floor controls
        floor_label = QLabel("Floor:")
        main_toolbar.addWidget(floor_label)
        
        self.floor_combo = QComboBox()
        self.floor_combo.addItem("All Floors", None)
        self.floor_combo.currentIndexChanged.connect(self._on_floor_changed)
        main_toolbar.addWidget(self.floor_combo)
        
        main_toolbar.addSeparator()
        
        # Wall Opacity controls
        opacity_label = QLabel("Wall Opacity:")
        main_toolbar.addWidget(opacity_label)
        
        self.opacity_slider = QSlider(Qt.Horizontal)
        self.opacity_slider.setRange(0, 100)
        self.opacity_slider.setValue(100)
        self.opacity_slider.setMaximumWidth(100)
        self.opacity_slider.valueChanged.connect(self._on_wall_opacity_changed)
        main_toolbar.addWidget(self.opacity_slider)
        
        # Simulation Toolbar
        sim_toolbar = QToolBar("Simulation", self)
        sim_toolbar.setMovable(True)
        self.addToolBar(Qt.RightToolBarArea, sim_toolbar)
        
        # Time display
        self.time_label = QLabel("Time: 00:00:00")
        sim_toolbar.addWidget(self.time_label)
        
        sim_toolbar.addSeparator()
        
        # Agent count
        self.agent_count_label = QLabel("Agents: 0")
        sim_toolbar.addWidget(self.agent_count_label)
        
    def _create_status_bar(self):
        """Create the status bar."""
        self.statusbar = QStatusBar()
        self.setStatusBar(self.statusbar)
        
        self.statusbar.showMessage("Ready")
        
        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setMaximumWidth(200)
        self.progress_bar.setVisible(False)
        self.statusbar.addPermanentWidget(self.progress_bar)
        
        # Model info label
        self.model_info_label = QLabel("No model loaded")
        self.statusbar.addPermanentWidget(self.model_info_label)
        
    def _create_central_widget(self):
        """Create the central widget with 3D viewer."""
        central = QWidget()
        self.setCentralWidget(central)
        
        layout = QVBoxLayout(central)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        
        # Create 3D visualization widget
        self.viz_container = QWidget()
        viz_layout = QVBoxLayout(self.viz_container)
        viz_layout.setContentsMargins(0, 0, 0, 0)
        
        # Placeholder for 3D view
        self.viz_placeholder = QLabel("3D Viewer - Open an IFC model to begin")
        self.viz_placeholder.setAlignment(Qt.AlignCenter)
        self.viz_placeholder.setStyleSheet("""
            QLabel {
                background-color: #f0f0f5;
                color: #666;
                font-size: 16px;
                border: 2px dashed #ccc;
                margin: 20px;
            }
        """)
        viz_layout.addWidget(self.viz_placeholder)
        
        layout.addWidget(self.viz_container)
        
    def _create_dock_panels(self):
        """Create dockable side panels."""
        # Project Panel - Left
        self._create_project_panel()
        
        # Properties Panel - Left
        self._create_properties_panel()
        
        # Simulation Panel - Right
        self._create_simulation_panel()
        
        # Analytics Panel - Right
        self._create_analytics_panel()
        
        # Console Panel - Bottom
        self._create_console_panel()
        
    def _create_project_panel(self):
        """Create the project explorer panel."""
        dock = QDockWidget("Project", self)
        dock.setAllowedAreas(Qt.LeftDockWidgetArea | Qt.RightDockWidgetArea)
        
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # IFC Tree
        self.ifc_tree = QTreeWidget()
        self.ifc_tree.setHeaderLabel("IFC Model Structure")
        self.ifc_tree.itemClicked.connect(self._on_tree_item_clicked)
        layout.addWidget(self.ifc_tree)
        
        # Model info
        self.model_summary = QTextEdit()
        self.model_summary.setReadOnly(True)
        self.model_summary.setMaximumHeight(150)
        layout.addWidget(self.model_summary)
        
        dock.setWidget(widget)
        self.addDockWidget(Qt.LeftDockWidgetArea, dock)
        self.panels["Project"] = dock
        
    def _create_room_config_panel(self):
        """Create the comprehensive room configurator panel."""
        dock = QDockWidget("Room Configurator", self)
        dock.setAllowedAreas(Qt.LeftDockWidgetArea | Qt.RightDockWidgetArea)
        
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setSpacing(8)
        
        # --- Section: Selected Space Info ---
        info_group = QGroupBox("Selected Space")
        info_layout = QVBoxLayout(info_group)
        
        self.lbl_selected_room = QLabel("Selected Room: None")
        self.lbl_selected_room.setStyleSheet("font-weight: bold; font-size: 12px;")
        info_layout.addWidget(self.lbl_selected_room)
        
        self.lbl_room_ifc_category = QLabel("IFC Category: —")
        info_layout.addWidget(self.lbl_room_ifc_category)
        
        self.lbl_room_occupancy = QLabel("IFC Occupancy: N/A")
        info_layout.addWidget(self.lbl_room_occupancy)
        
        layout.addWidget(info_group)
        
        # --- Section: Category Override ---
        cat_group = QGroupBox("Space Type Override")
        cat_layout = QVBoxLayout(cat_group)
        
        self.combo_category_override = QComboBox()
        self.combo_category_override.addItem("Use IFC Default", None)
        for cat in ["room", "corridor", "staircase", "elevator", "restroom", "office", "meeting", "public", "entrance", "exit", "lobby", "void"]:
            self.combo_category_override.addItem(cat.replace("_", " ").title(), cat)
        self.combo_category_override.setEnabled(False)
        self.combo_category_override.currentIndexChanged.connect(self._on_category_override_changed)
        cat_layout.addWidget(self.combo_category_override)
        
        layout.addWidget(cat_group)
        
        # --- Section: Exit & Safety ---
        safety_group = QGroupBox("Exit & Safety")
        safety_layout = QVBoxLayout(safety_group)
        
        self.chk_mark_exit = QCheckBox("Mark as Exit")
        self.chk_mark_exit.setEnabled(False)
        self.chk_mark_exit.setToolTip("This space is an exit / leads to the outside")
        safety_layout.addWidget(self.chk_mark_exit)
        
        self.chk_start_fire = QCheckBox("Set as Fire Origin")
        self.chk_start_fire.setEnabled(False)
        safety_layout.addWidget(self.chk_start_fire)
        
        self.chk_block_path = QCheckBox("Block Path (unreachable)")
        self.chk_block_path.setEnabled(False)
        self.chk_block_path.setToolTip("Agents cannot walk through this space")
        safety_layout.addWidget(self.chk_block_path)
        
        layout.addWidget(safety_group)
        
        # --- Section: Agents ---
        agents_group = QGroupBox("Agents")
        agents_layout = QVBoxLayout(agents_group)
        
        agents_row = QHBoxLayout()
        agents_row.addWidget(QLabel("Number of Agents:"))
        self.spin_room_agents = QSpinBox()
        self.spin_room_agents.setRange(0, 1000)
        self.spin_room_agents.setEnabled(False)
        agents_row.addWidget(self.spin_room_agents)
        agents_layout.addLayout(agents_row)
        
        layout.addWidget(agents_group)
        
        # --- Section: Evacuation Path ---
        path_group = QGroupBox("Evacuation Path")
        path_layout = QVBoxLayout(path_group)
        
        self.lbl_path_preview = QLabel("Path: —")
        self.lbl_path_preview.setWordWrap(True)
        path_layout.addWidget(self.lbl_path_preview)
        
        btn_show_path = QPushButton("Show Path from This Room")
        btn_show_path.clicked.connect(self._on_show_path_clicked)
        btn_show_path.setEnabled(False)
        self.btn_show_path = btn_show_path
        path_layout.addWidget(btn_show_path)
        
        layout.addWidget(path_group)
        
        # --- Section: Virtual Exits (global) ---
        v_exit_group = QGroupBox("Virtual Exits")
        v_exit_layout = QVBoxLayout(v_exit_group)
        
        self.lbl_virtual_exits = QLabel("No virtual exits added")
        self.lbl_virtual_exits.setWordWrap(True)
        v_exit_layout.addWidget(self.lbl_virtual_exits)
        
        btn_add_v_exit = QPushButton("Add Virtual Exit at Selected Space")
        btn_add_v_exit.clicked.connect(self._on_add_virtual_exit)
        btn_add_v_exit.setEnabled(False)
        self.btn_add_v_exit = btn_add_v_exit
        v_exit_layout.addWidget(btn_add_v_exit)
        
        layout.addWidget(v_exit_group)
        
        # --- Section: Virtual Spaces (global) ---
        v_space_group = QGroupBox("Virtual Spaces")
        v_space_layout = QVBoxLayout(v_space_group)
        
        self.lbl_virtual_spaces = QLabel("No virtual spaces added")
        self.lbl_virtual_spaces.setWordWrap(True)
        v_space_layout.addWidget(self.lbl_virtual_spaces)
        
        btn_add_v_space = QPushButton("Add Virtual Space at Camera Center")
        btn_add_v_space.clicked.connect(self._on_add_virtual_space)
        btn_add_v_space.setToolTip("Creates a virtual room at the current camera focal point")
        v_space_layout.addWidget(btn_add_v_space)
        
        layout.addWidget(v_space_group)
        
        # --- Section: Global Actions ---
        global_group = QGroupBox("Global Actions")
        global_layout = QVBoxLayout(global_group)
        
        btn_show_all = QPushButton("Show All Escape Routes")
        btn_show_all.clicked.connect(self._on_show_all_paths_clicked)
        global_layout.addWidget(btn_show_all)
        
        btn_hide_all = QPushButton("Hide All Paths")
        btn_hide_all.clicked.connect(self._on_hide_paths_clicked)
        global_layout.addWidget(btn_hide_all)
        
        btn_recompute = QPushButton("Recompute Spatial Graph")
        btn_recompute.clicked.connect(self._on_recompute_graph_clicked)
        btn_recompute.setStyleSheet("background-color: #0078d4; color: white;")
        global_layout.addWidget(btn_recompute)
        
        layout.addWidget(global_group)
        
        # --- Apply Button ---
        self.btn_apply_room = QPushButton("Apply & Recompute")
        self.btn_apply_room.setEnabled(False)
        self.btn_apply_room.setStyleSheet("background-color: #28a745; color: white; font-weight: bold;")
        self.btn_apply_room.clicked.connect(self._on_apply_room_config)
        layout.addWidget(self.btn_apply_room)
        
        layout.addStretch()
        
        dock.setWidget(widget)
        self.addDockWidget(Qt.RightDockWidgetArea, dock)
        self.panels["RoomConfigurator"] = dock

    def _on_space_selected(self, space_id: str):
        self.currently_selected_space_id = space_id
        
        has_model = self.current_model is not None and space_id in self.current_model.spaces
        
        self.combo_category_override.setEnabled(has_model)
        self.chk_mark_exit.setEnabled(has_model)
        self.chk_start_fire.setEnabled(has_model)
        self.chk_block_path.setEnabled(has_model)
        self.spin_room_agents.setEnabled(has_model)
        self.btn_show_path.setEnabled(has_model)
        self.btn_add_v_exit.setEnabled(has_model)
        self.btn_apply_room.setEnabled(has_model)
        
        if not has_model:
            self.lbl_selected_room.setText("Selected Room: None")
            self.lbl_room_ifc_category.setText("IFC Category: —")
            self.lbl_room_occupancy.setText("IFC Occupancy: N/A")
            self.lbl_path_preview.setText("Path: —")
            return
        
        space = self.current_model.spaces[space_id]
        self.lbl_selected_room.setText(f"Selected Room: {space.name}")
        self.lbl_room_ifc_category.setText(f"IFC Category: {space.category}")
        
        occupancy = space.occupancy_capacity
        if occupancy is not None:
            self.lbl_room_occupancy.setText(f"IFC Occupancy: {occupancy}")
        else:
            self.lbl_room_occupancy.setText("IFC Occupancy: N/A")
        
        # Load annotations if they exist
        ann = None
        if self.current_model.annotations and space_id in self.current_model.annotations.space_annotations:
            ann = self.current_model.annotations.space_annotations[space_id]
        
        # Category override
        current_cat = ann.category_override if ann else None
        idx = self.combo_category_override.findData(current_cat)
        self.combo_category_override.setCurrentIndex(idx if idx >= 0 else 0)
        
        # Safety checkboxes
        self.chk_mark_exit.setChecked(ann.is_exit if ann else False)
        self.chk_start_fire.setChecked(ann.is_fire_origin if ann else False)
        self.chk_block_path.setChecked(ann.block_path if ann else False)
        
        # Agent count
        agent_count = ann.agent_count if ann else (occupancy or 0)
        self.spin_room_agents.setValue(agent_count)
        
        # Path preview
        self._update_path_preview(space_id)
        
        # Update virtual exit list
        self._update_virtual_exit_list()
        
        # Update virtual space list
        self._update_virtual_space_list()

    def _on_apply_room_config(self):
        """Apply all room configurator settings and recompute."""
        if not self.currently_selected_space_id or not self.current_model:
            return
        
        space_id = self.currently_selected_space_id
        
        # Ensure annotations exist
        if not self.current_model.annotations:
            from core.model_annotations import ModelAnnotations
            self.current_model.annotations = ModelAnnotations()
        
        ann = self.current_model.annotations
        
        # Save annotation values
        cat_override = self.combo_category_override.currentData()
        ann.set_category_override(space_id, cat_override)
        ann.set_is_exit(space_id, self.chk_mark_exit.isChecked())
        ann.set_is_fire_origin(space_id, self.chk_start_fire.isChecked())
        ann.set_block_path(space_id, self.chk_block_path.isChecked())
        ann.set_agent_count(space_id, self.spin_room_agents.value())
        
        self.log_message(f"Applied annotations to space {space_id}")
        
        # Recompute spatial graph immediately
        self._on_recompute_graph_clicked()
        
        # Update simulation if one exists and is not running
        if self.current_simulation and not self.is_simulation_running:
            self.log_message("Re-initializing simulation with updated annotations...")
            try:
                scenario = self.current_simulation.scenario
                
                # Update custom room agents from all annotations
                scenario.custom_room_agents.clear()
                for sid, a in ann.space_annotations.items():
                    if a.agent_count > 0:
                        scenario.custom_room_agents[sid] = a.agent_count
                
                # Rebuild fire events from annotations
                scenario.events = [e for e in scenario.events if e.get("type") != "fire"]
                for sid, a in ann.space_annotations.items():
                    if a.is_fire_origin and sid in self.current_model.spaces:
                        space = self.current_model.spaces[sid]
                        if space.center:
                            scenario.events.append({
                                "time": 5,
                                "type": "fire",
                                "location": list(space.center),
                                "spread_rate": 0.5,
                                "hazard_intensity": 1.0,
                                "smoke_level": 0.8
                            })
                
                sim_model = self.simulation_engine.initialize_simulation(
                    self.current_model, self.spatial_engine, scenario
                )
                self.current_simulation = sim_model
                self.visualization.load_simulation(sim_model)
                self.log_message("Simulation re-initialized with new annotations", "SUCCESS")
            except Exception as e:
                self.log_message(f"Simulation re-init failed: {e}", "ERROR")

    def _on_category_override_changed(self, index):
        """Handle category override dropdown change."""
        if not self.currently_selected_space_id or not self.current_model:
            return
        data = self.combo_category_override.itemData(index)
        if self.current_model.annotations:
            self.current_model.annotations.set_category_override(self.currently_selected_space_id, data)

    def _update_path_preview(self, space_id: str):
        """Update the path preview label for the selected space."""
        if not self.spatial_engine:
            self.lbl_path_preview.setText("Path: —")
            return
        path = self.spatial_engine.get_evacuation_path(space_id)
        if path and len(path) > 1:
            names = []
            for sid in path:
                if sid in self.current_model.spaces:
                    names.append(self.current_model.spaces[sid].name)
                else:
                    names.append(sid)
            self.lbl_path_preview.setText("Path:\n" + " → ".join(names))
        elif path and len(path) == 1:
            self.lbl_path_preview.setText("Path: This space is the exit")
        else:
            self.lbl_path_preview.setText("Path: No route to exit found")

    def _update_virtual_exit_list(self):
        """Update the virtual exit list label."""
        if not self.current_model or not self.current_model.annotations:
            self.lbl_virtual_exits.setText("No virtual exits added")
            return
        exits = self.current_model.annotations.exits
        if not exits:
            self.lbl_virtual_exits.setText("No virtual exits added")
            return
        lines = [f"• {e.name} at ({e.position[0]:.1f}, {e.position[1]:.1f}, {e.position[2]:.1f})" for e in exits]
        self.lbl_virtual_exits.setText("\n".join(lines))

    def _update_virtual_space_list(self):
        """Update the virtual space list label."""
        if not self.current_model or not self.current_model.annotations:
            self.lbl_virtual_spaces.setText("No virtual spaces added")
            return
        spaces = self.current_model.annotations.virtual_spaces
        if not spaces:
            self.lbl_virtual_spaces.setText("No virtual spaces added")
            return
        lines = [f"• {s.name} at ({s.position[0]:.1f}, {s.position[1]:.1f}, {s.position[2]:.1f})" for s in spaces]
        self.lbl_virtual_spaces.setText("\n".join(lines))

    def _on_show_path_clicked(self):
        """Show the evacuation path for the currently selected room."""
        if not self.currently_selected_space_id:
            return
        self.visualization.highlight_evacuation_path(self.currently_selected_space_id)
        self.log_message(f"Highlighted evacuation path for {self.currently_selected_space_id}")

    def _on_show_all_paths_clicked(self):
        """Show all evacuation paths."""
        self.visualization.set_show_evacuation_paths(True)
        self.log_message("Showing all evacuation paths")

    def _on_hide_paths_clicked(self):
        """Hide all evacuation paths."""
        self.visualization.set_show_evacuation_paths(False)
        self.log_message("Hiding all evacuation paths")

    def _on_add_virtual_exit(self):
        """Add a virtual exit at the selected space's center."""
        if not self.currently_selected_space_id or not self.current_model:
            return
        space = self.current_model.spaces[self.currently_selected_space_id]
        if not space.center:
            QMessageBox.warning(self, "Warning", "Selected space has no center position")
            return
        
        name, ok = QInputDialog.getText(self, "Virtual Exit", "Enter exit name:", text=f"Exit - {space.name}")
        if not ok or not name:
            return
        
        if self.current_model.annotations:
            self.current_model.annotations.add_virtual_exit(
                name=name,
                position=space.center,
                level_id=space.level,
                width=1.2
            )
            self.log_message(f"Added virtual exit '{name}' at {space.name}")
            self._update_virtual_exit_list()

    def _on_add_virtual_space(self):
        """Add a virtual space at the camera focal point, extracting boundaries via raycasting."""
        if not self.current_model or not self.visualization.plotter:
            QMessageBox.warning(self, "Warning", "No model loaded or 3D view not ready")
            return
            
        # Get camera focal point
        focal_point = self.visualization.plotter.camera.GetFocalPoint()
        cx, cy, cz = focal_point
        
        name, ok = QInputDialog.getText(self, "Virtual Space", "Enter space name:", text=f"Virtual Room {len(self.current_model.annotations.virtual_spaces) + 1 if self.current_model.annotations else 1}")
        if not ok or not name:
            return
            
        from core.model_annotations import VALID_SPACE_CATEGORIES
        category, ok = QInputDialog.getItem(self, "Virtual Space Category", "Select category:", VALID_SPACE_CATEGORIES, 0, False)
        if not ok or not category:
            return
            
        # Raycasting to find boundary
        self.statusbar.showMessage("Extracting room boundary using raycasting...")
        import math
        from shapely.geometry import box, Point, LineString, Polygon
        from shapely.ops import unary_union
        
        obstacles = []
        for elem in self.current_model.elements.values():
            if elem.category in [ElementCategory.WALL, ElementCategory.WINDOW, ElementCategory.DOOR, ElementCategory.COLUMN]:
                if elem.bounds:
                    (min_x, min_y, min_z), (max_x, max_y, max_z) = elem.bounds
                    # Check if it intersects the focal point Z loosely
                    if min_z - 2.0 <= cz <= max_z + 2.0:
                        obstacles.append(box(min_x, min_y, max_x, max_y))
                        
        boundary_coords = None
        radius = 3.0
        computed_area = radius * radius * 3.14159
        
        if obstacles:
            all_walls = unary_union(obstacles)
            center = Point(cx, cy)
            points = []
            for angle in range(0, 360, 5):
                rad = math.radians(angle)
                ray = LineString([center, Point(cx + math.cos(rad)*50, cy + math.sin(rad)*50)])
                inter = ray.intersection(all_walls)
                if not inter.is_empty:
                    closest = None
                    min_dist = float('inf')
                    if inter.geom_type == 'Point': closest = inter
                    elif inter.geom_type == 'MultiPoint': closest = min(inter.geoms, key=lambda p: center.distance(p))
                    elif inter.geom_type == 'LineString': closest = Point(inter.coords[0])
                    elif inter.geom_type == 'MultiLineString': closest = min([Point(ls.coords[0]) for ls in inter.geoms], key=lambda p: center.distance(p))
                    elif inter.geom_type == 'GeometryCollection':
                        for geom in inter.geoms:
                            pt = None
                            if geom.geom_type == 'Point': pt = geom
                            elif geom.geom_type == 'LineString': pt = Point(geom.coords[0])
                            if pt and center.distance(pt) < min_dist:
                                min_dist = center.distance(pt)
                                closest = pt
                    
                    if closest: points.append(closest)
                else:
                    # No intersection, just use fixed distance
                    points.append(Point(cx + math.cos(rad)*10, cy + math.sin(rad)*10))
                    
            if len(points) >= 3:
                poly = Polygon(points)
                # Simplify polygon to remove tiny jagged edges
                poly = poly.simplify(0.1)
                if not poly.is_empty and poly.area > 1.0:
                    boundary_coords = list(poly.exterior.coords)
                    # Update focal point and area based on true room polygon
                    focal_point = (poly.centroid.x, poly.centroid.y, cz)
                    computed_area = poly.area
                    radius = math.sqrt(poly.area / math.pi)

        self.statusbar.clearMessage()
            
        if not self.current_model.annotations:
            from core.model_annotations import ModelAnnotations
            self.current_model.annotations = ModelAnnotations()
            
        # Add virtual space
        v_space = self.current_model.annotations.add_virtual_space(
            name=name,
            position=focal_point,
            level_id=self.visualization.settings.level_filter or "",
            radius=radius,
            category=category,
            boundary=boundary_coords
        )
        
        self.log_message(f"Added virtual space '{name}' at {focal_point} with area {computed_area:.1f}sqm")
        self._update_virtual_space_list()
        
        # Inject into model immediately so it renders and gets processed
        from core.bim_processor import BIMSpace
        synthetic_space = BIMSpace(
            id=v_space.id,
            global_id=v_space.id,
            name=v_space.name,
            long_name=v_space.name,
            level=v_space.level_id,
            area=computed_area,
            volume=0.0,
            bounds=None,
            center=v_space.position,
            geometry={"boundary": boundary_coords} if boundary_coords else None,
            category=v_space.category,
            occupancy_capacity=max(1, int(computed_area / 10))
        )
        self.current_model.spaces[v_space.id] = synthetic_space
        
        # Add a default annotation so it can be edited like a normal space
        ann = self.current_model.annotations.get_or_create_space(v_space.id)
        ann.category_override = v_space.category
        
        # Refresh visualization
        self.visualization.refresh()

    def _on_recompute_graph_clicked(self):
        """Recompute the spatial graph with current annotations."""
        if not self.current_model:
            QMessageBox.warning(self, "Warning", "No model loaded")
            return
        
        self.log_message("Recomputing spatial graph with annotations...")
        self.statusbar.showMessage("Recomputing spatial graph...")
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        
        try:
            self.progress_bar.setValue(30)
            annotations = self.current_model.annotations
            spatial_graph = self.spatial_engine.process_model(self.current_model, annotations)
            self.progress_bar.setValue(80)
            
            # Update visualization with new paths
            self.visualization.spatial_graph = spatial_graph
            self.visualization.display_exits()
            if self.visualization.settings.show_evacuation_paths:
                self.visualization.display_evacuation_paths()
            
            self.progress_bar.setValue(100)
            self.log_message("Spatial graph recomputed successfully", "SUCCESS")
            
            # Update path preview if a room is selected
            if self.currently_selected_space_id:
                self._update_path_preview(self.currently_selected_space_id)
            
        except Exception as e:
            self.log_message(f"Error recomputing spatial graph: {e}", "ERROR")
            QMessageBox.critical(self, "Error", f"Recompute failed:\n{str(e)}")
        finally:
            self.progress_bar.setVisible(False)
            self.statusbar.showMessage("Ready")

    def _create_properties_panel(self):
        """Create the properties panel."""
        dock = QDockWidget("Properties", self)
        dock.setAllowedAreas(Qt.LeftDockWidgetArea | Qt.RightDockWidgetArea)
        
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # Element properties
        self.properties_table = QTableWidget()
        self.properties_table.setColumnCount(2)
        self.properties_table.setHorizontalHeaderLabels(["Property", "Value"])
        self.properties_table.horizontalHeader().setStretchLastSection(True)
        self.properties_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeToContents)
        layout.addWidget(self.properties_table)
        
        dock.setWidget(widget)
        self.addDockWidget(Qt.LeftDockWidgetArea, dock)
        self.panels["Properties"] = dock
        
    def _create_simulation_panel(self):
        """Create the simulation control panel."""
        dock = QDockWidget("Simulation", self)
        dock.setAllowedAreas(Qt.LeftDockWidgetArea | Qt.RightDockWidgetArea)
        
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # Scenario selector
        scenario_group = QGroupBox("Scenario")
        scenario_layout = QVBoxLayout(scenario_group)
        
        self.scenario_combo = QComboBox()
        self.scenario_combo.addItem("Select a scenario...")
        scenario_layout.addWidget(self.scenario_combo)
        
        self.scenario_description = QTextEdit()
        self.scenario_description.setReadOnly(True)
        self.scenario_description.setMaximumHeight(80)
        scenario_layout.addWidget(self.scenario_description)
        
        layout.addWidget(scenario_group)
        
        # Agent Library
        agent_group = QGroupBox("Agent Library")
        agent_layout = QVBoxLayout(agent_group)
        
        self.agent_tree = QTreeWidget()
        self.agent_tree.setHeaderLabel("Agents")
        agent_layout.addWidget(self.agent_tree)
        
        btn_add_agent = QPushButton("Add Agent Type")
        btn_add_agent.clicked.connect(self._on_add_agent_type)
        agent_layout.addWidget(btn_add_agent)
        
        layout.addWidget(agent_group)
        
        # Simulation Settings
        settings_group = QGroupBox("Settings")
        settings_layout = QVBoxLayout(settings_group)
        
        # Duration
        duration_layout = QHBoxLayout()
        duration_layout.addWidget(QLabel("Duration (s):"))
        self.duration_spin = QSpinBox()
        self.duration_spin.setRange(60, 86400)
        self.duration_spin.setValue(3600)
        self.duration_spin.setSingleStep(300)
        duration_layout.addWidget(self.duration_spin)
        settings_layout.addLayout(duration_layout)
        
        # Time step
        timestep_layout = QHBoxLayout()
        timestep_layout.addWidget(QLabel("Time Step (s):"))
        self.timestep_spin = QDoubleSpinBox()
        self.timestep_spin.setRange(0.1, 10.0)
        self.timestep_spin.setValue(1.0)
        self.timestep_spin.setSingleStep(0.1)
        timestep_layout.addWidget(self.timestep_spin)
        settings_layout.addLayout(timestep_layout)
        
        # Random seed
        seed_layout = QHBoxLayout()
        seed_layout.addWidget(QLabel("Random Seed:"))
        self.seed_spin = QSpinBox()
        self.seed_spin.setRange(0, 999999)
        self.seed_spin.setValue(42)
        seed_layout.addWidget(self.seed_spin)
        settings_layout.addLayout(seed_layout)
        
        # Show trails toggle
        self.chk_show_trails = QCheckBox("Show Agent Trails")
        self.chk_show_trails.setChecked(False)
        self.chk_show_trails.setToolTip("Draw path trace lines behind each agent")
        self.chk_show_trails.stateChanged.connect(self._on_show_trails_changed)
        settings_layout.addWidget(self.chk_show_trails)
        
        # Show spatial graph toggle
        self.chk_show_spatial_graph = QCheckBox("Show Spatial Graph")
        self.chk_show_spatial_graph.setChecked(True)
        self.chk_show_spatial_graph.setToolTip("Display the generated navigation graph nodes and connections")
        self.chk_show_spatial_graph.stateChanged.connect(self._on_show_spatial_graph_changed)
        settings_layout.addWidget(self.chk_show_spatial_graph)
        
        layout.addWidget(settings_group)
        
        # Metrics display
        metrics_group = QGroupBox("Live Metrics")
        metrics_layout = QVBoxLayout(metrics_group)
        
        self.metrics_labels = {}
        for metric_name in ["Active Agents", "Moving", "Waiting", "Avg Speed", "Avg Distance", "Congestion Events"]:
            lbl = QLabel(f"{metric_name}: 0")
            metrics_layout.addWidget(lbl)
            self.metrics_labels[metric_name] = lbl
            
        layout.addWidget(metrics_group)
        
        layout.addStretch()
        dock.setWidget(widget)
        self.addDockWidget(Qt.RightDockWidgetArea, dock)
        self.panels["Simulation"] = dock
        
    def _create_analytics_panel(self):
        """Create the analytics dashboard panel."""
        dock = QDockWidget("Analytics", self)
        dock.setAllowedAreas(Qt.LeftDockWidgetArea | Qt.RightDockWidgetArea)
        
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # Analytics tabs
        tabs = QTabWidget()
        
        # Movement tab
        movement_tab = QWidget()
        movement_layout = QVBoxLayout(movement_tab)
        
        self.movement_table = QTableWidget()
        self.movement_table.setColumnCount(4)
        self.movement_table.setHorizontalHeaderLabels(["Agent", "Distance (m)", "Time (s)", "Avg Speed (m/s)"])
        self.movement_table.horizontalHeader().setStretchLastSection(True)
        movement_layout.addWidget(self.movement_table)
        
        tabs.addTab(movement_tab, "Movement")
        
        # Occupancy tab
        occupancy_tab = QWidget()
        occupancy_layout = QVBoxLayout(occupancy_tab)
        
        self.occupancy_table = QTableWidget()
        self.occupancy_table.setColumnCount(5)
        self.occupancy_table.setHorizontalHeaderLabels(["Space", "Agents", "Density", "Capacity %", "Status"])
        self.occupancy_table.horizontalHeader().setStretchLastSection(True)
        occupancy_layout.addWidget(self.occupancy_table)
        
        tabs.addTab(occupancy_tab, "Occupancy")
        
        # Accessibility tab
        accessibility_tab = QWidget()
        accessibility_layout = QVBoxLayout(accessibility_tab)
        
        self.accessibility_text = QTextEdit()
        self.accessibility_text.setReadOnly(True)
        accessibility_layout.addWidget(self.accessibility_text)
        
        tabs.addTab(accessibility_tab, "Accessibility")
        
        # Social tab
        social_tab = QWidget()
        social_layout = QVBoxLayout(social_tab)
        
        self.social_text = QTextEdit()
        self.social_text.setReadOnly(True)
        social_layout.addWidget(self.social_text)
        
        tabs.addTab(social_tab, "Social Analysis")
        
        layout.addWidget(tabs)
        
        # Export buttons
        btn_layout = QHBoxLayout()
        
        btn_export_csv = QPushButton("Export CSV")
        btn_export_csv.clicked.connect(self._on_export_csv)
        btn_layout.addWidget(btn_export_csv)
        
        btn_export_report = QPushButton("Generate Report")
        btn_export_report.clicked.connect(self._on_generate_report)
        btn_layout.addWidget(btn_export_report)
        
        layout.addLayout(btn_layout)
        
        dock.setWidget(widget)
        self.addDockWidget(Qt.RightDockWidgetArea, dock)
        self.panels["Analytics"] = dock
        
    def _create_console_panel(self):
        """Create the console/output panel."""
        dock = QDockWidget("Console", self)
        dock.setAllowedAreas(Qt.TopDockWidgetArea | Qt.BottomDockWidgetArea)
        
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        self.console_output = QTextEdit()
        self.console_output.setReadOnly(True)
        self.console_output.setStyleSheet("""
            QTextEdit {
                background-color: #1e1e1e;
                color: #d4d4d4;
                font-family: Consolas, monospace;
                font-size: 11px;
            }
        """)
        layout.addWidget(self.console_output)
        
        dock.setWidget(widget)
        self.addDockWidget(Qt.BottomDockWidgetArea, dock)
        self.panels["Console"] = dock
        
    def _toggle_panel(self, name: str, visible: bool):
        """Toggle panel visibility."""
        if name in self.panels:
            self.panels[name].setVisible(visible)
            
    def log_message(self, message: str, level: str = "INFO"):
        """Log a message to the console."""
        import datetime
        timestamp = datetime.datetime.now().strftime("%H:%M:%S")
        
        color_map = {
            "INFO": "#d4d4d4",
            "WARNING": "#ffcc00",
            "ERROR": "#ff4444",
            "SUCCESS": "#44ff44"
        }
        color = color_map.get(level, "#d4d4d4")
        
        self.console_output.append(
            f'<span style="color: #858585;">[{timestamp}]</span> '
            f'<span style="color: {color};">{level}: {message}</span>'
        )
        
    # Event Handlers
    
    def _populate_floor_combo(self, model: BIMModel):
        """Populate floor selector."""
        self.floor_combo.blockSignals(True)
        self.floor_combo.clear()
        self.floor_combo.addItem("All Floors", None)
        for level_id, level in sorted(model.levels.items(), key=lambda x: x[1].elevation):
            self.floor_combo.addItem(level.name, level_id)
        self.floor_combo.blockSignals(False)

    def _on_floor_changed(self, index):
        """Handle floor selection."""
        if not self.current_model:
            return
        level_id = self.floor_combo.currentData()
        self.visualization.settings.level_filter = level_id
        if level_id:
            self.visualization.set_view("top")
            self.view_combo.setCurrentText("Top")
        self.visualization.refresh()

    def _on_wall_opacity_changed(self, value):
        """Handle wall opacity slider change."""
        self.visualization.settings.wall_opacity = value / 100.0
        self.visualization.refresh()

    def _on_show_trails_changed(self, state):
        """Handle show trails checkbox toggle."""
        show = state == Qt.Checked.value if hasattr(Qt.Checked, 'value') else bool(state)
        self.visualization.settings.show_trails = show
        if not show:
            # Clear existing trail actors
            for actor in self.visualization.trail_actors.values():
                try:
                    self.visualization.plotter.remove_actor(actor)
                except Exception:
                    pass
            self.visualization.trail_actors.clear()
            if self.visualization.plotter:
                try:
                    self.visualization.plotter.render()
                except Exception:
                    pass
                    
    def _on_show_spatial_graph_changed(self, state):
        """Handle show spatial graph checkbox toggle."""
        show = state == Qt.Checked.value if hasattr(Qt.Checked, 'value') else bool(state)
        self.visualization.settings.show_spatial_graph = show
        self.visualization.refresh()
    
    def _on_open_ifc(self):
        """Handle open IFC file action."""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Open IFC File",
            "",
            "IFC Files (*.ifc *.IFC);;All Files (*.*)"
        )
        
        if file_path:
            self._load_ifc_file(file_path)
            
    def _load_ifc_file(self, file_path: str):
        """Load an IFC file and update the UI."""
        self.log_message(f"Loading IFC file: {file_path}")
        self.statusbar.showMessage(f"Loading {os.path.basename(file_path)}...")
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(10)
        
        try:
            # Process IFC
            def update_progress(msg, val):
                self.statusbar.showMessage(msg)
                self.progress_bar.setValue(val)
                QApplication.processEvents()
                
            model = self.bim_processor.load_ifc(file_path, progress_callback=update_progress)
            self.current_model = model
            
            self.progress_bar.setValue(60)
            
            # Process spatial intelligence
            self.log_message("Processing spatial intelligence...")
            annotations = self.current_model.annotations
            spatial_graph = self.spatial_engine.process_model(self.current_model, annotations)
            
            self.progress_bar.setValue(80)
            
            # Update visualization with paths
            self.visualization.spatial_graph = spatial_graph
            if self.visualization.settings.show_evacuation_paths:
                self.visualization.display_evacuation_paths()
            self.visualization.display_exits()
            
            # Update UI
            self._update_project_tree(model)
            self._update_model_info(model)
            self._populate_floor_combo(model)
            
            # Initialize 3D visualization
            self._initialize_3d_view(model)
            
            self.progress_bar.setValue(100)
            self.log_message(f"Model loaded: {model.name}", "SUCCESS")
            self.statusbar.showMessage(f"Loaded: {model.name}")
            self.model_info_label.setText(f"Model: {os.path.basename(file_path)}")
            
            # Emit signal
            self.model_loaded.emit(model)
            
        except Exception as e:
            self.log_message(f"Error loading IFC: {str(e)}", "ERROR")
            QMessageBox.critical(self, "Error", f"Failed to load IFC file:\n{str(e)}")
            
        finally:
            self.progress_bar.setVisible(False)
            
    def _update_project_tree(self, model: BIMModel):
        """Update the project tree widget."""
        self.ifc_tree.clear()
        
        # Root
        root = QTreeWidgetItem(self.ifc_tree)
        root.setText(0, model.name)
        root.setExpanded(True)
        
        # Levels
        levels_item = QTreeWidgetItem(root)
        levels_item.setText(0, f"Levels ({len(model.levels)})")
        
        for level_id, level in sorted(model.levels.items(), key=lambda x: x[1].elevation):
            level_item = QTreeWidgetItem(levels_item)
            level_item.setText(0, f"{level.name} (Elevation: {level.elevation:.2f}m)")
            level_item.setData(0, Qt.UserRole, ("level", level_id))
            
            # Spaces on this level
            level_spaces = [s for s in model.spaces.values() if s.level == level_id]
            if level_spaces:
                spaces_item = QTreeWidgetItem(level_item)
                spaces_item.setText(0, f"Spaces ({len(level_spaces)})")
                
                for space in sorted(level_spaces, key=lambda s: s.name):
                    space_item = QTreeWidgetItem(spaces_item)
                    space_item.setText(0, f"{space.name} ({space.category})")
                    space_item.setData(0, Qt.UserRole, ("space", space.id))
                    
        # Elements by category
        from collections import Counter
        categories = Counter(e.category.value for e in model.elements.values())
        
        elements_item = QTreeWidgetItem(root)
        elements_item.setText(0, f"Elements ({len(model.elements)})")
        
        for cat_name, count in sorted(categories.items()):
            cat_item = QTreeWidgetItem(elements_item)
            cat_item.setText(0, f"{cat_name} ({count})")
            
            # Add elements
            for elem in sorted(
                [e for e in model.elements.values() if e.category.value == cat_name],
                key=lambda e: e.name
            )[:50]:  # Limit to first 50 per category
                elem_item = QTreeWidgetItem(cat_item)
                elem_item.setText(0, elem.name)
                elem_item.setData(0, Qt.UserRole, ("element", elem.id))
                
    def _update_model_info(self, model: BIMModel):
        """Update model information display."""
        summary = self.bim_processor.export_summary(model)
        self.model_summary.setText(summary)
        
    def _initialize_3d_view(self, model: BIMModel):
        """Initialize the 3D visualization view."""
        # Remove placeholder
        if self.viz_placeholder:
            self.viz_placeholder.deleteLater()
            self.viz_placeholder = None
            
        # Create visualization plotter
        try:
            # create_plotter returns a QtInteractor which is itself a QWidget
            qt_widget = self.visualization.create_plotter(parent=self.viz_container, show=True)

            # Add the interactor widget to the container layout
            layout = self.viz_container.layout()
            layout.addWidget(qt_widget)

            # Load the model
            self.visualization.load_bim_model(model)

            self.log_message("3D visualization initialized")

        except Exception as e:
            self.log_message(f"Error initializing 3D view: {e}", "ERROR")
            # Fallback to placeholder
            self.viz_placeholder = QLabel(f"3D View Error: {e}\nPyVista may not be available")
            self.viz_placeholder.setAlignment(Qt.AlignCenter)
            layout = self.viz_container.layout()
            layout.addWidget(self.viz_placeholder)
            
    def _on_tree_item_clicked(self, item: QTreeWidgetItem, column: int):
        """Handle tree item click."""
        data = item.data(0, Qt.UserRole)
        if not data:
            return
            
        item_type, item_id = data
        
        if item_type == "space" and self.current_model:
            space = self.current_model.spaces.get(item_id)
            if space:
                self._show_space_properties(space)
                self.visualization.focus_on_space(item_id)
                
        elif item_type == "element" and self.current_model:
            element = self.current_model.elements.get(item_id)
            if element:
                self._show_element_properties(element)
                
        elif item_type == "level" and self.current_model:
            level = self.current_model.levels.get(item_id)
            if level:
                self._show_level_properties(level)
                
    def _show_space_properties(self, space: BIMSpace):
        """Show space properties in the properties panel."""
        self.properties_table.setRowCount(0)
        
        props = [
            ("ID", space.id),
            ("Global ID", space.global_id),
            ("Name", space.name),
            ("Long Name", space.long_name),
            ("Category", space.category),
            ("Level", space.level),
            ("Area (m²)", f"{space.area:.2f}"),
            ("Volume (m³)", f"{space.volume:.2f}"),
            ("Center", str(space.center)),
        ]
        
        for key, value in props:
            row = self.properties_table.rowCount()
            self.properties_table.insertRow(row)
            self.properties_table.setItem(row, 0, QTableWidgetItem(key))
            self.properties_table.setItem(row, 1, QTableWidgetItem(str(value)))
            
    def _show_element_properties(self, element: BIMElement):
        """Show element properties in the properties panel."""
        self.properties_table.setRowCount(0)
        
        props = [
            ("ID", element.id),
            ("Global ID", element.global_id),
            ("Name", element.name),
            ("Type", element.element_type),
            ("Category", element.category.value),
            ("Level", element.level),
            ("Area (m²)", f"{element.area:.2f}"),
            ("Volume (m³)", f"{element.volume:.2f}"),
            ("Center", str(element.center)),
        ]
        
        for key, value in props:
            row = self.properties_table.rowCount()
            self.properties_table.insertRow(row)
            self.properties_table.setItem(row, 0, QTableWidgetItem(key))
            self.properties_table.setItem(row, 1, QTableWidgetItem(str(value)))
            
        # Add custom properties
        for prop_name, prop_value in element.properties.items():
            row = self.properties_table.rowCount()
            self.properties_table.insertRow(row)
            self.properties_table.setItem(row, 0, QTableWidgetItem(prop_name))
            self.properties_table.setItem(row, 1, QTableWidgetItem(str(prop_value)))
            
    def _show_level_properties(self, level):
        """Show level properties."""
        self.properties_table.setRowCount(0)
        
        props = [
            ("ID", level.id),
            ("Name", level.name),
            ("Elevation", f"{level.elevation:.2f}m"),
            ("Height", f"{level.height:.2f}m"),
            ("Elements", str(len(level.elements))),
        ]
        
        for key, value in props:
            row = self.properties_table.rowCount()
            self.properties_table.insertRow(row)
            self.properties_table.setItem(row, 0, QTableWidgetItem(key))
            self.properties_table.setItem(row, 1, QTableWidgetItem(str(value)))
            
    def _set_visualization_mode(self, mode: str):
        """Set the visualization mode."""
        mode_map = {
            "BIM Only": VisualizationMode.BIM_ONLY,
            "Agents": VisualizationMode.AGENTS,
            "Density": VisualizationMode.DENSITY,
            "Heat Map": VisualizationMode.HEAT_MAP,
            "Evacuation": VisualizationMode.EVACUATION,
            "Navigation": VisualizationMode.NAVIGATION
        }
        
        if mode in mode_map:
            self.visualization.set_visualization_mode(mode_map[mode])
            self.log_message(f"Visualization mode: {mode}")
            
        # Update radio buttons
        for m, action in self.viz_mode_group.items():
            action.setChecked(m == mode)
            
    def _on_view_changed(self, view: str):
        """Handle view selection change."""
        view_map = {
            "Perspective": "isometric",
            "Top": "top",
            "Front": "front",
            "Side": "side",
            "Isometric": "isometric"
        }
        
        if view in view_map:
            self.visualization.set_view(view_map[view])
            
    def _on_validate_model(self):
        """Handle model validation."""
        if not self.current_model:
            QMessageBox.warning(self, "Warning", "No model loaded")
            return
            
        report = self.current_model.validation_report
        
        msg = f"""Model Validation Report:

File Valid: {report['file_valid']}
Total Elements: {report['stats'].get('total_elements', 0)}
Total Spaces: {report['stats'].get('total_spaces', 0)}
Total Levels: {report['stats'].get('total_levels', 0)}
Unique GlobalIds: {report['stats'].get('unique_global_ids', 0)}
Elements with Geometry: {report['stats'].get('elements_with_geometry', 0)}
Elements without Geometry: {report['stats'].get('elements_without_geometry', 0)}

Warnings: {len(report['warnings'])}
Errors: {len(report['errors'])}
"""
        
        if report['warnings']:
            msg += "\nWarnings:\n" + "\n".join(f"- {w}" for w in report['warnings'][:10])
            
        if report['errors']:
            msg += "\nErrors:\n" + "\n".join(f"- {e}" for e in report['errors'][:10])
            
        QMessageBox.information(self, "Model Validation", msg)
        
    def _on_process_spatial(self):
        """Handle spatial intelligence processing."""
        if not self.current_model:
            QMessageBox.warning(self, "Warning", "No model loaded")
            return
            
        self.log_message("Processing spatial intelligence...")
        self.statusbar.showMessage("Processing spatial intelligence...")
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        
        try:
            self.progress_bar.setValue(50)
            annotations = self.current_model.annotations
            spatial_graph = self.spatial_engine.process_model(self.current_model, annotations)
            self.progress_bar.setValue(100)
            
            # Update accessibility info
            accessibility = self.spatial_engine.analyze_accessibility()
            accessibility_text = f"""Accessibility Analysis:

Total Spaces: {accessibility.get('total_spaces', 0)}
Total Connections: {accessibility.get('total_connections', 0)}
Average Connections per Space: {accessibility.get('avg_connections_per_space', 0):.2f}

Connection Types:
"""
            for conn_type, count in accessibility.get('connection_types', {}).items():
                accessibility_text += f"  {conn_type}: {count}\n"
                
            if accessibility.get('isolated_spaces'):
                accessibility_text += f"\nIsolated Spaces: {len(accessibility['isolated_spaces'])}\n"
                
            if accessibility.get('accessible_issues'):
                accessibility_text += f"\nAccessibility Issues: {len(accessibility['accessible_issues'])}\n"
                
            self.accessibility_text.setText(accessibility_text)
            
            self.log_message("Spatial intelligence processed", "SUCCESS")
            
        except Exception as e:
            self.log_message(f"Error processing spatial intelligence: {e}", "ERROR")
            QMessageBox.critical(self, "Error", f"Spatial processing failed:\n{str(e)}")
            
        finally:
            self.progress_bar.setVisible(False)
            self.statusbar.showMessage("Ready")
            
    def _on_model_info(self):
        """Show model information dialog."""
        if not self.current_model:
            QMessageBox.warning(self, "Warning", "No model loaded")
            return
            
        summary = self.bim_processor.export_summary(self.current_model)
        
        dialog = QMessageBox(self)
        dialog.setWindowTitle("Model Information")
        dialog.setText(summary[:1000] + "..." if len(summary) > 1000 else summary)
        dialog.setDetailedText(summary)
        dialog.exec()
        
    def _on_new_scenario(self):
        """Create a new simulation scenario."""
        if not self.current_model:
            QMessageBox.warning(self, "Warning", "Please load a BIM model first")
            return
            
        self.log_message("Creating new scenario...")
        # TODO: Implement scenario builder dialog
        QMessageBox.information(self, "Info", "Scenario builder will be implemented in the next version.")
        
    def _load_preset(self, preset_name: str):
        """Load a simulation preset."""
        if not self.current_model:
            QMessageBox.warning(self, "Warning", "Please load a BIM model first")
            return
            
        self.log_message(f"Loading preset: {preset_name}")
        
        try:
            if preset_name == "office":
                scenario = ScenarioPresets.office_scenario()
            elif preset_name == "evacuation":
                scenario = ScenarioPresets.evacuation_scenario()
            elif preset_name == "hospital":
                scenario = ScenarioPresets.hospital_scenario()
            elif preset_name == "university":
                scenario = ScenarioPresets.university_scenario()
            elif preset_name == "fire_evacuation":
                scenario = ScenarioPresets.fire_evacuation_scenario()
            else:
                return
                
            # Clear previous custom room settings on new preset
            self.custom_room_settings.clear()
            if self.currently_selected_space_id:
                self._on_space_selected(self.currently_selected_space_id)
                
            # Initialize simulation
            sim_model = self.simulation_engine.initialize_simulation(
                self.current_model,
                self.spatial_engine,
                scenario
            )
            
            self.current_simulation = sim_model
            
            # Update UI
            self.scenario_combo.addItem(scenario.name, scenario.id)
            self.scenario_combo.setCurrentText(scenario.name)
            self.scenario_description.setText(scenario.description)
            
            # Update agent tree
            self._update_agent_tree(scenario)
            
            # Load into visualization
            self.visualization.load_simulation(sim_model)
            
            self.log_message(f"Preset loaded: {scenario.name}", "SUCCESS")
            
        except Exception as e:
            self.log_message(f"Error loading preset: {e}", "ERROR")
            QMessageBox.critical(self, "Error", f"Failed to load preset:\n{str(e)}")
            
    def _update_agent_tree(self, scenario):
        """Update the agent tree in the simulation panel."""
        self.agent_tree.clear()
        
        for profile in scenario.agent_profiles:
            count = scenario.agent_counts.get(profile.id, 1)
            item = QTreeWidgetItem(self.agent_tree)
            item.setText(0, f"{profile.name} ({count})")
            item.setData(0, Qt.UserRole, profile.id)
            
    def _on_add_agent_type(self):
        """Add a new agent type to the scenario."""
        # TODO: Implement agent type dialog
        pass
        
    def _on_start_simulation(self):
        """Start the simulation."""
        if not self.current_simulation:
            QMessageBox.warning(self, "Warning", "No simulation loaded")
            return
            
        self.simulation_engine.start()
        self.is_simulation_running = True
        
        # Update UI
        self.btn_start.setEnabled(False)
        self.btn_pause.setEnabled(True)
        self.btn_stop.setEnabled(True)
        
        # Start timer
        speed = self.speed_slider.value()
        interval = max(10, 1000 // speed)
        self.simulation_timer.start(interval)
        
        # Start visualization
        self.visualization.start_simulation_visualization()
        
        self.simulation_started.emit()
        self.log_message("Simulation started")
        
    def _on_pause_simulation(self):
        """Pause the simulation."""
        self.simulation_engine.pause()
        self.is_simulation_running = False
        
        self.btn_start.setEnabled(True)
        self.btn_pause.setEnabled(False)
        
        self.simulation_timer.stop()
        self.visualization.stop_simulation_visualization()
        
        self.simulation_paused.emit()
        self.log_message("Simulation paused")
        
    def _on_stop_simulation(self):
        """Stop the simulation."""
        self.simulation_engine.stop()
        self.is_simulation_running = False
        
        self.btn_start.setEnabled(True)
        self.btn_pause.setEnabled(False)
        self.btn_stop.setEnabled(False)
        
        self.simulation_timer.stop()
        self.visualization.stop_simulation_visualization()
        
        self.simulation_stopped.emit()
        self.log_message("Simulation stopped")
        
    def _on_step_forward(self):
        """Step simulation forward by one step."""
        if not self.current_simulation:
            return
            
        self.simulation_engine.step()
        self._update_simulation_ui()
        
    def _simulation_step(self):
        """Timer callback for simulation steps."""
        if self.is_simulation_running and self.current_simulation:
            self.simulation_engine.step()
            self._update_simulation_ui()
            
    def _update_simulation_ui(self):
        """Update UI with current simulation state."""
        if not self.current_simulation:
            return
            
        metrics = self.current_simulation.get_current_metrics()
        
        # Update time
        hours = int(metrics.timestamp // 3600)
        minutes = int((metrics.timestamp % 3600) // 60)
        seconds = int(metrics.timestamp % 60)
        self.time_label.setText(f"Time: {hours:02d}:{minutes:02d}:{seconds:02d}")
        
        # Update metrics
        self.agent_count_label.setText(f"Agents: {metrics.agent_count}")
        self.metrics_labels["Active Agents"].setText(f"Active Agents: {metrics.agent_count}")
        self.metrics_labels["Moving"].setText(f"Moving: {metrics.agents_moving}")
        self.metrics_labels["Waiting"].setText(f"Waiting: {metrics.agents_waiting}")
        self.metrics_labels["Avg Speed"].setText(f"Avg Speed: {metrics.avg_speed:.2f} m/s")
        
        # Update occupancy table
        self._update_occupancy_table()
        
        # Update movement table
        self._update_movement_table()
        
        # Check if simulation complete
        if self.current_simulation.schedule.steps >= self.current_simulation.max_steps:
            self._on_stop_simulation()
            self.log_message("Simulation completed", "SUCCESS")
            
    def _update_occupancy_table(self):
        """Update the occupancy analytics table."""
        if not self.current_simulation:
            return
            
        occupancy = self.current_simulation.get_space_occupancy()
        
        self.occupancy_table.setRowCount(0)
        for space_id, data in sorted(occupancy.items(), key=lambda x: x[1]['agent_count'], reverse=True)[:20]:
            row = self.occupancy_table.rowCount()
            self.occupancy_table.insertRow(row)
            
            self.occupancy_table.setItem(row, 0, QTableWidgetItem(data['space_name']))
            self.occupancy_table.setItem(row, 1, QTableWidgetItem(str(data['agent_count'])))
            self.occupancy_table.setItem(row, 2, QTableWidgetItem(f"{data['density']:.3f}"))
            self.occupancy_table.setItem(row, 3, QTableWidgetItem(f"{data['capacity_ratio']:.1%}"))
            
            status = "Overcrowded" if data['is_overcrowded'] else "Normal"
            status_item = QTableWidgetItem(status)
            if data['is_overcrowded']:
                status_item.setBackground(QColor(255, 200, 200))
            self.occupancy_table.setItem(row, 4, status_item)
            
    def _update_movement_table(self):
        """Update the movement analytics table."""
        if not self.current_simulation:
            return
            
        agents = list(self.current_simulation.schedule.agents)[:20]
        
        self.movement_table.setRowCount(0)
        for agent in agents:
            row = self.movement_table.rowCount()
            self.movement_table.insertRow(row)
            
            self.movement_table.setItem(row, 0, QTableWidgetItem(str(agent.unique_id)))
            self.movement_table.setItem(row, 1, QTableWidgetItem(f"{agent.traveled_distance:.2f}"))
            self.movement_table.setItem(row, 2, QTableWidgetItem(f"{agent.travel_time:.2f}"))
            
            avg_speed = agent.traveled_distance / agent.travel_time if agent.travel_time > 0 else 0
            self.movement_table.setItem(row, 3, QTableWidgetItem(f"{avg_speed:.2f}"))
            
    def _on_reset_simulation(self):
        """Reset the simulation."""
        self.simulation_engine.reset()
        
        self.btn_start.setEnabled(True)
        self.btn_pause.setEnabled(False)
        self.btn_stop.setEnabled(False)
        
        self.simulation_timer.stop()
        self.visualization.stop_simulation_visualization()
        
        self.log_message("Simulation reset")
        
    def _on_save_project(self):
        """Save the current project."""
        # TODO: Implement project save
        QMessageBox.information(self, "Info", "Project save will be implemented in the next version.")
        
    def _on_export_results(self):
        """Export simulation results."""
        if not self.current_simulation:
            QMessageBox.warning(self, "Warning", "No simulation to export")
            return
            
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Export Results",
            "",
            "JSON Files (*.json);;CSV Files (*.csv);;All Files (*.*)"
        )
        
        if file_path:
            try:
                import json
                results = self.simulation_engine.get_results()
                
                with open(file_path, 'w') as f:
                    json.dump(results, f, indent=2, default=str)
                    
                self.log_message(f"Results exported: {file_path}", "SUCCESS")
                
            except Exception as e:
                self.log_message(f"Export error: {e}", "ERROR")
                QMessageBox.critical(self, "Error", f"Export failed:\n{str(e)}")
                
    def _on_export_csv(self):
        """Export analytics as CSV."""
        # TODO: Implement CSV export
        pass
        
    def _on_generate_report(self):
        """Generate simulation report."""
        if not self.current_simulation:
            QMessageBox.warning(self, "Warning", "No simulation to report on")
            return
            
        results = self.simulation_engine.get_results()
        
        report = f"""
Simulation Report
=================

Scenario: {results['scenario']['name']}
Description: {results['scenario']['description']}
Duration: {results['scenario']['duration']} steps

Final Metrics:
  Active Agents: {results['final_metrics']['agent_count']}
  Moving: {results['final_metrics']['agents_moving']}
  Waiting: {results['final_metrics']['agents_waiting']}
  Average Speed: {results['final_metrics']['avg_speed']:.2f} m/s
  
Results:
  Total Social Interactions: {results['total_interactions']}
  Evacuated Agents: {results['evacuated_agents']}
  Congestion Events: {len(results['congestion_events'])}
  Completed Events: {len(results['completed_events'])}

This report was generated by BIM-CrowdSim.
"""
        
        # Show in dialog
        dialog = QMessageBox(self)
        dialog.setWindowTitle("Simulation Report")
        dialog.setText(report)
        dialog.exec()
        
    def _on_preferences(self):
        """Open preferences dialog."""
        # TODO: Implement preferences dialog
        QMessageBox.information(self, "Info", "Preferences will be implemented in the next version.")
        
    def _on_about(self):
        """Show about dialog."""
        about_text = """
<h2>BIM-CrowdSim</h2>
<p><b>Version:</b> 1.2.0</p>
<p><b>BIM-native Agent-Based Modeling and Simulation Platform</b></p>
<p>BIM-CrowdSim transforms BIM models (IFC files) into intelligent simulation 
environments where people, vehicles, services, and resources can interact within 
a digital representation of a building.</p>

<p><b>Technology Stack:</b></p>
<ul>
<li>GUI: PySide6</li>
<li>BIM Processing: IfcOpenShell</li>
<li>Simulation: Mesa</li>
<li>Graph Engine: NetworkX</li>
<li>Visualization: PyVista + VTK</li>
<li>Data Analysis: Pandas</li>
<li>Storage: SQLite</li>
</ul>

<p>Copyright 2024</p>
"""
        
        QMessageBox.about(self, "About BIM-CrowdSim", about_text)
