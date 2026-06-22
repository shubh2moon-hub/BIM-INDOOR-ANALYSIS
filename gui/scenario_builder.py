"""
Scenario Builder Dialog
Allows users to create and edit simulation scenarios without coding.
"""

import logging
from typing import Optional, List, Dict, Any

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QDialog, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QLineEdit, QTextEdit, QSpinBox, QDoubleSpinBox,
    QComboBox, QPushButton, QTreeWidget, QTreeWidgetItem,
    QTableWidget, QTableWidgetItem, QTabWidget,
    QGroupBox, QFormLayout, QDialogButtonBox,
    QMessageBox, QHeaderView, QListWidget, QListWidgetItem,
    QCheckBox, QTimeEdit, QScrollArea, QFrame
)
from PySide6.QtGui import QFont

from engine.simulation_engine import (
    SimulationEngine, SimulationScenario, AgentProfile,
    AgentType, HumanRole
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AgentProfileDialog(QDialog):
    """Dialog for creating/editing agent profiles."""
    
    profile_saved = Signal(object)  # AgentProfile
    
    def __init__(self, parent=None, profile: Optional[AgentProfile] = None):
        super().__init__(parent)
        
        self.setWindowTitle("Agent Profile Editor")
        self.setMinimumSize(500, 600)
        
        self.profile = profile
        
        self._setup_ui()
        
        if profile:
            self._load_profile(profile)
            
    def _setup_ui(self):
        """Setup the dialog UI."""
        layout = QVBoxLayout(self)
        
        # Basic Info
        basic_group = QGroupBox("Basic Information")
        basic_form = QFormLayout(basic_group)
        
        self.name_edit = QLineEdit()
        basic_form.addRow("Name:", self.name_edit)
        
        self.type_combo = QComboBox()
        self.type_combo.addItems(["Human", "Vehicle", "Service", "Autonomous"])
        basic_form.addRow("Agent Type:", self.type_combo)
        
        self.role_combo = QComboBox()
        self.role_combo.addItems([
            "Office Worker", "Student", "Patient", "Visitor",
            "Resident", "Staff", "Cleaning", "Security",
            "Maintenance", "Delivery Robot", "Drone", "Service Bot"
        ])
        basic_form.addRow("Role:", self.role_combo)
        
        layout.addWidget(basic_group)
        
        # Movement Properties
        movement_group = QGroupBox("Movement Properties")
        movement_form = QFormLayout(movement_group)
        
        self.speed_spin = QDoubleSpinBox()
        self.speed_spin.setRange(0.1, 5.0)
        self.speed_spin.setValue(1.2)
        self.speed_spin.setSingleStep(0.1)
        self.speed_spin.setSuffix(" m/s")
        movement_form.addRow("Base Speed:", self.speed_spin)
        
        self.max_speed_spin = QDoubleSpinBox()
        self.max_speed_spin.setRange(0.1, 10.0)
        self.max_speed_spin.setValue(1.5)
        self.max_speed_spin.setSingleStep(0.1)
        self.max_speed_spin.setSuffix(" m/s")
        movement_form.addRow("Max Speed:", self.max_speed_spin)
        
        self.size_spin = QDoubleSpinBox()
        self.size_spin.setRange(0.1, 2.0)
        self.size_spin.setValue(0.5)
        self.size_spin.setSingleStep(0.1)
        self.size_spin.setSuffix(" m")
        movement_form.addRow("Size (radius):", self.size_spin)
        
        self.vision_spin = QDoubleSpinBox()
        self.vision_spin.setRange(1.0, 50.0)
        self.vision_spin.setValue(10.0)
        self.vision_spin.setSingleStep(1.0)
        self.vision_spin.setSuffix(" m")
        movement_form.addRow("Vision Range:", self.vision_spin)
        
        layout.addWidget(movement_group)
        
        # Behavior Properties
        behavior_group = QGroupBox("Behavior Properties")
        behavior_form = QFormLayout(behavior_group)
        
        self.patience_slider = QSpinBox()
        self.patience_slider.setRange(0, 100)
        self.patience_slider.setValue(50)
        behavior_form.addRow("Patience (%):", self.patience_slider)
        
        self.sociability_slider = QSpinBox()
        self.sociability_slider.setRange(0, 100)
        self.sociability_slider.setValue(50)
        behavior_form.addRow("Sociability (%):", self.sociability_slider)
        
        self.risk_slider = QSpinBox()
        self.risk_slider.setRange(0, 100)
        self.risk_slider.setValue(50)
        behavior_form.addRow("Risk Tolerance (%):", self.risk_slider)
        
        layout.addWidget(behavior_group)
        
        # Accessibility
        accessibility_group = QGroupBox("Accessibility")
        accessibility_layout = QVBoxLayout(accessibility_group)
        
        self.needs_accessible = QCheckBox("Requires Accessible Routes")
        accessibility_layout.addWidget(self.needs_accessible)
        
        self.can_use_stairs = QCheckBox("Can Use Stairs")
        self.can_use_stairs.setChecked(True)
        accessibility_layout.addWidget(self.can_use_stairs)
        
        self.can_use_elevator = QCheckBox("Can Use Elevator")
        self.can_use_elevator.setChecked(True)
        accessibility_layout.addWidget(self.can_use_elevator)
        
        layout.addWidget(accessibility_group)
        
        # Buttons
        button_box = QDialogButtonBox(
            QDialogButtonBox.Save | QDialogButtonBox.Cancel
        )
        button_box.accepted.connect(self._on_save)
        button_box.rejected.connect(self.reject)
        layout.addWidget(button_box)
        
        layout.addStretch()
        
    def _load_profile(self, profile: AgentProfile):
        """Load existing profile data."""
        self.name_edit.setText(profile.name)
        
        type_index = self.type_combo.findText(profile.agent_type.value.title())
        if type_index >= 0:
            self.type_combo.setCurrentIndex(type_index)
            
        if profile.role:
            role_index = self.role_combo.findText(profile.role.value.replace("_", " ").title())
            if role_index >= 0:
                self.role_combo.setCurrentIndex(role_index)
                
        self.speed_spin.setValue(profile.base_speed)
        self.max_speed_spin.setValue(profile.max_speed)
        self.size_spin.setValue(profile.size)
        self.vision_spin.setValue(profile.vision_range)
        self.patience_slider.setValue(int(profile.patience * 100))
        self.sociability_slider.setValue(int(profile.sociability * 100))
        self.risk_slider.setValue(int(profile.risk_tolerance * 100))
        self.needs_accessible.setChecked(profile.needs_accessible)
        self.can_use_stairs.setChecked(profile.can_use_stairs)
        self.can_use_elevator.setChecked(profile.can_use_elevator)
        
    def _on_save(self):
        """Save the profile."""
        name = self.name_edit.text().strip()
        if not name:
            QMessageBox.warning(self, "Warning", "Please enter a profile name")
            return
            
        # Map type string to enum
        type_map = {
            "Human": AgentType.HUMAN,
            "Vehicle": AgentType.VEHICLE,
            "Service": AgentType.SERVICE,
            "Autonomous": AgentType.AUTONOMOUS
        }
        
        role_map = {
            "Office Worker": HumanRole.OFFICE_WORKER,
            "Student": HumanRole.STUDENT,
            "Patient": HumanRole.PATIENT,
            "Visitor": HumanRole.VISITOR,
            "Resident": HumanRole.RESIDENT,
            "Staff": HumanRole.STAFF,
            "Cleaning": HumanRole.STAFF,
            "Security": HumanRole.STAFF,
            "Maintenance": HumanRole.STAFF,
        }
        
        agent_type = type_map.get(self.type_combo.currentText(), AgentType.HUMAN)
        
        role = None
        if agent_type == AgentType.HUMAN:
            role = role_map.get(self.role_combo.currentText())
            
        profile = AgentProfile(
            id=self.profile.id if self.profile else "",
            name=name,
            agent_type=agent_type,
            role=role,
            base_speed=self.speed_spin.value(),
            max_speed=self.max_speed_spin.value(),
            size=self.size_spin.value(),
            vision_range=self.vision_spin.value(),
            patience=self.patience_slider.value() / 100.0,
            sociability=self.sociability_slider.value() / 100.0,
            risk_tolerance=self.risk_slider.value() / 100.0,
            needs_accessible=self.needs_accessible.isChecked(),
            can_use_stairs=self.can_use_stairs.isChecked(),
            can_use_elevator=self.can_use_elevator.isChecked()
        )
        
        self.profile_saved.emit(profile)
        self.accept()


class EventEditorDialog(QDialog):
    """Dialog for creating simulation events."""
    
    event_saved = Signal(object)  # Dict
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        self.setWindowTitle("Event Editor")
        self.setMinimumSize(400, 300)
        
        self._setup_ui()
        
    def _setup_ui(self):
        """Setup the dialog UI."""
        layout = QVBoxLayout(self)
        
        form = QFormLayout()
        
        # Event type
        self.type_combo = QComboBox()
        self.type_combo.addItems([
            "spawn_agents", "evacuate", "fire", "block_path",
            "set_destination", "change_behavior"
        ])
        form.addRow("Event Type:", self.type_combo)
        
        # Time
        self.time_spin = QSpinBox()
        self.time_spin.setRange(0, 86400)
        self.time_spin.setValue(0)
        self.time_spin.setSuffix(" seconds")
        form.addRow("Trigger Time:", self.time_spin)
        
        # Parameters
        self.params_edit = QTextEdit()
        self.params_edit.setPlaceholderText("Enter parameters as key=value, one per line...")
        self.params_edit.setMaximumHeight(100)
        form.addRow("Parameters:", self.params_edit)
        
        layout.addLayout(form)
        
        # Description
        self.desc_label = QLabel("Select an event type to see description")
        self.desc_label.setWordWrap(True)
        layout.addWidget(self.desc_label)
        
        # Buttons
        button_box = QDialogButtonBox(
            QDialogButtonBox.Save | QDialogButtonBox.Cancel
        )
        button_box.accepted.connect(self._on_save)
        button_box.rejected.connect(self.reject)
        layout.addWidget(button_box)
        
    def _on_save(self):
        """Save the event."""
        event_type = self.type_combo.currentText()
        time = self.time_spin.value()
        
        # Parse parameters
        params = {}
        for line in self.params_edit.toPlainText().strip().split("\n"):
            if "=" in line:
                key, value = line.split("=", 1)
                params[key.strip()] = value.strip()
                
        event = {
            "type": event_type,
            "time": time,
            **params
        }
        
        self.event_saved.emit(event)
        self.accept()


class ScenarioBuilderDialog(QDialog):
    """Main scenario builder dialog."""
    
    scenario_created = Signal(object)  # SimulationScenario
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        self.setWindowTitle("Scenario Builder")
        self.setMinimumSize(800, 700)
        
        self.profiles: List[AgentProfile] = []
        self.events: List[Dict] = []
        self.agent_counts: Dict[str, int] = {}
        
        self._setup_ui()
        
    def _setup_ui(self):
        """Setup the dialog UI."""
        layout = QVBoxLayout(self)
        
        # Tabs
        tabs = QTabWidget()
        
        # General tab
        general_tab = QWidget()
        general_layout = QFormLayout(general_tab)
        
        self.name_edit = QLineEdit()
        self.name_edit.setPlaceholderText("Enter scenario name...")
        general_layout.addRow("Name:", self.name_edit)
        
        self.desc_edit = QTextEdit()
        self.desc_edit.setPlaceholderText("Enter scenario description...")
        self.desc_edit.setMaximumHeight(80)
        general_layout.addRow("Description:", self.desc_edit)
        
        self.duration_spin = QSpinBox()
        self.duration_spin.setRange(60, 86400)
        self.duration_spin.setValue(3600)
        self.duration_spin.setSuffix(" seconds")
        self.duration_spin.setSingleStep(300)
        general_layout.addRow("Duration:", self.duration_spin)
        
        self.timestep_spin = QDoubleSpinBox()
        self.timestep_spin.setRange(0.1, 10.0)
        self.timestep_spin.setValue(1.0)
        self.timestep_spin.setSuffix(" seconds")
        general_layout.addRow("Time Step:", self.timestep_spin)
        
        tabs.addTab(general_tab, "General")
        
        # Agents tab
        agents_tab = QWidget()
        agents_layout = QVBoxLayout(agents_tab)
        
        # Toolbar
        agents_toolbar = QHBoxLayout()
        
        btn_add_profile = QPushButton("Add Agent Profile")
        btn_add_profile.clicked.connect(self._on_add_profile)
        agents_toolbar.addWidget(btn_add_profile)
        
        btn_edit_profile = QPushButton("Edit")
        btn_edit_profile.clicked.connect(self._on_edit_profile)
        agents_toolbar.addWidget(btn_edit_profile)
        
        btn_remove_profile = QPushButton("Remove")
        btn_remove_profile.clicked.connect(self._on_remove_profile)
        agents_toolbar.addWidget(btn_remove_profile)
        
        agents_toolbar.addStretch()
        agents_layout.addLayout(agents_toolbar)
        
        # Profiles list
        self.profiles_table = QTableWidget()
        self.profiles_table.setColumnCount(5)
        self.profiles_table.setHorizontalHeaderLabels([
            "Name", "Type", "Role", "Count", "Speed (m/s)"
        ])
        self.profiles_table.horizontalHeader().setStretchLastSection(True)
        agents_layout.addWidget(self.profiles_table)
        
        tabs.addTab(agents_tab, "Agents")
        
        # Events tab
        events_tab = QWidget()
        events_layout = QVBoxLayout(events_tab)
        
        # Toolbar
        events_toolbar = QHBoxLayout()
        
        btn_add_event = QPushButton("Add Event")
        btn_add_event.clicked.connect(self._on_add_event)
        events_toolbar.addWidget(btn_add_event)
        
        btn_remove_event = QPushButton("Remove")
        btn_remove_event.clicked.connect(self._on_remove_event)
        events_toolbar.addWidget(btn_remove_event)
        
        events_toolbar.addStretch()
        events_layout.addLayout(events_toolbar)
        
        # Events list
        self.events_table = QTableWidget()
        self.events_table.setColumnCount(3)
        self.events_table.setHorizontalHeaderLabels([
            "Time (s)", "Type", "Parameters"
        ])
        self.events_table.horizontalHeader().setStretchLastSection(True)
        events_layout.addWidget(self.events_table)
        
        tabs.addTab(events_tab, "Events")
        
        # Summary tab
        summary_tab = QWidget()
        summary_layout = QVBoxLayout(summary_tab)
        
        self.summary_text = QTextEdit()
        self.summary_text.setReadOnly(True)
        summary_layout.addWidget(self.summary_text)
        
        btn_refresh_summary = QPushButton("Refresh Summary")
        btn_refresh_summary.clicked.connect(self._update_summary)
        summary_layout.addWidget(btn_refresh_summary)
        
        tabs.addTab(summary_tab, "Summary")
        
        layout.addWidget(tabs)
        
        # Buttons
        button_box = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel
        )
        button_box.accepted.connect(self._on_create)
        button_box.rejected.connect(self.reject)
        layout.addWidget(button_box)
        
    def _on_add_profile(self):
        """Open dialog to add agent profile."""
        dialog = AgentProfileDialog(self)
        dialog.profile_saved.connect(self._on_profile_saved)
        dialog.exec()
        
    def _on_profile_saved(self, profile: AgentProfile):
        """Handle saved profile."""
        self.profiles.append(profile)
        self.agent_counts[profile.id] = 10  # Default count
        self._update_profiles_table()
        
    def _on_edit_profile(self):
        """Edit selected profile."""
        row = self.profiles_table.currentRow()
        if row < 0 or row >= len(self.profiles):
            return
            
        profile = self.profiles[row]
        dialog = AgentProfileDialog(self, profile)
        dialog.profile_saved.connect(lambda p: self._on_profile_updated(row, p))
        dialog.exec()
        
    def _on_profile_updated(self, row: int, profile: AgentProfile):
        """Handle profile update."""
        self.profiles[row] = profile
        self._update_profiles_table()
        
    def _on_remove_profile(self):
        """Remove selected profile."""
        row = self.profiles_table.currentRow()
        if row < 0 or row >= len(self.profiles):
            return
            
        profile = self.profiles.pop(row)
        if profile.id in self.agent_counts:
            del self.agent_counts[profile.id]
        self._update_profiles_table()
        
    def _update_profiles_table(self):
        """Update the profiles table."""
        self.profiles_table.setRowCount(0)
        
        for profile in self.profiles:
            row = self.profiles_table.rowCount()
            self.profiles_table.insertRow(row)
            
            self.profiles_table.setItem(row, 0, QTableWidgetItem(profile.name))
            self.profiles_table.setItem(row, 1, QTableWidgetItem(profile.agent_type.value))
            
            role = profile.role.value if profile.role else ""
            self.profiles_table.setItem(row, 2, QTableWidgetItem(role))
            
            count = self.agent_counts.get(profile.id, 0)
            self.profiles_table.setItem(row, 3, QTableWidgetItem(str(count)))
            
            self.profiles_table.setItem(row, 4, QTableWidgetItem(f"{profile.base_speed:.1f}"))
            
    def _on_add_event(self):
        """Open dialog to add event."""
        dialog = EventEditorDialog(self)
        dialog.event_saved.connect(self._on_event_saved)
        dialog.exec()
        
    def _on_event_saved(self, event: Dict):
        """Handle saved event."""
        self.events.append(event)
        self._update_events_table()
        
    def _on_remove_event(self):
        """Remove selected event."""
        row = self.events_table.currentRow()
        if row < 0 or row >= len(self.events):
            return
            
        self.events.pop(row)
        self._update_events_table()
        
    def _update_events_table(self):
        """Update the events table."""
        self.events_table.setRowCount(0)
        
        # Sort by time
        sorted_events = sorted(self.events, key=lambda e: e.get("time", 0))
        
        for event in sorted_events:
            row = self.events_table.rowCount()
            self.events_table.insertRow(row)
            
            self.events_table.setItem(row, 0, QTableWidgetItem(str(event.get("time", 0))))
            self.events_table.setItem(row, 1, QTableWidgetItem(event.get("type", "")))
            
            # Format parameters
            params = {k: v for k, v in event.items() if k not in ["type", "time"]}
            params_str = ", ".join(f"{k}={v}" for k, v in params.items())
            self.events_table.setItem(row, 2, QTableWidgetItem(params_str))
            
    def _update_summary(self):
        """Update the summary text."""
        summary = f"""Scenario Summary
================

Name: {self.name_edit.text() or 'Unnamed'}
Description: {self.desc_edit.toPlainText() or 'None'}
Duration: {self.duration_spin.value()} seconds
Time Step: {self.timestep_spin.value()} seconds

Agent Profiles: {len(self.profiles)}
"""
        
        for profile in self.profiles:
            count = self.agent_counts.get(profile.id, 0)
            summary += f"  - {profile.name} ({profile.agent_type.value}): {count} agents\n"
            
        summary += f"\nScheduled Events: {len(self.events)}\n"
        
        for event in sorted(self.events, key=lambda e: e.get("time", 0)):
            summary += f"  - T+{event.get('time', 0)}s: {event.get('type', '')}\n"
            
        self.summary_text.setText(summary)
        
    def _on_create(self):
        """Create the scenario."""
        name = self.name_edit.text().strip()
        if not name:
            QMessageBox.warning(self, "Warning", "Please enter a scenario name")
            return
            
        if not self.profiles:
            QMessageBox.warning(self, "Warning", "Please add at least one agent profile")
            return
            
        engine = SimulationEngine()
        
        scenario = engine.create_scenario(
            name=name,
            description=self.desc_edit.toPlainText(),
            duration=self.duration_spin.value(),
            time_step=self.timestep_spin.value(),
            agent_profiles=self.profiles.copy(),
            agent_counts=self.agent_counts.copy(),
            events=self.events.copy()
        )
        
        self.scenario_created.emit(scenario)
        self.accept()
