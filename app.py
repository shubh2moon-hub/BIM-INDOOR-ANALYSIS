#!/usr/bin/env python3
"""
BIM-Agent Studio - Main Application Entry Point
BIM-native Agent-Based Modeling and Simulation Platform
"""

import sys
import os
import logging
import argparse
from pathlib import Path

# Ensure the project directory is in path
project_dir = Path(__file__).parent
if str(project_dir) not in sys.path:
    sys.path.insert(0, str(project_dir))

def setup_logging(verbose: bool = False):
    """Configure application logging."""
    level = logging.DEBUG if verbose else logging.INFO
    
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler('bim_agent_studio.log', mode='w')
        ]
    )

def check_dependencies():
    """Check if required dependencies are available."""
    missing = []
    
    try:
        import PySide6
    except ImportError:
        missing.append("PySide6")
        
    try:
        import ifcopenshell
    except ImportError:
        missing.append("ifcopenshell")
        
    try:
        import mesa
    except ImportError:
        missing.append("mesa")
        
    try:
        import networkx
    except ImportError:
        missing.append("networkx")
        
    try:
        import pyvista
    except ImportError:
        missing.append("pyvista")
        
    try:
        import vtk
    except ImportError:
        missing.append("vtk")
        
    try:
        import pandas
    except ImportError:
        missing.append("pandas")
        
    try:
        import numpy
    except ImportError:
        missing.append("numpy")
        
    try:
        import scipy
    except ImportError:
        missing.append("scipy")
        
    try:
        import shapely
    except ImportError:
        missing.append("shapely")
        
    if missing:
        print("Missing required dependencies:")
        for dep in missing:
            print(f"  - {dep}")
        print("\nInstall with: pip install -r requirements.txt")
        return False
        
    return True

def main():
    """Main application entry point."""
    parser = argparse.ArgumentParser(
        description='BIM-Agent Studio - BIM-native Agent-Based Modeling and Simulation Platform'
    )
    parser.add_argument(
        '--ifc', '-i',
        type=str,
        help='Path to IFC file to open on startup'
    )
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose logging'
    )
    parser.add_argument(
        '--no-3d',
        action='store_true',
        help='Disable 3D visualization (fallback mode)'
    )
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.verbose)
    logger = logging.getLogger(__name__)
    
    logger.info("=" * 60)
    logger.info("BIM-Agent Studio Starting")
    logger.info("=" * 60)
    
    # Check dependencies
    if not check_dependencies():
        logger.error("Missing dependencies, exiting")
        sys.exit(1)
        
    logger.info("All dependencies verified")
    
    # Import GUI components (after dependency check)
    from PySide6.QtWidgets import QApplication
    from PySide6.QtCore import Qt
    from PySide6.QtGui import QFont, QFontDatabase
    
    # Enable high DPI support
    QApplication.setHighDpiScaleFactorRoundingPolicy(
        Qt.HighDpiScaleFactorRoundingPolicy.PassThrough
    )
    
    # Create application
    app = QApplication(sys.argv)
    app.setApplicationName("BIM-Agent Studio")
    app.setApplicationVersion("1.0.0")
    app.setOrganizationName("BIM-Agent Studio")
    
    # Set application font
    font = QFont("Segoe UI", 9)
    app.setFont(font)
    
    # Set application style
    app.setStyle("Fusion")
    
    # Apply stylesheet
    app.setStyleSheet("""
        QMainWindow {
            background-color: #f5f5f7;
        }
        QDockWidget {
            titlebar-close-icon: url(close.png);
        }
        QDockWidget::title {
            background: #e8e8ec;
            padding: 6px;
            border: 1px solid #d0d0d5;
        }
        QGroupBox {
            font-weight: bold;
            border: 1px solid #d0d0d5;
            border-radius: 4px;
            margin-top: 8px;
            padding-top: 8px;
        }
        QGroupBox::title {
            subcontrol-origin: margin;
            left: 8px;
            padding: 0 4px;
        }
        QPushButton {
            background-color: #0078d4;
            color: white;
            border: none;
            padding: 6px 16px;
            border-radius: 3px;
        }
        QPushButton:hover {
            background-color: #006cbd;
        }
        QPushButton:pressed {
            background-color: #005a9e;
        }
        QPushButton:disabled {
            background-color: #cccccc;
            color: #888888;
        }
        QTreeWidget {
            border: 1px solid #d0d0d5;
            alternate-background-color: #f8f8fa;
        }
        QTableWidget {
            border: 1px solid #d0d0d5;
            alternate-background-color: #f8f8fa;
        }
        QTextEdit {
            border: 1px solid #d0d0d5;
        }
        QComboBox {
            border: 1px solid #d0d0d5;
            padding: 4px;
            border-radius: 3px;
        }
        QSpinBox, QDoubleSpinBox {
            border: 1px solid #d0d0d5;
            padding: 4px;
            border-radius: 3px;
        }
        QSlider::groove:horizontal {
            border: 1px solid #d0d0d5;
            height: 6px;
            background: #e8e8ec;
            border-radius: 3px;
        }
        QSlider::handle:horizontal {
            background: #0078d4;
            width: 16px;
            border-radius: 8px;
        }
        QProgressBar {
            border: 1px solid #d0d0d5;
            border-radius: 3px;
            text-align: center;
        }
        QProgressBar::chunk {
            background-color: #0078d4;
        }
        QTabWidget::pane {
            border: 1px solid #d0d0d5;
            background-color: white;
        }
        QTabBar::tab {
            background: #e8e8ec;
            padding: 6px 12px;
            border: 1px solid #d0d0d5;
            border-bottom: none;
            border-top-left-radius: 3px;
            border-top-right-radius: 3px;
        }
        QTabBar::tab:selected {
            background: white;
            border-bottom: 1px solid white;
        }
        QStatusBar {
            background-color: #0078d4;
            color: white;
        }
        QLabel {
            color: #333333;
        }
        QToolBar {
            background-color: #f0f0f4;
            border: 1px solid #d0d0d5;
            spacing: 4px;
            padding: 4px;
        }
        QMenuBar {
            background-color: #f0f0f4;
            border-bottom: 1px solid #d0d0d5;
        }
        QMenuBar::item:selected {
            background-color: #0078d4;
            color: white;
        }
        QMenu {
            background-color: white;
            border: 1px solid #d0d0d5;
        }
        QMenu::item:selected {
            background-color: #0078d4;
            color: white;
        }
    """)
    
    # Create and show main window
    from gui.main_window import MainWindow
    
    logger.info("Creating main window...")
    window = MainWindow()
    window.show()
    
    # Open IFC file if specified
    if args.ifc:
        if os.path.exists(args.ifc):
            logger.info(f"Opening IFC file: {args.ifc}")
            window.log_message(f"Opening IFC file: {args.ifc}")
            window._load_ifc_file(args.ifc)
        else:
            logger.warning(f"IFC file not found: {args.ifc}")
            window.log_message(f"Warning: IFC file not found: {args.ifc}", "WARNING")
    
    logger.info("Application started successfully")
    
    # Run application
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
