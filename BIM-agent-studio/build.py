#!/usr/bin/env python3
"""
Build script for BIM-Agent Studio
Uses PyInstaller to create standalone executables.
"""

import os
import sys
import shutil
import subprocess
from pathlib import Path


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

APP_NAME = "BIM-Agent-Studio"
APP_VERSION = "1.0.0"
MAIN_SCRIPT = "app.py"
ICON_PATH = os.path.join("assets", "icon.ico")

# Directories to bundle as data
DATA_DIRS = [
    ("core", "core"),
    ("gui", "gui"),
    ("engine", "engine"),
    ("visualization", "visualization"),
    ("data", "data"),
]

# Hidden imports that PyInstaller cannot detect automatically
HIDDEN_IMPORTS = [
    # --- IFC / OpenShell ---
    "ifcopenshell",
    "ifcopenshell.geom",
    "ifcopenshell.util",
    "ifcopenshell.util.element",
    "ifcopenshell.util.shape",
    # --- Mesa ABM ---
    "mesa",
    "mesa.time",
    "mesa.space",
    "mesa.datacollection",
    # --- NetworkX ---
    "networkx",
    "networkx.algorithms",
    # --- PyVista / VTK ---
    "pyvista",
    "pyvistaqt",
    "vtkmodules",
    "vtkmodules.all",
    "vtkmodules.util",
    "vtkmodules.util.numpy_support",
    "vtkmodules.numpy_interface",
    "vtkmodules.numpy_interface.dataset_adapter",
    # --- Scientific ---
    "numpy",
    "pandas",
    "scipy",
    "scipy.spatial",
    "scipy.spatial.distance",
    "shapely",
    "shapely.geometry",
    "shapely.ops",
    "matplotlib",
    "matplotlib.backends",
    "matplotlib.backends.backend_agg",
    # --- PySide6 / Qt ---
    "PySide6",
    "PySide6.QtCore",
    "PySide6.QtGui",
    "PySide6.QtWidgets",
    "PySide6.QtOpenGL",
    "PySide6.QtOpenGLWidgets",
    # --- Pillow ---
    "PIL",
    "PIL.Image",
]

# Packages for which PyInstaller should collect *all* submodules and data
COLLECT_ALL = [
    "pyvista",
    "vtkmodules",
    "ifcopenshell",
    "PySide6",
]

COLLECT_DATA = [
    "pyvista",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ensure_pyinstaller():
    """Make sure PyInstaller is available."""
    try:
        import PyInstaller
        print(f"  PyInstaller version: {PyInstaller.__version__}")
    except ImportError:
        print("  Installing PyInstaller …")
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "pyinstaller"],
            check=True,
        )


def _clean(project_dir: Path):
    """Remove artefacts from previous builds."""
    for name in ("build", "dist"):
        d = project_dir / name
        if d.exists():
            print(f"  Removing {d}")
            shutil.rmtree(d)
    spec = project_dir / f"{APP_NAME}.spec"
    if spec.exists():
        spec.unlink()


def _build_cmd(mode: str) -> list[str]:
    """Assemble the PyInstaller command line."""
    cmd = [
        sys.executable, "-m", "PyInstaller",
        f"--name={APP_NAME}",
        "--windowed",
        f"--{mode}",
        "--clean",
        "--noconfirm",
    ]

    # Icon
    icon = Path(ICON_PATH)
    if icon.exists():
        cmd += ["--icon", str(icon)]

    # Data directories
    for src, dst in DATA_DIRS:
        if Path(src).exists():
            cmd += ["--add-data", f"{src}{os.pathsep}{dst}"]

    # Hidden imports
    for imp in HIDDEN_IMPORTS:
        cmd += ["--hidden-import", imp]

    # Collect-all
    for pkg in COLLECT_ALL:
        cmd += ["--collect-all", pkg]

    # Collect-data
    for pkg in COLLECT_DATA:
        cmd += ["--collect-data", pkg]

    # Main script
    cmd.append(MAIN_SCRIPT)
    return cmd


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build(mode: str = "onedir"):
    """Build the application.

    Parameters
    ----------
    mode : str
        ``"onedir"`` (default, fast startup) or ``"onefile"`` (single exe).
    """
    print("=" * 60)
    print(f"BIM-Agent Studio Build Script  (mode={mode})")
    print("=" * 60)

    project_dir = Path(__file__).parent
    os.chdir(project_dir)

    _ensure_pyinstaller()
    _clean(project_dir)

    cmd = _build_cmd(mode)

    print()
    print("Building application …")
    print("This may take several minutes.")
    print("-" * 60)

    result = subprocess.run(cmd, text=True)

    print("-" * 60)
    if result.returncode == 0:
        if mode == "onedir":
            exe = project_dir / "dist" / APP_NAME / f"{APP_NAME}.exe"
        else:
            exe = project_dir / "dist" / f"{APP_NAME}.exe"
        print("Build completed successfully!")
        print(f"\nExecutable: {exe}")
    else:
        print("Build FAILED!")
        print(f"Return code: {result.returncode}")

    return result.returncode


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Build BIM-Agent Studio")
    parser.add_argument(
        "--mode",
        choices=["onefile", "onedir"],
        default="onedir",
        help="Build mode (default: onedir)",
    )
    args = parser.parse_args()
    sys.exit(build(args.mode))
