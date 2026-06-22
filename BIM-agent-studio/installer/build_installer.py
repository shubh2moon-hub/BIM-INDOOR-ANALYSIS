#!/usr/bin/env python3
"""
Full build pipeline for BIM-Agent Studio.

Steps:
  1. Run PyInstaller (via build.py) to create the onedir bundle.
  2. Locate the Inno Setup compiler (ISCC.exe).
  3. Compile the installer script into a distributable Setup.exe.

Usage:
    python installer/build_installer.py          # full pipeline
    python installer/build_installer.py --skip-build   # installer only (reuse existing dist/)
"""

import os
import sys
import shutil
import subprocess
from pathlib import Path


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PROJECT_DIR = Path(__file__).resolve().parent.parent
DIST_DIR = PROJECT_DIR / "dist"
APP_NAME = "BIM-Agent-Studio"
ISS_FILE = PROJECT_DIR / "installer" / "setup.iss"

# Common Inno Setup install locations
ISCC_SEARCH_PATHS = [
    Path(os.environ.get("LOCALAPPDATA", "")) / "Programs" / "Inno Setup 6" / "ISCC.exe",
    Path(os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)")) / "Inno Setup 6" / "ISCC.exe",
    Path(os.environ.get("ProgramFiles", r"C:\Program Files")) / "Inno Setup 6" / "ISCC.exe",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def find_iscc() -> Path | None:
    """Locate the Inno Setup Compiler (ISCC.exe)."""
    # Check PATH first
    for p in os.environ.get("PATH", "").split(os.pathsep):
        candidate = Path(p) / "ISCC.exe"
        if candidate.is_file():
            return candidate

    # Check common locations
    for candidate in ISCC_SEARCH_PATHS:
        if candidate.is_file():
            return candidate

    return None


def run_pyinstaller():
    """Run the PyInstaller build via build.py."""
    print("=" * 60)
    print("Step 1: PyInstaller Build")
    print("=" * 60)

    build_script = PROJECT_DIR / "build.py"
    result = subprocess.run(
        [sys.executable, str(build_script), "--mode", "onedir"],
        cwd=PROJECT_DIR,
    )

    if result.returncode != 0:
        print("\n[FAIL] PyInstaller build failed.")
        sys.exit(1)

    exe_path = DIST_DIR / APP_NAME / f"{APP_NAME}.exe"
    if not exe_path.is_file():
        print(f"\n[FAIL] Expected executable not found: {exe_path}")
        sys.exit(1)

    print(f"\n[OK] PyInstaller build succeeded: {exe_path}")
    return exe_path


def compile_installer():
    """Compile the Inno Setup script into Setup.exe."""
    print()
    print("=" * 60)
    print("Step 2: Inno Setup Installer")
    print("=" * 60)

    iscc = find_iscc()
    if iscc is None:
        print(
            "\n[FAIL] Inno Setup compiler (ISCC.exe) not found.\n"
            "  Install Inno Setup from https://jrsoftware.org/isinfo.php\n"
            "  or run:  winget install -e --id JRSoftware.InnoSetup"
        )
        sys.exit(1)

    print(f"  ISCC.exe found at: {iscc}")

    if not ISS_FILE.is_file():
        print(f"\n[FAIL] Installer script not found: {ISS_FILE}")
        sys.exit(1)

    result = subprocess.run(
        [str(iscc), str(ISS_FILE)],
        cwd=PROJECT_DIR,
    )

    if result.returncode != 0:
        print("\n[FAIL] Inno Setup compilation failed.")
        sys.exit(1)

    # Find the output
    setup_files = list(DIST_DIR.glob("BIM-Agent-Studio-Setup-*.exe"))
    if setup_files:
        setup_exe = setup_files[0]
        print(f"\n[OK] Installer created: {setup_exe}")
        print(f"  Size: {setup_exe.stat().st_size / (1024*1024):.1f} MB")
        return setup_exe
    else:
        print("\n[FAIL] Setup exe not found in dist/")
        sys.exit(1)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    import argparse

    parser = argparse.ArgumentParser(description="Build BIM-Agent Studio installer")
    parser.add_argument(
        "--skip-build",
        action="store_true",
        help="Skip PyInstaller build (reuse existing dist/)",
    )
    args = parser.parse_args()

    os.chdir(PROJECT_DIR)

    if not args.skip_build:
        run_pyinstaller()
    else:
        exe_path = DIST_DIR / APP_NAME / f"{APP_NAME}.exe"
        if not exe_path.is_file():
            print(f"✗ No existing build found at {exe_path}. Run without --skip-build.")
            sys.exit(1)
        print(f"Reusing existing build: {exe_path}")

    setup_exe = compile_installer()

    print()
    print("=" * 60)
    print("BUILD COMPLETE")
    print("=" * 60)
    print(f"  Installer: {setup_exe}")
    print()
    print("  To install on any Windows machine, just run the Setup.exe.")
    print("  It will create Start Menu shortcuts and an uninstaller.")
    print("=" * 60)


if __name__ == "__main__":
    main()
