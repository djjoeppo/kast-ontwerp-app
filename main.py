#!/usr/bin/env python3
"""
Main entry point script for the Acoustics Platform Desktop Application.
Automatically configures the Python search path to the project root and
launches the PyQt6 graphical user interface with PyVista 3D visualization.
"""

import os
import sys
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from acoustics_platform.ui.gui import run_gui


def main():
    """Launch Desktop GUI application."""
    sys.exit(run_gui())


if __name__ == "__main__":
    main()
