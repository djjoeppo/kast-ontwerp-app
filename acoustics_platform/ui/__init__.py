"""
User interface module (CLI and GUI).
"""

from acoustics_platform.ui.cli import main as cli_main
from acoustics_platform.ui.gui import run_gui, PYQT_AVAILABLE

__all__ = ["cli_main", "run_gui", "PYQT_AVAILABLE"]
