"""Launcher for the HSI Viewer (PySide6).

    python HSI_Viewer.py                 # open the viewer
    python HSI_Viewer.py path/to/cube.tiff
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from hsi_viewer.app import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
