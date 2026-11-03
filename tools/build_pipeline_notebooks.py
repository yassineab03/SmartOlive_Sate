"""Compatibility entry point: build the CURRENT satellite notebooks."""
from pathlib import Path
import runpy
runpy.run_path(str(Path(__file__).with_name('build_current_satellite_notebooks.py')),run_name='__main__')
