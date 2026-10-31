"""Compatibility entry point; delegates to the maintained API."""
import runpy
import sys
from pathlib import Path

if __name__ == '__main__':
    app_path = Path(__file__).resolve().parent / 'app'
    sys.path.insert(0, str(app_path))
    runpy.run_path(str(app_path / 'main.py'), run_name='__main__')
