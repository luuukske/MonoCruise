"""Root conftest: adds repo root to sys.path so core.* imports resolve."""
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

