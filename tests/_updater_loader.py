"""Import updater/updater.py for tests (standalone sibling imports, not a package)."""
from __future__ import annotations

import importlib.util
import os
import sys

UPDATER_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "updater"
)
if UPDATER_DIR not in sys.path:
    sys.path.insert(0, UPDATER_DIR)


def load_updater():
    spec = importlib.util.spec_from_file_location(
        "mc_updater", os.path.join(UPDATER_DIR, "updater.py")
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
