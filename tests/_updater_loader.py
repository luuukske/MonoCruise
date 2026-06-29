"""Helper to import updater/updater.py inside the test suite.

The updater ships as a standalone exe and uses sibling-absolute imports
(`from styles import ...`, `from github_api import ...`), so it expects its own
directory on sys.path rather than being imported as a package. This helper puts
that directory on the path and loads the module under a private name.
"""

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
