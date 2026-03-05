"""
UI entry point — creates and returns the MonoCruiseWindow.

This is NOT a thread.  The main window lives on the main thread alongside
the QApplication event loop, exactly like PopupWindow.  Qt requires all
GUI objects (especially those with QTimers) to live on the thread that
owns the QApplication.

Usage from the root main.py::

    from ui.main_window.main import create_main_window

    window = create_main_window(settings, version="v2.0.0")
    # window is now shown and managed by the main-thread app.exec()

Other threads read UI state via the registry::

    registry.get_thread("telemetry_thread").data  # etc.

The window reads thread state via its internal QTimer poll (already in
window.py), so no extra glue is needed.
"""

from __future__ import annotations

import logging
import sys
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from core.settings import Settings
    from ui.main_window.window import MonoCruiseWindow

logger = logging.getLogger(__name__)


def create_main_window(
    settings: "Settings",
    version: str = "v2.0.0",
) -> "MonoCruiseWindow":
    """
    Instantiate and return the MonoCruiseWindow.

    Must be called from the main thread after QApplication has been created.
    The caller (root main.py) is responsible for calling ``app.exec()``.

    Args:
        settings: The loaded Settings instance (shared with all threads).
        version:  Version string displayed in the UI.

    Returns:
        The MonoCruiseWindow instance (not shown yet).
    """
    from ui.main_window.window import MonoCruiseWindow

    window = MonoCruiseWindow(settings, version=version)
    logger.info("Main window created")
    return window