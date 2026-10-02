"""Main-window factory on the Qt main thread (not a worker thread)."""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from core.settings import Settings
    from ui.main_window.window import MonoCruiseWindow

logger = logging.getLogger(__name__)


def create_main_window(
    settings: "Settings",
    version: str = "v2.0.0",
) -> "MonoCruiseWindow":
    """Create ``MonoCruiseWindow``; call from the main thread after ``QApplication`` exists."""
    from ui.main_window.window import MonoCruiseWindow

    window = MonoCruiseWindow(settings, version=version)
    logger.info("Main window created")
    return window
