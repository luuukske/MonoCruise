"""
Lightweight UI components for MonoCruise.

These modules intentionally contain **no threading logic**.  They are
instantiated and driven exclusively by UiThread.
"""

from .hud import HUD
from .popup import PopupManager

__all__ = ["HUD", "PopupManager"]

