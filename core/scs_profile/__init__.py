"""Selected ETS2/ATS profile settings. See README.md."""

from .intensity import (
    BrakeIntensityCache,
    apply_brake_intensity,
    learn_decel_scale,
)
from .reader import SelectedProfileSettings, brake_ui_scale, read_selected_profile

__all__ = [
    "BrakeIntensityCache",
    "SelectedProfileSettings",
    "apply_brake_intensity",
    "brake_ui_scale",
    "learn_decel_scale",
    "read_selected_profile",
]
