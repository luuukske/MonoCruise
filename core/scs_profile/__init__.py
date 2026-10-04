"""Selected ETS2/ATS profile settings. See README.md."""

from .intensity import (
    BrakeIntensityCache,
    LowBrakeIntensityAebWarning,
    apply_brake_intensity,
    effective_brake_pedal,
)
from .reader import SelectedProfileSettings, brake_ui_scale, read_selected_profile

__all__ = [
    "BrakeIntensityCache",
    "LowBrakeIntensityAebWarning",
    "SelectedProfileSettings",
    "apply_brake_intensity",
    "brake_ui_scale",
    "effective_brake_pedal",
    "read_selected_profile",
]
