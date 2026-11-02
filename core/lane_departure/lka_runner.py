"""The LA core run as the lane keeping assist: its settings, its pulse logs and ``lka_debug.csv``.

``Settings.lka_enabled`` and friends, config only. Calibration is shared with LA (one wheel, one truck), and so are
the LA's fixed calibration keys. See core/lane_departure/README.md, "Lane keeping assist".
"""

from __future__ import annotations

import logging
import math
from dataclasses import replace

from core.la.controller import STRENGTH_MAX, AssistTuning, Wheel, WheelTuning
from core.la.runner import _ARM_MARGIN_KMH, AssistRunner, Request, _setting
from core.lane_departure.command import PluginStatus
from core.settings import Settings

from .detector import SIDE_LEFT, Frame
from .lka import Pulse

logger = logging.getLogger(__name__)

_SKIPS = {"rate": "the per-minute limit", "swerve": "a swerve in the last seconds", "override": "the driver took over"}


class LkaRunner(AssistRunner):
    LABEL = "LKA"
    IDLE = AssistRunner.IDLE | {"standby"}
    DEBUG_NAME = "lka_debug.csv"

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._closest = math.inf
        self._one_sided_warned = False

    def _wheel(self, status: PluginStatus) -> Wheel:
        """A pulse needs the plugin's one-sided servo, so it can never push back toward the line; older ones steer
        both ways, and it stays off there (``plugin``).
        """
        wheel = super()._wheel(status)
        if status.present and not status.one_sided:
            if not self._one_sided_warned:
                self._one_sided_warned = True
                logger.info("LKA needs the FFB plugin with the one-sided servo; the installed one steers both ways, "
                            "so LKA stays off")
            return replace(wheel, ok=False)
        return wheel

    def step_pulse(self, frame: Frame | None, repeat: bool, status: PluginStatus, tmp: bool, pulse: Pulse,
                   edge) -> Request:
        """This tick's request for ``pulse``; ``edge`` is the detector's view of that pulse's line, for the logs."""
        request = self.step(frame, repeat, status, tmp, engage=pulse.side)
        self._log_pulse(pulse, edge)
        return request

    def _log_pulse(self, pulse: Pulse, edge) -> None:
        if pulse.side and edge is not None:
            self._closest = min(self._closest, edge.gap)
        if pulse.event == "skipped":
            logger.info("LKA skips a departure at the %s line: %s", "left" if pulse.line == SIDE_LEFT else "right",
                        _SKIPS.get(pulse.reason, pulse.reason))
        elif pulse.event == "start":
            self._closest = edge.gap if edge is not None else math.inf
            logger.info("LKA steers back from the %s line (%s past it): gap %.2f m, drift %.2f m/s, time to line "
                        "%.2f s, %.0f km/h, %s", "left" if pulse.side == SIDE_LEFT else "right", pulse.beyond,
                        edge.gap if edge else math.nan, edge.drift if edge else math.nan,
                        edge.tlc if edge else math.nan, self.output.speed_kmh, self.output.reason)
        elif pulse.event:
            logger.info("LKA pulse ends (%s) after %.2f s: closest %.2f m inside the line (negative: past it)",
                        pulse.event, pulse.elapsed_s, self._closest)

    def _tuning(self) -> tuple[tuple, AssistTuning, tuple]:
        arm = _setting("ldw_min_speed_kmh", 60.0)
        strength = max(0.0, min(STRENGTH_MAX, _setting("lka_strength", 1.0)))
        inertia = max(50.0, min(20000.0, _setting("la_wheel_inertia", 0.0) or WheelTuning().inertia))
        key = (arm, strength, bool(getattr(Settings, "lka_tmp_enabled", False)), inertia)
        line = ("LKA settings: from %.0f km/h, strength %.2f, TruckersMP %s, wheel inertia %.0f", *key)
        return key, AssistTuning(
            pulse=True, settle_s=1.0, arm_kmh=arm, disarm_kmh=max(0.0, arm - _ARM_MARGIN_KMH),
            wheel=WheelTuning(strength=strength, inertia=inertia), allow_tmp=key[2],
        ), line
