"""The LA core run as the lane keeping assist: its settings, its pulse logs and ``lka_debug.csv``.

``Settings.lka_enabled`` and friends, config only. Calibration is shared with LA (one wheel, one truck), and so are
the LA's fixed calibration keys. See core/lane_departure/README.md, "Lane keeping assist".
"""

from __future__ import annotations

import logging
import math
from dataclasses import replace

from core.la.actuator import design_overrides, push_ahead_overrides
from core.la.controller import STRENGTH_MAX, AssistTuning, Wheel, WheelTuning
from core.la.runner import _ARM_MARGIN_KMH, AssistRunner, Request, _setting
from core.lane_departure.command import PluginStatus
from core.settings import Settings

from .detector import SIDE_LEFT, Frame
from .lka import Pulse

logger = logging.getLogger(__name__)

_SKIPS = {"swerve": "a swerve in the last seconds", "override": "the driver took over",
          "drift": "the truck itself is not drifting toward it (lane fit)"}
# Past the pulse's own push by this much is the driver steering through it; still wheels read up to 500 (2026-11-02).
_BEYOND_UNITS = 500.0


class LkaRunner(AssistRunner):
    LABEL = "LKA"
    IDLE = AssistRunner.IDLE | {"standby"}
    DEBUG_NAME = "lka_debug.csv"
    TRACE_NAME = "lka_ffb_debug.csv"

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._closest = math.inf
        self._one_sided_warned = False
        self._pushed = self._against = self._beyond = math.nan
        self._beyond_at = math.nan

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
        if pulse.event == "start":
            self._pushed = self._against = self._beyond = self._beyond_at = math.nan
        if pulse.side:
            self._note_push(pulse.elapsed_s)
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
            logger.info("LKA pulse ends (%s) after %.2f s: closest %.2f m inside the line (negative: past it); %s",
                        pulse.event, pulse.elapsed_s, self._closest, self._push_note())

    def _note_push(self, elapsed_s: float) -> None:
        """Peaks of the pulse's own push and the driver's push against it, and when the driver first went past it."""
        p = self.push
        if not (math.isfinite(p.push) and math.isfinite(p.against)):
            return
        self._pushed = p.push if math.isnan(self._pushed) else max(self._pushed, p.push)
        self._against = p.against if math.isnan(self._against) else max(self._against, p.against)
        self._beyond = p.beyond if math.isnan(self._beyond) else max(self._beyond, p.beyond)
        if math.isnan(self._beyond_at) and p.beyond >= _BEYOND_UNITS:
            self._beyond_at = elapsed_s

    def _push_note(self) -> str:
        if math.isnan(self._pushed):
            return "driver's push not estimated"
        note = f"pushed up to {self._pushed:.0f}, the driver against it up to {self._against:.0f}"
        if math.isnan(self._beyond_at):
            return note + ", never past the pulse's own push"
        return note + f", past the pulse's own push from {self._beyond_at:.2f} s (by up to {self._beyond:.0f})"

    def _tuning(self) -> tuple[tuple, AssistTuning, tuple]:
        arm = _setting("ldw_min_speed_kmh", 60.0)
        strength = max(0.0, min(STRENGTH_MAX, _setting("lka_strength", 1.0)))
        inertia = max(50.0, min(20000.0, _setting("la_wheel_inertia", 0.0) or WheelTuning().inertia))
        design = (_setting("la_bandwidth_hz", 0.0), _setting("la_damping", 0.0),
                  bool(getattr(Settings, "la_damping_first", False)))
        ahead = (_setting("la_ff_inertia", 0.0), _setting("la_ff_damping", 0.0), _setting("la_ff_friction", 0.0),
                 _setting("la_servo_lead_s", 0.0), _setting("la_ff_plan_s", 0.0), _setting("la_ff_breakaway", 0.0))
        learn = bool(getattr(Settings, "la_wheel_learning", True))
        key = (arm, strength, bool(getattr(Settings, "lka_tmp_enabled", False)), inertia, *design, *ahead, learn)
        line = ("LKA settings: from %.0f km/h, strength %.2f, TruckersMP %s, wheel inertia %.0f, servo design %.2f Hz, "
                "damping %.2f, damping first %s, push ahead inertia %.0f, damping %.0f, friction %.0f, lead %.3f s, "
                "plan %.3f s, breakaway %.0f, wheel learning %s",
                *key)
        return key, AssistTuning(
            pulse=True, settle_s=1.0, arm_kmh=arm, disarm_kmh=max(0.0, arm - _ARM_MARGIN_KMH),
            wheel=WheelTuning(strength=strength, inertia=inertia, **design_overrides(*design),
                              **push_ahead_overrides(*ahead), hold=learn, learn=learn), allow_tmp=key[2],
        ), line
