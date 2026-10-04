"""Per-truck AEB brake capacity, learned from firm braking only. See README.md."""

from __future__ import annotations

import logging
import math
import time

from core.scs_profile.intensity import TUNE_BRAKE_INTENSITY, clamp_brake_intensity
from core.settings import Settings

from .accel_to_pedals import brake_curve_fraction

logger = logging.getLogger(__name__)

# A truck AEB has not measured yet plans slightly short of the model.
_PRIOR_SCALE: float = 0.95
_SCALE_MIN: float = 0.35
_SCALE_MAX: float = 1.00
# Light braking reads 30-50% low through the brake curve, so only firm samples teach.
_FIRM_TUNE_PEDAL: float = 0.5
_FULL_SENT_PEDAL: float = 0.9
_ALPHA: float = 0.04
_MAX_TRUCKS: int = 64
_SAVE_THRESHOLD: float = 0.02
_SAVE_COOLDOWN_S: float = 30.0


def truck_key(game: str | None, truck_id: str, trailer_count: int) -> str:
    """One learned scale per game, truck model and trailer count."""
    return f"{game or '?'}|{truck_id or '?'}|{max(0, int(trailer_count))}"


def aeb_capacity_ms2(baseline_ms2: float, scale: float, intensity: float | None) -> float:
    """Physical decel at pedal 1.0 for AEB. The lower of the slider cut and the learner's
    own full-pedal reading, so a low slider is not counted twice. Never above scale x baseline."""
    share = min(clamp_brake_intensity(intensity) / TUNE_BRAKE_INTENSITY, 1.0)
    learned = scale * brake_curve_fraction(share) / brake_curve_fraction(1.0)
    return max(float(baseline_ms2), 0.0) * min(learned, share)


def _sent_share(tune_pedal: float, intensity: float | None) -> float:
    """Sent pedal behind a tune-unit reading, capped at 1."""
    return min(tune_pedal * TUNE_BRAKE_INTENSITY / clamp_brake_intensity(intensity), 1.0)


def is_firm_sample(tune_pedal: float, intensity: float | None) -> bool:
    """Firm braking, or a full sent pedal on a slider too low to reach the firm band."""
    return (tune_pedal >= _FIRM_TUNE_PEDAL
            or _sent_share(tune_pedal, intensity) >= _FULL_SENT_PEDAL)


def _finite(value: object) -> float:
    try:
        result = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return 0.0
    return result if math.isfinite(result) else 0.0


class AebCapacityStore:
    """Learned AEB brake scale per truck key, persisted in `Settings.aeb_brake_scales`."""

    def __init__(self) -> None:
        self._scales: dict[str, float] = {}
        self._saved: dict[str, float] = {}
        self._last_save_mono: float = 0.0

    def load_persisted(self) -> None:
        raw = getattr(Settings, "aeb_brake_scales", None)
        self._scales = {}
        if isinstance(raw, dict):
            for key, value in list(raw.items())[-_MAX_TRUCKS:]:
                scale = _finite(value)
                if isinstance(key, str) and scale > 0.0:
                    self._scales[key] = min(max(scale, _SCALE_MIN), _SCALE_MAX)
        self._saved = dict(self._scales)

    def scale(self, key: str) -> float:
        return self._scales.get(key, _PRIOR_SCALE)

    def is_measured(self, key: str) -> bool:
        return key in self._scales

    def observe(
        self,
        key: str,
        candidate_scale: float,
        tune_pedal: float,
        intensity: float | None,
        now: float | None = None,
    ) -> None:
        """Feed one settled braking sample from `PedalCapacityTracker`. Light ones are ignored."""
        if not is_firm_sample(tune_pedal, intensity):
            return
        candidate = _finite(candidate_scale)
        if candidate <= 0.0:
            return
        candidate = min(max(candidate, _SCALE_MIN), _SCALE_MAX)
        firmness = max(min(tune_pedal, 1.0), _sent_share(tune_pedal, intensity))
        # Re-inserting keeps the dict in least-recently-used order for the size cap.
        current = self._scales.pop(key, _PRIOR_SCALE)
        self._scales[key] = current + _ALPHA * firmness ** 3 * (candidate - current)
        while len(self._scales) > _MAX_TRUCKS:
            self._scales.pop(next(iter(self._scales)))
        self._maybe_save(time.monotonic() if now is None else now)

    def _maybe_save(self, now: float) -> None:
        if self._last_save_mono > 0.0 and now - self._last_save_mono < _SAVE_COOLDOWN_S:
            return
        changed = any(
            abs(value - self._saved.get(key, 0.0)) > _SAVE_THRESHOLD * value
            for key, value in self._scales.items()
        )
        if not changed:
            return
        snapshot = {key: round(value, 4) for key, value in self._scales.items()}
        try:
            Settings.save(values={"aeb_brake_scales": snapshot})
        except Exception:
            logger.debug("aeb_brake_scales save failed", exc_info=True)
            return
        self._saved = dict(self._scales)
        self._last_save_mono = now
        logger.debug("aeb_brake_scales saved: %d truck(s)", len(snapshot))
