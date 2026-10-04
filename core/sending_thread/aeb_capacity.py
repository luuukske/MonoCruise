"""Per-truck AEB brake capacity: firm braking for the scale, full-pedal stops for the
top of the travel. See README.md."""

from __future__ import annotations

import logging
import math
import time

from core.scs_profile.intensity import TUNE_BRAKE_INTENSITY, clamp_brake_intensity
from core.settings import Settings

from .accel_to_pedals import brake_curve_fraction, brake_curve_pedal

logger = logging.getLogger(__name__)

_F1: float = brake_curve_fraction(1.0)
# A truck AEB has not measured yet plans slightly short of the model.
_PRIOR_SCALE: float = 0.95
_SCALE_MIN: float = 0.35
_SCALE_MAX: float = 1.00
# Light braking reads 30-50% low through the brake curve, so only firm samples teach.
_FIRM_TUNE_PEDAL: float = 0.5
_FULL_SENT_PEDAL: float = 0.9
_FULL_PEDAL: float = 0.999
_ALPHA: float = 0.04
_MAX_TRUCKS: int = 64
_SAVE_THRESHOLD: float = 0.02
_SAVE_COOLDOWN_S: float = 30.0
# Full-pedal stops: a sent 1.0 held long enough to settle, kept per load within 10%.
FULL_SENT: float = 0.97
_FULL_SETTLE_S: float = 0.3
_FULL_RUN_MIN_S: float = 0.5
_FULL_MIN_SPEED_MS: float = 5.0
_FULL_MIN_SAMPLES: int = 8
_FULL_MAX_SAMPLES: int = 4000
_FULL_MASS_TOL: float = 0.10
_FULL_KEEP_STOPS: int = 5
_FULL_LOADS_PER_KEY: int = 4
_FULL_MIN_STOPS: int = 2
_FULL_RATIO_MIN: float = 0.3
_FULL_RATIO_MAX: float = 2.5
# The room a clean 100% user has below full pedal; a measured full pedal keeps it.
_FULL_PEDAL_ROOM: float = 1.1


def truck_key(game: str | None, truck_id: str, trailer_count: int) -> str:
    """One learned scale per game, truck model and trailer count."""
    return f"{game or '?'}|{truck_id or '?'}|{max(0, int(trailer_count))}"


def full_pedal_key(truck: str, intensity: float | None) -> str:
    """Full-pedal stops also depend on the slider: the extra above 100% varies by rig."""
    return f"{truck}|{clamp_brake_intensity(intensity):.2f}"


def aeb_capacity_ms2(
    baseline_ms2: float,
    scale: float,
    intensity: float | None,
    full_ratio: float | None = None,
) -> float:
    """Decel asymptote AEB plans with. Unmeasured: the lower of the slider cut and the
    learner's full-pedal reading, never above scale x baseline. A measured full pedal
    (`full_ratio`, decel / (baseline x frac(1))) raises it to that less the usual room."""
    base = max(float(baseline_ms2), 0.0)
    share = min(clamp_brake_intensity(intensity) / TUNE_BRAKE_INTENSITY, 1.0)
    learned = scale * brake_curve_fraction(share) / _F1
    capacity = base * min(learned, share)
    if full_ratio is None:
        return capacity
    return max(capacity, base * full_ratio / _FULL_PEDAL_ROOM)


def aeb_pedal_gain(intensity: float | None) -> float:
    """How much harder a sent pedal brakes than the tune at this slider; 1 at or below it."""
    return max(clamp_brake_intensity(intensity) / TUNE_BRAKE_INTENSITY, 1.0)


class AebPedalAxis:
    """AEB's pedal, a point on its capacity's brake curve, on the sent axis and back.

    Identity at or below the tune slider. Above it the tune range runs through the remap,
    and a measured full pedal spreads the extra linearly over the rest of the travel.
    """

    def __init__(
        self,
        intensity: float | None,
        scale: float,
        capacity_ratio: float,
        full_ratio: float | None = None,
    ) -> None:
        self._gain = aeb_pedal_gain(intensity)
        self._scale = max(float(scale), 1e-3)
        self._capacity = max(float(capacity_ratio), 1e-3)
        self._tune_top = self._scale * _F1
        self._full = (full_ratio * _F1
                      if full_ratio is not None and full_ratio > self._scale else None)
        self._tune_end = 1.0 / self._gain

    def sent(self, pedal: float) -> float:
        if pedal >= _FULL_PEDAL:
            return 1.0
        p = max(float(pedal), 0.0)
        if self._gain <= 1.0:
            return p
        want = self._capacity * brake_curve_fraction(p)
        if want <= self._tune_top or self._full is None:
            return brake_curve_pedal(want / self._scale) * self._tune_end
        share = (want - self._tune_top) / (self._full - self._tune_top)
        return min(self._tune_end + share * (1.0 - self._tune_end), 1.0)

    def applied(self, sent: float) -> float:
        """The sent pedal back in AEB's units: what its plant model sees."""
        s = min(max(float(sent), 0.0), 1.0)
        if self._gain <= 1.0:
            return s
        if s <= self._tune_end:
            decel = self._scale * brake_curve_fraction(s * self._gain)
        elif self._full is None:
            decel = self._tune_top
        else:
            share = (s - self._tune_end) / (1.0 - self._tune_end)
            decel = self._tune_top + share * (self._full - self._tune_top)
        return brake_curve_pedal(decel / self._capacity)


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


def _valid_ratio(value: object) -> float | None:
    ratio = _finite(value)
    return ratio if _FULL_RATIO_MIN <= ratio <= _FULL_RATIO_MAX else None


class FullPedalRun:
    """One full-pedal stop, measured off its speed trace when the pedal comes off.

    A least-squares slope over the settled part, not the differentiated decel: a slam's
    20 Hz physics staircase swings that by +-2 m/s2 and failed every settle gate.
    """

    def __init__(self) -> None:
        self._reset()

    def _reset(self) -> None:
        self._key: str | None = None
        self._mass_kg = 0.0
        self._baseline_ms2 = 0.0
        self._start: float | None = None
        self._samples: list[tuple[float, float, float]] = []

    def tick(
        self,
        full_sent: bool,
        key: str,
        mass_kg: float,
        baseline_ms2: float,
        speed_ms: float,
        road_load_ms2: float,
        now: float,
    ) -> tuple[str, float, float] | None:
        """Returns (key, mass_kg, decel / (baseline x frac(1))) on the tick a stop ends."""
        if full_sent:
            if self._start is None:
                self._key, self._mass_kg, self._start = key, mass_kg, now
                self._baseline_ms2 = baseline_ms2
            settled = now - self._start >= _FULL_SETTLE_S
            if (key == self._key and settled and speed_ms >= _FULL_MIN_SPEED_MS
                    and len(self._samples) < _FULL_MAX_SAMPLES):
                self._samples.append((now, speed_ms, road_load_ms2))
            return None
        done = self._measure()
        self._reset()
        return done

    def _measure(self) -> tuple[str, float, float] | None:
        samples = self._samples
        if (self._key is None or len(samples) < _FULL_MIN_SAMPLES
                or samples[-1][0] - samples[0][0] < _FULL_RUN_MIN_S
                or self._baseline_ms2 <= 0.0):
            return None
        n = len(samples)
        mean_t = sum(t for t, _, _ in samples) / n
        mean_v = sum(v for _, v, _ in samples) / n
        spread = sum((t - mean_t) ** 2 for t, _, _ in samples)
        if spread <= 0.0:
            return None
        slope = sum((t - mean_t) * (v - mean_v) for t, v, _ in samples) / spread
        road = sum(r for _, _, r in samples) / n
        ratio = _valid_ratio((-slope - road) / (self._baseline_ms2 * _F1))
        return None if ratio is None else (self._key, self._mass_kg, ratio)


def _closest_load(loads: list, mass_kg: float) -> list | None:
    best = None
    for entry in loads:
        err = abs(mass_kg - entry[0]) / entry[0]
        if err <= _FULL_MASS_TOL and (best is None or err < best[0]):
            best = (err, entry)
    return None if best is None else best[1]


class FullPedalStore:
    """Measured full-pedal stops per truck and slider, per load. `Settings.aeb_full_pedal`."""

    def __init__(self) -> None:
        self._entries: dict[str, list] = {}

    def load_persisted(self) -> None:
        raw = getattr(Settings, "aeb_full_pedal", None)
        self._entries = {}
        if not isinstance(raw, dict):
            return
        for key, loads in list(raw.items())[-_MAX_TRUCKS:]:
            if not isinstance(key, str) or not isinstance(loads, list):
                continue
            clean = []
            for item in loads[-_FULL_LOADS_PER_KEY:]:
                if not isinstance(item, list) or len(item) != 2 or not isinstance(item[1], list):
                    continue
                mass = _finite(item[0])
                ratios = [r for r in (_valid_ratio(x) for x in item[1]) if r is not None]
                if mass > 0.0 and ratios:
                    clean.append([mass, ratios[-_FULL_KEEP_STOPS:]])
            if clean:
                self._entries[key] = clean

    def stops(self, key: str, mass_kg: float) -> int:
        entry = _closest_load(self._entries.get(key, []), mass_kg)
        return 0 if entry is None else len(entry[1])

    def ratio(self, key: str, mass_kg: float) -> float | None:
        """Weakest of this load's recent stops, once there are enough of them. Stops on
        one truck and slider spread +-12% between drives, as much as the room."""
        entry = _closest_load(self._entries.get(key, []), mass_kg)
        if entry is None or len(entry[1]) < _FULL_MIN_STOPS:
            return None
        return min(entry[1])

    def observe_stop(self, key: str, mass_kg: float, ratio: float) -> None:
        mass = _finite(mass_kg)
        valid = _valid_ratio(ratio)
        if mass <= 0.0 or valid is None:
            return
        # Re-inserting keeps keys and loads in least-recently-used order for the caps.
        loads = self._entries.pop(key, [])
        entry = _closest_load(loads, mass)
        if entry is None:
            entry = [mass, []]
        else:
            loads.remove(entry)
        loads.append(entry)
        entry[1].append(valid)
        del entry[1][:-_FULL_KEEP_STOPS]
        del loads[:-_FULL_LOADS_PER_KEY]
        self._entries[key] = loads
        while len(self._entries) > _MAX_TRUCKS:
            self._entries.pop(next(iter(self._entries)))
        snapshot = {
            k: [[round(m, 1), [round(r, 4) for r in rs]] for m, rs in v]
            for k, v in self._entries.items()
        }
        try:
            Settings.save(values={"aeb_full_pedal": snapshot})
        except Exception:
            logger.debug("aeb_full_pedal save failed", exc_info=True)
