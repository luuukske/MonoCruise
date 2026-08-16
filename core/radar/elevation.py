"""Shared road-surface elevation gate for AEB and ACC. See core/radar/README.md §15."""

from __future__ import annotations

import math
from dataclasses import dataclass

from .traffic import Vehicle


# Traffic position.y sits this fraction of the body height above its road
# surface; ego coordinateY is the road surface itself (fit: README §15).
BODY_DATUM_FRAC: float = 0.58
# Body height is only trusted inside road-vehicle range before it scales a datum.
_MIN_BODY_HEIGHT_M: float = 0.5
_MAX_BODY_HEIGHT_M: float = 5.0
# Widest plausible body datum, added to the band when size.height is missing.
_UNKNOWN_DATUM_SLACK_M: float = 2.2

# Profile band: base + curvature term. Envelope of the measured predictor error.
_BAND_BASE_M: float = 1.2
_BAND_CURV: float = 0.0011
_BAND_MAX_M: float = 15.0
# Slack proportional to the datum this vehicle's height bought, so a body the
# datum fit does not describe can never be gated on that correction alone.
_BAND_DATUM_FRAC: float = 0.25
# Nothing this far off ego's road plane is ever on it, whatever the model says.
_HARD_CAP_M: float = 20.0

# Vertical-curvature budget: max(floor, near / s^2), 1/m of grade per metre.
_K_FLOOR: float = 0.006
_K_NEAR: float = 6.0
# Below this range the grade test is noise-dominated; the band alone decides.
_MIN_GRADE_RANGE_M: float = 12.0

# Target rotation outside these is not a road grade (crashed, spun, jackknifed).
_MAX_TARGET_ROLL_DEG: float = 15.0
_MAX_TARGET_PITCH_DEG: float = 20.0
# Ego pitch beyond this is not a road at all (measured p100 0.126), so it reads
# level; the EMA below is what absorbs suspension bounce at a fixed 30 Hz.
_MAX_EGO_GRADE: float = 0.18
_GRADE_EMA_ALPHA: float = 0.15

# A heading this far off ego's axis carries no along-axis grade evidence.
_MIN_GRADE_ALIGN_COS: float = 0.30

# Fallback: the target's own tangent run back to ego, immune to ego pitch. Only
# valid on an observed target grade, or it is just the ego tangent again.
_FB_BASE_M: float = 2.0
_FB_GRADE: float = 0.03

# Ego elevation history: distance grid, quadratic fit, clamped forward term.
_FIT_SPAN_M: float = 45.0
_FIT_MIN_SPAN_M: float = 18.0
_FIT_STEP_M: float = 0.5
_FIT_MIN_SAMPLES: int = 6
_FIT_MAX_RMS_M: float = 0.15
_CURV_TERM_CLAMP_M: float = 2.0
_MAX_VERT_CURVATURE: float = 0.01

# Suppression needs this many consecutive failing frames unless it is gross.
SUPPRESS_CONFIRM_FRAMES: int = 3
_GROSS_RATIO: float = 1.7


@dataclass(frozen=True)
class RoadSurface:
    """Ego's road plane plus the vertical curvature its recent path implies."""

    ego_y: float = 0.0
    grade: float = 0.0            # dy/ds along ego forward
    curvature: float = 0.0        # d(grade)/ds, 0 when the fit is unusable
    curvature_ok: bool = False

    def predict(self, s: float) -> float:
        """Road height above ego at forward range ``s`` (signed)."""
        y = self.grade * s
        if self.curvature_ok:
            term = 0.5 * self.curvature * s * s
            y += max(-_CURV_TERM_CLAMP_M, min(_CURV_TERM_CLAMP_M, term))
        return y


class EgoElevationTrack:
    """Ego (arc length, elevation) samples on a distance grid, for the fit."""

    __slots__ = ("_samples", "_last_x", "_last_z", "_s", "_grade")

    def __init__(self) -> None:
        self._samples: list[tuple[float, float]] = []
        self._last_x: float | None = None
        self._last_z: float | None = None
        self._s: float = 0.0
        self._grade: float | None = None

    def clear(self) -> None:
        self._samples.clear()
        self._last_x = None
        self._last_z = None
        self._s = 0.0
        self._grade = None

    def smooth_grade(self, raw: float) -> float:
        """EMA of ego grade. A truck pitches on its suspension over bumps and
        level crossings; the road under it does not."""
        if self._grade is None:
            self._grade = raw
        else:
            self._grade += _GRADE_EMA_ALPHA * (raw - self._grade)
        return self._grade

    def push(self, x: float, z: float, y: float) -> None:
        if self._last_x is None:
            self._samples.append((0.0, y))
            self._last_x, self._last_z = x, z
            return
        step = math.hypot(x - self._last_x, z - self._last_z)
        if step < _FIT_STEP_M:
            return
        # A teleport (ferry, fast travel) invalidates the whole window.
        if step > _FIT_SPAN_M:
            self.clear()
            self.push(x, z, y)
            return
        self._s += step
        self._samples.append((self._s, y))
        self._last_x, self._last_z = x, z
        while len(self._samples) > 2 and self._s - self._samples[0][0] > _FIT_SPAN_M:
            self._samples.pop(0)

    def curvature(self) -> tuple[float, bool]:
        """Vertical curvature (1/m) from a quadratic LS fit; (0.0, False) if unusable."""
        pts = self._samples
        n = len(pts)
        if n < _FIT_MIN_SAMPLES:
            return 0.0, False
        s_end, y_end = pts[-1]
        if s_end - pts[0][0] < _FIT_MIN_SPAN_M:
            return 0.0, False
        fit = _quadratic_fit([(s - s_end, y - y_end) for s, y in pts])
        if fit is None:
            return 0.0, False
        c2, rms = fit
        if rms > _FIT_MAX_RMS_M:
            return 0.0, False
        kappa = 2.0 * c2
        if not math.isfinite(kappa) or abs(kappa) > _MAX_VERT_CURVATURE:
            return 0.0, False
        return kappa, True


def _quadratic_fit(pts: list[tuple[float, float]]) -> tuple[float, float] | None:
    """LS ``y = c0 + c1 s + c2 s^2``; returns ``(c2, rms)`` or None when singular."""
    n = len(pts)
    mom = [0.0] * 5
    rhs = [0.0, 0.0, 0.0]
    for s, y in pts:
        p = 1.0
        for k in range(5):
            mom[k] += p
            p *= s
        rhs[0] += y
        rhs[1] += y * s
        rhs[2] += y * s * s
    aug = [
        [mom[0], mom[1], mom[2], rhs[0]],
        [mom[1], mom[2], mom[3], rhs[1]],
        [mom[2], mom[3], mom[4], rhs[2]],
    ]
    for i in range(3):
        piv = max(range(i, 3), key=lambda r: abs(aug[r][i]))
        if abs(aug[piv][i]) < 1e-12:
            return None
        aug[i], aug[piv] = aug[piv], aug[i]
        for r in range(3):
            if r == i:
                continue
            f = aug[r][i] / aug[i][i]
            for c in range(i, 4):
                aug[r][c] -= f * aug[i][c]
    coef = [aug[i][3] / aug[i][i] for i in range(3)]
    err = sum((coef[0] + coef[1] * s + coef[2] * s * s - y) ** 2 for s, y in pts)
    return coef[2], math.sqrt(err / n)


def build_surface(
    ego_y: float, ego_pitch_rad: float, track: EgoElevationTrack | None,
) -> RoadSurface:
    """Ego road plane for this frame. Pitch is the grade; history gives curvature."""
    grade = math.tan(ego_pitch_rad) if math.isfinite(ego_pitch_rad) else 0.0
    if not math.isfinite(grade) or abs(grade) > _MAX_EGO_GRADE:
        grade = 0.0
    kappa, ok = (0.0, False)
    if track is not None:
        grade = track.smooth_grade(grade)
        kappa, ok = track.curvature()
    return RoadSurface(ego_y=ego_y, grade=grade, curvature=kappa, curvature_ok=ok)


def road_y_offset(v: Vehicle, ego_y: float) -> tuple[float, float]:
    """Target road surface minus ego road surface, and the body height it used.

    A height of 0.0 means the datum could not be applied; the caller widens the
    band by the full plausible datum instead."""
    h = float(getattr(v.size, "height", 0.0) or 0.0)
    if not math.isfinite(h) or h < _MIN_BODY_HEIGHT_M:
        return v.position.y - ego_y, 0.0
    h = min(h, _MAX_BODY_HEIGHT_M)
    return v.position.y - BODY_DATUM_FRAC * h - ego_y, h


def target_grade(
    v: Vehicle, ego_grade: float, yaw_diff_rad: float,
) -> tuple[float, bool]:
    """Road grade under the target along ego forward, and whether it was observed.

    Traffic euler pitch is the negated nose-up angle along the target's own
    heading, so only its ego-axis component is evidence (README §15). Returns
    ego's own grade, unobserved, whenever that evidence is missing."""
    if v.rotation.is_zero():
        return ego_grade, False
    pitch_deg, _, roll_deg = v.rotation.euler()
    if not (math.isfinite(pitch_deg) and math.isfinite(roll_deg)):
        return ego_grade, False
    if abs(roll_deg) > _MAX_TARGET_ROLL_DEG or abs(pitch_deg) > _MAX_TARGET_PITCH_DEG:
        return ego_grade, False
    c = math.cos(yaw_diff_rad)
    sn = math.sin(yaw_diff_rad)
    grade = ego_grade * sn * sn - math.tan(math.radians(pitch_deg)) * c
    return grade, abs(c) >= _MIN_GRADE_ALIGN_COS


def profile_band(s: float, body_height: float) -> float:
    """Half-width of the plausible road-height band at forward range ``s``."""
    a = abs(s)
    band = min(_BAND_BASE_M + _BAND_CURV * a * a, _BAND_MAX_M)
    if body_height <= 0.0:
        return band + _UNKNOWN_DATUM_SLACK_M
    return band + _BAND_DATUM_FRAC * body_height


def required_curvature(s: float, dy: float, m0: float, m1: float) -> float:
    """Peak |d(grade)/ds| of the cubic joining (0, 0, m0) to (s, dy, m1)."""
    if abs(s) < 1.0:
        return 0.0
    dev = dy - m0 * s
    dm = m1 - m0
    return max(abs(6.0 * dev / (s * s) - 2.0 * dm / s),
               abs(4.0 * dm / s - 6.0 * dev / (s * s)))


def max_curvature(s: float) -> float:
    """Vertical-curvature budget a real road may use over forward range ``s``."""
    a = max(abs(s), 1.0)
    return max(_K_FLOOR, _K_NEAR / (a * a))


@dataclass(frozen=True)
class ElevationVerdict:
    """Per-vehicle outcome. ``off_surface`` is the raw test, before persistence."""

    off_surface: bool = False
    gross: bool = False
    dy: float = 0.0
    residual: float = 0.0
    band: float = 0.0
    k_req: float = 0.0
    k_max: float = 0.0


_PASS = ElevationVerdict()


def fallback_window(s: float) -> float:
    """Window for the target-perspective tangent check."""
    return _FB_BASE_M + _FB_GRADE * abs(s)


def evaluate_vehicle(
    v: Vehicle, surface: RoadSurface, s: float, yaw_diff_rad: float,
) -> ElevationVerdict:
    """Is this vehicle off ego's road surface? See core/radar/README.md §15."""
    dy, body_height = road_y_offset(v, surface.ego_y)
    if not math.isfinite(dy):
        return _PASS
    if abs(dy) > _HARD_CAP_M:
        return ElevationVerdict(True, True, dy, abs(dy), 0.0, 0.0, 0.0)

    band = profile_band(s, body_height)
    residual = abs(dy - surface.predict(s))
    m1, m1_seen = target_grade(v, surface.grade, yaw_diff_rad)
    if residual > band:
        if m1_seen and _fallback_holds(dy, m1, s):
            return ElevationVerdict(False, False, dy, residual, band, 0.0, 0.0)
        return ElevationVerdict(
            True, residual > band * _GROSS_RATIO, dy, residual, band, 0.0, 0.0,
        )

    if abs(s) < _MIN_GRADE_RANGE_M:
        return ElevationVerdict(False, False, dy, residual, band, 0.0, 0.0)

    k_req = required_curvature(s, dy, surface.grade, m1)
    k_max = max_curvature(s)
    if k_req > k_max and not (m1_seen and _fallback_holds(dy, m1, s)):
        return ElevationVerdict(
            True, k_req > k_max * _GROSS_RATIO, dy, residual, band, k_req, k_max,
        )
    return ElevationVerdict(False, False, dy, residual, band, k_req, k_max)


def _fallback_holds(dy: float, m1: float, s: float) -> bool:
    """Target's own road tangent, run back to ego, lands on ego's road.

    Independent of ego pitch, which is what spikes over level crossings and
    crests. See core/radar/README.md §15."""
    return abs(dy - m1 * s) <= fallback_window(s)


class ElevationGate:
    """Per-id persistence around ``evaluate_vehicle``: marginal failures must repeat."""

    __slots__ = ("_strikes",)

    def __init__(self) -> None:
        self._strikes: dict[int, int] = {}

    def clear(self) -> None:
        self._strikes.clear()

    def step(
        self,
        vehicles: list[Vehicle],
        surface: RoadSurface,
        ego_x: float,
        ego_z: float,
        ego_yaw_rad: float,
    ) -> frozenset[int]:
        """Ids that are confirmed off ego's road surface this frame."""
        sin_y = math.sin(ego_yaw_rad)
        cos_y = math.cos(ego_yaw_rad)
        strikes = self._strikes
        seen: dict[int, int] = {}
        off: set[int] = set()
        for v in vehicles:
            vid = v.id
            s = (v.position.x - ego_x) * sin_y + (v.position.z - ego_z) * cos_y
            yaw_diff = ego_yaw_rad - _vehicle_yaw_rad(v)
            verdict = evaluate_vehicle(v, surface, s, yaw_diff)
            if not verdict.off_surface:
                seen[vid] = 0
                continue
            if verdict.gross:
                seen[vid] = SUPPRESS_CONFIRM_FRAMES
                off.add(vid)
                continue
            n = strikes.get(vid, 0) + 1
            seen[vid] = n
            if n >= SUPPRESS_CONFIRM_FRAMES:
                off.add(vid)
        self._strikes = seen
        return frozenset(off)


def _vehicle_yaw_rad(v: Vehicle) -> float:
    if v._smooth_yaw is not None:
        return v._smooth_yaw
    return math.radians(v.rotation.euler()[1])
