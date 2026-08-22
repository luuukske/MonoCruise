"""What does the CC accel envelope command, and how long does that take?

Offline: imports the real `core.longitudinal.accel_envelope` and integrates it.
No game, no threads, no settings file touched. See tools/README.md.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.longitudinal.accel_envelope import (  # noqa: E402
    HEADROOM_FRAC,
    PROFILES,
    AccelProfile,
    envelope_ms2,
    shape_ceiling_ms2,
)

# Today's flat ceiling, the thing the envelope replaces. Every table prints it
# as the reference column so a regression at any speed is visible at a glance.
LEGACY_CEILING_MS2: float = 1.0

TABLE_SPEEDS_KMH: tuple[int, ...] = (10, 20, 30, 40, 50, 60, 70, 85, 90, 110, 130)
TIME_TARGETS_KMH: tuple[int, ...] = (50, 90)

# Mirrors PedalCapacityTracker.accel_gain_for_gear and AccelToPedals.weight_factor.
# Replicated, not imported, so the probe never pulls in core.settings.
_ANCHOR_GEAR: int = 6
_DEFAULT_RATIO: float = 1.27
# Cold-start seed: shipped pedal_capacity_max_accel_ms2, learned around gear 8
# and projected back to the anchor gear. Mass-normalized, so a rig divides it out.
_DEFAULT_ANCHOR_NORM_MS2: float = 2.124 * _DEFAULT_RATIO ** 2
_REFERENCE_MASS_T: float = 20.0
_WEIGHT_SPAN_T: float = 12.7
_WEIGHT_STRENGTH: float = 0.27
_WEIGHT_MIN_FACTOR: float = 0.55
_WEIGHT_MAX_FACTOR: float = 1.85
_TRAILER_WEIGHT_BIAS: float = 1.02
# Speed at which each gear is engaged, top of the band, from a 12-speed stack.
_GEAR_TOP_KMH: tuple[tuple[int, float], ...] = (
    (1, 8.0), (2, 12.0), (3, 17.0), (4, 23.0), (5, 30.0), (6, 38.0),
    (7, 47.0), (8, 57.0), (9, 68.0), (10, 80.0), (11, 95.0), (12, 200.0),
)

# Rig presets as (mass in tonnes, has_trailer). "ideal" disables the capability
# term entirely, which is the comfort shape on its own.
RIGS: dict[str, tuple[float, bool] | None] = {
    "ideal": None,
    "solo": (8.0, False),
    "midweight": (20.0, True),
    "loaded": (40.0, True),
}


RigSpec = tuple[float, bool] | None


def weight_factor(mass_t: float, has_trailer: bool) -> float:
    factor = 1.0 + ((mass_t - _REFERENCE_MASS_T) / _WEIGHT_SPAN_T) * _WEIGHT_STRENGTH
    factor = max(_WEIGHT_MIN_FACTOR, min(_WEIGHT_MAX_FACTOR, factor))
    if has_trailer:
        factor = min(_WEIGHT_MAX_FACTOR, factor * _TRAILER_WEIGHT_BIAS)
    return factor


def gear_for_speed(speed_kmh: float) -> int:
    for gear, top in _GEAR_TOP_KMH:
        if speed_kmh <= top:
            return gear
    return _GEAR_TOP_KMH[-1][0]


def capacity_ms2(speed_ms: float, rig: tuple[float, bool] | None, ratio: float, anchor: float) -> float | None:
    """Capability at gas=1.0 for this rig at this speed, in m/s2."""
    if rig is None:
        return None
    gear = gear_for_speed(speed_ms * 3.6)
    normalized = anchor * (ratio ** (_ANCHOR_GEAR - gear))
    return normalized / weight_factor(*rig)


def ceiling_ms2(
    speed_ms: float, profile: AccelProfile, rig: RigSpec, ratio: float, anchor: float
) -> float:
    return envelope_ms2(speed_ms, profile, capacity_ms2(speed_ms, rig, ratio, anchor))


def legacy_ceiling_ms2(speed_ms: float, rig: RigSpec, ratio: float, anchor: float) -> float:
    """Today: a flat request that rails the pedal, so the truck delivers what it has."""
    cap = capacity_ms2(speed_ms, rig, ratio, anchor)
    if cap is None:
        return LEGACY_CEILING_MS2
    return min(LEGACY_CEILING_MS2, cap)


def integrate_to_kmh(target_kmh: float, accel_at, dt: float = 0.001, limit_s: float = 600.0) -> float:
    """Integrate dv/dt = accel_at(v) from rest. Flat road, no road load."""
    v, t = 0.0, 0.0
    target_ms = target_kmh / 3.6
    while v < target_ms and t < limit_s:
        a = accel_at(v)
        if a <= 0.0:
            return math.inf
        v += a * dt
        t += dt
    return t if t < limit_s else math.inf


def profile_time_to_kmh(
    target_kmh: float, profile: AccelProfile, rig: RigSpec, ratio: float, anchor: float
) -> float:
    return integrate_to_kmh(target_kmh, lambda v: ceiling_ms2(v, profile, rig, ratio, anchor))


def legacy_time_to_kmh(target_kmh: float, rig: RigSpec, ratio: float, anchor: float) -> float:
    return integrate_to_kmh(target_kmh, lambda v: legacy_ceiling_ms2(v, rig, ratio, anchor))


def _fmt(value: float) -> str:
    return "  n/a" if not math.isfinite(value) else f"{value:5.1f}"


def _rig_caption(name: str, rig: RigSpec, ratio: float, anchor: float) -> str:
    if rig is None:
        return f"rig: {name} (capability term disabled, comfort shape only)"
    mass_t, trailer = rig
    return (
        f"rig: {name} ({mass_t:.0f} t{', trailer' if trailer else ''}, "
        f"weight factor {weight_factor(*rig):.2f}, anchor {anchor:.2f} at gear "
        f"{_ANCHOR_GEAR}, ratio {ratio:.2f})"
    )


def report_text(name: str, rig: RigSpec, ratio: float, anchor: float) -> str:
    lines = [_rig_caption(name, rig, ratio, anchor), f"headroom fraction: {HEADROOM_FRAC:.2f}", ""]

    header = f"{'km/h':>5} {'gear':>5} {'cap':>6} {'today':>6}"
    header += "".join(f"{p.label:>12}" for p in PROFILES)
    lines.append(header)
    for kmh in TABLE_SPEEDS_KMH:
        v = kmh / 3.6
        cap = capacity_ms2(v, rig, ratio, anchor)
        cap_txt = "   n/a" if cap is None else f"{cap:6.2f}"
        row = f"{kmh:>5} {gear_for_speed(v * 3.6):>5} {cap_txt} "
        row += f"{legacy_ceiling_ms2(v, rig, ratio, anchor):>6.2f}"
        for p in PROFILES:
            value = ceiling_ms2(v, p, rig, ratio, anchor)
            bound = "c" if cap is not None and value < shape_ceiling_ms2(v, p) - 1e-9 else " "
            row += f"{value:>11.2f}{bound}"
        lines.append(row)

    lines += ["", "'c' marks a cell where capability binds instead of the comfort shape.", ""]

    lines.append(f"{'profile':>12}" + "".join(f"{f'0-{t}':>8}" for t in TIME_TARGETS_KMH))
    legacy = [_fmt(legacy_time_to_kmh(t, rig, ratio, anchor)) for t in TIME_TARGETS_KMH]
    lines.append(f"{'today':>12}" + "".join(f"{x:>8}" for x in legacy))
    for p in PROFILES:
        row = f"{p.label:>12}"
        for target in TIME_TARGETS_KMH:
            row += f"{_fmt(profile_time_to_kmh(target, p, rig, ratio, anchor)):>8}"
        lines.append(row)
    lines += [
        "",
        "Times are seconds from rest on a flat road with no road load, so they",
        "are optimistic bounds, not predictions.",
    ]
    return "\n".join(lines)


def report_json(name: str, rig: RigSpec, ratio: float, anchor: float) -> dict:
    return {
        "rig": name,
        "mass_t": None if rig is None else rig[0],
        "has_trailer": None if rig is None else rig[1],
        "weight_factor": None if rig is None else round(weight_factor(*rig), 4),
        "anchor_norm_ms2": anchor,
        "ratio_step": ratio,
        "headroom_frac": HEADROOM_FRAC,
        "legacy_ceiling_ms2": LEGACY_CEILING_MS2,
        "legacy_seconds_from_rest": {
            str(t): round(legacy_time_to_kmh(t, rig, ratio, anchor), 2)
            for t in TIME_TARGETS_KMH
        },
        "profiles": {
            p.label: {
                "launch_ms2": p.launch_ms2,
                "knee_kmh": round(p.knee_ms * 3.6, 1),
                "taper_power": p.taper_power,
                "floor_ms2": p.floor_ms2,
                "rise_jerk_ms3": p.rise_jerk_ms3,
                "ceiling_by_kmh": {
                    str(kmh): round(ceiling_ms2(kmh / 3.6, p, rig, ratio, anchor), 4)
                    for kmh in TABLE_SPEEDS_KMH
                },
                "seconds_from_rest": {
                    str(t): round(profile_time_to_kmh(t, p, rig, ratio, anchor), 2)
                    for t in TIME_TARGETS_KMH
                },
            }
            for p in PROFILES
        },
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--rig", default="loaded", choices=sorted(RIGS) + ["all"],
        help="capability model; 'ideal' disables it, 'loaded' is a 40 t rig",
    )
    ap.add_argument(
        "--anchor", type=float, default=_DEFAULT_ANCHOR_NORM_MS2,
        help=f"mass-normalized gain at gear {_ANCHOR_GEAR} (default is the cold-start seed)",
    )
    ap.add_argument("--ratio", type=float, default=_DEFAULT_RATIO, help="per-gear gain step")
    ap.add_argument("--report", default="text", choices=("text", "json"))
    args = ap.parse_args(argv)

    names = sorted(RIGS) if args.rig == "all" else [args.rig]
    render = report_json if args.report == "json" else report_text
    results = [render(name, RIGS[name], args.ratio, args.anchor) for name in names]

    if args.report == "json":
        print(json.dumps(results if len(results) > 1 else results[0], indent=2))
    else:
        print("\n\n".join(results))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
