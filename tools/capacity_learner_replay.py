"""Replay accel_to_pedals_debug.csv into PedalCapacityTracker.update_accel, offline.

The log is written at about 10 Hz and rows are linearly interpolated to `--hz` before they
are fed in; the learner's rates are per second, so `--hz` only sets the interpolation. The
mapper log only covers ticks the mapper commanded: `--csv gas_learn_debug.csv` (debug builds)
replays every tick the learner saw, the driver's own pedal included, with its mass and trailer.
On the mapper log mass comes from `gain_scale` (20 t / mass) and every row is treated as solo.
Saves are disabled, so the live config.json is never touched. See core/sending_thread/README.md.

`--bands` adds the online prediction error per mass band: on each logged pull (gas at least
0.35, accel over 0.4 m/s2, clutch closed) the measured authority against what the learner
believed just before it, and the same on pulls taken after a mass change of over 1.4x since
the anchor last learned, which is what the mass law alone decides. `--legacy-mass-law` replays
the pre-2026-10-21 linear `weight_factor`.
"""

from __future__ import annotations

import argparse
import csv
import math
import statistics
import sys
import types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import core.sending_thread.accel_to_pedals as atp  # noqa: E402
import core.sending_thread.pedal_capacity as pc  # noqa: E402

_BANDS_T = (0.0, 13.0, 19.0, 27.0, 35.0, 45.0, 1000.0)
_PULL_MIN_GAS = 0.35
_PULL_MIN_ACCEL_MS2 = 0.4
_TRANSFER_MASS_RATIO = 1.4

_FIELDS = ("t_s", "speed_ms", "gear", "gas_cmd", "raw_ms2", "road_load_ms2",
           "slope_rad", "game_clutch", "gain_scale", "est_accel_ms2")


def _lerp(a: dict, b: dict, w: float, fields: tuple[str, ...]) -> dict:
    return {k: a[k] + w * (b[k] - a[k]) for k in fields}


def _band(mass_t: float) -> str:
    for lo, hi in zip(_BANDS_T, _BANDS_T[1:]):
        if lo <= mass_t < hi:
            return f"{lo:.0f}-{hi:.0f} t"
    return "?"


def _pull_error(tr, row: dict, mass: float) -> float | None:
    """log(measured / believed) authority on a clean pull, or None when the row is not one."""
    gear, gas = int(row["gear"]), row["gas_cmd"]
    if (gear <= 0 or gas < _PULL_MIN_GAS or row["raw_ms2"] < _PULL_MIN_ACCEL_MS2
            or row["game_clutch"] > 0.05 or abs(row["slope_rad"]) > 0.10):
        return None
    measured = row["raw_ms2"] + row["road_load_ms2"] - tr._accel_zero_offset_ms2
    believed = tr.accel_slope_for_gear(gear) / max(weight_factor(mass, False), 1e-6) * gas
    if measured <= 0.05 or believed <= 0.0:
        return None
    return math.log(measured / believed)


def weight_factor(mass: float, has_trailer: bool) -> float:
    return pc.weight_factor(mass, has_trailer)


def replay(path: Path, date: str, anchor: float, ratio: float, offset: float, hz: float):
    clock = [0.0]
    pc.time = types.SimpleNamespace(monotonic=lambda: clock[0])
    pc.PedalCapacityTracker._maybe_save = lambda self, now: None
    tr = pc.PedalCapacityTracker()
    tr._accel_anchor_gain_ms2, tr._accel_ratio_step, tr._accel_zero_offset_ms2 = anchor, ratio, offset
    trace, updates, last_anchor, prev = [], 0, anchor, None
    errors: dict[str, list[float]] = {}
    learned_mass = None
    with path.open(newline="") as f:
        reader = csv.reader(f)
        hdr = next(reader)
        ix = {h: i for i, h in enumerate(hdr)}
        # The gas learning log carries mass and trailer; the mapper log only `gain_scale`.
        own_log = "mass_kg" in ix
        extra = ("mass_kg", "has_trailer") if own_log else ("gain_scale",)
        fields = tuple(k for k in _FIELDS if k != "gain_scale") + extra
        for row in reader:
            if len(row) != len(hdr) or not row[ix["utc"]].startswith(date):
                continue
            try:
                cur = {k: float(row[ix[k]]) for k in fields}
            except ValueError:
                continue
            cur["utc"] = row[ix["utc"]][11:19]
            if prev is None or not 0.0 < cur["t_s"] - prev["t_s"] < 1.0:
                prev = cur
                continue
            mass = cur["mass_kg"] if own_log else 20_000.0 / max(cur["gain_scale"], 0.1)
            trailer = own_log and cur["has_trailer"] > 0.5
            err = _pull_error(tr, cur, mass)
            if err is not None:
                errors.setdefault(_band(mass / 1000.0), []).append(err)
                if learned_mass and abs(math.log(mass / learned_mass)) > math.log(_TRANSFER_MASS_RATIO):
                    kind = "lighter" if mass < learned_mass else "heavier"
                    errors.setdefault(f"after change, {kind}", []).append(err)
            steps = max(1, round((cur["t_s"] - prev["t_s"]) * hz))
            for k in range(steps):
                at = _lerp(prev, cur, (k + 1) / steps, fields)
                clock[0] = at["t_s"]
                if at["gas_cmd"] > 0.01:
                    tr.update_accel(at["gas_cmd"], max(0.0, at["raw_ms2"]), at["speed_ms"],
                                    at["slope_rad"], at["game_clutch"], int(cur["gear"]), mass,
                                    trailer, road_load_ms2=at["road_load_ms2"])
            if tr._accel_anchor_gain_ms2 != last_anchor:
                updates, last_anchor = updates + 1, tr._accel_anchor_gain_ms2
                learned_mass = mass
            gear = int(cur["gear"])
            if gear > 0:
                est = tr.accel_gain_for_gear(gear, mass, trailer) / weight_factor(mass, trailer)
                trace.append((cur["utc"], gear, est, cur["est_accel_ms2"]))
            prev = cur
    return tr, trace, updates, errors


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("date", help="UTC day (or prefix: 2026-10 for a month) to replay")
    ap.add_argument("--csv", default="accel_to_pedals_debug.csv")
    ap.add_argument("--anchor", type=float, default=2.164, help="starting slope anchor at gear 6")
    ap.add_argument("--ratio", type=float, default=1.073)
    ap.add_argument("--offset", type=float, default=0.694)
    ap.add_argument("--hz", type=float, default=100.0, help="interpolation rate, the live loop's")
    ap.add_argument("--trace-every", type=int, default=0, help="print every Nth high-gear row")
    ap.add_argument("--bands", action="store_true", help="print prediction error per mass band")
    ap.add_argument("--legacy-mass-law", action="store_true",
                    help="replay with the pre-2026-10-21 linear weight_factor")
    args = ap.parse_args(argv)
    if args.legacy_mass_law:
        pc.weight_factor = atp.legacy_weight_factor
    tr, trace, updates, errors = replay(Path(args.csv), args.date, args.anchor, args.ratio, args.offset, args.hz)
    high = [e for _, g, e, _ in trace if g >= 10]
    lo = min(high) if high else float("nan")
    spread = max(high) / lo if high else float("nan")
    print(f"{args.date}: updates {updates}, high-gear estimate min {lo:.2f} spread {spread:.2f}x, "
          f"end anchor {tr._accel_anchor_gain_ms2:.3f} ratio {tr._accel_ratio_step:.4f} "
          f"offset {tr._accel_zero_offset_ms2:.3f}")
    if args.bands:
        for band in sorted(errors, key=lambda b: (b.startswith("after"), b if b.startswith("after") else float(b.split("-")[0]))):
            e = sorted(errors[band])
            q = statistics.quantiles(e, n=4) if len(e) >= 4 else [e[0], e[len(e) // 2], e[-1]]
            print(f"  {band:>22}: n {len(e):6d}  measured/believed median {math.exp(q[1]):.2f} "
                  f"(p25 {math.exp(q[0]):.2f}, p75 {math.exp(q[2]):.2f})")
    if args.trace_every > 0:
        for utc, gear, est, live in [r for r in trace if r[1] >= 10][::args.trace_every]:
            print(f"  {utc} gear {gear:2d} replay {est:.2f} live {live:.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
