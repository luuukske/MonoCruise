"""Replay accel_to_pedals_debug.csv into PedalCapacityTracker.update_accel, offline.

The log is written at about 10 Hz; the sending thread learns at the polling rate, so rows
are linearly interpolated to `--hz` before they are fed in. Mass comes from `gain_scale`
(20 t / mass) and every row is treated as solo, which the log does not record. Saves are
disabled, so the live config.json is never touched. See core/sending_thread/README.md.
"""

from __future__ import annotations

import argparse
import csv
import sys
import types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import core.sending_thread.pedal_capacity as pc  # noqa: E402
from core.sending_thread.accel_to_pedals import weight_factor  # noqa: E402

_FIELDS = ("t_s", "speed_ms", "gear", "gas_cmd", "raw_ms2", "road_load_ms2",
           "slope_rad", "game_clutch", "gain_scale", "est_accel_ms2")


def _lerp(a: dict, b: dict, w: float) -> dict:
    return {k: a[k] + w * (b[k] - a[k]) for k in _FIELDS}


def replay(path: Path, date: str, anchor: float, ratio: float, offset: float, hz: float):
    clock = [0.0]
    pc.time = types.SimpleNamespace(monotonic=lambda: clock[0])
    pc.PedalCapacityTracker._maybe_save = lambda self, now: None
    tr = pc.PedalCapacityTracker()
    tr._accel_anchor_gain_ms2, tr._accel_ratio_step, tr._accel_zero_offset_ms2 = anchor, ratio, offset
    trace, updates, last_anchor, prev = [], 0, anchor, None
    with path.open(newline="") as f:
        reader = csv.reader(f)
        hdr = next(reader)
        ix = {h: i for i, h in enumerate(hdr)}
        for row in reader:
            if len(row) != len(hdr) or not row[ix["utc"]].startswith(date):
                continue
            try:
                cur = {k: float(row[ix[k]]) for k in _FIELDS}
            except ValueError:
                continue
            cur["utc"] = row[ix["utc"]][11:19]
            if prev is None or not 0.0 < cur["t_s"] - prev["t_s"] < 1.0:
                prev = cur
                continue
            mass = 20_000.0 / max(cur["gain_scale"], 0.1)
            steps = max(1, round((cur["t_s"] - prev["t_s"]) * hz))
            for k in range(steps):
                at = _lerp(prev, cur, (k + 1) / steps)
                clock[0] = at["t_s"]
                if at["gas_cmd"] > 0.01:
                    tr.update_accel(at["gas_cmd"], max(0.0, at["raw_ms2"]), at["speed_ms"],
                                    at["slope_rad"], at["game_clutch"], int(cur["gear"]), mass,
                                    False, road_load_ms2=at["road_load_ms2"])
            if tr._accel_anchor_gain_ms2 != last_anchor:
                updates, last_anchor = updates + 1, tr._accel_anchor_gain_ms2
            gear = int(cur["gear"])
            if gear > 0:
                est = tr.accel_gain_for_gear(gear, mass, False) / weight_factor(mass, False)
                trace.append((cur["utc"], gear, est, cur["est_accel_ms2"]))
            prev = cur
    return tr, trace, updates


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("date", help="UTC day to replay, YYYY-MM-DD")
    ap.add_argument("--csv", default="accel_to_pedals_debug.csv")
    ap.add_argument("--anchor", type=float, default=2.164, help="starting slope anchor at gear 6")
    ap.add_argument("--ratio", type=float, default=1.073)
    ap.add_argument("--offset", type=float, default=0.694)
    ap.add_argument("--hz", type=float, default=100.0, help="learning rate of the live loop")
    ap.add_argument("--trace-every", type=int, default=0, help="print every Nth high-gear row")
    args = ap.parse_args(argv)
    tr, trace, updates = replay(Path(args.csv), args.date, args.anchor, args.ratio, args.offset, args.hz)
    high = [e for _, g, e, _ in trace if g >= 10]
    lo = min(high) if high else float("nan")
    spread = max(high) / lo if high else float("nan")
    print(f"{args.date}: updates {updates}, high-gear estimate min {lo:.2f} spread {spread:.2f}x, "
          f"end anchor {tr._accel_anchor_gain_ms2:.3f} ratio {tr._accel_ratio_step:.4f} "
          f"offset {tr._accel_zero_offset_ms2:.3f}")
    if args.trace_every > 0:
        for utc, gear, est, live in [r for r in trace if r[1] >= 10][::args.trace_every]:
            print(f"  {utc} gear {gear:2d} replay {est:.2f} live {live:.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
