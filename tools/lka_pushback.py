"""How hard the driver pushed against each lane keeping pulse, from ``lka_debug.csv`` (written with ``debug`` on).

Per pulse: the pulse's own push away from its line, the driver's push against it, and whether and when the driver
went past the pulse's own push (only a driver steering toward the line can). Files written before the estimate was
recorded are re-estimated from the plugin's raw readings. Stdlib only, never imports ``core.settings``.

  python tools/lka_pushback.py
  python tools/lka_pushback.py la_debug.csv --json
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.la.hand_push import HandPushEstimator, PushTuning

_GAP_S = 0.2
_BEYOND_UNITS = 500.0


def _num(row: dict, key: str) -> float:
    try:
        return float(row.get(key, ""))
    except ValueError:
        return math.nan


def _peak(values: list[float]) -> int | None:
    finite = [v for v in values if math.isfinite(v)]
    return round(max(finite)) if finite else None


def _estimate(rows: list[dict], inertia: float, direction: int) -> None:
    """Fill ``pulse_push`` and ``hand_against`` for a file that predates them; the side comes from the offset."""
    est = HandPushEstimator(PushTuning(inertia=inertia))
    last_t = None
    side = 0
    for row in rows:
        t = _num(row, "t_s")
        dt = t - last_t if last_t is not None else 0.0
        last_t = t
        if row["reason"] != "pulse":
            side = 0
        elif not side:
            side = 1 if _num(row, "offset_m") > 0.0 else -1
        push = est.update(dt, _num(row, "steer_rate"), _num(row, "game_force"), _num(row, "applied"),
                          _num(row, "cancel_applied"), direction, side)
        row["pulse_push"], row["hand_against"] = push.push, push.against


def pulses(rows: list[dict]) -> list[dict]:
    out = []
    i = 0
    while i < len(rows):
        if rows[i]["reason"] != "pulse":
            i += 1
            continue
        j = i
        while j + 1 < len(rows) and rows[j + 1]["reason"] == "pulse" and \
                _num(rows[j + 1], "t_s") - _num(rows[j], "t_s") <= _GAP_S:
            j += 1
        seg = rows[i:j + 1]
        t0 = _num(seg[0], "t_s")
        push = [_num(r, "pulse_push") for r in seg]
        against = [_num(r, "hand_against") for r in seg]
        beyond = [a - max(0.0, p) for p, a in zip(push, against)]
        past = next((_num(r, "t_s") - t0 for r, b in zip(seg, beyond) if b >= _BEYOND_UNITS), None)
        side = 1 if _num(seg[0], "offset_m") > 0.0 else -1
        after = rows[j + 1]["reason"] if j + 1 < len(rows) else ""
        out.append({
            "start": datetime.fromisoformat(seg[0]["utc"]).astimezone().strftime("%H:%M:%S.%f")[:10],
            "seconds": round(_num(seg[-1], "t_s") - t0, 2),
            "line": "right" if side > 0 else "left",
            "then": after,
            "pushed": _peak(push),
            "against": _peak(against),
            "beyond": _peak(beyond),
            "past_at_s": None if past is None else round(past, 2),
            "wheel_home": round(side * (_num(seg[-1], "steer") - _num(seg[0], "steer")), 3),
            "offset_m": round(_num(seg[0], "offset_m"), 2),
            "drift_ms": round(_num(seg[0], "drift_ms"), 2),
        })
        i = j + 1
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("csv", nargs="?", default=str(ROOT / "lka_debug.csv"))
    parser.add_argument("--inertia", type=float, default=650.0, help="la_wheel_inertia, for files that predate it")
    parser.add_argument("--direction", type=int, choices=(1, -1), default=1,
                        help="sign of the LA calibration's cornering fit, for files that predate it")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    path = Path(args.csv)
    if not path.exists():
        print(f"{path.name} not found: drive with LKA and debug on first")
        return 1
    with path.open(newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    if rows and "hand_against" not in rows[0]:
        _estimate(rows, args.inertia, args.direction)
    found = pulses(rows)
    if args.json:
        print(json.dumps(found, indent=1))
        return 0
    head = ("start", "seconds", "line", "then", "pushed", "against", "beyond", "past_at_s", "wheel_home", "offset_m",
            "drift_ms")
    print("  ".join(f"{h:>10}" for h in head))
    for p in found:
        print("  ".join(f"{'-' if p[h] is None else p[h]!s:>10}" for h in head))
    past = sum(p["past_at_s"] is not None for p in found)
    print(f"{len(found)} pulses; the driver went {_BEYOND_UNITS:.0f} or more past the pulse's own push in {past}")
    print("pushed: the pulse's push away from its line; against: the driver's added push toward it (passive hands")
    print("reach at most 'pushed'); wheel_home: how far the wheel turned away from the line, in lock.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
