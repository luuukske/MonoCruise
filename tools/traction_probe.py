"""Live view and CSV log of the traction probe plugin (plugins/traction/README.md).

python tools/traction_probe.py              table at 2 Hz
python tools/traction_probe.py --csv x.csv  also log every wheel at --hz (default 20)
"""
from __future__ import annotations

import argparse
import csv
import mmap
import struct
import sys
import time
from dataclasses import dataclass

TAG = "Local\\MonoCruiseTraction"
HEADER = struct.Struct("<6I3f12f10I")
NAMES = 32
NAME_LEN = 16
WHEEL = struct.Struct("<BBH11f")
MAX_WHEELS = 32
TRUCK_WHEELS = 16
SIZE = HEADER.size + NAMES * NAME_LEN + MAX_WHEELS * WHEEL.size
WEATHER_LIVE = 2
TRAILER_FLAG = 128
ON_GROUND = 1
POWERED = 2


@dataclass
class Wheel:
    unit: str
    index: int
    on_ground: bool
    powered: bool
    substance: str
    radius: float
    rim_speed: float
    ground_speed: float
    slip: float
    slip_peak: float
    deflection: float
    pos_x: float
    pos_z: float


@dataclass
class Snapshot:
    version: int
    status: int
    frames: int
    weather: int
    weather_target: int
    wetness: float | None
    speed: float
    wheels: list[Wheel]


def _read(buf: mmap.mmap) -> Snapshot | None:
    for _ in range(20):
        raw = bytes(buf[:SIZE])
        seq = struct.unpack_from("<I", raw, 4)[0]
        if seq % 2 == 0 and struct.unpack_from("<I", bytes(buf[4:8]))[0] == seq:
            break
        time.sleep(0.001)
    else:
        return None
    h = HEADER.unpack_from(raw, 0)
    version, _, status, frames, cur, target = h[:6]
    wet, _, speed = h[6:9]
    truck_n, trailer_n = h[21], h[22]
    if version == 0:
        return None
    names = []
    for i in range(NAMES):
        chunk = raw[HEADER.size + i * NAME_LEN: HEADER.size + (i + 1) * NAME_LEN]
        names.append(chunk.split(b"\0")[0].decode("ascii", "replace"))
    wheels = []
    base = HEADER.size + NAMES * NAME_LEN
    for k in list(range(truck_n)) + [TRUCK_WHEELS + i for i in range(trailer_n)]:
        f = WHEEL.unpack_from(raw, base + k * WHEEL.size)
        sub = f[2]
        wheels.append(Wheel(
            unit="trailer" if f[0] & TRAILER_FLAG else "truck",
            index=k % TRUCK_WHEELS,
            on_ground=bool(f[0] & ON_GROUND),
            powered=bool(f[0] & POWERED),
            substance=names[sub] if sub < len(names) and names[sub] else str(sub),
            radius=f[3], rim_speed=f[5], ground_speed=f[6], slip=f[7], slip_peak=f[8],
            deflection=f[9], pos_x=f[10], pos_z=f[12],
        ))
    return Snapshot(version, status, frames, cur, target,
                    wet if status & WEATHER_LIVE else None, speed, wheels)


def _table(s: Snapshot) -> str:
    wet = "n/a" if s.wetness is None else f"{s.wetness:.3f}"
    lines = [f"frames {s.frames}  speed {s.speed * 3.6:6.1f} km/h  weather {s.weather}->{s.weather_target}"
             f"  wetness {wet}  status {s.status:#x}"]
    for w in s.wheels:
        lines.append(f"  {w.unit:7} {w.index:2} {'P' if w.powered else ' '} {'G' if w.on_ground else ' '}"
                     f" {w.substance:13} slip {w.slip:+7.3f} peak {w.slip_peak:+7.3f}"
                     f" rim {w.rim_speed:6.2f} ground {w.ground_speed:6.2f} defl {w.deflection:6.3f}")
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv")
    ap.add_argument("--hz", type=float, default=20.0)
    ap.add_argument("--seconds", type=float, default=0.0)
    args = ap.parse_args()
    buf = mmap.mmap(-1, SIZE, tagname=TAG, access=mmap.ACCESS_READ)
    out = None
    if args.csv:
        fh = open(args.csv, "w", newline="")
        out = csv.writer(fh)
        out.writerow(["t", "speed_ms", "weather", "target", "wetness", "unit", "wheel", "powered", "on_ground",
                      "substance", "slip", "slip_peak", "rim_ms", "ground_ms", "deflection_m"])
    t0 = time.time()
    last_print = 0.0
    period = 1.0 / args.hz if out else 0.5
    while not args.seconds or time.time() - t0 < args.seconds:
        s = _read(buf)
        now = time.time()
        if s is None:
            if now - last_print > 2.0:
                print("no writer: is the plugin installed and the game running?", file=sys.stderr)
                last_print = now
        else:
            if out:
                for w in s.wheels:
                    out.writerow([f"{now - t0:.3f}", f"{s.speed:.3f}", s.weather, s.weather_target,
                                  "" if s.wetness is None else f"{s.wetness:.4f}", w.unit, w.index,
                                  int(w.powered), int(w.on_ground), w.substance, f"{w.slip:.4f}",
                                  f"{w.slip_peak:.4f}", f"{w.rim_speed:.3f}", f"{w.ground_speed:.3f}",
                                  f"{w.deflection:.4f}"])
            if now - last_print >= 0.5:
                print("\033[2J\033[H" + _table(s) if sys.stdout.isatty() else _table(s), flush=True)
                last_print = now
        time.sleep(period)
    return 0


if __name__ == "__main__":
    sys.exit(main())
