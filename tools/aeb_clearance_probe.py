"""Live AEB ego-body clearance probe against radar traffic.

Park bumper-flush against a target, then run from the project root::

    .venv\\Scripts\\python tools\\aeb_clearance_probe.py

Phase 1 reports bodies ahead and behind (face distance -> ego_half_length when
flush; behind covers own trailer when coupled). Phase 2 reports the body to the
right (left face distance -> ego_half_width when flush). Press Enter to freeze
each phase; Ctrl+C to quit.

Uses the same world->ego transform as ``core/aeb/debug_window.py::_w2e``.
Needs the game + SCS telemetry SDK + ETS2LA traffic plugin; MonoCruise may
also be running.
"""

from __future__ import annotations

import math
import os
import sys
import time
from dataclasses import dataclass

from core.aeb.calibration import AEBCalibration
from core.radar.reader import TrafficReader
from core.radar.traffic import Vehicle

_REFRESH_S = 0.15
_CAL = AEBCalibration()


def _w2e(wx: float, wz: float, ex: float, ez: float, ey: float) -> tuple[float, float]:
    """Ego-space (rx, rz); rz > 0 ahead, rx > 0 left of ego (AEB debug convention)."""
    dx = wx - ex
    dz = wz - ez
    c = math.cos(-ey)
    s = math.sin(-ey)
    return (-dx) * c - dz * s, (-dx) * s + dz * c


@dataclass(frozen=True)
class BodyInEgo:
    vehicle: Vehicle
    # Corner extents in ego frame (debug _w2e): +rz ahead, +rx left.
    min_rx: float
    max_rx: float
    min_rz: float
    max_rz: float
    center_rx: float
    center_rz: float

    @property
    def tag(self) -> str:
        v = self.vehicle
        kind = "trailer" if v.is_trailer else "veh"
        parked = " parked" if v.is_parked else ""
        tmp = " tmp" if v.is_tmp else ""
        return f"id={v.id} {kind}{parked}{tmp} L={v.size.length:.2f} W={v.size.width:.2f}"


def _body_in_ego(v: Vehicle, ex: float, ez: float, ey: float) -> BodyInEgo:
    corners = [_w2e(cx, cz, ex, ez, ey) for cx, cz in v.get_corners()]
    rxs = [c[0] for c in corners]
    rzs = [c[1] for c in corners]
    crx, crz = _w2e(v.position.x, v.position.z, ex, ez, ey)
    return BodyInEgo(
        vehicle=v,
        min_rx=min(rxs),
        max_rx=max(rxs),
        min_rz=min(rzs),
        max_rz=max(rzs),
        center_rx=crx,
        center_rz=crz,
    )


def _collect_bodies(
    vehicles: list[Vehicle],
    trailer_vehicles: list[Vehicle],
    ex: float,
    ez: float,
    ey: float,
) -> list[BodyInEgo]:
    # Nested trailers (ACC list) are separate bodies; TMP trailer-as-vehicle
    # slots already sit in ``vehicles`` with is_trailer=True.
    seen: set[int] = set()
    out: list[BodyInEgo] = []
    for v in list(vehicles) + list(trailer_vehicles):
        if v.id in seen:
            continue
        seen.add(v.id)
        out.append(_body_in_ego(v, ex, ez, ey))
    return out


def _in_lateral_band(b: BodyInEgo, band: float) -> bool:
    # +rx = left of ego (_w2e). Ego strip is roughly rx in [-band, +band].
    return not (b.max_rx < -band or b.min_rx > band)


def _pick_front(bodies: list[BodyInEgo], ego_hl: float, ego_hw: float) -> BodyInEgo | None:
    """Closest body whose nearest face is ahead of ego origin, in the ego width band."""
    band = ego_hw + 1.5
    candidates = [
        b for b in bodies
        if b.min_rz > 0.05 and _in_lateral_band(b, band)
    ]
    if not candidates:
        return None
    return min(candidates, key=lambda b: b.min_rz)


def _pick_behind(bodies: list[BodyInEgo], ego_hl: float, ego_hw: float) -> BodyInEgo | None:
    """Closest body behind ego (own trailer when coupled / reverse-parked)."""
    band = ego_hw + 1.5
    candidates = [
        b for b in bodies
        if b.max_rz < -0.05 and _in_lateral_band(b, band)
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda b: b.max_rz)


def _pick_right(bodies: list[BodyInEgo], ego_hl: float, ego_hw: float) -> BodyInEgo | None:
    """Closest body whose leftmost face is to the right of ego (negative rx)."""
    long_band = ego_hl + 2.0
    candidates = []
    for b in bodies:
        # Right of ego = negative rx in _w2e (debug draws +rx to screen left).
        if b.max_rx >= -0.05:
            continue
        if b.max_rz < -long_band or b.min_rz > long_band:
            continue
        candidates.append(b)
    if not candidates:
        return None
    # Nearest right face = least-negative max_rx (closest to ego from the right).
    return max(candidates, key=lambda b: b.max_rx)


def _fmt_longitudinal(front: BodyInEgo | None, behind: BodyInEgo | None,
                      ego_hl: float, ego_hw: float) -> str:
    lines = []
    if front is None:
        lines.append("FRONT: (no body ahead in ego band)")
    else:
        gap = front.min_rz - ego_hl
        lines.append(
            f"FRONT: {front.tag}\n"
            f"  nearest face rz={front.min_rz:+.3f} m  "
            f"(ego origin -> target rear)\n"
            f"  center  rx={front.center_rx:+.3f} rz={front.center_rz:+.3f} m\n"
            f"  body rx=[{front.min_rx:+.2f},{front.max_rx:+.2f}]  "
            f"rz=[{front.min_rz:+.2f},{front.max_rz:+.2f}]\n"
            f"  gap vs ego_half_length({ego_hl:.2f}) = {gap:+.3f} m\n"
            f"  if flush (~0 gap): suggested ego_half_length ~= {front.min_rz:.3f} m"
        )
    lines.append("")
    if behind is None:
        lines.append("BEHIND: (no body behind in ego band)")
    else:
        face = -behind.max_rz
        gap = face - ego_hl
        lines.append(
            f"BEHIND: {behind.tag}\n"
            f"  nearest face behind={face:+.3f} m  "
            f"(ego origin -> target front; own trailer when coupled)\n"
            f"  center  rx={behind.center_rx:+.3f} rz={behind.center_rz:+.3f} m\n"
            f"  body rx=[{behind.min_rx:+.2f},{behind.max_rx:+.2f}]  "
            f"rz=[{behind.min_rz:+.2f},{behind.max_rz:+.2f}]\n"
            f"  gap vs ego_half_length({ego_hl:.2f}) = {gap:+.3f} m\n"
            f"  if flush (~0 gap): suggested ego_half_length ~= {face:.3f} m"
        )
    return "\n".join(lines)


def _fmt_right(b: BodyInEgo | None, ego_hl: float, ego_hw: float) -> str:
    if b is None:
        return "RIGHT: (no body to the right of ego)"
    # max_rx is the leftmost face of a body sitting on the right (most toward ego).
    face_right = -b.max_rx  # metres to the right of ego origin
    gap = face_right - ego_hw
    return (
        f"RIGHT: {b.tag}\n"
        f"  nearest face right={face_right:+.3f} m  "
        f"(ego origin -> target left)\n"
        f"  center  rx={b.center_rx:+.3f} rz={b.center_rz:+.3f} m\n"
        f"  body rx=[{b.min_rx:+.2f},{b.max_rx:+.2f}]  "
        f"rz=[{b.min_rz:+.2f},{b.max_rz:+.2f}]\n"
        f"  gap vs ego_half_width({ego_hw:.2f}) = {gap:+.3f} m\n"
        f"  if flush (~0 gap): suggested ego_half_width ~= {face_right:.3f} m"
    )


def _read_ego(telemetry) -> tuple[float, float, float, float, float] | None:
    raw = telemetry.get_data()
    if not raw.get("sdkActive", False):
        return None
    return (
        float(raw["coordinateX"]),
        float(raw["coordinateY"]),
        float(raw["coordinateZ"]),
        float(raw["rotationX"]) * 2.0 * math.pi,
        float(raw.get("speed", 0.0)),
    )


def _clear() -> None:
    os.system("cls" if os.name == "nt" else "clear")


def _snapshot_msg(
    reader: TrafficReader,
    telemetry,
    mode: str,
    ego_hl: float,
    ego_hw: float,
) -> str:
    ego = _read_ego(telemetry)
    if ego is None:
        return "SDK inactive / no telemetry"
    ex, ey, ez, yaw, spd = ego
    frame = reader.read(ex, ey, ez, spd)
    if frame is None:
        return "Traffic buffer not open (ETS2LA traffic plugin?)"
    vehicles, trailer_vehicles = frame
    bodies = _collect_bodies(vehicles, trailer_vehicles, ex, ez, yaw)
    header = (
        f"ego speed={spd * 3.6:.1f} km/h  "
        f"tracked={len(vehicles)} nested_trailers={len(trailer_vehicles)}\n"
        f"cal: ego_half_length={ego_hl:.2f}  ego_half_width={ego_hw:.2f}\n\n"
    )
    if mode == "longitudinal":
        return header + _fmt_longitudinal(
            _pick_front(bodies, ego_hl, ego_hw),
            _pick_behind(bodies, ego_hl, ego_hw),
            ego_hl,
            ego_hw,
        )
    return header + _fmt_right(_pick_right(bodies, ego_hl, ego_hw), ego_hl, ego_hw)


def _phase(
    name: str,
    reader: TrafficReader,
    telemetry,
    mode: str,
    ego_hl: float,
    ego_hw: float,
) -> None:
    print(f"\n=== {name} ===")
    print("Live updating. Press Enter to freeze this reading.\n")
    last_line = ""
    while True:
        msg = _snapshot_msg(reader, telemetry, mode, ego_hl, ego_hw)
        if msg != last_line:
            _clear()
            print(f"=== {name} ===  (Enter = freeze, Ctrl+C = quit)\n")
            print(msg)
            last_line = msg
        if _enter_pressed():
            print("\n--- frozen ---")
            print(msg)
            return
        time.sleep(_REFRESH_S)


def _enter_pressed() -> bool:
    try:
        import msvcrt
    except ImportError:
        return False
    hit = False
    while msvcrt.kbhit():
        ch = msvcrt.getwch()
        if ch in ("\r", "\n"):
            hit = True
    return hit


def main() -> int:
    import truck_telemetry

    truck_telemetry.init()
    ego = _read_ego(truck_telemetry)
    if ego is None:
        print("SCS telemetry SDK not active. Start the game first.", file=sys.stderr)
        return 1

    reader = TrafficReader()
    if not reader.open():
        print(
            "Could not open Local\\ETS2LATraffic. Is the ETS2LA traffic plugin loaded?",
            file=sys.stderr,
        )
        return 1

    ego_hl = _CAL.ego_half_length
    ego_hw = _CAL.ego_half_width
    print(
        f"AEB clearance probe  "
        f"(cal ego_half_length={ego_hl:.2f} ego_half_width={ego_hw:.2f})"
    )
    print(
        "Phase 1: flush along the truck axis (nose into a target, or cab against "
        "own trailer). Phase 2: side-flush on the right."
    )

    try:
        _phase(
            "PHASE 1: longitudinal clearance (front + behind)",
            reader,
            truck_telemetry,
            "longitudinal",
            ego_hl,
            ego_hw,
        )
        input(
            "\nReposition for the RIGHT measurement, then press Enter to start phase 2..."
        )
        _phase(
            "PHASE 2: RIGHT clearance",
            reader,
            truck_telemetry,
            "right",
            ego_hl,
            ego_hw,
        )
    except KeyboardInterrupt:
        print("\nquit")
    finally:
        reader.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
