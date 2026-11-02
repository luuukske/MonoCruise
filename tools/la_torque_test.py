"""Push the wheel with a steady LA torque through the FFB plugin, no lane data, to check direction and feel.

Needs the game running and the truck driving (not paused), with a version 2 or later plugin
(plugins/ffb/build/monocruise_ffb.dll) in bin/win_x64/plugins. MonoCruise may stay open while
``la_enabled`` is off; with it on, MonoCruise overwrites the torque.

  python tools/la_torque_test.py
  python tools/la_torque_test.py --units 1200 --seconds 2
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.lane_departure.command import TORQUE_CAP_UNITS, TORQUE_VERSION, VERSION, PluginStatus, RumbleChannel

_TICK_S = 0.05
_WAIT_S = 10.0
_GAP_S = 1.5


def _wait_for_plugin(channel: RumbleChannel) -> PluginStatus:
    deadline = time.monotonic() + _WAIT_S
    status = PluginStatus()
    while time.monotonic() < deadline:
        status = channel.publish(False, 0, 0)
        if status.present:
            break
        time.sleep(_TICK_S)
    return status


def _hold(channel: RumbleChannel, torque: int, seconds: float) -> tuple[float, int]:
    """Keep the heartbeat alive at one torque; returns the game's mean force and the largest torque applied."""
    forces = []
    applied = 0
    end = time.monotonic() + seconds
    while time.monotonic() < end:
        status = channel.publish(False, 0, 0, torque=torque)
        forces.append(status.game_force)
        applied = max(applied, abs(status.applied))
        time.sleep(_TICK_S)
    return (sum(forces) / len(forces) if forces else 0.0), applied


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--units", type=int, default=800, help=f"torque out of 10000, capped at {TORQUE_CAP_UNITS}")
    parser.add_argument("--seconds", type=float, default=2.0, help="hold per direction")
    args = parser.parse_args()
    units = max(0, min(TORQUE_CAP_UNITS, args.units))

    channel = RumbleChannel()
    status = _wait_for_plugin(channel)
    if not status.present:
        print("plugin not found: start the game with monocruise_ffb.dll installed")
        return 1
    if status.version < TORQUE_VERSION:
        print(f"plugin version {status.version} only rumbles: install the version {VERSION} DLL")
        return 1
    if not status.hooked:
        print("constant force not hooked yet: drive for a few seconds and rerun")
        return 1
    print(f"plugin present, version {status.version}, game force now {status.game_force}")
    results = []
    for sign in (1, -1):
        print(f"holding {sign * units:+d} for {args.seconds:.1f} s: note which way the wheel is pushed")
        results.append(_hold(channel, sign * units, args.seconds))
        _hold(channel, 0, _GAP_S)
    channel.stop()
    channel.close()
    for sign, (force, applied) in zip((1, -1), results):
        print(f"{sign * units:+d}: applied up to {applied}, game force mean {force:+.0f}")
    print("A positive wheel value in the LA calibration log means a positive torque should push the wheel left.")
    return 0 if all(applied > 0 for _, applied in results) else 2


if __name__ == "__main__":
    sys.exit(main())
