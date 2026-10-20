"""Play one rumble through the FFB plugin with no lane data, to test the wheel path alone.

Needs the game running and the truck driving, with plugins/ffb/build/monocruise_ffb.dll
copied into the game's bin/win_x64/plugins folder. MonoCruise itself may stay closed.

  python tools/ldw_rumble_test.py
  python tools/ldw_rumble_test.py --seconds 3 --intensity 1.5
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.lane_departure.command import PluginStatus, RumbleChannel, intensity_units

_TICK_S = 0.05
_WAIT_S = 10.0


def _wait_for_plugin(channel: RumbleChannel) -> PluginStatus:
    deadline = time.monotonic() + _WAIT_S
    status = PluginStatus()
    while time.monotonic() < deadline:
        status = channel.publish(False, 0, 0)
        if status.present:
            break
        time.sleep(_TICK_S)
    return status


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--seconds", type=float, default=2.0)
    parser.add_argument("--intensity", type=float, default=1.0, help="scale of the 1600 peak, capped at 3000")
    parser.add_argument("--side", choices=("left", "right"), default="right", help="logged only, the cue is not directional")
    args = parser.parse_args()

    channel = RumbleChannel()
    status = _wait_for_plugin(channel)
    if not status.present:
        print("plugin not found: start the game with monocruise_ffb.dll installed")
        return 1
    print(f"plugin present, hooked={status.hooked} force_seen={status.force_seen} dead={status.dead}")
    if not status.hooked:
        print("constant force not hooked yet: drive for a few seconds and rerun")
    units = intensity_units(args.intensity)
    side = 1 if args.side == "left" else 2
    end = time.monotonic() + args.seconds
    rumbled = False
    while time.monotonic() < end:
        status = channel.publish(True, side, units)
        rumbled = rumbled or status.rumbling
        time.sleep(_TICK_S)
    channel.stop()
    time.sleep(0.5)
    status = channel.publish(False, 0, 0)
    channel.close()
    print(f"rumbled={rumbled} frames={status.frames} dead={status.dead}")
    return 0 if rumbled else 2


if __name__ == "__main__":
    sys.exit(main())
