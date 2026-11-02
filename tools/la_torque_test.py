"""Push the wheel with a steady LA torque through the FFB plugin, no lane data, to check direction and feel.

Needs the game running and the truck driving (not paused), with a version 2 or later plugin
(plugins/ffb/build/monocruise_ffb.dll) in bin/win_x64/plugins. MonoCruise may stay open while
``la_enabled`` is off; with it on, MonoCruise overwrites the torque.

It also checks the driver's push estimate (core/la/hand_push.py) against the known torque: run it standing still,
once with hands off the rim (the estimate should read about 0) and once holding the wheel still (about minus the
torque). Needs a version 4 plugin for the wheel's rate.

  python tools/la_torque_test.py
  python tools/la_torque_test.py --units 1200 --seconds 2
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.la.hand_push import HandPushEstimator, PushTuning
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


def _hold(channel: RumbleChannel, torque: int, seconds: float, est: HandPushEstimator,
          direction: int) -> tuple[float, int, float]:
    """Keep the heartbeat alive at one torque; returns the game's mean force, the largest torque applied and the
    driver's mean estimated push over the second half, as the torque does (positive turns it counterclockwise).
    """
    forces = []
    hands = []
    applied = 0
    start = last = time.monotonic()
    end = start + seconds
    while time.monotonic() < end:
        status = channel.publish(False, 0, 0, torque=torque)
        now = time.monotonic()
        push = est.update(now - last, status.steer_rate, status.game_force, status.applied, status.cancel_applied,
                          direction)
        last = now
        forces.append(status.game_force)
        applied = max(applied, abs(status.applied))
        if now - start >= 0.5 * seconds and math.isfinite(push.hand):
            hands.append(direction * push.hand)
        time.sleep(_TICK_S)
    hand = sum(hands) / len(hands) if hands else math.nan
    return (sum(forces) / len(forces) if forces else 0.0), applied, hand


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--units", type=int, default=800, help=f"torque out of 10000, capped at {TORQUE_CAP_UNITS}")
    parser.add_argument("--seconds", type=float, default=2.0, help="hold per direction")
    parser.add_argument("--direction", type=int, choices=(1, -1), default=1,
                        help="sign of the LA calibration's cornering fit (the torque sign that turns the wheel left)")
    parser.add_argument("--inertia", type=float, default=650.0, help="la_wheel_inertia")
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
    est = HandPushEstimator(PushTuning(inertia=args.inertia))
    results = []
    for sign in (1, -1):
        print(f"holding {sign * units:+d} for {args.seconds:.1f} s: note which way the wheel is pushed")
        results.append(_hold(channel, sign * units, args.seconds, est, args.direction))
        _hold(channel, 0, _GAP_S, est, args.direction)
    channel.stop()
    channel.close()
    for sign, (force, applied, hand) in zip((1, -1), results):
        print(f"{sign * units:+d}: applied up to {applied}, game force mean {force:+.0f}, driver's push estimate "
              f"{hand:+.0f} (hands off: about 0; holding the wheel still: about {-sign * units:+d})")
    print("A positive wheel value in the LA calibration log means a positive torque should push the wheel left.")
    if status.version < VERSION:
        print(f"the push estimate needs the version {VERSION} plugin's wheel rate; ignore it here")
    return 0 if all(applied > 0 for _, applied, _ in results) else 2


if __name__ == "__main__":
    sys.exit(main())
