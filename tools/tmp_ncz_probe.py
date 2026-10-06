"""Live view of the plugin's TruckersMP no-collision zone data. See core/radar/README.md section 18.

Run it next to the game while driving on TruckersMP:

    python tools/tmp_ncz_probe.py            # state line twice a second
    python tools/tmp_ncz_probe.py --players  # plus the nearest players' collision flags

Never imports ``core.settings`` and never reads Steam ids or latency from the player records.
"""

from __future__ import annotations

import argparse
import mmap
import struct
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.radar.tmp_state import (  # noqa: E402
    ENTER_CONFIRM_S, NoCollisionZoneGate, TmpStateReader,
)

_PLAYERS_TAG = r"Local\ETS2LAMpPlayers"
_PLAYER_SIZE = 183
_PLAYERS_SIZE = 40 * _PLAYER_SIZE
_HAS_COLLISION_OFFSET = 182
# Vehicle record: x, y, z, qw, qx, qy, qz, width, height, length, speed.
_VEHICLE_HEAD = "=11f"


def _player_rows(buf: mmap.mmap, limit: int) -> list[str]:
    rows = []
    for slot in range(40):
        base = slot * _PLAYER_SIZE
        x, _y, z, *_q, width, _h, length, speed = struct.unpack_from(_VEHICLE_HEAD, buf, base)
        if length <= 0.0 and x == 0.0 and z == 0.0:
            continue
        collides = buf[base + _HAS_COLLISION_OFFSET] != 0
        rows.append(
            f"    slot {slot:2d}  collides={'yes' if collides else 'NO '}  "
            f"x={x:10.1f} z={z:10.1f}  {length:4.1f} x {width:3.1f} m  {speed * 3.6:6.1f} km/h"
        )
        if len(rows) >= limit:
            break
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--players", action="store_true", help="also list the nearest players")
    ap.add_argument("--limit", type=int, default=8, help="players to list (default 8)")
    args = ap.parse_args()

    reader = TmpStateReader()
    gate = NoCollisionZoneGate()
    players = None
    if args.players:
        try:
            players = mmap.mmap(0, _PLAYERS_SIZE, _PLAYERS_TAG)
        except Exception as exc:
            print(f"player buffer unavailable: {exc}")

    print(f"gate opens after {ENTER_CONFIRM_S:.1f} s of agreement; Ctrl+C to stop")
    try:
        while True:
            now = time.monotonic()
            st = reader.read(now)
            active = gate.step(st, now)
            print(
                f"fresh={int(st.fresh)} connected={int(st.connected)} "
                f"in_zone={int(st.in_no_collision_zone)} streamed={st.players_streamed:3d} "
                f"collidable={st.players_collidable:3d}  ->  "
                f"{'AEB IGNORES TMP PLAYERS' if active else 'normal'}"
            )
            if players is not None:
                for row in _player_rows(players, args.limit):
                    print(row)
            time.sleep(0.5)
    except KeyboardInterrupt:
        return 0
    finally:
        reader.close()
        if players is not None:
            players.close()


if __name__ == "__main__":
    raise SystemExit(main())
