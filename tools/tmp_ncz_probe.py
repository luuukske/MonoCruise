"""Live view of the TruckersMP no-collision zone state. See core/radar/README.md section 18.

Run it next to the game while driving on TruckersMP, with MonoCruise's plugin
(monocruise_tmp.dll) installed:

    python tools/tmp_ncz_probe.py

Never imports ``core.settings``. The state carries no player identity.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.radar.tmp_state import (  # noqa: E402
    ENTER_CONFIRM_S, NoCollisionZoneGate, TmpStateReader,
)


def main() -> int:
    reader = TmpStateReader()
    gate = NoCollisionZoneGate()
    print(f"gate opens after {ENTER_CONFIRM_S:.1f} s in a zone; Ctrl+C to stop")
    t0 = time.monotonic()
    last_zone = last_active = None
    try:
        while True:
            now = time.monotonic()
            st = reader.read(now)
            active = gate.step(st, now)
            if st.in_no_collision_zone != last_zone or active != last_active:
                exit_note = "  (zone exit: overlapping players stay ghosts)" if gate.exited_zone else ""
                print(f">>> t={now - t0:6.1f} s  in_zone={int(st.in_no_collision_zone)}  "
                      f"gate={'OPEN' if active else 'closed'}{exit_note}")
                last_zone, last_active = st.in_no_collision_zone, active
            print(
                f"t={now - t0:6.1f} s  fresh={int(st.fresh)} connected={int(st.connected)} "
                f"in_zone={int(st.in_no_collision_zone)} streamed={st.players_streamed:3d}  ->  "
                f"{'AEB IGNORES TMP PLAYERS' if active else 'normal'}"
            )
            time.sleep(0.5)
    except KeyboardInterrupt:
        return 0
    finally:
        reader.close()


if __name__ == "__main__":
    raise SystemExit(main())
