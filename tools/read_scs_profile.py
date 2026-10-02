"""Read the selected ETS2/ATS profile's transmission and braking intensity."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.scs_profile import read_selected_profile


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Read transmission and brake intensity for the selected ETS2/ATS profile."
    )
    parser.add_argument("--game", choices=("ets2", "ats"), help="Force one game instead of detecting")
    parser.add_argument("--json", action="store_true", help="Machine-readable object on stdout")
    args = parser.parse_args(argv)
    settings = read_selected_profile(args.game)
    if args.json:
        json.dump(settings.to_dict(), sys.stdout, indent=2)
        sys.stdout.write("\n")
        return 0 if settings.profile_hex else 2
    _print_text(settings)
    return 0 if settings.profile_hex else 2


def _print_text(s) -> None:
    print(f"game: {s.game}")
    ident = s.profile_hex or "(none)"
    if s.profile_name:
        ident = f"{s.profile_name} ({s.profile_hex})"
    print(f"selected_profile: {ident}")
    print(f"store: {s.store or '(unknown)'}")
    print(f"identity_via: {s.identity_source}")
    trans = s.transmission_ui or s.transmission or "(unread)"
    extra = []
    if s.g_trans is not None:
        extra.append(f"g_trans={s.g_trans}")
    if s.trans_from:
        extra.append(f"from {s.trans_from}")
    suffix = f"  [{', '.join(extra)}]" if extra else ""
    print(f"transmission: {trans}{suffix}")
    adaptive = s.adaptive or "(unread)"
    if s.g_adaptive_shift is not None:
        adaptive = f"{adaptive}  [g_adaptive_shift={s.g_adaptive_shift:g}]"
    print(f"adaptive_gearbox: {adaptive}")
    if s.live_shifter_type:
        print(f"live_shifter_type: {s.live_shifter_type}")
    if s.g_brake_intensity is None:
        print("brake_intensity: (unread)")
    else:
        src = f", from {s.brake_from}" if s.brake_from else ""
        pct = (
            f", {s.brake_intensity_slider_pct:.0f}% UI"
            if s.brake_intensity_slider_pct is not None
            else ""
        )
        print(f"brake_intensity: {s.g_brake_intensity:g}{pct}{src}")
    if s.c_brake_dz is not None:
        print(f"brake_deadzone: {s.c_brake_dz:g}")


if __name__ == "__main__":
    raise SystemExit(main())
