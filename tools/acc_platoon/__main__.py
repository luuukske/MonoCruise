"""CLI: run convoy scenarios or the netcode calibration. See tools/acc_platoon/README.md."""
from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import sys
import tempfile
from dataclasses import asdict, replace
from pathlib import Path


def _sandbox_settings() -> Path:
    """Point core.settings at a throwaway copy before anything can load or save it."""
    import core.settings as settings_mod

    sandbox = Path(tempfile.mkdtemp(prefix="monocruise-platoon-"))
    real = Path(settings_mod.CONFIG_PATH)
    settings_mod.CONFIG_PATH = sandbox / "config.json"
    settings_mod.BACKUP_PATH = settings_mod.CONFIG_PATH.with_suffix(".json.bak")
    if real.is_file():
        shutil.copy2(str(real), str(settings_mod.CONFIG_PATH))
    return sandbox


def _scenario_report(name: str, seeds: list[int], level: int | None, as_json: bool) -> None:
    from . import metrics, scenarios
    from .netcode import CLEAN
    from .sim import run

    out = []
    for seed in seeds:
        factory = scenarios.ALL[name]
        sc = factory(seed, aeb=True) if args.aeb else factory(seed)
        if level is not None:
            sc = replace(sc, gap_level=level)
        if args.clean:
            sc = replace(sc, nets=(CLEAN,) * (sc.followers + 1))
        r = run(sc)
        stats = metrics.truck_stats(r, 0.0)
        if as_json:
            out.append({"scenario": name, "seed": seed, "gap_level": sc.gap_level,
                        "trucks": [asdict(s) for s in stats]})
            continue
        print(f"=== {name} seed {seed} gap level {sc.gap_level}")
        print(metrics.table(stats))
        if sc.v0_kmh - stats[0].min_speed_kmh > 1.0:
            gains = metrics.dip_gains(stats, sc.v0_kmh)
            print("speed dip vs lead: " + " ".join(f"{g:.2f}" for g in gains[1:]))
        print(f"contacts: {metrics.contacts(r)}  unprovoked brakes: "
              f"{sum(metrics.unprovoked_brakes(r))}")
    if as_json:
        json.dump(out, sys.stdout, indent=1)
        print()


def _calibration_report(clips: int | None, root: str | None) -> None:
    from core.aeb.clip_store import default_clip_root

    from . import calibrate

    streams = calibrate.corpus_streams(Path(root) if root else default_clip_root(), clips,
                                       workers=min(8, os.cpu_count() or 1))
    if not streams:
        raise SystemExit("no TMP streams found; is the clip store there?")
    synth = calibrate.model_streams()
    print(f"{len(streams)} corpus streams against {len(synth)} synthetic ones")
    print(calibrate.report(calibrate.raw_stats(streams), calibrate.raw_stats(synth)))
    print(calibrate.report(calibrate.filter_stats(streams), calibrate.filter_stats(synth)))
    print(calibrate.report(calibrate.artefact_response(streams),
                           calibrate.artefact_response(synth)))


parser = argparse.ArgumentParser(prog="python -m tools.acc_platoon", description=__doc__)
parser.add_argument("--scenario", default="steady", help="name, or 'all'; see scenarios.ALL")
parser.add_argument("--seeds", default="1", help="comma-separated seeds")
parser.add_argument("--gap-level", type=int, default=None, help="override the scenario's level")
parser.add_argument("--clean", action="store_true", help="no netcode artefacts, for A/B")
parser.add_argument("--aeb", action="store_true", help="run the headless AEB in every client")
parser.add_argument("--json", action="store_true")
parser.add_argument("--calibrate", action="store_true", help="compare the model with the clip store")
parser.add_argument("--clips", type=int, default=None, help="sample size; default reads every clip")
parser.add_argument("--clip-root", default=None)
args = parser.parse_args()
# AEB interventions raise popups; there is no window here to show them.
logging.getLogger("ui.popup.popup_window").setLevel(logging.ERROR)

_sandbox = _sandbox_settings()
try:
    if args.calibrate:
        _calibration_report(args.clips, args.clip_root)
    else:
        from .scenarios import ALL

        names = list(ALL) if args.scenario == "all" else [args.scenario]
        for scenario_name in names:
            _scenario_report(scenario_name, [int(s) for s in args.seeds.split(",")],
                             args.gap_level, args.json)
finally:
    shutil.rmtree(str(_sandbox), ignore_errors=True)
