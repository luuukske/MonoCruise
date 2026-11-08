"""The numbers in tools/la_bench/SMOOTH_RESULTS.md: the hold metrics of the servo as shipped before, with the
learner and the hold, and on the true wheel (the reference), over the held-out set and the five presets. Simulated.

    python -m tools.la_bench.smooth_eval held --out <file.json>
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict

from core.la.hold import HoldTuning
from core.la.wheel_learner import WheelLearner
from tools.la_bench.bench import sim_bench
from tools.la_bench.learn_sim import EPISODES, design_for, la_drive, manual_drive
from tools.la_bench.metrics import summarize
from tools.la_bench.plant import WheelParams
from tools.la_bench.population import Hold, hold_test, population, presets, split, table
from tools.la_bench.protocol import Design, follow, ramp

CHECKPOINTS = (0, 8, 16, EPISODES)
TRACK = (ramp(0.06, 0.05), follow(0.03, 0.4))


def tracking(p: WheelParams, design: Design) -> list[float | None]:
    """Error rms on a slow ramp and a slow sine (LA's own target moves at most 0.2 lock/s), the hold not involved."""
    bench = sim_bench(p)
    bench.say = lambda _m: None
    bench.ready()
    return [summarize(bench.run_stage(st, design, p.sign))["track"]["rms"] for st in TRACK]


def reference(p: WheelParams) -> Design:
    """The gentlest design of the tuning grid with the true rim's friction pushes: what a perfect fit would set."""
    top = p.breakaway * (1.0 + p.breakaway_spread)
    return Design(bandwidth_hz=0.5, damping=1.4, inertia=150.0, damping_first=True, ff_damping=p.damping,
                  ff_friction=p.stick, ff_breakaway=p.stick + 1.2 * max(0.0, top - p.stick), plan_s=p.delay_s)


def evaluate(wheels: list[tuple[str, WheelParams]]) -> dict:
    rows = []
    for i, (name, p) in enumerate(wheels):
        learner = WheelLearner()
        manual_drive(p, learner, seed=i)
        learned: dict[int, Hold] = {}
        done = 0
        for k in CHECKPOINTS:
            if k > done:
                la_drive(p, learner, episodes=k - done, seed=i + 100 * k)
                done = k
            m = learner.model
            learned[k] = hold_test(p, design_for(m), hold=HoldTuning(), quantum=m.quantum)
        rows.append({
            "name": name, "params": asdict(p), "model": learner.model.to_dict(),
            "default": asdict(hold_test(p, Design())),
            "learned": {k: asdict(h) for k, h in learned.items()},
            "reference": asdict(hold_test(p, reference(p), hold=HoldTuning(), quantum=p.quantum)),
            "reference_no_hold": asdict(hold_test(p, reference(p))),
            "tracking": {"default": tracking(p, Design()), "learned": tracking(p, design_for(learner.model))},
        })
        print(name, "learned ok" if learned[EPISODES].ok else "", flush=True, file=sys.stderr)
    return {"rows": rows, "tables": tables(rows)}


def tables(rows: list[dict]) -> dict:
    def tab(get) -> dict:
        return table([(r["name"], Hold(**get(r))) for r in rows])

    out = {"default": tab(lambda r: r["default"]), "reference": tab(lambda r: r["reference"]),
           "reference_no_hold": tab(lambda r: r["reference_no_hold"])}
    for k in CHECKPOINTS:
        out[f"learned_{k}"] = tab(lambda r, k=k: r["learned"][k])
    for who in ("default", "learned"):
        for i, st in enumerate(TRACK):
            vals = sorted(r["tracking"][who][i] for r in rows if r["tracking"][who][i] is not None)
            out[f"track_{who}_{st.name}"] = {"median": vals[len(vals) // 2], "p90": vals[int(0.9 * (len(vals) - 1))]}
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("which", choices=("held", "tune", "presets"))
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    tune, held = split(population())
    wheels = {"held": held, "tune": tune, "presets": presets()}[args.which]
    data = evaluate(wheels)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=1, default=str)
    print(json.dumps(data["tables"], indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
