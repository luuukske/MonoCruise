"""Does the ACC command move smoothly when the lead barely brakes, or step?

The response map in `acc_response_map.py` is a steady-state slice, so it cannot
see either failure this probe targets. See `tools/README.md`.

Two questions, one per panel row:

* **Corner.** Sweep `a_lead` finely through zero and differentiate the cap. A
  hard `min(0, demand)` in the feedforward makes the gain step at `a_lead = 0`,
  which rectifies jitter around that point into a one-sided mean brake.
* **Chatter.** Hold the lead at a small braking rate, add sample-and-hold noise
  of the kind laggy TMP traffic produces, and run closed loop. The old filter
  re-picked its time constant at a hard threshold on the innovation, so a lead
  hovering on that edge alternated between two bandwidths every tick.

Every knob this change added is keyed on a config field that disables it at 0,
so the HEAD baseline and the current controller run in one process against the
same checkout with no worktree. `BASELINE` must list *all* of them: an earlier
version omitted the two feedforward shares, which left the feedforward live in
both columns and reported a jerk improvement where there was a regression.
`tests/acc/test_gap_law_shaping.py` pins that each knob really does disable its
feature, so this file can trust the list.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from acc_probe_rig import Rig, StubLead, patched_clock

# Every feature knob off. Reproduces HEAD's lead_law bit exactly.
BASELINE = {"lead_brake_ff_share": 0.0, "lead_accel_nudge_share": 0.0,
            "a_lead_tau_ramp_ms2": 0.0, "lead_brake_ff_soft_ms2": 0.0,
            "lead_law_floor_soft_ms2": 0.0}

VARIANTS = (("HEAD", BASELINE), ("current", {}))


def sweep_corner(rig: Rig, ego_ms, dist_m, closing_ms, decels):
    """Cap across a fine `a_lead` sweep, plus its numeric gain."""
    v_lead = ego_ms - closing_ms
    with patched_clock(rig.clock):
        caps = np.array([rig.cap(ego_ms, dist_m, v_lead, -float(d)) for d in decels])
    return caps, np.gradient(caps, decels)


def jitter_trace(rig: Rig, ego_ms, dist_m, base_decel, sigma, seed, horizon=14.0,
                 hold_hz=8.0):
    """Closed loop against a lead whose reported a_lead is noisy and held.

    Sample-and-hold at `hold_hz` is what a telemetry rate below the control rate
    looks like from inside the controller."""
    rng = np.random.default_rng(seed)
    ctrl = rig.controller()
    v_ego = v_lead = ego_ms
    gap, t = dist_m, 0.0
    hold_dt = 1.0 / hold_hz
    next_sample, noise = 0.0, 0.0
    ts, cmds, a_leads = [], [], []
    with patched_clock(rig.clock):
        while t < horizon and gap > 0.5 and v_ego > 0.05:
            if t >= next_sample:
                noise = float(rng.normal(0.0, sigma))
                next_sample += hold_dt
            a_lead_seen = -base_decel + noise
            rig.stub.data.publish(StubLead(gap, v_lead, a_lead_seen, rig.score))
            rig.clock.advance()
            cap = ctrl.accel_cap_ms2(v_ego)
            a_ego = min(0.0, cap)
            ts.append(t)
            cmds.append(cap)
            a_leads.append(a_lead_seen)
            v_ego = max(0.0, v_ego + a_ego * rig.dt)
            v_lead = max(0.0, v_lead - base_decel * rig.dt)
            gap += (v_lead - v_ego) * rig.dt
            t += rig.dt
    return np.array(ts), np.array(cmds), np.array(a_leads)


def corner_stats(decels, caps, gain):
    """Worst gain step across the sweep: the number the corner fix moves."""
    d2 = np.abs(np.diff(gain))
    k = int(np.argmax(d2)) if d2.size else 0
    return {
        "worst_gain_jump": float(d2[k]) if d2.size else 0.0,
        "at_decel": float(decels[k + 1]) if d2.size else 0.0,
        "gain_span": float(gain.max() - gain.min()),
        "cap_at_zero": float(caps[int(np.abs(decels).argmin())]),
    }


def jitter_stats(ts, cmds, dt):
    """Chatter and rectification: RMS jerk, and how far the mean is dragged."""
    jerk = np.diff(cmds) / dt
    return {
        "rms_jerk": float(np.sqrt(np.mean(jerk ** 2))) if jerk.size else 0.0,
        "peak_jerk": float(np.max(np.abs(jerk))) if jerk.size else 0.0,
        "mean_cmd": float(np.mean(cmds)),
        "cmd_std": float(np.std(cmds)),
    }


def mean_jitter_stats(rig, ego_ms, dist_m, base_decel, sigma, seeds, dt):
    """Averaged over seeds: one noise realisation ranks variants wrongly."""
    runs = [jitter_stats(*jitter_trace(rig, ego_ms, dist_m, base_decel, sigma, s)[:2],
                         dt) for s in range(seeds)]
    return {k: float(np.mean([r[k] for r in runs])) for k in runs[0]}


def run(repo: Path, gap_level: int, ego_kmh: float, dist_m: float, closing_ms: float,
        base_decel: float, sigma: float, seed: int, dt: float, seeds: int) -> dict:
    ego_ms = ego_kmh / 3.6
    decels = np.linspace(-0.6, 1.6, 441)
    out: dict = {"decels": decels, "ego_kmh": ego_kmh, "dist_m": dist_m,
                 "closing_ms": closing_ms, "base_decel": base_decel,
                 "sigma": sigma, "seeds": seeds, "variants": {}}
    for name, overrides in VARIANTS:
        rig = Rig(repo, gap_level, dt, 8, 5.0)
        rig.overrides = dict(overrides)
        caps, gain = sweep_corner(rig, ego_ms, dist_m, closing_ms, decels)
        ts, cmds, a_leads = jitter_trace(
            rig, ego_ms, dist_m, base_decel, sigma, seed)
        stats = mean_jitter_stats(
            rig, ego_ms, dist_m, base_decel, sigma, seeds, dt)
        out["variants"][name] = {
            "caps": caps, "gain": gain, "ts": ts, "cmds": cmds, "a_leads": a_leads,
            "corner": corner_stats(decels, caps, gain),
            "jitter": stats,
        }
        out.setdefault("rev", rig.describe(name)["rev"])
        out.setdefault("branch", rig.describe(name)["branch"])
        rig.cleanup()
    return out


def text_report(res: dict) -> str:
    lines = [
        f"ACC transition probe  ({res['branch']} {res['rev']})",
        f"{res['ego_kmh']:.0f} km/h, gap {res['dist_m']:.0f} m, "
        f"closing {res['closing_ms']:+.1f} m/s",
        "",
        "corner: cap gain d(cap)/d(a_lead) swept through a_lead = 0",
        f"  {'variant':8} {'worst gain jump':>16} {'at decel':>10} {'gain span':>11}",
    ]
    for name, _ in VARIANTS:
        c = res["variants"][name]["corner"]
        lines.append(f"  {name:8} {c['worst_gain_jump']:>16.4f} "
                     f"{c['at_decel']:>10.3f} {c['gain_span']:>11.4f}")
    lines += [
        "",
        f"chatter: lead holding -{res['base_decel']:.1f} m/s^2 with "
        f"{res['sigma']:.2f} m/s^2 sample-and-hold noise, "
        f"mean of {res['seeds']} seeds",
        f"  {'variant':8} {'rms jerk':>10} {'peak jerk':>11} {'cmd std':>9} "
        f"{'mean cmd':>10}",
    ]
    for name, _ in VARIANTS:
        j = res["variants"][name]["jitter"]
        lines.append(f"  {name:8} {j['rms_jerk']:>10.3f} {j['peak_jerk']:>11.3f} "
                     f"{j['cmd_std']:>9.4f} {j['mean_cmd']:>10.4f}")
    base, after = (res["variants"][n]["jitter"] for n, _ in VARIANTS)
    lines += [
        "",
        f"rms jerk {base['rms_jerk']:.3f} -> {after['rms_jerk']:.3f} m/s^3 "
        f"({100.0 * (after['rms_jerk'] / max(base['rms_jerk'], 1e-9) - 1.0):+.0f}%)",
        f"mean command {base['mean_cmd']:+.4f} -> {after['mean_cmd']:+.4f} m/s^2 "
        "(zero-mean noise, so the gap is rectification)",
    ]
    return "\n".join(lines)


def render(res: dict, out_path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {"HEAD": "#c0392b", "current": "#1f6fb4"}
    fig, axes = plt.subplots(2, 2, figsize=(13.0, 8.0), layout="constrained")
    decels = res["decels"]

    for name, _ in VARIANTS:
        v = res["variants"][name]
        axes[0][0].plot(decels, v["caps"], color=colors[name], lw=1.6, label=name)
        axes[0][1].plot(decels, v["gain"], color=colors[name], lw=1.6, label=name)
        axes[1][0].plot(v["ts"], v["cmds"], color=colors[name], lw=1.0, label=name)
        jerk = np.diff(v["cmds"]) / (v["ts"][1] - v["ts"][0])
        axes[1][1].plot(v["ts"][1:], jerk, color=colors[name], lw=0.8, label=name)

    axes[0][0].set_title("cap through the transition zone", fontsize=9)
    axes[0][0].set_xlabel("lead deceleration (m/s$^2$)", fontsize=8)
    axes[0][0].set_ylabel("ACC accel cap (m/s$^2$)", fontsize=8)
    axes[0][1].set_title("gain  d(cap) / d(lead decel):  the step is the problem",
                         fontsize=9)
    axes[0][1].set_xlabel("lead deceleration (m/s$^2$)", fontsize=8)
    axes[0][1].set_ylabel("gain (m/s$^2$ per m/s$^2$)", fontsize=8)
    axes[1][0].set_title(
        f"command, lead at -{res['base_decel']:.1f} with "
        f"{res['sigma']:.2f} m/s$^2$ telemetry noise", fontsize=9)
    axes[1][0].set_xlabel("time (s)", fontsize=8)
    axes[1][0].set_ylabel("ACC accel cap (m/s$^2$)", fontsize=8)
    axes[1][1].set_title("commanded jerk under the same noise", fontsize=9)
    axes[1][1].set_xlabel("time (s)", fontsize=8)
    axes[1][1].set_ylabel("jerk (m/s$^3$)", fontsize=8)

    for ax in axes.ravel():
        ax.axhline(0.0, color="#4d4d4d", lw=0.6, ls=":")
        ax.tick_params(labelsize=7)
        ax.legend(fontsize=8, frameon=False)
        plt.setp(ax.spines.values(), color="#9a9a9a")
    for ax in (axes[0][0], axes[0][1]):
        ax.axvline(0.0, color="#4d4d4d", lw=0.6, ls=":")

    fig.suptitle("ACC transition smoothness at small lead braking    "
                 f"({res['branch']} {res['rev']})", fontsize=11, fontweight="bold")
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def build_parser() -> argparse.ArgumentParser:
    here = Path(__file__).resolve().parent
    p = argparse.ArgumentParser(
        description="Smoothness of the ACC command at small lead deceleration.")
    p.add_argument("--repo", default=str(here.parent))
    p.add_argument("--report", choices=("text", "png", "both"), default="both")
    p.add_argument("--out", default="")
    p.add_argument("--gap-level", type=int, default=3)
    p.add_argument("--ego-kmh", type=float, default=80.0)
    p.add_argument("--dist-m", type=float, default=40.0)
    p.add_argument("--closing-ms", type=float, default=0.5)
    p.add_argument("--base-decel", type=float, default=0.30,
                   help="lead decel to hover at; default sits on the deadband")
    p.add_argument("--sigma", type=float, default=0.35,
                   help="telemetry noise on a_lead, m/s^2")
    p.add_argument("--seed", type=int, default=7,
                   help="noise seed for the plotted trace only")
    p.add_argument("--seeds", type=int, default=12,
                   help="how many seeds the reported statistics average over")
    p.add_argument("--dt", type=float, default=1.0 / 30.0)
    return p


def main(argv: list[str] | None = None) -> int:
    here = Path(__file__).resolve().parent
    args = build_parser().parse_args(argv)
    res = run(Path(args.repo), args.gap_level, args.ego_kmh, args.dist_m,
              args.closing_ms, args.base_decel, args.sigma, args.seed, args.dt,
              args.seeds)
    if args.report in ("text", "both"):
        print(text_report(res))
    if args.report in ("png", "both"):
        out = Path(args.out) if args.out else (
            here / "acc_response_map_out" / "acc_transition.png")
        out.parent.mkdir(parents=True, exist_ok=True)
        render(res, out)
        print(f"png  {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
