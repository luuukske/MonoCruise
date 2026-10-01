"""ACC response map: what the gap law commands, as a function of the lead's state.

Full usage, agent recipes and context-budget guidance live in `tools/README.md`.
The short version:

* The surface is the accel cap over closing speed (`v_ego - v_lead`, positive =
  catching up) and lead deceleration (positive = braking harder).
* It runs the real `AdaptiveCruiseController`, not a reimplementation: a stub
  `acc_thread` goes in the registry, one synthetic lead is published per cell,
  and `accel_cap_ms2` is called on a patched clock.
* `--report text` prints numbers and an ASCII field and renders nothing. Use it
  when the caller is an agent; a PNG costs far more context than it returns.

Three query kinds, picked by flag:

* map (default): the 2D grid, optionally diffed against a baseline `.npz`.
* `--probe`: exact values at named points, no grid, no image.
* `--trace`: closed-loop run of a lead braking from matched speed, which is what
  measures response lag rather than steady-state shape.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from acc_probe_rig import Rig, run_probes, run_sweep, run_trace

# How close to the free-flow cap counts as "ACC is not binding here". Bit
# equality never happens: a lead at any modelled range still moves the cap.
NOT_BINDING_TOL_MS2 = 0.05

CONTOUR_LEVELS = (-6.0, -5.0, -4.0, -3.0, -2.0, -1.0, -0.5, 0.5, 1.0)
DELTA_CONTOURS = (-1.0, -0.5, -0.25, 0.25, 0.5, 1.0)

# Closing speeds and lead decels the text report reports sensitivity at.
SENS_CLOSING = (0.0, 2.0, 5.0, 10.0)
SENS_DECEL = (0.0, 2.0, 4.0, 6.0)

ASCII_BANDS = ((0.5, "+"), (-0.5, ":"), (-1.0, "1"), (-2.0, "2"), (-3.0, "3"),
               (-4.0, "4"), (-5.0, "5"), (-6.0, "6"))
ASCII_FLOOR = "#"
ASCII_LEGEND = (
    "  + >= +0.5   : -0.5..+0.5   1..6 = that many m/s^2 of braking   "
    "# <= -6   (blank = lead speed would be negative)"
)

SCENARIOS: dict[str, dict] = {
    "highway": {
        "kind": "map", "ego_kmh": [80.0], "dist_m": [20.0, 40.0, 80.0],
        "doc": "one speed, three gaps around the level-3 wanted gap",
    },
    "highway-pair": {
        "kind": "map", "ego_kmh": [50.0, 80.0], "dist_m": [20.0, 40.0, 80.0],
        "doc": "two speeds by three gaps, the default comparison grid",
    },
    "town": {
        "kind": "map", "ego_kmh": [30.0, 50.0], "dist_m": [10.0, 20.0, 40.0],
        "doc": "low speed, short gaps",
    },
    "cutin": {
        "kind": "map", "ego_kmh": [80.0], "dist_m": [12.0, 18.0, 25.0],
        "vrel_min": -2.0, "vrel_max": 8.0, "decel_min": 0.0, "decel_max": 6.0,
        "doc": "someone just took your gap: inside the wanted gap, low closing",
    },
    "matched-speed": {
        "kind": "map", "ego_kmh": [50.0, 80.0], "dist_m": [30.0, 40.0, 60.0],
        "vrel_min": -1.0, "vrel_max": 3.0, "decel_min": 0.0, "decel_max": 8.0,
        "doc": "zoom on the band where lead braking should matter",
    },
    "brake-onset": {
        "kind": "trace",
        "traces": [
            "ego=80,gap=40,decel=2", "ego=80,gap=40,decel=4", "ego=80,gap=40,decel=6",
            "ego=80,gap=30,decel=2", "ego=80,gap=30,decel=4", "ego=80,gap=30,decel=6",
            "ego=50,gap=26,decel=2", "ego=50,gap=26,decel=4", "ego=50,gap=26,decel=6",
        ],
        "doc": "closed loop: how long after the lead brakes does the cap respond",
    },
    "stopped-lead": {
        "kind": "probe",
        "probes": [
            f"ego={e},gap={g},closing={e / 3.6:.4f},decel=0"
            for e in (20, 30, 50, 70, 90)
            for g in (20, 30, 50, 80, 120, 160, 200)
        ],
        "doc": "cap against a stopped, non-accelerating lead, by speed and gap",
    },
}




def _panel_conditions(res: dict) -> list[tuple[float, float]]:
    return [(e, d) for e in res["ego_kmh"] for d in res["dist_m"]]


def _nearest(vals: np.ndarray, target: float) -> int:
    return int(np.abs(vals - target).argmin())


def panel_ceiling(res: dict, idx: int) -> float:
    """Measured no-lead cap for this panel, or the constant for an older .npz."""
    ceilings = res.get("ceilings")
    if ceilings is None or idx >= len(ceilings):
        return float(res["ceiling"])
    return float(ceilings[idx])


def panel_stats(res: dict, idx: int) -> dict:
    """Per-panel numbers, including how much lead decel moves the command."""
    grid = res["grids"][idx]
    x_vals, y_vals = res["x_vals"], res["y_vals"]
    finite = grid[np.isfinite(grid)]

    decel_sens = {}
    for closing in SENS_CLOSING:
        if not (x_vals[0] <= closing <= x_vals[-1]):
            continue
        col = grid[:, _nearest(x_vals, closing)]
        col = col[np.isfinite(col)]
        if col.size:
            decel_sens[f"{closing:g}"] = float(col.max() - col.min())

    closing_sens = {}
    for decel in SENS_DECEL:
        if not (y_vals[0] <= decel <= y_vals[-1]):
            continue
        row = grid[_nearest(y_vals, decel), :]
        row = row[np.isfinite(row)]
        if row.size:
            closing_sens[f"{decel:g}"] = float(row.max() - row.min())

    zero_at = None
    row0 = grid[_nearest(y_vals, 0.0), :]
    sign = np.where(np.isfinite(row0), np.sign(row0), np.nan)
    flips = np.nonzero(np.diff(sign) != 0)[0]
    if flips.size:
        zero_at = float(x_vals[flips[0]])

    n = max(finite.size, 1)
    return {
        "min_ms2": float(finite.min()) if finite.size else None,
        "max_ms2": float(finite.max()) if finite.size else None,
        "zero_crossing_closing_ms": zero_at,
        "span_over_decel_axis": decel_sens,
        "span_over_closing_axis": closing_sens,
        "frac_at_decel_clamp": float((finite <= res["cap_lo"] + 1e-3).sum() / n),
        "frac_at_ceiling": float(
            (finite >= panel_ceiling(res, idx) - NOT_BINDING_TOL_MS2).sum() / n),
    }


def ascii_field(res: dict, idx: int, width: int = 34, height: int = 13) -> list[str]:
    grid = res["grids"][idx]
    x_vals, y_vals = res["x_vals"], res["y_vals"]
    cols = np.linspace(0, x_vals.size - 1, min(width, x_vals.size)).astype(int)
    rows = np.linspace(0, y_vals.size - 1, min(height, y_vals.size)).astype(int)

    lines = []
    for iy in rows[::-1]:
        chars = []
        for ix in cols:
            v = grid[iy, ix]
            if not np.isfinite(v):
                chars.append(" ")
                continue
            sym = ASCII_FLOOR
            for threshold, glyph in ASCII_BANDS:
                if v >= threshold:
                    sym = glyph
                    break
            chars.append(sym)
        lines.append(f"  decel {y_vals[iy]:+5.1f} |{''.join(chars)}")
    ruler = " " * 14 + "+" + "-" * len(cols)
    axis = f"{' ' * 15}closing {x_vals[0]:+.1f}"
    axis += " " * max(1, len(cols) - len(axis) + 6) + f"{x_vals[-1]:+.1f} m/s"
    lines.extend([ruler, axis])
    return lines


def text_report(res: dict, compare: dict | None, show_ascii: bool) -> str:
    out: list[str] = []
    out.append(f"ACC response map  {res['label']}  ({res['branch']} {res['rev']})")
    out.append(f"gap level {res['gap_level']}, T = {res['headway_s']:.2f} s, "
               f"lead score {res['score']:.1f} (anticipation inactive, single lead)")
    out.append(f"axes: closing {res['x_vals'][0]:+.1f}..{res['x_vals'][-1]:+.1f} m/s, "
               f"decel {res['y_vals'][0]:+.1f}..{res['y_vals'][-1]:+.1f} m/s^2, "
               f"{res['x_vals'].size}x{res['y_vals'].size}")
    out.append(f"settle residual {res['residual']:.1e} m/s^2 at 4x ticks")
    out.append("")

    for idx, (ego, dist) in enumerate(_panel_conditions(res)):
        st = panel_stats(res, idx)
        out.append(f"panel {ego:.0f} km/h, gap {dist:.0f} m")
        out.append(f"  cap {st['min_ms2']:+.2f} .. {st['max_ms2']:+.2f} m/s^2"
                   + (f", zero at closing {st['zero_crossing_closing_ms']:+.1f} m/s"
                      if st["zero_crossing_closing_ms"] is not None else ""))
        spans = ", ".join(f"closing {k}: {v:.2f}"
                          for k, v in st["span_over_decel_axis"].items())
        out.append(f"  cap span over the whole decel axis   [{spans}]")
        spans = ", ".join(f"decel {k}: {v:.2f}"
                          for k, v in st["span_over_closing_axis"].items())
        out.append(f"  cap span over the whole closing axis [{spans}]")
        out.append(f"  at decel clamp {st['frac_at_decel_clamp'] * 100:.1f}%, "
                   f"at ceiling {st['frac_at_ceiling'] * 100:.1f}%")
        if compare is not None:
            d = res["grids"][idx] - compare["grids"][idx]
            dv = d[np.isfinite(d)]
            out.append(f"  vs {compare['label']}: mean {dv.mean():+.3f}, "
                       f"softer by up to {dv.max():+.3f}, firmer by up to {dv.min():+.3f}")
        if show_ascii:
            out.extend(ascii_field(res, idx))
        out.append("")

    if show_ascii:
        out.append(ASCII_LEGEND)
    return "\n".join(out)


X_LABEL = "closing speed  $v_{ego} - v_{lead}$  (m/s)"
Y_LABEL = "lead deceleration (m/s$^2$)"


def _shade(base, grid, res, norm, ax, contours, ceiling=None):
    x_vals, y_vals = res["x_vals"], res["y_vals"]
    cmap = base.copy()
    cmap.set_bad("#e4e4e8")
    im = ax.imshow(
        np.ma.masked_invalid(grid), origin="lower", aspect="auto",
        extent=(x_vals[0], x_vals[-1], y_vals[0], y_vals[-1]),
        cmap=cmap, norm=norm, interpolation="nearest",
    )
    if not contours:
        return im
    xx, yy = np.meshgrid(x_vals, y_vals)
    with np.errstate(invalid="ignore"):
        levels = list(DELTA_CONTOURS if contours == "delta" else CONTOUR_LEVELS)
        inside = [lv for lv in levels
                  if np.nanmin(grid) < lv < np.nanmax(grid)]
        if inside:
            cs = ax.contour(xx, yy, grid, levels=inside,
                            colors="#1a1a1a", linewidths=0.5, alpha=0.5)
            ax.clabel(cs, inline=True, fontsize=6, fmt="%.2g")
        if np.nanmin(grid) < 0.0 < np.nanmax(grid):
            zero = ax.contour(xx, yy, grid, levels=[0.0],
                              colors="#111111", linewidths=1.6)
            ax.clabel(zero, inline=True, fontsize=7, fmt="0")
        if contours != "delta":
            idle = (grid >= (res["ceiling"] if ceiling is None else ceiling)
                    - NOT_BINDING_TOL_MS2).astype(float)
            if idle.max() > 0.5:
                ax.contourf(xx, yy, idle, levels=[0.5, 1.5],
                            colors="none", hatches=["////"])
    return im


def _decorate(ax, title, *, show_x: bool, show_y: bool):
    import matplotlib.pyplot as plt

    ax.axhline(0.0, color="#4d4d4d", lw=0.6, ls=":")
    ax.axvline(0.0, color="#4d4d4d", lw=0.6, ls=":")
    ax.set_title(title, fontsize=8.5)
    ax.tick_params(labelsize=7)
    ax.set_xlabel(X_LABEL if show_x else "", fontsize=8)
    ax.set_ylabel(Y_LABEL if show_y else "", fontsize=8)
    plt.setp(ax.spines.values(), color="#9a9a9a")


def _subtitle(res: dict, compare: dict | None = None, delta_only: bool = False) -> str:
    head = f"gap level {res['gap_level']}  (T = {res['headway_s']:.2f} s)"
    if compare is not None:
        head += (
            f" vs level {compare['gap_level']} (T = {compare['headway_s']:.2f} s) "
            "on the baseline"
        )
        if abs(res["headway_s"] - compare["headway_s"]) > 1e-6:
            head += "   [HEADWAY DIFFERS, the change column mixes two effects]"
    head += ("   |   "
             f"one lead, tracker score {res['score']:.1f}, so anticipation is inactive\n")
    if delta_only:
        return head + (
            "purple = ACC brakes harder now, orange = ACC brakes less now, "
            "black line = no change.   grey = lead speed would be negative"
        )
    return head + (
        "red = ACC braking, blue = ACC allowing accel, black line = zero.   "
        "hatched = cap at the no-lead ceiling (ACC not binding), "
        "grey = lead speed would be negative"
    )


def _delta_pack(res, compare):
    from matplotlib.colors import Normalize

    delta = res["grids"] - compare["grids"]
    span = float(np.nanmax(np.abs(delta))) or 1.0
    return delta, Normalize(vmin=-span, vmax=span)


def _render_single(res):
    import matplotlib.pyplot as plt
    from matplotlib.colors import TwoSlopeNorm

    conds = _panel_conditions(res)
    ncols, nrows = len(res["dist_m"]), len(res["ego_kmh"])
    norm = TwoSlopeNorm(vmin=res["cap_lo"], vcenter=0.0, vmax=res["cap_hi"])
    cmap = plt.get_cmap("RdBu")

    fig, axes = plt.subplots(
        nrows, ncols, figsize=(4.0 * ncols + 1.4, 3.7 * nrows + 1.4),
        squeeze=False, layout="constrained",
    )
    im = None
    for idx, (ego, dist) in enumerate(conds):
        row, col = idx // ncols, idx % ncols
        im = _shade(cmap, res["grids"][idx], res, norm, axes[row][col], "cap",
                    panel_ceiling(res, idx))
        _decorate(axes[row][col], f"{ego:.0f} km/h   gap {dist:.0f} m",
                  show_x=row == nrows - 1, show_y=col == 0)
    cbar = fig.colorbar(im, ax=axes, fraction=0.035, pad=0.015)
    cbar.set_label("ACC accel cap (m/s$^2$)", fontsize=8)
    cbar.ax.tick_params(labelsize=7)
    return fig


def _render_delta_only(res, compare):
    import matplotlib.pyplot as plt

    conds = _panel_conditions(res)
    ncols, nrows = len(res["dist_m"]), len(res["ego_kmh"])
    delta, dnorm = _delta_pack(res, compare)
    dcmap = plt.get_cmap("PuOr_r")

    fig, axes = plt.subplots(
        nrows, ncols, figsize=(4.0 * ncols + 1.4, 3.7 * nrows + 1.4),
        squeeze=False, layout="constrained",
    )
    dim = None
    for idx, (ego, dist) in enumerate(conds):
        row, col = idx // ncols, idx % ncols
        dim = _shade(dcmap, delta[idx], res, dnorm, axes[row][col], "delta")
        _decorate(axes[row][col], f"{ego:.0f} km/h   gap {dist:.0f} m",
                  show_x=row == nrows - 1, show_y=col == 0)
    cb = fig.colorbar(dim, ax=axes, fraction=0.035, pad=0.015)
    cb.set_label(f"{res['label']} minus {compare['label']} (m/s$^2$)\n"
                 "purple = more braking now", fontsize=8)
    cb.ax.tick_params(labelsize=7)
    return fig


def _render_compare(res, compare):
    import matplotlib.pyplot as plt
    from matplotlib.colors import TwoSlopeNorm

    conds = _panel_conditions(res)
    nrows = len(conds)
    norm = TwoSlopeNorm(vmin=res["cap_lo"], vcenter=0.0, vmax=res["cap_hi"])
    cmap = plt.get_cmap("RdBu")
    delta, dnorm = _delta_pack(res, compare)
    dcmap = plt.get_cmap("PuOr_r")

    fig, axes = plt.subplots(
        nrows, 3, figsize=(13.5, 3.7 * nrows + 1.4), squeeze=False, layout="constrained",
    )
    im = dim = None
    for idx, (ego, dist) in enumerate(conds):
        cond = f"{ego:.0f} km/h, gap {dist:.0f} m"
        last = idx == nrows - 1
        im = _shade(cmap, res["grids"][idx], res, norm, axes[idx][0], "cap",
                    panel_ceiling(res, idx))
        _decorate(axes[idx][0], f"now: {res['label']}   |   {cond}",
                  show_x=last, show_y=True)
        _shade(cmap, compare["grids"][idx], compare, norm, axes[idx][1], "cap",
               panel_ceiling(compare, idx))
        _decorate(axes[idx][1], f"was: {compare['label']}   |   {cond}",
                  show_x=last, show_y=False)
        dim = _shade(dcmap, delta[idx], res, dnorm, axes[idx][2], "delta")
        _decorate(axes[idx][2], f"change   |   {cond}", show_x=last, show_y=False)

    cb = fig.colorbar(im, ax=axes[:, :2], fraction=0.025, pad=0.015)
    cb.set_label("ACC accel cap (m/s$^2$)", fontsize=8)
    cb.ax.tick_params(labelsize=7)
    cd = fig.colorbar(dim, ax=axes[:, 2], fraction=0.05, pad=0.015)
    cd.set_label("now - was (m/s$^2$):  purple = more braking now", fontsize=8)
    cd.ax.tick_params(labelsize=7)
    return fig


def render(res: dict, out_path: Path, compare: dict | None, delta_only: bool) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams["hatch.linewidth"] = 0.4
    plt.rcParams["hatch.color"] = "#ffffff"

    if compare and delta_only:
        fig = _render_delta_only(res, compare)
        head = f"ACC accel cap change    {res['label']} vs {compare['label']}"
    elif compare:
        fig = _render_compare(res, compare)
        head = f"ACC accel cap    {res['label']} vs {compare['label']}"
    else:
        fig = _render_single(res)
        head = f"ACC accel cap over closing speed and lead braking    {res['label']}"
    if res["rev"]:
        head += f"  ({res['branch']} {res['rev']})"
    fig.suptitle(head, fontsize=11, fontweight="bold")
    fig.text(0.5, -0.01, _subtitle(res, compare, bool(compare and delta_only)),
             ha="center", va="top", fontsize=8, color="#333333")
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def save_npz(res: dict, path: Path) -> None:
    np.savez_compressed(path, **{k: np.asarray(v) for k, v in res.items()})


def load_npz(path: Path) -> dict:
    with np.load(path, allow_pickle=False) as data:
        out = {k: data[k] for k in data.files}
    for key in ("gap_level", "score", "cap_lo", "cap_hi", "ceiling", "residual", "headway_s"):
        if key in out:
            out[key] = out[key].item()
    for key in ("label", "rev", "branch"):
        if key in out:
            out[key] = str(out[key])
    return out


def check_comparable(new: dict, old: dict) -> None:
    for key in ("x_vals", "y_vals", "ego_kmh", "dist_m"):
        if new[key].shape != old[key].shape or not np.allclose(new[key], old[key]):
            raise SystemExit(
                f"baseline {key} does not match this run; re-run both with the same axes",
            )


def summary_json(res: dict, compare: dict | None) -> dict:
    out = {
        "label": res["label"], "rev": res["rev"], "branch": res["branch"],
        "gap_level": res["gap_level"], "headway_s": res["headway_s"],
        "score": res["score"], "residual": res["residual"],
        "axes": {
            "closing_ms": [float(res["x_vals"][0]), float(res["x_vals"][-1])],
            "decel_ms2": [float(res["y_vals"][0]), float(res["y_vals"][-1])],
            "shape": [int(res["y_vals"].size), int(res["x_vals"].size)],
        },
        "panels": [],
    }
    for idx, (ego, dist) in enumerate(_panel_conditions(res)):
        panel = {"ego_kmh": float(ego), "gap_m": float(dist), **panel_stats(res, idx)}
        if compare is not None:
            d = res["grids"][idx] - compare["grids"][idx]
            dv = d[np.isfinite(d)]
            panel["delta"] = {
                "baseline": compare["label"], "mean": float(dv.mean()),
                "max_softer": float(dv.max()), "max_firmer": float(dv.min()),
            }
        out["panels"].append(panel)
    return out


def _apply_scenario(args) -> dict:
    if not args.scenario:
        return {}
    if args.scenario not in SCENARIOS:
        raise SystemExit(
            f"unknown scenario {args.scenario!r}. Available: "
            + ", ".join(f"{k} ({v['doc']})" for k, v in SCENARIOS.items()),
        )
    preset = SCENARIOS[args.scenario]
    given = {a.lstrip("-").replace("-", "_") for a in sys.argv if a.startswith("--")}
    for key, value in preset.items():
        if key in ("kind", "doc", "traces", "probes") or key in given:
            continue
        setattr(args, key, list(value) if isinstance(value, list) else value)
    if preset["kind"] == "trace" and not args.trace:
        args.trace = list(preset["traces"])
    if preset["kind"] == "probe" and not args.probe:
        args.probe = list(preset["probes"])
    return preset


def _print_probes(rows: list[dict]) -> None:
    print(f"{'ego':>6} {'gap':>7} {'closing':>9} {'lead decel':>11} "
          f"{'v_lead':>8} {'cap':>8}")
    for r in rows:
        cap = "masked" if not np.isfinite(r["cap_ms2"]) else f"{r['cap_ms2']:+.3f}"
        print(f"{r['ego_kmh']:>6.0f} {r['gap_m']:>7.1f} {r['closing_ms']:>9.2f} "
              f"{r['lead_decel_ms2']:>11.2f} {r['v_lead_ms']:>8.2f} {cap:>8}")


def _print_traces(rows: list[dict]) -> None:
    print("lead brakes from matched speed, perfect actuator (real lag adds to this)")
    print(f"{'case':26} {'-0.5':>6} {'-1.0':>6} {'-2.0':>6} {'-4.0':>6} "
          f"{'peak':>7} {'min gap':>8} {'clamp':>8}")
    for r in rows:
        case = f"{r['ego_kmh']:.0f} km/h gap {r['gap0_m']:.0f} m, lead -{r['lead_decel_ms2']:.0f}"
        cells = " ".join(
            f"{r['onset_s'][k]:>6.2f}" if r["onset_s"][k] is not None else f"{'-':>6}"
            for k in ("0.5", "1.0", "2.0", "4.0")
        )
        clamp = f"{r['decel_clamp_s']:.2f} s" if r["decel_clamp_s"] is not None else "no"
        print(f"{case:26} {cells} {r['peak_ms2']:>7.2f} {r['min_gap_m']:>8.1f} {clamp:>8}")


def build_parser() -> argparse.ArgumentParser:
    here = Path(__file__).resolve().parent
    p = argparse.ArgumentParser(
        description="ACC accel-cap response map, point probes and closed-loop traces.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="scenarios:\n  " + "\n  ".join(
            f"{k:<15} [{v['kind']}] {v['doc']}" for k, v in SCENARIOS.items()),
    )
    p.add_argument("--repo", default=str(here.parent),
                   help="checkout to probe (default: this script's repo)")
    p.add_argument("--label", default="", help="name for this run in the title")
    p.add_argument("--scenario", default="", help="named preset, see the list below")
    p.add_argument("--report", choices=("text", "png", "both"), default="both",
                   help="text is the cheap one for agents; png renders only the figure")
    p.add_argument("--json", default="", help="write the summary as JSON ('-' for stdout)")
    p.add_argument("--no-ascii", action="store_true", help="drop the ASCII field from --report text")
    p.add_argument("--out", default="", help="PNG path (default: tools/acc_response_map_out/)")
    p.add_argument("--compare", default="", help="baseline .npz to diff against")
    p.add_argument("--delta-only", action="store_true",
                   help="with --compare, render just the change panels")
    p.add_argument("--probe", action="append", default=[],
                   metavar="ego=..,gap=..,closing=..,decel=..",
                   help="exact point query, repeatable; skips the grid entirely")
    p.add_argument("--trace", action="append", default=[],
                   metavar="ego=..,gap=..,decel=..[,horizon=..]",
                   help="closed-loop brake-onset run, repeatable; skips the grid")
    p.add_argument("--ego-kmh", nargs="+", type=float, default=[80.0])
    p.add_argument("--dist-m", nargs="+", type=float, default=[20.0, 40.0, 80.0])
    p.add_argument("--gap-level", type=int, default=3)
    p.add_argument("--vrel-min", type=float, default=-5.0)
    p.add_argument("--vrel-max", type=float, default=15.0)
    p.add_argument("--decel-min", type=float, default=-1.5)
    p.add_argument("--decel-max", type=float, default=8.0)
    p.add_argument("--nx", type=int, default=161)
    p.add_argument("--ny", type=int, default=153)
    p.add_argument("--score", type=float, default=5.0,
                   help="tracker score of the synthetic lead (5 = full confidence)")
    p.add_argument("--settle-ticks", type=int, default=8)
    p.add_argument("--dt", type=float, default=1.0 / 30.0)
    return p


def main(argv: list[str] | None = None) -> int:
    here = Path(__file__).resolve().parent
    parser = build_parser()
    args = parser.parse_args(argv)
    _apply_scenario(args)

    rig = Rig(Path(args.repo), args.gap_level, args.dt, args.settle_ticks, args.score)
    payload: dict = {}

    if args.probe or args.trace:
        if args.probe:
            rows = run_probes(rig, args.probe)
            payload["probes"] = rows
            _print_probes(rows)
        if args.trace:
            rows = [run_trace(rig, spec) for spec in args.trace]
            payload["traces"] = rows
            if args.probe:
                print()
            _print_traces(rows)
        rig.cleanup()
        if args.json:
            blob = json.dumps(payload, indent=2)
            if args.json == "-":
                print(blob)
            else:
                Path(args.json).write_text(blob, encoding="utf-8")
        return 0

    x_vals = np.linspace(args.vrel_min, args.vrel_max, args.nx)
    y_vals = np.linspace(args.decel_min, args.decel_max, args.ny)
    res = run_sweep(rig, args.ego_kmh, args.dist_m, x_vals, y_vals, args.label)
    rig.cleanup()

    baseline = None
    if args.compare:
        baseline = load_npz(Path(args.compare))
        check_comparable(res, baseline)

    out = Path(args.out) if args.out else (here / "acc_response_map_out" / f"{res['label']}.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    npz = out.with_suffix(".npz")
    save_npz(res, npz)

    if args.report in ("png", "both"):
        render(res, out, baseline, args.delta_only)
    if args.report in ("text", "both"):
        print(text_report(res, baseline, not args.no_ascii))
    if args.json:
        blob = json.dumps(summary_json(res, baseline), indent=2)
        if args.json == "-":
            print(blob)
        else:
            Path(args.json).write_text(blob, encoding="utf-8")

    print(f"npz  {npz}")
    if args.report in ("png", "both"):
        print(f"png  {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
