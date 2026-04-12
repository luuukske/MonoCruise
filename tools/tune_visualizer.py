"""Live tuning visualizer for accel_to_pedals_debug.csv.

Run from the project root:
    .venv/Scripts/python tools/tune_visualizer.py

Reads accel_to_pedals_debug.csv and updates plots every second.
Shows the last N seconds of data (default 60s, set with --window).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import pandas as pd
from matplotlib.animation import FuncAnimation

_DEFAULT_CSV = Path(__file__).resolve().parents[1] / "accel_to_pedals_debug.csv"
_UPDATE_MS = 1000  # refresh interval in ms
_GEARSHIFT_COLOR = "#ff9933"
_INTEGRAL_CLAMP_COLOR = "#cc0000"


def _load(path: Path, window_s: float) -> pd.DataFrame:
    try:
        df = pd.read_csv(path, low_memory=False)
    except (FileNotFoundError, pd.errors.EmptyDataError):
        return pd.DataFrame()
    for col in df.columns:
        if col not in ("utc",):
            df[col] = pd.to_numeric(df[col], errors="coerce")
    if "t_s" not in df.columns or df.empty:
        return df
    t_max = df["t_s"].max()
    return df[df["t_s"] >= t_max - window_s].copy()


def _shade_gearshifts(ax: plt.Axes, df: pd.DataFrame) -> None:
    """Shade regions where gearshift_active == 1."""
    if "gearshift_active" not in df.columns:
        return
    in_shift = False
    start = None
    for _, row in df.iterrows():
        active = row["gearshift_active"] == 1
        if active and not in_shift:
            start = row["t_s"]
            in_shift = True
        elif not active and in_shift:
            ax.axvspan(start, row["t_s"], alpha=0.15, color=_GEARSHIFT_COLOR, zorder=0)
            in_shift = False
    if in_shift and start is not None:
        ax.axvspan(start, df["t_s"].max(), alpha=0.15, color=_GEARSHIFT_COLOR, zorder=0)


def _mark_clamped(ax: plt.Axes, df: pd.DataFrame, y_col: str) -> None:
    """Mark rows where integral was clamped."""
    if "integral_clamped" not in df.columns or y_col not in df.columns:
        return
    clamped = df[df["integral_clamped"] == 1]
    ax.scatter(
        clamped["t_s"], clamped[y_col],
        color=_INTEGRAL_CLAMP_COLOR, s=12, zorder=5, label="clamped", marker="x",
    )


def build_figure() -> tuple[plt.Figure, list[plt.Axes]]:
    fig, axes = plt.subplots(4, 1, figsize=(14, 10), sharex=True)
    fig.suptitle("accel_to_pedals live tuning", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    return fig, axes


def update_plots(axes: list[plt.Axes], df: pd.DataFrame) -> None:
    if df.empty or "t_s" not in df.columns:
        return

    t = df["t_s"]

    # ── Panel 1: acceleration signals ─────────────────────────────────────
    ax = axes[0]
    ax.cla()
    ax.set_title("Acceleration (m/s²)", fontsize=9)
    ax.axhline(0, color="gray", lw=0.5)
    if "wanted_smooth" in df.columns:
        ax.plot(t, df["wanted_smooth"], label="wanted_smooth", lw=1.2, color="steelblue")
    if "raw_smooth" in df.columns:
        ax.plot(t, df["raw_smooth"], label="raw_smooth", lw=1.0, color="darkorange", alpha=0.8)
    if "error_ms2" in df.columns:
        ax.plot(t, df["error_ms2"], label="error (raw−wanted)", lw=0.8, color="purple", alpha=0.7)
    if "road_load_ms2" in df.columns:
        ax.plot(t, df["road_load_ms2"], label="road_load", lw=0.7, color="gray", ls="--", alpha=0.6)
    _shade_gearshifts(ax, df)
    ax.legend(fontsize=7, loc="upper left", ncol=4)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))

    # ── Panel 2: integral and factor ──────────────────────────────────────
    ax = axes[1]
    ax.cla()
    ax.set_title("Integral correction + factor", fontsize=9)
    ax.axhline(0, color="gray", lw=0.5)
    if "integral_correction" in df.columns:
        ax.plot(t, df["integral_correction"], label="integral_correction", lw=1.2, color="crimson")
        _mark_clamped(ax, df, "integral_correction")
    ax2 = ax.twinx()
    if "integral_factor" in df.columns:
        ax2.fill_between(t, df["integral_factor"], alpha=0.15, color="green", label="integral_factor")
        ax2.set_ylim(-0.05, 1.2)
        ax2.set_ylabel("factor", fontsize=7)
        ax2.tick_params(labelsize=7)
    _shade_gearshifts(ax, df)
    ax.legend(fontsize=7, loc="upper left")
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.3f"))

    # ── Panel 3: pedal commands vs game response ──────────────────────────
    ax = axes[2]
    ax.cla()
    ax.set_title("Pedal commands vs game response", fontsize=9)
    if "gas_cmd" in df.columns:
        ax.plot(t, df["gas_cmd"], label="gas_cmd", lw=1.2, color="green")
    if "brake_cmd" in df.columns:
        ax.plot(t, -df["brake_cmd"], label="−brake_cmd", lw=1.2, color="red")
    if "game_throttle" in df.columns:
        ax.plot(t, df["game_throttle"], label="game_throttle", lw=0.8, color="limegreen", ls="--", alpha=0.8)
    if "game_clutch" in df.columns:
        ax.fill_between(t, df["game_clutch"], alpha=0.2, color=_GEARSHIFT_COLOR, label="game_clutch")
    ax.axhline(0, color="gray", lw=0.5)
    ax.set_ylim(-1.05, 1.05)
    _shade_gearshifts(ax, df)
    ax.legend(fontsize=7, loc="upper left", ncol=4)

    # ── Panel 4: speed + gear + sensitivity ───────────────────────────────
    ax = axes[3]
    ax.cla()
    ax.set_title("Speed / gear / sensitivity", fontsize=9)
    if "speed_ms" in df.columns:
        ax.plot(t, df["speed_ms"], label="speed (m/s)", lw=1.2, color="navy")
    ax.set_xlabel("t (s)", fontsize=8)
    ax3 = ax.twinx()
    if "accel_sensitivity" in df.columns:
        ax3.plot(t, df["accel_sensitivity"], label="accel_sens", lw=0.8, color="darkorange", ls="--")
    if "brake_sensitivity" in df.columns:
        ax3.plot(t, df["brake_sensitivity"], label="brake_sens", lw=0.8, color="firebrick", ls="--")
    if "gear" in df.columns:
        ax3.scatter(t, df["gear"] * 0.1, s=4, color="gray", alpha=0.4, label="gear×0.1")
    ax3.set_ylabel("sensitivity / gear", fontsize=7)
    ax3.tick_params(labelsize=7)
    _shade_gearshifts(ax, df)
    ax.legend(fontsize=7, loc="upper left")
    ax3.legend(fontsize=7, loc="upper right")


def main() -> None:
    parser = argparse.ArgumentParser(description="Live visualizer for accel_to_pedals_debug.csv")
    parser.add_argument("--csv", type=Path, default=_DEFAULT_CSV, help="Path to CSV file")
    parser.add_argument("--window", type=float, default=60.0, help="Time window in seconds (default 60)")
    parser.add_argument("--interval", type=int, default=_UPDATE_MS, help="Refresh interval ms (default 1000)")
    args = parser.parse_args()

    if not args.csv.exists():
        print(f"CSV not found: {args.csv}", file=sys.stderr)
        print("Start MonoCruise with cruise control active to generate the file.")
        sys.exit(1)

    fig, axes = build_figure()

    orange_patch = plt.matplotlib.patches.Patch(color=_GEARSHIFT_COLOR, alpha=0.3, label="gearshift")
    fig.legend(handles=[orange_patch], loc="lower right", fontsize=7)

    def _animate(_frame: int) -> None:
        df = _load(args.csv, args.window)
        update_plots(axes, df)
        fig.tight_layout(rect=[0, 0, 1, 0.97])

    _anim = FuncAnimation(fig, _animate, interval=args.interval, cache_frame_data=False)
    plt.show()


if __name__ == "__main__":
    main()
