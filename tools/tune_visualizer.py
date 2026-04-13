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


def _load(path: Path, window_s: float) -> pd.DataFrame:
    try:
        df = pd.read_csv(path, low_memory=False)
    except (FileNotFoundError, pd.errors.EmptyDataError):
        return pd.DataFrame()
    for col in df.columns:
        if col not in ("utc", "pedal_state"):
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


def build_figure() -> tuple[plt.Figure, list[plt.Axes]]:
    fig, axes = plt.subplots(4, 1, figsize=(14, 10), sharex=True)
    fig.suptitle("accel_to_pedals live tuning", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    return fig, axes


def update_plots(axes: list[plt.Axes], df: pd.DataFrame) -> None:
    if df.empty or "t_s" not in df.columns:
        return

    t = df["t_s"]

    # Panel 1: acceleration signals
    ax = axes[0]
    ax.cla()
    ax.set_title("Acceleration (m/s\u00b2)", fontsize=9)
    ax.axhline(0, color="gray", lw=0.5)
    if "wanted_smooth" in df.columns:
        ax.plot(t, df["wanted_smooth"], label="wanted_smooth", lw=1.2, color="steelblue")
    if "raw_smooth" in df.columns:
        ax.plot(t, df["raw_smooth"], label="raw_smooth", lw=1.0, color="darkorange", alpha=0.8)
    if "error_ms2" in df.columns:
        ax.plot(t, df["error_ms2"], label="error (raw\u2212wanted)", lw=0.8, color="purple", alpha=0.7)
    if "road_load_ms2" in df.columns:
        ax.plot(t, df["road_load_ms2"], label="road_load", lw=0.7, color="gray", ls="--", alpha=0.6)
    _shade_gearshifts(ax, df)
    ax.legend(fontsize=7, loc="upper left", ncol=4)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))

    # Panel 2: Gas PID terms
    ax = axes[1]
    ax.cla()
    ax.set_title("Gas PID (pedal units)", fontsize=9)
    ax.axhline(0, color="gray", lw=0.5)
    if "gas_p" in df.columns:
        ax.plot(t, df["gas_p"], label="P", lw=1.0, color="steelblue")
    if "gas_i" in df.columns:
        ax.plot(t, df["gas_i"], label="I", lw=1.2, color="crimson")
    if "gas_d" in df.columns:
        ax.plot(t, df["gas_d"], label="D", lw=0.8, color="green", alpha=0.8)
    if "gas_cmd" in df.columns:
        ax.plot(t, df["gas_cmd"], label="gas_cmd", lw=1.4, color="black", ls="--")
    _shade_gearshifts(ax, df)
    ax.legend(fontsize=7, loc="upper left", ncol=4)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.3f"))

    # Panel 3: Brake feedforward + trim + pedal commands
    ax = axes[2]
    ax.cla()
    ax.set_title("Brake FF + trim / pedal commands", fontsize=9)
    ax.axhline(0, color="gray", lw=0.5)
    if "brake_ff" in df.columns:
        ax.plot(t, df["brake_ff"], label="brake FF", lw=1.0, color="salmon")
    if "brake_trim_i" in df.columns:
        ax.plot(t, df["brake_trim_i"], label="brake trim I", lw=0.8, color="firebrick", alpha=0.8)
    if "brake_cmd" in df.columns:
        ax.plot(t, -df["brake_cmd"], label="\u2212brake_cmd", lw=1.2, color="red")
    if "gas_cmd" in df.columns:
        ax.plot(t, df["gas_cmd"], label="gas_cmd", lw=1.2, color="green")
    if "game_throttle" in df.columns:
        ax.plot(t, df["game_throttle"], label="game_throttle", lw=0.8, color="limegreen", ls="--", alpha=0.8)
    if "game_clutch" in df.columns:
        ax.fill_between(t, df["game_clutch"], alpha=0.2, color=_GEARSHIFT_COLOR, label="game_clutch")
    ax.set_ylim(-1.05, 1.05)
    _shade_gearshifts(ax, df)
    ax.legend(fontsize=7, loc="upper left", ncol=5)

    # Panel 4: speed + gear + gain_scale + brake_multiplier
    ax = axes[3]
    ax.cla()
    ax.set_title("Speed / gear / gain_scale / brake_mult", fontsize=9)
    if "speed_ms" in df.columns:
        ax.plot(t, df["speed_ms"], label="speed (m/s)", lw=1.2, color="navy")
    ax.set_xlabel("t (s)", fontsize=8)
    ax3 = ax.twinx()
    if "gain_scale" in df.columns:
        ax3.plot(t, df["gain_scale"], label="gain_scale", lw=0.8, color="darkorange", ls="--")
    if "brake_multiplier" in df.columns:
        ax3.plot(t, df["brake_multiplier"], label="brake_mult", lw=0.8, color="firebrick", ls="--")
    if "gear" in df.columns:
        ax3.scatter(t, df["gear"] * 0.1, s=4, color="gray", alpha=0.4, label="gear\u00d70.1")
    ax3.set_ylabel("scale / gear", fontsize=7)
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
