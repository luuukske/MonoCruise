"""Plot coast-down raw decel vs speed. Ignores slope (noisy on hilly roads).

Expects coast_debug.csv in project root.
Output: coast_plot.png next to the CSV.
"""
from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CSV_PATH = PROJECT_ROOT / "coast_debug.csv"
OUT_PATH = PROJECT_ROOT / "coast_plot.png"


def main() -> None:
    speed: list[float] = []
    raw_a: list[float] = []
    clutch: list[float] = []
    gear: list[int] = []

    with CSV_PATH.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                v = float(row["speed_ms"])
                a = float(row["raw_accel_ms2"])
                c = float(row["game_clutch"])
                g = int(row["gear"])
            except (KeyError, ValueError):
                continue
            if v < 1.0:
                continue
            speed.append(v)
            raw_a.append(a)
            clutch.append(c)
            gear.append(g)

    speed_arr = np.array(speed)
    decel_arr = -np.array(raw_a)  # positive = decelerating
    clutch_arr = np.array(clutch)
    gear_arr = np.array(gear)

    engine_disengaged = (clutch_arr > 0.5) | (gear_arr == 0)
    engine_engaged = ~engine_disengaged

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(
        speed_arr[engine_engaged], decel_arr[engine_engaged],
        s=6, alpha=0.35, color="tab:blue", label=f"engine engaged (n={engine_engaged.sum()})",
    )
    ax.scatter(
        speed_arr[engine_disengaged], decel_arr[engine_disengaged],
        s=6, alpha=0.55, color="tab:orange", label=f"clutch in / neutral (n={engine_disengaged.sum()})",
    )

    # Binned median of engine-engaged samples (typical cruising case)
    if engine_engaged.sum() > 20:
        v_eng = speed_arr[engine_engaged]
        d_eng = decel_arr[engine_engaged]
        edges = np.linspace(v_eng.min(), v_eng.max(), 20)
        centers = 0.5 * (edges[:-1] + edges[1:])
        medians = []
        for lo, hi in zip(edges[:-1], edges[1:]):
            mask = (v_eng >= lo) & (v_eng < hi)
            medians.append(np.median(d_eng[mask]) if mask.sum() else np.nan)
        medians = np.array(medians)
        ok = ~np.isnan(medians)
        ax.plot(centers[ok], medians[ok], color="tab:red", lw=2, label="binned median (engine engaged)")

    ax.set_xlabel("speed (m/s)")
    ax.set_ylabel("deceleration (m/s²)  [positive = slowing down]")
    ax.set_title("Coast-down decel vs speed  (slope ignored: hilly road)")
    ax.axhline(0.0, color="gray", lw=0.5)
    ax.grid(alpha=0.3)
    ax.legend(loc="best")

    fig.tight_layout()
    fig.savefig(OUT_PATH, dpi=120)
    print(f"wrote {OUT_PATH}")


if __name__ == "__main__":
    main()

