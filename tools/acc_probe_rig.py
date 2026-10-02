"""Measurement half of the ACC probe: load a checkout, feed it a lead, read the cap.

Everything here touches the real controller. `acc_response_map.py` holds the
axes, statistics, rendering and CLI on top of it. Import this directly to build
a new ACC probe without inheriting the map's presentation. See `tools/README.md`.

The rig runs the shipped `AdaptiveCruiseController`: a stub `acc_thread` goes in
the registry, one synthetic lead is published per query, and `time.monotonic` is
patched to a deterministic clock so EMA and jerk state step by a fixed dt.
"""
from __future__ import annotations

import contextlib
import importlib
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

import numpy as np

CONTROLLER_CANDIDATES = (
    "core.cruise_control_thread.acc_controller",
    "core.cruise_control_thread.acc",
    "core.longitudinal.acc_controller",
)

# A closing speed a hair above ego rounds v_lead below zero. Snap those to a
# stopped lead rather than masking the cell, which is a real state.
V_LEAD_TOL_MS: float = 1e-3


class Clock:
    """Deterministic monotonic source so EMA and jerk state step by a fixed dt."""

    def __init__(self, dt: float) -> None:
        self.dt = dt
        self.t = 10_000.0

    def advance(self) -> None:
        self.t += self.dt


@contextlib.contextmanager
def patched_clock(clock: Clock):
    real_mono, real_perf = time.monotonic, time.perf_counter
    time.monotonic = lambda: clock.t
    time.perf_counter = lambda: clock.t
    try:
        yield
    finally:
        time.monotonic = real_mono
        time.perf_counter = real_perf


class StubVehicle:
    def __init__(self, vid: int) -> None:
        self.id = vid
        self.is_parked = False
        self.is_trailer = False


class StubLead:
    def __init__(self, dist_m: float, v_lead_ms: float, a_lead_ms2: float, score: float) -> None:
        self.vehicle = StubVehicle(1)
        self.score = score
        self.dist_m = dist_m
        self.effective_speed_ms = v_lead_ms
        self.effective_accel_ms2 = a_lead_ms2
        self.rel_speed_ms = v_lead_ms


class StubACCData:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.enabled = True
        self.has_lead = False
        self.lead_id = -1
        self.lead_dist_m = 0.0
        self.lead_rel_speed_ms = 0.0
        self.lead_score = 0.0
        self.leads: list[StubLead] = []
        self.indicated_lead = None
        self.blinker_b_eff = 0.0
        self.blinker_lane_offset_m = 0.0
        self.blinker_committed = False
        self.ego_lat_vel_ms = 0.0
        self.t_mono = 0.0

    def publish(self, lead: StubLead | None) -> None:
        with self._lock:
            self.leads = [lead] if lead is not None else []
            self.has_lead = lead is not None
            if lead is not None:
                self.lead_id = lead.vehicle.id
                self.lead_dist_m = lead.dist_m
                self.lead_score = lead.score


class StubACCThread:
    name = "acc_thread"

    def __init__(self) -> None:
        self.data = StubACCData()

    def is_alive(self) -> bool:
        return True


def git(repo: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", *args], cwd=str(repo), capture_output=True, text=True, check=False,
    )
    return result.stdout.strip() if result.returncode == 0 else ""


def import_from_repo(repo: Path):
    """Put `repo` first on the path and import the ACC controller living there."""
    repo = repo.resolve()
    sys.path.insert(0, str(repo))
    core = importlib.import_module("core")
    # `core` is a namespace package, so __file__ is None and __path__ is the truth.
    roots = {Path(p).resolve().parent for p in list(getattr(core, "__path__", []))}
    if repo not in roots:
        raise SystemExit(
            f"imported 'core' from {sorted(str(r) for r in roots)}, not the requested repo. "
            "Run this script by absolute path from outside the other checkout.",
        )

    settings_mod = importlib.import_module("core.settings")
    sandbox = Path(tempfile.mkdtemp(prefix="monocruise-accmap-"))
    real_config = settings_mod.CONFIG_PATH
    settings_mod.CONFIG_PATH = sandbox / "config.json"
    settings_mod.BACKUP_PATH = settings_mod.CONFIG_PATH.with_suffix(
        settings_mod.CONFIG_PATH.suffix + ".bak",
    )
    if Path(real_config).is_file():
        shutil.copy2(str(real_config), str(settings_mod.CONFIG_PATH))

    for name in CONTROLLER_CANDIDATES:
        try:
            mod = importlib.import_module(name)
        except ImportError:
            continue
        cls = getattr(mod, "AdaptiveCruiseController", None)
        if cls is not None and hasattr(cls, "accel_cap_ms2"):
            return mod, cls, settings_mod, sandbox
    raise SystemExit(
        "no AdaptiveCruiseController with accel_cap_ms2 found in "
        + ", ".join(CONTROLLER_CANDIDATES),
    )


def _force_gap_level(settings_mod, level: int) -> int | None:
    """Pin the headway setting so two revisions are compared at the same level."""
    settings_cls = settings_mod.Settings
    if "acc_gap_level" not in getattr(settings_cls, "__dataclass_fields__", {}):
        return None
    instance = settings_cls.instance() if hasattr(settings_cls, "instance") else settings_cls()
    instance.acc_gap_level = int(level)
    return int(settings_cls.acc_gap_level)


def _headway_for(mod, level: int) -> float:
    table = getattr(mod, "T_HEADWAY_BY_LEVEL_S", None)
    if table and 1 <= level < len(table):
        return float(table[level])
    return float(getattr(mod, "T_HEADWAY_S", float("nan")))


def _cfg_value(mod, name: str, fallback_const: str, default: float) -> float:
    cfg = mod.ACConfig() if hasattr(mod, "ACConfig") else None
    return float(getattr(cfg, name, getattr(mod, fallback_const, default)))


class Rig:
    """A loaded revision plus the stub wiring needed to call its controller."""

    def __init__(self, repo: Path, gap_level: int, dt: float, settle: int, score: float) -> None:
        self.repo = Path(repo).resolve()
        self.mod, self.cls, settings_mod, self.sandbox = import_from_repo(self.repo)
        registry = importlib.import_module("core.thread_management.registry").registry
        self.stub = StubACCThread()
        registry.replace(self.stub)
        self.level = _force_gap_level(settings_mod, gap_level)
        self.headway = _headway_for(self.mod, gap_level)
        self.clock = Clock(dt)
        self.dt = dt
        self.settle = settle
        self.score = score
        self.overrides: dict[str, float] = {}
        self.cap_lo = _cfg_value(self.mod, "max_decel_ms2", "MAX_DECEL_MS2", -6.55)
        self.cap_hi = _cfg_value(self.mod, "max_accel_ms2", "MAX_ACCEL_MS2", 1.5)
        self.ceiling = _cfg_value(self.mod, "no_lead_ceiling_ms2", "MAX_ACCEL_MS2", 1.5)

    def controller(self):
        """A fresh controller with `self.overrides` applied to its config."""
        ctrl = self.cls()
        for key, value in self.overrides.items():
            if not hasattr(ctrl.config, key):
                raise SystemExit(f"this revision's ACConfig has no field {key!r}")
            setattr(ctrl.config, key, value)
        return ctrl

    def cap(self, ego_ms: float, dist_m: float, v_lead_ms: float, a_lead_ms2: float,
            settle: int | None = None) -> float:
        """Steady-state accel cap for one constant lead state."""
        self.stub.data.publish(StubLead(dist_m, v_lead_ms, a_lead_ms2, self.score))
        ctrl = self.controller()
        out = float("nan")
        for _ in range(settle if settle is not None else self.settle):
            self.clock.advance()
            out = ctrl.accel_cap_ms2(ego_ms)
        return out

    def describe(self, label: str) -> dict:
        rev = git(self.repo, "rev-parse", "--short", "HEAD")
        # A figure that names a clean rev while plotting a modified tree is
        # misleading, and most so once it is published.
        if rev and git(self.repo, "status", "--porcelain"):
            rev += "+dirty"
        return {
            "gap_level": self.level if self.level is not None else -1,
            "headway_s": self.headway,
            "score": self.score,
            "cap_lo": self.cap_lo,
            "cap_hi": self.cap_hi,
            "ceiling": self.ceiling,
            "label": label or rev or self.repo.name,
            "rev": rev,
            "branch": git(self.repo, "rev-parse", "--abbrev-ref", "HEAD"),
        }

    def cleanup(self) -> None:
        shutil.rmtree(str(self.sandbox), ignore_errors=True)


def panel_ceiling(rig: Rig, ego_ms: float) -> float:
    """The cap with the lead pushed out of range, measured rather than assumed.

    `a_free` falls short of `a_max` by the IIDM speed term, so the configured
    ceiling constant is never reached and testing against it never fires."""
    return rig.cap(ego_ms, 10_000.0, ego_ms, 0.0)


def _sweep_panel(rig: Rig, ego_ms, dist_m, x_vals, y_vals) -> np.ndarray:
    grid = np.full((y_vals.size, x_vals.size), np.nan, dtype=float)
    for iy, decel in enumerate(y_vals):
        a_lead = -float(decel)
        for ix, v_close in enumerate(x_vals):
            v_lead = ego_ms - float(v_close)
            if v_lead < -V_LEAD_TOL_MS:
                continue
            grid[iy, ix] = rig.cap(ego_ms, dist_m, max(0.0, v_lead), a_lead)
    return grid


def _convergence_residual(rig: Rig, ego_ms, dist_m, x_vals, y_vals, grid) -> float:
    """Max change from quadrupling the settle count, on a coarse subsample."""
    worst = 0.0
    for iy in range(0, y_vals.size, 7):
        for ix in range(0, x_vals.size, 7):
            if not np.isfinite(grid[iy, ix]):
                continue
            v_lead = max(0.0, ego_ms - float(x_vals[ix]))
            out = rig.cap(ego_ms, dist_m, v_lead, -float(y_vals[iy]), settle=rig.settle * 4)
            worst = max(worst, abs(out - float(grid[iy, ix])))
    return worst


def run_sweep(rig: Rig, ego_kmh, dist_m, x_vals, y_vals, label: str) -> dict:
    """One grid per (ego speed, gap), plus the metadata a diff needs."""
    panels: list[np.ndarray] = []
    ceilings: list[float] = []
    residual = 0.0
    with patched_clock(rig.clock):
        for speed in ego_kmh:
            for dist in dist_m:
                ego_ms = speed / 3.6
                grid = _sweep_panel(rig, ego_ms, dist, x_vals, y_vals)
                panels.append(grid)
                ceilings.append(panel_ceiling(rig, ego_ms))
                residual = max(residual, _convergence_residual(
                    rig, ego_ms, dist, x_vals, y_vals, grid))
    return {
        "grids": np.stack(panels),
        "ceilings": np.asarray(ceilings, dtype=float),
        "x_vals": x_vals,
        "y_vals": y_vals,
        "ego_kmh": np.asarray(ego_kmh, dtype=float),
        "dist_m": np.asarray(dist_m, dtype=float),
        "residual": residual,
        **rig.describe(label),
    }


def parse_spec(spec: str, allowed: tuple[str, ...]) -> dict[str, float]:
    out: dict[str, float] = {}
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "=" not in part:
            raise SystemExit(f"bad spec fragment {part!r}, expected key=value")
        key, _, value = part.partition("=")
        key = key.strip()
        if key not in allowed:
            raise SystemExit(f"unknown spec key {key!r}, allowed: {', '.join(allowed)}")
        out[key] = float(value)
    return out


def run_probes(rig: Rig, specs: list[str]) -> list[dict]:
    """Exact cap at named states. No grid, so this is the cheapest query."""
    rows = []
    with patched_clock(rig.clock):
        for spec in specs:
            p = parse_spec(spec, ("ego", "gap", "closing", "decel"))
            ego_ms = p.get("ego", 80.0) / 3.6
            gap = p.get("gap", 40.0)
            closing = p.get("closing", 0.0)
            decel = p.get("decel", 0.0)
            v_lead = ego_ms - closing
            cap = (float("nan") if v_lead < -V_LEAD_TOL_MS
                   else rig.cap(ego_ms, gap, max(0.0, v_lead), -decel))
            rows.append({
                "ego_kmh": p.get("ego", 80.0), "gap_m": gap, "closing_ms": closing,
                "lead_decel_ms2": decel, "v_lead_ms": v_lead, "cap_ms2": cap,
            })
    return rows


def run_trace(rig: Rig, spec: str) -> dict:
    """Lead brakes from matched speed. Perfect actuator, so onsets are floors."""
    p = parse_spec(spec, ("ego", "gap", "decel", "horizon"))
    ego_kmh = p.get("ego", 80.0)
    gap0 = p.get("gap", 40.0)
    lead_decel = p.get("decel", 4.0)
    horizon = p.get("horizon", 12.0)

    v_ego = ego_kmh / 3.6
    v_lead = v_ego
    gap = float(gap0)
    ctrl = rig.controller()
    t = 0.0
    marks: dict[float, float] = {}
    peak = 0.0
    min_gap = gap
    slam = None
    with patched_clock(rig.clock):
        while t < horizon and gap > 0.5 and v_ego > 0.05:
            rig.stub.data.publish(StubLead(gap, v_lead, -lead_decel, rig.score))
            rig.clock.advance()
            cap = ctrl.accel_cap_ms2(v_ego)
            a_ego = min(0.0, cap)
            for level in (0.5, 1.0, 2.0, 4.0):
                if level not in marks and -a_ego >= level:
                    marks[level] = t
            if slam is None and cap <= rig.cap_lo + 0.05:
                slam = t
            peak = min(peak, a_ego)
            v_ego = max(0.0, v_ego + a_ego * rig.dt)
            v_lead = max(0.0, v_lead - lead_decel * rig.dt)
            gap += (v_lead - v_ego) * rig.dt
            min_gap = min(min_gap, gap)
            t += rig.dt
    return {
        "ego_kmh": ego_kmh, "gap0_m": gap0, "lead_decel_ms2": lead_decel,
        "onset_s": {str(k): marks.get(k) for k in (0.5, 1.0, 2.0, 4.0)},
        "peak_ms2": peak, "min_gap_m": min_gap, "decel_clamp_s": slam,
    }
