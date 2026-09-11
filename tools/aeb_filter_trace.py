"""Per-vehicle radar filter-chain trace for the review charts. Dev only, never shipped.

Every number here is read off the replayed ``Vehicle`` objects or rebuilt from
them with the production helpers, so the chart shows what the filter actually
did rather than a second implementation of it. See core/radar/README.md section 7
for the chain itself and tools/README.md for what the lanes mean.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from core.aeb.clip_replay import clip_t0, decode_radar_stream
from core.aeb.clip_schema import Clip
from core.radar import traffic as T
from core.radar.traffic import Vehicle

NAN = float("nan")

# Vehicles closer than this at any point in the clip are traced. Everything else
# is out of AEB and ACC range for the whole clip and only crowds the picker.
TRACE_RANGE_M: float = 220.0


@dataclass
class VehicleTrace:
    """One vehicle's filter signals over the clip, one sample per radar frame."""

    vid: int
    is_tmp: bool
    is_trailer: bool
    min_gap_m: float
    was_threat: bool
    t: list[float] = field(default_factory=list)
    series: dict[str, list[float]] = field(default_factory=dict)

    def label(self) -> str:
        kind = "TMP" if self.is_tmp else "AI"
        if self.is_trailer:
            kind += " trailer"
        threat = "  threat" if self.was_threat else ""
        return f"#{self.vid}  {kind}  min {self.min_gap_m:.0f} m{threat}"

    def at(self, t_rel: float) -> dict[str, float]:
        """Every signal at the sample nearest ``t_rel``; empty when the trace is."""
        if not self.t:
            return {}
        i = min(range(len(self.t)), key=lambda k: abs(self.t[k] - t_rel))
        return {name: col[i] for name, col in self.series.items()}


@dataclass
class ClipTrace:
    """Ego context plus one VehicleTrace per traced vehicle, on the clip clock."""

    duration: float
    ego_t: list[float] = field(default_factory=list)
    ego_speed: list[float] = field(default_factory=list)
    vehicles: dict[int, VehicleTrace] = field(default_factory=dict)
    order: list[int] = field(default_factory=list)

    def pick_default(self, target_vid: int | None) -> int | None:
        """Labelled target if it was traced, else the most relevant vehicle."""
        if target_vid is not None and target_vid in self.vehicles:
            return target_vid
        return self.order[0] if self.order else None


# Signal names in draw order per lane. The chart reads these, so a new signal is
# added once, here, and appears without touching the widget.
SPEED_SIGNALS = ("raw_sel", "raw_long", "speed_ema", "speed", "acc_corr", "acc_speed")
ACCEL_SIGNALS = ("accel", "acc_accel", "accel_trend", "accel_long", "brake_floor")
GATE_SIGNALS = ("accel_win", "ramp", "consistency", "ff_gate", "accel_factor",
                "speed_factor", "tau")
LAG_SIGNALS = ("lag_disp_ratio", "lag_rot_rate", "lag_raw_recent", "lag_raw_decay",
               "lag_freeze_dur", "lag_elapsed", "gap_ttc")
STATE_SIGNALS = ("st_frozen", "st_lag_confirmed", "st_raw_brake", "st_pos_mismatch",
                 "st_crash", "st_standstill", "st_stale", "st_subframe", "st_bypassed")

# Step 4 internals only exist on frames where the chain ran. On a sub-frame the
# filter genuinely still holds the last set, so the trace carries them forward.
_HELD_SIGNALS = ("acc_corr", "accel_trend", "accel_long", "ramp", "consistency",
                 "ff_gate", "accel_factor", "speed_factor", "tau")
_ALL_SIGNALS = SPEED_SIGNALS + ACCEL_SIGNALS + GATE_SIGNALS + LAG_SIGNALS + STATE_SIGNALS

# Residual between the rebuilt step 4 and the recorded acc_speed. Non-zero means
# the rebuild below has drifted from traffic.py and the gate lane is lying.
_RESIDUAL = "acc_speed_residual"


def _fwd(v: Vehicle) -> tuple[float, float]:
    yaw = v._smooth_yaw if v._smooth_yaw is not None else math.radians(v.rotation.euler()[1])
    return -math.sin(yaw), -math.cos(yaw)


def _clamp01(x: float) -> float:
    return max(0.0, min(1.0, x))


def _rot_rate(prev: Vehicle, cur: Vehicle, dt: float) -> float:
    """Largest per-axis rotation rate in deg/s, the lag entry gate 2 quantity."""
    if dt <= 1e-9:
        return NAN
    return max(
        abs((c - p + 180.0) % 360.0 - 180.0) for c, p in
        zip(cur.rotation.euler(), prev.rotation.euler())
    ) / dt


def _chain_ran(prev: Vehicle | None, cur: Vehicle, dt: float) -> bool:
    """Did the filter chain execute on this frame, or was it a copy or an early return?

    Sub-frames, clock re-anchors, lag freezes and position-mismatch holds all carry
    state forward without running steps 1 to 4, and the first full update seeds
    instead of filtering. The witness is the ACC history: only the real path appends
    to it, so a last sample stamped at this frame's own clock means the chain ran.
    """
    if prev is None or dt <= 1e-9 or prev._acc_speed_ema is None:
        return False
    history = cur._acc_speed_ema_history
    return bool(history) and abs(history[-1][0] - cur.time) < 1e-9


def _acc_chain_inputs(prev: Vehicle, cur: Vehicle, dt: float,
                      raw_long: float) -> float:
    """ACC step 3 output, rebuilt from the recorded step 1 state and the raw input."""
    acc_raw = (cur._raw_speed or 0.0) if cur.crash_confirmed else raw_long
    alpha_s = T._tmp_speed_ema_alpha(abs((prev._acc_speed_ema + acc_raw) * 0.5))
    tau_eff = dt * (1.0 - alpha_s) / alpha_s if alpha_s > 1e-6 else 0.0
    corr = max(-T._SPEED_CORR_CLAMP_MS, min(T._SPEED_CORR_CLAMP_MS, cur.acc_accel * tau_eff))
    return cur._acc_speed_ema + corr


def _step4_internals(prev: Vehicle, cur: Vehicle, dt: float,
                     acc_corr: float) -> dict[str, float]:
    """Rebuild the step 4 gates. Mirrors ``_acc_speed_step``; the residual proves it."""
    history = cur._acc_speed_ema_history
    accel_trend = T._accel_from_speed_history(history, T._ACC_SPEED_ACCEL_WINDOW_S)
    accel_long = T._accel_from_speed_history(history, T._ACC_SPEED_CONSIST_WINDOW_S)
    delta = acc_corr - prev.acc_speed

    ramp = _clamp01((abs(delta) - T._ACC_SPEED_DEADBAND_MS) / T._ACC_SPEED_DEADBAND_MS)
    if abs(accel_trend) <= T._ACC_SPEED_FF_GATE_LO_MS2 or accel_trend * delta <= 0.0:
        ramp = 0.0
    speed_factor = T._ACC_SPEED_SMOOTH_MIN + (1.0 - T._ACC_SPEED_SMOOTH_MIN) * min(
        1.0, abs(acc_corr) / T._ACC_SPEED_SMOOTH_REF_MS)

    if accel_trend * accel_long <= 0.0:
        consistency = 0.0
    else:
        ratio = min(1.0, abs(accel_long) / max(abs(accel_trend), 1e-6))
        mag_span = T._ACC_SPEED_CONSIST_MAG_HI_MS2 - T._ACC_SPEED_CONSIST_MAG_LO_MS2
        mag = _clamp01((abs(accel_long) - T._ACC_SPEED_CONSIST_MAG_LO_MS2) / mag_span)
        consistency = max(ratio, mag)

    accel_span = T._ACC_SPEED_ACCEL_HI_MS2 - T._ACC_SPEED_ACCEL_LO_MS2
    accel_ramp = _clamp01(
        (abs(accel_trend) - T._ACC_SPEED_ACCEL_LO_MS2) / accel_span) * consistency
    accel_factor = 1.0 - (1.0 - T._ACC_SPEED_ACCEL_FLOOR) * accel_ramp
    tau = T._ACC_SPEED_TAU_SLOW_S + (T._ACC_SPEED_TAU_FAST_S - T._ACC_SPEED_TAU_SLOW_S) * ramp
    tau *= speed_factor * accel_factor

    ff_gate = _clamp01(
        (abs(accel_trend) - T._ACC_SPEED_FF_GATE_LO_MS2)
        / (T._ACC_SPEED_FF_GATE_HI_MS2 - T._ACC_SPEED_FF_GATE_LO_MS2)) * consistency
    accel_ff = max(-T._ACC_SPEED_FF_ACCEL_CLAMP_MS2,
                   min(T._ACC_SPEED_FF_ACCEL_CLAMP_MS2, cur.acc_accel))
    predicted = prev.acc_speed + accel_ff * dt * ff_gate
    alpha_a = dt / (tau + dt) if (tau + dt) > 1e-9 else 1.0
    rebuilt = predicted + alpha_a * (acc_corr - predicted)

    return {
        "accel_trend": accel_trend, "accel_long": accel_long,
        "ramp": ramp, "consistency": consistency, "ff_gate": ff_gate,
        "accel_factor": accel_factor, "speed_factor": speed_factor, "tau": tau,
        _RESIDUAL: NAN if cur._acc_standstill else rebuilt - cur.acc_speed,
    }


def _lag_gates(prev: Vehicle, cur: Vehicle, dt: float,
               gap_3d: float, ego_speed: float) -> dict[str, float]:
    """The four lag entry gates and the freeze clock, each against its own threshold."""
    freeze_dur = T._lag_freeze_duration(gap_3d, ego_speed)
    elapsed = NAN if cur._lag_since is None else cur.time - cur._lag_since
    out = {
        "lag_rot_rate": _rot_rate(prev, cur, dt),
        "lag_freeze_dur": freeze_dur,
        "lag_elapsed": elapsed,
        "st_frozen": float(elapsed == elapsed and elapsed < freeze_dur),
        "lag_disp_ratio": NAN, "lag_raw_recent": NAN, "lag_raw_decay": NAN,
    }
    if cur._raw_x is not None and prev._raw_x is not None and dt > 1e-9:
        raw_disp = math.hypot(cur._raw_x - prev._raw_x, cur._raw_z - prev._raw_z)
        expected = abs(prev.speed) * dt
        if expected > 1e-6:
            out["lag_disp_ratio"] = raw_disp / expected

    window = T._LAG_ENTRY_WINDOW
    recent = T._raw_path_speed(cur._position_history, -(window + 1), None)
    if recent is not None:
        out["lag_raw_recent"] = recent
        older = T._raw_path_speed(cur._position_history, -(2 * window + 1), -window)
        if older is not None and older > 1e-9:
            out["lag_raw_decay"] = recent / older
    return out


def _sample(prev: Vehicle | None, cur: Vehicle, ego,
            stale: bool) -> dict[str, float]:
    """Every traced signal for one vehicle on one radar frame."""
    fwd_x, fwd_z = _fwd(cur)
    raw_long = T._raw_speed_from_position_history(cur._position_history, fwd_x, fwd_z)
    if raw_long is None:
        raw_long = NAN
    gap_3d = math.dist(
        (cur.position.x, cur.position.y, cur.position.z),
        (ego.coordinateX, ego.coordinateY, ego.coordinateZ),
    )
    brake_floor = NAN
    if cur._raw_brake_active:
        decel = T._hard_brake_decel_from_position_history(
            cur._position_history, fwd_x, fwd_z)
        if decel is not None:
            brake_floor = -min(decel, T._ACC_SPEED_FF_ACCEL_CLAMP_MS2)

    row = {
        "raw_sel": cur._raw_speed if cur._raw_speed is not None else NAN,
        "raw_long": raw_long,
        "speed_ema": cur._speed_ema if cur._speed_ema is not None else NAN,
        "speed": cur.speed,
        "acc_corr": NAN, "acc_speed": cur.acc_speed,
        "accel": cur.acceleration, "acc_accel": cur.acc_accel,
        "accel_win": (T._ACCEL_FIT_WINDOW_S
                      * T._accel_window_scale(cur._speed_ema or 0.0)),
        "brake_floor": brake_floor,
        "accel_trend": NAN, "accel_long": NAN,
        "ramp": NAN, "consistency": NAN, "ff_gate": NAN,
        "accel_factor": NAN, "speed_factor": NAN, "tau": NAN, _RESIDUAL: NAN,
        "lag_disp_ratio": NAN, "lag_rot_rate": NAN, "lag_raw_recent": NAN,
        "lag_raw_decay": NAN, "lag_freeze_dur": NAN, "lag_elapsed": NAN,
        "gap_ttc": gap_3d / max(abs(ego.speed), T._LAG_FREEZE_EGO_SPEED_FLOOR),
        "st_frozen": 0.0,
        "st_lag_confirmed": float(cur.lag_confirmed),
        "st_raw_brake": float(cur._raw_brake_active),
        "st_pos_mismatch": float(cur._pos_mismatch_frames > 0),
        "st_crash": float(cur.crash_confirmed),
        "st_standstill": float(cur._acc_standstill),
        "st_stale": float(stale),
        "st_subframe": 0.0,
        "st_bypassed": 0.0,
    }
    dt = (cur.time - prev.time) if prev is not None else 0.0
    if prev is None or dt <= 1e-9:
        # Sub-frames copy the chain forward untouched, so this is not an anomaly.
        row["st_subframe"] = float(prev is not None)
        return row
    row.update(_lag_gates(prev, cur, dt, gap_3d, ego.speed))
    if not _chain_ran(prev, cur, dt):
        row["st_bypassed"] = 1.0
        return row
    acc_corr = _acc_chain_inputs(prev, cur, dt, raw_long)
    row["acc_corr"] = acc_corr
    row.update(_step4_internals(prev, cur, dt, acc_corr))
    return row


def _threat_ids(clip: Clip) -> set[int]:
    """Ids the live AEB ever tracked or suppressed: the ones a filter bug shows up on."""
    ids: set[int] = set()
    for tick in clip.aeb_ticks:
        live = tick.live_aeb
        ids.update(int(i) for i in live.colliding_ids)
        ids.update(int(i) for i in live.suppressed_ids)
    return ids


def build_trace(clip: Clip, stream=None) -> ClipTrace:
    """Trace every in-range vehicle. ``stream`` reuses an existing decode_radar_stream."""
    veh_by_t, ego_by_t, frame_t, _off = stream if stream is not None else decode_radar_stream(clip)
    t0 = clip_t0(clip)
    trace = ClipTrace(duration=max((frame_t[-1] - t0) if frame_t else 1.0, 1e-3))
    threats = _threat_ids(clip)
    prev_veh: dict[int, Vehicle] = {}
    columns: dict[int, dict[str, list[float]]] = {}

    for ft in frame_t:
        ego = ego_by_t.get(ft)
        if ego is None:
            continue
        t_rel = ft - t0
        trace.ego_t.append(t_rel)
        trace.ego_speed.append(ego.speed)
        for v in veh_by_t.get(ft, []):
            vid = int(v.id)
            prev = prev_veh.get(vid)
            stale = prev is v
            gap = math.dist(
                (v.position.x, v.position.z), (ego.coordinateX, ego.coordinateZ))
            vt = trace.vehicles.get(vid)
            if vt is None:
                if gap > TRACE_RANGE_M and vid not in threats:
                    continue
                vt = VehicleTrace(vid=vid, is_tmp=bool(v.is_tmp),
                                  is_trailer=bool(getattr(v, "is_trailer", False)),
                                  min_gap_m=gap, was_threat=vid in threats)
                vt.series = {name: [] for name in _ALL_SIGNALS + (_RESIDUAL,)}
                trace.vehicles[vid] = vt
                columns[vid] = vt.series
            vt.min_gap_m = min(vt.min_gap_m, gap)
            vt.t.append(t_rel)
            row = _sample(prev, v, ego, stale)
            cols = columns[vid]
            if row["st_subframe"]:
                for name in _HELD_SIGNALS:
                    row[name] = cols[name][-1] if cols[name] else NAN
            for name, col in cols.items():
                col.append(row.get(name, NAN))
            prev_veh[vid] = v

    trace.order = sorted(
        trace.vehicles,
        key=lambda vid: (not trace.vehicles[vid].was_threat, trace.vehicles[vid].min_gap_m),
    )
    return trace
