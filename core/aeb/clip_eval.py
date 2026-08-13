"""Stateful AEBThread replay over clips: parity vs live_aeb and candidate cal evaluation."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from core.aeb.calibration import AEBCalibration, DEFAULT as _CAL_DEFAULT
from core.aeb.clip_replay import decode_radar_stream, nearest_frame_t
from core.aeb.clip_schema import Clip
from core.aeb.filters import VehicleCurvatureBlender, build_pipeline
from core.aeb.thread import (
    AEBSnapshot, AEBState, AEBThread, _INF,
    _addressing_brake_from_sources, _user_braking_from_sources,
)


@dataclass
class EvalTick:
    """One re-run tick's decision (candidate or parity)."""

    t_rel: float
    t_mono: float
    aeb_warn: bool
    aeb_brake: bool
    engaged: bool
    target_decel_ms2: float
    required_decel_ms2: float
    time_to_brake: float
    time_to_collision: float
    colliding_ids: set = field(default_factory=set)
    suppressed_ids: set = field(default_factory=set)
    braking_worsens_ids: set = field(default_factory=set)


class _NoSound:
    def start_warning(self) -> None: ...
    def stop_warning(self) -> None: ...
    def cleanup(self) -> None: ...


def _pitch_deg(rotation_y: float) -> float:
    pv = (rotation_y + 0.5) % 1.0 - 0.5
    return math.degrees(-pv * 2.0 * math.pi)


def _make_headless(cal: AEBCalibration) -> AEBThread:
    t = AEBThread()
    t._cal = cal
    t._pipeline = build_pipeline(cal)
    t._curvature_blender = VehicleCurvatureBlender(cal)
    t.running = True
    t._radar_visualizer = None
    t._sound_handler = _NoSound()
    t._aeb_active_fn = lambda: True
    t._capture_aeb_tick = lambda *a, **k: None   # never record during replay
    return t


def _apply_warm_state(t: AEBThread, ws) -> None:
    t._engaged = bool(ws.engaged)
    t._latched_threat_ids = {int(i) for i in ws.latched_threat_ids}
    t._latched_scope_ok_mono = {
        int(k): float(v)
        for k, v in getattr(ws, "latched_scope_ok_mono", {}).items()
    }
    t._latched_filter_ego_kmh = ws.latched_filter_ego_kmh
    t._published_target_ms2 = float(ws.target_decel_ms2)
    if ws.brake_hold_until_mono is not None:
        t._prev_state = AEBState.BRAKE
        t._state_hold_until = float(ws.brake_hold_until_mono)
    elif ws.warn_hold_until_mono is not None:
        t._prev_state = AEBState.WARN
        t._state_hold_until = float(ws.warn_hold_until_mono)


def _snapshot_tuple(ego, vehicles, radar_t_mono: float):
    """Build the tuple _read_radar_snapshot returns, from clip ego + vehicles."""
    return (
        vehicles,
        ego.coordinateX, ego.coordinateY, ego.coordinateZ,
        ego.rotationX * 2.0 * math.pi,
        ego.speed,
        _pitch_deg(ego.rotationY),
        ego.userSteer,
        bool(ego.ego_has_trailer),
        None,                       # ego_curvature: AEB uses the yaw-rate proxy
        any(v.is_tmp for v in vehicles),
        bool(ego.paused),
        radar_t_mono,
    )


def run_headless(clip: Clip, cal: AEBCalibration = _CAL_DEFAULT,
                 warm: bool = True) -> list[EvalTick]:
    """Re-run the AEB pipeline over the clip under ``cal``; one EvalTick per tick."""
    veh_by_t, ego_by_t, frame_t = decode_radar_stream(clip)
    t = _make_headless(cal)
    if warm:
        _apply_warm_state(t, clip.metadata.aeb_warm_state)

    ticks = sorted(clip.aeb_ticks, key=lambda x: x.t_mono)
    all_t = [f.t_mono for f in clip.radar_frames] + [tk.t_mono for tk in ticks]
    t0 = min(all_t) if all_t else 0.0

    out: list[EvalTick] = []
    for tk in ticks:
        ft = nearest_frame_t(frame_t, tk.radar_t_mono)
        if ft is None:
            continue
        vehicles = veh_by_t.get(ft, [])
        ego = ego_by_t.get(ft)
        if ego is None:
            continue

        snap = _snapshot_tuple(ego, vehicles, ft)
        t._read_radar_snapshot = lambda s=snap: s
        t._read_max_brake_ms2 = lambda mb=tk.consumed.max_brake_ms2: mb
        t._read_user_braking = (
            lambda c=tk.consumed: _user_braking_from_sources(
                c.brakeval, c.opdbrakeval, c.program_brake,
            )
        )
        t._read_addressing_brake = (
            lambda c=tk.consumed: _addressing_brake_from_sources(
                c.brakeval, c.program_brake,
            )
        )
        t._now = lambda tm=tk.t_mono: tm
        t._aeb_active_fn = lambda en=tk.consumed.aeb_enabled: bool(en)

        t.loop()

        d = t.data
        snap_out = d.snapshot
        out.append(EvalTick(
            t_rel=tk.t_mono - t0,
            t_mono=tk.t_mono,
            aeb_warn=bool(d.AEB_warn),
            aeb_brake=bool(d.AEB_brake),
            engaged=bool(t._engaged),
            target_decel_ms2=float(d.AEB_target_decel_ms2),
            required_decel_ms2=float(d.AEB_required_decel_ms2),
            time_to_brake=float(d.time_to_brake),
            time_to_collision=float(snap_out.time_to_collision),
            colliding_ids=set(snap_out.colliding_ids),
            suppressed_ids=set(snap_out.suppressed_ids),
            braking_worsens_ids=set(snap_out.braking_worsens_ids),
        ))
    return out


@dataclass
class ParityMismatch:
    t_rel: float
    field: str
    recorded: object
    replayed: object


@dataclass
class ParityReport:
    total_ticks: int
    compared: int                       # ticks after burn-in
    warn_agree: int
    brake_agree: int
    engaged_agree: int
    colliding_agree: int
    mismatches: list[ParityMismatch] = field(default_factory=list)

    @property
    def brake_agreement(self) -> float:
        return self.brake_agree / self.compared if self.compared else 1.0

    @property
    def warn_agreement(self) -> float:
        return self.warn_agree / self.compared if self.compared else 1.0

    @property
    def full_agree(self) -> int:
        return min(self.warn_agree, self.brake_agree, self.engaged_agree,
                   self.colliding_agree)


def parity_report(clip: Clip, burn_in_s: float | None = None,
                  max_mismatches: int = 20) -> ParityReport:
    """Replay under the capture calibration; compare to recorded live_aeb (plan 8)."""
    burn = burn_in_s if burn_in_s is not None else clip.metadata.burn_in_s
    evals = run_headless(clip, cal=_CAL_DEFAULT, warm=True)
    recorded = sorted(clip.aeb_ticks, key=lambda x: x.t_mono)

    rep = ParityReport(total_ticks=len(recorded), compared=0,
                       warn_agree=0, brake_agree=0, engaged_agree=0, colliding_agree=0)
    for rec, ev in zip(recorded, evals):
        if ev.t_rel < burn:
            continue
        rep.compared += 1
        la = rec.live_aeb
        rec_coll = {int(i) for i in la.colliding_ids}
        checks = [
            ("aeb_warn", bool(la.aeb_warn), ev.aeb_warn),
            ("aeb_brake", bool(la.aeb_brake), ev.aeb_brake),
            ("engaged", bool(la.engaged), ev.engaged),
            ("colliding_ids", rec_coll, ev.colliding_ids),
        ]
        for name, r, e in checks:
            agree = r == e
            if name == "aeb_warn" and agree:
                rep.warn_agree += 1
            elif name == "aeb_brake" and agree:
                rep.brake_agree += 1
            elif name == "engaged" and agree:
                rep.engaged_agree += 1
            elif name == "colliding_ids" and agree:
                rep.colliding_agree += 1
            if not agree and len(rep.mismatches) < max_mismatches:
                rep.mismatches.append(ParityMismatch(round(ev.t_rel, 3), name, r, e))
    return rep


@dataclass
class CandidateDiff:
    """Per-tick difference between two decision streams (baseline vs candidate)."""

    t_rel: float
    baseline_brake: bool
    candidate_brake: bool
    baseline_warn: bool
    candidate_warn: bool


@dataclass
class Outcome:
    """A calibration's outcome on one labelled clip, judged against its window.

    Verdicts: ``true_positive``, ``false_negative``, ``late``, ``false_positive``
    (must-not-trigger + brake), ``false_warn`` (must-not-trigger + warn only),
    ``true_negative`` (must-not-trigger + silent), ``unlabeled``.
    """

    label_class: str
    should_trigger: tuple | None      # (from_t, to_t) or None = must-not-trigger
    engaged: bool
    brake_window: tuple | None        # (from_t, to_t) t_rel where the candidate braked
    peak_decel_ms2: float
    verdict: str
    warn_window: tuple | None = None  # (from_t, to_t) t_rel where the candidate warned


def outcome_under(clip: Clip, cal: AEBCalibration = _CAL_DEFAULT,
                  burn_in_s: float = 0.0) -> Outcome:
    """Classify candidate run vs label after ``burn_in_s``; verdicts on ``Outcome``."""
    lbl = clip.metadata.label
    evs = [e for e in run_headless(clip, cal=cal) if e.t_rel >= burn_in_s]
    braked = [e for e in evs if e.aeb_brake]
    warned = [e for e in evs if e.aeb_warn]
    bw = (braked[0].t_rel, braked[-1].t_rel) if braked else None
    ww = (warned[0].t_rel, warned[-1].t_rel) if warned else None
    peak = max((e.target_decel_ms2 for e in evs), default=0.0)
    engaged = bool(braked)

    st = None
    if lbl is not None and lbl.should_trigger:
        st = (float(lbl.should_trigger["from_t"]), float(lbl.should_trigger["to_t"]))

    if lbl is None:
        verdict = "unlabeled"
    elif st is None:
        if engaged:
            verdict = "false_positive"
        elif ww is not None:
            verdict = "false_warn"
        else:
            verdict = "true_negative"
    elif not engaged:
        verdict = "false_negative"
    elif bw[0] > st[1]:
        verdict = "late"
    else:
        verdict = "true_positive"

    return Outcome(
        label_class=(lbl.class_ if lbl else "unlabeled"),
        should_trigger=st, engaged=engaged, brake_window=bw,
        peak_decel_ms2=peak, verdict=verdict, warn_window=ww,
    )


def diff_calibrations(clip: Clip, candidate: AEBCalibration,
                      baseline: AEBCalibration = _CAL_DEFAULT) -> list[CandidateDiff]:
    """Ticks where baseline and candidate calibrations decide differently."""
    a = run_headless(clip, cal=baseline)
    b = run_headless(clip, cal=candidate)
    out: list[CandidateDiff] = []
    for x, y in zip(a, b):
        if x.aeb_brake != y.aeb_brake or x.aeb_warn != y.aeb_warn:
            out.append(CandidateDiff(
                round(x.t_rel, 3), x.aeb_brake, y.aeb_brake, x.aeb_warn, y.aeb_warn,
            ))
    return out
