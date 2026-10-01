"""One clip as text: metadata, label, scene features, tick timeline, ASCII scenes.

This is what an agent reads instead of watching the clip in the review UI. It is
deliberately verbose about the *measurement* and quiet about AEB's opinion, so a
reader can disagree with the recorded decision on evidence rather than deference.
"""

from __future__ import annotations

import bisect
import math
from pathlib import Path

from core.aeb.clip_replay import replay_clip
from core.aeb.clip_score import class_window_warning

from tools.aeb_agent.corpus import ClipRow
from tools.aeb_agent.features import ClipFeatures, extract, world_to_ego
from tools.aeb_agent.scene import glyph_map, render

_INF_AT = 1e8
_TIMELINE_MAX_ROWS = 34


def _f(value: float, digits: int = 2) -> str:
    if value is None:
        return "-"
    if value >= _INF_AT:
        return "inf"
    return f"{value:.{digits}f}"


def _header(row: ClipRow, feats: ClipFeatures) -> list[str]:
    mass = ("?" if feats.ego_mass_kg is None
            else f"{feats.ego_mass_kg / 1000.0:.1f} t")
    return [
        f"CLIP {row.clip_id}  ({row.origin} store)",
        f"  file        {Path(row.path).name}",
        f"  captured    {row.captured_at}  client {row.client_version}",
        f"  session     {row.session_kind}   trigger {row.trigger_source}"
        f"   brake_reached={row.brake_reached}",
        f"  duration    {feats.duration_s:.2f} s over {feats.tick_count} AEB ticks"
        f"   ({feats.n_vehicles} vehicles tracked)",
        f"  ego         {feats.ego_speed_min_kmh:.0f}-"
        f"{feats.ego_speed_max_kmh:.0f} km/h, {mass}, "
        f"trailer={feats.ego_has_trailer}, "
        f"max|steer| {feats.ego_max_abs_steer:.3f} "
        f"(kappa {feats.ego_max_kappa:.4f})",
        f"  driver      brake max {feats.user_brake_max:.2f} "
        f"({feats.user_brake_frac:.0%} of ticks), gas max {feats.user_gas_max:.2f}, "
        f"program brake max {feats.program_brake_max:.2f}",
    ]


def _label_block(row: ClipRow) -> list[str]:
    if not row.labelled:
        return ["LABEL       (none) - this clip is unreviewed"]
    warn = class_window_warning(row.label_class, row.window is not None)
    out = [
        f"LABEL       class={row.label_class}  severity={row.severity}  "
        f"target_vid={row.target_vid}",
        "  window      " + ("must NOT trigger" if not row.window else
                            f"{row.window[0]:.2f} .. {row.window[1]:.2f} s "
                            f"({row.window_span:.2f} s)"),
    ]
    if row.desired_peak_decel_ms2:
        out.append(f"  desired     peak decel {row.desired_peak_decel_ms2:.2f} m/s2")
    if row.notes:
        out.append(f"  notes       {row.notes}")
    if warn:
        out.append(f"  INCONSISTENT {warn}")
    return out


def _live_block(feats: ClipFeatures) -> list[str]:
    band = ("none" if not feats.recorded_band else
            f"{feats.recorded_band[0]:.2f} .. {feats.recorded_band[1]:.2f} s")
    return [
        "RECORDED AEB (what the shipped build did at capture time)",
        f"  warn {feats.warn_ticks} ticks, brake {feats.brake_ticks} ticks, "
        f"engaged {feats.engaged_ticks} ticks",
        f"  first warn {_f(feats.first_warn_t) if feats.first_warn_t is not None else '-'}"
        f"  first brake "
        f"{_f(feats.first_brake_t) if feats.first_brake_t is not None else '-'}",
        f"  peak target {feats.peak_target_ms2:.2f} m/s2, peak required "
        f"{_f(feats.peak_required_ms2)} m/s2, capacity {feats.max_brake_ms2:.2f}",
        f"  min ttc {_f(feats.min_ttc_s)} s, min ttb {_f(feats.min_ttb_s)} s",
        f"  reaction band {band}   (proposal only, never ground truth)",
    ]


def _driver_block(feats: ClipFeatures) -> list[str]:
    """What the driver did and what would have happened had they not.

    This block outranks the recorded geometry. A swerve leaves a near miss on the
    record, and reading that as "no threat" is the single most common way to
    mislabel a clip.
    """
    inter = feats.intervention or {}
    cf = feats.counterfactual or {}
    out = ["DRIVER AND COUNTERFACTUAL"]
    if not inter.get("found"):
        out.append("  intervention  none detected near the action")
        out.append("  counterfactual not needed: the recorded path already is "
                   "the no-action path")
        return out
    bits = [f"{inter.get('kind')} at t={inter.get('t', 0):.2f}s"]
    if "swerve" in (inter.get("kind") or ""):
        bits.append(f"steer {inter.get('steer_before', 0):+.3f} -> "
                    f"{inter.get('steer_peak', 0):+.3f} "
                    f"(peak rate {inter.get('steer_rate_peak', 0):.2f}/s)")
    if "brake" in (inter.get("kind") or ""):
        bits.append(f"brake to {inter.get('brake_peak', 0):.2f}")
    if "lift" in (inter.get("kind") or ""):
        bits.append(f"gas was {inter.get('gas_before', 0):.2f}")
    if inter.get("blinker"):
        bits.append(f"blinker {inter['blinker']}")
    out.append("  intervention  " + ", ".join(bits))
    if not cf.get("ran"):
        out.append(f"  counterfactual not run: {cf.get('reason', '?')}")
        return out
    mode = cf.get("mode", "")
    held = {
        "heading": (f"held {inter.get('yaw_rate_before', 0):+.3f} rad/s and "
                    f"kept the recorded speed profile"),
        "speed": (f"kept the recorded heading and held "
                  f"{cf.get('ghost_speed_kmh', 0):.0f} km/h"),
        "both": (f"held {inter.get('yaw_rate_before', 0):+.3f} rad/s and "
                 f"{cf.get('ghost_speed_kmh', 0):.0f} km/h"),
    }.get(mode, "?")
    out.append(f"  ghost         {held}, for {cf.get('horizon_s', 0):.2f} s "
               f"(replaced: {mode})")
    out.append(f"  verdict       {cf.get('verdict', '?').upper()}: nearest "
               f"approach {cf.get('min_separation_m', 0):+.1f} m to vid "
               f"{cf.get('target_vid')} at {cf.get('dt_at_min_s', 0):.2f} s past "
               f"the fork")
    out.append(f"  error bar     ghost drifts +/-{cf.get('drift_at_min_m', 0):.1f} m "
               f"by then (measured p90)"
               + ("" if cf.get("credible")
                  else "; past 1.5 s this is not trustworthy"))
    if cf.get("verdict") in ("collides", "likely"):
        out.append("  READ THIS     the driver avoided contact. The clean pass on "
                   "the record is the rescue, not the scene.")
    return out


def _target_block(feats: ClipFeatures, limit: int = 6) -> list[str]:
    out = ["TARGETS (ego frame: fwd +ahead, lat +right)"]
    if not feats.targets:
        out.append("  none decoded")
        return out
    out.append("   vid   role         ticks  minRange  corrGap   tgeom  "
               "lat@min  fwd@min  dot    speed   flags")
    for t in feats.targets[:limit]:
        mark = "*" if t["vid"] == feats.primary_vid else " "
        role = _role(t)
        flags = []
        if t["colliding_ticks"]:
            flags.append(f"coll x{t['colliding_ticks']}")
        if t["suppressed_ticks"]:
            flags.append(f"supp x{t['suppressed_ticks']}")
        if t["lag_ticks"] or t["teleport_ticks"]:
            flags.append(f"lag {t['lag_ticks']}/tp {t['teleport_ticks']}")
        if t["stall_run_max"] >= 10:
            flags.append(f"stall run {t['stall_run_max']}")
        if t["is_tmp"]:
            flags.append("tmp")
        if t["has_trailer"] or t["is_trailer"]:
            flags.append("trailer")
        out.append(
            f"  {mark}{t['vid']:<6}{role:<12} {t['ticks']:>5}  "
            f"{_f(t['min_range_m'], 1):>8}  {_f(t['min_corridor_gap_m'], 1):>7}  "
            f"{_f(t['min_geom_ttc_s'], 2):>6}  {t['lat_at_min_m']:+7.1f}  "
            f"{t['fwd_at_min_m']:+7.1f}  {t['heading_dot_at_min']:+.2f}  "
            f"{t['speed_at_min_kmh']:5.1f}   {', '.join(flags)}")
    for t in feats.targets[:limit]:
        if t["suppression_stages"]:
            stages = ", ".join(f"{k} x{v}" for k, v in
                               sorted(t["suppression_stages"].items(),
                                      key=lambda kv: -kv[1]))
            out.append(f"    vid {t['vid']} suppressed by: {stages}")
    return out


def _role(t: dict) -> str:
    if t["speed_max_kmh"] < 2.0:
        return "stationary"
    dot = t["heading_dot_at_min"]
    if dot >= 0.6:
        return "codir" if t["ahead_frac"] >= 0.5 else "overtaker"
    if dot <= -0.6:
        return "oncoming"
    return "crossing"


def _timeline(frames, feats: ClipFeatures, vid: int | None,
              steer_at=None) -> list[str]:
    """Per-tick table around the action, thinned to a readable row count."""
    if not frames:
        return ["TIMELINE    (no ticks)"]
    lo = max(0.0, feats.action_t - 4.0)
    hi = feats.action_t + 3.0
    window = [f for f in frames if lo <= f.t_rel <= hi] or frames
    step = max(1, len(window) // _TIMELINE_MAX_ROWS)
    out = [
        f"TIMELINE    ticks {lo:.1f}-{hi:.1f} s, every {step} tick(s); "
        f"tracking vid {vid}",
        "     t     ego   state  ttc    ttb    req    raw   targ  "
        "fwd    lat   vspd  steer  gas  brk  colliding",
    ]
    for f in window[::step]:
        snap = f.snapshot
        la = f.live_aeb
        fwd = lat = vspd = None
        for veh in snap.vehicles:
            if int(veh["vid"]) == vid:
                fwd, lat = world_to_ego(veh["x"], veh["z"], snap.ego_x,
                                        snap.ego_z, snap.ego_yaw)
                vspd = veh["speed_kmh"]
                break
        coll = ",".join(str(i) for i in sorted(snap.colliding_ids)[:4]) or "-"
        state = snap.aeb_state.name[:5]
        out.append(
            f"  {f.t_rel:6.2f} {snap.ego_speed * 3.6:6.1f}  {state:<6}"
            f"{_f(la.time_to_collision):>5} {_f(la.time_to_brake):>6} "
            f"{_f(min(la.required_decel_ms2, 99.0)):>6} "
            f"{f.raw_target_ms2:6.2f} {la.target_decel_ms2:6.2f} "
            f"{('  -  ' if fwd is None else f'{fwd:+6.1f}')} "
            f"{('  -  ' if lat is None else f'{lat:+6.1f}')} "
            f"{('  -  ' if vspd is None else f'{vspd:5.1f}')} "
            f"{(steer_at(f.t_mono) if steer_at else 0.0):+6.3f} "
            f"{f.consumed.gasval:4.2f} "
            f"{f.consumed.brakeval:4.2f}  {coll}")
    return out


def _steer_lookup(clip):
    """t_mono -> userSteer for the nearest radar frame. ReviewFrame does not carry it."""
    frames = sorted(clip.radar_frames, key=lambda f: f.t_mono)
    times = [f.t_mono for f in frames]
    steers = [f.ego.userSteer for f in frames]

    def at(t_mono: float) -> float:
        if not times:
            return 0.0
        idx = bisect.bisect_left(times, t_mono)
        if idx <= 0:
            return steers[0]
        if idx >= len(times):
            return steers[-1]
        before, after = times[idx - 1], times[idx]
        return steers[idx - 1] if (t_mono - before) <= (after - t_mono) else steers[idx]

    return at


def _near_vids(frames, radius_m: float = 150.0) -> set[int]:
    """Ids that came within `radius_m` at some point, so glyphs stay unique."""
    out: set[int] = set()
    for f in frames:
        snap = f.snapshot
        for veh in snap.vehicles:
            fwd, lat = world_to_ego(veh["x"], veh["z"], snap.ego_x, snap.ego_z,
                                    snap.ego_yaw)
            if math.hypot(fwd, lat) <= radius_m:
                out.add(int(veh["vid"]))
    return out


def _scene_times(feats: ClipFeatures, frames) -> list[float]:
    base = feats.action_t
    picks = [base - 1.5, base, base + 1.0]
    span = [f.t_rel for f in frames]
    lo, hi = min(span), max(span)
    return [min(max(t, lo), hi) for t in picks]


def replay_block(clip) -> list[str]:
    """What the working-tree AEB does on this clip now, versus the label."""
    from core.aeb.clip_eval import outcome_under, run_headless

    evs = run_headless(clip)
    out = outcome_under(clip, burn_in_s=clip.metadata.burn_in_s)
    braked = [e for e in evs if e.aeb_brake]
    warned = [e for e in evs if e.aeb_warn]
    clears = sorted({vid for e in evs for vid in e.clearance_clears_ids})
    lines = [
        "CURRENT CODE (working tree replay, not the recorded decision)",
        f"  verdict     {out.verdict}   engaged={out.engaged}   "
        f"peak {out.peak_decel_ms2:.2f} m/s2",
        f"  brake       {'-' if not braked else f'{braked[0].t_rel:.2f} .. {braked[-1].t_rel:.2f} s'}"
        f"   warn "
        f"{'-' if not warned else f'{warned[0].t_rel:.2f} .. {warned[-1].t_rel:.2f} s'}",
        f"  peak clearance demand "
        f"{max((e.clearance_required_ms2 for e in evs), default=0.0):.2f} m/s2"
        f"   ids that clear the corridor: {clears or 'none'}",
    ]
    return lines


def ab_block(result) -> list[str]:
    """How commit 9eee3dd's clearance rewrite moves this clip, if it does."""
    if result is None:
        return []
    head = f"FLAG A/B    {result.flag}: {result.summary()}"
    if not result.changed or result.error:
        return [head]
    return [head,
            f"  off (pre-9eee3dd) brake {result.brake_ticks_off} ticks, peak "
            f"required {result.peak_required_off:.2f} m/s2",
            f"  on  (shipped)     brake {result.brake_ticks_on} ticks, peak "
            f"required {result.peak_required_on:.2f} m/s2"]


def build(row: ClipRow, clip, *, feats: ClipFeatures | None = None,
          scenes: int = 3, timeline: bool = True, replay: bool = False,
          ab_result=None) -> str:
    """Full text dossier for one clip."""
    feats = feats if feats is not None else extract(clip, row.path)
    frames = replay_clip(clip)
    vid = feats.primary_vid
    parts: list[list[str]] = [
        _header(row, feats),
        _label_block(row),
        ["SCENE       " + (", ".join(feats.scenario) or "(untagged)")],
        (["FLAGS       " + f for f in feats.flags] if feats.flags
         else ["FLAGS       none"]),
        _driver_block(feats),
        _live_block(feats),
        _target_block(feats),
    ]
    ab_lines = ab_block(ab_result)
    if ab_lines:
        parts.append(ab_lines)
    text = "\n".join("\n".join(block) for block in parts)
    if replay:
        text += "\n" + "\n".join(replay_block(clip))
    if timeline:
        text += "\n" + "\n".join(_timeline(frames, feats, vid))
    if scenes > 0 and frames:
        glyphs = glyph_map(_near_vids(frames))
        times = _scene_times(feats, frames)[:scenes]
        picked = []
        for t in times:
            best = min(frames, key=lambda f: abs(f.t_rel - t))
            if best not in picked:
                picked.append(best)
        text += "\n\nSCENES\n" + "\n\n".join(
            render(f, glyphs=glyphs) for f in picked)
    return text


def one_line(row: ClipRow, feats: ClipFeatures | None) -> str:
    """Compact listing row: id, label, scenario, the numbers that decide most calls."""
    lbl = f"{row.label_class or '-':<17}s{row.severity}"
    if feats is None:
        return f"{row.short}  {row.origin:<6} {row.session_kind:<3} {lbl}  (no features)"
    prim = feats.primary() or {}
    gap = _f(prim.get("min_corridor_gap_m", math.inf), 1)
    return (f"{row.short}  {row.origin:<6} {row.session_kind:<3} {lbl}  "
            f"{feats.ego_speed_at_action_kmh:5.1f}km/h  gap {gap:>7}  "
            f"ttc {_f(feats.min_ttc_s):>5}  "
            f"w{feats.warn_ticks:<3} b{feats.brake_ticks:<3}  "
            f"{','.join(feats.scenario)}")
