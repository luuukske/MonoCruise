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


def _header(row: ClipRow, feats: ClipFeatures, full: bool = False) -> list[str]:
    """Clip header. Capture time, client version and file name only with `full`.

    Agent dossiers leave them out: an agent never needs them, and contributed clips
    go to a hosted model with as little attached as possible.
    """
    mass = ("?" if feats.ego_mass_kg is None
            else f"{feats.ego_mass_kg / 1000.0:.1f} t")
    ident = [f"CLIP {row.clip_id}  ({row.origin} store)"]
    if full:
        ident += [f"  file        {Path(row.path).name}",
                  f"  captured    {row.captured_at}  client {row.client_version}"]
    return ident + [
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


def _live_block(feats: ClipFeatures, flags: dict | None = None) -> list[str]:
    band = ("none" if not feats.recorded_band else
            f"{feats.recorded_band[0]:.2f} .. {feats.recorded_band[1]:.2f} s")
    episodes = (flags or {}).get("brake_episodes")
    return [
        "RECORDED AEB (what the shipped build did at capture time)",
        f"  warn {feats.warn_ticks} ticks, brake {feats.brake_ticks} ticks, "
        f"engaged {feats.engaged_ticks} ticks"
        + (f", {episodes} separate brake episode(s)" if episodes is not None else ""),
        f"  first warn {_f(feats.first_warn_t) if feats.first_warn_t is not None else '-'}"
        f"  first brake "
        f"{_f(feats.first_brake_t) if feats.first_brake_t is not None else '-'}",
        f"  peak target {feats.peak_target_ms2:.2f} m/s2, peak required "
        f"{_f(feats.peak_required_ms2)} m/s2, capacity {feats.max_brake_ms2:.2f}",
        f"  min ttc {_f(feats.min_ttc_s)} s, min ttb {_f(feats.min_ttb_s)} s",
        f"  reaction band {band}   (proposal only, never ground truth)",
    ]


def _lane_cf_line(ctx: dict | None) -> list[str]:
    """The map-lane ghost's verdict, when the context ran one."""
    from tools.aeb_agent.lane_ghost import grade

    lcf = (ctx or {}).get("lane_cf") or {}
    if not lcf:
        return []
    if not lcf.get("ran"):
        return [f"  lane ghost    not run: {lcf.get('reason', '?')}"]
    g = grade(lcf)
    held = "held its speed" if lcf.get("mode") == "lane_held_speed" else "kept the recorded speed"
    line = (f"  lane ghost    followed ego's map lane and {held} for {lcf.get('horizon_s', 0):.2f} s"
            + (" (stopped at a junction)" if lcf.get("stops_at_branch") else ""))
    verdict = (f"  lane verdict  {g['verdict'].upper()}: nearest approach "
               f"{lcf.get('min_separation_m', 0):+.1f} m to vid {lcf.get('target_vid')} at "
               f"{lcf.get('dt_at_min_s', 0):.2f} s past the fork, drift +/-{g['drift_m']:.1f} m"
               + ("" if g["credible"] else "; not trustworthy this far out"))
    return [line, verdict]


def _driver_block(feats: ClipFeatures, ctx: dict | None = None) -> list[str]:
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
    lane = _lane_cf_line(ctx)
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
        return out + lane
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
    out += lane
    if cf.get("verdict") in ("collides", "likely") or any(
            v in line for line in lane for v in ("COLLIDES", "LIKELY")):
        out.append("  READ THIS     the driver avoided contact. The clean pass on "
                   "the record is the rescue, not the scene.")
    if lane and "swerve" in (inter.get("kind") or "") and "brake" not in (inter.get("kind") or ""):
        out.append("  which ghost   prefer the lane ghost: after a swerve it drifts 1.5 m by "
                   "2 s where the held-yaw ghost drifts 4.6 m (p90)")
    elif lane:
        out.append("  which ghost   after a brake or lift both ghosts guess the speed and "
                   "drift alike; weigh them equally")
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


_REL_WORDS = {
    "ego_lane": "in ego's lane",
    "ego_lane_wrong_way": "in ego's lane, driving toward ego",
    "across_ego_lane": "across ego's lane (crossing or turning)",
    "adjacent_left": "in the next lane on the left, same direction",
    "adjacent_right": "in the next lane on the right, same direction",
    "oncoming_left": "in an oncoming lane on the left",
    "oncoming_right": "in an oncoming lane on the right",
    "crossing_left": "crossing ego's road, now on the left",
    "crossing_right": "crossing ego's road, now on the right",
    "beside_road_left": "beside ego's road on the left",
    "beside_road_right": "beside ego's road on the right",
    "beyond_route": "beyond the stretch of road the map route covers",
    "off_lane": "not on any mapped lane (parked, yard or shoulder)",
    "unknown": "no map relation",
}


def rel_words(rel: str) -> str:
    """Plain words for a context `map_rel` code."""
    if not rel:
        return _REL_WORDS["unknown"]
    behind = rel.startswith("behind_")
    core, _, extra = rel.removeprefix("behind_").partition("+")
    text = _REL_WORDS.get(core, core)
    if behind:
        text = "behind ego, " + text
    if extra == "off_lane":
        text += ", and not on any mapped lane (parked, yard or shoulder)"
    return text


def _what(t: dict) -> str:
    """Short description of a context target: who drives it and roughly what it is."""
    who = "TMP" if t.get("is_tmp") else "AI"
    length = t.get("length_m") or 0.0
    if t.get("is_trailer"):
        body = "trailer"
    elif length < 6.5:
        body = "car"
    elif length < 13.0:
        body = "truck"
    else:
        body = "long rig"
    return f"{who} {body}"


def _ref(t: dict) -> dict:
    return t.get("approach") or t.get("action") or {}


def _path_words(t: dict, ctx: dict) -> str:
    """What ego's real path says about one target, in a sentence fragment."""
    p = _ref(t)
    d, s = p.get("path_d"), p.get("path_s")
    if d is None or s is None:
        return "ego's real path gives no reading"
    if s < 0.0:
        return "it was already behind ego"
    verdict = _VERDICT_WORDS.get(t.get("in_path", ""), "")
    if p.get("path_reached"):
        when = f", ego got there at {p['t_arrive']:.2f} s" if p.get("t_arrive") is not None else ""
        return (f"ego's real path later passed {abs(d):.1f} m from its centre{when} "
                f"({verdict})")
    how = ("its map lane" if ctx.get("map_matched") and (ctx.get("ego") or {}).get("on_lane")
           else "a straight line")
    return (f"ego ended {p.get('path_short_m', 0.0):.1f} m short of its centre; continuing ego's path "
            f"along {how} passes {abs(d):.1f} m from its centre ({verdict})")


_VERDICT_WORDS = {
    "in_path": "bodies overlap: it was in ego's path",
    "brushing": "within 1 m of touching",
    "clear": "clear of ego's path",
    "unknown": "no reading",
}


def _ego_words(ctx: dict) -> list[str]:
    if not ctx.get("map_matched"):
        return ["Road: this clip did not match the local game map (area not installed, "
                "or a TruckersMP layout that has changed). Judge lanes from ego's real path."]
    ego = ctx.get("ego") or {}
    if not ego.get("on_lane"):
        return ["Road: ego is not on any mapped lane at the action (yard, car park, fuel "
                "station or off-road). Lane words below are weak here."]
    kind = "a junction lane" if ego.get("lane_kind") == "prefab" else "a road lane"
    lanes = ""
    if ego.get("lanes_total"):
        same = ego.get("lanes_same_dir")
        lanes = (f" ({same} lane(s) its way, {ego['lanes_total']} in all)" if same
                 else f" ({ego['lanes_total']} lanes in all)")
    out = [f"Road: ego on {kind}{lanes}, {ego.get('offset_m', 0.0):+.1f} m right of the "
           f"lane centre, heading {ego.get('heading_err_deg', 0.0):+.0f} deg off it."]
    b50, b100 = ego.get("bend_50_deg"), ego.get("bend_100_deg")
    if b50 is None and b100 is None:
        shape = "unknown past the map's reach"
    elif max(abs(b50 or 0.0), abs(b100 or 0.0)) < 5.0:
        shape = "straight for 100 m"
    else:
        ref = b100 if b100 is not None else b50
        turn = "right" if ref > 0 else "left"
        shape = (f"bends {turn}: {abs(b50 or 0.0):.0f} deg within 50 m, "
                 f"{abs(b100 or 0.0):.0f} deg within 100 m")
    bits = [f"Road ahead {shape}."]
    if ego.get("branch_s") is not None:
        bits.append(f"Junction or exit {ego['branch_s']:.0f} m ahead: ego's route past it "
                    f"is not certain.")
    if ego.get("merge_s") is not None:
        bits.append(f"Two lanes merge {ego['merge_s']:.0f} m ahead.")
    if ego.get("lane_change_t") is not None:
        bits.append(f"Ego changed lane at {ego['lane_change_t']:.2f} s.")
    out.append(" ".join(bits))
    return out


def _conflicts(t: dict) -> list[str]:
    """Where the straight-ahead numbers, the map and ego's real path disagree."""
    p = _ref(t)
    out = []
    lat, d, rel = p.get("lat"), p.get("path_d"), p.get("map_rel") or ""
    reach = 1.265 + (t.get("half_w_m") or 1.0)
    in_path = t.get("in_path") == "in_path"
    if lat is not None and d is not None and p.get("path_s", -1) is not None \
            and (p.get("path_s") or -1) >= 0:
        if abs(lat) <= 1.95 and abs(d) > reach + 1.0:
            out.append(f"vid {t['vid']}: the straight-ahead view puts it in ego's lane "
                       f"(lat {lat:+.1f}), but ego's real path passed {abs(d):.1f} m off it.")
        elif abs(lat) > 1.95 + 1.0 and in_path:
            out.append(f"vid {t['vid']}: the straight-ahead view puts it {lat:+.1f} m to the "
                       f"side, but it was in ego's path (bend or turn).")
    if rel.startswith("ego_lane") and d is not None and abs(d) > reach + 1.0:
        out.append(f"vid {t['vid']}: map says ego's lane, ego's real path passed "
                   f"{abs(d):.1f} m off: a lane change, a junction turn or a map offset.")
    if (rel.startswith("adjacent") or rel.startswith("oncoming")) and in_path:
        out.append(f"vid {t['vid']}: map says another lane, yet ego's real path ran through "
                   f"it: ego changed lane, or the map lane is offset here.")
    return out


OVERLAP_M = -0.5


def overlapping(ctx: dict | None) -> list[dict]:
    """Vehicles whose body went more than half a metre into ego's, deepest first."""
    hits = [t for t in (ctx or {}).get("targets") or []
            if t.get("real_min_sep_m", 1e9) < OVERLAP_M]
    return sorted(hits, key=lambda t: t["real_min_sep_m"])


def clip_flags(clip, frames) -> dict:
    """Per-clip facts the cached features lack: NCZ share and separate AEB brake episodes."""
    radar = list(clip.radar_frames)
    ncz = sum(1 for f in radar if getattr(f, "tmp_ncz", False)) / max(len(radar), 1)
    episodes, last = 0, None
    for f in frames:
        if f.live_aeb.aeb_brake:
            if last is None or f.t_rel - last > 1.0:
                episodes += 1
            last = f.t_rel
    return {"ncz_frac": round(ncz, 3), "brake_episodes": episodes}


def situation_block(feats: ClipFeatures, ctx: dict | None) -> list[str]:
    """Plain-language summary of the scene, computed from the recording and the map."""
    out = ["SITUATION (computed from the recording and the local map, not AEB's opinion)"]
    if not ctx or ctx.get("error") == "too few usable ticks":
        out.append("  no scene context for this clip")
        return out
    for line in _ego_words(ctx):
        out.append("  " + line)
    out.append(f"  Ego: {feats.ego_speed_at_action_kmh:.0f} km/h at the action "
               f"(t={ctx.get('t_action', 0.0):.2f} s), {ctx.get('ego_end_speed_kmh', 0.0):.0f} "
               f"km/h at the end of the clip, {ctx.get('ego_travel_m', 0.0):.0f} m driven in all.")
    over = overlapping(ctx)
    if over:
        who = ", ".join(f"vid {t['vid']} ({t['real_min_sep_m']:+.1f} m)" for t in over[:4])
        out.append(f"  OVERLAP {len(over)} vehicle(s) drove into ego's body without a crash "
                   f"response: {who}. In TruckersMP that is ghosting (no-collision zone or desync).")
    ncz = (ctx.get("_clip_flags") or {}).get("ncz_frac")
    if ncz:
        out.append(f"  TruckersMP no-collision zone was active on {ncz:.0%} of frames.")
    targets = {t["vid"]: t for t in ctx.get("targets") or []}
    hp = ctx.get("hindsight_primary")
    if hp is not None and hp in targets:
        t = targets[hp]
        p = _ref(t)
        out.append(f"  Primary by hindsight: vid {hp} ({_what(t)}, {t.get('length_m', 0):.1f} m, "
                   f"{t.get('speed_kmh', 0):.0f} km/h), {rel_words(p.get('map_rel', ''))} "
                   f"at {p.get('t', 0):.2f} s; {_path_words(t, ctx)}. Closest real gap "
                   f"{t.get('real_min_sep_m', 0):.1f} m at {t.get('t_real_min', 0):.2f} s.")
    else:
        out.append("  Primary by hindsight: none. No vehicle sat in ego's real or "
                   "extended path at the approach.")
    if feats.primary_vid is not None and feats.primary_vid != hp:
        t = targets.get(feats.primary_vid)
        where = (f"{rel_words(_ref(t).get('map_rel', ''))}; {_path_words(t, ctx)}"
                 if t else "not near ego's path at the approach")
        out.append(f"  The straight-corridor primary is vid {feats.primary_vid} instead: {where}.")
    for t in (ctx.get("targets") or [])[:5]:
        if t["vid"] not in (hp, feats.primary_vid) and t.get("real_min_sep_m", 1e9) > 5.0:
            continue
        for line in _conflicts(t):
            out.append("  CONFLICT " + line)
    return out


def evidence_block(ctx: dict | None, limit: int = 8) -> list[str]:
    """One row per nearby vehicle: ego's real path, the map lane, and the real outcome."""
    if not ctx or not ctx.get("targets"):
        return []
    out = [f"EVIDENCE (at the approach, t={ctx.get('t_approach', 0):.2f} s; path_d is "
           f"how far right of ego's real path the vehicle sat, + right / - left)",
           "   vid   what         inPath    path_d  path_s  reached     map               "
           "map_d  laneDist  realGap@t"]
    hp = ctx.get("hindsight_primary")
    for t in ctx["targets"][:limit]:
        p = _ref(t)
        mark = "*" if t["vid"] == hp else " "
        reached = ("yes" if p.get("path_reached")
                   else f"short {p.get('path_short_m', 0):.0f}m" if p.get("path_s") is not None
                   else "-")
        rel = (p.get("map_rel") or "unknown")[:17]
        out.append(
            f"  {mark}{t['vid']:<6}{_what(t):<13}{t.get('in_path', '?'):<9}"
            f"{_num(p.get('path_d'), '+7.1f')} {_num(p.get('path_s'), '7.1f')}  "
            f"{reached:<10}  {rel:<17} {_num(p.get('map_d'), '+6.1f')}  "
            f"{_num(p.get('lane_dist_m'), '7.1f')}   "
            f"{t.get('real_min_sep_m', 0):5.1f}@{t.get('t_real_min', 0):.2f}")
    return out


def _num(value, fmt: str) -> str:
    if value is None:
        width = int("".join(ch for ch in fmt.split(".")[0] if ch.isdigit()) or 6)
        return "-".rjust(width)
    return format(value, fmt)


def _role(t: dict) -> str:
    if t["speed_max_kmh"] < 2.0:
        return "stationary"
    dot = t["heading_dot_at_min"]
    if dot >= 0.6:
        return "codir" if t["ahead_frac"] >= 0.5 else "overtaker"
    if dot <= -0.6:
        return "oncoming"
    return "crossing"


def _demand_at(demand: list, t: float) -> float | None:
    """Hindsight demand nearest `t`, from the context's (t, req, gap, v) rows."""
    if not demand:
        return None
    times = [row[0] for row in demand]
    j = bisect.bisect_left(times, t)
    cands = [k for k in (j - 1, j) if 0 <= k < len(demand)]
    k = min(cands, key=lambda i: abs(times[i] - t))
    return demand[k][1] if abs(times[k] - t) <= 0.1 else None


def _timeline(frames, feats: ClipFeatures, vid: int | None,
              steer_at=None, demand=None) -> list[str]:
    """Per-tick table around the action, thinned to a readable row count."""
    if not frames:
        return ["TIMELINE    (no ticks)"]
    lo = max(0.0, feats.action_t - 4.0)
    hi = feats.action_t + 3.0
    window = [f for f in frames if lo <= f.t_rel <= hi] or frames
    step = max(1, len(window) // _TIMELINE_MAX_ROWS)
    out = [
        f"TIMELINE    ticks {lo:.1f}-{hi:.1f} s, every {step} tick(s); "
        f"tracking vid {vid}"
        + ("; hsReq = decel needed to stop closing on it along ego's real path"
           if demand else ""),
        "     t     ego   state  ttc    ttb    req    raw   targ  "
        "fwd    lat   vspd  steer  gas  brk  " + ("hsReq  " if demand else "") + "colliding",
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
        if demand:
            hs = _demand_at(demand, f.t_rel)
            coll = ("   -   " if hs is None else f"{hs:6.2f} ") + coll
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
          ab_result=None, ctx: dict | None = None, full_header: bool = False) -> str:
    """Full text dossier for one clip.

    `ctx` adds the SITUATION and EVIDENCE blocks and real-path scenes.
    """
    feats = feats if feats is not None else extract(clip, row.path)
    frames = replay_clip(clip)
    flags = clip_flags(clip, frames)
    if ctx is not None:
        ctx = dict(ctx, _clip_flags=flags)
    vid = feats.primary_vid
    if ctx and ctx.get("hindsight_primary") is not None:
        vid = ctx["hindsight_primary"]
    parts: list[list[str]] = [_header(row, feats, full=full_header), _label_block(row)]
    if ctx is not None:
        parts.append(situation_block(feats, ctx))
    parts += [
        ["SCENE       " + (", ".join(feats.scenario) or "(untagged)")],
        (["FLAGS       " + f for f in feats.flags] if feats.flags
         else ["FLAGS       none"]),
        _driver_block(feats, ctx),
        _live_block(feats, flags),
    ]
    evidence = evidence_block(ctx)
    if evidence:
        parts.append(evidence)
    parts.append(_target_block(feats))
    ab_lines = ab_block(ab_result)
    if ab_lines:
        parts.append(ab_lines)
    text = "\n".join("\n".join(block) for block in parts)
    if replay:
        text += "\n" + "\n".join(replay_block(clip))
    if timeline:
        demand_vid = (ctx or {}).get("demand_vid", (ctx or {}).get("window_target"))
        demand = (ctx or {}).get("demand") if demand_vid == vid else None
        text += "\n" + "\n".join(_timeline(frames, feats, vid, demand=demand))
    if scenes > 0 and frames:
        glyphs = glyph_map(_near_vids(frames))
        overlays = (ctx or {}).get("overlays") or {}
        if overlays:
            times = [overlays[k]["t"] for k in ("approach", "action") if k in overlays]
            times.append(feats.action_t + 1.0)
        else:
            times = _scene_times(feats, frames)
        picked = []
        for t in times[:scenes]:
            best = min(frames, key=lambda f: abs(f.t_rel - t))
            if best not in picked:
                picked.append(best)
        rendered = []
        for f in picked:
            overlay = next((o for o in overlays.values() if abs(o["t"] - f.t_rel) <= 0.06), None)
            rendered.append(render(f, glyphs=glyphs, overlay=overlay))
        text += "\n\nSCENES\n" + "\n\n".join(rendered)
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
