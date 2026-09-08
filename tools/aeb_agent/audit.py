"""Mistag heuristics over the labelled corpus.

Every rule here is a *suspicion*, not a verdict. It reads recorded geometry and
the label, never the current AEB decision, so a rule cannot fire merely because
the tuning moved. Each hit names the evidence so a reviewer can disagree with it
from the same numbers. Nothing in this module writes a label.
"""

from __future__ import annotations

from dataclasses import dataclass

from tools.aeb_agent.corpus import ClipRow
from tools.aeb_agent.features import ClipFeatures

POSITIVE_CLASSES = ("tp", "fn", "good_intervention")
NEGATIVE_CLASSES = ("fp", "tn")

# A positive label needs something that could actually have been hit.
FAR_ONLY_M = 45.0
# Driver authority: above this the driver, not AEB, was handling the scene.
DRIVER_BRAKE = 0.35
# Below this a "severity 4-5" claim is not supported by the geometry.
WEAK_DEMAND_MS2 = 3.0
HOT_TTC_S = 0.9
# Anything this close counts as a near miss whether or not the corridor test saw
# it: a body alongside at 4 m did not need to be "ahead" to have mattered.
NEAR_MISS_M = 15.0
_SENTINEL = 1e8


def _dist(value: float) -> str:
    return "never in corridor" if value >= _SENTINEL else f"{value:.0f} m"


@dataclass
class Suspect:
    """One rule firing on one clip."""

    clip_id: str
    short: str
    rule: str
    severity: str          # high | medium | low, how likely the label is wrong
    message: str
    suggestion: str = ""

    def line(self) -> str:
        tail = f"  -> {self.suggestion}" if self.suggestion else ""
        return f"{self.short}  [{self.severity:<6}] {self.rule:<22} {self.message}{tail}"


def _primary(feats: ClipFeatures) -> dict:
    return feats.primary() or {}


def _evaded(feats: ClipFeatures) -> bool:
    """True when the driver intervened and holding course was not clearly safe.

    Any rule that reads "nothing was ever in the corridor" as evidence of no
    threat must consult this first: after a swerve the recorded geometry is the
    result of the rescue, not of the scene.
    """
    cf = feats.counterfactual or {}
    if not cf.get("ran"):
        return False
    return cf.get("verdict") in ("collides", "likely", "close")


def _cf_detail(feats: ClipFeatures) -> str:
    cf = feats.counterfactual or {}
    inter = feats.intervention or {}
    return (f"{inter.get('kind', '?')} at t={inter.get('t', 0):.2f}s, ghost "
            f"separation {cf.get('min_separation_m', 0):+.1f} m at "
            f"{cf.get('dt_at_min_s', 0):.2f} s (drift band "
            f"+/-{cf.get('drift_at_min_m', 0):.1f} m)")


def _closest_approach(feats: ClipFeatures) -> tuple[float, bool]:
    """Nearest range over every tracked target, and whether any entered the corridor.

    The rules that ask "could anything have been hit" must scan all targets, not
    only the primary: a clip can carry one distant crosser and two bodies inside
    touching distance.
    """
    if not feats.targets:
        return 1e9, False
    nearest = min(t.get("min_range_m", 1e9) for t in feats.targets)
    entered = any(t.get("corridor_ticks", 0) > 0 for t in feats.targets)
    return nearest, entered


def _desync(prim: dict) -> bool:
    """Same test as TargetTrack.lag_suspect, over a serialized track.

    Deliberately not the stall counters: measured against known-good clips they
    track the TMP update rate rather than a harmful desync. See the README.
    """
    return prim.get("lag_ticks", 0) >= 3 or prim.get("teleport_ticks", 0) >= 2


def _desync_detail(prim: dict) -> str:
    return (f"lag_confirmed x{prim.get('lag_ticks', 0)}, "
            f"{prim.get('teleport_ticks', 0)} teleports, max jump "
            f"{prim.get('max_jump_m', 0):.0f} m, "
            f"{prim.get('stall_ticks', 0)} stream-stall ticks")


def rule_class_window(row: ClipRow, feats: ClipFeatures) -> Suspect | None:
    warn = row.label_warning()
    if warn is None:
        return None
    return Suspect(row.clip_id, row.short, "class_window", "high", warn,
                   "fix the class or the window in the review UI")


def rule_positive_no_target(row: ClipRow, feats: ClipFeatures) -> Suspect | None:
    if row.label_class not in POSITIVE_CLASSES:
        return None
    if _evaded(feats):
        return None
    prim = _primary(feats)
    if not prim:
        return Suspect(row.clip_id, row.short, "positive_no_target", "high",
                       "positive label but no vehicle decoded at all",
                       "reclass to ignore")
    gap = prim.get("min_corridor_gap_m", 1e9)
    rng, entered = _closest_approach(feats)
    if not entered and rng > NEAR_MISS_M:
        return Suspect(row.clip_id, row.short, "positive_no_target", "high",
                       f"positive label but no target entered ego's corridor "
                       f"(closest range over every target {rng:.0f} m)",
                       "reclass to fp/tn or ignore")
    if gap > FAR_ONLY_M:
        return Suspect(row.clip_id, row.short, "positive_no_target", "medium",
                       f"positive label but the closest corridor gap was "
                       f"{gap:.0f} m", "check whether a real threat exists")
    return None


def rule_driver_handled(row: ClipRow, feats: ClipFeatures) -> Suspect | None:
    """A scene the driver braked through, and AEB never did, is not a miss.

    Gated on AEB staying silent: a driver dabbing the brake during a genuine
    intervention is ordinary and says nothing about the label.
    """
    if row.label_class not in POSITIVE_CLASSES:
        return None
    if feats.brake_ticks > 0 or feats.user_brake_max < DRIVER_BRAKE:
        return None
    if _evaded(feats):
        return None
    return Suspect(
        row.clip_id, row.short, "driver_handled", "medium",
        f"{row.label_class} label, AEB never braked, driver brake reached "
        f"{feats.user_brake_max:.2f} on {feats.user_brake_frac:.0%} of ticks",
        "if the driver resolved it, ignore rather than count it as a miss")


def rule_lag_positive(row: ClipRow, feats: ClipFeatures) -> Suspect | None:
    """TMP desync makes a frozen car look like a sudden obstacle."""
    if row.label_class not in POSITIVE_CLASSES:
        return None
    prim = _primary(feats)
    if not prim:
        return None
    if not _desync(prim):
        return None
    return Suspect(
        row.clip_id, row.short, "lag_positive", "high",
        f"primary vid {prim.get('vid')} is lag/desync suspect "
        f"({_desync_detail(prim)}) under a {row.label_class} label",
        "a frozen TMP body is not a threat the model should learn: ignore")


def rule_lag_negative(row: ClipRow, feats: ClipFeatures) -> Suspect | None:
    """The same desync on the FP side: the engagement was against a ghost."""
    if row.label_class not in NEGATIVE_CLASSES:
        return None
    prim = _primary(feats)
    if not prim or feats.brake_ticks == 0:
        return None
    if not _desync(prim):
        return None
    return Suspect(
        row.clip_id, row.short, "lag_negative", "medium",
        f"{row.label_class} clip whose primary is lag/desync suspect "
        f"({_desync_detail(prim)})",
        "scoring this as a tuning failure charges AEB for a TMP artefact")


def rule_crossing_cleared(row: ClipRow, feats: ClipFeatures) -> Suspect | None:
    """A crosser that was gone before ego arrived is not a miss."""
    if row.label_class != "fn":
        return None
    prim = _primary(feats)
    if not prim or "crossing" not in feats.scenario or _evaded(feats):
        return None
    ttc = prim.get("min_geom_ttc_s", 1e9)
    gap = prim.get("min_corridor_gap_m", 1e9)
    rng, _entered = _closest_approach(feats)
    if ttc < 2.0 or gap < 8.0 or rng <= NEAR_MISS_M:
        return None
    return Suspect(
        row.clip_id, row.short, "crossing_cleared", "high",
        f"crossing target closest range {rng:.0f} m, corridor gap "
        f"{_dist(gap)} (geometric ttc {ttc:.1f} s), yet is labelled fn",
        "perpendicular traffic that clears is a true negative, not a miss")


def rule_negative_looks_real(row: ClipRow, feats: ClipFeatures) -> Suspect | None:
    if row.label_class not in NEGATIVE_CLASSES:
        return None
    prim = _primary(feats)
    if not prim:
        return None
    ttc = prim.get("min_geom_ttc_s", 1e9)
    gap = prim.get("min_corridor_gap_m", 1e9)
    in_lane = prim.get("in_lane_frac", 0.0)
    if ttc > HOT_TTC_S or gap > 6.0 or in_lane < 0.2:
        return None
    return Suspect(
        row.clip_id, row.short, "negative_looks_real", "medium",
        f"{row.label_class} clip with an in-lane target at {gap:.1f} m and "
        f"geometric ttc {ttc:.2f} s ({in_lane:.0%} of ticks in lane)",
        "check whether this was a genuine threat mislabelled as a phantom")


def rule_evasion_rescued(row: ClipRow, feats: ClipFeatures) -> Suspect | None:
    """A negative label on a scene the driver steered or braked out of.

    This is the label error that recorded geometry cannot show: the swerve makes
    a real threat look like a clean pass, so the clip reads as a true negative.

    One-directional on purpose. A `clear` verdict never gets a matching rule that
    demotes a positive label: a wrong `clear` deletes a real threat from the
    corpus silently, while a wrong `collides` only sends a clip back for review.
    Same asymmetry as `core/aeb/README.md` on measured misses, which may remove
    certainty and never grant it.
    """
    if row.label_class not in ("fp", "tn", "ignore"):
        return None
    cf = feats.counterfactual or {}
    if not cf.get("ran") or not cf.get("credible"):
        return None
    verdict = cf.get("verdict")
    if verdict not in ("collides", "likely"):
        return None
    hit = next((x for x in feats.targets
                if x.get("vid") == cf.get("target_vid")), None)
    if hit is not None and _desync(hit):
        return None
    sev = "high" if verdict == "collides" else "medium"
    return Suspect(
        row.clip_id, row.short, "evasion_rescued", sev,
        f"{row.label_class} label, but the driver intervened and holding course "
        f"{verdict}: {_cf_detail(feats)}",
        "the driver avoided this, AEB did not: reconsider fn or a positive class")


def rule_evasion_unlabelled_positive(row: ClipRow,
                                     feats: ClipFeatures) -> Suspect | None:
    """A positive window opening after an evasive swerve describes the rescue.

    Gated on the swerve actually being evasive. Firing on any swerve flagged 74
    clips whose "swerve" was a routine corner four seconds before the threat, and
    reopening a window onto that corner would poison it.
    """
    if row.label_class not in ("tp", "fn", "good_intervention") or not row.window:
        return None
    inter = feats.intervention or {}
    cf = feats.counterfactual or {}
    if not inter.get("found") or "swerve" not in inter.get("kind", ""):
        return None
    if not cf.get("credible") or cf.get("verdict") not in ("collides", "likely"):
        return None
    if row.window[0] <= inter.get("t", 0.0) + 0.1:
        return None
    return Suspect(
        row.clip_id, row.short, "window_after_evasion", "medium",
        f"window opens at {row.window[0]:.2f}s but the driver made an evasive "
        f"swerve at {inter.get('t', 0):.2f}s, so it starts after the rescue",
        "a should-trigger window must open before the driver had to act")


def rule_severity_outlier(row: ClipRow, feats: ClipFeatures) -> Suspect | None:
    if not row.labelled or row.label_class == "ignore":
        return None
    prim = _primary(feats)
    ttc = prim.get("min_geom_ttc_s", 1e9) if prim else 1e9
    if row.severity >= 4 and feats.peak_required_ms2 < WEAK_DEMAND_MS2 and ttc > 2.5:
        return Suspect(
            row.clip_id, row.short, "severity_outlier", "low",
            f"severity {row.severity} but peak required decel was only "
            f"{feats.peak_required_ms2:.1f} m/s2 and geometric ttc {ttc:.1f} s",
            "severity looks inflated")
    if row.severity and row.severity <= 1 and ttc < 0.7:
        return Suspect(
            row.clip_id, row.short, "severity_outlier", "low",
            f"severity {row.severity} but geometric ttc reached {ttc:.2f} s",
            "severity looks understated")
    return None


def rule_window_bounds(row: ClipRow, feats: ClipFeatures) -> Suspect | None:
    if not row.window:
        return None
    lo, hi = row.window
    if hi <= lo:
        return Suspect(row.clip_id, row.short, "window_bounds", "high",
                       f"window {lo:.2f}..{hi:.2f} is empty or reversed",
                       "re-mark the window")
    if lo < -0.01 or hi > feats.duration_s + 0.5:
        return Suspect(row.clip_id, row.short, "window_bounds", "high",
                       f"window {lo:.2f}..{hi:.2f} falls outside the clip "
                       f"(0..{feats.duration_s:.2f} s)", "re-mark the window")
    if hi - lo > 8.0:
        return Suspect(row.clip_id, row.short, "window_bounds", "low",
                       f"window spans {hi - lo:.1f} s, wider than a plausible "
                       f"reaction band", "tighten the window")
    return None


def rule_target_vid_absent(row: ClipRow, feats: ClipFeatures) -> Suspect | None:
    if row.target_vid is None or not feats.targets:
        return None
    known = {t["vid"] for t in feats.targets}
    if row.target_vid in known:
        return None
    return Suspect(row.clip_id, row.short, "target_vid_absent", "low",
                   f"label target_vid {row.target_vid} is not among the tracked "
                   f"vehicles", "the id may predate a re-decode; re-pick it")


def rule_window_matches_recorded(row: ClipRow, feats: ClipFeatures) -> Suspect | None:
    """A window identical to AEB's own reaction band may have been seeded, not judged."""
    if row.label_class != "fn" or not row.window or not feats.recorded_band:
        return None
    lo_d = abs(row.window[0] - feats.recorded_band[0])
    hi_d = abs(row.window[1] - feats.recorded_band[1])
    if lo_d > 0.05 or hi_d > 0.05:
        return None
    return Suspect(
        row.clip_id, row.short, "window_is_recorded", "medium",
        "fn window equals AEB's own reaction band to within 0.05 s",
        "an fn window seeded from AEB output encodes 'AEB was right'; re-mark it")


def rule_data_quality(row: ClipRow, feats: ClipFeatures) -> Suspect | None:
    if row.label_class in ("", "ignore"):
        return None
    reasons = [f for f in feats.flags
               if f.startswith(("paused", "short clip", "only ", "AEB disabled",
                                "no traffic"))]
    if not reasons:
        return None
    return Suspect(row.clip_id, row.short, "data_quality", "medium",
                   "; ".join(reasons), "unreliable capture: consider ignore")


RULES = (
    rule_class_window,
    rule_evasion_rescued,
    rule_evasion_unlabelled_positive,
    rule_positive_no_target,
    rule_driver_handled,
    rule_lag_positive,
    rule_lag_negative,
    rule_crossing_cleared,
    rule_negative_looks_real,
    rule_severity_outlier,
    rule_window_bounds,
    rule_target_vid_absent,
    rule_window_matches_recorded,
    rule_data_quality,
)
RULE_NAMES = tuple(
    [fn.__name__.removeprefix("rule_") for fn in RULES] + ["duplicate_clip"])
_ORDER = {"high": 0, "medium": 1, "low": 2}


def cross_store_duplicates(rows: list[ClipRow]) -> list[Suspect]:
    """One clip_id present in both stores: a relabel must reach both files."""
    seen: dict[str, list[ClipRow]] = {}
    for row in rows:
        seen.setdefault(row.clip_id, []).append(row)
    out = []
    for clip_id, group in seen.items():
        if len(group) < 2:
            continue
        classes = {r.label_class for r in group}
        sev = "high" if len(classes) > 1 else "low"
        out.append(Suspect(
            clip_id, group[0].short, "duplicate_clip", sev,
            f"present in {len(group)} stores ({', '.join(r.origin for r in group)}) "
            f"with class(es) {', '.join(sorted(c or '-' for c in classes))}",
            "apply writes every copy; check both if you edit by hand"))
    return out


def run(rows: list[ClipRow], feats_by_id: dict[str, ClipFeatures],
        only: str = "") -> list[Suspect]:
    """Every rule over every labelled clip that has features, ranked by suspicion."""
    wanted = {n.strip() for n in only.split(",") if n.strip()}
    out: list[Suspect] = []
    for row in rows:
        feats = feats_by_id.get(row.clip_id)
        if feats is None or not row.labelled:
            continue
        for fn in RULES:
            name = fn.__name__.removeprefix("rule_")
            if wanted and name not in wanted:
                continue
            hit = fn(row, feats)
            if hit is not None:
                out.append(hit)
    if not wanted or "duplicate_clip" in wanted:
        out.extend(cross_store_duplicates(rows))
    out.sort(key=lambda s: (_ORDER.get(s.severity, 3), s.rule, s.short))
    return out


def summarize(suspects: list[Suspect]) -> dict:
    by_rule: dict[str, int] = {}
    by_sev: dict[str, int] = {}
    for s in suspects:
        by_rule[s.rule] = by_rule.get(s.rule, 0) + 1
        by_sev[s.severity] = by_sev.get(s.severity, 0) + 1
    return {
        "total": len(suspects),
        "clips": len({s.clip_id for s in suspects}),
        "by_rule": dict(sorted(by_rule.items(), key=lambda kv: -kv[1])),
        "by_severity": by_sev,
    }
