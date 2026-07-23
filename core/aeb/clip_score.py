"""Corpus objective over labelled clip replays: safety-weighted FN/late, comfort FP, TP reward."""

from __future__ import annotations

from dataclasses import dataclass, field

from core.aeb.calibration import AEBCalibration, DEFAULT as _CAL_DEFAULT
from core.aeb.clip_eval import Outcome, outcome_under
from core.aeb.clip_schema import Clip

# Safety must dominate comfort in the objective.
SAFETY_SCALE: float = 10.0
COMFORT_SCALE: float = 1.0
TP_REWARD: float = 2.0
LATE_SAFETY_FRACTION: float = 0.5
# A false positive always costs at least this fraction of a full-duration brake,
# so even a brief phantom brake is penalised.
FP_MIN_DURATION_S: float = 0.3


@dataclass
class ClipScore:
    clip_id: str
    label_class: str
    severity: int
    verdict: str
    cost: float
    detail: str = ""


@dataclass
class CorpusScore:
    total_cost: float = 0.0
    scored: int = 0
    verdict_counts: dict = field(default_factory=dict)
    per_clip: list = field(default_factory=list)   # list[ClipScore]


def class_window_warning(class_: str, has_window: bool) -> str | None:
    """Label class vs should-trigger window consistency; None if OK."""
    if class_ == "ignore":
        return None
    if class_ == "fp" and has_window:
        return "class 'fp' but has a should-trigger window (fp must be must-not-trigger)"
    if class_ in ("tp", "good_intervention", "fn") and not has_window:
        return f"class '{class_}' but no should-trigger window"
    return None


def label_inconsistency(clip: Clip) -> str | None:
    """Flag a clip whose label class disagrees with its should-trigger window."""
    lbl = clip.metadata.label
    if lbl is None:
        return None
    return class_window_warning(lbl.class_, lbl.should_trigger is not None)


def audit_corpus(clips) -> list[tuple[str, str]]:
    """List (clip_id, message) for every clip with a class/window mismatch."""
    out = []
    for clip in clips:
        msg = label_inconsistency(clip)
        if msg is not None:
            out.append((clip.metadata.clip_id, msg))
    return out


def _severity(clip: Clip) -> int:
    lbl = clip.metadata.label
    return max(1, int(lbl.severity)) if (lbl and lbl.severity) else 3


def _overlap(a: tuple | None, b: tuple | None) -> float:
    if a is None or b is None:
        return 0.0
    lo = max(a[0], b[0])
    hi = min(a[1], b[1])
    return max(0.0, hi - lo)


def _tp_quality(clip: Clip, o: Outcome) -> float:
    """0..1: window overlap fraction, scaled by closeness to desired peak decel."""
    st = o.should_trigger
    st_len = (st[1] - st[0]) if st else 0.0
    if st_len > 1e-6:
        overlap = min(1.0, _overlap(o.brake_window, st) / st_len)
    else:
        overlap = 1.0
    peak_match = 1.0
    desired = clip.metadata.label.desired_peak_decel_ms2 if clip.metadata.label else None
    if desired:
        peak_match = max(0.0, 1.0 - abs(o.peak_decel_ms2 - desired) / max(desired, 1e-3))
    return overlap * peak_match


def _cost(clip: Clip, o: Outcome, sev: int, cal: AEBCalibration) -> tuple[float, str]:
    if o.verdict == "false_negative":
        return sev * SAFETY_SCALE, "MISS: should have braked, never did"
    if o.verdict == "late":
        return sev * SAFETY_SCALE * LATE_SAFETY_FRACTION, "LATE: engaged after the window"
    if o.verdict == "false_positive":
        dur = (o.brake_window[1] - o.brake_window[0]) if o.brake_window else 0.0
        intensity = o.peak_decel_ms2 / max(cal.full_brake_decel, 1e-3)
        return (sev * COMFORT_SCALE * intensity * (FP_MIN_DURATION_S + dur),
                f"FP: braked {dur:.2f}s @ {o.peak_decel_ms2:.1f} m/s2")
    if o.verdict == "true_positive":
        q = _tp_quality(clip, o)
        return -sev * TP_REWARD * q, f"TP: correct (quality {q:.2f})"
    if o.verdict == "true_negative":
        return 0.0, "TN: correctly silent"
    return 0.0, o.verdict


def score_clip(clip: Clip, cal: AEBCalibration = _CAL_DEFAULT,
               burn_in_s: float | None = None) -> ClipScore | None:
    """Score one labelled clip under ``cal``. None for unlabelled / ignore clips."""
    lbl = clip.metadata.label
    if lbl is None or lbl.class_ == "ignore":
        return None
    burn = burn_in_s if burn_in_s is not None else clip.metadata.burn_in_s
    o = outcome_under(clip, cal, burn_in_s=burn)
    sev = _severity(clip)
    cost, detail = _cost(clip, o, sev, cal)
    return ClipScore(clip.metadata.clip_id, lbl.class_, sev, o.verdict, cost, detail)


def score_corpus(clips, cal: AEBCalibration = _CAL_DEFAULT,
                 burn_in_s: float | None = None) -> CorpusScore:
    """Aggregate cost over labelled clips. Lower total is safer + comfier."""
    cs = CorpusScore()
    for clip in clips:
        sc = score_clip(clip, cal, burn_in_s)
        if sc is None:
            continue
        cs.scored += 1
        cs.total_cost += sc.cost
        cs.verdict_counts[sc.verdict] = cs.verdict_counts.get(sc.verdict, 0) + 1
        cs.per_clip.append(sc)
    return cs


def score_store(store, cal: AEBCalibration = _CAL_DEFAULT) -> CorpusScore:
    """Score every labelled clip in a ClipStore under ``cal``."""
    clips = []
    for info in store.list_clips():
        clip = store.load(info.path)
        if clip is not None:
            clips.append(clip)
    return score_corpus(clips, cal)


def format_corpus(cs: CorpusScore, title: str = "") -> str:
    lines = []
    if title:
        lines.append(title)
    for sc in cs.per_clip:
        lines.append(
            f"  {sc.cost:+7.2f}  {sc.verdict:15s} sev{sc.severity} "
            f"[{sc.label_class}] {sc.clip_id[:8]}  {sc.detail}"
        )
    counts = "  ".join(f"{k}={v}" for k, v in sorted(cs.verdict_counts.items()))
    lines.append(f"  TOTAL {cs.total_cost:+.2f}  over {cs.scored} clips   ({counts})")
    return "\n".join(lines)
