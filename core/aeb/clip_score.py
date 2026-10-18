"""Corpus objective over labelled clip replays: filtering, not brake timing (README corpus scoring).

Positives are judged on whether the geometry would let AEB engage inside the window,
so the engage bar and the rest of the longitudinal tuning cannot move them. Every false
trigger costs the same flat amount, however long it held or however hard it braked.
"""

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
# One false trigger: a brake in a must-not-trigger clip, or a stray one in a positive.
FP_TRIGGER_COST: float = 1.0
# Warn-only FP vs a brake FP (beep << slam).
FP_WARN_SCALE: float = 0.25


@dataclass
class ClipScore:
    clip_id: str
    label_class: str
    severity: int
    verdict: str
    cost: float
    detail: str = ""
    stray: bool = False              # positive clip that also braked outside its window
    braked_in_window: bool = False   # diagnostic, never scored


_POSITIVE_VERDICTS = ("true_positive", "false_negative", "late")


@dataclass
class CorpusScore:
    total_cost: float = 0.0
    scored: int = 0
    verdict_counts: dict = field(default_factory=dict)
    per_clip: list = field(default_factory=list)   # list[ClipScore]

    @property
    def stray_clips(self) -> int:
        return sum(1 for sc in self.per_clip if sc.stray)

    @property
    def positives_braked(self) -> tuple[int, int]:
        """(braked inside the window, positives): what the geometric verdict cannot see."""
        pos = [sc for sc in self.per_clip if sc.verdict in _POSITIVE_VERDICTS]
        return sum(1 for sc in pos if sc.braked_in_window), len(pos)


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


def _cost(o: Outcome, sev: int) -> tuple[float, str]:
    """Verdict cost; a positive with stray onsets also pays one flat false trigger."""
    if o.verdict == "false_positive":
        dur = (o.brake_window[1] - o.brake_window[0]) if o.brake_window else 0.0
        return (sev * COMFORT_SCALE * FP_TRIGGER_COST,
                f"FP: braked {dur:.2f}s @ {o.peak_decel_ms2:.1f} m/s2")
    if o.verdict == "false_warn":
        dur = (o.warn_window[1] - o.warn_window[0]) if o.warn_window else 0.0
        return (sev * COMFORT_SCALE * FP_TRIGGER_COST * FP_WARN_SCALE,
                f"FP-warn: {dur:.2f}s")
    if o.verdict == "true_negative":
        return 0.0, "TN: correctly silent"
    if o.verdict == "false_negative":
        cost, detail = sev * SAFETY_SCALE, "MISS: geometry never let AEB engage in the window"
    elif o.verdict == "late":
        cost = sev * SAFETY_SCALE * LATE_SAFETY_FRACTION
        detail = "LATE: geometry let AEB engage only after the window"
    elif o.verdict == "true_positive":
        cost = -sev * TP_REWARD
        detail = ("TP: braked in the window" if o.braked_in_window
                  else "TP: geometry engages in the window, no brake")
    else:
        return 0.0, o.verdict
    if o.stray_onsets:
        cost += sev * COMFORT_SCALE * FP_TRIGGER_COST
        onsets = ", ".join(f"{t:.2f}" for t in o.stray_onsets)
        detail += f"; stray trigger at {onsets} s"
    return cost, detail


def score_clip(clip: Clip, cal: AEBCalibration = _CAL_DEFAULT,
               burn_in_s: float | None = None) -> ClipScore | None:
    """Score one labelled clip under ``cal``. None for unlabelled / ignore clips."""
    lbl = clip.metadata.label
    if lbl is None or lbl.class_ == "ignore":
        return None
    burn = burn_in_s if burn_in_s is not None else clip.metadata.burn_in_s
    o = outcome_under(clip, cal, burn_in_s=burn)
    sev = _severity(clip)
    cost, detail = _cost(o, sev)
    return ClipScore(clip.metadata.clip_id, lbl.class_, sev, o.verdict, cost, detail,
                     stray=bool(o.stray_onsets), braked_in_window=o.braked_in_window)


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
    braked, positives = cs.positives_braked
    lines.append(f"  stray triggers in {cs.stray_clips} positive clips; "
                 f"braked in window {braked}/{positives} positives (not scored)")
    return "\n".join(lines)
