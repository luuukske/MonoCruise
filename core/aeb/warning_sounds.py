"""AEB warning sounds: the original recording plus three synthesized styles, and how each repeats.

The styles are synthesized here, never sampled from a real car, so no third-party
audio ships with MonoCruise. Pure Python apart from ``load`` and
``load``; ``core/aeb/warning_player.py`` plays them. See ``core/aeb/README.md``
section 17.
"""

from __future__ import annotations

import logging
import math
from array import array
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

logger = logging.getLogger(__name__)

ORIGINAL_PATH = Path(__file__).resolve().parent / "AEB_warning.wav"

DEFAULT_VOLUME_PCT: int = 100
# Mixer level at 100%: the fixed 0.8 every warning played at before the volume setting.
FULL_VOLUME: float = 0.8
# A stray 0 must not silently mute the collision warning.
MIN_VOLUME_PCT: int = 10
MAX_VOLUME_PCT: int = 100

# Length of the settings test cue: about a warning plus the brake that follows it.
TEST_CUE_S: float = 2.5

# Edge ramps on every tone, so a cut never clicks.
_ATTACK_S: float = 0.004
_RELEASE_S: float = 0.006


@dataclass(frozen=True)
class Tone:
    """One beep: a fundamental plus partials, then silence."""

    freq_hz: float
    dur_s: float
    gap_s: float
    # (frequency ratio, amplitude relative to the fundamental)
    partials: tuple[tuple[float, float], ...] = ()
    # Exponential decay rate in 1/s; 0 holds the level flat.
    decay: float = 0.0
    attack_s: float = _ATTACK_S
    # Extra level at the onset, as a multiple of the held level, dying over strike_tau_s.
    strike: float = 0.0
    strike_tau_s: float = 0.004
    # Level of the whole tone against the others in its sound (ringing sounds only).
    level: float = 1.0
    # Ringing sounds: after dur_s the tone fades with this time constant, running on
    # over the gap and into the next tone instead of stopping.
    release_tau_s: float = 0.0

    @property
    def level_sum(self) -> float:
        return 1.0 + sum(a for _, a in self.partials)


@dataclass(frozen=True)
class WarningSound:
    """One cycle of a warning and the rules for repeating it.

    The cycle loops while the cue holds. ``min_cycles`` always play, however short
    the cue, and ``stop_extra_cycles`` play after it ends.
    """

    label: str
    tones: tuple[Tone, ...] = ()
    # Silence closing the cycle, before the next one starts.
    tail_s: float = 0.0
    # The next cycle starts this early. Only the original recording needs it.
    overlap_s: float = 0.0
    min_cycles: int = 1
    stop_extra_cycles: int = 0
    # Level, matched to the original's loudness (README section 17).
    gain: float = 0.35
    # Ringing after each tone, per frequency: (frequency Hz, decay time constant s,
    # phase rad against the tone), at ring_level of the tone's own level.
    rings: tuple[tuple[float, float, float], ...] = ()
    ring_level: float = 0.0
    # Sound once per AEB event instead of looping; a new event needs the cue clear
    # for rearm_s first, so a flickering cue inside one event stays a single warning.
    one_shot: bool = False
    rearm_s: float = 0.0

    @property
    def synthesized(self) -> bool:
        return bool(self.tones)

    @property
    def ringing(self) -> bool:
        return bool(self.rings) or any(t.release_tau_s > 0.0 for t in self.tones)

    @property
    def cycle_s(self) -> float:
        return sum(t.dur_s + t.gap_s for t in self.tones) + self.tail_s


# The recording shipped since 1.0: seamless loop, one extra pass after the cue.
ORIGINAL = WarningSound("Original", overlap_s=0.15, stop_extra_cycles=1)

# A piezo buzzer: steady high beeps while the cue holds, silent after the beep in flight.
SIMPLE = WarningSound(
    "Simple",
    tones=(Tone(2800.0, 0.10, 0.10, partials=((3.0, 0.12),)),),
    gain=0.493,
)

# Two groups of three struck pips ringing out of phase, the third accented, once per AEB
# event. Fitted to a reference clip's per-tone envelopes, see README section 17.
def _volvo_pip(level: float, octave: float, harmonic: float, gap_s: float) -> Tone:
    return Tone(1568.0, 0.0736, gap_s, partials=((0.5, octave), (2.0, harmonic)),
                attack_s=0.0022, strike=5.0, strike_tau_s=0.0024, level=level,
                release_tau_s=0.0236)


_VOLVO_TONES = (
    _volvo_pip(1.000, 0.306, 0.074, 0.0304),
    _volvo_pip(1.000, 0.306, 0.074, 0.0364),
    _volvo_pip(0.905, 0.529, 0.006, 0.0864),
    _volvo_pip(1.000, 0.306, 0.074, 0.0324),
    _volvo_pip(1.000, 0.306, 0.074, 0.0354),
    _volvo_pip(0.905, 0.529, 0.006, 0.0864),
)
VOLVO = WarningSound(
    "Volvo style", tones=_VOLVO_TONES, tail_s=0.7510, gain=0.95,
    rings=((784.0, 0.080, 1.99), (1568.0, 0.102, 3.17), (3136.0, 0.014, 3.16)),
    ring_level=0.295, one_shot=True, rearm_s=1.0,
)

# Five flat beeps, 1100 Hz beating against 1165 Hz, one burst a second; a burst
# always completes. Timing and pitch measured off a reference clip, not copied from it.
_TESLA_BEEP = Tone(1100.0, 0.090, 0.055, partials=((1165.0 / 1100.0, 0.25),))
TESLA = WarningSound("Tesla style", tones=(_TESLA_BEEP,) * 5, tail_s=0.275, gain=0.646)

SOUNDS: dict[str, WarningSound] = {s.label: s for s in (ORIGINAL, SIMPLE, VOLVO, TESLA)}
SOUND_LABELS: tuple[str, ...] = tuple(SOUNDS)
DEFAULT_SOUND: str = ORIGINAL.label


def resolve(label: object) -> WarningSound:
    """The named sound, or the original for anything unknown."""
    return SOUNDS.get(str(label), ORIGINAL)


def clamp_volume_pct(value: object) -> int:
    """A stored or typed volume as a whole percent inside the allowed range."""
    try:
        pct = int(round(float(value)))
    except (TypeError, ValueError):
        return DEFAULT_VOLUME_PCT
    return max(MIN_VOLUME_PCT, min(MAX_VOLUME_PCT, pct))


def current_choice() -> tuple[WarningSound, float]:
    """The sound and the volume fraction the settings ask for. Never raises."""
    try:
        from core.settings import Settings

        sound = resolve(getattr(Settings, "aeb_sound", DEFAULT_SOUND))
        pct = clamp_volume_pct(getattr(Settings, "aeb_sound_volume", DEFAULT_VOLUME_PCT))
    except Exception:
        logger.debug("could not read the AEB sound settings", exc_info=True)
        return ORIGINAL, FULL_VOLUME * DEFAULT_VOLUME_PCT / 100.0
    return sound, FULL_VOLUME * pct / 100.0


def _tone_samples(tone: Tone, rate: int, gain: float, norm: float) -> list[float]:
    n = int(round(tone.dur_s * rate))
    attack = max(1, int(tone.attack_s * rate))
    release = max(1, int(_RELEASE_S * rate))
    partials = ((1.0, 1.0),) + tone.partials
    out = []
    for i in range(n):
        t = i / rate
        env = min(1.0, i / attack, (n - i) / release)
        if tone.decay:
            env *= math.exp(-tone.decay * t)
        if tone.strike:
            env *= 1.0 + tone.strike * math.exp(-t / tone.strike_tau_s)
        s = sum(a * math.sin(2.0 * math.pi * tone.freq_hz * r * t) for r, a in partials)
        out.append(gain * env * s / norm)
    out.extend([0.0] * int(round(tone.gap_s * rate)))
    return out


def _ringing_shape(tone: Tone, rate: int) -> list[float]:
    """A ringing tone's level from its onset: rise, strike, hold, then exponential fade."""
    release = tone.release_tau_s
    length = int((tone.dur_s + (8.0 * release if release > 0.0 else _RELEASE_S)) * rate)
    attack = max(1.0, tone.attack_s * rate)
    shape = []
    for j in range(length):
        t = j / rate
        v = min(1.0, j / attack)
        if tone.strike:
            v *= 1.0 + tone.strike * math.exp(-t / tone.strike_tau_s)
        if tone.decay:
            v *= math.exp(-tone.decay * t)
        if t > tone.dur_s:
            past = t - tone.dur_s
            v *= math.exp(-past / release) if release > 0.0 else max(0.0, 1.0 - past / _RELEASE_S)
        shape.append(v)
    return shape


def _render_ringing(sound: WarningSound, rate: int) -> list[float]:
    """Overlapping tones on shared oscillators, plus the room ringing after them.

    A looping sound is built steady: tails wrap past the cycle end and the ring runs
    a warm-up cycle, so each cycle carries the previous one's ring. A one-shot starts
    from silence and needs a tail_s long enough for its ring to die out.
    """
    wrap = not sound.one_shot
    n = int(round(sound.cycle_s * rate))
    levels: dict[float, list[float]] = {}
    onset = 0.0
    for tone in sound.tones:
        shape = _ringing_shape(tone, rate)
        i0 = int(round(onset * rate))
        for ratio, amp in ((1.0, 1.0),) + tone.partials:
            env = levels.setdefault(round(tone.freq_hz * ratio, 3), [0.0] * n)
            a = tone.level * amp
            for j, v in enumerate(shape):
                if wrap or i0 + j < n:
                    env[(i0 + j) % n] += a * v
        onset += tone.dur_s + tone.gap_s
    rings = {round(f, 3): (tau, phase) for f, tau, phase in sound.rings}
    out = [0.0] * n
    for freq, env in levels.items():
        w = 2.0 * math.pi * freq / rate
        ring = rings.get(freq)
        if ring is None:
            for i, v in enumerate(env):
                out[i] += v * math.sin(w * i)
            continue
        tau, phase = ring
        k = 1.0 - math.exp(-1.0 / (rate * tau))
        y = 0.0
        for v in env if wrap else ():
            y += k * (v - y)
        for i, v in enumerate(env):
            y += k * (v - y)
            out[i] += v * math.sin(w * i) + sound.ring_level * y * math.sin(w * i + phase)
    peak = max(abs(v) for v in out) or 1.0
    return [v * sound.gain / peak for v in out]



@lru_cache(maxsize=16)
def render(sound: WarningSound, rate: int = 44100, channels: int = 2) -> bytes:
    """One cycle as signed 16-bit interleaved PCM, for ``pygame.mixer.Sound(buffer=...)``."""
    if not sound.synthesized:
        raise ValueError(f"{sound.label} is a recording, not a synthesized sound")
    if sound.ringing:
        mono = _render_ringing(sound, rate)
    else:
        # One scale for the whole cycle, so a tone with louder partials keeps its accent.
        norm = max(t.level_sum for t in sound.tones)
        mono = []
        for tone in sound.tones:
            mono.extend(_tone_samples(tone, rate, sound.gain, norm))
        mono.extend([0.0] * int(round(sound.tail_s * rate)))
    pcm = array("h")
    for v in mono:
        q = int(max(-1.0, min(1.0, v)) * 32767)
        pcm.extend([q] * max(1, channels))
    return pcm.tobytes()


def load(sound: WarningSound, mixer_init: tuple[int, int, int]):
    """A ``pygame.mixer.Sound`` for ``sound`` on a mixer opened as ``mixer_init``."""
    import pygame

    if not sound.synthesized:
        return pygame.mixer.Sound(str(ORIGINAL_PATH))
    rate, fmt, channels = mixer_init
    if fmt != -16:
        logger.warning("AEB sound: mixer format %s unsupported, using the original", fmt)
        return pygame.mixer.Sound(str(ORIGINAL_PATH))
    return pygame.mixer.Sound(buffer=render(sound, rate, channels))
