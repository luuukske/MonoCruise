"""AEB warning sounds: the original recording plus five synthesized styles, and how each repeats.

The styles are synthesized here, never sampled from a real car, so no third-party
audio ships with MonoCruise. Pure Python apart from ``load`` and
``load_brake``; ``core/aeb/warning_player.py`` plays them. See ``core/aeb/README.md``
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

# Length of the settings test cue. Warning only: the brake layer is for a real AEB brake.
TEST_CUE_S: float = 1.0

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
class Partial:
    """One frequency of a layered sound: a note in each slot of the bar, each on its own envelope."""

    freq_hz: float
    # Level of the note in each slot in dB, or None for no note in that slot.
    slots_db: tuple[float | None, ...]
    # Onset after the slot starts, rise time, flat hold, then fade time constant.
    delay_s: float = 0.0
    attack_s: float = 0.001
    hold_s: float = 0.0
    tau_s: float = 0.02


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
    # Layered sounds (README section 17): ``brake_bar`` loops in step on top of ``bar``.
    # ``brake_gate_s`` is where the brake layer is silent, the only place it is switched.
    bar: tuple[Partial, ...] = ()
    brake_bar: tuple[Partial, ...] = ()
    bar_s: float = 0.0
    # Where each slot starts, from the bar start; empty spaces the slots evenly.
    slot_starts_s: tuple[float, ...] = ()
    # Where the last note of the bar has faded, from the bar start.
    sounding_s: float = 0.0
    brake_gate_s: tuple[float, float] = (0.0, 0.0)

    @property
    def synthesized(self) -> bool:
        return bool(self.tones) or bool(self.bar)

    @property
    def layered(self) -> bool:
        return bool(self.brake_bar)

    @property
    def ringing(self) -> bool:
        return bool(self.rings) or any(t.release_tau_s > 0.0 for t in self.tones)

    @property
    def cycle_s(self) -> float:
        if self.bar:
            return self.bar_s
        return sum(t.dur_s + t.gap_s for t in self.tones) + self.tail_s

    @property
    def closing_s(self) -> float:
        """Silence closing the cycle, where a soft stop lands."""
        if self.bar:
            return self.bar_s - self.sounding_s
        return self.tones[-1].gap_s + self.tail_s


# The recording shipped since 1.0: seamless loop, one extra pass after the cue.
ORIGINAL = WarningSound("Original", overlap_s=0.15, stop_extra_cycles=1)

# A piezo buzzer: steady high beeps while the cue holds, silent after the beep in flight.
SIMPLE = WarningSound(
    "Simple",
    tones=(Tone(2800.0, 0.10, 0.10, partials=((3.0, 0.12),)),),
    gain=0.493,
)

# Groups of three struck pips ringing out of phase, the third accented, once per AEB event;
# the last group is 20 to 25 dB down. Fitted to a reference clip, see README section 17.
def _cars_pip(level: float, octave: float, harmonic: float, gap_s: float) -> Tone:
    return Tone(1568.0, 0.0736, gap_s, partials=((0.5, octave), (2.0, harmonic)),
                attack_s=0.0022, strike=5.0, strike_tau_s=0.0024, level=level,
                release_tau_s=0.0236)


_CARS_TONES = (
    _cars_pip(1.000, 0.306, 0.074, 0.0304),
    _cars_pip(1.000, 0.306, 0.074, 0.0364),
    _cars_pip(0.905, 0.529, 0.006, 0.0864),
    _cars_pip(1.000, 0.306, 0.074, 0.0324),
    _cars_pip(1.000, 0.306, 0.074, 0.0354),
    _cars_pip(0.905, 0.529, 0.006, 0.0864),
    _cars_pip(0.055, 0.306, 0.074, 0.0324),
    _cars_pip(0.102, 0.306, 0.074, 0.0354),
    _cars_pip(0.057, 0.529, 0.006, 0.0864),
)
VOLVO_CARS = WarningSound(
    "Volvo Cars style", tones=_CARS_TONES, tail_s=0.376, gain=0.95,
    rings=((784.0, 0.080, 1.99), (1568.0, 0.102, 3.17), (3136.0, 0.014, 3.16)),
    ring_level=0.295, one_shot=True, rearm_s=1.0,
)

# Foundation and brake layers share one 0.5 s bar of six slots; see README section 17.
# Levels, delays and fades are least-squares fits to a reference's per-partial envelopes.
_NO_NOTE = (None,) * 2


def _truck_partial(freq: float, levels: tuple[float, ...], *, delay: float = 0.0,
                   attack: float = 0.0005, hold: float = 0.0, tau: float) -> Partial:
    return Partial(freq, levels + _NO_NOTE, delay, attack, hold, tau)


_TRUCK_FOUNDATION = (
    _truck_partial(443.0, (-28.8, -36.6, -36.0, -34.3), delay=0.0027, attack=0.0028,
                   hold=0.0718, tau=0.0358),
    _truck_partial(1317.0, (-48.6, -57.0, -55.6, -55.8), attack=0.0042, hold=0.0025,
                   tau=0.0626),
    _truck_partial(1744.0, (-39.0, -43.5, -43.1, -43.4), delay=0.0039, attack=0.0304,
                   tau=0.0192),
    _truck_partial(2195.0, (-48.2, -54.5, -55.7, -55.5), tau=0.0431),
    _truck_partial(2641.0, (-50.4, -57.2, -59.0, -56.7), tau=0.0580),
    _truck_partial(3083.0, (-45.0, -61.2, -61.5, -73.0), tau=0.0121),
    _truck_partial(3520.0, (-50.0, -57.5, -58.6, -58.5), tau=0.0269),
)
# Pings lead the slot by 12 ms: that is where the reference peak sits. The 443 Hz tone
# holds the pause. The recording's brake ticks are the truck, not this layer (README).
_TRUCK_BRAKE = (
    Partial(1743.7, (None, None, -38.5, -38.5, -38.5, -38.5), -0.012, 0.003, 0.024, 0.020),
    Partial(443.0, (None, None, None, -42.1, None, None), 0.0585, 0.0184, 0.14, 0.016),
)
VOLVO_TRUCKS = WarningSound(
    "Volvo Trucks style", bar=_TRUCK_FOUNDATION, brake_bar=_TRUCK_BRAKE,
    bar_s=0.50123, sounding_s=0.470, brake_gate_s=(0.012, 0.145), min_cycles=4, gain=0.8976,
)

# Four notes, high low low high, on an uneven grid: harmonics 1, 2, 4, 8 of 269.5 Hz.
# One pattern for warn and brake. Fitted to a reference, see README section 17.
_SCANIA_F0 = 269.5
_SCANIA_NOTES = (
    Partial(_SCANIA_F0, (-33.8, -32.9, -32.4, None), 0.005, 0.0035, 0.0568, 0.0336),
    Partial(_SCANIA_F0, (None, None, None, -34.1), 0.0089, 0.0035, 0.1051, 0.015),
    Partial(2 * _SCANIA_F0, (-28.5, -25.5, -25.5, None), 0.005, 0.0053, 0.0636, 0.0188),
    Partial(2 * _SCANIA_F0, (None, None, None, -30.7), 0.0199, 0.0073, 0.0917, 0.0285),
    Partial(4 * _SCANIA_F0, (-18.3, None, None, -22.8), 0.002, 0.0061, 0.0276, 0.0272),
    Partial(4 * _SCANIA_F0, (None, -30.2, -29.9, None), 0.002, 0.0056, 0.0683, 0.0185),
    Partial(8 * _SCANIA_F0, (-29.5, -32.0, -31.7, -31.8), 0.002, 0.0005, 0.0, 0.032),
)
SCANIA = WarningSound(
    "Scania style", bar=_SCANIA_NOTES, bar_s=0.4714, slot_starts_s=(0.0, 0.095, 0.200, 0.303),
    sounding_s=0.430, gain=0.97,
)

# Five flat beeps, 1100 Hz beating against 1165 Hz, one burst a second; a burst
# always completes. Timing and pitch measured off a reference clip, not copied from it.
_TESLA_BEEP = Tone(1100.0, 0.090, 0.055, partials=((1165.0 / 1100.0, 0.25),))
TESLA = WarningSound("Tesla style", tones=(_TESLA_BEEP,) * 5, tail_s=0.275, gain=0.646)

SOUNDS: dict[str, WarningSound] = {
    s.label: s for s in (ORIGINAL, SIMPLE, VOLVO_CARS, VOLVO_TRUCKS, SCANIA, TESLA)
}
SOUND_LABELS: tuple[str, ...] = tuple(SOUNDS)
DEFAULT_SOUND: str = ORIGINAL.label

# Labels a settings file may still carry from before the Trucks style was added.
_RENAMED = {"Volvo style": VOLVO_CARS.label}


def resolve(label: object) -> WarningSound:
    """The named sound, or the original for anything unknown."""
    name = str(label)
    return SOUNDS.get(_RENAMED.get(name, name), ORIGINAL)


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



def _note_shape(p: Partial, rate: int) -> tuple[int, list[float]]:
    """A note's level from its onset sample: rise, hold, then an exponential fade."""
    onset = int(round(p.delay_s * rate))
    rise = max(1.0, p.attack_s * rate)
    hold = p.hold_s * rate
    tau = p.tau_s * rate
    shape = []
    for j in range(int(rise + hold + 7.0 * tau)):
        v = min(1.0, j / rise)
        if j > rise + hold:
            v *= math.exp(-(j - rise - hold) / tau)
        shape.append(v)
    return onset, shape


def _render_layer(parts: tuple[Partial, ...], n: int, rate: int,
                  starts_s: tuple[float, ...] = ()) -> list[float]:
    """One layer of a bar. Notes wrap past the bar end and each frequency fits whole
    cycles into the bar, so the loop has neither a gap nor a phase jump."""
    out = [0.0] * n
    for p in parts:
        onset, shape = _note_shape(p, rate)
        env = [0.0] * n
        for slot, db in enumerate(p.slots_db):
            if db is None:
                continue
            amp = 10.0 ** (db / 20.0)
            if starts_s:
                start = int(round(starts_s[slot] * rate)) + onset
            else:
                start = int(round(slot * n / len(p.slots_db))) + onset
            for j, v in enumerate(shape):
                env[(start + j) % n] += amp * v
        w = 2.0 * math.pi * max(1, round(p.freq_hz * n / rate)) / n
        for i, e in enumerate(env):
            out[i] += e * math.sin(w * i)
    return out


@lru_cache(maxsize=8)
def _render_bar(sound: WarningSound, rate: int) -> tuple[list[float], list[float]]:
    """The foundation and brake layers, scaled so the foundation's peak is ``gain``.

    The brake layer uses that same scale, so braking does not change the foundation.
    """
    n = int(round(sound.bar_s * rate))
    base = _render_layer(sound.bar, n, rate, sound.slot_starts_s)
    brake = _render_layer(sound.brake_bar, n, rate, sound.slot_starts_s)
    k = sound.gain / (max(abs(v) for v in base) or 1.0)
    return [v * k for v in base], [v * k for v in brake]


def _pcm(mono: list[float], channels: int) -> bytes:
    pcm = array("h")
    for v in mono:
        q = int(max(-1.0, min(1.0, v)) * 32767)
        pcm.extend([q] * max(1, channels))
    return pcm.tobytes()


def render_brake(sound: WarningSound, rate: int = 44100, channels: int = 2) -> bytes:
    """The brake layer of a layered sound: one bar, on the same samples as ``render``."""
    if not sound.layered:
        raise ValueError(f"{sound.label} has no brake layer")
    return _pcm(_render_bar(sound, rate)[1], channels)


def render_with_brake(sound: WarningSound, rate: int = 44100, channels: int = 2) -> bytes:
    """One bar with the brake layer added on the foundation's own samples."""
    if not sound.layered:
        raise ValueError(f"{sound.label} has no brake layer")
    base, brake = _render_bar(sound, rate)
    return _pcm([a + b for a, b in zip(base, brake)], channels)


@lru_cache(maxsize=16)
def render(sound: WarningSound, rate: int = 44100, channels: int = 2) -> bytes:
    """One cycle as signed 16-bit interleaved PCM, for ``pygame.mixer.Sound(buffer=...)``."""
    if not sound.synthesized:
        raise ValueError(f"{sound.label} is a recording, not a synthesized sound")
    if sound.bar:
        return _pcm(_render_bar(sound, rate)[0], channels)
    if sound.ringing:
        mono = _render_ringing(sound, rate)
    else:
        # One scale for the whole cycle, so a tone with louder partials keeps its accent.
        norm = max(t.level_sum for t in sound.tones)
        mono = []
        for tone in sound.tones:
            mono.extend(_tone_samples(tone, rate, sound.gain, norm))
        mono.extend([0.0] * int(round(sound.tail_s * rate)))
    return _pcm(mono, channels)


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


def load_brake(sound: WarningSound, mixer_init: tuple[int, int, int]):
    """The brake layer of ``sound`` as a ``pygame.mixer.Sound``, or None without one."""
    import pygame

    rate, fmt, channels = mixer_init
    if not sound.layered or fmt != -16:
        return None
    return pygame.mixer.Sound(buffer=render_brake(sound, rate, channels))


def load_mixed(sound: WarningSound, mixer_init: tuple[int, int, int]):
    """Foundation plus brake in one buffer, or None when the mixer cannot play it."""
    import pygame

    rate, fmt, channels = mixer_init
    if not sound.layered or fmt != -16:
        return None
    return pygame.mixer.Sound(buffer=render_with_brake(sound, rate, channels))
