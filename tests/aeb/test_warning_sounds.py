"""AEB warning styles: synthesis, settings, and each style's repeat rules."""
from __future__ import annotations

import threading
from array import array

import pytest

from core.aeb import warning_sounds as ws
from core.aeb.warning_player import SoundState, WarningPlayer


@pytest.mark.parametrize("sound", [s for s in ws.SOUNDS.values() if s.synthesized],
                         ids=lambda s: s.label)
def test_a_rendered_cycle_is_exactly_one_cycle_long(sound):
    pcm = array("h", ws.render(sound, rate=44100, channels=2))
    assert len(pcm) / 2 == pytest.approx(sound.cycle_s * 44100, abs=len(sound.tones) + 1)


@pytest.mark.parametrize("sound", [s for s in ws.SOUNDS.values() if s.synthesized],
                         ids=lambda s: s.label)
def test_every_sound_ends_quiet_where_it_can_be_cut(sound):
    """A loop's soft stop lands halfway into its closing gap; a one-shot ends at its buffer end."""
    pcm = array("h", ws.render(sound, rate=44100, channels=1))
    peak = max(abs(v) for v in pcm)
    assert peak > 3000
    if sound.one_shot:
        assert max(abs(v) for v in pcm[-441:]) <= peak // 300
        return
    closing = int((sound.tones[-1].gap_s + sound.tail_s) * 44100)
    assert closing > 0
    assert max(abs(v) for v in pcm[-closing // 2:]) == 0


def test_a_ringing_cycle_loops_without_a_phase_jump():
    """Each ringing frequency fits a whole number of periods into one cycle."""
    for freq, _tau, _phase in ws.VOLVO.rings:
        periods = freq * ws.VOLVO.cycle_s
        assert periods == pytest.approx(round(periods), abs=1e-9)


def test_unknown_labels_fall_back_to_the_original():
    assert ws.resolve("Klaxon") is ws.ORIGINAL
    assert ws.resolve(None) is ws.ORIGINAL
    assert ws.ORIGINAL.label == "Original"


@pytest.mark.parametrize("raw, pct", [(80, 80), (0, ws.MIN_VOLUME_PCT), (250, 100),
                                      ("55", 55), ("loud", ws.DEFAULT_VOLUME_PCT)])
def test_volume_is_clamped_and_never_mutes(raw, pct):
    assert ws.clamp_volume_pct(raw) == pct


def test_full_volume_is_the_level_the_original_always_played_at():
    from core.settings import Settings

    Settings.save({"aeb_sound": ws.ORIGINAL.label, "aeb_sound_volume": 100})
    try:
        assert ws.current_choice() == (ws.ORIGINAL, pytest.approx(0.8))
    finally:
        Settings.save({"aeb_sound": ws.DEFAULT_SOUND, "aeb_sound_volume": ws.DEFAULT_VOLUME_PCT})


def test_the_choice_follows_the_settings():
    from core.settings import Settings

    Settings.save({"aeb_sound": ws.TESLA.label, "aeb_sound_volume": 40})
    try:
        assert ws.current_choice() == (ws.TESLA, pytest.approx(0.32))
    finally:
        Settings.save({"aeb_sound": ws.DEFAULT_SOUND, "aeb_sound_volume": ws.DEFAULT_VOLUME_PCT})


def test_each_style_repeats_like_the_system_it_imitates():
    # Original keeps its shipped tail; a buzzer stops after the beep in flight;
    # a Volvo-style warning (two groups of three) and a Tesla-style burst complete.
    assert (ws.ORIGINAL.min_cycles, ws.ORIGINAL.stop_extra_cycles) == (1, 1)
    assert (ws.SIMPLE.min_cycles, ws.SIMPLE.stop_extra_cycles) == (1, 0)
    assert (ws.VOLVO.min_cycles, ws.VOLVO.stop_extra_cycles) == (1, 0)
    assert (ws.TESLA.min_cycles, ws.TESLA.stop_extra_cycles) == (1, 0)
    assert len(ws.VOLVO.tones) == 6 and len(ws.TESLA.tones) == 5
    assert ws.VOLVO.ringing and not ws.TESLA.ringing and not ws.SIMPLE.ringing
    assert ws.TESLA.cycle_s == pytest.approx(1.0)


@pytest.mark.parametrize("sound", [s for s in ws.SOUNDS.values() if s.synthesized],
                         ids=lambda s: s.label)
def test_no_style_clips(sound):
    """A limited Volvo strike was heard as clipping; no style may get near full scale."""
    pcm = array("h", ws.render(sound, rate=48000, channels=1))
    assert max(abs(v) for v in pcm) < 32000


def test_repeated_cycles_never_run_together():
    """A gap must separate one cycle from the next, or repeats sound mashed together."""
    for sound in ws.SOUNDS.values():
        if sound.synthesized:
            assert sound.tones[-1].gap_s + sound.tail_s >= 0.04, sound.label


def _handler(style: ws.WarningSound, cycles_played: int) -> WarningPlayer:
    h = object.__new__(WarningPlayer)
    h._sound = object()
    h._style = style
    h._state = SoundState.RUNNING
    h._lock = threading.Lock()
    h._stop_extra_replays = style.stop_extra_cycles
    h._min_cycles = style.min_cycles
    h._cycles_played = cycles_played
    h._replays_remaining = 0
    h._cue_active = True
    h._cleared_at = float("-inf")
    h._one_shot = None
    return h


def test_a_short_cue_still_gets_the_style_minimum():
    style = ws.WarningSound("two", tones=ws.SIMPLE.tones, min_cycles=2)
    h = _handler(style, cycles_played=1)
    h.stop_warning()
    assert h._state == SoundState.SHUTTING_DOWN
    assert h._replays_remaining == 1


def test_a_long_cue_ends_with_the_cycle_in_flight():
    h = _handler(ws.TESLA, cycles_played=3)
    h.stop_warning()
    assert h._replays_remaining == 0


class _FakePlayer:
    """Records the cue a WarningTest drives, standing in for the mixer."""

    made = 0

    def __init__(self):
        type(self).made += 1
        self.events: list[str] = []
        self.sounding = False

    def start_warning(self):
        self.events.append("start")
        self.sounding = True

    def stop_warning(self):
        self.events.append("stop")
        self.sounding = False

    def busy(self):
        return self.sounding


def test_the_settings_test_holds_one_cue_then_soft_stops():
    import time
    from core.aeb.warning_player import WarningTest

    _FakePlayer.made = 0
    test = WarningTest(cue_s=0.05, player_factory=_FakePlayer)
    test.trigger()
    player = test._player
    time.sleep(0.2)
    assert player.events == ["start", "stop"]


def test_a_second_press_while_sounding_holds_the_same_warning_longer():
    import time
    from core.aeb.warning_player import WarningTest

    _FakePlayer.made = 0
    test = WarningTest(cue_s=0.15, player_factory=_FakePlayer)
    test.trigger()
    time.sleep(0.1)
    test.trigger()
    time.sleep(0.1)
    assert test._player.events == ["start", "start"]
    time.sleep(0.15)
    assert test._player.events == ["start", "start", "stop"]
    assert _FakePlayer.made == 1


def test_a_press_after_the_warning_ended_starts_a_fresh_player():
    import time
    from core.aeb.warning_player import WarningTest

    _FakePlayer.made = 0
    test = WarningTest(cue_s=0.02, player_factory=_FakePlayer)
    test.trigger()
    time.sleep(0.1)
    test.trigger()
    assert _FakePlayer.made == 2


class _Channel:
    def __init__(self):
        self.playing = True
        self.stopped = False

    def get_busy(self):
        return self.playing

    def stop(self):
        self.playing = False
        self.stopped = True


class _Sound:
    def __init__(self):
        self.channels: list[_Channel] = []

    def play(self, loops=0):
        ch = _Channel()
        self.channels.append(ch)
        return ch

    def set_volume(self, _v):
        pass

    def stop(self):
        for ch in self.channels:
            ch.stop()


def _one_shot_player(monkeypatch) -> tuple[WarningPlayer, _Sound]:
    from core.aeb import warning_player

    monkeypatch.setattr(warning_player.warning_sounds, "current_choice",
                        lambda: (ws.VOLVO, 0.8))
    sound = _Sound()
    h = _handler(ws.VOLVO, cycles_played=0)
    h._state = SoundState.STOPPED
    h._sound_thread = None
    h._sound = sound
    h._loaded = {ws.VOLVO.label: sound}
    h._cue_active = False
    return h, sound


def _cue(h: WarningPlayer, ticks: int) -> None:
    for _ in range(ticks):
        h.start_warning()


def test_a_one_shot_style_sounds_once_however_long_the_cue_holds(monkeypatch):
    h, sound = _one_shot_player(monkeypatch)
    _cue(h, 90)
    sound.channels[0].playing = False
    _cue(h, 90)
    assert len(sound.channels) == 1


def test_a_cue_flickering_inside_one_event_does_not_sound_again(monkeypatch):
    h, sound = _one_shot_player(monkeypatch)
    _cue(h, 5)
    sound.channels[0].playing = False
    h.stop_warning()
    _cue(h, 5)
    assert len(sound.channels) == 1
    assert h.busy()


def test_a_new_event_after_the_rearm_window_sounds_again(monkeypatch):
    h, sound = _one_shot_player(monkeypatch)
    _cue(h, 5)
    sound.channels[0].playing = False
    h.stop_warning()
    h._cleared_at -= ws.VOLVO.rearm_s + 0.01
    assert not h.busy()
    _cue(h, 5)
    assert len(sound.channels) == 2


def test_a_soft_stop_lets_the_one_warning_play_out_and_a_hard_stop_cuts_it(monkeypatch):
    h, sound = _one_shot_player(monkeypatch)
    _cue(h, 3)
    h.stop_warning()
    assert not sound.channels[0].stopped
    h.stop_warning(hard=True)
    assert sound.channels[0].stopped


def test_looping_styles_are_not_one_shot():
    assert ws.VOLVO.one_shot
    assert not any(s.one_shot for s in (ws.ORIGINAL, ws.SIMPLE, ws.TESLA))


# K-weighting (ITU-R BS.1770) at 48 kHz: the loudness the ear hears, not the peak.
_K_STAGES = (
    ((1.53512485958697, -2.69169618940638, 1.19839281085285),
     (1.0, -1.69065929318241, 0.73248077421585)),
    ((1.0, -2.0, 1.0), (1.0, -1.99004745483398, 0.99007225036621)),
)


def _momentary_max_db(samples: list[float], rate: int = 48000) -> float:
    """Loudest 400 ms of K-weighted signal, in dB: how loud a short warning sounds."""
    import math

    y = samples
    for (b0, b1, b2), (_a0, a1, a2) in _K_STAGES:
        out, x1, x2, y1, y2 = [], 0.0, 0.0, 0.0, 0.0
        for x in y:
            v = b0 * x + b1 * x1 + b2 * x2 - a1 * y1 - a2 * y2
            out.append(v)
            x2, x1, y2, y1 = x1, x, y1, v
        y = out
    win = int(0.4 * rate)
    acc = sum(v * v for v in y[:win])
    best = acc
    for i in range(win, len(y)):
        acc += y[i] * y[i] - y[i - win] * y[i - win]
        best = max(best, acc)
    return 10.0 * math.log10(best / win + 1e-12)


def _played(sound: ws.WarningSound, seconds: float = 2.0) -> list[float]:
    """What a held cue sounds like for ``seconds``, mono at 48 kHz."""
    import wave

    rate = 48000
    n = int(seconds * rate)
    if sound.synthesized:
        cycle = [v / 32768.0 for v in array("h", ws.render(sound, rate=rate, channels=1))]
        if sound.one_shot:
            return (cycle + [0.0] * n)[:n]
        return (cycle * (int(seconds / sound.cycle_s) + 1))[:n]
    with wave.open(str(ws.ORIGINAL_PATH)) as w:
        assert w.getframerate() == rate and w.getsampwidth() == 3
        raw = w.readframes(w.getnframes())
        ch = w.getnchannels()
    frame = 3 * ch
    rec = [sum(int.from_bytes(raw[i + 3 * c:i + 3 * c + 3], "little", signed=True)
               for c in range(ch)) / (ch * 8388608.0) for i in range(0, len(raw), frame)]
    period = len(rec) - int(sound.overlap_s * rate)
    out = [0.0] * (n + len(rec))
    for start in range(0, n, period):
        for j, v in enumerate(rec):
            out[start + j] += v
    return out[:n]


# Volvo sits a little under the rest by request: matching it clipped its strike.
_LOUDNESS_OFFSET_DB = {ws.VOLVO.label: -4.2}


def test_every_style_sounds_as_loud_as_the_original():
    """Same volume setting, same perceived level as the original recording, within 1 dB."""
    reference = _momentary_max_db(_played(ws.ORIGINAL))
    for sound in ws.SOUNDS.values():
        if sound.synthesized:
            want = reference + _LOUDNESS_OFFSET_DB.get(sound.label, 0.0)
            assert _momentary_max_db(_played(sound)) == pytest.approx(want, abs=1.0), sound.label
