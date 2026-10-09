"""AEB warning styles: synthesis, settings, and each style's repeat rules."""
from __future__ import annotations

import math
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
    closing = int(sound.closing_s * 44100)
    assert closing > 0
    if sound.layered:
        # The brake layer's last ping is still fading at the loop point, far under the notes.
        assert max(abs(v) for v in pcm[-closing // 2:]) <= peak // 50
        return
    if sound is ws.SCANIA:
        # The last note's low tone is still dying out where a stop fades it: no note starts there.
        win = closing // 4
        levels = [max(abs(v) for v in pcm[-closing + k * win:][:win]) for k in range(4)]
        assert levels == sorted(levels, reverse=True) and levels[-1] <= peak // 12
        return
    assert max(abs(v) for v in pcm[-closing // 2:]) == 0


def test_a_ringing_cycle_loops_without_a_phase_jump():
    """Each ringing frequency fits a whole number of periods into one cycle."""
    for freq, _tau, _phase in ws.VOLVO_CARS.rings:
        periods = freq * ws.VOLVO_CARS.cycle_s
        assert periods == pytest.approx(round(periods), abs=1e-9)


def test_volvo_cars_ends_with_soft_retriggers():
    """The third group is 20 to 25 dB under the main pips, its last pip quieter than the one before.
    Drivers hear it as two soft retriggers, the last barely audible."""
    main = [t.level for t in ws.VOLVO_CARS.tones[:6]]
    soft = [t.level for t in ws.VOLVO_CARS.tones[6:]]
    assert len(soft) == 3 and max(soft) < 0.15 * min(main)
    assert soft[2] < soft[1]
    rate = 44100
    pcm = array("h", ws.render(ws.VOLVO_CARS, rate, 1))
    peak = max(abs(v) for v in pcm)
    onset = sum(t.dur_s + t.gap_s for t in ws.VOLVO_CARS.tones[:7])
    a = int(onset * rate)
    second = max(abs(v) for v in pcm[a:a + int(0.07 * rate)])
    assert peak / 20 < second < peak / 6


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
    # Every looping style plays one full extra cycle after the cue; Volvo Cars is a one-shot.
    assert (ws.ORIGINAL.min_cycles, ws.ORIGINAL.stop_extra_cycles) == (1, 1)
    assert (ws.SIMPLE.min_cycles, ws.SIMPLE.stop_extra_cycles) == (1, 1)
    assert (ws.VOLVO_CARS.min_cycles, ws.VOLVO_CARS.stop_extra_cycles) == (1, 0)
    assert (ws.TESLA.min_cycles, ws.TESLA.stop_extra_cycles) == (1, 1)
    assert len(ws.VOLVO_CARS.tones) == 9 and len(ws.TESLA.tones) == 5
    assert ws.VOLVO_CARS.ringing and not ws.TESLA.ringing and not ws.SIMPLE.ringing
    assert ws.TESLA.cycle_s == pytest.approx(1.0)
    # A truck warning sounds at least three bars, however short the cue.
    assert (ws.VOLVO_TRUCKS.min_cycles, ws.VOLVO_TRUCKS.stop_extra_cycles) == (3, 1)
    assert ws.VOLVO_TRUCKS.cycle_s == pytest.approx(0.50123)
    # Scania: one pattern for warn and brake.
    assert (ws.SCANIA.min_cycles, ws.SCANIA.stop_extra_cycles) == (1, 1)
    assert ws.SCANIA.cycle_s == pytest.approx(0.4714) and not ws.SCANIA.layered


@pytest.mark.parametrize("sound", [s for s in ws.SOUNDS.values() if s.synthesized],
                         ids=lambda s: s.label)
def test_no_style_clips(sound):
    """A limited Volvo strike was heard as clipping; no style may get near full scale."""
    pcm = array("h", ws.render(sound, rate=48000, channels=1))
    if sound.layered:
        brake = array("h", ws.render_brake(sound, rate=48000, channels=1))
        pcm = array("h", [a + b for a, b in zip(pcm, brake)])
    assert max(abs(v) for v in pcm) < 32000


def test_repeated_cycles_never_run_together():
    """A gap must separate one cycle from the next, or repeats sound mashed together."""
    for sound in ws.SOUNDS.values():
        if sound.synthesized:
            assert sound.closing_s >= (0.025 if sound.layered else 0.04), sound.label


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
    h._brake_sound = None
    h._brake_loaded = {}
    h._braking = False
    h._brake_audible = False
    return h


def test_a_short_cue_still_gets_the_style_minimum():
    style = ws.WarningSound("two", tones=ws.SIMPLE.tones, min_cycles=2)
    h = _handler(style, cycles_played=1)
    h.stop_warning()
    assert h._state == SoundState.SHUTTING_DOWN
    assert h._replays_remaining == 1


def test_a_long_cue_ends_with_one_extra_cycle_after_the_one_in_flight():
    h = _handler(ws.TESLA, cycles_played=3)
    h.stop_warning()
    assert h._replays_remaining == 1


def test_a_short_truck_cue_gets_three_bars_in_all():
    h = _handler(ws.VOLVO_TRUCKS, cycles_played=1)
    h.stop_warning()
    assert h._replays_remaining == 2


class _FakePlayer:
    """Records the cue a WarningTest drives, standing in for the mixer."""

    made = 0

    def __init__(self):
        type(self).made += 1
        self.events: list[str] = []
        self.sounding = False

    def start_warning(self, braking=False):
        self.events.append("brake" if braking else "start")
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
                        lambda: (ws.VOLVO_CARS, 0.8))
    sound = _Sound()
    h = _handler(ws.VOLVO_CARS, cycles_played=0)
    h._state = SoundState.STOPPED
    h._sound_thread = None
    h._sound = sound
    h._loaded = {ws.VOLVO_CARS.label: sound}
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
    h._cleared_at -= ws.VOLVO_CARS.rearm_s + 0.01
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
    assert ws.VOLVO_CARS.one_shot
    assert not any(s.one_shot for s in (ws.ORIGINAL, ws.SIMPLE, ws.VOLVO_TRUCKS, ws.SCANIA,
                                        ws.TESLA))


def test_a_settings_file_from_before_the_trucks_style_still_resolves():
    assert ws.resolve("Volvo style") is ws.VOLVO_CARS
    assert ws.resolve(ws.VOLVO_TRUCKS.label) is ws.VOLVO_TRUCKS
    assert ws.VOLVO_CARS.label != ws.VOLVO_TRUCKS.label


def _bar(sound: ws.WarningSound, *, brake: bool, rate: int = 48000) -> list[float]:
    pcm = ws.render_brake(sound, rate, 1) if brake else ws.render(sound, rate, 1)
    return [v / 32768.0 for v in array("h", pcm)]


def _slot_peaks(x: list[float], slots: int = 6) -> list[float]:
    n = len(x) // slots
    return [max(abs(v) for v in x[k * n:(k + 1) * n]) for k in range(slots)]


def test_the_truck_foundation_is_four_notes_then_a_pause():
    peaks = _slot_peaks(_bar(ws.VOLVO_TRUCKS, brake=False))
    assert all(p > 0.1 for p in peaks[:4])
    # The last note's tone rings into the first part of the pause, then nothing.
    assert peaks[4] < peaks[3] and peaks[5] < 0.1 * peaks[0]


def test_the_truck_brake_layer_fills_the_last_four_slots():
    peaks = _slot_peaks(_bar(ws.VOLVO_TRUCKS, brake=True))
    assert all(p > 0.04 for p in peaks[2:])


def test_the_brake_layer_is_mixed_onto_the_foundation_samples():
    """A second clip can start late; the brake has to be on the foundation's own samples."""
    from core.aeb.warning_player import _next_bar_kind

    rate = 48000
    base = array("h", ws.render(ws.VOLVO_TRUCKS, rate, 1))
    brake = array("h", ws.render_brake(ws.VOLVO_TRUCKS, rate, 1))
    mixed = array("h", ws.render_with_brake(ws.VOLVO_TRUCKS, rate, 1))
    assert all(abs(m - (a + b)) <= 1 for a, b, m in zip(base, brake, mixed))
    steps = [abs(b - a) for a, b in zip(base, base[1:])]
    limit = max(steps)
    assert abs(mixed[0] - base[-1]) <= limit
    assert abs(base[0] - mixed[-1]) <= limit
    assert abs(mixed[0] - mixed[-1]) <= limit
    assert _next_bar_kind(True, 2, None) == "brake"
    assert _next_bar_kind(False, 2, None) == "plain"
    assert _next_bar_kind(True, 4, 4) == "silence"


def test_a_truck_stop_fades_out_before_the_next_bar():
    """The pause is shorter than the usual fade, which used to catch the next beep."""
    from core.aeb.warning_player import _stop_fade_ms

    cycle, closing = ws.VOLVO_TRUCKS.cycle_s, ws.VOLVO_TRUCKS.closing_s
    assert _stop_fade_ms(3, 4, 0.20, cycle, closing) is None
    phase = cycle - closing
    while phase < cycle - 0.002:
        fade = _stop_fade_ms(4, 4, phase, cycle, closing)
        assert fade is not None and phase + fade / 1000.0 <= cycle - 0.002
        phase += 0.003


def test_the_brake_gate_lies_inside_the_brake_layers_silence():
    lo, hi = ws.VOLVO_TRUCKS.brake_gate_s
    x = _bar(ws.VOLVO_TRUCKS, brake=True)
    peak = max(abs(v) for v in x)
    window = x[int(lo * 48000):int(hi * 48000)]
    assert 0.0 < lo < hi and max(abs(v) for v in window) < 0.06 * peak


@pytest.mark.parametrize("sound, brake", [(ws.VOLVO_TRUCKS, False), (ws.VOLVO_TRUCKS, True),
                                          (ws.SCANIA, False)])
def test_a_truck_bar_loops_without_a_jump(sound, brake):
    """Every frequency fits whole cycles into the bar, so the wrap is as smooth as any sample."""
    x = _bar(sound, brake=brake)
    steps = [abs(b - a) for a, b in zip(x, x[1:])]
    assert abs(x[0] - x[-1]) <= max(steps)


def _tone_level(x: list[float], freq: float, start_s: float, dur_s: float,
                rate: int = 48000) -> float:
    a = int(start_s * rate)
    seg = x[a:a + int(dur_s * rate)]
    w = 2.0 * math.pi * freq / rate
    re = sum(v * math.cos(w * i) for i, v in enumerate(seg))
    im = sum(v * math.sin(w * i) for i, v in enumerate(seg))
    return math.hypot(re, im)


def test_the_scania_pattern_is_high_low_low_high():
    """Notes one and four lead with the high tone, two and three with the low one."""
    x = _bar(ws.SCANIA, brake=False)
    f0 = ws._SCANIA_F0
    for k, start in enumerate(ws.SCANIA.slot_starts_s):
        high = _tone_level(x, 4 * f0, start + 0.01, 0.06)
        low = _tone_level(x, 2 * f0, start + 0.01, 0.06)
        assert (high > 1.5 * low) if k in (0, 3) else (low > high), k


def test_scania_notes_start_on_its_uneven_grid():
    """Each note's strike lands on its own start, not on an even split of the bar."""
    x = _bar(ws.SCANIA, brake=False)
    lead = 0.02
    looped = x[-int(lead * 48000):] + x
    for start in ws.SCANIA.slot_starts_s:
        before = _tone_level(looped, 8 * ws._SCANIA_F0, start + lead - 0.015, 0.015)
        after = _tone_level(looped, 8 * ws._SCANIA_F0, start + lead, 0.015)
        assert after > 3.0 * before, start


def test_the_scania_tone_keeps_its_low_fundamental():
    """Every tone is a harmonic of one fundamental, and that fundamental sounds in every note.
    Without it the first version was heard as too high."""
    f0 = ws._SCANIA_F0
    assert all(p.freq_hz / f0 in (1, 2, 4, 8) for p in ws.SCANIA.bar)
    x = _bar(ws.SCANIA, brake=False)
    for start in ws.SCANIA.slot_starts_s:
        low = _tone_level(x, f0, start + 0.02, 0.05)
        assert low > 0.3 * _tone_level(x, 2 * f0, start + 0.02, 0.05), start


def test_the_layers_are_the_same_length_so_they_stay_in_step():
    assert len(_bar(ws.VOLVO_TRUCKS, brake=False)) == len(_bar(ws.VOLVO_TRUCKS, brake=True))
    assert ws.VOLVO_TRUCKS.layered and not ws.TESLA.layered
    with pytest.raises(ValueError):
        ws.render_brake(ws.TESLA)


class _GateChannel:
    def __init__(self):
        self.volumes: list[float] = []

    def set_volume(self, v):
        self.volumes.append(v)


def test_the_brake_layer_only_switches_inside_its_silent_stretch():
    h = _handler(ws.VOLVO_TRUCKS, cycles_played=1)
    channel = _GateChannel()
    lo, hi = ws.VOLVO_TRUCKS.brake_gate_s
    h._braking = True
    h._gate_brake_layer(channel, hi + 0.05)
    assert channel.volumes == [] and not h._brake_audible
    h._gate_brake_layer(channel, (lo + hi) / 2)
    assert channel.volumes == [1.0] and h._brake_audible
    h._braking = False
    h._gate_brake_layer(channel, hi + 0.05)
    assert channel.volumes == [1.0] and h._brake_audible
    h._gate_brake_layer(channel, lo + 0.001)
    assert channel.volumes == [1.0, 0.0] and not h._brake_audible


def test_the_braking_flag_follows_the_cue():
    h = _handler(ws.VOLVO_TRUCKS, cycles_played=1)
    h._sound = _Sound()
    h.start_warning(braking=True)
    assert h._braking
    h.start_warning(braking=False)
    assert not h._braking
    h.start_warning(braking=True)
    h.stop_warning()
    assert not h._braking


def test_the_settings_test_is_one_second_of_warning_and_does_not_brake():
    from core.aeb.warning_player import WarningTest

    _FakePlayer.made = 0
    test = WarningTest(player_factory=_FakePlayer)
    test.trigger()
    assert test.cue_s == ws.TEST_CUE_S == 1.0
    assert test.brake_after_s is None
    assert test._player.events == ["start"]
    assert len(test._timers) == 1
    for timer in test._timers:
        timer.cancel()


def test_the_settings_test_adds_the_brake_after_a_warning_then_ends_both():
    import time
    from core.aeb.warning_player import WarningTest

    _FakePlayer.made = 0
    test = WarningTest(cue_s=0.3, player_factory=_FakePlayer, brake_after_s=0.1)
    test.trigger()
    time.sleep(0.45)
    assert test._player.events == ["start", "brake", "stop"]


def test_a_test_cue_shorter_than_the_brake_delay_never_brakes():
    import time
    from core.aeb.warning_player import WarningTest

    _FakePlayer.made = 0
    test = WarningTest(cue_s=0.05, player_factory=_FakePlayer, brake_after_s=0.2)
    test.trigger()
    time.sleep(0.3)
    assert test._player.events == ["start", "stop"]


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


def _played(sound: ws.WarningSound, seconds: float = 2.0, braking: bool = False) -> list[float]:
    """What a held cue sounds like for ``seconds``, mono at 48 kHz."""
    import wave

    rate = 48000
    n = int(seconds * rate)
    if sound.synthesized:
        cycle = [v / 32768.0 for v in array("h", ws.render(sound, rate=rate, channels=1))]
        if braking:
            layer = array("h", ws.render_brake(sound, rate=rate, channels=1))
            cycle = [a + b / 32768.0 for a, b in zip(cycle, layer)]
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
_LOUDNESS_OFFSET_DB = {ws.VOLVO_CARS.label: -4.2}


def test_every_style_sounds_as_loud_as_the_original():
    """Same volume setting, same perceived level as the original recording, within 1 dB."""
    reference = _momentary_max_db(_played(ws.ORIGINAL))
    for sound in ws.SOUNDS.values():
        if sound.synthesized:
            want = reference + _LOUDNESS_OFFSET_DB.get(sound.label, 0.0)
            assert _momentary_max_db(_played(sound)) == pytest.approx(want, abs=1.0), sound.label
            if sound.layered:
                braking = _played(sound, braking=True)
                assert _momentary_max_db(braking) == pytest.approx(want, abs=1.0), sound.label
