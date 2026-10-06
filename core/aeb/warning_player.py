"""Plays AEB warning styles the way a real warning repeats, for AEB and for the settings test.

The AEB thread owns one ``WarningPlayer``; the settings panel's test button drives
its own through ``WarningTest``. See ``core/aeb/README.md`` section 17.
"""

from __future__ import annotations

import enum
import logging
import threading
import time

from core.aeb import warning_sounds

logger = logging.getLogger(__name__)

# Fade at a soft stop. Shortened to fit a closing gap smaller than this.
_STOP_FADE_MS: int = 30
# Kept between the fade and the next cycle, so the fade cannot catch its first note.
_STOP_MARGIN_S: float = 0.004


def _silence_sound(mixer_init: tuple[int, int, int]):
    """A few milliseconds of silence, queued in place of a bar that must not play."""
    import pygame

    rate, _fmt, channels = mixer_init
    n = max(1, int(rate * 0.01))
    return pygame.mixer.Sound(buffer=b"\x00" * (n * max(1, channels) * 2))


def _next_bar_kind(braking: bool, playing: int, stop_cycles: int | None) -> str:
    """What to queue after bar ``playing``: ``plain``, ``brake``, or ``silence``."""
    if stop_cycles is not None and playing + 1 > stop_cycles:
        return "silence"
    return "brake" if braking else "plain"


def _stop_fade_ms(cycle_index: int, stop_cycles: int, phase: float,
                  cycle: float, closing: float) -> int | None:
    """Fade length once the last cycle is in its closing silence, else None.

    Ends before the next cycle, so a stop cannot catch that cycle's first note.
    """
    if cycle_index < stop_cycles:
        return None
    remain = cycle - phase
    if cycle_index == stop_cycles and remain > closing:
        return None
    if cycle_index > stop_cycles:
        return 5
    fade_s = min(_STOP_FADE_MS / 1000.0, max(0.001, remain - _STOP_MARGIN_S))
    return max(1, int(round(fade_s * 1000)))


try:
    import pygame
    _PYGAME_AVAILABLE = True
except ImportError:
    _PYGAME_AVAILABLE = False


class SoundState(enum.IntEnum):
    STOPPED = 0
    RUNNING = 1
    SHUTTING_DOWN = 2


class WarningPlayer:
    """Pygame AEB warning loop with non-blocking stop and per-style repeat rules.

    The style and volume are read from settings whenever a warning starts from
    silence, so a change in the settings panel applies to the next warning.
    """

    def __init__(self) -> None:
        self._sound = None
        self._style: warning_sounds.WarningSound = warning_sounds.ORIGINAL
        self._state = SoundState.STOPPED
        self._sound_thread: threading.Thread | None = None
        self._lock = threading.Lock()
        self._stop_extra_replays = self._style.stop_extra_cycles
        self._min_cycles = self._style.min_cycles
        self._cycles_played = 0
        self._replays_remaining = 0
        self._loaded: dict[str, object] = {}
        # Brake mixed onto the foundation, so the two share one timeline.
        self._mixed_loaded: dict[str, object] = {}
        self._mixed_sound = None
        self._brake_sound = None
        self._silence = None
        self._channel = None
        self._braking = False
        self._brake_audible = False
        # Cue edges, for styles that sound once per AEB event (README section 17).
        self._cue_active = False
        self._cleared_at = float("-inf")
        self._one_shot = None

        if not _PYGAME_AVAILABLE:
            logger.warning("pygame not available: AEB sound disabled")
            return

        try:
            if not pygame.mixer.get_init():
                pygame.mixer.pre_init(frequency=44100, size=-16, channels=2, buffer=256)
                pygame.mixer.init()
            # Built up front: synthesis takes up to ~70 ms, too long for an AEB tick.
            init = pygame.mixer.get_init()
            for style in warning_sounds.SOUNDS.values():
                self._loaded[style.label] = warning_sounds.load(style, init)
                if style.layered:
                    self._mixed_loaded[style.label] = warning_sounds.load_mixed(style, init)
            self._silence = _silence_sound(init)
        except Exception as exc:
            logger.error("AEB sound init failed (%s): sound disabled", exc)
            self._loaded = {}
            self._mixed_loaded = {}
            return
        self._apply_choice()

    def _apply_choice(self) -> None:
        """Take the style and volume from settings. Only while no loop is running."""
        style, volume = warning_sounds.current_choice()
        sound = self._loaded.get(style.label) or self._loaded.get(warning_sounds.ORIGINAL.label)
        if sound is None:
            return
        mixed = getattr(self, "_mixed_loaded", {}).get(style.label)
        try:
            sound.set_volume(volume)
            if mixed is not None:
                mixed.set_volume(volume)
        except Exception:
            logger.debug("AEB sound: set_volume failed", exc_info=True)
        self._sound = sound
        self._mixed_sound = mixed
        self._brake_sound = None
        self._style = style
        self._stop_extra_replays = style.stop_extra_cycles
        self._min_cycles = style.min_cycles

    def start_warning(self, braking: bool = False) -> None:
        """Called every tick the cue holds. Loops, or sounds once per event for a one-shot style.

        ``braking`` brings in the brake layer of a layered style while AEB brakes.
        """
        if self._sound is None:
            return
        with self._lock:
            rising = not self._cue_active
            self._cue_active = True
            self._braking = braking
            if self._one_shot_playing():
                return
            if self._state == SoundState.RUNNING:
                return
            self._replays_remaining = 0
            if self._sound_thread is not None and self._sound_thread.is_alive():
                self._state = SoundState.RUNNING
                logger.debug("AEB sound: existing loop resumed")
                return
            if self._style.one_shot and not rising:
                return
            self._apply_choice()
            if self._style.one_shot:
                self._start_one_shot()
                return
            self._state = SoundState.RUNNING
            self._sound_thread = threading.Thread(
                target=self._sound_loop_manager, daemon=True
            )
            self._sound_thread.start()
            logger.debug("AEB sound: %s warning started", self._style.label)

    def _start_one_shot(self) -> None:
        """Sound the whole warning once, if the cue was clear for the re-arm window."""
        if time.monotonic() - self._cleared_at < self._style.rearm_s:
            return
        try:
            self._one_shot = self._sound.play()
        except Exception:
            logger.debug("AEB sound: one-shot play failed", exc_info=True)
            return
        logger.debug("AEB sound: %s warning played once", self._style.label)

    def _one_shot_playing(self) -> bool:
        channel = self._one_shot
        try:
            return channel is not None and channel.get_busy()
        except Exception:
            return False

    def stop_warning(self, *, hard: bool = False) -> None:
        """Soft: finish the cycle, then any style minimum or extra cycles. Hard: silence now.

        Called every tick the cue is clear. A one-shot warning always plays out.
        """
        if self._sound is None:
            return
        with self._lock:
            self._braking = False
            if self._cue_active:
                self._cue_active = False
                self._cleared_at = time.monotonic()
            if hard:
                if self._one_shot_playing():
                    self._one_shot.stop()
                self._one_shot = None
                if self._state == SoundState.STOPPED:
                    return
                self._state = SoundState.STOPPED
                self._replays_remaining = 0
                try:
                    self._sound.stop()
                    mixed = getattr(self, "_mixed_sound", None)
                    if mixed is not None:
                        mixed.stop()
                    channel = getattr(self, "_channel", None)
                    if channel is not None:
                        channel.stop()
                except Exception:
                    pass
                logger.debug("AEB sound: hard stop")
                return
            if self._state == SoundState.RUNNING:
                self._state = SoundState.SHUTTING_DOWN
                self._replays_remaining = max(
                    self._stop_extra_replays, self._min_cycles - self._cycles_played)
                logger.debug(
                    "AEB sound: stop requested: %d more cycle(s) then finishing",
                    self._replays_remaining,
                )

    def _play(self, loops: int = 0):
        self._cycles_played += 1
        return self._sound.play(loops=loops)

    def _sound_loop_manager(self) -> None:
        while True:
            with self._lock:
                self._cycles_played = 0
                seamless = self._style.synthesized
            if seamless:
                aborted = self._run_seamless()
            else:
                aborted = self._run_overlapped()
            with self._lock:
                # A cue that returned while the last cycle played out keeps sounding;
                # exiting here used to leave the state RUNNING with no loop behind it.
                if self._state == SoundState.RUNNING and not aborted:
                    continue
                self._state = SoundState.STOPPED
            logger.debug("AEB sound: finished playing, thread closing")
            return

    def _run_seamless(self) -> bool:
        """Loop a synthesized cycle in the mixer, stopping inside a cycle's closing silence.

        The mixer keeps the rhythm sample-accurate; a sleep-timed replay jitters by
        the Windows clock step. Returns True when a hard stop cut it short.
        """
        style = self._style
        if style.layered and self._mixed_sound is not None:
            return self._run_synced_bars()
        cycle = style.cycle_s
        closing = style.closing_s
        with self._lock:
            channel = self._play(loops=-1)
            brake_channel = self._start_brake_layer()
            t0 = time.monotonic()
        stop_cycles: int | None = None
        while True:
            time.sleep(0.01)
            with self._lock:
                if self._state == SoundState.STOPPED:
                    return True
                now = time.monotonic()
                elapsed = now - t0
                self._cycles_played = int(elapsed / cycle) + 1
                phase = elapsed % cycle
                self._gate_brake_layer(brake_channel, phase)
                if self._state == SoundState.RUNNING:
                    stop_cycles = None
                    continue
                if stop_cycles is None:
                    stop_cycles = self._cycles_played + self._replays_remaining
                fade_ms = _stop_fade_ms(
                    self._cycles_played, stop_cycles, phase, cycle, closing)
                if fade_ms is None:
                    continue
                for ch in (channel, brake_channel):
                    if ch is not None:
                        ch.fadeout(fade_ms)
                return False

    def _layer_sound(self, kind: str):
        if kind == "brake" and self._mixed_sound is not None:
            return self._mixed_sound
        if kind == "silence" and self._silence is not None:
            return self._silence
        return self._sound

    def _run_synced_bars(self) -> bool:
        """One channel, one bar at a time. The brake is mixed into that bar, not a second clip."""
        with self._lock:
            kind = _next_bar_kind(self._braking, 0, None)
            channel = self._layer_sound(kind).play(loops=0)
            if channel is None:
                return True
            self._channel = channel
            nxt = _next_bar_kind(self._braking, 1, None)
            channel.queue(self._layer_sound(nxt))
            self._cycles_played = 1
            queued = True
            queued_kind = nxt
            saw_queue = False
        stop_cycles: int | None = None
        while True:
            time.sleep(0.01)
            with self._lock:
                if self._state == SoundState.STOPPED:
                    channel.stop()
                    return True
                try:
                    pending = channel.get_queue()
                    busy = channel.get_busy()
                except Exception:
                    logger.debug("AEB sound: bar channel unreadable", exc_info=True)
                    return True
                if not busy:
                    return False
                if pending is not None:
                    saw_queue = True
                elif queued and saw_queue:
                    if queued_kind != "silence":
                        self._cycles_played += 1
                    queued = False
                    saw_queue = False
                if self._state == SoundState.RUNNING:
                    stop_cycles = None
                elif stop_cycles is None:
                    stop_cycles = self._cycles_played + self._replays_remaining
                kind = _next_bar_kind(self._braking, self._cycles_played, stop_cycles)
                # The closing sliver only replaces a bar already queued. Do not chain more.
                if kind == "silence" and not queued:
                    continue
                if not queued or kind != queued_kind:
                    try:
                        channel.queue(self._layer_sound(kind))
                        queued = True
                        queued_kind = kind
                        saw_queue = False
                    except Exception:
                        logger.debug("AEB sound: could not queue the next bar", exc_info=True)

    def _start_brake_layer(self):
        """Start the brake layer looping, silent, in step with the foundation just started."""
        self._brake_audible = False
        if self._brake_sound is None:
            return None
        try:
            channel = self._brake_sound.play(loops=-1)
            channel.set_volume(0.0)
            return channel
        except Exception:
            logger.debug("AEB sound: brake layer failed to start", exc_info=True)
            return None

    def _gate_brake_layer(self, channel, phase_s: float) -> None:
        """Switch the brake layer in or out, only where its bar is silent, so it never clicks."""
        if channel is None or self._braking == self._brake_audible:
            return
        lo, hi = self._style.brake_gate_s
        if lo <= phase_s <= hi:
            try:
                channel.set_volume(1.0 if self._braking else 0.0)
            except Exception:
                logger.debug("AEB sound: brake layer gate failed", exc_info=True)
            self._brake_audible = self._braking

    def _run_overlapped(self) -> bool:
        """Replay a recording ``overlap_s`` before it ends. Returns True on a hard stop."""
        slice_s = 0.02
        with self._lock:
            last_channel = self._play()
            period = max(0.0, self._sound.get_length() - self._style.overlap_s)
        while True:
            deadline = time.monotonic() + period
            while time.monotonic() < deadline:
                time.sleep(slice_s)
                with self._lock:
                    if self._state == SoundState.STOPPED:
                        return True
            with self._lock:
                if self._state == SoundState.RUNNING:
                    last_channel = self._play()
                elif (self._state == SoundState.SHUTTING_DOWN
                      and self._replays_remaining > 0):
                    self._replays_remaining -= 1
                    last_channel = self._play()
                elif self._state == SoundState.SHUTTING_DOWN:
                    logger.debug("AEB sound: last cycle playing out")
                    break
                else:
                    return True
        while last_channel is not None and last_channel.get_busy():
            time.sleep(0.01)
            with self._lock:
                if self._state == SoundState.STOPPED:
                    return True
        return False

    def busy(self) -> bool:
        """True while this warning event lasts: sounding, playing out, or not yet re-armed."""
        thread = self._sound_thread
        if thread is not None and thread.is_alive():
            return True
        with self._lock:
            if self._one_shot_playing():
                return True
            if not self._style.one_shot:
                return False
            return self._cue_active or time.monotonic() - self._cleared_at < self._style.rearm_s

    def cleanup(self) -> None:
        """Silence immediately, wait for the loop thread, then quit the mixer."""
        self.stop_warning(hard=True)
        if self._sound_thread and self._sound_thread.is_alive():
            self._sound_thread.join()
        if _PYGAME_AVAILABLE and pygame.mixer.get_init():
            pygame.mixer.quit()
        logger.debug("AEB sound: cleanup complete")


class WarningTest:
    """One simulated AEB cue of ``cue_s`` through a real ``WarningPlayer``.

    The settings button is one second of warning and does not play the brake layer.
    ``brake_after_s``, when set and shorter than the cue, still brings that layer in.
    A trigger while the warning still sounds holds the cue longer, exactly as a
    warning that returns during an AEB event; an idle trigger starts a fresh one.
    """

    def __init__(self, cue_s: float = warning_sounds.TEST_CUE_S, player_factory=None,
                 brake_after_s: float | None = None) -> None:
        self.cue_s = cue_s
        self.brake_after_s = brake_after_s
        self._factory = player_factory or WarningPlayer
        self._player = None
        self._braking = False
        self._timers: list[threading.Timer] = []
        self._lock = threading.Lock()

    def trigger(self) -> None:
        """Start or extend the test cue. Never raises."""
        try:
            with self._lock:
                if self._player is None or not self._player.busy():
                    # Fresh each time: AEB teardown can quit the mixer under an old one.
                    self._player = self._factory()
                    self._braking = False
                player = self._player
                player.start_warning(braking=self._braking)
                for timer in self._timers:
                    timer.cancel()
                self._timers = [threading.Timer(self.cue_s, self._end, args=(player,))]
                if (self.brake_after_s is not None and not self._braking
                        and self.brake_after_s < self.cue_s):
                    self._timers.append(
                        threading.Timer(self.brake_after_s, self._brake, args=(player,)))
                for timer in self._timers:
                    timer.daemon = True
                    timer.start()
        except Exception:
            logger.warning("AEB warning test could not play", exc_info=True)

    def _brake(self, player) -> None:
        with self._lock:
            self._braking = True
            player.start_warning(braking=True)

    def _end(self, player) -> None:
        with self._lock:
            self._braking = False
            player.stop_warning()
