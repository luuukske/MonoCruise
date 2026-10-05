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

# Fade at a soft stop; well inside every style's closing gap.
_STOP_FADE_MS: int = 30

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
        except Exception as exc:
            logger.error("AEB sound init failed (%s): sound disabled", exc)
            self._loaded = {}
            return
        self._apply_choice()

    def _apply_choice(self) -> None:
        """Take the style and volume from settings. Only while no loop is running."""
        style, volume = warning_sounds.current_choice()
        sound = self._loaded.get(style.label) or self._loaded.get(warning_sounds.ORIGINAL.label)
        if sound is None:
            return
        try:
            sound.set_volume(volume)
        except Exception:
            logger.debug("AEB sound: set_volume failed", exc_info=True)
        self._sound = sound
        self._style = style
        self._stop_extra_replays = style.stop_extra_cycles
        self._min_cycles = style.min_cycles

    def start_warning(self) -> None:
        """Called every tick the cue holds. Loops, or sounds once per event for a one-shot style."""
        if self._sound is None:
            return
        with self._lock:
            rising = not self._cue_active
            self._cue_active = True
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
        cycle = style.cycle_s
        closing = style.tones[-1].gap_s + style.tail_s
        with self._lock:
            channel = self._play(loops=-1)
            t0 = time.monotonic()
        stop_at: float | None = None
        while True:
            time.sleep(0.01)
            with self._lock:
                if self._state == SoundState.STOPPED:
                    return True
                now = time.monotonic()
                self._cycles_played = int((now - t0) / cycle) + 1
                if self._state == SoundState.RUNNING:
                    stop_at = None
                    continue
                if stop_at is None:
                    cycles = self._cycles_played + self._replays_remaining
                    stop_at = t0 + cycles * cycle - closing / 2.0
                if now >= stop_at:
                    # A fade, not a cut: a ringing style still carries a faint tail here.
                    if channel is not None:
                        channel.fadeout(_STOP_FADE_MS)
                    return False

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

    A trigger while the warning still sounds holds the cue longer, exactly as a
    warning that returns during an AEB event; an idle trigger starts a fresh one.
    """

    def __init__(self, cue_s: float = warning_sounds.TEST_CUE_S,
                 player_factory=None) -> None:
        self.cue_s = cue_s
        self._factory = player_factory or WarningPlayer
        self._player = None
        self._timer: threading.Timer | None = None
        self._lock = threading.Lock()

    def trigger(self) -> None:
        """Start or extend the test cue. Never raises."""
        try:
            with self._lock:
                if self._player is None or not self._player.busy():
                    # Fresh each time: AEB teardown can quit the mixer under an old one.
                    self._player = self._factory()
                player = self._player
                player.start_warning()
                if self._timer is not None:
                    self._timer.cancel()
                self._timer = threading.Timer(self.cue_s, player.stop_warning)
                self._timer.daemon = True
                self._timer.start()
        except Exception:
            logger.warning("AEB warning test could not play", exc_info=True)
