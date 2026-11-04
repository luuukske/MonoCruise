"""Pedal input must survive a pygame video init that does not raise.

pygame.init() counts a failed display and leaves get_init() true. event.get()
then raises "video system not initialized" and used to kill main_pedal_thread.
See core/main_pedal_thread/README.md. pygame is not installed in CI, so the
thread is checked from its source and the bring-up is driven with a fake.
"""

from __future__ import annotations

import ast
import logging
import threading
from pathlib import Path
from types import SimpleNamespace

from core.main_pedal_thread.pygame_video import (
    PygameVideo,
    ensure_pygame_video,
    publish_released,
)

REPO = Path(__file__).resolve().parents[1]
PEDAL_SRC = REPO / "core" / "main_pedal_thread" / "thread.py"


class _Display:
    def __init__(self, env: dict[str, str]) -> None:
        self.env = env
        self.up = False
        self.inits = 0
        self.driver = ""
        self.fail = False

    def get_init(self) -> bool:
        return self.up

    def get_driver(self) -> str:
        if not self.up:
            raise RuntimeError("video system not initialized")
        return self.driver

    def init(self) -> None:
        self.inits += 1
        if self.fail:
            raise RuntimeError("No available video device")
        driver = self.env.get("SDL_VIDEODRIVER", "")
        if driver == "not_a_driver":
            raise RuntimeError("not_a_driver not available")
        self.up = True
        self.driver = driver or "windows"


class _Joystick:
    def __init__(self) -> None:
        self.up = False
        self.inits = 0
        self.quits = 0

    def get_init(self) -> bool:
        return self.up

    def init(self) -> None:
        self.inits += 1
        self.up = True

    def quit(self) -> None:
        self.quits += 1
        self.up = False


class _Events:
    def __init__(self) -> None:
        self.pending: list[object] = []
        self.calls = 0
        self.error: BaseException | None = None

    def get(self) -> list[object]:
        self.calls += 1
        if self.error is not None:
            raise self.error
        return list(self.pending)


class _Pygame:
    def __init__(self, env: dict[str, str]) -> None:
        self.env = env
        self.display = _Display(env)
        self.joystick = _Joystick()
        self.event = _Events()
        self.up = False
        self.inits = 0
        self.quits = 0
        # None: pg.init() brings the display up. Otherwise only that driver does.
        self.video_needs: str | None = None
        self.joystick_from_init = True

    def get_init(self) -> bool:
        return self.up

    def init(self) -> tuple[int, int]:
        self.inits += 1
        self.up = True
        if self.joystick_from_init:
            self.joystick.up = True
        driver = self.env.get("SDL_VIDEODRIVER", "")
        if self.video_needs is None or driver == self.video_needs:
            self.display.up = True
            self.display.driver = driver or "windows"
            return (6, 0)
        self.display.up = False
        return (5, 1)

    def quit(self) -> None:
        self.quits += 1
        self.up = False
        self.display.up = False
        self.joystick.up = False


def _fn(name: str) -> ast.FunctionDef:
    tree = ast.parse(PEDAL_SRC.read_text(encoding="utf-8-sig"), filename=str(PEDAL_SRC))
    found = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == name
    )
    return found


def _calls(node: ast.AST) -> list[str]:
    return [
        ast.unparse(call.func)
        for call in ast.walk(node)
        if isinstance(call, ast.Call)
    ]


def test_working_video_keeps_the_current_driver(caplog):
    env = {"SDL_VIDEODRIVER": "windows"}
    pg = _Pygame(env)
    with caplog.at_level(logging.INFO):
        assert ensure_pygame_video(pg, env) is True
    assert env["SDL_VIDEODRIVER"] == "windows"
    assert pg.inits == 1
    assert pg.quits == 0
    assert pg.display.inits == 0
    assert "6 passed, 0 failed, video windows" in caplog.text


def test_failed_video_retries_with_the_dummy_driver(caplog):
    env = {"SDL_VIDEODRIVER": "not_a_driver"}
    pg = _Pygame(env)
    pg.video_needs = "dummy"
    with caplog.at_level(logging.INFO):
        assert ensure_pygame_video(pg, env) is True
    assert env["SDL_VIDEODRIVER"] == "dummy"
    assert pg.display.get_driver() == "dummy"
    assert pg.quits == 0
    assert "not_a_driver not available" in caplog.text
    assert "SDL_VIDEODRIVER='not_a_driver'" in caplog.text
    assert "video dummy" in caplog.text


def test_direct_display_retry_does_not_switch_driver_when_it_recovers(caplog):
    env: dict[str, str] = {}
    pg = _Pygame(env)
    pg.video_needs = "windows"
    # pg.init() leaves the display down; display.init() itself succeeds.
    pg.display.fail = False

    def init() -> tuple[int, int]:
        pg.inits += 1
        pg.up = True
        pg.joystick.up = True
        pg.display.up = False
        return (4, 1)

    pg.init = init  # type: ignore[method-assign]
    with caplog.at_level(logging.INFO):
        assert ensure_pygame_video(pg, env) is True
    assert "SDL_VIDEODRIVER" not in env
    assert pg.display.driver == "windows"
    assert pg.quits == 0
    assert "dummy driver" not in caplog.text


def test_still_down_returns_false_without_quitting_pygame(caplog):
    env: dict[str, str] = {}
    pg = _Pygame(env)
    pg.display.fail = True
    pg.video_needs = "never"
    with caplog.at_level(logging.WARNING):
        assert ensure_pygame_video(pg, env) is False
    assert pg.quits == 0
    assert env["SDL_VIDEODRIVER"] == "dummy"
    assert "still down" in caplog.text


def test_joystick_is_initialised_when_pygame_init_leaves_it_down():
    env: dict[str, str] = {}
    pg = _Pygame(env)
    pg.joystick_from_init = False
    assert ensure_pygame_video(pg, env) is True
    assert pg.joystick.inits == 1
    assert pg.joystick.up is True


def test_already_up_does_not_init_again():
    env: dict[str, str] = {}
    pg = _Pygame(env)
    pg.up = True
    pg.display.up = True
    pg.display.driver = "windows"
    pg.joystick.up = True
    assert ensure_pygame_video(pg, env) is True
    assert pg.inits == 0
    assert pg.joystick.inits == 0


def test_read_pumps_events_after_the_dummy_fallback():
    env = {"SDL_VIDEODRIVER": "not_a_driver"}
    pg = _Pygame(env)
    pg.video_needs = "dummy"
    pg.event.pending = ["axis"]
    video = PygameVideo()
    assert video.read(pg, env) == ["axis"]
    assert video.failed is False
    assert pg.event.calls == 1


def test_read_does_not_call_event_get_when_video_stays_down():
    env: dict[str, str] = {}
    pg = _Pygame(env)
    pg.display.fail = True
    pg.video_needs = "never"
    pg.event.error = RuntimeError("video system not initialized")
    video = PygameVideo()
    assert video.read(pg, env) == []
    assert video.failed is True
    assert pg.event.calls == 0
    assert video.read(pg, env) == []
    assert pg.inits == 1
    assert pg.display.inits == 2


def test_event_get_video_error_gives_up_and_releases_the_joystick():
    env: dict[str, str] = {}
    pg = _Pygame(env)
    pg.up = True
    pg.display.up = True
    pg.display.driver = "windows"
    pg.joystick.up = True
    pg.event.error = RuntimeError("video system not initialized")
    video = PygameVideo()
    video.ok = True
    assert video.read(pg, env) == []
    assert video.failed is True
    assert video.take_failure() is True
    assert video.take_failure() is False
    video.release(pg)
    assert pg.joystick.quits == 1
    assert pg.joystick.up is False
    video.release(pg)
    assert pg.joystick.quits == 1
    assert video.read(pg, env) == []
    assert pg.event.calls == 1


def test_other_pump_errors_are_logged_once_and_retried(caplog):
    env: dict[str, str] = {}
    pg = _Pygame(env)
    pg.up = True
    pg.display.up = True
    pg.display.driver = "windows"
    pg.joystick.up = True
    pg.event.error = RuntimeError("event queue busy")
    video = PygameVideo()
    video.ok = True
    with caplog.at_level(logging.WARNING):
        assert video.read(pg, env) == []
        assert video.read(pg, env) == []
    assert video.failed is False
    assert pg.event.calls == 2
    assert caplog.text.count("event queue busy") == 1


def test_gave_up_start_does_not_init_again():
    env: dict[str, str] = {}
    pg = _Pygame(env)
    pg.display.fail = True
    pg.video_needs = "never"
    video = PygameVideo()
    assert video.start(pg, env) is False
    inits = pg.inits
    assert video.start(pg, env) is False
    assert pg.inits == inits


def test_publish_released_clears_pedals_and_button_holds():
    data = SimpleNamespace(
        _lock=threading.Lock(),
        gasval=0.4,
        brakeval=0.9,
        gas_output=0.4,
        brake_output=1.0,
        opdgasval=0.2,
        opdbrakeval=0.8,
        em_stop=True,
        cc_start_held=True,
        cc_inc_held=True,
        cc_dec_held=True,
        acc_dist_inc_held=True,
        acc_dist_dec_held=True,
        device_lost=False,
    )
    publish_released(data)
    assert data.gasval == data.brakeval == 0.0
    assert data.gas_output == data.brake_output == 0.0
    assert data.opdgasval == data.opdbrakeval == 0.0
    assert data.em_stop is False
    assert data.cc_start_held is False
    assert data.cc_inc_held is False
    assert data.cc_dec_held is False
    assert data.acc_dist_inc_held is False
    assert data.acc_dist_dec_held is False
    assert data.device_lost is False


def test_pedal_thread_pumps_through_the_video_helper():
    process = _fn("_process_pygame_events")
    calls = _calls(process)
    assert "pygame.event.get" not in calls
    assert "pygame.init" not in calls
    assert "self._video.read" in calls

    setup = _fn("setup")
    setup_calls = _calls(setup)
    assert "pygame.init" not in setup_calls
    assert "self._video.start" in setup_calls
    returns = [node for node in ast.walk(setup) if isinstance(node, ast.Return)]
    assert returns, "setup() must return before opening joysticks when video failed"

    loop = _fn("loop")
    failed = next(
        node
        for node in loop.body
        if isinstance(node, ast.If)
        and isinstance(node.test, ast.Attribute)
        and node.test.attr == "failed"
    )
    assert any(isinstance(node, ast.Return) for node in failed.body)
    process_calls = [
        node for node in ast.walk(loop) if isinstance(node, ast.Call)
        and ast.unparse(node.func) == "self._process_pygame_events"
    ]
    assert process_calls
    assert failed.lineno < min(call.lineno for call in process_calls)
