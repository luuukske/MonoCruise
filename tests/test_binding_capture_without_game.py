"""Button assignment must work with the game closed.

`main_pedal_thread.loop()` returns early when telemetry is not connected. When
the joystick snapshot sat below that gate, joystick assignment saw no presses
and the HID scan never opened, because its gate waits on
`joystick_capture_ready`, which that same snapshot publishes. Keyboard
assignment kept working (OS hook), which is what made the failure look like a
HID/pygame problem. See core/main_pedal_thread/README.md.

The pedal thread is checked through the AST: pygame is not installed in CI.
"""

from __future__ import annotations

import ast
import threading
from pathlib import Path

import pytest

from core.button_device_thread.thread import ButtonDeviceThread
from core.thread_management.registry import registry

REPO = Path(__file__).resolve().parents[1]
PEDAL_SRC = REPO / "core" / "main_pedal_thread" / "thread.py"


def _game_closed_branch() -> ast.If:
    """The `if tel is None:` early-return branch of MainPedalThread.loop."""
    tree = ast.parse(PEDAL_SRC.read_text(encoding="utf-8-sig"), filename=str(PEDAL_SRC))
    loop = next(
        node
        for cls in tree.body
        if isinstance(cls, ast.ClassDef) and cls.name == "MainPedalThread"
        for node in cls.body
        if isinstance(node, ast.FunctionDef) and node.name == "loop"
    )
    branches = [
        node
        for node in ast.walk(loop)
        if isinstance(node, ast.If)
        and isinstance(node.test, ast.Compare)
        and isinstance(node.test.left, ast.Name)
        and node.test.left.id == "tel"
        and isinstance(node.test.ops[0], ast.Is)
        and isinstance(node.test.comparators[0], ast.Constant)
        and node.test.comparators[0].value is None
    ]
    assert len(branches) == 1, "expected one `if tel is None:` branch in loop()"
    return branches[0]


def _self_calls(node: ast.AST) -> set[str]:
    return {
        call.func.attr
        for call in ast.walk(node)
        if isinstance(call, ast.Call)
        and isinstance(call.func, ast.Attribute)
        and isinstance(call.func.value, ast.Name)
        and call.func.value.id == "self"
    }


def test_joystick_snapshot_runs_with_the_game_closed():
    called = _self_calls(_game_closed_branch())
    assert "_update_joystick_states" in called, (
        "no joystick snapshot while the game is closed: joystick assignment sees "
        "no presses and joystick_capture_ready never gates the HID scan open"
    )
    assert "_ensure_button_devices" in called, (
        "bound joystick devices go untracked while the game is closed, so the "
        "settings panel shows no pressed highlight"
    )


def test_game_closed_branch_cannot_act_on_a_button():
    """Publishing button state there must not reach the cruise buttons."""
    assert "_read_cc_button_states" not in _self_calls(_game_closed_branch())


class _Data:
    def __init__(self, ready: bool) -> None:
        self._lock = threading.Lock()
        self.joystick_capture_ready = ready


class _FakePedalThread:
    name = "main_pedal_thread"

    def __init__(self, *, alive: bool, ready: bool) -> None:
        self.data = _Data(ready)
        self._alive = alive

    def is_alive(self) -> bool:
        return self._alive


@pytest.fixture
def pedal_thread():
    def register(*, alive: bool, ready: bool) -> _FakePedalThread:
        t = _FakePedalThread(alive=alive, ready=ready)
        registry.replace(t)
        return t

    yield register
    registry.unregister("main_pedal_thread")


def test_hid_capture_waits_for_the_joystick_snapshot(pedal_thread):
    pedal_thread(alive=True, ready=False)
    assert ButtonDeviceThread._pygame_capture_ready() is False


def test_hid_capture_opens_once_the_snapshot_is_published(pedal_thread):
    pedal_thread(alive=True, ready=True)
    assert ButtonDeviceThread._pygame_capture_ready() is True


def test_hid_capture_does_not_wait_on_a_dead_pedal_thread(pedal_thread):
    """Nothing owns the joysticks then, and the flag would never arrive."""
    pedal_thread(alive=False, ready=False)
    assert ButtonDeviceThread._pygame_capture_ready() is True


def test_hid_capture_does_not_wait_when_the_thread_is_missing():
    registry.unregister("main_pedal_thread")
    assert ButtonDeviceThread._pygame_capture_ready() is True
