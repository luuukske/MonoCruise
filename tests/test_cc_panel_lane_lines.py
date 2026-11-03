"""CC panel lane lines: LDW/LKA state, warning hold, fades, and where the panel draws them."""
from __future__ import annotations

import time

import pytest
from PySide6.QtCore import QPoint
from PySide6.QtGui import QColor, QFontMetrics, QImage, QPainter, QRegion
from PySide6.QtWidgets import QApplication, QWidget

from core.lane_departure import display as ll
from core.lane_departure.display import ALERT, DISABLED, LEFT, OFF, RIGHT, STANDBY, WARN, LaneLines


@pytest.mark.parametrize("reason, plugin_ok, state", [
    ("clear", True, STANDBY),
    ("blinker", True, STANDBY),
    ("deliberate", True, STANDBY),
    ("returning", True, STANDBY),
    ("warn", True, STANDBY),
    ("speed", True, DISABLED),
    ("bend_sharp", True, DISABLED),
    ("branch", True, DISABLED),
    ("lane_settle", True, DISABLED),
    ("no_map", True, OFF),
    ("off", True, OFF),
    ("stale", True, OFF),
    ("clear", False, OFF),
    ("something_new", True, DISABLED),
])
def test_detector_reasons_map_to_line_states(reason, plugin_ok, state):
    assert ll.base_state(reason, plugin_ok) == state


def test_the_reason_buckets_do_not_overlap():
    assert not (ll._OFF_REASONS & ll._DISABLED_REASONS)
    assert not (ll._OFF_REASONS & ll._STANDBY_REASONS)
    assert not (ll._DISABLED_REASONS & ll._STANDBY_REASONS)


def test_detector_sides_map_to_panel_sides():
    assert [ll.panel_side(s) for s in (0, 1, 2, 7)] == [0, LEFT, RIGHT, 0]


def _run(lines: LaneLines, t0: float, t1: float, dt: float = 0.02) -> float:
    t = t0
    while t < t1:
        t = min(t + dt, t1)
        lines.step(t)
    return t


def _settled(base: str = STANDBY) -> tuple[LaneLines, float]:
    lines = LaneLines()
    lines.set(base, 0, 0.0)
    return lines, _run(lines, 0.0, 2.0)


def test_a_warning_shows_at_once_on_its_side_only():
    lines, t = _settled()
    lines.set(STANDBY, LEFT, t)
    assert lines.weights(LEFT)[WARN] == 1.0
    assert lines.weights(RIGHT)[STANDBY] == 1.0
    assert lines.weights(RIGHT)[WARN] == 0.0


def test_a_warning_is_held_one_second_after_it_stops():
    lines, t = _settled()
    lines.set(STANDBY, RIGHT, t)
    t = _run(lines, t, t + 3.0)
    assert lines.weights(RIGHT)[WARN] == 1.0
    lines.set(STANDBY, 0, t)
    end = t + ll.HOLD_S
    _run(lines, t, end - 0.02)
    assert lines.weights(RIGHT)[WARN] == 1.0
    _run(lines, end - 0.02, end + ll.FADE_S + 0.05)
    assert lines.weights(RIGHT)[WARN] == 0.0
    assert lines.weights(RIGHT)[STANDBY] == 1.0


def test_firing_again_inside_the_hold_restarts_it():
    lines, t = _settled()
    lines.set(STANDBY, LEFT, t)
    lines.set(STANDBY, 0, t + 0.5)
    lines.set(STANDBY, LEFT, t + 1.2)
    lines.set(STANDBY, 0, t + 1.4)
    _run(lines, t, t + 2.3)
    assert lines.weights(LEFT)[WARN] == 1.0


def test_the_other_side_takes_over_the_warning():
    lines, t = _settled()
    lines.set(STANDBY, LEFT, t)
    lines.set(STANDBY, RIGHT, t + 0.5)
    assert lines.weights(RIGHT)[WARN] == 1.0
    _run(lines, t + 0.5, t + 0.5 + ll.FADE_S + 0.05)
    assert lines.weights(LEFT)[WARN] == 0.0
    assert lines.weights(LEFT)[STANDBY] == 1.0


def test_a_short_blip_never_shows():
    lines, t = _settled()
    lines.set(DISABLED, 0, t)
    lines.set(STANDBY, 0, t + ll.SETTLE_S - 0.1)
    _run(lines, t, t + 2.0)
    assert lines.weights(LEFT)[DISABLED] == 0.0
    assert lines.weights(LEFT)[STANDBY] == 1.0


def test_a_change_settles_then_fades_from_where_it_was_asked():
    lines, t = _settled()
    t = _run(lines, t, t + 10.0, dt=5.0)
    lines.set(DISABLED, 0, t)
    lines.step(t + ll.SETTLE_S)
    assert lines.weights(LEFT)[STANDBY] == 1.0
    lines.step(t + ll.SETTLE_S + 0.05)
    assert 0.7 < lines.weights(LEFT)[STANDBY] < 1.0
    _run(lines, t + ll.SETTLE_S + 0.05, t + ll.SETTLE_S + ll.FADE_S + 0.05)
    assert lines.weights(LEFT)[DISABLED] == 1.0


def test_off_hides_the_lines_and_the_timer_rests():
    lines, t = _settled()
    assert lines.visible()
    assert lines.wake_in(t) is None
    lines.set(OFF, 0, t)
    assert lines.wake_in(t) == pytest.approx(ll.SETTLE_S)
    _run(lines, t, t + 1.0)
    assert not lines.visible()
    assert lines.wake_in(t + 1.0) is None


def test_the_timer_wakes_for_the_end_of_the_hold():
    lines, t = _settled()
    lines.set(STANDBY, LEFT, t)
    assert lines.wake_in(t) is None
    lines.set(STANDBY, 0, t + 2.0)
    assert lines.wake_in(t + 2.0) == pytest.approx(ll.HOLD_S)


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


@pytest.fixture
def panel(qapp):
    from ui.cc_panel.main import cc_panel

    p = cc_panel("80 km/h", "Cruise control", True, scale_mult=1.0)
    yield p
    p.stop()


def _render(p, lane_state=STANDBY, lane_side=0, **fields) -> QImage:
    state = dict(
        text_content="80 km/h", cc_mode="Cruise control", cc_enabled=True, acc_enabled=True, acc_locked=True,
        acc_truck=False, distance_to_lead=2, AEB_warn=False, lead_vehicle_speed=None,
        lane_state=lane_state, lane_side=lane_side, _complete_update=True,
    )
    state.update(fields)
    p._widget._on_update(state)
    p._widget._lane_timer.stop()
    now = time.monotonic()
    _run(p._lane, now, now + 2.0)
    p._vehicle_anim_layout_key = ()
    img = QImage(p._panel_w, p._panel_h, QImage.Format.Format_ARGB32_Premultiplied)
    img.fill(QColor("#000000"))
    pa = QPainter(img)
    p._widget.render(pa, QPoint(0, 0), QRegion(), QWidget.RenderFlag.DrawChildren)
    pa.end()
    p._widget._vehicle_anim_timer.stop()
    return img


def _count(img: QImage, test, x0: int = 0, x1: int | None = None) -> int:
    n = 0
    for y in range(img.height()):
        for x in range(x0, img.width() if x1 is None else x1):
            if test(QColor(img.pixel(x, y))):
                n += 1
    return n


def _white(c: QColor) -> bool:
    return min(c.red(), c.green(), c.blue()) > 200


def _red(c: QColor) -> bool:
    return c.red() > 200 and c.green() < 80 and c.blue() < 80


def _icon_mid(p) -> int:
    icon_sz = int(QFontMetrics(p._font).height() * 1.8 * 0.85)
    return p._panel_w - int(20 * p._scale_mult) - icon_sz // 2


def test_off_looks_like_today(panel):
    assert _count(_render(panel, lane_state=OFF), _white) == 0


@pytest.mark.parametrize("distance", [1, 2, 3, 4])
@pytest.mark.parametrize("lead", ["car", "truck", "none"])
def test_standby_lines_show_at_every_distance(panel, distance, lead):
    img = _render(panel, distance_to_lead=distance, acc_locked=lead != "none", acc_truck=lead == "truck")
    mid = _icon_mid(panel)
    assert _count(img, _white, 0, mid) > 20
    assert _count(img, _white, mid) > 20


def _orange(c: QColor) -> bool:
    return c.red() > 200 and 100 < c.green() < 180 and c.blue() < 60


def test_a_warning_is_steady_orange_on_its_side_only(panel):
    img = _render(panel, lane_side=LEFT)
    mid = _icon_mid(panel)
    assert _count(img, _orange, 0, mid) > 20
    assert _count(img, _red, 0, mid) == 0
    assert _count(img, _orange, mid) == 0
    assert _count(img, _white, mid) > 20


def test_an_alert_is_red_on_its_side_only(panel, monkeypatch):
    monkeypatch.setattr(ll, "BLINK_S", 10.0)
    img = _render(panel, lane_side=RIGHT, lane_alert=True)
    mid = _icon_mid(panel)
    assert _count(img, _red, mid) > 20
    assert _count(img, _orange, mid) == 0
    assert _count(img, _red, 0, mid) == 0


def test_lines_show_only_with_the_acc_distance_lines(panel):
    assert _count(_render(panel, acc_enabled=False), _white) == 0
    assert _count(_render(panel, cc_mode="Speed limiter"), _white) == 0


def test_disabled_lines_are_dim(panel):
    img = _render(panel, lane_state=DISABLED)
    assert _count(img, _white) == 0
    assert _count(img, lambda c: 50 < c.red() < 200 and abs(c.red() - c.blue()) < 30) > 0


def test_an_alert_shows_at_once_and_blinks_from_lit():
    lines, t = _settled()
    lines.set(STANDBY, RIGHT, t, alert=True)
    assert lines.weights(RIGHT)[ALERT] == 1.0
    assert lines.shown(RIGHT, t)[ALERT] == 1.0
    assert lines.shown(RIGHT, t + ll.BLINK_S + 0.01)[ALERT] == 0.0
    assert lines.shown(RIGHT, t + 2.0 * ll.BLINK_S + 0.01)[ALERT] == 1.0
    assert lines.shown(LEFT, t)[STANDBY] == 1.0


def test_an_alert_stays_one_for_its_episode_hold_included():
    lines, t = _settled()
    lines.set(STANDBY, RIGHT, t, alert=True)
    lines.set(STANDBY, RIGHT, t + 0.2, alert=False)
    assert lines.target(RIGHT, t + 0.2) == ALERT
    lines.set(STANDBY, 0, t + 0.5)
    _run(lines, t + 0.5, t + 0.5 + ll.HOLD_S - 0.02)
    assert lines.target(RIGHT, t + 0.5 + ll.HOLD_S - 0.02) == ALERT
    t = _run(lines, t + 0.5 + ll.HOLD_S - 0.02, t + 3.0)
    lines.set(STANDBY, RIGHT, t)
    assert lines.weights(RIGHT)[WARN] == 1.0, "a new episode starts as a plain warning"


def test_a_warning_turns_into_an_alert_at_once_and_its_blink_starts_lit():
    lines, t = _settled()
    lines.set(STANDBY, LEFT, t)
    lines.set(STANDBY, LEFT, t + 0.3, alert=True)
    assert lines.weights(LEFT)[ALERT] == 1.0
    assert lines.shown(LEFT, t + 0.3)[ALERT] == 1.0


def test_the_timer_wakes_for_each_blink():
    lines, t = _settled()
    lines.set(STANDBY, LEFT, t, alert=True)
    assert lines.wake_in(t) == pytest.approx(ll.BLINK_S)
    assert lines.wake_in(t + 0.05) == pytest.approx(ll.BLINK_S - 0.05)
