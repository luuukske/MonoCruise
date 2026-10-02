"""Consent overlay: the read gate, the screenshot opt-out, and the shipped document."""
from __future__ import annotations

import os

import pytest

from PySide6.QtWidgets import QApplication, QWidget

from ui.main_window.constants import WINDOW_HEIGHT, WINDOW_WIDTH
from ui.main_window.consent_overlay import _BODY_HEIGHT, CONSENT_MARKDOWN, ConsentOverlay


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


@pytest.fixture()
def host(qapp):
    w = QWidget()
    w.resize(900, 700)
    w.show()          # the read gate refuses to evaluate a widget never laid out
    qapp.processEvents()
    yield w
    w.hide()
    w.deleteLater()


def _overlay(host, tmp_path, body: str, **kw) -> ConsentOverlay:
    md = tmp_path / "consent.md"
    md.write_text(body, encoding="utf-8")
    calls = {}
    ov = ConsentOverlay(
        host,
        markdown_path=str(md),
        on_accept=lambda shot: calls.setdefault("accept", shot),
        on_decline=lambda: calls.setdefault("decline", True),
        **kw,
    )
    ov._calls = calls
    return ov


def _assert_footer_clear_of_body(ov: ConsentOverlay) -> None:
    """Body must end above the checkbox; checkbox and buttons must stay in the card."""
    body = ov._body.geometry()
    shot = ov._screenshot.geometry()
    btn = ov._accept_btn.geometry()
    card = ov._card.geometry()
    assert body.bottom() < shot.top(), (
        f"body overlaps checkbox: body.bottom={body.bottom()} shot.top={shot.top()}"
    )
    assert shot.bottom() < btn.top(), (
        f"checkbox overlaps buttons: shot.bottom={shot.bottom()} btn.top={btn.top()}"
    )
    assert btn.bottom() <= card.height(), (
        f"buttons clipped by card: btn.bottom={btn.bottom()} card.h={card.height()}"
    )
    assert card.bottom() <= ov.height(), (
        f"card overflows overlay: card.bottom={card.bottom()} ov.h={ov.height()}"
    )


def test_the_shipped_document_and_image_are_both_present():
    """They live under ui/main_window/assets so the spec's datas entry bundles them."""
    assert os.path.isfile(CONSENT_MARKDOWN)
    image = os.path.join(os.path.dirname(CONSENT_MARKDOWN), "clip_screenshot_example.jpg")
    assert os.path.isfile(image)
    assert "clip_screenshot_example.jpg" in open(CONSENT_MARKDOWN, encoding="utf-8").read()


def test_shipped_document_renders_with_an_inlined_image(host, qapp):
    ov = ConsentOverlay(host, on_accept=lambda _s: None)
    qapp.processEvents()
    assert "data:image/png;base64," in ov._body.toHtml()


def test_accept_is_locked_until_scrolled_to_the_end(host, tmp_path, qapp):
    ov = _overlay(host, tmp_path, "para\n\n" * 400)
    qapp.processEvents()
    assert not ov._accept_btn.isEnabled()

    bar = ov._body.verticalScrollBar()
    assert bar.maximum() > 0, "test document must be long enough to scroll"
    bar.setValue(bar.maximum())
    assert ov._accept_btn.isEnabled()


def test_a_document_that_does_not_scroll_unlocks_immediately(host, tmp_path, qapp):
    """Otherwise a short document could never be accepted at all."""
    ov = _overlay(host, tmp_path, "one short line")
    qapp.processEvents()
    assert ov._body.verticalScrollBar().maximum() == 0
    assert ov._accept_btn.isEnabled()


def test_scrolling_back_up_does_not_relock(host, tmp_path, qapp):
    ov = _overlay(host, tmp_path, "para\n\n" * 400)
    qapp.processEvents()
    bar = ov._body.verticalScrollBar()
    bar.setValue(bar.maximum())
    bar.setValue(0)
    assert ov._accept_btn.isEnabled()


def test_decline_always_works_even_while_locked(host, tmp_path, qapp):
    ov = _overlay(host, tmp_path, "para\n\n" * 400)
    qapp.processEvents()
    assert not ov._accept_btn.isEnabled()
    ov._decline()
    assert ov._calls.get("decline") is True


def test_screenshot_checkbox_is_the_settings_checkmark_widget(host, tmp_path, qapp):
    """Plain QCheckBox has no glyph under our QSS; settings uses CheckBox for that."""
    from ui.main_window.widgets import CheckBox

    ov = _overlay(host, tmp_path, "short")
    qapp.processEvents()
    assert isinstance(ov._screenshot, CheckBox)
    assert ov._screenshot.isChecked()
    assert ov._screenshot.width() == 24 and ov._screenshot.height() == 24


def test_screenshot_is_on_by_default_and_reported_on_accept(host, tmp_path, qapp):
    ov = _overlay(host, tmp_path, "short")
    qapp.processEvents()
    assert ov.include_screenshot is True
    ov._accept()
    assert ov._calls.get("accept") is True


def test_screenshot_can_be_opted_out_without_blocking_accept(host, tmp_path, qapp):
    ov = _overlay(host, tmp_path, "short")
    qapp.processEvents()
    ov._screenshot.setChecked(False)
    assert ov._accept_btn.isEnabled()
    ov._accept()
    assert ov._calls.get("accept") is False


def test_an_unreadable_document_cannot_be_accepted_blindly(host, qapp, tmp_path):
    """A missing file must not silently present an empty, instantly-acceptable page."""
    ov = ConsentOverlay(host, markdown_path=str(tmp_path / "gone.md"), on_accept=lambda _s: None)
    qapp.processEvents()
    assert "could not be loaded" in ov._body.toPlainText()


def test_body_uses_full_preferred_height_when_the_window_is_tall(host, tmp_path, qapp):
    ov = _overlay(host, tmp_path, "para\n\n" * 400)
    qapp.processEvents()
    assert ov._body.height() == _BODY_HEIGHT
    _assert_footer_clear_of_body(ov)


@pytest.mark.parametrize("height", [WINDOW_HEIGHT, 450, 400, 350, 300, 250])
def test_body_shrinks_instead_of_covering_the_footer(qapp, tmp_path, height):
    """Default window height is shorter than the card's preferred size; body must yield."""
    host = QWidget()
    host.resize(WINDOW_WIDTH, height)
    host.show()
    qapp.processEvents()
    try:
        ov = _overlay(host, tmp_path, "para\n\n" * 400)
        qapp.processEvents()
        assert ov._body.height() <= _BODY_HEIGHT
        if height < 520:
            assert ov._body.height() < _BODY_HEIGHT
        _assert_footer_clear_of_body(ov)
    finally:
        host.hide()
        host.deleteLater()


def test_shrinking_the_host_keeps_the_footer_clear(host, tmp_path, qapp):
    ov = _overlay(host, tmp_path, "para\n\n" * 400)
    qapp.processEvents()
    assert ov._body.height() == _BODY_HEIGHT

    for height in (500, 420, 360, 280):
        host.resize(WINDOW_WIDTH, height)
        qapp.processEvents()
        assert ov._body.height() < _BODY_HEIGHT
        _assert_footer_clear_of_body(ov)


def test_read_gate_still_works_when_the_body_is_compressed(qapp, tmp_path):
    host = QWidget()
    host.resize(WINDOW_WIDTH, WINDOW_HEIGHT)
    host.show()
    qapp.processEvents()
    try:
        ov = _overlay(host, tmp_path, "para\n\n" * 400)
        qapp.processEvents()
        assert ov._body.height() < _BODY_HEIGHT
        assert not ov._accept_btn.isEnabled()
        bar = ov._body.verticalScrollBar()
        assert bar.maximum() > 0
        bar.setValue(bar.maximum())
        assert ov._accept_btn.isEnabled()
    finally:
        host.hide()
        host.deleteLater()
