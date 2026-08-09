"""Consent overlay: the read gate, the screenshot opt-out, and the shipped document."""
from __future__ import annotations

import os

import pytest

from PySide6.QtWidgets import QApplication, QWidget

from ui.main_window.consent_overlay import CONSENT_MARKDOWN, ConsentOverlay


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
