"""
MonoCruise – Reusable widget factory functions.

Every factory inserts its widget into the *parent*'s QGridLayout at the
specified (row, col) and returns the widget reference.  The factories mirror
the original ``new_label``, ``new_checkbutton``, ``new_entry``,
``new_optionmenu`` helpers from MonoCruise.py.
"""

from __future__ import annotations

from typing import Any, Callable, Sequence

from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QSizePolicy,
    QWidget,
)

from ui.main_window.constants import BG_COLOR, SETTINGS_COLOR


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _grid(parent: QWidget) -> QGridLayout:
    layout = parent.layout()
    assert isinstance(layout, QGridLayout), "Parent widget must have a QGridLayout"
    return layout


# ---------------------------------------------------------------------------
# Label  (mirrors ``new_label``)
# ---------------------------------------------------------------------------

def new_label(
    parent: QWidget,
    row: int,
    col: int,
    text: str,
    *,
    object_name: str | None = None,
    alignment: Qt.AlignmentFlag = Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
    col_span: int = 1,
) -> QLabel:
    lbl = QLabel(text)
    if object_name:
        lbl.setObjectName(object_name)
    lbl.setAlignment(alignment)
    lbl.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
    _grid(parent).addWidget(lbl, row, col, 1, col_span)
    return lbl


# ---------------------------------------------------------------------------
# Section header  (full‑width, spanning both columns)
# ---------------------------------------------------------------------------

def new_section_header(parent: QWidget, row: int, text: str) -> QLabel:
    lbl = QLabel(text)
    lbl.setObjectName("sectionHeader")
    lbl.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
    _grid(parent).addWidget(lbl, row, 0, 1, 2)
    return lbl


# ---------------------------------------------------------------------------
# Subtext / description
# ---------------------------------------------------------------------------

def new_subtext(
    parent: QWidget,
    row: int,
    col: int,
    text: str,
    *,
    col_span: int = 1,
) -> QLabel:
    lbl = QLabel(text)
    lbl.setObjectName("subtext")
    lbl.setWordWrap(True)
    lbl.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
    lbl.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
    _grid(parent).addWidget(lbl, row, col, 1, col_span)
    return lbl


# ---------------------------------------------------------------------------
# Checkbox  (24×24, no text)
# ---------------------------------------------------------------------------

def new_checkbutton(
    parent: QWidget,
    row: int,
    col: int,
    checked: bool = False,
    *,
    callback: Callable[..., Any] | None = None,
) -> QCheckBox:
    cb = QCheckBox()
    cb.setFixedSize(24, 24)
    cb.setChecked(checked)
    if callback:
        cb.toggled.connect(callback)
    _grid(parent).addWidget(
        cb, row, col,
        alignment=Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter,
    )
    return cb


# ---------------------------------------------------------------------------
# Entry (QLineEdit) – 50 px wide, 2‑second debounce, min/max validation
# ---------------------------------------------------------------------------

def new_entry(
    parent: QWidget,
    row: int,
    col: int,
    *,
    value: Any = "",
    value_type: type = float,
    minimum: float | int | None = None,
    maximum: float | int | None = None,
    callback: Callable[[Any], Any] | None = None,
) -> QLineEdit:
    le = QLineEdit(str(value))
    le.setFixedWidth(50)
    le.setAlignment(Qt.AlignmentFlag.AlignCenter)

    # -- debounce timer (2 s, matching original ``master.after(2000, …)``) --
    _debounce = QTimer()
    _debounce.setSingleShot(True)
    _debounce.setInterval(2000)

    _last_good = [value]  # mutable container so the closure can update it

    def _validate() -> None:
        raw = le.text().strip()
        try:
            v = value_type(raw)
        except (ValueError, TypeError):
            le.setText(str(_last_good[0]))
            le.clearFocus()
            return
        if minimum is not None and v < minimum:
            v = value_type(minimum)
        if maximum is not None and v > maximum:
            v = value_type(maximum)
        le.setText(str(v))
        _last_good[0] = v
        le.clearFocus()
        if callback:
            callback(v)

    _debounce.timeout.connect(_validate)
    le.textChanged.connect(lambda _: _debounce.start())
    le.returnPressed.connect(lambda: (_debounce.stop(), _validate()))

    # Escape reverts
    _orig_key_press = le.keyPressEvent

    def _key_override(event):
        if event.key() == Qt.Key.Key_Escape:
            le.setText(str(_last_good[0]))
            le.clearFocus()
        else:
            _orig_key_press(event)

    le.keyPressEvent = _key_override  # type: ignore[assignment]

    _grid(parent).addWidget(
        le, row, col,
        alignment=Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter,
    )
    return le


# ---------------------------------------------------------------------------
# Option menu (QComboBox wrapped in a bordered QFrame)
# ---------------------------------------------------------------------------

def new_optionmenu(
    parent: QWidget,
    row: int,
    col: int,
    values: Sequence[str],
    *,
    default: str = "",
    callback: Callable[[str], Any] | None = None,
) -> QComboBox:
    # Bordered wrapper matching the original CTkFrame border‑style
    wrapper = QFrame()
    wrapper.setStyleSheet(
        f"QFrame {{ border: 1.5px solid {SETTINGS_COLOR}; "
        f"border-radius: 5px; background-color: transparent; }}"
    )
    wrapper_lay = QHBoxLayout(wrapper)
    wrapper_lay.setContentsMargins(2, 2, 2, 2)

    combo = QComboBox()
    combo.setFixedWidth(100)
    combo.addItems([str(v) for v in values])
    if default and str(default) in [str(v) for v in values]:
        combo.setCurrentText(str(default))
    if callback:
        combo.currentTextChanged.connect(callback)
    wrapper_lay.addWidget(combo)

    _grid(parent).addWidget(
        wrapper, row, col,
        alignment=Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter,
    )
    # Return the combo itself so callers can read/set its value directly
    return combo


# ---------------------------------------------------------------------------
# Clickable label  (emulates the "button detection" click‑to‑assign pattern)
# ---------------------------------------------------------------------------

class ClickableLabel(QLabel):
    """QLabel that invokes a callback on mouse press."""

    def __init__(self, text: str, callback: Callable[..., Any] | None = None) -> None:
        super().__init__(text)
        self.setObjectName("clickable")
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self._callback = callback

    def mousePressEvent(self, event) -> None:
        if self._callback:
            self._callback()
        super().mousePressEvent(event)


def new_clickable_label(
    parent: QWidget,
    row: int,
    col: int,
    text: str,
    *,
    callback: Callable[..., Any] | None = None,
) -> ClickableLabel:
    lbl = ClickableLabel(text, callback)
    _grid(parent).addWidget(
        lbl, row, col,
        alignment=Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter,
    )
    return lbl


# ---------------------------------------------------------------------------
# BETA pill badge
# ---------------------------------------------------------------------------

def new_beta_pill() -> QLabel:
    """Create a free‑standing BETA pill (caller positions it)."""
    lbl = QLabel("BETA")
    lbl.setObjectName("pillBeta")
    lbl.setFixedSize(40, 20)
    lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
    return lbl