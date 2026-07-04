"""
MonoCruise – Reusable widget factory functions.

Every factory inserts its widget into the *parent*'s QGridLayout at the
specified (row, col) and returns the widget reference.  The factories mirror
the original ``new_label``, ``new_checkbutton``, ``new_entry``,
``new_optionmenu`` helpers from MonoCruise.py.
"""

from __future__ import annotations

from typing import Any, Callable, Sequence

from PySide6.QtCore import Qt, QTimer, QPointF
from PySide6.QtGui import QColor, QPainter, QPainterPath, QPen
from PySide6.QtWidgets import (
    QCheckBox,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QSizePolicy,
    QStyle,
    QStyleOptionButton,
    QWidget,
)

from shared.dropdown import Dropdown
from ui.main_window.constants import (
    BG_COLOR,
    FONT_SIZE,
    RADIUS_INPUT,
    SETTINGS_COLOR,
    TEXT_COLOR,
    WAITING_COLOR,
)


# Internal helpers

def _grid(parent: QWidget) -> QGridLayout:
    layout = parent.layout()
    assert isinstance(layout, QGridLayout), "Parent widget must have a QGridLayout"
    return layout


# Label  (mirrors ``new_label``)

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


# Section header  (full‑width, spanning both columns)

def new_section_header(parent: QWidget, row: int, text: str) -> QLabel:
    lbl = QLabel(text)
    lbl.setObjectName("sectionHeader")
    lbl.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
    _grid(parent).addWidget(lbl, row, 0, 1, 2)
    return lbl


# Subtext / description

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


# Checkbox  (24×24, no text)

class CheckBox(QCheckBox):
    """QCheckBox that paints a checkmark glyph when checked.

    QSS styles the indicator box (border, fill) but cannot draw a vector
    checkmark without an image asset, so the glyph is painted on top here.
    """

    def paintEvent(self, event) -> None:
        super().paintEvent(event)
        if not self.isChecked():
            return
        opt = QStyleOptionButton()
        self.initStyleOption(opt)
        rect = self.style().subElementRect(
            QStyle.SubElement.SE_CheckBoxIndicator, opt, self
        )
        if rect.isEmpty():
            rect = self.rect()
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        pen = QPen(QColor("white"), 2.0)
        pen.setCapStyle(Qt.PenCapStyle.RoundCap)
        pen.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
        x, y, w, h = rect.x(), rect.y(), rect.width(), rect.height()
        path = QPainterPath()
        path.moveTo(QPointF(x + 0.28 * w, y + 0.54 * h))
        path.lineTo(QPointF(x + 0.44 * w, y + 0.70 * h))
        path.lineTo(QPointF(x + 0.73 * w, y + 0.34 * h))
        p.strokePath(path, pen)
        p.end()


def new_checkbutton(
    parent: QWidget,
    row: int,
    col: int,
    checked: bool = False,
    *,
    callback: Callable[..., Any] | None = None,
) -> QCheckBox:
    cb = CheckBox()
    cb.setFixedSize(24, 24)
    cb.setChecked(checked)
    if callback:
        cb.toggled.connect(callback)
    _grid(parent).addWidget(
        cb, row, col,
        alignment=Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter,
    )
    return cb


# Entry (QLineEdit) – 50 px wide, 2‑second debounce, min/max validation

def new_entry(
    parent: QWidget,
    row: int,
    col: int,
    *,
    value: Any = "",
    value_type: type = float,
    minimum: float | int | None = None,
    maximum: float | int | None = None,
    optional: bool = False,
    suffix: str | None = None,
    callback: Callable[[Any], Any] | None = None,
) -> QLineEdit:
    def _format_display(v: Any) -> str:
        if optional and v is None:
            return ""
        return str(v)

    le = QLineEdit(_format_display(value))
    # 50x30 to line up with the other input controls (bind buttons, dropdowns).
    le.setFixedSize(50, 30)
    le.setAlignment(Qt.AlignmentFlag.AlignCenter)

    # debounce timer (2 s, matching original ``master.after(2000, …)``)
    _debounce = QTimer()
    _debounce.setSingleShot(True)
    _debounce.setInterval(2000)

    _last_good = [value]  # mutable container so the closure can update it

    def _validate() -> None:
        raw = le.text().strip()
        if optional and not raw:
            v = None
            le.setText("")
            _last_good[0] = v
            le.clearFocus()
            if callback:
                callback(v)
            return
        try:
            v = value_type(raw)
        except (ValueError, TypeError):
            le.setText(_format_display(_last_good[0]))
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
            le.setText(_format_display(_last_good[0]))
            le.clearFocus()
        else:
            _orig_key_press(event)

    le.keyPressEvent = _key_override  # type: ignore[assignment]

    align = Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
    if suffix:
        wrapper = QWidget()
        wrapper.setObjectName("transparentRow")
        wrapper.setStyleSheet("QWidget#transparentRow { background: transparent; }")
        lay = QHBoxLayout(wrapper)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(4)
        lay.addStretch()
        lay.addWidget(le, alignment=Qt.AlignmentFlag.AlignVCenter)
        unit_lbl = QLabel(suffix)
        unit_lbl.setAlignment(Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft)
        lay.addWidget(unit_lbl, alignment=Qt.AlignmentFlag.AlignVCenter)
        _grid(parent).addWidget(wrapper, row, col, alignment=align)
    else:
        _grid(parent).addWidget(le, row, col, alignment=align)
    return le


# Option menu (shared custom-painted Dropdown, QComboBox-compatible API)

def new_optionmenu(
    parent: QWidget,
    row: int,
    col: int,
    values: Sequence[str],
    *,
    default: str = "",
    callback: Callable[[str], Any] | None = None,
) -> Dropdown:
    # Themed to match the other settings controls: input-field fill, 2px
    # border, 5px radius, blue hover accent, 13px text, 30px field height.
    combo = Dropdown(
        100,
        field_bg=BG_COLOR,
        border_w=2,
        border_color=SETTINGS_COLOR,
        border_hover=WAITING_COLOR,
        radius=RADIUS_INPUT,
        text_color=TEXT_COLOR,
        font_px=FONT_SIZE,
        pad_y=6,
    )
    combo.addItems([str(v) for v in values])
    if default and str(default) in [str(v) for v in values]:
        combo.setCurrentText(str(default))
    if callback:
        combo.currentTextChanged.connect(callback)

    _grid(parent).addWidget(
        combo, row, col,
        alignment=Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter,
    )
    return combo


# Button-binding configure button

class BindButton(QPushButton):
    """Click-to-assign binding button for the settings panel.

    Visual states are driven by dynamic QSS properties (see the
    ``QPushButton#bindButton`` rules in constants.STYLESHEET):

    - idle: input-field infill, 2px border; "None" in dark gray when unassigned
    - configuring: green border while listening for the next input
    - held: blue infill while the assigned input is physically pressed
    """

    _CONFIGURING_TEXT = "listening..."

    def __init__(self) -> None:
        super().__init__()
        self.setObjectName("bindButton")
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self._binding_text = "None"
        self._is_none = True
        self._state = "idle"
        self._hovered = False
        self.setProperty("bindState", self._state)
        self.setProperty("bindNone", True)
        self.setProperty("bindHover", False)
        self.setText(self._binding_text)

    # Hover is tracked explicitly via a dynamic property: the forced
    # repolish makes the hover restyle deterministic regardless of how the
    # ancestor chain is styled or composited.
    def enterEvent(self, event) -> None:
        super().enterEvent(event)
        self._hovered = True
        self._apply_properties()

    def leaveEvent(self, event) -> None:
        super().leaveEvent(event)
        self._hovered = False
        self._apply_properties()

    def set_binding_text(self, text: str, *, is_none: bool) -> None:
        """Set the displayed binding name; *is_none* renders it in dark gray."""
        self._binding_text = text
        self._is_none = is_none
        if self._state != "configuring":
            self.setText(text)
        self._apply_properties()

    def set_configuring(self, configuring: bool) -> None:
        if configuring:
            self._state = "configuring"
            self.setText(self._CONFIGURING_TEXT)
        else:
            if self._state != "configuring":
                return
            self._state = "idle"
            self.setText(self._binding_text)
        self._apply_properties()

    def set_held(self, held: bool) -> None:
        if self._state == "configuring":
            return
        new_state = "held" if held else "idle"
        if new_state == self._state:
            return
        self._state = new_state
        self._apply_properties()

    def _apply_properties(self) -> None:
        # Gray "None" text only in idle state: configuring shows the
        # listening hint and held shows white-on-blue.
        self.setProperty("bindState", self._state)
        self.setProperty("bindNone", self._is_none and self._state == "idle")
        self.setProperty("bindHover", self._hovered)
        style = self.style()
        style.unpolish(self)
        style.polish(self)
        self.update()


# Clickable label  (emulates the "button detection" click‑to‑assign pattern)

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


# BETA pill badge

def new_beta_pill() -> QLabel:
    """Create a free‑standing BETA pill (caller positions it)."""
    lbl = QLabel("BETA")
    lbl.setObjectName("pillBeta")
    lbl.setFixedSize(40, 20)
    lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
    return lbl