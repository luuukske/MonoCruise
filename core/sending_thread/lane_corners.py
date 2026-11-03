"""Bottom screen corner on the side LDW/LKA acted on, orange or blinking red; part of the live bottom bar.

See README, Visualization bar.
"""

from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QColor, QLinearGradient, QPainter
from PySide6.QtWidgets import QWidget

from core.lane_departure.display import ALERT_COLOR, LEFT, WARN_COLOR

# Each leg runs this share of the screen height from the corner; the first part stays fully red, then fades.
EXTENT_OF_HEIGHT = 0.25
SOLID_SHARE = 0.40
THICKNESS_PX = 5
_FADE_STOPS = 8


def _alpha_at(f: float) -> float:
    """Opacity along a leg, 0 at the corner and 1 at the far end."""
    if f <= SOLID_SHARE:
        return 1.0
    t = min(1.0, (f - SOLID_SHARE) / (1.0 - SOLID_SHARE))
    return 1.0 - t * t * (3.0 - 2.0 * t)


class LaneCorner(QWidget):
    """One bottom corner: a leg along the bottom edge and one up the side, transparent and click-through."""

    def __init__(self, side: int, parent: QWidget | None = None):
        super().__init__(parent)
        self.side = side
        self.warn = 0.0
        self.alert = 0.0
        self.extent = 1
        self.setWindowFlags(
            Qt.FramelessWindowHint
            | Qt.WindowStaysOnTopHint
            | Qt.Tool
            | Qt.WindowDoesNotAcceptFocus
        )
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.setAttribute(Qt.WA_NoSystemBackground)
        self.setAttribute(Qt.WA_TransparentForMouseEvents)
        self.setAttribute(Qt.WA_ShowWithoutActivating)

    def place(self, left: int, top: int, width: int, height: int) -> None:
        """Fit the corner to a screen geometry (the bar's screen)."""
        self.extent = max(1, int(round(height * EXTENT_OF_HEIGHT)))
        x = left if self.side == LEFT else left + width - self.extent
        self.setGeometry(x, top + height - self.extent, self.extent, self.extent)

    def set_weights(self, warn: float, alert: float) -> None:
        """How much of the orange warning and of the red alert shows, each 0 to 1 (the alert's blink applied)."""
        warn, alert = min(1.0, max(0.0, warn)), min(1.0, max(0.0, alert))
        if (warn, alert) != (self.warn, self.alert):
            self.warn, self.alert = warn, alert
            self.update()

    @staticmethod
    def _gradient(start: QPointF, end: QPointF, color: QColor, weight: float) -> QLinearGradient:
        g = QLinearGradient(start, end)
        for i in range(_FADE_STOPS + 1):
            f = SOLID_SHARE + (1.0 - SOLID_SHARE) * i / _FADE_STOPS
            c = QColor(color)
            c.setAlphaF(weight * _alpha_at(f))
            g.setColorAt(f, c)
        c = QColor(color)
        c.setAlphaF(weight)
        g.setColorAt(0.0, c)
        return g

    def paintEvent(self, _event):
        layers = [(QColor(code), w) for code, w in ((WARN_COLOR, self.warn), (ALERT_COLOR, self.alert)) if w > 0.0]
        if not layers:
            return
        e, t = float(self.extent), float(THICKNESS_PX)
        corner_x, far_x = (0.0, e) if self.side == LEFT else (e, 0.0)
        side_x = 0.0 if self.side == LEFT else e - t
        painter = QPainter(self)
        for color, weight in layers:
            # The bottom leg owns the corner square; the side leg starts above it so a fade never doubles up there.
            along = self._gradient(QPointF(corner_x, 0.0), QPointF(far_x, 0.0), color, weight)
            up = self._gradient(QPointF(0.0, e), QPointF(0.0, 0.0), color, weight)
            painter.fillRect(QRectF(0.0, e - t, e, t), along)
            painter.fillRect(QRectF(side_x, 0.0, t, e - t), up)
        painter.end()
