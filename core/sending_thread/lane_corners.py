"""Red bottom screen corner on the side LDW/LKA acted on; part of the live bottom bar. See README, Visualization bar."""

from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QColor, QLinearGradient, QPainter
from PySide6.QtWidgets import QWidget

from core.lane_departure.display import LEFT

# Each leg runs this share of the screen height from the corner; the first part stays fully red, then fades.
EXTENT_OF_HEIGHT = 0.25
SOLID_SHARE = 0.40
THICKNESS_PX = 5
_RED = QColor("red")
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
        self.weight = 0.0
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

    def set_weight(self, weight: float) -> None:
        weight = min(1.0, max(0.0, weight))
        if weight != self.weight:
            self.weight = weight
            self.update()

    def _gradient(self, start: QPointF, end: QPointF) -> QLinearGradient:
        g = QLinearGradient(start, end)
        for i in range(_FADE_STOPS + 1):
            f = SOLID_SHARE + (1.0 - SOLID_SHARE) * i / _FADE_STOPS
            c = QColor(_RED)
            c.setAlphaF(self.weight * _alpha_at(f))
            g.setColorAt(f, c)
        c = QColor(_RED)
        c.setAlphaF(self.weight)
        g.setColorAt(0.0, c)
        return g

    def paintEvent(self, _event):
        if self.weight <= 0.0:
            return
        e, t = float(self.extent), float(THICKNESS_PX)
        corner_x, far_x = (0.0, e) if self.side == LEFT else (e, 0.0)
        side_x = 0.0 if self.side == LEFT else e - t
        painter = QPainter(self)
        # The bottom leg owns the corner square; the side leg starts above it so a fade never doubles up there.
        painter.fillRect(QRectF(0.0, e - t, e, t), self._gradient(QPointF(corner_x, 0.0), QPointF(far_x, 0.0)))
        painter.fillRect(QRectF(side_x, 0.0, t, e - t), self._gradient(QPointF(0.0, e), QPointF(0.0, 0.0)))
        painter.end()
