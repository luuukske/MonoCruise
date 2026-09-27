"""Standalone top-down lane map. Not part of the ACC debug window yet.

North is up and the truck stays in the centre. Sectors load on a side thread
when the view can reach them.
"""

from __future__ import annotations

import logging
import math

from PySide6.QtCore import QObject, QPointF, QThread, Qt, QTimer, Signal, Slot
from PySide6.QtGui import QColor, QFont, QPainter, QPaintEvent, QPen, QPolygonF
from PySide6.QtWidgets import QWidget

from core.radar.scs_pose import ScsPose, ScsPoseReader
from core.road_layout.live import LiveMap, MapFrame, MapLine

log = logging.getLogger("road_layout")

_BG = QColor(15, 15, 20)
_GRID = QColor(40, 40, 50)
_ROAD = QColor(224, 154, 69)
_JUNCTION = QColor(106, 168, 224)
_TRUCK = QColor(242, 242, 242)
_TEXT = QColor(215, 215, 215)
_MUTED = QColor(150, 150, 160)
_VIEW_M = 200.0
_TICK_MS = 50
_MOVE_M = 50.0


class _Worker(QObject):
    ready = Signal(object)
    status = Signal(str)
    failed = Signal(str)

    def __init__(self) -> None:
        super().__init__()
        self._map: LiveMap | None = None

    @Slot()
    def open_catalogs(self) -> None:
        self.status.emit("Indexing the map")
        try:
            self._map = LiveMap.open_installed()
        except FileNotFoundError as exc:
            log.warning("lane map index failed: %s", exc)
            self.failed.emit(str(exc))
            return
        except OSError:
            log.exception("lane map index failed")
            self.failed.emit("Could not open the game map.")
            return
        self.status.emit("Waiting for the game")

    @Slot(float, float)
    def on_pose(self, x: float, z: float) -> None:
        if self._map is None:
            return
        if self._map.stale(x, z):
            self.status.emit("Loading sector")
        try:
            frame = self._map.frame_for(x, z)
        except Exception:
            log.exception("lane sector load failed")
            self.status.emit("Sector load failed")
            return
        if frame is not None:
            self.ready.emit(frame)
        self.status.emit("")


class LaneMapWindow(QWidget):
    """Ego-centred lane map. Call ``start`` once the widget is shown."""

    _ask = Signal(float, float)

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Lane map")
        self.resize(960, 720)
        self.setFocusPolicy(Qt.FocusPolicy.WheelFocus)
        self._font = QFont()
        self._font.setPointSize(10)
        self._reader = ScsPoseReader()
        self._pose: ScsPose | None = None
        self._frame: MapFrame | None = None
        self._status = "Indexing the map"
        self._catalog_ready = False
        self._zoom = 1.0
        self._req: tuple[float, float] | None = None
        self._thread = QThread(self)
        self._worker = _Worker()
        self._worker.moveToThread(self._thread)
        self._thread.started.connect(self._worker.open_catalogs)
        self._ask.connect(self._worker.on_pose)
        self._worker.ready.connect(self._on_frame)
        self._worker.status.connect(self._on_status)
        self._worker.failed.connect(self._on_failed)
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._tick)

    def start(self) -> None:
        self._thread.start()
        self._timer.start(_TICK_MS)

    def closeEvent(self, event) -> None:
        self._timer.stop()
        self._thread.quit()
        self._thread.wait(3000)
        self._reader.close()
        super().closeEvent(event)

    def wheelEvent(self, event) -> None:
        step = 1.12 if event.angleDelta().y() > 0 else 1 / 1.12
        self._zoom = min(6.0, max(0.35, self._zoom * step))
        self.update()

    def _on_frame(self, frame: object) -> None:
        if isinstance(frame, MapFrame):
            self._frame = frame
            self.update()

    def _on_status(self, text: str) -> None:
        if text == "Waiting for the game":
            self._catalog_ready = True
            self._status = ""
        elif text:
            self._status = text
        elif self._catalog_ready:
            self._status = ""
        self.update()

    def _on_failed(self, text: str) -> None:
        self._status = text
        self._catalog_ready = False
        self.update()

    def _tick(self) -> None:
        pose = self._reader.read()
        self._pose = pose
        if pose is not None and self._catalog_ready:
            moved = (
                self._req is None
                or math.hypot(pose.x - self._req[0], pose.z - self._req[1]) > _MOVE_M
            )
            if moved:
                self._req = (pose.x, pose.z)
                self._ask.emit(pose.x, pose.z)
        elif pose is None and self._catalog_ready:
            self._status = "Waiting for the game"
        self.update()

    def paintEvent(self, event: QPaintEvent) -> None:
        del event
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        painter.fillRect(self.rect(), _BG)
        painter.setFont(self._font)
        if self._pose is None or self._frame is None:
            painter.setPen(QPen(_TEXT))
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, self._status or "Waiting for the game")
            painter.end()
            return
        self._draw_map(painter, self._pose, self._frame)
        self._draw_hud(painter, self._pose, self._frame)
        painter.end()

    def _ppm(self) -> float:
        short = max(1.0, float(min(self.width(), self.height())))
        return short * 0.5 / _VIEW_M * self._zoom

    def _to_screen(self, x: float, z: float, pose: ScsPose, ppm: float) -> QPointF:
        east = x - pose.x
        north = -(z - pose.z)
        return QPointF(self.width() * 0.5 + east * ppm, self.height() * 0.5 - north * ppm)

    def _draw_map(self, painter: QPainter, pose: ScsPose, frame: MapFrame) -> None:
        ppm = self._ppm()
        half_x = self.width() * 0.5 / ppm + 30.0
        half_z = self.height() * 0.5 / ppm + 30.0
        self._draw_grid(painter, pose, ppm)
        road_pen = QPen(_ROAD, 1.6)
        junction_pen = QPen(_JUNCTION, 1.6)
        for line in frame.lines:
            if line.max_x < pose.x - half_x or line.min_x > pose.x + half_x:
                continue
            if line.max_z < pose.z - half_z or line.min_z > pose.z + half_z:
                continue
            if len(line.pts) < 2:
                continue
            painter.setPen(junction_pen if line.kind == "prefab" else road_pen)
            painter.drawPolyline(QPolygonF([
                self._to_screen(x, z, pose, ppm) for x, z in line.pts
            ]))
        self._draw_truck(painter, pose, ppm)

    def _draw_grid(self, painter: QPainter, pose: ScsPose, ppm: float) -> None:
        painter.setPen(QPen(_GRID, 1.0))
        reach_x = self.width() * 0.5 / ppm
        reach_z = self.height() * 0.5 / ppm
        step = 50.0
        x0 = math.floor((pose.x - reach_x) / step) * step
        x1 = pose.x + reach_x
        z0 = math.floor((pose.z - reach_z) / step) * step
        z1 = pose.z + reach_z
        z = z0
        while z <= z1:
            painter.drawLine(self._to_screen(pose.x - reach_x, z, pose, ppm), self._to_screen(pose.x + reach_x, z, pose, ppm))
            z += step
        x = x0
        while x <= x1:
            painter.drawLine(self._to_screen(x, pose.z - reach_z, pose, ppm), self._to_screen(x, pose.z + reach_z, pose, ppm))
            x += step

    def _draw_truck(self, painter: QPainter, pose: ScsPose, ppm: float) -> None:
        yaw = pose.yaw_norm * 2.0 * math.pi
        fwd_x = -math.sin(yaw)
        fwd_z = -math.cos(yaw)
        tip = self._to_screen(pose.x + fwd_x * 16.0, pose.z + fwd_z * 16.0, pose, ppm)
        center = self._to_screen(pose.x, pose.z, pose, ppm)
        painter.setPen(QPen(_TRUCK, 2.0))
        painter.drawLine(center, tip)
        painter.setBrush(_TRUCK)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.drawEllipse(center, 4.0, 4.0)

    def _draw_hud(self, painter: QPainter, pose: ScsPose, frame: MapFrame) -> None:
        dist, kind = _nearest(frame.lines, pose.x, pose.z)
        speed = pose.speed * 3.6
        where = "junction" if kind == "prefab" else "road lane" if kind else "none"
        dist_txt = "n/a" if dist is None else f"{dist:.1f} m {where}"
        lines = [
            "North up. Scroll to zoom.",
            f"{speed:.0f} km/h" if speed >= 0.5 else "Stopped",
            f"Nearest lane {dist_txt}",
            f"Sectors {frame.sectors}",
        ]
        if self._status:
            lines.append(self._status)
        painter.setPen(QPen(_MUTED))
        y = 16
        for text in lines:
            painter.setPen(QPen(_TEXT if text == lines[1] else _MUTED))
            painter.drawText(16, y, text)
            y += 18


def _nearest(lines: tuple[MapLine, ...], x: float, z: float) -> tuple[float | None, str]:
    best: float | None = None
    kind = ""
    for line in lines:
        if x < line.min_x - 25.0 or x > line.max_x + 25.0:
            continue
        if z < line.min_z - 25.0 or z > line.max_z + 25.0:
            continue
        pts = line.pts
        for i in range(len(pts) - 1):
            dist = _segment(x, z, pts[i], pts[i + 1])
            if best is None or dist < best:
                best = dist
                kind = line.kind
    return best, kind


def _segment(x: float, z: float, a: tuple[float, float], b: tuple[float, float]) -> float:
    dx = b[0] - a[0]
    dz = b[1] - a[1]
    length2 = dx * dx + dz * dz
    if length2 < 1e-12:
        return math.hypot(x - a[0], z - a[1])
    t = max(0.0, min(1.0, ((x - a[0]) * dx + (z - a[1]) * dz) / length2))
    return math.hypot(x - (a[0] + t * dx), z - (a[1] + t * dz))
