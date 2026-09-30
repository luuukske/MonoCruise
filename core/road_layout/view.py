"""Standalone top-down lane map. Not part of the ACC debug window yet.

North is up and the truck stays in the centre. Sectors load on a side thread
when the view can reach them. The bright line is the lane the truck is in, drawn forward until just before
the sixth prefab, or 200 m. The pink line is the road model.
Map objects out to 400 m each have their own color. Point at one to read its name.
"""

from __future__ import annotations

import logging
import math

from PySide6.QtCore import QObject, QPointF, QThread, Qt, QTimer, Signal, Slot
from PySide6.QtGui import QColor, QFont, QFontMetrics, QPainter, QPaintEvent, QPen, QPolygonF
from PySide6.QtWidgets import QWidget

from core.acc.lane_direction import LaneDirection, centreline_world, lane_direction_samples
from core.acc.road_model import fit_road_model
from core.radar.scs_pose import ScsPose, ScsPoseReader
from core.road_layout.ego_lane import (
    EgoLaneChoice,
    EgoLaneTrack,
    LineIndex,
    extend_lane,
    path_curvature,
    ranked_lanes,
)
from core.road_layout.live import LiveMap, MapFrame, MapLine

log = logging.getLogger("road_layout")

_BG = QColor(15, 15, 20)
_GRID = QColor(40, 40, 50)
_ROAD = QColor(224, 154, 69)
_JUNCTION = QColor(106, 168, 224)
_CHOSEN = QColor(190, 235, 130)
_MODEL = QColor(255, 120, 210)
_TRUCK = QColor(242, 242, 242)
_TEXT = QColor(215, 215, 215)
_MUTED = QColor(150, 150, 160)
_VIEW_M = 200.0
_PROP_M = 400.0
_PICK_PX = 12.0
_TICK_MS = 50
_MOVE_M = 50.0
_HOLD_M = 2.5
_HOLD_DOT = 0.992


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
        self.setMouseTracking(True)
        self._font = QFont()
        self._font.setPointSize(10)
        self._reader = ScsPoseReader()
        self._pose: ScsPose | None = None
        self._frame: MapFrame | None = None
        self._track = EgoLaneTrack()
        self._near: LineIndex | None = None
        self._near_frame: MapFrame | None = None
        self._pts: tuple[tuple[tuple[float, float], ...], ...] = ()
        self._held: tuple | None = None
        self._model: tuple[tuple[float, float], ...] = ()
        self._status = "Indexing the map"
        self._catalog_ready = False
        self._zoom = 1.0
        self._pointer: QPointF | None = None
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

    def mouseMoveEvent(self, event) -> None:
        self._pointer = event.position()
        self.update()

    def leaveEvent(self, event) -> None:
        self._pointer = None
        self.update()
        super().leaveEvent(event)

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
        choice, chain = self._highlight(self._pose, self._frame)
        self._draw_map(painter, self._pose, self._frame, chain, self._model)
        self._draw_hud(painter, self._pose, self._frame, choice)
        painter.end()

    def _ppm(self) -> float:
        short = max(1.0, float(min(self.width(), self.height())))
        return short * 0.5 / _VIEW_M * self._zoom

    def _to_screen(self, x: float, z: float, pose: ScsPose, ppm: float) -> QPointF:
        east = x - pose.x
        north = -(z - pose.z)
        return QPointF(self.width() * 0.5 + east * ppm, self.height() * 0.5 - north * ppm)

    def _highlight(self, pose: ScsPose, frame: MapFrame):
        fx, fz = _nose(pose)
        held = self._held
        if (
            held is not None
            and held[0] is frame
            and math.hypot(pose.x - held[1], pose.z - held[2]) < _HOLD_M
            and fx * held[3] + fz * held[4] > _HOLD_DOT
        ):
            self._track.note(pose.x, pose.z)
            return held[5], held[6]
        near = self._line_index(frame)
        choice = self._track.choose(self._pts, pose.x, pose.z, fx, fz, near=near)
        chain: tuple = ()
        if choice is not None:
            chain = extend_lane(frame.lines, choice.index, pose.x, pose.z, fx, fz, near)
        self._model = _road_model_line(
            self._track, self._pts, frame.lines, near, pose, fx, fz, choice, chain,
        )
        self._held = (frame, pose.x, pose.z, fx, fz, choice, chain)
        return choice, chain

    def _line_index(self, frame: MapFrame) -> LineIndex:
        if self._near_frame is not frame or self._near is None:
            self._pts = tuple(line.pts for line in frame.lines)
            self._near = LineIndex(frame.lines)
            self._near_frame = frame
        return self._near

    def _draw_map(
        self,
        painter: QPainter,
        pose: ScsPose,
        frame: MapFrame,
        chain: tuple,
        model: tuple[tuple[float, float], ...],
    ) -> None:
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
        self._draw_props(painter, pose, frame, ppm)
        if chain:
            painter.setPen(QPen(_CHOSEN, 3.0))
            for span in chain:
                if len(span.pts) < 2:
                    continue
                painter.drawPolyline(QPolygonF([
                    self._to_screen(x, z, pose, ppm) for x, z in span.pts
                ]))
        if len(model) >= 2:
            painter.setPen(QPen(_MODEL, 2.6))
            painter.drawPolyline(QPolygonF([
                self._to_screen(x, z, pose, ppm) for x, z in model
            ]))
        self._draw_truck(painter, pose, ppm)

    def _draw_props(self, painter: QPainter, pose: ScsPose, frame: MapFrame, ppm: float) -> None:
        reach = _PROP_M
        reach2 = reach * reach
        for line in frame.props:
            if line.max_x < pose.x - reach or line.min_x > pose.x + reach:
                continue
            if line.max_z < pose.z - reach or line.min_z > pose.z + reach:
                continue
            if not line.color:
                continue
            painter.setPen(QPen(QColor(line.color), 2.2))
            run: list[QPointF] = []
            for x, z in line.pts:
                dx = x - pose.x
                dz = z - pose.z
                if dx * dx + dz * dz > reach2:
                    if len(run) >= 2:
                        painter.drawPolyline(QPolygonF(run))
                    run = []
                    continue
                run.append(self._to_screen(x, z, pose, ppm))
            if len(run) >= 2:
                painter.drawPolyline(QPolygonF(run))
        painter.setPen(Qt.PenStyle.NoPen)
        for dot in frame.dots:
            dx = dot.x - pose.x
            dz = dot.z - pose.z
            if dx * dx + dz * dz > reach2:
                continue
            if not dot.color:
                continue
            painter.setBrush(QColor(dot.color))
            radius = 3.2
            painter.drawEllipse(self._to_screen(dot.x, dot.z, pose, ppm), radius, radius)

    def _draw_pick(self, painter: QPainter, pose: ScsPose, frame: MapFrame, ppm: float) -> None:
        text = self._pick_label(pose, frame, ppm)
        color = _TEXT
        if text.startswith("#") and len(text) >= 7:
            color = QColor(text[:7])
        shown = QFontMetrics(self._font).elidedText(
            text, Qt.TextElideMode.ElideMiddle, max(40, self.width() - 32),
        )
        painter.setPen(QPen(color))
        painter.drawText(16, self.height() - 16, shown)

    def _pick_label(self, pose: ScsPose, frame: MapFrame, ppm: float) -> str:
        if self._pointer is None:
            return "Point at an object to read its name and color."
        px, py = self._pointer.x(), self._pointer.y()
        limit = _PICK_PX * _PICK_PX
        best = limit
        found = ""
        reach2 = _PROP_M * _PROP_M
        for dot in frame.dots:
            dx = dot.x - pose.x
            dz = dot.z - pose.z
            if dx * dx + dz * dz > reach2 or not dot.label:
                continue
            screen = self._to_screen(dot.x, dot.z, pose, ppm)
            dist = (screen.x() - px) ** 2 + (screen.y() - py) ** 2
            if dist < best:
                best = dist
                found = f"{dot.color}  {dot.label}"
        wx, wz = self._cursor_world(pose, ppm)
        pad = _PICK_PX / max(ppm, 0.05)
        for line in frame.props:
            if not line.label or line.max_x < wx - pad or line.min_x > wx + pad:
                continue
            if line.max_z < wz - pad or line.min_z > wz + pad:
                continue
            previous: QPointF | None = None
            for x, z in line.pts:
                screen = self._to_screen(x, z, pose, ppm)
                if previous is not None:
                    dist = _segment_dist2(px, py, previous.x(), previous.y(), screen.x(), screen.y())
                    if dist < best:
                        best = dist
                        found = f"{line.color}  {line.label}"
                previous = screen
        return found or "Point at an object to read its name and color."

    def _cursor_world(self, pose: ScsPose, ppm: float) -> tuple[float, float]:
        if self._pointer is None:
            return pose.x, pose.z
        east = (self._pointer.x() - self.width() * 0.5) / ppm
        north = (self.height() * 0.5 - self._pointer.y()) / ppm
        return pose.x + east, pose.z - north

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
        fwd_x, fwd_z = _nose(pose)
        tip = self._to_screen(pose.x + fwd_x * 16.0, pose.z + fwd_z * 16.0, pose, ppm)
        center = self._to_screen(pose.x, pose.z, pose, ppm)
        painter.setPen(QPen(_TRUCK, 2.0))
        painter.drawLine(center, tip)
        painter.setBrush(_TRUCK)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.drawEllipse(center, 4.0, 4.0)

    def _draw_hud(
        self,
        painter: QPainter,
        pose: ScsPose,
        frame: MapFrame,
        choice: EgoLaneChoice | None,
    ) -> None:
        speed = pose.speed * 3.6
        lines = [
            "North up. Scroll to zoom. Objects to 400 m.",
            "Pink line is the road model.",
            f"{speed:.0f} km/h" if speed >= 0.5 else "Stopped",
            _lane_text(frame.lines, choice),
            f"Sectors {frame.sectors}",
        ]
        if self._status:
            lines.append(self._status)
        painter.setPen(QPen(_MUTED))
        y = 16
        for text in lines:
            painter.setPen(QPen(
                _MODEL if text == lines[1] else _TEXT if text == lines[2] else _MUTED
            ))
            painter.drawText(16, y, text)
            y += 18
        self._draw_pick(painter, pose, frame, self._ppm())


def _segment_dist2(px: float, py: float, ax: float, ay: float, bx: float, by: float) -> float:
    dx = bx - ax
    dy = by - ay
    length2 = dx * dx + dy * dy
    if length2 < 1e-6:
        return (px - ax) ** 2 + (py - ay) ** 2
    t = max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / length2))
    return (px - (ax + t * dx)) ** 2 + (py - (ay + t * dy)) ** 2


def _nose(pose: ScsPose) -> tuple[float, float]:
    yaw = pose.yaw_norm * 2.0 * math.pi
    return -math.sin(yaw), -math.cos(yaw)


def _road_model_line(track, pts, lines, near, pose, fx, fz, choice, chain):
    """World points of the fitted centreline. Reuses the lane already extended."""
    kappa = path_curvature(track.points())
    ranked = ranked_lanes(pts, pose.x, pose.z, fx, fz, near, kappa)
    routes = []
    for index, dist, cost in ranked:
        spans = chain if choice is not None and index == choice.index else extend_lane(
            lines, index, pose.x, pose.z, fx, fz, near,
        )
        route = _join_spans(spans)
        if len(route) >= 2:
            routes.append(LaneDirection(route, dist, cost))
    samples = lane_direction_samples(routes, pose.x, pose.z, fx, fz)
    ego = []
    for hx, hz in track.points():
        dx = hx - pose.x
        dz = hz - pose.z
        ego.append((dx * fx + dz * fz, dx * (-fz) + dz * fx))
    model = fit_road_model(ego, [], kappa, samples or None)
    return centreline_world(model, pose.x, pose.z, fx, fz)


def _join_spans(spans) -> tuple[tuple[float, float], ...]:
    pts: list[tuple[float, float]] = []
    for span in spans:
        chunk = list(span.pts)
        if len(chunk) < 2:
            continue
        if pts:
            last = pts[-1]
            if _gap(chunk[-1], last) < _gap(chunk[0], last):
                chunk.reverse()
            if _gap(chunk[0], last) < 0.5:
                chunk = chunk[1:]
        pts.extend(chunk)
    return tuple(pts)


def _gap(a, b) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _lane_text(lines: tuple[MapLine, ...], choice: EgoLaneChoice | None) -> str:
    if choice is None or choice.index < 0 or choice.index >= len(lines):
        return "No lane"
    kind = lines[choice.index].kind
    where = "junction" if kind == "prefab" else "road lane"
    return f"{where} {choice.distance_m:.1f} m, {math.degrees(choice.angle_rad):.0f}°"
