"""Read-only road background and map controls for clip replay."""
from pathlib import Path

from PySide6.QtCore import QPointF, Qt, QTimer
from PySide6.QtGui import QColor, QPen
from PySide6.QtWidgets import QCheckBox, QFileDialog, QHBoxLayout, QLabel, QPushButton, QVBoxLayout, QWidget

from core.road_map.service import MapService


class RoadMapOverlay:
    def init_road_map(self):
        self.show_roads = True
        self.road_map = MapService()

    def _draw_map_background(self, painter, snap):
        if not self.show_roads:
            return
        painter.save()
        painter.setPen(QPen(QColor('#75a9d5'), 3))
        for a, b in self.road_map.preview_segments(snap.ego_x, snap.ego_z, 0):
            start = self._ws(a[0], a[1], snap.ego_x, snap.ego_z, snap.ego_yaw)
            end = self._ws(b[0], b[1], snap.ego_x, snap.ego_z, snap.ego_yaw)
            painter.drawLine(QPointF(*start), QPointF(*end))
        painter.setPen(QPen(QColor('#f1cc7a'), 2, Qt.PenStyle.DashLine))
        for a, b in self.road_map.preview_boundaries(snap.ego_x, snap.ego_z, 0):
            start = self._ws(a[0], a[1], snap.ego_x, snap.ego_z, snap.ego_yaw)
            end = self._ws(b[0], b[1], snap.ego_x, snap.ego_z, snap.ego_yaw)
            painter.drawLine(QPointF(*start), QPointF(*end))
        painter.setPen(QPen(QColor('#f39b66'), 2, Qt.PenStyle.DotLine))
        for a, b in self.road_map.preview_model_extents(snap.ego_x, snap.ego_z, 0, all_heights=True):
            start = self._ws(a[0], a[1], snap.ego_x, snap.ego_z, snap.ego_yaw)
            end = self._ws(b[0], b[1], snap.ego_x, snap.ego_z, snap.ego_yaw)
            painter.drawLine(QPointF(*start), QPointF(*end))
        painter.restore()

    def set_roads(self, on):
        self.show_roads = bool(on)
        self.update()


class RoadMapControls(QWidget):
    def __init__(self, scene):
        super().__init__()
        self.scene = scene
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        row = QHBoxLayout()
        toggle = QCheckBox('Show roads')
        toggle.setChecked(True)
        toggle.toggled.connect(scene.set_roads)
        row.addWidget(toggle)
        choose = QPushButton('Choose map JSON')
        choose.clicked.connect(self.choose_map)
        row.addWidget(choose)
        layout.addLayout(row)
        self.status = QLabel()
        self.status.setWordWrap(True)
        layout.addWidget(self.status)
        scene.road_map.load()
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.refresh)
        self.timer.start(500)
        self.refresh()

    def choose_map(self):
        path, _ = QFileDialog.getOpenFileName(self, 'Choose an ETS2 road map', '', 'JSON (*.json)')
        if path:
            self.scene.road_map.load(Path(path))
            self.refresh()

    def refresh(self):
        snap = self.scene._snap
        count = (len(self.scene.road_map.preview_segments(snap.ego_x, snap.ego_z, 0))
                 if snap is not None else 0)
        model_count = (len(self.scene.road_map.preview_model_extents(snap.ego_x, snap.ego_z, 0, all_heights=True))
                       if snap is not None else 0)
        self.status.setText(
            f'{self.scene.road_map.status} | Nearby road segments: {count}\n'
            f'Model surface candidates: {model_count} segments\n'
            'Height is unavailable in this replay; position-specific checks are in the live view.\n'
            'Blue: road centre. Dashed yellow: definition width extent; '
            'orange: unverified model surface, not a lane/shoulder boundary. '
            'Unknown models and junction interiors may be missing. Braking decisions are unchanged.')
        self.scene.update()
