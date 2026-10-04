"""Read-only live radar and road context on the main window."""
from __future__ import annotations

import math
import time

from PySide6.QtCore import QPointF, QRectF, Qt, QTimer
from PySide6.QtGui import QColor, QPainter, QPen
from PySide6.QtWidgets import QApplication, QCheckBox, QLabel, QPushButton, QVBoxLayout, QWidget

from core.road_map.service import service
from core.road_map.driven_routes import routes
from core.road_map.validation import description
from core.settings import Settings
from core.thread_management.registry import registry
from ui.main_window.map_preparation import MapPreparationButton


class RadarView(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(300, 260)
        self.pose = None
        self.vehicles = []
        self.roads = []
        self.boundaries = []
        self.model_extents = []
        self.driven = []
        self.elevation = 0

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setBrush(QColor('#111d30'))
        painter.setPen(Qt.PenStyle.NoPen)
        painter.drawRoundedRect(QRectF(self.rect()), 16, 16)
        cx, cy = self.width()/2, self.height()*0.72
        scale = min(self.width()/120, self.height()/120)
        painter.setPen(QPen(QColor('#273850'), 1))
        for radius in (20, 40, 60):
            painter.drawEllipse(QPointF(cx, cy), radius*scale, radius*scale)
        if self.pose is None:
            painter.setPen(QColor('#a1b3cc'))
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, 'Oyun verisi bekleniyor')
            return
        x, z, yaw = self.pose

        def screen(px, pz):
            dx, dz = px-x, pz-z
            right = math.cos(yaw)*dx - math.sin(yaw)*dz
            forward = -math.sin(yaw)*dx - math.cos(yaw)*dz
            return QPointF(cx+right*scale, cy-forward*scale)

        for a, b in self.roads:
            same_height = abs(self.elevation-(a[2]+b[2])/2) <= 8
            painter.setPen(QPen(QColor('#75a9d5' if same_height else '#485b73'),
                                3 if same_height else 1))
            painter.drawLine(screen(a[0], a[1]), screen(b[0], b[1]))
        painter.setPen(QPen(QColor('#f1cc7a'), 2, Qt.PenStyle.DashLine))
        for a, b in self.boundaries:
            painter.drawLine(screen(a[0], a[1]), screen(b[0], b[1]))
        painter.setPen(QPen(QColor('#f39b66'), 2, Qt.PenStyle.DotLine))
        for a, b in self.model_extents:
            painter.drawLine(screen(a[0], a[1]), screen(b[0], b[1]))
        painter.setPen(QPen(QColor('#c995eb'), 2, Qt.PenStyle.DotLine))
        for a, b in self.driven:
            painter.drawLine(screen(a[0], a[1]), screen(b[0], b[1]))
        for px, pz, identity in self.vehicles:
            point = screen(px, pz)
            if not self.rect().contains(point.toPoint()):
                continue
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(QColor('#f5b85b'))
            painter.drawRoundedRect(QRectF(point.x()-4, point.y()-7, 8, 14), 2, 2)
        painter.setBrush(QColor('#55d5c3'))
        painter.setPen(Qt.PenStyle.NoPen)
        painter.drawRoundedRect(QRectF(cx-6, cy-12, 12, 24), 3, 3)
        painter.setPen(QColor('#c5d5e9'))
        painter.drawText(14, 24, 'CANLI RADAR  /  metre')
        painter.drawText(14, self.height()-30, 'Blue: centre   Yellow: width   Purple: trace')
        painter.drawText(14, self.height()-14, 'Orange: model surface (not a lane boundary)')


class LivePanel(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 8, 12, 12)
        layout.setSpacing(12)
        title = QLabel('Driving view')
        title.setStyleSheet('font-size: 23px; font-weight: 600; color: #edf5ff;')
        layout.addWidget(title)
        subtitle = QLabel('Radar, harita ve AEB durumu tek ekranda')
        subtitle.setStyleSheet('color: #9aacc5;')
        layout.addWidget(subtitle)
        self.view = RadarView()
        layout.addWidget(self.view, 1)
        self.status = QLabel('Waiting for game connection')
        self.status.setWordWrap(True)
        self.status.setMinimumHeight(72)
        self.status.setStyleSheet('background: #192940; padding: 14px; border-radius: 12px;')
        layout.addWidget(self.status)
        self.map_status = QLabel(service.status)
        self.map_status.setWordWrap(True)
        self.map_status.setMinimumHeight(86)
        self.map_status.setStyleSheet('color: #9aacc5; font-size: 12px;')
        layout.addWidget(self.map_status)
        reload_button = QPushButton('Reload road map')
        reload_button.clicked.connect(lambda: service.load())
        layout.addWidget(reload_button)
        self.map_preparation = MapPreparationButton(self)
        layout.addWidget(self.map_preparation)
        self.route_recording = QCheckBox('Record my driven routes automatically (local)')
        self.route_recording.setChecked(Settings.driven_routes_enabled)
        self.route_recording.toggled.connect(self._toggle_routes)
        layout.addWidget(self.route_recording)
        routes.start()
        app = QApplication.instance()
        if app is not None:
            app.aboutToQuit.connect(routes.close)
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.refresh)
        self.timer.start(200)

    def _toggle_routes(self, enabled):
        Settings().driven_routes_enabled = bool(enabled)
        Settings().save()
        routes.reset()

    def _record_route(self):
        if not Settings.driven_routes_enabled or Settings.last_game != 1:
            routes.reset()
            return
        try:
            radar = registry.get_thread('radar_thread')
            if radar is None or not radar.is_alive():
                routes.reset()
                return
            with radar.data._lock:
                pose = (float(radar.data.ego_x), float(radar.data.ego_z), float(radar.data.ego_y),
                        float(radar.data.ego_speed), float(radar.data.t_mono))
                paused = bool(radar.data.paused)
            routes.observe(*pose, valid=not paused and 0 <= time.monotonic()-pose[4] <= 1)
        except (KeyError, AttributeError, TypeError, ValueError):
            routes.reset()

    def refresh(self):
        self._record_route()
        if not self.isVisible():
            return
        self.map_status.setText(service.status + '\nMap information is advisory; braking decisions are unchanged.')
        self.view.pose, self.view.vehicles, self.view.roads = None, [], []
        self.view.boundaries = []
        self.view.model_extents = []
        self.view.driven = []
        try:
            radar = registry.get_thread('radar_thread')
            if radar is None or not radar.is_alive():
                raise KeyError('radar offline')
            with radar.data._lock:
                x, z = float(radar.data.ego_x), float(radar.data.ego_z)
                elevation, yaw = float(radar.data.ego_y), float(radar.data.ego_yaw_rad)
                speed = float(radar.data.ego_speed)
                stamp = float(radar.data.t_mono)
                paused = bool(radar.data.paused)
                vehicles = [(float(v.position.x), float(v.position.z), v.id)
                            for v in radar.data.vehicles[:128]]
            if time.monotonic()-stamp > 1 or not all(math.isfinite(v) for v in (x,z,elevation,yaw,speed)):
                raise KeyError('stale radar')
            self.view.pose = (x, z, yaw)
            self.view.elevation = elevation
            self.view.vehicles = [(px,pz,identity) for px,pz,identity in vehicles
                                  if math.isfinite(px) and math.isfinite(pz)]
            self.view.roads = service.preview_segments(x, z, elevation)
            self.view.boundaries = service.preview_boundaries(x, z, elevation)
            self.view.model_extents = service.preview_model_extents(x, z, elevation)
            validation_note = description(service.validation_at(x, z, elevation, yaw))
            self.view.driven = routes.preview(x, z, elevation) if Settings.last_game == 1 else []
            count = len(self.view.roads)
            note = (f'Drawn road segments: {count}' if count else
                    'No road lines here; check the map source and position.')
            self.map_status.setText(service.status + f'\n{note}\n'
                                    f'Boundary segments: {len(self.view.boundaries)} | Position: X {x:.0f} / Z {z:.0f}\n'
                                    f'Model surface candidates: {len(self.view.model_extents)} segments (unverified)\n'
                                    f'Automatic check: {validation_note}\n'
                                    f'\nPurple trace: {len(self.view.driven)} segments | {routes.status}\n'
                                    'Driven traces are not road edges; braking decisions are unchanged.')
            state = 'AEB verisi bekleniyor'
            try:
                aeb = registry.get_thread('aeb_thread')
                if aeb is not None and aeb.is_alive():
                    with aeb.data._lock:
                        state = ('BRAKE INTERVENTION' if aeb.data.AEB_brake else
                                 'COLLISION WARNING' if aeb.data.AEB_warn else 'No intervention')
                        edges = getattr(aeb.data, 'road_edge_context', {}) or {}
                        outside = sum(v.get('state') == 'stationary_outside_candidate' for v in edges.values())
                        if edges:
                            state += f'\nStationary outside-boundary candidates: {outside} (braking unchanged)'
                            model_outside = sum(v.get('model_surface', {}).get('state') ==
                                                'stationary_outside_candidate' for v in edges.values())
                            state += f'\nOutside-model-surface candidates: {model_outside} (unverified)'
            except (KeyError, AttributeError):
                pass
            self.status.setText(f'{speed*3.6:.0f} km/h  •  {len(self.view.vehicles)} vehicles\n'
                                f'{"Game paused" if paused else state}')
        except (KeyError, AttributeError, TypeError, ValueError):
            self.status.setText('No live game data\nThe view starts when ETS2/TMP connects.')
        self.view.update()
