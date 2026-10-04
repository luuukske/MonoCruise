import threading
import time
from types import SimpleNamespace

from PySide6.QtWidgets import QApplication

from ui.main_window.live_panel import LivePanel


def test_live_panel_updates_and_clears_stale_radar(monkeypatch):
    app = QApplication.instance() or QApplication([])
    data = SimpleNamespace(_lock=threading.Lock(), ego_x=0, ego_z=0, ego_y=0,
                           ego_yaw_rad=0, ego_speed=10, t_mono=time.monotonic(),
                           paused=False, vehicles=[SimpleNamespace(position=SimpleNamespace(x=5,z=-10),id=1)])
    radar = SimpleNamespace(data=data, is_alive=lambda: True)
    def get_thread(name):
        if name == 'radar_thread':
            return radar
        raise KeyError(name)
    monkeypatch.setattr('ui.main_window.live_panel.registry.get_thread', get_thread)
    panel = LivePanel()
    panel.show()
    app.processEvents()
    panel.refresh()
    assert panel.view.pose == (0,0,0)
    assert '36 km/h' in panel.status.text()
    assert len(panel.view.vehicles) == 1
    assert not panel.grab().isNull()
    data.t_mono = time.monotonic()-5
    panel.refresh()
    assert panel.view.pose is None
    assert not panel.view.vehicles
    assert 'No live game data' in panel.status.text()
    panel.timer.stop()
    panel.close()


def test_radar_view_draws_loaded_road_geometry():
    from ui.main_window.live_panel import RadarView

    app = QApplication.instance() or QApplication([])
    view = RadarView()
    view.resize(400, 400)
    view.pose = (0, 0, 0)
    view.roads = [((0, -60, 0), (0, 40, 0))]
    view.show()
    app.processEvents()
    image = view.grab().toImage()
    assert any(image.pixelColor(200, y).name() == '#75a9d5' for y in range(80, 240))
    view.close()
