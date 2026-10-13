"""Live lane map while driving. Separate from the ACC debug window.

Run from the repo root, with the game already in a truck:

  python tools/road_layout_view.py

North is up. The truck stays centred. The bright line is the lane it is in,
drawn forward until just before the sixth prefab, or 200 m. The pink line
is the road model. Other map objects out to 400 m each have their own color.
Point at one and the bottom line shows its name and hex color.
A sector is read when the view can reach it, and dropped after the truck has
moved on.
"""

from __future__ import annotations

import sys
from pathlib import Path

from PySide6.QtWidgets import QApplication

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.road_layout.view import LaneMapWindow
from tools.map_obstacle_props import ObstacleProps


def main() -> int:
    app = QApplication(sys.argv)
    window = LaneMapWindow(props_factory=ObstacleProps)
    window.show()
    window.start()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
