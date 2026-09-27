"""Live lane map while driving. Separate from the ACC debug window.

Run from the repo root, with the game already in a truck:

  python tools/road_layout_view.py

North is up. The truck stays centred. A sector is read when the view can
reach it, and dropped after the truck has moved on.
"""

from __future__ import annotations

import sys
from pathlib import Path

from PySide6.QtWidgets import QApplication

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.road_layout.view import LaneMapWindow


def main() -> int:
    app = QApplication(sys.argv)
    window = LaneMapWindow()
    window.show()
    window.start()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
