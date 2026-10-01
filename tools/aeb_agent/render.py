"""Offscreen PNG sequence of a clip's debug view, for video work. Dev only.

Renders the same top-down scene the review UI draws (`AEBDebugWindow`), one frame
per AEB tick, so a clip can become footage without re-driving it. Runs on the
offscreen Qt platform, so it needs no display.

The output is an abstraction of the recorded geometry: arcs, bodies and the AEB
state. It carries no screenshot, no player names and no place names.
"""

from __future__ import annotations

import os
from pathlib import Path

_FPS_DEFAULT = 30


def _system_font_dir() -> Path | None:
    """Platform font directory, derived from the environment, never hardcoded."""
    windir = os.environ.get("WINDIR")
    if windir:
        candidate = Path(windir) / "Fonts"
        return candidate if candidate.is_dir() else None
    for path in ("/usr/share/fonts", "/Library/Fonts"):
        if Path(path).is_dir():
            return Path(path)
    return None


def _ensure_fonts() -> int:
    """Register a few real fonts when the platform plugin exposes none.

    The offscreen plugin can come up with an empty font database, which draws
    every HUD string as tofu boxes. Rendered frames are for video, so unreadable
    text is a broken output rather than a cosmetic issue.
    """
    from PySide6.QtGui import QFontDatabase

    if QFontDatabase.families():
        return len(QFontDatabase.families())
    root = _system_font_dir()
    if root is None:
        return 0
    wanted = ("segoeui.ttf", "segoeuib.ttf", "arial.ttf", "arialbd.ttf",
              "DejaVuSans.ttf", "DejaVuSans-Bold.ttf")
    for name in wanted:
        hit = next(root.rglob(name), None)
        if hit is not None:
            QFontDatabase.addApplicationFont(str(hit))
    return len(QFontDatabase.families())


def _app():
    """A QApplication, preferring the native platform so real fonts are present."""
    from PySide6.QtWidgets import QApplication

    existing = QApplication.instance()
    app = existing if existing is not None else QApplication([])
    _ensure_fonts()
    return app


CREDIT_LINE = "anonymous contributed clip"
CREDIT_FILE = "CREDIT.txt"


def write_credit(out_dir: Path, clip_id: str) -> Path:
    """Drop the required on-screen credit beside the frames of a pulled clip.

    The credit is a condition of using contributed footage, and a sidecar in the
    render folder is what makes it hard to lose between here and the edit.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / CREDIT_FILE
    body = "\n".join([
        CREDIT_LINE,
        "",
        f"Rendered from contributed clip {clip_id[:8]}.",
        "It must carry the on-screen credit above wherever it is published.",
        "",
    ])
    path.write_text(body, encoding="utf-8")
    return path


def render_frames(clip, out_dir: Path, *, width: int = 1280, height: int = 720,
                  start_t: float | None = None, end_t: float | None = None,
                  vehicle_paths: bool = True, progress=None) -> dict:
    """Write one PNG per replayed tick in the window. Returns a summary dict."""
    from core.aeb.clip_replay import replay_clip

    from tools.aeb_review_widgets import SceneWidget

    _app()
    frames = replay_clip(clip)
    if not frames:
        return {"written": 0, "reason": "clip replayed to no frames"}
    lo = start_t if start_t is not None else frames[0].t_rel
    hi = end_t if end_t is not None else frames[-1].t_rel
    picked = [f for f in frames if lo <= f.t_rel <= hi]
    if not picked:
        return {"written": 0, "reason": f"no ticks between {lo} and {hi}"}

    out_dir.mkdir(parents=True, exist_ok=True)
    scene = SceneWidget()
    scene.set_vehicle_paths(vehicle_paths)
    scene.resize(width, height)
    written = 0
    for i, frame in enumerate(picked):
        scene.set_snapshot(frame.snapshot)
        scene.grab().save(str(out_dir / f"frame_{i:05d}.png"), "PNG")
        written += 1
        if progress is not None and written % 50 == 0:
            progress(f"  {written}/{len(picked)}")
    span = picked[-1].t_rel - picked[0].t_rel
    fps = (written / span) if span > 1e-6 else _FPS_DEFAULT
    return {
        "written": written,
        "dir": str(out_dir),
        "from_t": round(picked[0].t_rel, 2),
        "to_t": round(picked[-1].t_rel, 2),
        "native_fps": round(fps, 1),
        "ffmpeg": (f"ffmpeg -framerate {fps:.0f} -i frame_%05d.png "
                   f"-c:v libx264 -pix_fmt yuv420p out.mp4"),
    }
