"""Clip to video: replay, resample, frame, paint, encode. Dev only, never shipped."""

from __future__ import annotations

import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from tools.clip_export import camera as camera_mod
from tools.clip_export.timeline import Timeline, aeb_state, auto_window, with_rerun_decisions

CREDIT_LINE = "anonymous contributed clip"


@dataclass(frozen=True)
class ExportOptions:
    width: int = 1920
    height: int = 1080
    fps: int = 60
    speed: float = 1.0
    start: float | None = None
    end: float | None = None
    hud: bool = True
    png: bool = False
    crf: int = 16
    recompute: bool = True


def ego_has_trailer(clip) -> bool:
    """Either flag: `ego_has_trailer` read constant False before schema v4 fixed it."""
    return any(f.ego.ego_has_trailer or f.ego.trailer_count > 0 for f in clip.radar_frames)


def build_timeline(clip, *, recompute: bool = True) -> Timeline:
    """Replayed clip; with ``recompute`` the AEB decisions are the working tree's, not the recording's."""
    from core.aeb.clip_replay import decode_radar_stream, replay_clip

    stream = decode_radar_stream(clip)
    recorded = replay_clip(clip, stream=stream)
    frames = recorded
    if recompute:
        from core.aeb.clip_eval import run_headless

        frames = with_rerun_decisions(recorded, run_headless(clip, stream=stream))
    label = clip.metadata.label
    timeline = Timeline(frames, ego_has_trailer=ego_has_trailer(clip),
                        label_vid=label.target_vid if label is not None else None)
    rec = {f.t_mono: aeb_state(f.live_aeb) for f in recorded}
    timeline.decisions = "recomputed" if recompute else "recorded"
    timeline.changed_ticks = sum(1 for f in frames if rec[f.t_mono] != aeb_state(f.live_aeb))
    timeline.recorded_brake_t = first_brake(recorded)
    timeline.shown_brake_t = first_brake(frames)
    return timeline


def first_brake(frames) -> float | None:
    return next((f.t_rel for f in frames if f.live_aeb.aeb_brake), None)


def open_loop_warning(timeline: Timeline) -> str | None:
    """Set when the video shows ego braking from a recorded AEB decision the re-run did not make."""
    rec, shown = timeline.recorded_brake_t, timeline.shown_brake_t
    if timeline.decisions != "recomputed" or rec is None:
        return None
    if shown is not None and shown <= rec + 0.1:
        return None
    when = "never" if shown is None else f"at {shown:.2f} s"
    return (f"recorded AEB braked at {rec:.2f} s, the current pipeline brakes {when}; ego motion "
            f"after {rec:.2f} s is the recorded braking, so the truck slows before the label "
            "says so. Use --recorded or another clip for footage.")


def output_times(timeline: Timeline, opts: ExportOptions) -> list[float]:
    """Output clock. The auto start waits for the threat to fit the widest shot (camera README)."""
    lo, hi = auto_window(timeline.frames)
    if opts.start is None:
        ticks = timeline.states(timeline.tick_ts, corridors=False)
        entry = camera_mod.entry_time(ticks, opts.width, opts.height, timeline.primary_span)
        if entry is not None:
            lo = max(lo, entry)
    lo = max(timeline.t_first, opts.start if opts.start is not None else lo)
    hi = min(timeline.t_last, opts.end if opts.end is not None else hi)
    if hi <= lo:
        raise ValueError(f"empty export window {lo:.2f}..{hi:.2f}")
    step = opts.speed / opts.fps
    n = int((hi - lo) / step) + 1
    return [lo + i * step for i in range(n)]


class FfmpegWriter:
    """Raw BGRA frames piped into x264, tagged BT.709 so players do not shift the colours."""

    def __init__(self, path: Path, width: int, height: int, fps: int, crf: int) -> None:
        exe = shutil.which("ffmpeg")
        if exe is None:
            raise FileNotFoundError("ffmpeg not on PATH; use --png for a frame sequence")
        if width % 2 or height % 2:
            raise ValueError("x264 yuv420p needs even width and height")
        pix = "bgra" if sys.byteorder == "little" else "argb"
        cmd = [
            exe, "-y", "-loglevel", "error",
            "-f", "rawvideo", "-pix_fmt", pix, "-s", f"{width}x{height}", "-r", str(fps),
            "-i", "-",
            "-vf", "scale=out_color_matrix=bt709:out_range=tv,format=yuv420p",
            "-c:v", "libx264", "-preset", "slow", "-tune", "animation", "-crf", str(crf),
            "-colorspace", "bt709", "-color_primaries", "bt709", "-color_trc", "bt709",
            "-color_range", "tv", "-movflags", "+faststart", str(path),
        ]
        self._proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
        self._size = width * height * 4

    def write(self, image) -> None:
        data = image.constBits().tobytes()
        if len(data) != self._size:
            raise ValueError("frame buffer size does not match the stream")
        self._proc.stdin.write(data)

    def close(self) -> None:
        self._proc.stdin.close()
        if self._proc.wait() != 0:
            raise RuntimeError(f"ffmpeg exited with {self._proc.returncode}")


@dataclass
class Prepared:
    """Everything a render loop needs, built once per clip."""

    timeline: Timeline
    times: list[float]
    states: list
    cams: list
    renderer: object


def prepare(clip, opts: ExportOptions, *, credit: str | None = None,
            times: list[float] | None = None, timeline: Timeline | None = None) -> Prepared:
    from tools.clip_export.painter import ShowcaseRenderer

    timeline = timeline if timeline is not None else build_timeline(clip, recompute=opts.recompute)
    times = times if times is not None else output_times(timeline, opts)
    states = timeline.states(times)
    cams = camera_mod.solve(states, opts.width, opts.height, timeline.primary_span)
    renderer = ShowcaseRenderer(opts.width, opts.height, origin=timeline.origin,
                                hud=opts.hud, credit=credit)
    return Prepared(timeline, times, states, cams, renderer)


def export_clip(clip, out: Path, opts: ExportOptions, *, credit: str | None = None,
                progress=None) -> dict:
    """Render ``clip`` to ``out`` (an .mp4, or a folder of PNGs with ``opts.png``)."""
    job = prepare(clip, opts, credit=credit)
    out = Path(out)
    writer = None
    if opts.png:
        out.mkdir(parents=True, exist_ok=True)
    else:
        out.parent.mkdir(parents=True, exist_ok=True)
        writer = FfmpegWriter(out, opts.width, opts.height, opts.fps, opts.crf)
    try:
        for i, (st, cam) in enumerate(zip(job.states, job.cams)):
            image = job.renderer.render(st, cam)
            if writer is not None:
                writer.write(image)
            else:
                image.save(str(out / f"frame_{i:05d}.png"), "PNG")
            if progress is not None and (i + 1) % 60 == 0:
                progress(f"  {i + 1}/{len(job.states)} frames")
    finally:
        if writer is not None:
            writer.close()
    return {
        "out": str(out),
        "frames": len(job.states),
        "from_t": round(job.times[0], 2),
        "to_t": round(job.times[-1], 2),
        "fps": opts.fps,
        "speed": opts.speed,
        "seconds": round(len(job.states) / opts.fps, 2),
        "primary_vid": job.timeline.primary_vid,
        "decisions": job.timeline.decisions,
        "ticks_differing_from_recording": job.timeline.changed_ticks,
        "recorded_first_brake_t": job.timeline.recorded_brake_t,
        "shown_first_brake_t": job.timeline.shown_brake_t,
        "warning": open_loop_warning(job.timeline),
        "credit": credit,
    }


def render_still(clip, t: float, out: Path, opts: ExportOptions, *, credit: str | None = None) -> dict:
    """One PNG at clip time ``t``, framed as it would be inside the export window.

    A ``t`` outside the window widens it to the whole clip so the camera still has context.
    """
    job = prepare(clip, opts, credit=credit)
    if not job.times[0] <= t <= job.times[-1]:
        whole = ExportOptions(fps=opts.fps, speed=opts.speed, start=-1e9, end=1e9)
        job = prepare(clip, opts, credit=credit, times=output_times(job.timeline, whole),
                      timeline=job.timeline)
    k = min(range(len(job.times)), key=lambda i: abs(job.times[i] - t))
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    job.renderer.render(job.states[k], job.cams[k]).save(str(out), "PNG")
    return {"out": str(out), "t": round(job.times[k], 2), "primary_vid": job.timeline.primary_vid,
            "decisions": job.timeline.decisions}
