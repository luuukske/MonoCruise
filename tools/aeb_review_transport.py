"""Playback and the frame readout for the AEB clip review window. Not shipped."""
from __future__ import annotations

from tools.aeb_review_widgets import _fmt


class ReviewTransport:
    """Frame stepping, play, seek and the per-frame decision line. Mixed into ReviewWindow."""

    def _cur_t(self) -> float:
        return self._frames[self._idx].t_rel if self._frames else 0.0

    def _toggle_play(self) -> None:
        if self._play_timer.isActive():
            self._play_timer.stop()
            self._play_btn.setText("Play")
        elif self._frames:
            self._play_timer.start(33)
            self._play_btn.setText("Pause")

    def _advance(self) -> None:
        if self._idx >= len(self._frames) - 1:
            self._play_timer.stop()
            self._play_btn.setText("Play")
            return
        self._idx += 1
        self._refresh()

    def _step(self, delta: int) -> None:
        if not self._frames:
            return
        self._idx = max(0, min(len(self._frames) - 1, self._idx + delta))
        self._refresh()

    def _first(self) -> None:
        self._step(-len(self._frames))

    def _last(self) -> None:
        self._step(len(self._frames))

    def _prev(self) -> None:
        self._step(-1)

    def _next(self) -> None:
        self._step(1)

    def _seek_time(self, t_rel: float) -> None:
        if not self._frames:
            return
        self._idx = min(range(len(self._frames)),
                        key=lambda i: abs(self._frames[i].t_rel - t_rel))
        self._refresh()

    def _refresh(self) -> None:
        if not self._frames:
            self._scene.set_snapshot(None)
            return
        f = self._frames[self._idx]
        self._scene.set_snapshot(f.snapshot)
        self._strip.set_cursor(f.t_rel)
        if self._charts is not None:
            self._charts.set_cursor(f.t_rel)
        self._time_lbl.setText(f"t={f.t_rel:.2f}s  ({self._idx + 1}/{len(self._frames)})")
        la = f.live_aeb
        state = "BRAKE" if la.aeb_brake else ("WARN" if la.aeb_warn else "standby")
        in_win = self._window is not None and self._window[0] <= f.t_rel <= self._window[1]
        truth = "should-trigger" if in_win else ("must-not" if self._window is None else "outside")
        self._decision_lbl.setText(
            f"recorded: {state}  demand={f.raw_target_ms2:.1f}"
            f" (sent {la.target_decel_ms2:.1f})  ttc={_fmt(la.time_to_collision)}"
            f"  |  truth: {truth}"
        )

