"""The measured line bent by both turns (README, OppositeLaneFilter bend miss)."""
from __future__ import annotations

import math
from collections import deque
from dataclasses import replace

import pytest

from core.aeb.calibration import DEFAULT as CAL
from core.aeb.thread import _los_bend_shift, _los_predicted_miss


def _track(r_ego: float, r_tgt: float, v_ego: float, v_tgt: float, ahead_m: float):
    """Ego and an oncoming car on concentric circles round the origin; ego turns left."""
    out = deque()
    dt = 1.0 / 30.0
    n = int(CAL.los_veto_window_s / dt) + 2
    t_now = 100.0
    for i in range(n):
        t = t_now - (n - 1 - i) * dt
        a_e = v_ego * (t - t_now) / r_ego
        a_t = ahead_m / r_ego - v_tgt * (t - t_now) / r_tgt
        out.append((t, r_tgt * math.cos(a_t), r_tgt * math.sin(a_t),
                    r_ego * math.cos(a_e), r_ego * math.sin(a_e)))
    # Travel directions now: ego along +angle, the car along -angle.
    e_f = (0.0, 1.0)
    a_t = ahead_m / r_ego
    t_f = (math.sin(a_t), -math.cos(a_t))
    return out, t_now, e_f, t_f


def _kappa(fwd, centre_dir):
    """Signed curvature whose ArcPath centre lies toward ``centre_dir``."""
    fx, fz = fwd
    return 1.0 if (fz * centre_dir[0] - fx * centre_dir[1]) > 0.0 else -1.0


def test_a_car_in_the_inner_lane_of_a_bend_passes_a_lane_over_by_the_bent_line():
    """Straight, the line runs through ego's centre; bent by both turns it passes a lane over."""
    r_e, r_t, v_e, v_t = 178.0, 172.5, 11.6, 26.2
    trk, now, e_f, t_f = _track(r_e, r_t, v_e, v_t, 60.0)
    straight = _los_predicted_miss(trk, now, CAL)
    a_t = 60.0 / r_e
    k_e = _kappa(e_f, (-1.0, 0.0)) / r_e
    k_t = _kappa(t_f, (-math.cos(a_t), -math.sin(a_t))) / r_t
    shift = _los_bend_shift(trk, now, CAL, (v_e, k_e, *e_f), (v_t, k_t, *t_f))
    assert straight is not None and straight < 1.0
    assert shift is not None and straight + shift > 4.0


def test_without_a_turn_the_bent_line_is_the_straight_one():
    trk = deque()
    for i in range(20):
        t = 10.0 + i / 30.0
        trk.append((t, -3.5, -80.0 + 30.0 * (t - 10.0), 0.0, -20.0 * (t - 10.0)))
    now = trk[-1][0]
    assert _los_bend_shift(trk, now, CAL, (20.0, 0.0, 0.0, -1.0), (30.0, 0.0, 0.0, 1.0)) == 0.0


def test_a_swerve_is_no_steady_bend():
    """01e038e7: ego's curvature ran -0.0035 to -0.0137 in 0.3 s; a held bend stays inside the band."""
    from core.aeb.thread import _steady_kappa

    held, swerve = {}, {}
    steady = [_steady_kappa(held, -1, 10.0 + i / 30.0, 0.0056 + 0.0001 * (i % 3), CAL)
              for i in range(20)]
    turning = [_steady_kappa(swerve, -1, 10.0 + i / 30.0, -0.0035 - 0.0011 * i, CAL)
               for i in range(20)]
    assert not steady[0] and steady[-1]
    assert not any(turning)


def _clip(name: str):
    import os
    from pathlib import Path

    from core.aeb.clip_store import ClipStore
    base = os.environ.get("LOCALAPPDATA")
    p = Path(base) / "MonoCruise" / "aeb_clips" / name if base else None
    if p is None or not p.is_file():
        pytest.skip(f"clip {name} not in local clip store")
    return ClipStore().load(p)


def _onset(clip, cal) -> float | None:
    from core.aeb.clip_eval import run_headless
    braked = [e.t_rel for e in run_headless(clip, cal=cal) if e.aeb_brake]
    return braked[0] if braked else None


@pytest.mark.needs_clips
def test_132557d1_the_oncoming_truck_a_lane_over_on_the_bend_no_longer_brakes():
    """The straight line read 0.97 to 0.13 m; the truck passed a lane over (map 5.5 m)."""
    clip = _clip("20260807T193343Z_auto_engagement_132557d1.json.gz")
    assert _onset(clip, CAL) is None
    old = _onset(clip, replace(CAL, los_bend_miss_enabled=False))
    assert old is not None and 7.85 <= old <= 8.0


@pytest.mark.needs_clips
def test_249e9d51_keeps_its_brake_with_the_bend_miss():
    """A junction turner is no road bend: the onset stays at 3.54 s either way."""
    clip = _clip("20261124T154554Z_auto_crash_249e9d51.json.gz")
    on = _onset(clip, CAL)
    assert on is not None and 3.126 <= on <= 3.58
    assert on == _onset(clip, replace(CAL, los_bend_miss_enabled=False))
