"""What a convoy run did, per truck and along the string. See tools/acc_platoon/README.md."""
from __future__ import annotations

import math
from dataclasses import dataclass

from core.cruise_control_thread.acc_controller import S0_M
from core.sending_thread.hold_controller import STATE_HOLDING, STATE_ROLLING, STATE_STOPPING

from .sim import Run

# A brake episode starts below this command and ends back above half of it.
BRAKE_EVENT_MS2: float = 1.0
MOVING_MS: float = 1.0
STOPPED_MS: float = 0.3
AT_REST_MS: float = 0.05


@dataclass
class TruckStats:
    idx: int
    label: str
    min_speed_kmh: float
    max_speed_kmh: float
    speed_std_kmh: float
    peak_decel_ms2: float
    peak_cmd_decel_ms2: float
    brake_events: int
    accel_rms_ms2: float
    jerk_rms_ms3: float
    stopped: bool
    min_gap_drawn_m: float = math.inf
    min_gap_true_m: float = math.inf
    overlay_ticks: int = 0
    aeb_ticks: int = 0
    disarmed: bool = False


def _window(run: Run, t_from: float, t_to: float | None) -> range:
    t_to = run.t[-1] if t_to is None else t_to
    lo = next((i for i, t in enumerate(run.t) if t >= t_from), len(run.t))
    hi = next((i for i, t in enumerate(run.t) if t > t_to), len(run.t))
    return range(lo, hi)


def _brake_events(cmd: list[float]) -> int:
    events, braking = 0, False
    for c in cmd:
        if not braking and c < -BRAKE_EVENT_MS2:
            events, braking = events + 1, True
        elif braking and c > -0.5 * BRAKE_EVENT_MS2:
            braking = False
    return events


def truck_stats(run: Run, t_from: float = 0.0, t_to: float | None = None) -> list[TruckStats]:
    win = _window(run, t_from, t_to)
    dt = run.t[1] - run.t[0]
    out: list[TruckStats] = []
    for k, tr in enumerate(run.traces):
        v = [tr.v[i] for i in win]
        a = [tr.a[i] for i in win]
        cmd = [tr.cmd[i] for i in win]
        mean_v = sum(v) / len(v)
        jerk = [(a[i + 1] - a[i]) / dt for i in range(len(a) - 1)]
        st = TruckStats(
            idx=k, label=run.specs[k].label,
            min_speed_kmh=min(v) * 3.6, max_speed_kmh=max(v) * 3.6,
            speed_std_kmh=math.sqrt(sum((x - mean_v) ** 2 for x in v) / len(v)) * 3.6,
            peak_decel_ms2=max(0.0, -min(a)), peak_cmd_decel_ms2=max(0.0, -min(cmd)),
            brake_events=_brake_events(cmd),
            accel_rms_ms2=math.sqrt(sum(x * x for x in a) / len(a)),
            jerk_rms_ms3=math.sqrt(sum(x * x for x in jerk) / max(len(jerk), 1)),
            stopped=min(v) < STOPPED_MS,
        )
        if k:
            st.min_gap_drawn_m = min(tr.gap_drawn[i] for i in win)
            st.min_gap_true_m = min(tr.gap_true[i] for i in win)
            st.overlay_ticks = sum(1 for i in win if tr.overlay[i])
            st.aeb_ticks = sum(1 for i in win if tr.aeb[i])
            st.disarmed = not all(tr.armed[i] for i in win)
        out.append(st)
    return out


def contacts(run: Run) -> list[int]:
    """Followers that touched the truck ahead as their own client draws it."""
    return [k for k, tr in enumerate(run.traces[1:], 1) if min(tr.gap_drawn) <= 0.0]


def unprovoked_brakes(run: Run, t_from: float = 0.0, calm_s: float = 3.0,
                      calm_ms2: float = 0.5) -> list[int]:
    """Brake episodes each truck starts while the truck ahead has not braked for `calm_s`."""
    win = _window(run, t_from, None)
    steps = max(1, int(round(calm_s / (run.t[1] - run.t[0]))))
    out = [0]
    for k in range(1, len(run.traces)):
        tr, ahead = run.traces[k], run.traces[k - 1]
        count, braking = 0, False
        for i in win:
            c = tr.cmd[i]
            if not braking and c < -BRAKE_EVENT_MS2:
                braking = True
                if min(ahead.a[max(0, i - steps):i + 1]) >= -calm_ms2:
                    count += 1
            elif braking and c > -0.5 * BRAKE_EVENT_MS2:
                braking = False
        out.append(count)
    return out


def hop_gains(stats: list[TruckStats], v_ref_kmh: float) -> list[float]:
    """Each follower's speed dip over the dip of the truck directly ahead."""
    dips = [max(v_ref_kmh - s.min_speed_kmh, 1e-6) for s in stats]
    return [dips[k] / dips[k - 1] for k in range(1, len(dips))]


def dip_gains(stats: list[TruckStats], v_ref_kmh: float) -> list[float]:
    """Each truck's speed dip below `v_ref_kmh` over the lead truck's dip."""
    lead_dip = max(v_ref_kmh - stats[0].min_speed_kmh, 1e-6)
    return [(v_ref_kmh - s.min_speed_kmh) / lead_dip for s in stats]


def relaunch_times(run: Run, t_go: float) -> list[float | None]:
    """When each truck rolls off from standstill after `t_go`, lead included.

    A truck still rolling into the queue at `t_go` counts from the stop it comes to later.
    """
    win = _window(run, t_go, None)
    out: list[float | None] = []
    for tr in run.traces:
        stood = tr.v[win[0]] < STOPPED_MS
        when = None
        for i in win:
            if tr.v[i] < STOPPED_MS:
                stood = True
            elif stood and tr.v[i] > MOVING_MS:
                when = run.t[i]
                break
        out.append(when)
    return out


def standstill_creep(run: Run, t_from: float, t_to: float) -> list[float]:
    """How far each follower crept closer than it was held, while the truck ahead stood.

    Counting starts when the hold owns the truck behind a standing one; a relaunch decided
    while the truck ahead still rolled is a crawl. Closing up from farther than s0 is the
    intended close-up (ACC_ARCHITECTURE.md §10.1), so only ground below min(gap, s0) counts.
    """
    win = _window(run, t_from, t_to)
    out = [0.0]
    for k in range(1, len(run.traces)):
        tr, ahead = run.traces[k], run.traces[k - 1]
        floor = None
        creep = 0.0
        for i in win:
            if ahead.v[i] >= AT_REST_MS:
                floor = None
            elif floor is None:
                if tr.hold[i] == STATE_HOLDING:
                    floor = min(tr.gap_drawn[i], S0_M)
            else:
                creep = max(creep, floor - tr.gap_drawn[i])
        out.append(creep)
    return out


def restops(run: Run, t_from: float) -> list[int]:
    """How often each truck came back to rest after first rolling off at `t_from` or later.

    Behind a lead that never stops, every one of these is a lurch."""
    out = []
    for tr in run.traces:
        rolled, count = False, 0
        for i in _window(run, t_from, None):
            if tr.v[i] > MOVING_MS / 2.0:
                rolled = True
            elif rolled and tr.v[i] < STOPPED_MS / 3.0:
                rolled, count = False, count + 1
        out.append(count)
    return out


def hold_captures(run: Run, t_from: float, t_to: float) -> list[int]:
    """How often the hold FSM took each truck from ROLLING into STOPPING in the window."""
    win = _window(run, t_from, t_to)
    return [sum(1 for a, b in zip(win, win[1:])
                if tr.hold[a] == STATE_ROLLING and tr.hold[b] == STATE_STOPPING)
            for tr in run.traces]


def moved_after_disarm(run: Run) -> list[float]:
    """How far each follower rolled after an AEB stop switched its ACC off (0 if never)."""
    out = [0.0]
    for tr in run.traces[1:]:
        off = next((i for i in range(1, len(tr.armed)) if tr.armed[i - 1] and not tr.armed[i]),
                   None)
        out.append(0.0 if off is None else max(tr.s[i] - tr.s[off] for i in range(off, len(tr.s))))
    return out


def stop_gaps(run: Run, t: float) -> list[float]:
    """Drawn gap each follower stands at, at scenario time `t`."""
    i = _window(run, t, None)[0]
    return [math.nan] + [tr.gap_drawn[i] for tr in run.traces[1:]]


def convoy_summary(run: Run, stats: list[TruckStats], bucket: int = 10) -> str:
    """Long-convoy report: followers grouped in buckets of `bucket`, lead excluded."""
    followers = stats[1:]
    groups = [followers[i:i + bucket] for i in range(0, len(followers), bucket)]
    brakes = unprovoked_brakes(run)[1:]
    head = (f"followers {len(followers)}  gap level {run.scenario.gap_level}  "
            f"stopped {sum(s.stopped for s in followers)}  contacts {len(contacts(run))}  "
            f"unprovoked brakes {sum(brakes)} (max one truck {max(brakes)})")
    lines = [head]
    if run.scenario.v0_kmh - stats[0].min_speed_kmh > 1.0:
        dips = dip_gains(stats, run.scenario.v0_kmh)[1:]
        lines.append(f"dip gain max {max(dips):.2f} at truck {dips.index(max(dips)) + 1}  "
                     f"hop gain max {max(hop_gains(stats, run.scenario.v0_kmh)):.2f}")
    lines += [f"min speed {min(s.min_speed_kmh for s in followers):.1f} km/h",
              "trucks      speed std   min km/h   peak dec   jerk rms   brakes"]
    for k, grp in enumerate(groups):
        lo, hi = grp[0].idx, grp[-1].idx
        n = len(grp)
        lines.append(
            f"{lo:3d}..{hi:<3d}  {sum(s.speed_std_kmh for s in grp) / n:9.2f}  "
            f"{sum(s.min_speed_kmh for s in grp) / n:9.1f}  "
            f"{max(s.peak_decel_ms2 for s in grp):9.2f}  "
            f"{sum(s.jerk_rms_ms3 for s in grp) / n:9.3f}  "
            f"{sum(brakes[lo - 1:hi]):6d}")
    return "\n".join(lines)


def table(stats: list[TruckStats]) -> str:
    """Plain-text per-truck table for reports and assertion messages."""
    lines = [" #  truck         v min..max km/h  std   peak dec  cmd dec  brakes  gap drawn  true  overlay  aeb  disarm"]
    for s in stats:
        gap = "" if s.idx == 0 else (f"{s.min_gap_drawn_m:9.1f} {s.min_gap_true_m:5.1f} "
                                     f"{s.overlay_ticks:8d} {s.aeb_ticks:4d}  {'yes' if s.disarmed else ''}")
        lines.append(
            f"{s.idx:2d}  {s.label:12s} {s.min_speed_kmh:6.1f}..{s.max_speed_kmh:5.1f}  "
            f"{s.speed_std_kmh:5.2f}  {s.peak_decel_ms2:8.2f} {s.peak_cmd_decel_ms2:8.2f}  "
            f"{s.brake_events:6d} {gap}")
    return "\n".join(lines)
