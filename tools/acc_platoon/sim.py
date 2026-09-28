"""Closed-loop convoy: a scripted lead truck and N MonoCruise clients behind it.

Every client runs the shipped code from the radar filter chain up: `Vehicle`
kinematics on TMP-drawn positions, `AdaptiveCruiseController` through the
longitudinal ACC wrapper, `CruiseController`, the orchestrator's min-arbitration,
the headless `AEBThread` and `HoldController`. The tracker is replaced by
straight-road geometry, the mapper and truck by `plant.TruckPlant`. See
tools/acc_platoon/README.md.
"""
from __future__ import annotations

import contextlib
import random
import types
from dataclasses import dataclass, field, replace

from core.cruise_control_thread import acc_controller
from core.cruise_control_thread.acc_controller import T_HEADWAY_BY_LEVEL_S
from core.longitudinal import cc as long_cc
from core.settings import Settings

from .netcode import NORMAL, PHYSICS_HZ, SYNC_DELAY_S, Glitch, NetProfile, TmpStream
from .plant import HEAVY, LIGHT, MEDIUM, RIG_LEN_M, TruckSpec
from .stack import Client, Registry

DT: float = 1.0 / PHYSICS_HZ
# Physics steps between radar reads, measured on the clip corpus (1 to 3, mostly 2).
FRAME_STEPS: tuple[int, ...] = (1, 2, 2, 2, 2, 2, 3, 3)
# AEB and ACC threads run at 30 Hz: every other physics step.
AEB_EVERY_STEPS: int = 2
# Scenario time 0 on the game clock. Radar sentinels assume a positive clock, as live.
T_BASE_S: float = 1000.0
CC_FIELDS: tuple[str, ...] = (
    "cc_kp", "cc_ki", "cc_kd", "cc_integral_clamp", "cc_accel_min_ms2",
    "cc_accel_max_ms2", "cc_accel_profile", "global_speed_limit_kmh",
)


@dataclass(frozen=True)
class Phase:
    """From `at_s` on, the lead driver goes for `kmh` using at most these rates."""

    at_s: float
    kmh: float
    accel: float = 0.8
    decel: float = 1.5


@dataclass
class Scenario:
    name: str
    phases: tuple[Phase, ...] = ()
    followers: int = 10
    v0_kmh: float = 80.0
    set_kmh: float = 90.0
    duration_s: float = 60.0
    warmup_s: float = 8.0
    # Open-loop share of the warm-up: every truck holds v0 so radar filters start warm.
    preroll_s: float = 3.0
    gap_level: int = 2
    seed: int = 1
    fleet: tuple[TruckSpec, ...] = (HEAVY, MEDIUM, LIGHT)
    # One profile per truck, lead first; shorter tuples repeat NORMAL.
    nets: tuple[NetProfile, ...] = ()
    sync_delay_s: float = SYNC_DELAY_S
    # (sender index, receiver index) -> glitches; sender 0 is the lead truck.
    glitches: dict[tuple[int, int], tuple[Glitch, ...]] = field(default_factory=dict)
    split_trailers: bool = True
    # AEB ships disabled; scenarios opt in.
    aeb: bool = False
    # After an AEB stop disarms ACC, the driver taps resume this long after the truck
    # ahead pulls away. None: nobody resumes.
    resume_after_s: float | None = 2.0
    # Lead driver on a keyboard: throttle and lift bursts of this size, 1 to 4 s each.
    lead_wobble_ms2: float = 0.0


@dataclass
class Trace:
    """Per-step record of one truck. Gaps and follower fields stay empty for the lead."""

    s: list[float] = field(default_factory=list)
    v: list[float] = field(default_factory=list)
    a: list[float] = field(default_factory=list)
    cmd: list[float] = field(default_factory=list)
    hold: list[str] = field(default_factory=list)
    cap: list[float] = field(default_factory=list)
    overlay: list[bool] = field(default_factory=list)
    aeb: list[bool] = field(default_factory=list)
    armed: list[bool] = field(default_factory=list)
    gap_drawn: list[float] = field(default_factory=list)
    gap_true: list[float] = field(default_factory=list)
    lead_v_seen: list[float] = field(default_factory=list)
    lead_a_seen: list[float] = field(default_factory=list)


@dataclass
class Run:
    scenario: Scenario
    t: list[float]
    traces: list[Trace]
    specs: list[TruckSpec]
    nets: list[NetProfile]
    delays_s: list[float]
    resumes: list[int]


class LeadDriver:
    """The scripted human in the first truck."""

    def __init__(self, sc: Scenario, rng: random.Random) -> None:
        self.phases = tuple(sorted(sc.phases, key=lambda p: p.at_s))
        self.v0 = sc.v0_kmh / 3.6
        self.wobble = sc.lead_wobble_ms2
        self._rng = rng
        self._burst = 0.0
        self._burst_until = -1e9

    def command(self, t: float, v: float) -> float:
        target, accel, decel = self.v0, 0.8, 1.5
        for p in self.phases:
            if p.at_s <= t:
                target, accel, decel = p.kmh / 3.6, p.accel, p.decel
        if target <= 0.0:
            return -decel if v > 0.05 else 0.0
        cmd = max(-decel, min(accel, 1.2 * (target - v)))
        if self.wobble > 0.0 and abs(target - v) < 1.0:
            if t >= self._burst_until:
                sign = -1.0 if self._burst > 0.0 else 1.0
                self._burst = sign * self._rng.uniform(0.5, 1.0) * self.wobble
                self._burst_until = t + self._rng.uniform(1.0, 4.0)
            cmd += self._burst
        return cmd


class _Clock:
    def __init__(self) -> None:
        self.t = 0.0

    def monotonic(self) -> float:
        return self.t


@contextlib.contextmanager
def _runtime(level: int, reg: Registry, clock: _Clock):
    """Pin the settings the stack reads and point its registry and clock at the sim."""
    s = Settings.instance()
    fields = Settings.__dataclass_fields__
    pinned = {name: Settings._dataclass_field_default(fields[name]) for name in CC_FIELDS}
    pinned.update(acc_enabled=True, cc_mode="Cruise control", acc_gap_level=level)
    saved = {name: getattr(s, name) for name in pinned}
    saved_mods = (acc_controller.registry, long_cc.registry, acc_controller.time)
    try:
        for name, value in pinned.items():
            setattr(s, name, value)
        acc_controller.registry = reg
        long_cc.registry = reg
        acc_controller.time = types.SimpleNamespace(monotonic=clock.monotonic)
        yield
    finally:
        for name, value in saved.items():
            setattr(s, name, value)
        acc_controller.registry, long_cc.registry, acc_controller.time = saved_mods


def _build(sc: Scenario, clock: _Clock) -> tuple[list[Client], list[float]]:
    rng = random.Random(sc.seed)
    n = sc.followers + 1
    specs = [sc.fleet[rng.randrange(len(sc.fleet))] for _ in range(n)]
    nets = [sc.nets[i] if i < len(sc.nets) else NORMAL for i in range(n)]
    v0 = sc.v0_kmh / 3.6
    t0 = T_BASE_S - sc.warmup_s
    headway = T_HEADWAY_BY_LEVEL_S[sc.gap_level]
    clients: list[Client] = []
    delays = [0.0]
    front = 0.0
    for i in range(n):
        if i:
            delay = sc.sync_delay_s + 0.5 * (nets[i].ping_s + nets[i - 1].ping_s)
            delays.append(delay)
            front -= RIG_LEN_M + acc_controller.S0_M + v0 * headway + v0 * delay
        clients.append(Client(i, specs[i], nets[i], front, v0, sc.set_kmh, t0, DT, clock, sc.aeb))
    for i, c in enumerate(clients):
        for j in range(i):
            glitches = tuple(replace(g, at_s=g.at_s + T_BASE_S)
                             for g in sc.glitches.get((j, i), ()))
            c.streams[j] = TmpStream(clients[j].path, nets[j], nets[i],
                                     random.Random(rng.getrandbits(64)), t0,
                                     sc.sync_delay_s, glitches)
        c.next_frame = rng.randrange(3)
    return clients, delays


def run(sc: Scenario) -> Run:
    """Simulate the scenario; scenario time 0 is the end of the warm-up."""
    clock = _Clock()
    clients, delays = _build(sc, clock)
    lead, followers = clients[0], clients[1:]
    driver = LeadDriver(sc, random.Random(sc.seed * 104729 + 3))
    frame_rng = random.Random(sc.seed * 7919 + 17)
    reg = Registry()
    traces = [Trace() for _ in clients]
    times: list[float] = []
    v0 = sc.v0_kmh / 3.6
    steps = int(round((sc.warmup_s + sc.duration_s) * PHYSICS_HZ))
    t_closed = -sc.warmup_s + sc.preroll_s
    with _runtime(sc.gap_level, reg, clock):
        for k in range(steps + 1):
            t = -sc.warmup_s + k * DT
            t_game = T_BASE_S + t
            closed = t >= t_closed
            for c in clients:
                if closed:
                    c.truck.step(c.cmd, DT)
                else:
                    c.truck.s += v0 * DT
            for c in followers:
                _contact(c, clients[c.idx - 1], t_game)
            for c in clients:
                c.path.append(c.truck.s)
            clock.t = t_game
            lead.cmd = driver.command(t, lead.truck.v) if closed else 0.0
            for c in followers:
                for stream in c.streams.values():
                    stream.advance(t_game, DT)
                c.acc_data.leads = c.next_leads
                if k >= c.next_frame:
                    c.radar_frame(t_game, sc.split_trailers)
                    c.next_frame = k + frame_rng.choice(FRAME_STEPS)
                reg.client = c
                acc_out = None
                if closed:
                    if (k + c.idx) % AEB_EVERY_STEPS == 0:
                        c.aeb_tick()
                    acc_out = c.control_tick(t_game, DT)
                    c.driver(t_game, c.streams[c.idx - 1].position(t_game), sc.resume_after_s)
                _record_follower(traces[c.idx], c, clients[c.idx - 1], acc_out, t_game)
            reg.client = None
            times.append(t)
            for c in clients:
                tr = traces[c.idx]
                tr.s.append(c.truck.s)
                tr.v.append(c.truck.v)
                tr.a.append(c.truck.a)
                tr.cmd.append(c.cmd)
                tr.hold.append(c.truck.hold_out.state)
    return Run(sc, times, traces, [c.spec for c in clients], [c.net for c in clients], delays,
               [c.resumes for c in clients])


def _contact(c: Client, ahead: Client, t: float) -> None:
    """A truck cannot drive through the one ahead as its own client draws it."""
    rear = c.streams[ahead.idx].position(t) - RIG_LEN_M
    if c.truck.s > rear:
        c.truck.s = rear
        c.truck.v = min(c.truck.v, ahead.truck.v)
        c.contacts += 1


def _record_follower(tr: Trace, c: Client, ahead: Client, acc_out, t: float) -> None:
    cap = acc_out.wanted_ms2 if acc_out is not None else None
    tr.cap.append(cap if cap is not None else float("nan"))
    tr.overlay.append(c.traced.overlay)
    tr.aeb.append(c.aeb_brake)
    tr.armed.append(c.cc.enabled)
    drawn = c.streams[ahead.idx].position(t)
    tr.gap_drawn.append(drawn - RIG_LEN_M - c.truck.s)
    tr.gap_true.append(ahead.truck.s - RIG_LEN_M - c.truck.s)
    view = c.views.get(ahead.idx)
    veh = view.vehicle if view is not None else None
    tr.lead_v_seen.append(veh.acc_speed if veh is not None else float("nan"))
    tr.lead_a_seen.append(veh.acc_accel if veh is not None else float("nan"))
