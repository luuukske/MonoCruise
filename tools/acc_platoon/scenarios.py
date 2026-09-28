"""Named convoy scenarios: ten ACC clients at gap level 2 behind a scripted lead truck.

AEB ships disabled (`Settings.AEB_enabled`), so every scenario runs ACC alone
unless `aeb=True` is passed.
"""
from __future__ import annotations

from dataclasses import replace

from .netcode import LAGGY, NORMAL, Glitch
from .sim import Phase, Scenario

# Scenario time the lead's driver acts in every disturbance scenario.
EVENT_S: float = 5.0
CRUISE_KMH: float = 80.0
HARSH_MS2: float = 6.5
# A full-pedal stop on a loaded rig; the fitted plants top out at 10.9 to 13.9 m/s^2.
STOP_MS2: float = 8.0
# Emergency stop: how long the lead stands before it drives off again.
STANDSTILL_S: float = 15.0
# Queue stop: when the lead drives off again.
QUEUE_GO_S: float = EVENT_S + 20.0
LAGGY_CLIENT: int = 5


def steady(seed: int = 1, aeb: bool = False) -> Scenario:
    """Lead holds 80 km/h: anything the followers do is the netcode."""
    return Scenario("steady", duration_s=40.0, seed=seed, aeb=aeb)


def human_lead(seed: int = 1, aeb: bool = False) -> Scenario:
    """Lead holds about 80 km/h on a keyboard, throttle and lift in bursts."""
    return Scenario("human_lead", duration_s=40.0, seed=seed, aeb=aeb, lead_wobble_ms2=0.3)


def slowdown(seed: int = 1, aeb: bool = False) -> Scenario:
    """A comfortable slowdown, 80 to 60 km/h at 2 m/s^2."""
    return Scenario("slowdown", phases=(Phase(EVENT_S, 60.0, decel=2.0),), duration_s=30.0,
                    seed=seed, aeb=aeb)


def hard_brake(seed: int = 1, aeb: bool = False) -> Scenario:
    """Sudden harsh braking, 80 to 30 km/h at 6.5 m/s^2, then the lead holds 30."""
    return Scenario("hard_brake", phases=(Phase(EVENT_S, 30.0, decel=HARSH_MS2),),
                    duration_s=30.0, seed=seed, aeb=aeb)


def emergency_stop(seed: int = 1, aeb: bool = False) -> Scenario:
    """Lead stops dead from 80 km/h on full pedal, stands, then drives off."""
    go = EVENT_S + STANDSTILL_S
    return Scenario("emergency_stop",
                    phases=(Phase(EVENT_S, 0.0, decel=STOP_MS2), Phase(go, CRUISE_KMH, accel=0.8)),
                    duration_s=go + 40.0, seed=seed, aeb=aeb)


def stationary_lock(seed: int = 1, aeb: bool = True) -> Scenario:
    """Emergency stop with AEB, and nobody taps resume: ACC stays disarmed where AEB stopped."""
    sc = emergency_stop(seed, aeb)
    return replace(sc, name="stationary_lock", resume_after_s=None, duration_s=35.0)


def queue_stop(seed: int = 1, aeb: bool = False) -> Scenario:
    """Traffic ahead stops: 50 km/h to a standstill at 2 m/s^2, stands, then drives off."""
    return Scenario("queue_stop", v0_kmh=50.0, set_kmh=60.0,
                    phases=(Phase(EVENT_S, 0.0, decel=2.0), Phase(QUEUE_GO_S, 50.0, accel=0.8)),
                    duration_s=QUEUE_GO_S + 40.0, seed=seed, aeb=aeb)


def stop_and_go(seed: int = 1, aeb: bool = False) -> Scenario:
    """Heavy traffic ahead of the lead: 60 and 25 km/h in turns."""
    phases = []
    for k in range(3):
        at = EVENT_S + 16.0 * k
        phases += [Phase(at, 25.0, decel=2.0), Phase(at + 8.0, 60.0, accel=0.8)]
    return Scenario("stop_and_go", phases=tuple(phases), v0_kmh=60.0, set_kmh=70.0,
                    duration_s=55.0, seed=seed, aeb=aeb)


def laggy_client(seed: int = 1, aeb: bool = False) -> Scenario:
    """Steady lead, one follower mid-convoy on a bad connection."""
    nets = tuple(LAGGY if i == LAGGY_CLIENT else NORMAL for i in range(11))
    return Scenario("laggy_client", duration_s=40.0, seed=seed, nets=nets, aeb=aeb)


def blackout_brake(seed: int = 1, aeb: bool = False) -> Scenario:
    """Harsh braking while the first follower gets no update for a second, the worst stall seen."""
    sc = hard_brake(seed, aeb)
    return replace(sc, name="blackout_brake",
                   glitches={(0, 1): (Glitch("freeze", EVENT_S - 0.1, 1.0),)})


def desync_snap(seed: int = 1, aeb: bool = False) -> Scenario:
    """Steady lead drawn 8 m closer for 0.3 s on the first follower: a TMP desync."""
    return Scenario("desync_snap", duration_s=20.0, seed=seed, aeb=aeb,
                    glitches={(0, 1): (Glitch("shift", 10.0, 0.3, shift_m=-8.0),)})


ALL = {f.__name__: f for f in (steady, human_lead, slowdown, hard_brake, emergency_stop,
                                stationary_lock, queue_stop, stop_and_go, laggy_client,
                                blackout_brake, desync_snap)}
