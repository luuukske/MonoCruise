"""Ten MonoCruise ACC clients in a TruckersMP convoy at gap level 2. See tools/acc_platoon/README.md.

Every client runs the shipped radar chain, ACC, cruise PID, hold FSM and, where a
scenario asks for it, AEB; the other trucks reach it through a TMP display model
calibrated on the clip corpus. Each scenario runs once per session, in parallel
worker processes, and several tests read it.

Two kinds of bound. `TARGET_*` is what the convoy should do. `BASELINE_*` is the
worst of seeds 1 to 3 when this landed (2026-09-28), where the stack does not meet
its target yet: lower it whenever the stack improves, never raise it to go green.
"""
from __future__ import annotations

import random
from dataclasses import replace

import pytest

from core.cruise_control_thread.acc_controller import S0_M, T_HEADWAY_BY_LEVEL_S
from tools.acc_platoon import metrics, scenarios
from tools.acc_platoon.batch import run_many
from tools.acc_platoon.netcode import CLEAN, LAGGY, NORMAL, PHYSICS_HZ, Glitch, TmpStream, TruePath
from tools.acc_platoon.sim import Run, Scenario

DT = 1.0 / PHYSICS_HZ
WANTED_GAP_TOL_M = 1.0
CLEAN_SPEED_STD_KMH = 0.3
# Movement a held truck may show behind a standing one, and after AEB switched ACC off.
CREEP_TOL_M = 0.3
# Least room AEB may leave when it rescues the convoy.
AEB_MIN_GAP_M = 0.5
# By then the creep's start wave has passed the first followers.
CREEP_STEADY_FROM_S = 15.0

TARGET_UNPROVOKED_BRAKES = 0
TARGET_SPEED_STD_KMH = 1.0
TARGET_HOP_GAIN = 1.0
TARGET_STOPPED = 0
TARGET_CONTACTS = 0
TARGET_UNDERSHOOT_KMH = 5.0
TARGET_DISARMED = 0
TARGET_STOP_GAP_M = 4.0
TARGET_RELAUNCH_HOP_S = 2.0
TARGET_SNAP_BRAKE_MS2 = 0.5
TARGET_PULL_AWAY_START_S = 0.5
TARGET_CREEP_RESTOPS = 0

BASELINE_STEADY_UNPROVOKED_BRAKES = 10
BASELINE_STEADY_SPEED_STD_KMH = 5.9
BASELINE_SLOWDOWN_HOP_GAIN = 1.32
BASELINE_HARD_BRAKE_STOPPED = 7
BASELINE_HARD_BRAKE_UNDERSHOOT_KMH = 18.2
BASELINE_HARD_BRAKE_AEB_DISARMED = 2
BASELINE_FULL_STOP_CONTACTS = 2
BASELINE_QUEUE_STOP_GAP_M = 3.1
BASELINE_QUEUE_RELAUNCH_HOP_S = 2.5
BASELINE_SNAP_BRAKE_MS2 = 1.8
BASELINE_LAGGY_UNPROVOKED_BRAKES = 8
# Landed 2026-09-30 with ACC_ARCHITECTURE.md §10.2; before it 1.37 s and 12 re-stops.
# 2026-10-08: 0.82 s with five published leads (ACC_ARCHITECTURE.md §9.9).
BASELINE_PULL_AWAY_START_S = 0.82
BASELINE_CREEP_RESTOPS = 7


def cases(seed: int = 1) -> dict[str, Scenario]:
    """Every run the tests read, keyed by name."""
    return {
        "clean": replace(scenarios.steady(seed), name="clean", nets=(CLEAN,) * 11,
                         duration_s=15.0),
        "steady": scenarios.steady(seed),
        "slowdown": scenarios.slowdown(seed),
        "hard_brake": scenarios.hard_brake(seed),
        "hard_brake_aeb": scenarios.hard_brake(seed, aeb=True),
        "full_stop": replace(scenarios.emergency_stop(seed), duration_s=25.0),
        "stationary_lock": scenarios.stationary_lock(seed),
        "queue_stop": scenarios.queue_stop(seed),
        "blackout_aeb": scenarios.blackout_brake(seed, aeb=True),
        "laggy_client": scenarios.laggy_client(seed),
        "desync_snap": scenarios.desync_snap(seed),
        "slow_pull_away": scenarios.slow_pull_away(seed),
        "creep": scenarios.creep(seed),
    }


@pytest.fixture(scope="module")
def runs() -> dict[str, Run]:
    todo = cases()
    return dict(zip(todo, run_many(list(todo.values()))))


def _why(run: Run, detail: str) -> str:
    return f"{run.scenario.name}: {detail}\n{metrics.table(metrics.truck_stats(run))}"


def relaunch_hops(run: Run) -> list[float | None]:
    """Seconds between a truck and the one ahead rolling off after the queue's lead goes."""
    launch = metrics.relaunch_times(run, scenarios.QUEUE_GO_S - DT)
    return [None if launch[k] is None or launch[k - 1] is None else launch[k] - launch[k - 1]
            for k in range(1, len(launch))]


def snap_brake(run: Run) -> float:
    """The first follower's hardest command while an 8 m desync of its lead plays out."""
    tr = run.traces[1]
    win = [i for i, t in enumerate(run.t) if 9.5 <= t <= 13.0]
    return max(0.0, -min(tr.cmd[i] for i in win))


def _drive(profile, seconds: float, seed: int = 1, glitches=(), v: float = 22.0):
    t = 1000.0
    path = TruePath(t, 0.0, v)
    stream = TmpStream(path, profile, profile, random.Random(seed), t, glitches=glitches)
    s, out = 0.0, []
    for _ in range(int(seconds * PHYSICS_HZ)):
        t += DT
        s += v * DT
        path.append(s)
        stream.advance(t, DT)
        out.append((t, stream.position(t), stream.tau))
    return stream, path, out


def test_a_clean_link_draws_the_sender_exactly_one_delay_late():
    stream, path, out = _drive(CLEAN, 30.0)
    assert max(abs(p - path.at(t - stream.delay_s)) for t, p, _ in out) < 1e-9


@pytest.mark.parametrize("profile", [NORMAL, LAGGY], ids=["normal", "laggy"])
def test_the_drawn_truck_is_never_ahead_of_the_real_one(profile):
    """TMP replays the past; a model that extrapolated would invent braking the lead never did."""
    for seed in range(12):
        _, _, out = _drive(profile, 60.0, seed)
        assert all(t - 2.0 < tau < t for t, _, tau in out)


def test_a_blackout_holds_the_drawn_position_byte_identical():
    """The radar's lag freeze keys on repeated positions, which is how real stalls arrive."""
    _, _, out = _drive(NORMAL, 10.0, glitches=(Glitch("freeze", 1004.0, 1.0),))
    assert len({p for t, p, _ in out if 1004.0 <= t < 1005.0}) == 1


def test_a_clean_network_convoy_holds_its_wanted_gap(runs):
    """Harness check: with no netcode artefacts ten trucks sit still at the level-2 gap."""
    r = runs["clean"]
    stats = metrics.truck_stats(r)
    wanted = S0_M + r.scenario.v0_kmh / 3.6 * T_HEADWAY_BY_LEVEL_S[r.scenario.gap_level]
    tail = range(len(r.t) // 2, len(r.t))
    worst = max(abs(tr.gap_drawn[i] - wanted) for tr in r.traces[1:] for i in tail)
    assert worst < WANTED_GAP_TOL_M, _why(r, f"gap strays {worst:.2f} m from {wanted:.1f} m")
    assert max(s.speed_std_kmh for s in stats[1:]) < CLEAN_SPEED_STD_KMH, _why(r, "speed wobble")
    assert sum(s.brake_events for s in stats[1:]) == 0, _why(r, "braked on a clean link")


def test_steady_convoy_brakes_for_nothing(runs):
    """The lead holds 80 km/h, yet netcode alone makes followers brake. Target 0."""
    r = runs["steady"]
    brakes = metrics.unprovoked_brakes(r)
    assert sum(brakes) <= BASELINE_STEADY_UNPROVOKED_BRAKES, _why(r, f"unprovoked {brakes}")


def test_steady_convoy_does_not_grow_netcode_noise_into_waves(runs):
    r = runs["steady"]
    worst = max(s.speed_std_kmh for s in metrics.truck_stats(r)[1:])
    assert worst <= BASELINE_STEADY_SPEED_STD_KMH, _why(r, f"speed std {worst:.2f} km/h")


def test_a_slowdown_does_not_grow_along_the_convoy(runs):
    """String stability: each truck dips no deeper than the one ahead. Target hop gain 1.0."""
    r = runs["slowdown"]
    gains = metrics.hop_gains(metrics.truck_stats(r), r.scenario.v0_kmh)
    assert max(gains) <= BASELINE_SLOWDOWN_HOP_GAIN, _why(r, f"hop gains {gains}")


def test_a_slowdown_to_60_does_not_stop_anyone(runs):
    r = runs["slowdown"]
    stopped = sum(s.stopped for s in metrics.truck_stats(r)[1:])
    assert stopped <= TARGET_STOPPED, _why(r, f"{stopped} trucks stopped")


def test_a_harsh_brake_to_30_does_not_become_a_standstill_jam(runs):
    """The lead never goes below 28 km/h; every truck brought to a stop is overreaction."""
    r = runs["hard_brake"]
    stopped = sum(s.stopped for s in metrics.truck_stats(r)[1:])
    assert stopped <= BASELINE_HARD_BRAKE_STOPPED, _why(r, f"{stopped} trucks stopped")


def test_the_first_follower_lands_near_the_lead_after_a_harsh_brake(runs):
    r = runs["hard_brake"]
    stats = metrics.truck_stats(r)
    under = stats[0].min_speed_kmh - stats[1].min_speed_kmh
    assert under <= BASELINE_HARD_BRAKE_UNDERSHOOT_KMH, _why(r, f"undershoot {under:.1f} km/h")


def test_acc_alone_in_a_harsh_brake_contacts(runs):
    """AEB ships disabled. ACC alone still lets the amplified wave reach trucks. Target 0."""
    r = runs["hard_brake"]
    hit = metrics.contacts(r)
    assert len(hit) <= TARGET_CONTACTS, _why(r, f"contacts at {hit}")


def test_acc_alone_in_a_full_pedal_stop_contacts(runs):
    """The lead stops at 8 m/s^2, past ACC's own authority; only AEB can stop this. Target 0."""
    r = runs["full_stop"]
    hit = metrics.contacts(r)
    assert len(hit) <= BASELINE_FULL_STOP_CONTACTS, _why(r, f"contacts at {hit}")


def test_with_aeb_a_harsh_brake_ends_without_contact(runs):
    r = runs["hard_brake_aeb"]
    stats = metrics.truck_stats(r)
    assert metrics.contacts(r) == [], _why(r, "contact")
    assert min(s.min_gap_drawn_m for s in stats[1:]) >= AEB_MIN_GAP_M, _why(r, "no room left")


def test_with_aeb_a_harsh_brake_leaves_acc_switched_off(runs):
    """Each AEB stop disarms ACC until the driver taps resume. Target 0: ACC should not need AEB."""
    r = runs["hard_brake_aeb"]
    disarmed = sum(s.disarmed for s in metrics.truck_stats(r)[1:])
    assert disarmed <= BASELINE_HARD_BRAKE_AEB_DISARMED, _why(r, f"{disarmed} disarmed")


def test_with_aeb_a_packet_blackout_during_a_harsh_brake_ends_without_contact(runs):
    r = runs["blackout_aeb"]
    assert metrics.contacts(r) == [], _why(r, "contact")


def test_with_aeb_a_full_pedal_stop_ends_without_contact(runs):
    r = runs["stationary_lock"]
    assert metrics.contacts(r) == [], _why(r, "contact")


def test_trucks_locked_by_an_aeb_stop_stay_put_and_the_rest_wait_behind_them(runs):
    """Nobody taps resume: disarmed trucks must not roll, and ACC behind them must not creep."""
    r = runs["stationary_lock"]
    rolled = metrics.moved_after_disarm(r)
    assert max(rolled) <= CREEP_TOL_M, _why(r, f"moved after disarm {rolled}")
    creep = metrics.standstill_creep(r, scenarios.EVENT_S, r.scenario.duration_s)
    assert max(creep) <= CREEP_TOL_M, _why(r, f"creep {creep}")


def test_a_queue_stop_ends_without_contact_or_creep(runs):
    r = runs["queue_stop"]
    assert metrics.contacts(r) == [], _why(r, "contact")
    creep = metrics.standstill_creep(r, scenarios.EVENT_S, scenarios.QUEUE_GO_S)
    assert max(creep) <= CREEP_TOL_M, _why(r, f"creep {creep}")


def test_a_queue_stops_with_room_between_trucks(runs):
    """ACC's standstill target is 5 m; trucks bunching up to a metre is the jam locking up."""
    r = runs["queue_stop"]
    gaps = metrics.stop_gaps(r, scenarios.QUEUE_GO_S - 0.5)[1:]
    assert min(gaps) >= BASELINE_QUEUE_STOP_GAP_M, _why(r, f"stop gaps {gaps}")


def test_a_stopped_queue_drives_off_again(runs):
    """No stationary lock: every truck rolls off after the one ahead, without the driver."""
    r = runs["queue_stop"]
    hops = relaunch_hops(r)
    assert None not in hops, _why(r, f"never drove off: {hops}")
    assert max(hops) <= BASELINE_QUEUE_RELAUNCH_HOP_S, _why(r, f"relaunch hops {hops}")


def test_a_gentle_pull_away_is_followed_promptly(runs):
    """Lead eases off at 0.3 m/s^2 from rest; the first follower rolls soon after."""
    r = runs["slow_pull_away"]
    launch = metrics.relaunch_times(r, scenarios.PULL_AWAY_S)
    assert None not in launch, _why(r, f"never drove off: {launch}")
    start = launch[1] - launch[0]
    assert start <= BASELINE_PULL_AWAY_START_S, _why(r, f"first follower {start:.2f} s")
    hops = [launch[k] - launch[k - 1] for k in range(1, len(launch))]
    assert max(hops) <= TARGET_RELAUNCH_HOP_S, _why(r, f"relaunch hops {hops}")
    assert metrics.contacts(r) == [], _why(r, "contact")


def test_a_creeping_lead_is_followed_without_stopping(runs):
    """Lead inches at 2 km/h, below where acc_speed reads non-zero, and never stops."""
    r = runs["creep"]
    stops = metrics.restops(r, scenarios.PULL_AWAY_S)
    assert sum(stops) <= BASELINE_CREEP_RESTOPS, _why(r, f"re-stops {stops}")
    assert metrics.contacts(r) == [], _why(r, "contact")


def test_the_first_followers_creep_along_without_the_hold_stopping_them(runs):
    """Steady speed keeping behind a crawl commands about zero; the hold must not read it as a stop."""
    r = runs["creep"]
    caps = metrics.hold_captures(r, CREEP_STEADY_FROM_S, r.scenario.duration_s)
    assert caps[1:3] == [0, 0], _why(r, f"hold captures {caps}")


def test_a_desync_snap_barely_moves_the_brake(runs):
    """Lead drawn 8 m closer for 0.3 s: the radar holds it as a rewind. Target 0.5 m/s^2."""
    r = runs["desync_snap"]
    peak = snap_brake(r)
    assert peak <= BASELINE_SNAP_BRAKE_MS2, _why(r, f"brake {peak:.2f} m/s^2")
    assert not any(r.traces[1].overlay), _why(r, "TTC overlay tripped")


def test_a_laggy_client_mid_convoy_causes_no_contact(runs):
    r = runs["laggy_client"]
    assert metrics.contacts(r) == [], _why(r, "contact")
    brakes = metrics.unprovoked_brakes(r)
    assert sum(brakes) <= BASELINE_LAGGY_UNPROVOKED_BRAKES, _why(r, f"unprovoked {brakes}")


def test_every_baseline_is_still_short_of_its_target():
    """Once a baseline reaches its target, assert the target and delete the baseline."""
    pairs = [
        (BASELINE_STEADY_UNPROVOKED_BRAKES, TARGET_UNPROVOKED_BRAKES, 1),
        (BASELINE_LAGGY_UNPROVOKED_BRAKES, TARGET_UNPROVOKED_BRAKES, 1),
        (BASELINE_STEADY_SPEED_STD_KMH, TARGET_SPEED_STD_KMH, 1),
        (BASELINE_SLOWDOWN_HOP_GAIN, TARGET_HOP_GAIN, 1),
        (BASELINE_HARD_BRAKE_STOPPED, TARGET_STOPPED, 1),
        (BASELINE_FULL_STOP_CONTACTS, TARGET_CONTACTS, 1),
        (BASELINE_HARD_BRAKE_UNDERSHOOT_KMH, TARGET_UNDERSHOOT_KMH, 1),
        (BASELINE_HARD_BRAKE_AEB_DISARMED, TARGET_DISARMED, 1),
        (BASELINE_QUEUE_RELAUNCH_HOP_S, TARGET_RELAUNCH_HOP_S, 1),
        (BASELINE_SNAP_BRAKE_MS2, TARGET_SNAP_BRAKE_MS2, 1),
        (BASELINE_QUEUE_STOP_GAP_M, TARGET_STOP_GAP_M, -1),
        (BASELINE_PULL_AWAY_START_S, TARGET_PULL_AWAY_START_S, 1),
        (BASELINE_CREEP_RESTOPS, TARGET_CREEP_RESTOPS, 1),
    ]
    for baseline, target, sign in pairs:
        assert sign * (baseline - target) > 0.0, (baseline, target)
