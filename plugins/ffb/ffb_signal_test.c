#include "ffb_signal.h"

#include <stdio.h>

static int g_fails;

static void expect(int cond, const char *msg) {
    if (!cond) {
        fprintf(stderr, "fail: %s\n", msg);
        g_fails++;
    }
}

static const double DT = 1.0 / 120.0;

static double run_on(FfbWave *w, double amp, double seconds, double *peak, double *sum) {
    double t;
    double last = 0.0;
    int n = 0;
    for (t = 0.0; t < seconds; t += DT) {
        last = ffb_wave_step(w, amp, 1, DT);
        if (peak && (n == 0 || fabs(last) > *peak)) {
            *peak = fabs(last);
        }
        if (sum) {
            *sum += last;
        }
        n++;
    }
    return last;
}

static void test_it_starts_at_zero_and_fades_in(void) {
    FfbWave w = {0.0, 0.0, 0.0};
    double first = ffb_wave_step(&w, 1600.0, 1, DT);
    expect(fabs(first) < 8.0, "first tick is not a step");
    expect(w.gain > 0.0 && w.gain < 0.1, "gain starts rising");
    run_on(&w, 1600.0, FFB_ATTACK_S + 0.1, NULL, NULL);
    expect(w.gain == 1.0, "gain reaches one after the attack");
}

static void test_the_hold_is_zero_mean_and_under_the_cap(void) {
    FfbWave w = {0.0, 0.0, 0.0};
    double peak = 0.0;
    double sum = 0.0;
    double start;
    run_on(&w, 1600.0, FFB_ATTACK_S + 0.05, NULL, NULL);
    start = 0.0;
    run_on(&w, 1600.0, 1.0, &peak, &sum);
    expect(peak > 1500.0 && peak <= 1600.0, "peak is the request");
    expect(fabs(sum * DT) < 1600.0 * 0.01, "twelve whole cycles average to zero");
    (void)start;
}

static void test_the_request_is_capped_in_the_plugin(void) {
    FfbWave w = {0.0, 0.0, 0.0};
    double peak = 0.0;
    run_on(&w, 9999.0, 1.0, &peak, NULL);
    expect(peak <= FFB_PEAK_CAP, "never above the cap");
    expect(peak > FFB_PEAK_CAP * 0.95, "reaches the cap when asked for more");
    expect(ffb_amp_clamp(-5.0) == 0.0, "negative amplitude is silence");
}

static void test_release_ramps_out_without_a_step(void) {
    FfbWave w = {0.0, 0.0, 0.0};
    double prev;
    double cur;
    double t;
    double worst = 0.0;
    run_on(&w, 2000.0, 1.0, NULL, NULL);
    prev = ffb_wave_step(&w, 0.0, 0, DT);
    for (t = DT; t < FFB_RELEASE_S + 0.1; t += DT) {
        cur = ffb_wave_step(&w, 0.0, 0, DT);
        if (fabs(cur - prev) > worst) {
            worst = fabs(cur - prev);
        }
        prev = cur;
    }
    expect(w.gain == 0.0, "gain ends at zero");
    expect(prev == 0.0 && w.phase == 0.0, "silent and phase reset");
    expect(worst < 2000.0 * 2.0 * FFB_PI * FFB_RUMBLE_HZ * DT * 1.1, "no tick jumps more than the carrier slope");
}

static void test_zero_intensity_makes_no_sound(void) {
    FfbWave w = {0.0, 0.0, 0.0};
    double peak = 0.0;
    run_on(&w, 0.0, 0.5, &peak, NULL);
    expect(peak == 0.0 && w.gain == 0.0, "an intensity of zero stays silent");
}

static void test_a_stalled_game_cannot_fast_forward_the_wave(void) {
    FfbWave a = {0.0, 0.0, 0.0};
    FfbWave b = {0.0, 0.0, 0.0};
    ffb_wave_step(&a, 1600.0, 1, 5.0);
    ffb_wave_step(&b, 1600.0, 1, FFB_DT_CAP_S);
    expect(a.gain == b.gain && a.phase == b.phase, "a long gap counts as the cap");
}

static void test_the_sum_with_the_game_force_is_clamped(void) {
    expect(ffb_apply(9500, 1600) == 10000, "positive clamp");
    expect(ffb_apply(-9500, -1600) == -10000, "negative clamp");
    expect(ffb_apply(1200, -300) == 900, "plain sum");
    expect(ffb_round(2.5) == 3 && ffb_round(-2.5) == -3, "rounds away from zero");
}

static double torque_run(FfbTorque *s, double target, int on, double seconds, double *worst_step) {
    double t;
    double prev = s->value;
    for (t = 0.0; t < seconds; t += DT) {
        double cur = ffb_torque_step(s, target, on, DT);
        if (worst_step && fabs(cur - prev) > *worst_step) {
            *worst_step = fabs(cur - prev);
        }
        prev = cur;
    }
    return prev;
}

static void test_torque_builds_without_a_step_and_settles_on_the_request(void) {
    FfbTorque s = {0.0, 0.0};
    double worst = 0.0;
    double first = ffb_torque_step(&s, 1500.0, 1, DT);
    expect(first > 0.0 && first <= FFB_TORQUE_RISE * DT + 1e-9, "first tick is bounded by the rise rate");
    expect(fabs(torque_run(&s, 1500.0, 1, 1.0, &worst) - 1500.0) < 1.0, "settles on the request");
    expect(worst <= FFB_TORQUE_RISE * DT + 1e-9, "never rises faster than the rise rate");
}

static void test_torque_is_capped_in_the_plugin(void) {
    FfbTorque s = {0.0, 0.0};
    expect(torque_run(&s, 30000.0, 1, 2.0, NULL) <= FFB_TORQUE_CAP, "never above the cap");
    s.smooth = s.value = 0.0;
    expect(torque_run(&s, -30000.0, 1, 2.0, NULL) >= -FFB_TORQUE_CAP, "never below minus the cap");
    expect(ffb_torque_clamp(nan("")) == 0.0, "nan is no torque");
}

static void test_a_lost_heartbeat_fades_the_torque_out(void) {
    FfbTorque s = {0.0, 0.0};
    double worst = 0.0;
    torque_run(&s, FFB_TORQUE_CAP, 1, 2.0, NULL);
    torque_run(&s, FFB_TORQUE_CAP, 0, 0.6, &worst);
    expect(s.value == 0.0 && s.smooth == 0.0, "fully released");
    expect(worst <= FFB_TORQUE_FALL * DT + 1e-9, "the release is a ramp, not a step");
    expect(worst > 0.0, "it did move");
}

static void test_a_sign_flip_goes_through_zero_at_the_rate_limits(void) {
    FfbTorque s = {0.0, 0.0};
    double worst = 0.0;
    torque_run(&s, 2000.0, 1, 2.0, NULL);
    expect(fabs(torque_run(&s, -2000.0, 1, 2.0, &worst) + 2000.0) < 1.0, "reaches the other side");
    expect(worst <= FFB_TORQUE_FALL * DT + 1e-9, "no tick jumps more than the fall rate");
}

static void test_the_game_force_low_pass_follows_without_overshoot(void) {
    double lp = 0.0;
    double t;
    for (t = 0.0; t < 1.0; t += DT) {
        lp = ffb_force_step(lp, 4000.0, DT);
        expect(lp <= 4000.0, "no overshoot");
    }
    expect(fabs(lp - 4000.0) < 1.0, "settles within a second");
}

static FfbServoIn servo_in(double target, double rate, int dir) {
    FfbServoIn in;
    in.target = target;
    in.rate = rate;
    in.kp = 57000.0;
    in.kd = 8600.0;
    in.ki = 13000.0;
    in.cap = 2500.0;
    in.dir = dir;
    in.only = 0;
    return in;
}

static void test_one_sided_it_never_pushes_the_other_way(void) {
    FfbServo s;
    FfbServoIn in = servo_in(0.01, 0.0, 1);
    double t;
    int ok = 1;
    in.only = 1;
    ffb_servo_init(&s);
    expect(fabs(ffb_servo_pid(&s, &in, 0.01, 0.0, 0.0, DT) - 570.0) < 2.0, "short of the target it pushes as before");
    ffb_servo_init(&s);
    expect(ffb_servo_pid(&s, &in, 0.01, 0.03, 0.0, DT) == 0.0, "past it, nothing pushes back");
    ffb_servo_init(&s);
    expect(ffb_servo_pid(&s, &in, 0.01, 0.01, 0.5, DT) == 0.0, "nor does the damping of a wheel turned on fast");
    ffb_servo_init(&s);
    for (t = 0.0; t < 1.0; t += DT) {
        ok = ok && ffb_servo_pid(&s, &in, 0.01, 0.03, 0.0, DT) == 0.0;
    }
    expect(ok && s.integ >= 0.0, "drive 12: a driver steering home was pushed back toward the line; held there, never");
    expect(ffb_servo_pid(&s, &in, 0.01, 0.0, 0.0, DT) > 500.0, "short of the target again, it pushes at once");
    in.dir = -1;
    ffb_servo_init(&s);
    expect(ffb_servo_pid(&s, &in, 0.01, 0.0, 0.0, DT) < -560.0 && ffb_servo_pid(&s, &in, 0.01, 0.03, 0.0, DT) == 0.0,
           "the way is counterclockwise whatever the wheel's torque sign");
}

static void test_the_pid_pushes_toward_the_target_in_the_wheel_sign(void) {
    FfbServo s;
    FfbServoIn in = servo_in(0.01, 0.0, 1);
    ffb_servo_init(&s);
    expect(fabs(ffb_servo_pid(&s, &in, 0.01, 0.0, 0.0, DT) - 570.0) < 2.0, "kp * error, counterclockwise");
    ffb_servo_init(&s);
    in.dir = -1;
    expect(ffb_servo_pid(&s, &in, 0.01, 0.0, 0.0, DT) < -560.0, "the other wheel sign");
    in.dir = 0;
    expect(ffb_servo_pid(&s, &in, 0.01, 0.0, 0.0, DT) == 0.0 && s.integ == 0.0, "dir 0 is off");
    in = servo_in(1.0, 0.0, 1);
    expect(ffb_servo_pid(&s, &in, 1.0, 0.0, 0.0, DT) == 2500.0, "the requested cap holds");
    in.cap = 1e6;
    expect(ffb_servo_pid(&s, &in, 1.0, 0.0, 0.0, DT) == FFB_SERVO_CAP, "never above the plugin's cap");
    in.kd = -1.0;
    expect(ffb_servo_pid(&s, &in, 1.0, 0.0, 0.0, DT) == 0.0, "a negative gain is refused");
}

static void test_the_damping_resists_the_wheel_moving_off_the_target_rate(void) {
    FfbServo s;
    FfbServoIn in = servo_in(0.0, 0.0, 1);
    ffb_servo_init(&s);
    expect(ffb_servo_pid(&s, &in, 0.0, 0.0, 0.1, DT) < -800.0, "a wheel turning counterclockwise is held back");
    in.rate = 0.1;
    ffb_servo_init(&s);
    expect(fabs(ffb_servo_pid(&s, &in, 0.0, 0.0, 0.1, DT)) < 1e-9, "moving with the target is free");
    in.rate = 0.0;
    ffb_servo_init(&s);
    expect(ffb_servo_pid(&s, &in, 0.0, 0.0, 5.0, DT) >= -FFB_D_CAP - 1e-9, "the damping is capped");
}

static void test_the_integral_takes_out_friction_and_does_not_wind_up(void) {
    FfbServo s;
    FfbServoIn in = servo_in(0.002, 0.0, 1);
    double t;
    double out = 0.0;
    ffb_servo_init(&s);
    for (t = 0.0; t < 1.0; t += DT) {
        out = ffb_servo_pid(&s, &in, 0.002, 0.0, 0.0, DT);
    }
    expect(out > 114.0 + 20.0 && s.integ > 20.0, "a wheel stuck short of the target gets more over time");
    in = servo_in(1.0, 0.0, 1);
    ffb_servo_init(&s);
    for (t = 0.0; t < 5.0; t += DT) {
        ffb_servo_pid(&s, &in, 1.0, 0.0, 0.0, DT);
    }
    expect(fabs(s.integ) < 1e-9, "pinned at the cap, the integral does not grow");
}

static void test_the_target_moves_on_between_updates_for_a_while(void) {
    expect(ffb_servo_target(0.01, 0.04, 0.05) == 0.01 + 0.04 * 0.05, "moved on at its rate");
    expect(ffb_servo_target(0.01, 0.04, 2.0) == 0.01 + 0.04 * FFB_EXTRAP_S, "for at most the extrapolation window");
    expect(ffb_servo_target(0.01, 0.04, -1.0) == 0.01, "never backwards");
    expect(ffb_servo_target(0.01, nan(""), 0.05) == 0.01, "a bad rate is ignored");
}

static void test_the_cancel_removes_the_slow_game_force_and_fades(void) {
    expect(ffb_cancel_term(1200.0, 1.0, 1.0) == -1200.0, "all of the steady force at full");
    expect(ffb_cancel_term(1200.0, 1.0, 0.5) == -600.0, "faded with the servo");
    expect(ffb_cancel_term(1200.0, 0.0, 1.0) == 0.0 && ffb_cancel_term(1200.0, -1.0, 1.0) == 0.0, "none when off");
    expect(ffb_cancel_term(1200.0, 3.0, 1.0) == -1200.0, "never more than the game force");
    expect(ffb_cancel_term(9000.0, 1.0, 1.0) == -FFB_CANCEL_CAP, "bounded");
}

static void test_the_servo_fades_in_and_out_without_a_step(void) {
    FfbServo s;
    double t;
    double prev = 0.0;
    double worst = 0.0;
    ffb_servo_init(&s);
    for (t = 0.0; t < 1.0; t += DT) {
        double cur = ffb_servo_step(&s, 2000.0, 1, 0.0, DT);
        worst = fabs(cur - prev) > worst ? fabs(cur - prev) : worst;
        prev = cur;
    }
    expect(fabs(prev - 2000.0) < 1.0, "reaches the request");
    for (t = 0.0; t < 0.5; t += DT) {
        double cur = ffb_servo_step(&s, 2000.0, 0, 0.0, DT);
        worst = fabs(cur - prev) > worst ? fabs(cur - prev) : worst;
        prev = cur;
    }
    expect(prev == 0.0 && s.gain == 0.0, "off means released");
    expect(worst <= FFB_SERVO_SLEW * DT + 1e-9, "never steps faster than the slew limit");
}

/* ``push`` and ``wheel`` swing side to side at 4 Hz (a flip every 15 ticks at 120 Hz) when set, else hold still. */
static double buzz_run(FfbServo *s, int push, int wheel, double seconds) {
    double t;
    int i = 0;
    for (t = 0.0; t < seconds; t += DT, i++) {
        int flip = (i / 15) % 2;
        ffb_servo_step(s, push ? (flip ? 1500.0 : -1500.0) : 1000.0, 1, wheel ? (flip ? 0.004 : -0.004) : 0.01, DT);
    }
    return s->scale;
}

static void test_a_ringing_wheel_halves_the_servo_and_it_comes_back_within_seconds(void) {
    FfbServo s;
    double t;
    ffb_servo_init(&s);
    buzz_run(&s, 0, 0, 0.5);
    expect(buzz_run(&s, 1, 1, 1.0) < 0.6, "the wheel swinging with the push flipping is a ring");
    for (t = 0.0; t < 4.0; t += DT) {
        ffb_servo_step(&s, 800.0, 1, 0.01, DT);
    }
    expect(s.scale == 1.0, "back within 4 s (18 s before drive 12)");
}

static void test_a_flipping_push_on_a_still_wheel_or_a_shaken_wheel_is_no_ring(void) {
    FfbServo s;
    ffb_servo_init(&s);
    buzz_run(&s, 0, 0, 0.5);
    expect(buzz_run(&s, 1, 0, 1.0) == 1.0, "drive 12: the push flipped near zero, the wheel held, pulses cut to 0.1");
    ffb_servo_init(&s);
    buzz_run(&s, 0, 0, 0.5);
    expect(buzz_run(&s, 0, 1, 1.0) == 1.0, "a wheel shaken by the road under a steady push");
}

static void test_the_plugins_own_rumble_never_counts_as_a_ring(void) {
    FfbServo s;
    double t;
    int i = 0;
    ffb_servo_init(&s);
    buzz_run(&s, 0, 0, 0.5);
    for (t = 0.0; t < 1.0; t += DT, i++) {
        ffb_servo_step(&s, (i / 5) % 2 ? 1500.0 : -1500.0, 1, NAN, DT);
    }
    expect(s.scale == 1.0, "drive 12: the 12 Hz rumble at the line shook the damping into flips, every pulse was cut");
    expect(buzz_run(&s, 1, 1, 1.0) < 0.6, "and a ring after it is caught as before");
}

int main(void) {
    test_the_pid_pushes_toward_the_target_in_the_wheel_sign();
    test_one_sided_it_never_pushes_the_other_way();
    test_the_damping_resists_the_wheel_moving_off_the_target_rate();
    test_the_integral_takes_out_friction_and_does_not_wind_up();
    test_the_target_moves_on_between_updates_for_a_while();
    test_the_cancel_removes_the_slow_game_force_and_fades();
    test_the_servo_fades_in_and_out_without_a_step();
    test_a_ringing_wheel_halves_the_servo_and_it_comes_back_within_seconds();
    test_a_flipping_push_on_a_still_wheel_or_a_shaken_wheel_is_no_ring();
    test_the_plugins_own_rumble_never_counts_as_a_ring();
    test_torque_builds_without_a_step_and_settles_on_the_request();
    test_torque_is_capped_in_the_plugin();
    test_a_lost_heartbeat_fades_the_torque_out();
    test_a_sign_flip_goes_through_zero_at_the_rate_limits();
    test_the_game_force_low_pass_follows_without_overshoot();
    test_it_starts_at_zero_and_fades_in();
    test_the_hold_is_zero_mean_and_under_the_cap();
    test_the_request_is_capped_in_the_plugin();
    test_release_ramps_out_without_a_step();
    test_zero_intensity_makes_no_sound();
    test_a_stalled_game_cannot_fast_forward_the_wave();
    test_the_sum_with_the_game_force_is_clamped();
    if (g_fails) {
        fprintf(stderr, "%d failed\n", g_fails);
        return 1;
    }
    printf("ffb signal ok\n");
    return 0;
}
