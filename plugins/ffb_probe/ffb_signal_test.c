#include "ffb_signal.h"

#include <stdio.h>

static int g_fails;

static void expect(int cond, const char *msg) {
    if (!cond) {
        fprintf(stderr, "fail: %s\n", msg);
        g_fails++;
    }
}

static void test_the_rumble_fades_in_and_shakes_both_ways(void) {
    const double dt = 1.0 / 120.0;
    double t;
    double sum = 0.0;
    double lo = 0.0;
    double hi = 0.0;
    int n = 0;
    expect(ffb_signal_sample(0.0) == 0.0, "starts at zero");
    expect(ffb_signal_sample(dt) > -8.0 && ffb_signal_sample(dt) < 8.0, "first tick is not a step");
    expect(ffb_signal_phase(0.1) == FFB_PH_RUMBLE, "rumble phase");
    for (t = FFB_RUMBLE_IN_S; t < FFB_RUMBLE_IN_S + FFB_RUMBLE_HOLD_S; t += dt) {
        double v = ffb_signal_sample(t);
        sum += v;
        if (n == 0 || v < lo) {
            lo = v;
        }
        if (n == 0 || v > hi) {
            hi = v;
        }
        n++;
        expect(v < 1700.0 && v > -1700.0, "rumble stays inside its amplitude");
    }
    expect(n > 0, "rumble hold was sampled");
    expect(sum / n < 160.0 && sum / n > -160.0, "rumble hold is zero mean");
    expect(hi > 1400.0 && lo < -1400.0, "rumble reaches both sides at the higher amplitude");
    expect(ffb_signal_sample(FFB_END_S + 0.5) == 0.0, "rumble ends at zero");
    expect(ffb_signal_phase(FFB_END_S + 0.1) == FFB_PH_DONE, "done phase");
}

static void test_a_stall_cannot_skip_the_ramp(void) {
    expect(ffb_advance(0.0, 1.0) == FFB_DT_CAP_S, "a one second stall moves one cap");
    expect(ffb_signal_sample(ffb_advance(0.0, 1.0)) > -80.0 && ffb_signal_sample(ffb_advance(0.0, 1.0)) < 80.0,
           "stall does not reach full rumble");
}

static void test_the_pattern_repeats_after_ten_seconds(void) {
    expect(ffb_signal_sample(FFB_END_S + 9.0) == 0.0, "the ten second gap stays silent");
    expect(ffb_signal_phase(FFB_END_S + 9.0) == FFB_PH_DONE, "the gap sits between repeats");
    expect(ffb_signal_phase(FFB_CYCLE_S + 0.2) == FFB_PH_RUMBLE, "the next cycle rumbles");
    expect(ffb_signal_sample(FFB_CYCLE_S + FFB_RUMBLE_IN_S + 0.05) != 0.0, "the next rumble is audible");
}

static void test_sum_keeps_the_game_force_inside_the_rail(void) {
    expect(ffb_round(-1600.2) == -1600, "rounds half away from zero on the negative side");
    expect(ffb_apply(0, 0) == 0, "zero stays zero");
    expect(ffb_apply(4000, -1600) == 2400, "game force is kept and summed");
    expect(ffb_apply(9500, 1600) == 10000, "sum stops at full scale");
    expect(ffb_apply(-9500, -1600) == -10000, "sum stops at negative full scale");
    expect(ffb_apply(10000, 500) == 10000, "no headroom leaves the game force");
    expect(ffb_apply(10000, -1600) == 8400, "a left add can leave the positive rail");
}

int main(void) {
    test_the_rumble_fades_in_and_shakes_both_ways();
    test_a_stall_cannot_skip_the_ramp();
    test_the_pattern_repeats_after_ten_seconds();
    test_sum_keeps_the_game_force_inside_the_rail();
    if (g_fails) {
        fprintf(stderr, "%d failed\n", g_fails);
        return 1;
    }
    printf("ok\n");
    return 0;
}
