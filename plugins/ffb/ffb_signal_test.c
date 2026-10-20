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

int main(void) {
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
