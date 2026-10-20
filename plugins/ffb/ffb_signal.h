#ifndef MONOCRUISE_FFB_SIGNAL_H
#define MONOCRUISE_FFB_SIGNAL_H

#include <math.h>

/* A 12 Hz carrier under a slow envelope. The carrier has zero mean and the envelope starts
   and ends at zero with zero slope, so the rumble never pushes the wheel to one side.
   Units are DirectInput magnitude, where 10000 is full scale. */

#define FFB_FULL_SCALE 10000.0
#define FFB_PEAK_CAP 3000.0
#define FFB_RUMBLE_HZ 12.0
#define FFB_ATTACK_S 0.25
#define FFB_RELEASE_S 0.30
#define FFB_DT_CAP_S 0.05
#define FFB_STALE_S 0.25
#define FFB_PI 3.14159265358979323846

typedef struct FfbWave {
    double gain;
    double phase;
    double amp;
} FfbWave;

static double ffb_ease(double u) {
    if (u <= 0.0) {
        return 0.0;
    }
    if (u >= 1.0) {
        return 1.0;
    }
    return 0.5 - 0.5 * cos(FFB_PI * u);
}

static double ffb_amp_clamp(double amp) {
    if (!(amp > 0.0)) {
        return 0.0;
    }
    return amp > FFB_PEAK_CAP ? FFB_PEAK_CAP : amp;
}

static double ffb_dt_clamp(double dt) {
    if (!(dt > 0.0)) {
        return 0.0;
    }
    return dt > FFB_DT_CAP_S ? FFB_DT_CAP_S : dt;
}

static double ffb_gain_step(double gain, int on, double dt) {
    gain += on ? dt / FFB_ATTACK_S : -dt / FFB_RELEASE_S;
    if (gain < 0.0) {
        return 0.0;
    }
    return gain > 1.0 ? 1.0 : gain;
}

/* One tick of the rumble. Asking for silence ramps the last amplitude out, never a step. */
static double ffb_wave_step(FfbWave *w, double amp, int on, double dt) {
    dt = ffb_dt_clamp(dt);
    if (on) {
        w->amp = ffb_amp_clamp(amp);
    }
    w->gain = ffb_gain_step(w->gain, on && w->amp > 0.0, dt);
    if (w->gain <= 0.0) {
        w->phase = 0.0;
        return 0.0;
    }
    w->phase += 2.0 * FFB_PI * FFB_RUMBLE_HZ * dt;
    if (w->phase >= 2.0 * FFB_PI) {
        w->phase -= 2.0 * FFB_PI;
    }
    return w->amp * ffb_ease(w->gain) * sin(w->phase);
}

static long ffb_round(double x) {
    return (long)(x >= 0.0 ? x + 0.5 : x - 0.5);
}

static long ffb_apply(long game, long delta) {
    long sum = game + delta;
    if (sum > 10000) {
        sum = 10000;
    }
    if (sum < -10000) {
        sum = -10000;
    }
    return sum;
}

#endif
