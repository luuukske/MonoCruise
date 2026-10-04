#ifndef MONOCRUISE_FFB_SIGNAL_H
#define MONOCRUISE_FFB_SIGNAL_H

#include <math.h>

/* Cosine ramps so the added force and its slope both start and end at zero.
   The rumble is zero-mean, so it does not steer. Units are DirectInput
   magnitude, where 10000 is full scale. */

#define FFB_FULL_SCALE 10000.0
#define FFB_RUMBLE_AMP 1600.0
#define FFB_RUMBLE_HZ 12.0
#define FFB_RUMBLE_IN_S 0.40
#define FFB_RUMBLE_HOLD_S 1.35
#define FFB_RUMBLE_OUT_S 0.45
#define FFB_REPEAT_S 10.0
#define FFB_DT_CAP_S 0.05

#define FFB_END_S (FFB_RUMBLE_IN_S + FFB_RUMBLE_HOLD_S + FFB_RUMBLE_OUT_S)
#define FFB_CYCLE_S (FFB_END_S + FFB_REPEAT_S)

#define FFB_PH_WAIT 0
#define FFB_PH_PUSH 1
#define FFB_PH_GAP 2
#define FFB_PH_RUMBLE 3
#define FFB_PH_DONE 4

static double ffb_ease(double u) {
    if (u <= 0.0) {
        return 0.0;
    }
    if (u >= 1.0) {
        return 1.0;
    }
    return 0.5 - 0.5 * cos(3.14159265358979323846 * u);
}

static double ffb_rumble(double t) {
    double env;
    double span = FFB_RUMBLE_IN_S + FFB_RUMBLE_HOLD_S;
    if (t < 0.0 || t >= FFB_END_S) {
        return 0.0;
    }
    if (t < FFB_RUMBLE_IN_S) {
        env = FFB_RUMBLE_AMP * ffb_ease(t / FFB_RUMBLE_IN_S);
    } else if (t < span) {
        env = FFB_RUMBLE_AMP;
    } else {
        env = FFB_RUMBLE_AMP * (1.0 - ffb_ease((t - span) / FFB_RUMBLE_OUT_S));
    }
    return env * sin(2.0 * 3.14159265358979323846 * FFB_RUMBLE_HZ * t);
}

static double ffb_wrap(double t) {
    double m;
    if (t <= 0.0) {
        return 0.0;
    }
    m = fmod(t, FFB_CYCLE_S);
    if (m < 0.0) {
        return 0.0;
    }
    return m;
}

static double ffb_signal_sample(double t) {
    t = ffb_wrap(t);
    if (t <= 0.0 || t >= FFB_END_S) {
        return 0.0;
    }
    return ffb_rumble(t);
}

static int ffb_signal_phase(double t) {
    t = ffb_wrap(t);
    if (t > 0.0 && t < FFB_END_S) {
        return FFB_PH_RUMBLE;
    }
    if (t <= 0.0) {
        return FFB_PH_WAIT;
    }
    return FFB_PH_DONE;
}

static double ffb_advance(double t, double dt) {
    if (dt < 0.0) {
        dt = 0.0;
    }
    if (dt > FFB_DT_CAP_S) {
        dt = FFB_DT_CAP_S;
    }
    return t + dt;
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
