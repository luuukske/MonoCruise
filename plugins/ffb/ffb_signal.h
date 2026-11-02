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

/* Steering torque for the LA: a bias, not a wave. The request is low-passed and then
   rate-limited, so neither MonoCruise's 20 Hz staircase nor a lost heartbeat reaches the rim as a step. */

#define FFB_TORQUE_CAP 3000.0
#define FFB_TORQUE_TAU_S 0.06
#define FFB_TORQUE_RISE 6000.0
#define FFB_TORQUE_FALL 12000.0
#define FFB_FORCE_TAU_S 0.10

typedef struct FfbTorque {
    double smooth;
    double value;
} FfbTorque;

static double ffb_torque_clamp(double torque) {
    if (!(torque == torque)) {
        return 0.0;
    }
    if (torque > FFB_TORQUE_CAP) {
        return FFB_TORQUE_CAP;
    }
    return torque < -FFB_TORQUE_CAP ? -FFB_TORQUE_CAP : torque;
}

/* One tick. ``on`` false fades the torque out at the fall rate; growing away from zero uses the rise rate. */
static double ffb_torque_step(FfbTorque *s, double target, int on, double dt) {
    double step;
    double limit;
    int rising;
    dt = ffb_dt_clamp(dt);
    target = on ? ffb_torque_clamp(target) : 0.0;
    s->smooth += (target - s->smooth) * (1.0 - exp(-dt / FFB_TORQUE_TAU_S));
    step = s->smooth - s->value;
    rising = fabs(s->smooth) > fabs(s->value) && s->smooth * s->value >= 0.0;
    limit = (rising ? FFB_TORQUE_RISE : FFB_TORQUE_FALL) * dt;
    if (step > limit) {
        step = limit;
    } else if (step < -limit) {
        step = -limit;
    }
    s->value += step;
    if (!on && fabs(s->value) < 0.5 && fabs(s->smooth) < 0.5) {
        s->value = 0.0;
        s->smooth = 0.0;
    }
    return s->value;
}

/* Steering servo (layout 4): a damped PID on the wheel angle at the game's FFB rate, added to the game's own force.
   MonoCruise sends the target with its rate, so the target moves between its 20 Hz updates instead of stepping. It can
   also cancel a share of the game's steady force (off by default). It fades in and out and halves on a buzz. */

#define FFB_SERVO_CAP 3000.0
#define FFB_ASSIST_CAP 4000.0
#define FFB_CANCEL_CAP 6000.0
#define FFB_SERVO_IN_S 0.3
#define FFB_SERVO_OUT_S 0.2
#define FFB_SERVO_SLEW 30000.0
#define FFB_STEER_STALE_S 0.15
#define FFB_STEER_RATE_TAU_S 0.04
#define FFB_RATE_FAST_TAU_S 0.006
#define FFB_EXTRAP_S 0.15
#define FFB_D_CAP 1500.0
#define FFB_I_SHARE 1.0
#define FFB_CANCEL_TAU_S 1.5
#define FFB_BUZZ_MIN 300.0
#define FFB_BUZZ_SWING 0.002
#define FFB_BUZZ_FLIPS 4
#define FFB_BUZZ_WINDOW_S 0.5
#define FFB_BUZZ_CUT 0.5
#define FFB_BUZZ_FLOOR 0.1
#define FFB_BUZZ_RECOVER 0.25

typedef struct FfbServo {
    double gain;
    double value;
    double scale;
    double clock;
    double integ;
    double flips[FFB_BUZZ_FLIPS];
    int nflips;
    int sign;
    int pushed;
    int swing;
    int seen;
    double peak;
} FfbServo;

/* What MonoCruise asks for: ``target`` and ``rate`` in counterclockwise lock, gains per lock, ``dir`` the torque sign
   that turns the wheel counterclockwise (0 is off), ``only`` +1 or -1 the one way it may push (0 both). */
typedef struct FfbServoIn {
    double target;
    double rate;
    double kp;
    double kd;
    double ki;
    double cap;
    int dir;
    int only;
} FfbServoIn;

static void ffb_servo_init(FfbServo *s) {
    int i;
    s->gain = 0.0;
    s->value = 0.0;
    s->scale = 1.0;
    s->clock = 0.0;
    s->integ = 0.0;
    s->nflips = 0;
    s->sign = 0;
    s->pushed = 0;
    s->swing = 0;
    s->seen = 0;
    s->peak = 0.0;
    for (i = 0; i < FFB_BUZZ_FLIPS; i++) {
        s->flips[i] = 0.0;
    }
}

static double ffb_clamp(double x, double cap) {
    if (!(x == x)) {
        return 0.0;
    }
    if (x > cap) {
        return cap;
    }
    return x < -cap ? -cap : x;
}

static double ffb_servo_cap(double cap) {
    return cap > 0.0 ? (cap < FFB_SERVO_CAP ? cap : FFB_SERVO_CAP) : 0.0;
}

/* The target ``age`` seconds after MonoCruise sent it, moved on at its rate for at most ``FFB_EXTRAP_S``. */
static double ffb_servo_target(double target, double rate, double age) {
    if (!(age > 0.0) || !(rate == rate)) {
        return target;
    }
    return target + rate * (age < FFB_EXTRAP_S ? age : FFB_EXTRAP_S);
}

/* One tick of the PID in the counterclockwise axis, turned into a torque sign by ``dir``. The integral grows only while
   the output is not pinned the same way, and stays within the cap. With ``only`` nothing pushes the other way at all. */
static double ffb_servo_pid(FfbServo *s, const FfbServoIn *in, double target, double angle, double rate, double dt) {
    double cap = ffb_servo_cap(in->cap);
    double e;
    double p;
    double d;
    double u;
    dt = ffb_dt_clamp(dt);
    if (!in->dir || !(in->kp >= 0.0) || !(in->kd >= 0.0) || !(in->ki >= 0.0)) {
        s->integ = 0.0;
        return 0.0;
    }
    e = target - angle;
    p = in->kp * e;
    d = ffb_clamp(in->kd * (in->rate - rate), FFB_D_CAP);
    u = p + d + s->integ;
    if (fabs(u) < cap || u * e < 0.0) {
        s->integ = ffb_clamp(s->integ + in->ki * e * dt, FFB_I_SHARE * cap);
    }
    if (in->only) {
        s->integ = s->integ * in->only < 0.0 ? 0.0 : s->integ;
        u = p + d + s->integ;
        return u * in->only < 0.0 ? 0.0 : (double)in->dir * ffb_clamp(u, cap);
    }
    return (double)in->dir * ffb_clamp(p + d + s->integ, cap);
}

/* A share of the game's steady force (low-passed at 1.5 s, so turn-in and road feel pass) removed while the servo
   holds: ``cancel`` of it, faded with the servo, bounded. */
static double ffb_cancel_term(double game_lp, double cancel, double fade) {
    if (!(cancel > 0.0) || !(game_lp == game_lp)) {
        return 0.0;
    }
    cancel = cancel > 1.0 ? 1.0 : cancel;
    return ffb_clamp(-cancel * fade * game_lp, FFB_CANCEL_CAP);
}

/* A swing is the wheel turning back by FFB_BUZZ_SWING from its last extreme with the push flipped since the last
   swing: a ring. ``angle`` NaN (the plugin's own rumble shakes the wheel) counts nothing and starts the watch over. */
static void ffb_buzz_note(FfbServo *s, double want, double angle) {
    int sign = want > FFB_BUZZ_MIN ? 1 : (want < -FFB_BUZZ_MIN ? -1 : 0);
    int i;
    int kept = 0;
    if (sign && sign != s->sign) {
        s->pushed = s->pushed || s->sign != 0;
        s->sign = sign;
    }
    if (!(angle == angle)) {
        s->seen = 0;
        s->swing = 0;
        s->pushed = 0;
        return;
    }
    if (!s->seen) {
        s->peak = angle;
        s->seen = 1;
        return;
    }
    if (!s->swing) {
        if (fabs(angle - s->peak) >= FFB_BUZZ_SWING) {
            s->swing = angle > s->peak ? 1 : -1;
            s->peak = angle;
        }
        return;
    }
    if ((angle - s->peak) * s->swing >= 0.0) {
        s->peak = angle;
        return;
    }
    if (fabs(angle - s->peak) < FFB_BUZZ_SWING) {
        return;
    }
    s->swing = -s->swing;
    s->peak = angle;
    if (!s->pushed) {
        return;
    }
    s->pushed = 0;
    for (i = 0; i < s->nflips; i++) {
        if (s->clock - s->flips[i] <= FFB_BUZZ_WINDOW_S) {
            s->flips[kept++] = s->flips[i];
        }
    }
    s->nflips = kept;
    if (s->nflips < FFB_BUZZ_FLIPS) {
        s->flips[s->nflips++] = s->clock;
    }
    if (s->nflips >= FFB_BUZZ_FLIPS) {
        s->scale *= FFB_BUZZ_CUT;
        if (s->scale < FFB_BUZZ_FLOOR) {
            s->scale = FFB_BUZZ_FLOOR;
        }
        s->nflips = 0;
    }
}

/* One tick. ``want`` is the servo torque, ``angle`` the wheel; returns what reaches the rim. Off clears the push. */
static double ffb_servo_step(FfbServo *s, double want, int on, double angle, double dt) {
    double target;
    double step;
    double limit;
    dt = ffb_dt_clamp(dt);
    s->clock += dt;
    s->scale += FFB_BUZZ_RECOVER * dt;
    if (s->scale > 1.0) {
        s->scale = 1.0;
    }
    want = on ? ffb_clamp(want, FFB_SERVO_CAP) : 0.0;
    if (on) {
        ffb_buzz_note(s, want, angle);
    } else {
        s->integ = 0.0;
    }
    s->gain += on ? dt / FFB_SERVO_IN_S : -dt / FFB_SERVO_OUT_S;
    s->gain = s->gain < 0.0 ? 0.0 : (s->gain > 1.0 ? 1.0 : s->gain);
    target = want * s->scale * ffb_ease(s->gain);
    step = target - s->value;
    limit = FFB_SERVO_SLEW * dt;
    s->value += step > limit ? limit : (step < -limit ? -limit : step);
    if (!on && s->gain <= 0.0 && fabs(s->value) < 0.5) {
        s->value = 0.0;
        s->sign = 0;
        s->pushed = 0;
        s->swing = 0;
        s->seen = 0;
        s->nflips = 0;
    }
    return s->value;
}

/* The game's own force, low-passed before MonoCruise samples it at 20 Hz. */
static double ffb_force_step(double smooth, double game, double dt) {
    return smooth + (game - smooth) * (1.0 - exp(-ffb_dt_clamp(dt) / FFB_FORCE_TAU_S));
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
