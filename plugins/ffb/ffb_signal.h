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

/* Steering servo (layout 5): a damped PID on the wheel angle at the game's FFB rate, added to the game's own force.
   MonoCruise sends the target with its rate and acceleration, so the target moves between its 20 Hz updates instead of
   stepping, and may ask for the rim's own force pushed ahead and the wheel read ahead by its delay (both off by
   default). It can also cancel a share of the game's steady force (off by default). It fades in and out and halves on
   a buzz. */

/* Hard maximums. MonoCruise asks for 3000 unless LA's la_servo_cap says more (Lukas, 2026-11-06), never above 6000. */
#define FFB_SERVO_CAP 6000.0
#define FFB_ASSIST_CAP 6500.0
#define FFB_CANCEL_CAP 6000.0
#define FFB_SERVO_IN_S 0.3
#define FFB_SERVO_OUT_S 0.2
/* Slew 150000 per second and damping up to half the servo cap: Lukas, 2026-11-06, for fast moves (was 30000, 1500). */
#define FFB_SERVO_SLEW 150000.0
#define FFB_STEER_STALE_S 0.15
#define FFB_STEER_RATE_TAU_S 0.04
#define FFB_RATE_FAST_TAU_S 0.006
#define FFB_EXTRAP_S 0.15
#define FFB_D_SHARE 0.5
#define FFB_LEAD_MAX_S 0.08
#define FFB_FF_STILL_RATE 0.02
#define FFB_FF_REST_BAND 0.002
#define FFB_STUCK_FROM 1.5
#define FFB_BREAK_RAMP_S 0.3
#define FFB_BREAK_BAND 0.003
#define FFB_BREAK_HOLD 0.0015
/* The settle law (servo flag 4, status 4096): the push finishes a move down to a band round the target, then parks. */
#define FFB_SETTLE_BAND 0.0005
#define FFB_SETTLE_RISE 0.0005
#define FFB_SETTLE_RAMP_S 1.0
#define FFB_SETTLE_DROP_S 0.02
#define FFB_SETTLE_STEP_MAX 0.005
#define FFB_SETTLE_RELEASE 0.001
#define FFB_I_MOVE_RATE 0.3
#define FFB_D_LAG_MAX_S 0.05
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
    double ff;
    double d_ref;
    int held;
    double held_u;
    double u_prev;
    double ff_prev;
} FfbServo;

/* What MonoCruise asks for: ``target`` and ``rate`` in counterclockwise lock, gains per lock, ``dir`` the torque sign
   that turns the wheel counterclockwise (0 is off), ``only`` +1 or -1 the one way it may push (0 both),
   ``damp_first`` 1 gives the damping its room under the cap before P and I get theirs. ``accel`` is the target's
   acceleration; ``ff_*`` the rim's inertia, damping and friction pushed ahead (0 is none) on ``ff_rate`` and
   ``ff_accel``: the target's own motion, or the motion MonoCruise plans a wheel delay from now. ``ff_breakaway`` is
   where the friction push climbs to as ``stuck`` (0 to 1) ramps while the wheel stands, when above ``ff_friction``. */
typedef struct FfbServoIn {
    double target;
    double rate;
    double kp;
    double kd;
    double ki;
    double cap;
    int dir;
    int only;
    int damp_first;
    double accel;
    double ff_inertia;
    double ff_damping;
    double ff_friction;
    double ff_breakaway;
    double stuck;
    double ff_rate;
    double ff_accel;
    double d_lag;
    double i_move;
    int settle;
    double still_s;
    double interval;
    double step;
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
    s->ff = 0.0;
    s->d_ref = 0.0;
    s->held = 0;
    s->held_u = 0.0;
    s->u_prev = 0.0;
    s->ff_prev = 0.0;
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

static double ffb_finite(double x) {
    return x == x && x < 1e30 && x > -1e30 ? x : 0.0;
}

/* The same with its acceleration: where the target is ``age`` seconds on, and how fast it moves there. */
static void ffb_servo_aim(double target, double rate, double accel, double age, double *aim, double *aim_rate) {
    double a = age > 0.0 ? (age < FFB_EXTRAP_S ? age : FFB_EXTRAP_S) : 0.0;
    rate = ffb_finite(rate);
    accel = ffb_finite(accel);
    *aim = target + rate * a + 0.5 * accel * a * a;
    *aim_rate = rate + accel * a;
}

/* How far ahead to read the wheel when MonoCruise asks for a lead (the wheel's own delay): that lead and the steering
   reading's age (it comes at the game's frame rate), together at most FFB_LEAD_MAX_S. No lead reads it as it is. */
static double ffb_servo_ahead(double lead, double reading_age) {
    double ahead;
    if (!(lead > 0.0)) {
        return 0.0;
    }
    ahead = lead + (reading_age > 0.0 ? reading_age : 0.0);
    return ahead < FFB_LEAD_MAX_S ? ahead : FFB_LEAD_MAX_S;
}

/* The breakaway ramp: 0 while the wheel's reading changed within FFB_STUCK_FROM steering readings, then up to 1 over
   FFB_BREAK_RAMP_S. Read off the angle, not its rate: a coarse encoder reads a slow slide as steps with still readings. */
static double ffb_servo_stuck_over(double still_s, double interval, double ramp) {
    double n;
    if (!(interval > 0.0) || !(still_s >= 0.0)) {
        return 0.0;
    }
    n = (still_s - FFB_STUCK_FROM * interval) / ramp;
    return n > 0.0 ? (n < 1.0 ? n : 1.0) : 0.0;
}

static double ffb_servo_stuck(double still_s, double interval) {
    return ffb_servo_stuck_over(still_s, interval, FFB_BREAK_RAMP_S);
}

/* The reading's step as the settle law sees it: the smallest change lately, creeping up 0.1 % a change, bounded. */
static double ffb_settle_step(double step, double change) {
    double grown = step * 1.001;
    if (!(change > 0.0)) {
        return step;
    }
    step = step > 0.0 ? (change < grown ? change : grown) : change;
    return step < FFB_SETTLE_STEP_MAX ? step : FFB_SETTLE_STEP_MAX;
}

/* The force the rim needs to move as asked: its inertia times ``ff_accel``, its damping times ``ff_rate`` and its
   friction in the way it moves, within the cap. At rest the friction push turns toward the target (``err``, full
   from FFB_FF_REST_BAND), so a wheel held short or past by friction finishes the move instead of creeping. A wheel
   that standing needs more to start than to slide has the push climb toward ``ff_breakaway`` while it stands, once
   the target moves or is past FFB_BREAK_HOLD (full from FFB_BREAK_BAND), and no rest push within FFB_BREAK_HOLD: the
   old wheel started on 311 to 526 from one try to the next, a fixed push left it standing or lurched it, and the rest
   push hunted it 1.5 mlock either side of the target at 4 Hz (2026-11-06). */
static double ffb_servo_ff(const FfbServoIn *in, double cap, double err) {
    double ff = 0.0;
    if (in->ff_inertia > 0.0) {
        ff += in->ff_inertia * ffb_finite(in->ff_accel);
    }
    if (in->ff_damping > 0.0) {
        ff += in->ff_damping * ffb_finite(in->ff_rate);
    }
    if (in->ff_friction > 0.0) {
        double moving = tanh(ffb_finite(in->ff_rate) / FFB_FF_STILL_RATE);
        double friction = in->ff_friction;
        double rest = (1.0 - fabs(moving)) * tanh(ffb_finite(err) / FFB_FF_REST_BAND);
        if (in->ff_breakaway > friction && in->settle) {
            double band = FFB_SETTLE_BAND + 0.5 * in->step;
            double near = (fabs(ffb_finite(err)) - band) / FFB_SETTLE_RISE;
            int close = fabs(ffb_finite(err)) < FFB_BREAK_BAND && fabs(moving) < 0.5;
            double base = close ? 0.0 : friction;
            double stuck = ffb_servo_stuck_over(in->still_s, in->interval, close ? FFB_SETTLE_RAMP_S : FFB_BREAK_RAMP_S);
            double gate;
            near = near > 0.0 ? (near < 1.0 ? near : 1.0) : 0.0;
            gate = fabs(moving) > near ? fabs(moving) : near;
            rest = close ? (1.0 - fabs(moving)) * copysign(near, ffb_finite(err)) : rest * near;
            friction = base + (ffb_finite(in->ff_breakaway) - base) * stuck * gate;
        } else if (in->ff_breakaway > friction) {
            double stuck = in->stuck > 0.0 ? (in->stuck < 1.0 ? in->stuck : 1.0) : 0.0;
            double off = (fabs(ffb_finite(err)) - FFB_BREAK_HOLD) / (FFB_BREAK_BAND - FFB_BREAK_HOLD);
            double gate;
            off = off > 0.0 ? (off < 1.0 ? off : 1.0) : 0.0;
            gate = fabs(moving) > off ? fabs(moving) : off;
            rest *= off;
            friction += (ffb_finite(in->ff_breakaway) - friction) * stuck * gate;
        }
        ff += friction * (moving + rest);
    }
    return ffb_clamp(ff, cap);
}

static void ffb_settle_reset(FfbServo *s) {
    s->held = 0;
    s->u_prev = 0.0;
    s->ff_prev = 0.0;
}

/* The settle law's part of a tick: the friction push lets go over FFB_SETTLE_DROP_S instead of at once, and a wheel
   standing within the band of a still target parks: the output is held as it was and the integral stands. */
static void ffb_settle_note(FfbServo *s, const FfbServoIn *in, double e, double dt) {
    double band = FFB_SETTLE_BAND + 0.5 * in->step;
    double release = in->step > FFB_SETTLE_RELEASE ? in->step : FFB_SETTLE_RELEASE;
    int standing;
    int calm;
    if (!in->settle) {
        s->held = 0;
        return;
    }
    if (fabs(s->ff) < fabs(s->ff_prev) && s->ff * s->ff_prev >= 0.0) {
        s->ff = s->ff_prev + (s->ff - s->ff_prev) * (1.0 - exp(-dt / FFB_SETTLE_DROP_S));
    }
    s->ff_prev = s->ff;
    standing = in->interval > 0.0 && in->still_s >= FFB_STUCK_FROM * in->interval;
    calm = fabs(ffb_finite(in->rate)) < FFB_FF_STILL_RATE && fabs(ffb_finite(in->ff_rate)) < FFB_FF_STILL_RATE;
    if (s->held && (fabs(e) > band + release || !calm)) {
        s->held = 0;
    } else if (!s->held && standing && calm && fabs(e) <= band) {
        s->held = 1;
        s->held_u = s->u_prev;
    }
}

/* With the plan the push ahead moves the rim on time, so the feedback must not also chase what the wheel reading
   shows late: the damping compares the wheel's rate with the target's lagged as the reading lags (half the interval
   between steering readings plus the rate filter), and the integral holds while the target moves faster than
   FFB_I_MOVE_RATE (it wound up on the reading's lag during fast moves and pushed past the stop, 2026-11-06). */
static void ffb_servo_plan_feedback(FfbServoIn *in, int plan, double reading_interval) {
    double lag = 0.5 * (reading_interval > 0.0 ? reading_interval : 0.0) + FFB_RATE_FAST_TAU_S;
    in->d_lag = plan ? (lag < FFB_D_LAG_MAX_S ? lag : FFB_D_LAG_MAX_S) : 0.0;
    in->i_move = plan ? FFB_I_MOVE_RATE : 0.0;
}

/* What the push ahead moves on: the plan MonoCruise sent (``plan`` 1), moved on at its acceleration for at most
   FFB_EXTRAP_S since it came, or else the target's own rate and acceleration where the servo aims. */
static void ffb_servo_ff_motion(FfbServoIn *in, int plan, double plan_rate, double plan_accel, double age) {
    double a = age > 0.0 ? (age < FFB_EXTRAP_S ? age : FFB_EXTRAP_S) : 0.0;
    if (plan) {
        in->ff_accel = ffb_finite(plan_accel);
        in->ff_rate = ffb_finite(plan_rate) + in->ff_accel * a;
    } else {
        in->ff_accel = in->accel;
        in->ff_rate = in->rate;
    }
}

/* The three terms summed in the counterclockwise axis and held to ``cap``. With ``damp_first`` P and I get only the room
   the cap leaves after D, so a servo pinned at its cap still brakes a wheel that swings through the target. */
static double ffb_servo_sum(double p, double d, double integ, double cap, int damp_first) {
    double pi = p + integ;
    double room;
    if (damp_first) {
        room = cap - fabs(d);
        pi = ffb_clamp(pi, room > 0.0 ? room : 0.0);
    }
    return ffb_clamp(pi + d, cap);
}

/* One tick of the PID in the counterclockwise axis, turned into a torque sign by ``dir``; ``in->rate`` is the target's
   rate at ``target``. The integral grows only while the output is not pinned the same way, and stays within the cap.
   The damping is capped at half the cap; the push ahead rides with P. With ``only`` nothing pushes the other way. */
static double ffb_servo_pid(FfbServo *s, const FfbServoIn *in, double target, double angle, double rate, double dt) {
    double cap = ffb_servo_cap(in->cap);
    double e;
    double p;
    double d;
    double u;
    double room;
    int moving;
    dt = ffb_dt_clamp(dt);
    s->ff = 0.0;
    if (!in->dir || !(in->kp >= 0.0) || !(in->kd >= 0.0) || !(in->ki >= 0.0)) {
        s->integ = 0.0;
        s->d_ref = 0.0;
        ffb_settle_reset(s);
        return 0.0;
    }
    e = target - angle;
    s->ff = ffb_servo_ff(in, cap, e);
    ffb_settle_note(s, in, e, dt);
    p = in->kp * e + s->ff;
    if (in->d_lag > 0.0) {
        s->d_ref += (in->rate - s->d_ref) * (1.0 - exp(-dt / in->d_lag));
    } else {
        s->d_ref = in->rate;
    }
    d = ffb_clamp(in->kd * (s->d_ref - rate), FFB_D_SHARE * cap);
    room = in->damp_first ? cap - fabs(d) : cap;
    u = in->damp_first ? p + s->integ : p + d + s->integ;
    moving = in->i_move > 0.0 && (fabs(in->rate) > in->i_move || fabs(ffb_finite(in->ff_rate)) > in->i_move);
    if (!moving && !s->held && (fabs(u) < room || u * e < 0.0)) {
        s->integ = ffb_clamp(s->integ + in->ki * e * dt, FFB_I_SHARE * cap);
    }
    if (in->only && s->integ * in->only < 0.0) {
        s->integ = 0.0;
    }
    u = s->held ? ffb_clamp(s->held_u, cap) : ffb_servo_sum(p, d, s->integ, cap, in->damp_first);
    s->u_prev = u;
    if (in->only) {
        return u * in->only < 0.0 ? 0.0 : (double)in->dir * u;
    }
    return (double)in->dir * u;
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
        ffb_settle_reset(s);
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
