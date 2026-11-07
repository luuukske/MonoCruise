#ifndef MONOCRUISE_FFB_CMD_H
#define MONOCRUISE_FFB_CMD_H

#include <stddef.h>
#include <stdint.h>

/* Command block shared with core/lane_departure/command.py. Change both together.
   Steering values are truck.input.steering units: -1..1 of lock, counterclockwise positive. */

#define FFB_CMD_TAG L"Local\\MonoCruiseFfbCmd"
#define FFB_CMD_VERSION 5
#define FFB_CMD_SIZE 128
#define FFB_CMD_NONE 0
#define FFB_CMD_RUMBLE 1

#define FFB_ST_HOOKED 1
#define FFB_ST_FORCE_SEEN 2
#define FFB_ST_RUMBLING 4
#define FFB_ST_DEAD 8
#define FFB_ST_TORQUE 16
#define FFB_ST_SERVO 32
#define FFB_ST_STEER 64
#define FFB_ST_ONE_SIDED 128
#define FFB_ST_DAMP_FIRST 256
#define FFB_ST_PUSH_AHEAD 512
#define FFB_ST_FF_PLAN 1024
#define FFB_ST_BREAKAWAY 2048

#define FFB_SF_DAMP_FIRST 1
#define FFB_SF_FF_PLAN 2

typedef struct FfbCmd {
    volatile uint32_t version;
    volatile uint32_t heartbeat;
    volatile uint8_t command;
    volatile uint8_t side;
    volatile uint16_t intensity;
    volatile uint32_t seq;
    volatile int16_t torque;
    volatile int16_t game_force;
    volatile uint32_t status;
    volatile uint32_t frames;
    volatile int16_t applied;
    volatile int16_t servo_applied;
    volatile float servo_target;
    volatile float servo_kp;
    volatile float servo_ki;
    volatile uint16_t servo_cap;
    volatile int8_t servo_dir;
    volatile int8_t servo_only;
    volatile float steer;
    volatile float steer_rate;
    volatile uint32_t steer_frames;
    volatile float servo_scale;
    volatile float servo_rate;
    volatile float servo_kd;
    volatile float servo_cancel;
    volatile uint32_t servo_stamp;
    volatile float servo_integ;
    volatile float frame_hz;
    volatile int16_t cancel_applied;
    volatile uint8_t servo_flags;
    volatile uint8_t reserved1;
    volatile float servo_accel;
    volatile float ff_inertia;
    volatile float ff_damping;
    volatile float ff_friction;
    volatile float servo_lead;
    volatile int16_t ff_applied;
    volatile int16_t reserved3;
    volatile float plan_rate;
    volatile float plan_accel;
    volatile float ff_breakaway;
} FfbCmd;

_Static_assert(sizeof(FfbCmd) == FFB_CMD_SIZE, "command block size");
_Static_assert(offsetof(FfbCmd, heartbeat) == 4, "heartbeat offset");
_Static_assert(offsetof(FfbCmd, command) == 8, "command offset");
_Static_assert(offsetof(FfbCmd, seq) == 12, "seq offset");
_Static_assert(offsetof(FfbCmd, torque) == 16, "torque offset");
_Static_assert(offsetof(FfbCmd, game_force) == 18, "game_force offset");
_Static_assert(offsetof(FfbCmd, status) == 20, "status offset");
_Static_assert(offsetof(FfbCmd, frames) == 24, "frames offset");
_Static_assert(offsetof(FfbCmd, applied) == 28, "applied offset");
_Static_assert(offsetof(FfbCmd, servo_applied) == 30, "servo_applied offset");
_Static_assert(offsetof(FfbCmd, servo_target) == 32, "servo_target offset");
_Static_assert(offsetof(FfbCmd, servo_kp) == 36, "servo_kp offset");
_Static_assert(offsetof(FfbCmd, servo_ki) == 40, "servo_ki offset");
_Static_assert(offsetof(FfbCmd, servo_cap) == 44, "servo_cap offset");
_Static_assert(offsetof(FfbCmd, servo_dir) == 46, "servo_dir offset");
_Static_assert(offsetof(FfbCmd, servo_only) == 47, "servo_only offset");
_Static_assert(offsetof(FfbCmd, steer) == 48, "steer offset");
_Static_assert(offsetof(FfbCmd, steer_rate) == 52, "steer_rate offset");
_Static_assert(offsetof(FfbCmd, steer_frames) == 56, "steer_frames offset");
_Static_assert(offsetof(FfbCmd, servo_scale) == 60, "servo_scale offset");
_Static_assert(offsetof(FfbCmd, servo_rate) == 64, "servo_rate offset");
_Static_assert(offsetof(FfbCmd, servo_kd) == 68, "servo_kd offset");
_Static_assert(offsetof(FfbCmd, servo_cancel) == 72, "servo_cancel offset");
_Static_assert(offsetof(FfbCmd, servo_stamp) == 76, "servo_stamp offset");
_Static_assert(offsetof(FfbCmd, servo_integ) == 80, "servo_integ offset");
_Static_assert(offsetof(FfbCmd, frame_hz) == 84, "frame_hz offset");
_Static_assert(offsetof(FfbCmd, cancel_applied) == 88, "cancel_applied offset");
_Static_assert(offsetof(FfbCmd, servo_flags) == 90, "servo_flags offset");
_Static_assert(offsetof(FfbCmd, servo_accel) == 92, "servo_accel offset");
_Static_assert(offsetof(FfbCmd, ff_inertia) == 96, "ff_inertia offset");
_Static_assert(offsetof(FfbCmd, ff_damping) == 100, "ff_damping offset");
_Static_assert(offsetof(FfbCmd, ff_friction) == 104, "ff_friction offset");
_Static_assert(offsetof(FfbCmd, servo_lead) == 108, "servo_lead offset");
_Static_assert(offsetof(FfbCmd, ff_applied) == 112, "ff_applied offset");
_Static_assert(offsetof(FfbCmd, plan_rate) == 116, "plan_rate offset");
_Static_assert(offsetof(FfbCmd, plan_accel) == 120, "plan_accel offset");
_Static_assert(offsetof(FfbCmd, ff_breakaway) == 124, "ff_breakaway offset");

#endif
