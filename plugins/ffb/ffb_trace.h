#ifndef MONOCRUISE_FFB_TRACE_H
#define MONOCRUISE_FFB_TRACE_H

#include <stddef.h>
#include <stdint.h>

/* Trace block read by core/lane_departure/ffb_trace.py: every constant-force update the plugin saw, and the effects
   the game made. MonoCruise only reads it (debug CSVs); nothing in it steers. Change both together. */

#define FFB_TRACE_TAG L"Local\\MonoCruiseFfbTrace"
#define FFB_TRACE_VERSION 1
#define FFB_TRACE_LEN 256
#define FFB_TRACE_SIZE 6336

#define FFB_TF_SERVO 1
#define FFB_TF_RUMBLE 2
#define FFB_TF_LIVE 4
#define FFB_TF_DRIVING 8

/* Effect kinds, in dinput.h order: GUID_ConstantForce is 13541C20, each next kind one more. */
#define FFB_FX_CONSTANT 0
#define FFB_FX_SPRING 7
#define FFB_FX_FRICTION 10
#define FFB_FX_KINDS 12
#define FFB_FX_OTHER 31

typedef struct FfbTraceSample {
    double t;
    float steer;
    int16_t game;
    int16_t out;
    int16_t servo;
    uint16_t steer_seq;
    uint16_t flags;
    int16_t reserved;
} FfbTraceSample;

/* A condition effect's first axis as the game last wrote it; spring, damper, inertia, friction in that order. */
typedef struct FfbCondition {
    int32_t offset;
    int32_t pos_coeff;
    int32_t neg_coeff;
    uint32_t pos_sat;
    uint32_t neg_sat;
    int32_t deadband;
    uint32_t gain;
    uint32_t sets;
} FfbCondition;

typedef struct FfbTrace {
    volatile uint32_t version;
    uint32_t capacity;
    volatile uint32_t head;
    volatile uint32_t effects;
    volatile uint32_t creates;
    volatile uint32_t other_sets;
    volatile uint32_t const_gain;
    uint32_t reserved[9];
    FfbCondition cond[4];
    FfbTraceSample samples[FFB_TRACE_LEN];
} FfbTrace;

_Static_assert(sizeof(FfbTraceSample) == 24, "trace sample size");
_Static_assert(sizeof(FfbCondition) == 32, "condition size");
_Static_assert(sizeof(FfbTrace) == FFB_TRACE_SIZE, "trace block size");
_Static_assert(offsetof(FfbTrace, head) == 8, "head offset");
_Static_assert(offsetof(FfbTrace, effects) == 12, "effects offset");
_Static_assert(offsetof(FfbTrace, creates) == 16, "creates offset");
_Static_assert(offsetof(FfbTrace, other_sets) == 20, "other_sets offset");
_Static_assert(offsetof(FfbTrace, const_gain) == 24, "const_gain offset");
_Static_assert(offsetof(FfbTrace, cond) == 64, "cond offset");
_Static_assert(offsetof(FfbTrace, samples) == 192, "samples offset");
_Static_assert(offsetof(FfbTraceSample, steer) == 8, "sample steer offset");
_Static_assert(offsetof(FfbTraceSample, game) == 12, "sample game offset");
_Static_assert(offsetof(FfbTraceSample, out) == 14, "sample out offset");
_Static_assert(offsetof(FfbTraceSample, servo) == 16, "sample servo offset");
_Static_assert(offsetof(FfbTraceSample, steer_seq) == 18, "sample steer_seq offset");
_Static_assert(offsetof(FfbTraceSample, flags) == 20, "sample flags offset");

#endif
