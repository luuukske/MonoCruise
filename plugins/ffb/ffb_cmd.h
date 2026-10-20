#ifndef MONOCRUISE_FFB_CMD_H
#define MONOCRUISE_FFB_CMD_H

#include <stddef.h>
#include <stdint.h>

/* Command block shared with core/lane_departure/command.py. Change both together. */

#define FFB_CMD_TAG L"Local\\MonoCruiseFfbCmd"
#define FFB_CMD_VERSION 1
#define FFB_CMD_SIZE 32
#define FFB_CMD_NONE 0
#define FFB_CMD_RUMBLE 1

#define FFB_ST_HOOKED 1
#define FFB_ST_FORCE_SEEN 2
#define FFB_ST_RUMBLING 4
#define FFB_ST_DEAD 8

typedef struct FfbCmd {
    volatile uint32_t version;
    volatile uint32_t heartbeat;
    volatile uint8_t command;
    volatile uint8_t side;
    volatile uint16_t intensity;
    volatile uint32_t seq;
    volatile int16_t torque;
    volatile uint16_t reserved0;
    volatile uint32_t status;
    volatile uint32_t frames;
    volatile uint32_t reserved1;
} FfbCmd;

_Static_assert(sizeof(FfbCmd) == FFB_CMD_SIZE, "command block size");
_Static_assert(offsetof(FfbCmd, heartbeat) == 4, "heartbeat offset");
_Static_assert(offsetof(FfbCmd, command) == 8, "command offset");
_Static_assert(offsetof(FfbCmd, seq) == 12, "seq offset");
_Static_assert(offsetof(FfbCmd, torque) == 16, "torque offset");
_Static_assert(offsetof(FfbCmd, status) == 20, "status offset");
_Static_assert(offsetof(FfbCmd, frames) == 24, "frames offset");

#endif
