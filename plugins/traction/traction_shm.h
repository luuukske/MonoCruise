/* Layout of Local\MonoCruiseTraction, written by traction.c and read by tools/traction_probe.py.
   Change both together. All fields little endian, the struct is packed by construction (no padding). */
#ifndef TRACTION_SHM_H
#define TRACTION_SHM_H

#include <stdint.h>

#define TRACTION_TAG L"Local\\MonoCruiseTraction"
#define TRACTION_VERSION 1
#define TRACTION_MAX_WHEELS 32
#define TRACTION_TRUCK_WHEELS 16
#define TRACTION_MAX_SUBSTANCES 32
#define TRACTION_NAME_LEN 16
#define TRACTION_NO_SUBSTANCE 0xFFFFu

#define TRACTION_ST_WEATHER_FOUND 1u
#define TRACTION_ST_WEATHER_LIVE 2u
#define TRACTION_ST_SUBSTANCES 4u
#define TRACTION_ST_TRUCK_CONFIG 8u
#define TRACTION_ST_TRAILER_CONFIG 16u
#define TRACTION_ST_PAUSED 32u
#define TRACTION_ST_TRAILER_CHANNELS 64u

#define TRACTION_WH_ON_GROUND 1u
#define TRACTION_WH_POWERED 2u
#define TRACTION_WH_STEERABLE 4u
#define TRACTION_WH_SIMULATED 8u
#define TRACTION_WH_CONFIGURED 16u
#define TRACTION_WH_TRAILER 128u

typedef struct TractionWheel {
    uint8_t flags;
    uint8_t axle_hint;
    uint16_t substance;
    float radius_m;
    float rot_speed;
    float rim_speed_ms;
    float ground_speed_ms;
    float slip;
    float slip_peak;
    float deflection_m;
    float pos_x;
    float pos_y;
    float pos_z;
    float steering;
} TractionWheel;

typedef struct TractionShm {
    uint32_t version;
    uint32_t seq;
    uint32_t status;
    uint32_t frames;
    uint32_t weather_current;
    uint32_t weather_target;
    float wetness;
    float weather_timer_s;
    float speed_ms;
    float truck_vel[3];
    float truck_angvel[3];
    float trailer_vel[3];
    float trailer_angvel[3];
    uint32_t truck_wheels;
    uint32_t trailer_wheels;
    uint32_t substance_count;
    uint32_t global_rva;
    uint32_t off_world;
    uint32_t off_weather;
    uint32_t off_target;
    uint32_t off_timer;
    uint32_t off_wetness;
    uint32_t game_version;
    char substances[TRACTION_MAX_SUBSTANCES][TRACTION_NAME_LEN];
    TractionWheel wheels[TRACTION_MAX_WHEELS];
} TractionShm;

#define TRACTION_SIZE ((uint32_t)sizeof(TractionShm))

#endif
