/* Traction probe: publishes per-wheel surface, slip and the game's road wetness in Local\MonoCruiseTraction.
   Experimental, read only: it never writes game memory. Built to merge later into one MonoCruise plugin. */

#define UNICODE
#define _UNICODE
#define WIN32_LEAN_AND_MEAN
#include <windows.h>

#include <math.h>
#include <stdarg.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include "traction_shm.h"

typedef void (WINAPI *ScsLogFn)(int type, const char *message);
typedef int (WINAPI *ScsRegEventFn)(unsigned int event, void *cb, void *ctx);
typedef void (WINAPI *ScsChannelCb)(const char *name, unsigned int index, const void *value, void *ctx);
typedef int (WINAPI *ScsRegChannelFn)(const char *name, unsigned int index, unsigned int type, unsigned int flags,
                                      ScsChannelCb cb, void *ctx);

typedef struct ScsSdkInitV100 {
    const char *game_name;
    const char *game_id;
    unsigned int game_version;
    unsigned int padding;
    ScsLogFn log;
} ScsSdkInitV100;

typedef struct ScsTelInitV100 {
    ScsSdkInitV100 common;
    ScsRegEventFn register_for_event;
    void *unregister_from_event;
    void *register_for_channel;
    void *unregister_from_channel;
} ScsTelInitV100;

typedef struct ScsValue {
    unsigned int type;
    unsigned int padding;
    union {
        unsigned char b;
        int s32;
        unsigned int u32;
        float f;
        float fv[3];
        const char *str;
        unsigned char raw[40];
    } v;
} ScsValue;

typedef struct ScsNamedValue {
    const char *name;
    unsigned int index;
    unsigned int padding;
    ScsValue value;
} ScsNamedValue;

typedef struct ScsConfiguration {
    const char *id;
    const ScsNamedValue *attributes;
} ScsConfiguration;

_Static_assert(sizeof(ScsValue) == 48, "scs_value_t layout");
_Static_assert(sizeof(ScsNamedValue) == 64, "scs_named_value_t layout");
_Static_assert(sizeof(TractionWheel) == 48, "wheel layout");

#define EV_FRAME_END 2u
#define EV_PAUSED 3u
#define EV_STARTED 4u
#define EV_CONFIGURATION 5u
#define T_BOOL 1u
#define T_U32 3u
#define T_FLOAT 5u
#define T_FVECTOR 7u
#define T_STRING 12u
#define CH_EACH_FRAME 1u
#define U32_NIL 0xFFFFFFFFu
#define TWO_PI 6.28318530718f
#define SLIP_MIN_SPEED 1.0f
#define PEAK_HOLD_S 1.0f
#define SDK_TRUCK_WHEELS 14u

typedef struct RawWheel {
    unsigned int substance;
    float rot;
    float defl;
    float steer;
    unsigned char on_ground;
} RawWheel;

static ScsLogFn g_log;
static HANDLE g_map_handle;
static TractionShm *g_shm;
static ScsRegChannelFn g_reg;
static RawWheel g_raw[TRACTION_MAX_WHEELS];
static float g_speed;
static float g_tvel[3], g_tang[3], g_lvel[3], g_lang[3];
static float g_peak_age[TRACTION_MAX_WHEELS];
static LARGE_INTEGER g_qpf, g_last;
static uintptr_t g_base;
static uint32_t g_rva_global, g_off_world, g_off_cur, g_off_target, g_off_timer, g_off_wet;
static int g_weather_found;
static int g_weather_faults;

static void log_msg(int type, const char *fmt, ...) {
    char buf[256];
    va_list ap;
    if (!g_log) {
        return;
    }
    va_start(ap, fmt);
    vsnprintf(buf, sizeof(buf), fmt, ap);
    va_end(ap);
    g_log(type, buf);
}

/* Masked byte pattern: 0x100 matches any byte. */
static const uint16_t k_weather_sig[] = {
    0x48, 0x8B, 0x8F, 0x100, 0x100, 0x00, 0x00,
    0x44, 0x39, 0xA9, 0x100, 0x100, 0x00, 0x00,
    0x74, 0x100,
    0x8B, 0x81, 0x100, 0x100, 0x00, 0x00,
    0x39, 0x81, 0x100, 0x100, 0x00, 0x00,
    0x75, 0x100,
    0xF3, 0x44, 0x0F, 0x10, 0x89, 0x100, 0x100, 0x00, 0x00,
    0xF3, 0x44, 0x0F, 0x11, 0x8E, 0x100, 0x100, 0x00, 0x00,
    0x8B, 0x81, 0x100, 0x100, 0x00, 0x00,
    0x89, 0x86, 0x100, 0x100, 0x00, 0x00,
    0x8B, 0x81, 0x100, 0x100, 0x00, 0x00,
    0x48, 0x8B, 0x8E, 0x100, 0x100, 0x00, 0x00,
    0x89, 0x86,
};

static uint32_t rd32(const uint8_t *p) {
    uint32_t v;
    memcpy(&v, p, 4);
    return v;
}

static int match_at(const uint8_t *p) {
    size_t i;
    for (i = 0; i < sizeof(k_weather_sig) / sizeof(k_weather_sig[0]); i++) {
        if (k_weather_sig[i] != 0x100 && p[i] != (uint8_t)k_weather_sig[i]) {
            return 0;
        }
    }
    return 1;
}

/* The game's save code copies the live weather into the save object; its operands name every field we read. */
static void find_weather(void) {
    const IMAGE_DOS_HEADER *dos;
    const IMAGE_NT_HEADERS64 *nt;
    const IMAGE_SECTION_HEADER *sec;
    const uint8_t *text = NULL, *hit = NULL;
    size_t text_len = 0, n, i, sig_len = sizeof(k_weather_sig) / sizeof(k_weather_sig[0]);
    int hits = 0;
    g_base = (uintptr_t)GetModuleHandleW(NULL);
    dos = (const IMAGE_DOS_HEADER *)g_base;
    nt = (const IMAGE_NT_HEADERS64 *)(g_base + dos->e_lfanew);
    sec = IMAGE_FIRST_SECTION((PIMAGE_NT_HEADERS64)nt);
    for (n = 0; n < nt->FileHeader.NumberOfSections; n++, sec++) {
        if (memcmp(sec->Name, ".text", 5) == 0) {
            text = (const uint8_t *)(g_base + sec->VirtualAddress);
            text_len = sec->Misc.VirtualSize;
        }
    }
    if (!text || text_len < sig_len) {
        log_msg(1, "traction: no .text section, weather off");
        return;
    }
    for (i = 0; i + sig_len < text_len; i++) {
        if (text[i] == 0x48 && text[i + 1] == 0x8B && text[i + 2] == 0x8F && match_at(text + i)) {
            hit = text + i;
            hits++;
        }
    }
    if (hits != 1) {
        log_msg(1, "traction: weather signature found %d times, weather off", hits);
        return;
    }
    for (i = 0; i < 0x600 && hit - i > text; i++) {
        const uint8_t *p = hit - i;
        if (p[0] == 0x48 && p[1] == 0x8B && p[2] == 0x3D) {
            int32_t rel;
            memcpy(&rel, p + 3, 4);
            g_rva_global = (uint32_t)((uintptr_t)(p + 7 + rel) - g_base);
            break;
        }
    }
    if (!g_rva_global) {
        log_msg(1, "traction: weather global not found, weather off");
        return;
    }
    g_off_world = rd32(hit + 3);
    g_off_target = rd32(hit + 0x12);
    g_off_cur = rd32(hit + 0x18);
    g_off_timer = rd32(hit + 0x23);
    g_off_wet = rd32(hit + 0x3E);
    g_weather_found = 1;
    log_msg(0, "traction: weather at rva %#x +%#x (weather +%#x, wetness +%#x)", g_rva_global, g_off_world, g_off_cur,
            g_off_wet);
}

static int read_weather(uint32_t *cur, uint32_t *target, float *timer, float *wet) {
    int ok = 0;
    if (!g_weather_found || g_weather_faults > 8) {
        return 0;
    }
    __try {
        uintptr_t game = *(const uintptr_t *)(g_base + g_rva_global);
        if (game > 0x10000 && game < 0x7FFFFFFFFFFFull) {
            uintptr_t w = *(const uintptr_t *)(game + g_off_world);
            if (w > 0x10000 && w < 0x7FFFFFFFFFFFull) {
                *cur = *(const uint32_t *)(w + g_off_cur);
                *target = *(const uint32_t *)(w + g_off_target);
                *timer = *(const float *)(w + g_off_timer);
                *wet = *(const float *)(w + g_off_wet);
                ok = *cur < 64 && *target < 64 && *wet >= -0.01f && *wet <= 2.0f;
            }
        }
    } __except (EXCEPTION_EXECUTE_HANDLER) {
        if (++g_weather_faults > 8) {
            log_msg(1, "traction: weather read faulted repeatedly, weather off until restart");
        }
        ok = 0;
    }
    return ok;
}

static void begin_write(void) {
    InterlockedIncrement((volatile LONG *)&g_shm->seq);
}

static void end_write(void) {
    InterlockedIncrement((volatile LONG *)&g_shm->seq);
}

static void WINAPI on_u32(const char *name, unsigned int index, const void *value, void *ctx) {
    unsigned int base = (unsigned int)(uintptr_t)ctx;
    (void)name;
    if (base + index < TRACTION_MAX_WHEELS) {
        g_raw[base + index].substance = value ? ((const ScsValue *)value)->v.u32 : TRACTION_NO_SUBSTANCE;
    }
}

static void WINAPI on_rot(const char *name, unsigned int index, const void *value, void *ctx) {
    unsigned int base = (unsigned int)(uintptr_t)ctx;
    (void)name;
    if (base + index < TRACTION_MAX_WHEELS) {
        g_raw[base + index].rot = value ? ((const ScsValue *)value)->v.f : 0.0f;
    }
}

static void WINAPI on_defl(const char *name, unsigned int index, const void *value, void *ctx) {
    unsigned int base = (unsigned int)(uintptr_t)ctx;
    (void)name;
    if (base + index < TRACTION_MAX_WHEELS) {
        g_raw[base + index].defl = value ? ((const ScsValue *)value)->v.f : 0.0f;
    }
}

static void WINAPI on_steer(const char *name, unsigned int index, const void *value, void *ctx) {
    unsigned int base = (unsigned int)(uintptr_t)ctx;
    (void)name;
    if (base + index < TRACTION_MAX_WHEELS) {
        g_raw[base + index].steer = value ? ((const ScsValue *)value)->v.f : 0.0f;
    }
}

static void WINAPI on_ground(const char *name, unsigned int index, const void *value, void *ctx) {
    unsigned int base = (unsigned int)(uintptr_t)ctx;
    (void)name;
    if (base + index < TRACTION_MAX_WHEELS) {
        g_raw[base + index].on_ground = value ? ((const ScsValue *)value)->v.b : 0;
    }
}

static void WINAPI on_vec(const char *name, unsigned int index, const void *value, void *ctx) {
    float *dst = (float *)ctx;
    (void)name;
    (void)index;
    if (value) {
        memcpy(dst, ((const ScsValue *)value)->v.fv, sizeof(float) * 3);
    } else {
        dst[0] = dst[1] = dst[2] = 0.0f;
    }
}

static void WINAPI on_speed(const char *name, unsigned int index, const void *value, void *ctx) {
    (void)name;
    (void)index;
    (void)ctx;
    g_speed = value ? ((const ScsValue *)value)->v.f : 0.0f;
}

/* The SDK takes truck wheel indices 0..13 (1.62); asking for more only logs an SDK error per channel. */
static int reg_wheels(const char *prefix, unsigned int base, unsigned int count) {
    char n[64];
    unsigned int i;
    int ok = 1;
    for (i = 0; i < count; i++) {
        void *ctx = (void *)(uintptr_t)base;
        snprintf(n, sizeof(n), "%s.wheel.substance", prefix);
        ok &= g_reg(n, i, T_U32, CH_EACH_FRAME, on_u32, ctx) == 0;
        snprintf(n, sizeof(n), "%s.wheel.angular_velocity", prefix);
        ok &= g_reg(n, i, T_FLOAT, CH_EACH_FRAME, on_rot, ctx) == 0;
        snprintf(n, sizeof(n), "%s.wheel.suspension.deflection", prefix);
        ok &= g_reg(n, i, T_FLOAT, CH_EACH_FRAME, on_defl, ctx) == 0;
        snprintf(n, sizeof(n), "%s.wheel.steering", prefix);
        ok &= g_reg(n, i, T_FLOAT, CH_EACH_FRAME, on_steer, ctx) == 0;
        snprintf(n, sizeof(n), "%s.wheel.on_ground", prefix);
        ok &= g_reg(n, i, T_BOOL, CH_EACH_FRAME, on_ground, ctx) == 0;
    }
    return ok;
}

static int reg_trailer(const char *prefix) {
    char n[64];
    int ok = reg_wheels(prefix, TRACTION_TRUCK_WHEELS, TRACTION_TRUCK_WHEELS);
    snprintf(n, sizeof(n), "%s.velocity.linear", prefix);
    ok &= g_reg(n, U32_NIL, T_FVECTOR, CH_EACH_FRAME, on_vec, g_lvel) == 0;
    snprintf(n, sizeof(n), "%s.velocity.angular", prefix);
    ok &= g_reg(n, U32_NIL, T_FVECTOR, CH_EACH_FRAME, on_vec, g_lang) == 0;
    if (ok) {
        log_msg(0, "traction: trailer channels on %s", prefix);
    }
    return ok;
}

static void apply_wheel_config(const ScsNamedValue *a, unsigned int base, uint32_t *count) {
    for (; a && a->name; a++) {
        unsigned int i = a->index == U32_NIL ? 0 : a->index;
        TractionWheel *w;
        if (strcmp(a->name, "wheels.count") == 0) {
            *count = a->value.v.u32 > TRACTION_TRUCK_WHEELS ? TRACTION_TRUCK_WHEELS : a->value.v.u32;
            continue;
        }
        if (i >= TRACTION_TRUCK_WHEELS) {
            continue;
        }
        w = &g_shm->wheels[base + i];
        if (strcmp(a->name, "wheel.radius") == 0) {
            w->radius_m = a->value.v.f;
            w->flags |= TRACTION_WH_CONFIGURED;
        } else if (strcmp(a->name, "wheel.position") == 0) {
            w->pos_x = a->value.v.fv[0];
            w->pos_y = a->value.v.fv[1];
            w->pos_z = a->value.v.fv[2];
        } else if (strcmp(a->name, "wheel.powered") == 0 && a->value.v.b) {
            w->flags |= TRACTION_WH_POWERED;
        } else if (strcmp(a->name, "wheel.steerable") == 0 && a->value.v.b) {
            w->flags |= TRACTION_WH_STEERABLE;
        } else if (strcmp(a->name, "wheel.simulated") == 0 && a->value.v.b) {
            w->flags |= TRACTION_WH_SIMULATED;
        }
    }
}

static void clear_unit(unsigned int base) {
    unsigned int i;
    for (i = 0; i < TRACTION_TRUCK_WHEELS; i++) {
        memset(&g_shm->wheels[base + i], 0, sizeof(TractionWheel));
        g_shm->wheels[base + i].substance = TRACTION_NO_SUBSTANCE;
        if (base) {
            g_shm->wheels[base + i].flags = TRACTION_WH_TRAILER;
        }
    }
}

static void WINAPI on_config(unsigned int event, const void *info, void *ctx) {
    const ScsConfiguration *c = (const ScsConfiguration *)info;
    (void)event;
    (void)ctx;
    if (!g_shm || !c || !c->id) {
        return;
    }
    begin_write();
    if (strcmp(c->id, "substances") == 0) {
        const ScsNamedValue *a;
        uint32_t n = 0;
        memset(g_shm->substances, 0, sizeof(g_shm->substances));
        for (a = c->attributes; a && a->name; a++) {
            if (strcmp(a->name, "id") == 0 && a->index < TRACTION_MAX_SUBSTANCES && a->value.v.str) {
                size_t len = strnlen(a->value.v.str, TRACTION_NAME_LEN - 1);
                memcpy(g_shm->substances[a->index], a->value.v.str, len);
                if (a->index + 1 > n) {
                    n = a->index + 1;
                }
            }
        }
        g_shm->substance_count = n;
        g_shm->status |= TRACTION_ST_SUBSTANCES;
    } else if (strcmp(c->id, "truck") == 0) {
        clear_unit(0);
        g_shm->truck_wheels = 0;
        apply_wheel_config(c->attributes, 0, &g_shm->truck_wheels);
        g_shm->status |= TRACTION_ST_TRUCK_CONFIG;
    } else if (strcmp(c->id, "trailer.0") == 0 || strcmp(c->id, "trailer") == 0) {
        clear_unit(TRACTION_TRUCK_WHEELS);
        g_shm->trailer_wheels = 0;
        apply_wheel_config(c->attributes, TRACTION_TRUCK_WHEELS, &g_shm->trailer_wheels);
        if (g_shm->trailer_wheels) {
            g_shm->status |= TRACTION_ST_TRAILER_CONFIG;
        } else {
            g_shm->status &= ~TRACTION_ST_TRAILER_CONFIG;
        }
    }
    end_write();
}

static void WINAPI on_pause(unsigned int event, const void *info, void *ctx) {
    (void)info;
    (void)ctx;
    if (!g_shm) {
        return;
    }
    begin_write();
    if (event == EV_PAUSED) {
        g_shm->status |= TRACTION_ST_PAUSED;
    } else {
        g_shm->status &= ~TRACTION_ST_PAUSED;
    }
    end_write();
}

/* Contact speed along the wheel's heading: body velocity plus yaw times the wheel's lever, rotated by steering. */
static void update_wheel(unsigned int k, const float *vel, const float *ang, float dt) {
    TractionWheel *w = &g_shm->wheels[k];
    const RawWheel *r = &g_raw[k];
    float yaw = ang[1] * TWO_PI, th = r->steer * TWO_PI;
    float vx = vel[0] + yaw * w->pos_z, vz = vel[2] - yaw * w->pos_x;
    float ground = -(vx * sinf(th) + vz * cosf(th));
    float rim = r->rot * TWO_PI * w->radius_m;
    float slip = 0.0f;
    w->substance = (uint16_t)(r->substance > 0xFFFE ? TRACTION_NO_SUBSTANCE : r->substance);
    w->rot_speed = r->rot;
    w->deflection_m = r->defl;
    w->steering = r->steer;
    w->flags = (uint8_t)((w->flags & ~TRACTION_WH_ON_GROUND) | (r->on_ground ? TRACTION_WH_ON_GROUND : 0));
    w->rim_speed_ms = rim;
    w->ground_speed_ms = ground;
    if (r->on_ground && w->radius_m > 0.0f && fabsf(ground) >= SLIP_MIN_SPEED) {
        slip = (rim - ground) / fabsf(ground);
    }
    w->slip = slip;
    g_peak_age[k] += dt;
    if (fabsf(slip) >= fabsf(w->slip_peak) || g_peak_age[k] > PEAK_HOLD_S) {
        w->slip_peak = slip;
        g_peak_age[k] = 0.0f;
    }
}

static void WINAPI on_frame_end(unsigned int event, const void *info, void *ctx) {
    LARGE_INTEGER now;
    float dt;
    unsigned int i;
    uint32_t cur = 0, target = 0;
    float timer = 0.0f, wet = 0.0f;
    (void)event;
    (void)info;
    (void)ctx;
    if (!g_shm) {
        return;
    }
    QueryPerformanceCounter(&now);
    dt = g_last.QuadPart ? (float)(now.QuadPart - g_last.QuadPart) / (float)g_qpf.QuadPart : 0.0f;
    g_last = now;
    begin_write();
    g_shm->frames++;
    g_shm->speed_ms = g_speed;
    memcpy(g_shm->truck_vel, g_tvel, sizeof(g_tvel));
    memcpy(g_shm->truck_angvel, g_tang, sizeof(g_tang));
    memcpy(g_shm->trailer_vel, g_lvel, sizeof(g_lvel));
    memcpy(g_shm->trailer_angvel, g_lang, sizeof(g_lang));
    for (i = 0; i < g_shm->truck_wheels; i++) {
        update_wheel(i, g_tvel, g_tang, dt);
    }
    for (i = 0; i < g_shm->trailer_wheels; i++) {
        update_wheel(TRACTION_TRUCK_WHEELS + i, g_lvel, g_lang, dt);
    }
    if (read_weather(&cur, &target, &timer, &wet)) {
        g_shm->weather_current = cur;
        g_shm->weather_target = target;
        g_shm->weather_timer_s = timer;
        g_shm->wetness = wet;
        g_shm->status |= TRACTION_ST_WEATHER_LIVE;
    } else {
        g_shm->status &= ~TRACTION_ST_WEATHER_LIVE;
    }
    end_write();
}

static void open_map(void) {
    g_map_handle = CreateFileMappingW(INVALID_HANDLE_VALUE, NULL, PAGE_READWRITE, 0, TRACTION_SIZE, TRACTION_TAG);
    if (g_map_handle) {
        g_shm = (TractionShm *)MapViewOfFile(g_map_handle, FILE_MAP_ALL_ACCESS, 0, 0, TRACTION_SIZE);
    }
    if (!g_shm) {
        log_msg(2, "traction: shared memory unavailable");
        return;
    }
    memset(g_shm, 0, TRACTION_SIZE);
    clear_unit(0);
    clear_unit(TRACTION_TRUCK_WHEELS);
}

BOOL WINAPI DllMain(HINSTANCE module, DWORD reason, LPVOID reserved) {
    (void)reserved;
    if (reason == DLL_PROCESS_ATTACH) {
        DisableThreadLibraryCalls(module);
    }
    return TRUE;
}

__declspec(dllexport) int WINAPI scs_telemetry_init(unsigned int version, const void *params) {
    const ScsTelInitV100 *init = (const ScsTelInitV100 *)params;
    if (version != 0x00010000 && version != 0x00010001) {
        return -1;
    }
    if (!init || !init->common.log || !init->register_for_event || !init->register_for_channel) {
        return -7;
    }
    g_log = init->common.log;
    QueryPerformanceFrequency(&g_qpf);
    open_map();
    if (!g_shm) {
        return 0;
    }
    __try {
        find_weather();
    } __except (EXCEPTION_EXECUTE_HANDLER) {
        g_weather_found = 0;
        log_msg(1, "traction: weather scan faulted, weather off");
    }
    g_shm->game_version = init->common.game_version;
    g_shm->global_rva = g_rva_global;
    g_shm->off_world = g_off_world;
    g_shm->off_weather = g_off_cur;
    g_shm->off_target = g_off_target;
    g_shm->off_timer = g_off_timer;
    g_shm->off_wetness = g_off_wet;
    g_shm->status = g_weather_found ? TRACTION_ST_WEATHER_FOUND : 0;
    g_reg = (ScsRegChannelFn)init->register_for_channel;
    init->register_for_event(EV_FRAME_END, (void *)on_frame_end, NULL);
    init->register_for_event(EV_PAUSED, (void *)on_pause, NULL);
    init->register_for_event(EV_STARTED, (void *)on_pause, NULL);
    init->register_for_event(EV_CONFIGURATION, (void *)on_config, NULL);
    g_reg("truck.speed", U32_NIL, T_FLOAT, CH_EACH_FRAME, on_speed, NULL);
    g_reg("truck.local.velocity.linear", U32_NIL, T_FVECTOR, CH_EACH_FRAME, on_vec, g_tvel);
    g_reg("truck.local.velocity.angular", U32_NIL, T_FVECTOR, CH_EACH_FRAME, on_vec, g_tang);
    if (!reg_wheels("truck", 0, SDK_TRUCK_WHEELS)) {
        log_msg(1, "traction: some truck wheel channels are unavailable");
    }
    if (reg_trailer("trailer.0") || reg_trailer("trailer")) {
        g_shm->status |= TRACTION_ST_TRAILER_CHANNELS;
    } else {
        log_msg(1, "traction: trailer channels unavailable, trailer slip off");
    }
    g_shm->version = TRACTION_VERSION;
    log_msg(0, "traction: ready (layout %u, %u bytes)", TRACTION_VERSION, TRACTION_SIZE);
    return 0;
}

__declspec(dllexport) void WINAPI scs_telemetry_shutdown(void) {
    if (g_shm) {
        TractionShm *s = g_shm;
        g_shm = NULL;
        MemoryBarrier();
        s->version = 0;
        UnmapViewOfFile(s);
    }
    if (g_map_handle) {
        CloseHandle(g_map_handle);
        g_map_handle = NULL;
    }
    g_log = NULL;
}
