/* Adds a zero-mean rumble and the LA (a damped steering servo on top of the game's own force) onto ETS2's
   constant-force effect. The steering axis is never written: the servo only pushes the rim, so a hand always wins.
   With no request the game force is passed through. Requests arrive in Local\MonoCruiseFfbCmd (ffb_cmd.h); a stale
   heartbeat fades everything out. */

#define UNICODE
#define _UNICODE
#define WIN32_LEAN_AND_MEAN
#define DIRECTINPUT_VERSION 0x0800
#include <windows.h>
#include <dinput.h>

#include <stdarg.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include "ffb_cmd.h"
#include "ffb_signal.h"
#include "ffb_trace.h"

#define VT_QI 0
#define VT_RELEASE 2
#define VT_CREATE_DEVICE 3
#define VT_GET_CAPS 3
#define VT_GET_EFFECT_GUID 4
#define VT_GET_PARAMS 5
#define VT_SET_PARAMS 6
#define VT_GET_STATE 9
#define VT_SET_FORMAT 11
#define VT_CREATE_EFFECT 18
#define VT_ENUM_EFFECTS 23

_Static_assert(sizeof(DIEFFECT) == 80, "DIEFFECT layout");
_Static_assert(offsetof(DIEFFECT, cbTypeSpecificParams) == 56, "DIEFFECT params");
_Static_assert(offsetof(DIEFFECT, lpvTypeSpecificParams) == 64, "DIEFFECT param pointer");

static const GUID k_constant_force = {
    0x13541C20, 0x8E33, 0x11D0, {0x9A, 0xD0, 0x00, 0xA0, 0xC9, 0xA0, 0x6E, 0x35}};
static const GUID k_iid_di8_w = {
    0xBF798031, 0x483A, 0x4DA2, {0xAA, 0x99, 0x5D, 0x64, 0xED, 0x36, 0x97, 0x00}};

typedef void (WINAPI *ScsLogFn)(int type, const char *message);
typedef int (WINAPI *ScsRegEventFn)(unsigned int event, void *cb, void *ctx);
typedef void (WINAPI *ScsChannelCb)(const char *name, unsigned int index, const void *value, void *ctx);
typedef int (WINAPI *ScsRegChannelFn)(const char *name, unsigned int index, unsigned int type, unsigned int flags,
                                      ScsChannelCb cb, void *ctx);

#define SCS_U32_NIL 0xFFFFFFFFu
#define SCS_VALUE_TYPE_FLOAT 5u
#define SCS_CHANNEL_EACH_FRAME 1u
#define SCS_VALUE_OFFSET 8
typedef HRESULT (WINAPI *CreateDevFn)(void *self, const GUID *guid, void **out, void *unk);
typedef HRESULT (WINAPI *SetFmtFn)(void *self, const DIDATAFORMAT *fmt);
typedef HRESULT (WINAPI *CreateFxFn)(void *self, const GUID *guid, const DIEFFECT *eff, void **out, void *unk);
typedef HRESULT (WINAPI *GdsFn)(void *self, DWORD cb, void *data);
typedef HRESULT (WINAPI *SetFn)(void *self, const DIEFFECT *eff, DWORD flags);
typedef HRESULT (WINAPI *GetCapsFn)(void *self, DIDEVCAPS *caps);
typedef HRESULT (WINAPI *GetGuidFn)(void *self, GUID *guid);
typedef HRESULT (WINAPI *GetParamsFn)(void *self, DIEFFECT *eff, DWORD flags);
typedef HRESULT (WINAPI *EnumFxFn)(void *self, LPDIENUMCREATEDEFFECTOBJECTSCALLBACK cb, void *ctx, DWORD flags);
typedef ULONG (WINAPI *RelFn)(void *self);

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

_Static_assert(sizeof(ScsTelInitV100) == 64, "telemetry init layout");

typedef struct Patch {
    void **slot;
    void *original;
} Patch;

static ScsLogFn g_log;
static FfbCmd *g_map;
static HANDLE g_map_handle;
static volatile LONG g_live;
static volatile LONG g_dead;
static volatile LONG g_driving;
static volatile LONG g_depth;
static volatile LONG g_busy;
static volatile LONG g_have_mag;
static volatile LONG g_first_mag;
static volatile LONG g_bad_size;
static volatile LONG g_rumbling;
static volatile LONG g_applied;
static volatile LONG g_servoing;
static volatile LONG g_steer_ok;
static volatile LONG g_steer_count;
static volatile float g_steer;
static volatile float g_steer_rate;
static volatile float g_steer_rate_fast;
static volatile float g_steer_interval;
static uint32_t g_stamp_last;
static LARGE_INTEGER g_stamp_at;
static double g_frame_hz;
static LARGE_INTEGER g_steer_at;
static LARGE_INTEGER g_steer_moved_at;
static LARGE_INTEGER g_qpf;
static LARGE_INTEGER g_qpc;
static LARGE_INTEGER g_hb_at;
static uint32_t g_hb_last;
static int g_seen;
static FfbWave g_wave;
static FfbTorque g_torque;
static FfbServo g_servo;
static double g_force_lp;
static double g_force_steady;
static Patch g_patches[8];
static int g_npatches;
static void *g_set_vt[2];
static SetFn g_set_orig[2];
static void *g_gds_vt[2];
static GdsFn g_gds_orig[2];
static void *g_ce_vt[2];
static CreateFxFn g_ce_orig[2];
static GUID g_devs[8];
static int g_ndev;
static void *g_skip[8];
static int g_nskip;
static int g_game_enum_done;
static FfbTrace *g_trace;
static HANDLE g_trace_handle;
static double g_trace_servo;
static int g_trace_live;

static void log_msg(const char *msg) {
    if (g_log && msg) {
        g_log(0, msg);
    }
}

static void log_hr(const char *what, HRESULT hr) {
    char buf[128];
    snprintf(buf, sizeof buf, "ffb: %s 0x%08lX", what, (unsigned long)hr);
    log_msg(buf);
}

static int guid_eq(const GUID *a, const GUID *b) {
    return memcmp(a, b, sizeof(GUID)) == 0;
}

static int in_dinput(void *fn) {
    HMODULE ours = NULL;
    HMODULE di = GetModuleHandleW(L"dinput8.dll");
    if (!di || !fn) {
        return 0;
    }
    if (!GetModuleHandleExW(
            GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
            (LPCWSTR)fn, &ours)) {
        return 0;
    }
    return ours == di;
}

static void remember(void **slot, void *original) {
    int i;
    for (i = 0; i < g_npatches; i++) {
        if (g_patches[i].slot == slot) {
            return;
        }
    }
    if (g_npatches >= 8) {
        return;
    }
    g_patches[g_npatches].slot = slot;
    g_patches[g_npatches].original = original;
    g_npatches++;
}

static int patch_slot(void **slot, void *hook) {
    DWORD old = 0;
    void *cur;
    if (!slot || !hook) {
        return 0;
    }
    if (!VirtualProtect(slot, sizeof(void *), PAGE_READWRITE, &old)) {
        return 0;
    }
    cur = *slot;
    if (cur == hook) {
        VirtualProtect(slot, sizeof(void *), old, &old);
        return 1;
    }
    if (!in_dinput(cur)) {
        VirtualProtect(slot, sizeof(void *), old, &old);
        return 0;
    }
    remember(slot, cur);
    *slot = hook;
    VirtualProtect(slot, sizeof(void *), old, &old);
    FlushInstructionCache(GetCurrentProcess(), slot, sizeof(void *));
    return 1;
}

static void release_obj(void *obj) {
    void **vt;
    RelFn rel;
    if (!obj) {
        return;
    }
    vt = *(void ***)obj;
    rel = (RelFn)vt[VT_RELEASE];
    if (in_dinput((void *)rel)) {
        rel(obj);
    }
}

static SetFn orig_set(void *self) {
    void *vt = *(void **)self;
    if (vt == g_set_vt[1]) {
        return g_set_orig[1];
    }
    return g_set_orig[0];
}

static GdsFn orig_gds(void *self) {
    void *vt = *(void **)self;
    if (vt == g_gds_vt[1]) {
        return g_gds_orig[1];
    }
    return g_gds_orig[0];
}

static CreateFxFn orig_create(void *self) {
    void *vt = *(void **)self;
    if (vt == g_ce_vt[1]) {
        return g_ce_orig[1];
    }
    return g_ce_orig[0];
}

/* An effect's kind (FFB_FX_*): dinput.h numbers the effect GUIDs in a run from GUID_ConstantForce. */
static int fx_kind(const GUID *g) {
    uint32_t n;
    if (!g || memcmp((const char *)g + 4, (const char *)&k_constant_force + 4, sizeof(GUID) - 4) != 0) {
        return FFB_FX_OTHER;
    }
    n = (uint32_t)(g->Data1 - k_constant_force.Data1);
    return n < FFB_FX_KINDS ? (int)n : FFB_FX_OTHER;
}

static int fx_kind_of(void *fx) {
    GUID guid;
    GetGuidFn get_guid;
    if (!fx) {
        return FFB_FX_OTHER;
    }
    get_guid = (GetGuidFn)(*(void ***)fx)[VT_GET_EFFECT_GUID];
    if (!in_dinput((void *)get_guid) || FAILED(get_guid(fx, &guid))) {
        return FFB_FX_OTHER;
    }
    return fx_kind(&guid);
}

static int16_t trace_i16(long x) {
    return (int16_t)(x > 32767 ? 32767 : (x < -32767 ? -32767 : x));
}

static int fx_condition(int kind) {
    return kind >= FFB_FX_SPRING && kind <= FFB_FX_FRICTION;
}

/* The gain and condition parameters the game gave an effect; read only, nothing is changed. */
static void trace_params(int kind, const DIEFFECT *eff, DWORD flags) {
    FfbTrace *tr = g_trace;
    FfbCondition *c;
    if (!tr || !eff || eff->dwSize != sizeof(DIEFFECT)) {
        return;
    }
    if (kind == FFB_FX_CONSTANT) {
        if (flags & DIEP_GAIN) {
            tr->const_gain = eff->dwGain;
        }
        return;
    }
    if (!fx_condition(kind)) {
        tr->other_sets++;
        return;
    }
    c = &tr->cond[kind - FFB_FX_SPRING];
    if ((flags & DIEP_TYPESPECIFICPARAMS) && eff->lpvTypeSpecificParams
        && eff->cbTypeSpecificParams >= sizeof(DICONDITION)) {
        const DICONDITION *p = (const DICONDITION *)eff->lpvTypeSpecificParams;
        c->offset = p->lOffset;
        c->pos_coeff = p->lPositiveCoefficient;
        c->neg_coeff = p->lNegativeCoefficient;
        c->pos_sat = p->dwPositiveSaturation;
        c->neg_sat = p->dwNegativeSaturation;
        c->deadband = p->lDeadBand;
    }
    if (flags & DIEP_GAIN) {
        c->gain = eff->dwGain;
    }
    MemoryBarrier();
    c->sets++;
}

static void trace_kind(int kind) {
    if (g_trace) {
        InterlockedOr((volatile LONG *)&g_trace->effects, (LONG)(1u << kind));
    }
}

static void trace_created(const GUID *guid, const DIEFFECT *eff) {
    int kind = fx_kind(guid);
    if (!g_trace) {
        return;
    }
    trace_kind(kind);
    InterlockedIncrement((volatile LONG *)&g_trace->creates);
    if (kind == FFB_FX_CONSTANT || fx_condition(kind)) {
        trace_params(kind, eff, DIEP_TYPESPECIFICPARAMS | DIEP_GAIN);
    }
}

/* An effect the game made before the hook: its kind, and what it holds now. */
static void trace_existing(void *fx, int kind) {
    GetParamsFn get;
    DICONDITION cond[2];
    DIEFFECT eff;
    DWORD flags = DIEP_GAIN;
    trace_kind(kind);
    if (!g_trace || !(kind == FFB_FX_CONSTANT || fx_condition(kind))) {
        return;
    }
    get = (GetParamsFn)(*(void ***)fx)[VT_GET_PARAMS];
    if (!in_dinput((void *)get)) {
        return;
    }
    ZeroMemory(&eff, sizeof eff);
    ZeroMemory(cond, sizeof cond);
    eff.dwSize = sizeof eff;
    if (fx_condition(kind)) {
        flags |= DIEP_TYPESPECIFICPARAMS;
        eff.cbTypeSpecificParams = sizeof cond;
        eff.lpvTypeSpecificParams = cond;
    }
    if (SUCCEEDED(get(fx, &eff, flags))) {
        trace_params(kind, &eff, flags);
    }
}

/* One constant-force update: what the game asked, what went to the wheel, the wheel angle the servo used. */
static void trace_push(LONG game, LONG out) {
    FfbTrace *tr = g_trace;
    FfbTraceSample *s;
    uint32_t head;
    uint16_t flags = 0;
    if (!tr || g_qpf.QuadPart == 0) {
        return;
    }
    flags |= g_servo.gain > 0.0 ? FFB_TF_SERVO : 0;
    flags |= g_wave.gain > 0.0 ? FFB_TF_RUMBLE : 0;
    flags |= g_trace_live ? FFB_TF_LIVE : 0;
    flags |= InterlockedOr((LONG *)&g_driving, 0) ? FFB_TF_DRIVING : 0;
    head = tr->head;
    s = &tr->samples[head % FFB_TRACE_LEN];
    s->t = (double)g_qpc.QuadPart / (double)g_qpf.QuadPart;
    s->steer = g_steer;
    s->game = trace_i16(game);
    s->out = trace_i16(out);
    s->servo = trace_i16(ffb_round(g_trace_servo));
    s->steer_seq = (uint16_t)((uint32_t)g_steer_count & 0xFFFFu);
    s->flags = flags;
    s->reserved = 0;
    MemoryBarrier();
    tr->head = head + 1;
}

static double tick_dt(void) {
    LARGE_INTEGER now;
    double dt;
    QueryPerformanceCounter(&now);
    if (g_qpc.QuadPart == 0 || g_qpf.QuadPart == 0) {
        g_qpc = now;
        return 0.0;
    }
    dt = (double)(now.QuadPart - g_qpc.QuadPart) / (double)g_qpf.QuadPart;
    g_qpc = now;
    if (dt < 0.0) {
        dt = 0.0;
    }
    return dt;
}

static int block_live(void) {
    LARGE_INTEGER now;
    uint32_t hb;
    double age;
    if (!g_map || g_map->version != FFB_CMD_VERSION || !InterlockedOr((LONG *)&g_driving, 0)) {
        return 0;
    }
    QueryPerformanceCounter(&now);
    hb = g_map->heartbeat;
    if (hb != g_hb_last || g_hb_at.QuadPart == 0) {
        g_hb_last = hb;
        g_hb_at = now;
    }
    age = g_qpf.QuadPart ? (double)(now.QuadPart - g_hb_at.QuadPart) / (double)g_qpf.QuadPart : 0.0;
    return hb != 0 && age <= FFB_STALE_S;
}

static int steer_fresh(void) {
    LARGE_INTEGER now;
    if (!InterlockedOr(&g_steer_ok, 0) || g_steer_at.QuadPart == 0 || g_qpf.QuadPart == 0) {
        return 0;
    }
    QueryPerformanceCounter(&now);
    return (double)(now.QuadPart - g_steer_at.QuadPart) / (double)g_qpf.QuadPart <= FFB_STEER_STALE_S;
}

static double target_age(uint32_t stamp) {
    LARGE_INTEGER now;
    QueryPerformanceCounter(&now);
    if (stamp != g_stamp_last || g_stamp_at.QuadPart == 0) {
        g_stamp_last = stamp;
        g_stamp_at = now;
        return 0.0;
    }
    return g_qpf.QuadPart ? (double)(now.QuadPart - g_stamp_at.QuadPart) / (double)g_qpf.QuadPart : 0.0;
}

static double steer_age(void) {
    LARGE_INTEGER now;
    if (g_steer_at.QuadPart == 0 || g_qpf.QuadPart == 0) {
        return 0.0;
    }
    QueryPerformanceCounter(&now);
    return (double)(now.QuadPart - g_steer_at.QuadPart) / (double)g_qpf.QuadPart;
}

/* Seconds since the steering reading last changed value; 0 before the first change. */
static double steer_still(void) {
    LARGE_INTEGER now;
    if (g_steer_moved_at.QuadPart == 0 || g_qpf.QuadPart == 0) {
        return 0.0;
    }
    QueryPerformanceCounter(&now);
    return (double)(now.QuadPart - g_steer_moved_at.QuadPart) / (double)g_qpf.QuadPart;
}

/* The target and the wheel are both taken ``ahead`` seconds on: where the target will be and where the wheel will
   be, read on at its rate, when the force reaches the rim. */
static double servo_delta(int live, double dt) {
    double raw = 0.0;
    int on = live && g_map && g_map->servo_dir != 0 && steer_fresh();
    if (on) {
        FfbServoIn in;
        double target;
        double age;
        double ahead = ffb_servo_ahead(ffb_finite((double)g_map->servo_lead), steer_age());
        in.target = (double)g_map->servo_target;
        in.kp = (double)g_map->servo_kp;
        in.kd = (double)g_map->servo_kd;
        in.ki = (double)g_map->servo_ki;
        in.cap = (double)g_map->servo_cap;
        in.dir = (int)g_map->servo_dir;
        in.only = g_map->servo_only > 0 ? 1 : (g_map->servo_only < 0 ? -1 : 0);
        in.damp_first = (g_map->servo_flags & FFB_SF_DAMP_FIRST) != 0;
        in.accel = ffb_finite((double)g_map->servo_accel);
        in.ff_inertia = ffb_finite((double)g_map->ff_inertia);
        in.ff_damping = ffb_finite((double)g_map->ff_damping);
        in.ff_friction = ffb_finite((double)g_map->ff_friction);
        in.ff_breakaway = ffb_finite((double)g_map->ff_breakaway);
        in.stuck = ffb_servo_stuck(steer_still(), (double)g_steer_interval);
        age = target_age(g_map->servo_stamp);
        ffb_servo_aim(in.target, (double)g_map->servo_rate, in.accel, age + ahead, &target, &in.rate);
        ffb_servo_ff_motion(&in, (g_map->servo_flags & FFB_SF_FF_PLAN) != 0, (double)g_map->plan_rate,
                            (double)g_map->plan_accel, age);
        ffb_servo_plan_feedback(&in, (g_map->servo_flags & FFB_SF_FF_PLAN) != 0, (double)g_steer_interval);
        raw = ffb_servo_pid(&g_servo, &in, target, (double)g_steer + (double)g_steer_rate_fast * ahead,
                            (double)g_steer_rate_fast, dt);
    } else {
        g_servo.ff = 0.0;
    }
    return ffb_servo_step(&g_servo, raw, on, g_wave.gain > 0.0 ? NAN : (double)g_steer, dt);
}

static long current_delta(LONG game_mag) {
    double dt = tick_dt();
    int live = block_live();
    double amp = g_map ? (double)g_map->intensity : 0.0;
    double want = g_map ? (double)g_map->torque : 0.0;
    double sum = ffb_wave_step(&g_wave, amp, live && g_map->command == FFB_CMD_RUMBLE, dt);
    double base = ffb_torque_step(&g_torque, want, live, dt);
    double servo = servo_delta(live, dt);
    double assist = ffb_clamp(base + servo, FFB_ASSIST_CAP);
    double cancel;
    g_trace_servo = servo;
    g_trace_live = live;
    g_force_lp = ffb_force_step(g_force_lp, (double)game_mag, dt);
    g_force_steady += ((double)game_mag - g_force_steady) * (1.0 - exp(-ffb_dt_clamp(dt) / FFB_CANCEL_TAU_S));
    cancel = ffb_cancel_term(g_force_steady, g_map ? (double)g_map->servo_cancel : 0.0, ffb_ease(g_servo.gain));
    sum += assist + cancel;
    if (dt > 0.0) {
        g_frame_hz = g_frame_hz > 0.0 ? g_frame_hz + (1.0 / dt - g_frame_hz) * 0.02 : 1.0 / dt;
    }
    if (g_map) {
        g_map->game_force = (int16_t)ffb_round(g_force_lp);
        g_map->applied = (int16_t)ffb_round(assist);
        g_map->servo_applied = (int16_t)ffb_round(servo);
        g_map->servo_scale = (float)g_servo.scale;
        g_map->servo_integ = (float)g_servo.integ;
        g_map->frame_hz = (float)g_frame_hz;
        g_map->cancel_applied = (int16_t)ffb_round(cancel);
        g_map->ff_applied = trace_i16(ffb_round(g_servo.ff * (double)(g_map->servo_dir ? g_map->servo_dir : 1)));
    }
    InterlockedExchange(&g_rumbling, g_wave.gain > 0.0);
    InterlockedExchange(&g_applied, (LONG)ffb_round(assist));
    InterlockedExchange(&g_servoing, g_servo.gain > 0.0);
    return ffb_round(sum);
}

static HRESULT hook_set_body(SetFn orig, void *self, const DIEFFECT *eff, DWORD flags) {
    LONG game_mag;
    long delta;
    DIEFFECT copy;
    DICONSTANTFORCE cf;
    HRESULT hr;
    if (InterlockedIncrement(&g_depth) != 1) {
        InterlockedDecrement(&g_depth);
        return orig(self, eff, flags);
    }
    if (!eff || eff->dwSize != sizeof(DIEFFECT) || !(flags & DIEP_TYPESPECIFICPARAMS)
        || eff->cbTypeSpecificParams != sizeof(DICONSTANTFORCE) || !eff->lpvTypeSpecificParams) {
        if (eff && eff->dwSize != sizeof(DIEFFECT) && eff->dwSize != 0 && !g_bad_size) {
            g_bad_size = (LONG)eff->dwSize;
        }
        if (g_trace && eff && eff->dwSize == sizeof(DIEFFECT)) {
            trace_params(fx_kind_of(self), eff, flags);
        }
        InterlockedDecrement(&g_depth);
        return orig(self, eff, flags);
    }
    g_seen = 1;
    if (g_map) {
        g_map->frames++;
    }
    if (g_trace && (flags & DIEP_GAIN)) {
        g_trace->const_gain = eff->dwGain;
    }
    game_mag = ((const DICONSTANTFORCE *)eff->lpvTypeSpecificParams)->lMagnitude;
    if (InterlockedOr((LONG *)&g_driving, 0) && !g_have_mag) {
        g_first_mag = game_mag;
        g_have_mag = 1;
    }
    delta = current_delta(game_mag);
    if (delta == 0) {
        trace_push(game_mag, game_mag);
        InterlockedDecrement(&g_depth);
        return orig(self, eff, flags);
    }
    cf.lMagnitude = ffb_apply(game_mag, delta);
    trace_push(game_mag, cf.lMagnitude);
    copy = *eff;
    copy.lpvTypeSpecificParams = &cf;
    hr = orig(self, &copy, flags);
    InterlockedDecrement(&g_depth);
    return hr;
}

static HRESULT WINAPI hook_set(void *self, const DIEFFECT *eff, DWORD flags) {
    SetFn orig = orig_set(self);
    if (!orig) {
        return E_FAIL;
    }
    if (!g_live || g_dead) {
        return orig(self, eff, flags);
    }
    __try {
        return hook_set_body(orig, self, eff, flags);
    } __except (EXCEPTION_EXECUTE_HANDLER) {
        g_dead = 1;
        g_depth = 0;
    }
    return orig(self, eff, flags);
}

static void hook_effect(void *fx) {
    void **vt;
    int i;
    if (!fx) {
        return;
    }
    vt = *(void ***)fx;
    for (i = 0; i < 2; i++) {
        if (g_set_vt[i] == vt) {
            return;
        }
    }
    for (i = 0; i < 2; i++) {
        if (!g_set_vt[i]) {
            void **slot = &vt[VT_SET_PARAMS];
            void *cur = *slot;
            if (!in_dinput(cur)) {
                return;
            }
            g_set_orig[i] = (SetFn)cur;
            g_set_vt[i] = vt;
            MemoryBarrier();
            patch_slot(slot, (void *)hook_set);
            return;
        }
    }
}

static BOOL CALLBACK on_game_effect(LPDIRECTINPUTEFFECT fx, LPVOID ctx) {
    int kind;
    (void)ctx;
    if (!fx) {
        return DIENUM_CONTINUE;
    }
    kind = fx_kind_of(fx);
    trace_existing(fx, kind);
    if (kind == FFB_FX_CONSTANT) {
        hook_effect(fx);
    }
    return DIENUM_CONTINUE;
}

static int skipped(void *obj) {
    int i;
    for (i = 0; i < g_nskip; i++) {
        if (g_skip[i] == obj) {
            return 1;
        }
    }
    return 0;
}

static void note_skip(void *obj) {
    if (g_nskip < 8 && !skipped(obj)) {
        g_skip[g_nskip++] = obj;
    }
}

static void try_enum_game(void *dev) {
    DIDEVCAPS caps;
    GetCapsFn get_caps;
    EnumFxFn enum_fx;
    if (!dev || g_game_enum_done || skipped(dev)) {
        return;
    }
    get_caps = (GetCapsFn)(*(void ***)dev)[VT_GET_CAPS];
    if (!in_dinput((void *)get_caps)) {
        note_skip(dev);
        return;
    }
    caps.dwSize = sizeof caps;
    if (FAILED(get_caps(dev, &caps)) || !(caps.dwFlags & DIDC_FORCEFEEDBACK)) {
        note_skip(dev);
        return;
    }
    g_game_enum_done = 1;
    enum_fx = (EnumFxFn)(*(void ***)dev)[VT_ENUM_EFFECTS];
    if (in_dinput((void *)enum_fx)) {
        enum_fx(dev, on_game_effect, NULL, 0);
    }
}

static HRESULT WINAPI hook_gds(void *self, DWORD cb, void *data) {
    GdsFn orig = orig_gds(self);
    HRESULT hr;
    if (!orig) {
        return E_FAIL;
    }
    hr = orig(self, cb, data);
    if (g_live && !g_dead && !g_game_enum_done && InterlockedCompareExchange(&g_busy, 1, 0) == 0) {
        __try {
            try_enum_game(self);
        } __except (EXCEPTION_EXECUTE_HANDLER) {
            g_dead = 1;
        }
        InterlockedExchange(&g_busy, 0);
    }
    return hr;
}

static HRESULT WINAPI hook_create(void *self, const GUID *guid, const DIEFFECT *eff, void **out, void *unk) {
    CreateFxFn orig = orig_create(self);
    HRESULT hr;
    if (!orig) {
        return E_FAIL;
    }
    hr = orig(self, guid, eff, out, unk);
    if (g_live && !g_dead && SUCCEEDED(hr) && out && *out && guid) {
        __try {
            trace_created(guid, eff);
            if (guid_eq(guid, &k_constant_force)) {
                hook_effect(*out);
            }
        } __except (EXCEPTION_EXECUTE_HANDLER) {
            g_dead = 1;
        }
    }
    return hr;
}

static void patch_device_methods(void *dev) {
    void **vt = *(void ***)dev;
    int i;
    for (i = 0; i < 2; i++) {
        if (g_gds_vt[i] == vt) {
            return;
        }
    }
    for (i = 0; i < 2; i++) {
        if (!g_gds_vt[i]) {
            void *gds = vt[VT_GET_STATE];
            void *create = vt[VT_CREATE_EFFECT];
            if (!in_dinput(gds) || !in_dinput(create)) {
                return;
            }
            g_gds_orig[i] = (GdsFn)gds;
            g_gds_vt[i] = vt;
            g_ce_orig[i] = (CreateFxFn)create;
            g_ce_vt[i] = vt;
            MemoryBarrier();
            patch_slot(&vt[VT_GET_STATE], (void *)hook_gds);
            patch_slot(&vt[VT_CREATE_EFFECT], (void *)hook_create);
            return;
        }
    }
}

static void prime_effect_vtable(void *dev) {
    SetFmtFn set_fmt = (SetFmtFn)(*(void ***)dev)[VT_SET_FORMAT];
    CreateFxFn create = (CreateFxFn)(*(void ***)dev)[VT_CREATE_EFFECT];
    DICONSTANTFORCE cf;
    DWORD axis = DIJOFS_X;
    LONG dir = 0;
    DIEFFECT eff;
    void *fx = NULL;
    HRESULT hr;
    if (!create) {
        log_msg("ffb: create-effect slot missing");
        return;
    }
    if (in_dinput((void *)set_fmt)) {
        set_fmt(dev, &c_dfDIJoystick);
    }
    ZeroMemory(&eff, sizeof eff);
    cf.lMagnitude = 0;
    eff.dwSize = sizeof eff;
    eff.dwFlags = DIEFF_CARTESIAN | DIEFF_OBJECTOFFSETS;
    eff.dwDuration = INFINITE;
    eff.dwGain = DI_FFNOMINALMAX;
    eff.dwTriggerButton = DIEB_NOTRIGGER;
    eff.cAxes = 1;
    eff.rgdwAxes = &axis;
    eff.rglDirection = &dir;
    eff.cbTypeSpecificParams = sizeof cf;
    eff.lpvTypeSpecificParams = &cf;
    hr = create(dev, &k_constant_force, &eff, &fx, NULL);
    if (FAILED(hr) || !fx) {
        log_hr("own constant force", hr);
        return;
    }
    hook_effect(fx);
    release_obj(fx);
}

static BOOL WINAPI on_enum_dev(LPCDIDEVICEINSTANCEW inst, LPVOID ctx) {
    char utf[160];
    char buf[220];
    (void)ctx;
    if (!inst || g_ndev >= 8) {
        return DIENUM_CONTINUE;
    }
    g_devs[g_ndev] = inst->guidInstance;
    g_ndev++;
    if (WideCharToMultiByte(CP_UTF8, 0, inst->tszProductName, -1, utf, (int)sizeof utf, NULL, NULL) > 0) {
        snprintf(buf, sizeof buf, "ffb: saw %s", utf);
        log_msg(buf);
    }
    return DIENUM_CONTINUE;
}

static void open_and_patch(IDirectInput8W *di, const GUID *guid) {
    IDirectInputDevice8W *dev = NULL;
    HRESULT hr;
    if (!di || !guid) {
        return;
    }
    hr = di->lpVtbl->CreateDevice(di, guid, &dev, NULL);
    if (FAILED(hr) || !dev) {
        log_hr("CreateDevice", hr);
        return;
    }
    prime_effect_vtable(dev);
    patch_device_methods(dev);
    release_obj(dev);
}

static void install_hook(void) {
    IDirectInput8W *diw = NULL;
    HRESULT hr;
    int i;
    hr = DirectInput8Create(GetModuleHandleW(NULL), 0x0800, &k_iid_di8_w, (void **)&diw, NULL);
    if (FAILED(hr) || !diw) {
        log_hr("DirectInput8Create", hr);
        return;
    }
    hr = diw->lpVtbl->EnumDevices(diw, DI8DEVCLASS_GAMECTRL, on_enum_dev, NULL, DIEDFL_ATTACHEDONLY);
    if (g_ndev == 0) {
        log_hr("enum game controllers", hr);
        hr = diw->lpVtbl->EnumDevices(diw, DI8DEVCLASS_ALL, on_enum_dev, NULL, DIEDFL_ATTACHEDONLY);
        log_hr("enum all devices", hr);
    }
    for (i = 0; i < g_ndev; i++) {
        open_and_patch(diw, &g_devs[i]);
    }
    release_obj(diw);
    if (g_ndev == 0) {
        log_msg("ffb: no controller enumerated");
    }
}

static void open_map(void) {
    g_map_handle = CreateFileMappingW(INVALID_HANDLE_VALUE, NULL, PAGE_READWRITE, 0, FFB_CMD_SIZE, FFB_CMD_TAG);
    if (g_map_handle) {
        g_map = (FfbCmd *)MapViewOfFile(g_map_handle, FILE_MAP_ALL_ACCESS, 0, 0, FFB_CMD_SIZE);
    }
    if (g_map) {
        g_map->command = FFB_CMD_NONE;
        g_map->torque = 0;
        g_map->game_force = 0;
        g_map->applied = 0;
        g_map->servo_applied = 0;
        g_map->servo_dir = 0;
        g_map->servo_only = 0;
        g_map->servo_flags = 0;
        g_map->servo_kp = 0.0f;
        g_map->servo_ki = 0.0f;
        g_map->servo_scale = 1.0f;
        g_map->servo_rate = 0.0f;
        g_map->servo_kd = 0.0f;
        g_map->servo_cancel = 0.0f;
        g_map->servo_stamp = 0;
        g_map->servo_integ = 0.0f;
        g_map->frame_hz = 0.0f;
        g_map->cancel_applied = 0;
        g_map->servo_accel = 0.0f;
        g_map->ff_inertia = 0.0f;
        g_map->ff_damping = 0.0f;
        g_map->ff_friction = 0.0f;
        g_map->servo_lead = 0.0f;
        g_map->plan_rate = 0.0f;
        g_map->plan_accel = 0.0f;
        g_map->ff_breakaway = 0.0f;
        g_map->ff_applied = 0;
        g_map->steer_frames = 0;
        g_map->status = 0;
        g_map->frames = 0;
        g_map->version = FFB_CMD_VERSION;
    }
}

static void open_trace(void) {
    g_trace_handle = CreateFileMappingW(INVALID_HANDLE_VALUE, NULL, PAGE_READWRITE, 0, FFB_TRACE_SIZE, FFB_TRACE_TAG);
    if (g_trace_handle) {
        g_trace = (FfbTrace *)MapViewOfFile(g_trace_handle, FILE_MAP_ALL_ACCESS, 0, 0, FFB_TRACE_SIZE);
    }
    if (g_trace) {
        g_trace->version = 0;
        MemoryBarrier();
        ZeroMemory((void *)g_trace, FFB_TRACE_SIZE);
        g_trace->capacity = FFB_TRACE_LEN;
        MemoryBarrier();
        g_trace->version = FFB_TRACE_VERSION;
    }
}

static void log_trace(void) {
    static const char *names[FFB_FX_KINDS] = {"constant", "ramp", "square", "sine", "triangle", "sawtooth up",
                                              "sawtooth down", "spring", "damper", "inertia", "friction", "custom"};
    static uint32_t logged_fx;
    static int logged_cond[4];
    char buf[256];
    uint32_t fx;
    int i;
    if (!g_trace) {
        return;
    }
    fx = g_trace->effects;
    if (fx != logged_fx) {
        int n = snprintf(buf, sizeof buf, "ffb: game effects:");
        for (i = 0; i < FFB_FX_KINDS && n > 0 && n < (int)sizeof buf; i++) {
            if (fx & (1u << i)) {
                n += snprintf(buf + n, sizeof buf - (size_t)n, " %s", names[i]);
            }
        }
        if ((fx & (1u << FFB_FX_OTHER)) && n > 0 && n < (int)sizeof buf) {
            snprintf(buf + n, sizeof buf - (size_t)n, " other");
        }
        log_msg(buf);
        logged_fx = fx;
    }
    for (i = 0; i < 4; i++) {
        const FfbCondition *c = &g_trace->cond[i];
        if (c->sets && !logged_cond[i]) {
            snprintf(buf, sizeof buf, "ffb: game %s: coefficient %ld/%ld, saturation %lu/%lu, offset %ld, dead band %ld, "
                     "gain %lu", names[FFB_FX_SPRING + i], (long)c->pos_coeff, (long)c->neg_coeff,
                     (unsigned long)c->pos_sat, (unsigned long)c->neg_sat, (long)c->offset, (long)c->deadband,
                     (unsigned long)c->gain);
            log_msg(buf);
            logged_cond[i] = 1;
        }
    }
}

static void unpatch_all(void);

static void publish_status(void) {
    uint32_t bits = 0;
    if (!g_map) {
        return;
    }
    if (g_set_orig[0]) {
        bits |= FFB_ST_HOOKED;
    }
    if (g_seen) {
        bits |= FFB_ST_FORCE_SEEN;
    }
    if (g_rumbling) {
        bits |= FFB_ST_RUMBLING;
    }
    if (g_applied) {
        bits |= FFB_ST_TORQUE;
    }
    if (g_servoing) {
        bits |= FFB_ST_SERVO;
    }
    if (g_steer_ok) {
        bits |= FFB_ST_STEER;
    }
    if (g_dead) {
        bits |= FFB_ST_DEAD;
    }
    g_map->status = bits | FFB_ST_ONE_SIDED | FFB_ST_DAMP_FIRST | FFB_ST_PUSH_AHEAD | FFB_ST_FF_PLAN | FFB_ST_BREAKAWAY;
}

static void WINAPI on_frame(unsigned int event, const void *info, void *ctx) {
    static int logged_rumble;
    static int logged_torque;
    static int logged_mag;
    static int logged_hook;
    static int logged_dead;
    int rumbling;
    int torque;
    (void)event;
    (void)info;
    (void)ctx;
    if (!g_log) {
        return;
    }
    publish_status();
    log_trace();
    if (g_dead && !logged_dead) {
        logged_dead = 1;
        unpatch_all();
        log_msg("ffb: a call faulted, plugin off until restart");
    }
    if (g_bad_size) {
        char buf[80];
        snprintf(buf, sizeof buf, "ffb: unexpected effect size %ld", (long)g_bad_size);
        log_msg(buf);
        g_bad_size = 0;
    }
    if (!logged_hook && g_set_orig[0]) {
        log_msg("ffb: constant force hooked");
        logged_hook = 1;
    }
    if (!logged_mag && g_have_mag) {
        char buf[80];
        snprintf(buf, sizeof buf, "ffb: game force sample %ld", (long)g_first_mag);
        log_msg(buf);
        logged_mag = 1;
    }
    rumbling = g_rumbling ? 1 : 0;
    if (rumbling != logged_rumble) {
        logged_rumble = rumbling;
        log_msg(rumbling ? "ffb: rumble on" : "ffb: rumble off");
    }
    torque = g_servoing ? 1 : 0;
    if (torque != logged_torque) {
        logged_torque = torque;
        log_msg(torque ? "ffb: steering servo on" : "ffb: steering servo off");
    }
}

/* truck.input.steering, every frame: the wheel's own angle, counterclockwise positive. */
static void WINAPI on_steering(const char *name, unsigned int index, const void *value, void *ctx) {
    static LARGE_INTEGER last;
    static float prev;
    LARGE_INTEGER now;
    float v;
    double dt;
    (void)name;
    (void)index;
    (void)ctx;
    if (!value || *(const unsigned int *)value != SCS_VALUE_TYPE_FLOAT) {
        return;
    }
    v = *(const float *)((const char *)value + SCS_VALUE_OFFSET);
    if (!(v == v)) {
        return;
    }
    QueryPerformanceCounter(&now);
    dt = (last.QuadPart && g_qpf.QuadPart) ? (double)(now.QuadPart - last.QuadPart) / (double)g_qpf.QuadPart : 0.0;
    if (dt > 0.0 && dt < 0.2) {
        double raw = (double)(v - prev) / dt;
        g_steer_rate = (float)((double)g_steer_rate + (raw - (double)g_steer_rate) * (1.0 - exp(-dt / FFB_STEER_RATE_TAU_S)));
        g_steer_rate_fast = (float)((double)g_steer_rate_fast
                                    + (raw - (double)g_steer_rate_fast) * (1.0 - exp(-dt / FFB_RATE_FAST_TAU_S)));
        g_steer_interval = g_steer_interval > 0.0f ? (float)((double)g_steer_interval + (dt - (double)g_steer_interval) * 0.05)
                                                   : (float)dt;
    } else {
        g_steer_rate = 0.0f;
        g_steer_rate_fast = 0.0f;
    }
    if (v != prev || g_steer_moved_at.QuadPart == 0) {
        g_steer_moved_at = now;
    }
    last = now;
    prev = v;
    g_steer = v;
    g_steer_at = now;
    InterlockedIncrement(&g_steer_count);
    if (g_map) {
        g_map->steer = v;
        g_map->steer_rate = g_steer_rate;
        g_map->steer_frames = (uint32_t)g_steer_count;
    }
}

static void WINAPI on_drive_state(unsigned int event, const void *info, void *ctx) {
    (void)info;
    (void)ctx;
    InterlockedExchange((LONG *)&g_driving, event == 4 ? 1 : 0);
}

static void unpatch_all(void) {
    int i;
    g_live = 0;
    MemoryBarrier();
    for (i = 0; i < g_npatches; i++) {
        DWORD old = 0;
        if (!g_patches[i].slot) {
            continue;
        }
        if (VirtualProtect(g_patches[i].slot, sizeof(void *), PAGE_READWRITE, &old)) {
            *g_patches[i].slot = g_patches[i].original;
            VirtualProtect(g_patches[i].slot, sizeof(void *), old, &old);
        }
    }
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
    if (!init || !init->common.log) {
        return -7;
    }
    g_log = init->common.log;
    g_live = 1;
    QueryPerformanceFrequency(&g_qpf);
    ffb_servo_init(&g_servo);
    open_map();
    open_trace();
    __try {
        install_hook();
    } __except (EXCEPTION_EXECUTE_HANDLER) {
        g_dead = 1;
    }
    if (init->register_for_event) {
        init->register_for_event(1, (void *)on_frame, NULL);
        init->register_for_event(3, (void *)on_drive_state, NULL);
        init->register_for_event(4, (void *)on_drive_state, NULL);
    }
    if (init->register_for_channel) {
        ScsRegChannelFn reg = (ScsRegChannelFn)init->register_for_channel;
        if (reg("truck.input.steering", SCS_U32_NIL, SCS_VALUE_TYPE_FLOAT, SCS_CHANNEL_EACH_FRAME, on_steering, NULL) == 0) {
            InterlockedExchange(&g_steer_ok, 1);
        } else {
            log_msg("ffb: steering channel unavailable, servo off");
        }
    }
    log_msg("ffb: ready, waiting for rumble and LA requests");
    return 0;
}

__declspec(dllexport) void WINAPI scs_telemetry_shutdown(void) {
    unpatch_all();
    if (g_map) {
        g_map->command = FFB_CMD_NONE;
        g_map->torque = 0;
        g_map->applied = 0;
        g_map->servo_dir = 0;
        g_map->servo_cancel = 0.0f;
        g_map->servo_applied = 0;
        g_map->cancel_applied = 0;
        g_map->ff_applied = 0;
        g_map->status = 0;
        g_map->version = 0;
        UnmapViewOfFile(g_map);
        g_map = NULL;
    }
    if (g_map_handle) {
        CloseHandle(g_map_handle);
        g_map_handle = NULL;
    }
    if (g_trace) {
        FfbTrace *tr = g_trace;
        g_trace = NULL;
        MemoryBarrier();
        tr->version = 0;
        UnmapViewOfFile((void *)tr);
    }
    if (g_trace_handle) {
        CloseHandle(g_trace_handle);
        g_trace_handle = NULL;
    }
    g_log = NULL;
}
