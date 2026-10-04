/* Sums a smooth left force, then a zero-mean rumble, onto ETS2's constant-force effect.
   The steering axis is never written. Outside the sequence the game force is passed through. */

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

#include "ffb_signal.h"

#define VT_QI 0
#define VT_RELEASE 2
#define VT_CREATE_DEVICE 3
#define VT_GET_CAPS 3
#define VT_GET_EFFECT_GUID 4
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
typedef HRESULT (WINAPI *CreateDevFn)(void *self, const GUID *guid, void **out, void *unk);
typedef HRESULT (WINAPI *SetFmtFn)(void *self, const DIDATAFORMAT *fmt);
typedef HRESULT (WINAPI *CreateFxFn)(void *self, const GUID *guid, const DIEFFECT *eff, void **out, void *unk);
typedef HRESULT (WINAPI *GdsFn)(void *self, DWORD cb, void *data);
typedef HRESULT (WINAPI *SetFn)(void *self, const DIEFFECT *eff, DWORD flags);
typedef HRESULT (WINAPI *GetCapsFn)(void *self, DIDEVCAPS *caps);
typedef HRESULT (WINAPI *GetGuidFn)(void *self, GUID *guid);
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

typedef struct ProbeMap {
    volatile uint32_t generation;
    volatile int32_t sign;
} ProbeMap;

typedef struct Patch {
    void **slot;
    void *original;
} Patch;

static ScsLogFn g_log;
static ProbeMap *g_map;
static HANDLE g_map_handle;
static volatile LONG g_live;
static volatile LONG g_driving;
static volatile LONG g_depth;
static volatile LONG g_busy;
static volatile LONG g_phase;
static volatile LONG g_have_mag;
static volatile LONG g_first_mag;
static volatile LONG g_bad_size;
static LARGE_INTEGER g_qpf;
static LARGE_INTEGER g_qpc;
static int g_seen;
static int g_running;
static int g_gen_ready;
static uint32_t g_gen;
static double g_t;
static double g_wait;
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

static void log_msg(const char *msg) {
    if (g_log && msg) {
        g_log(0, msg);
    }
}

static void log_hr(const char *what, HRESULT hr) {
    char buf[128];
    snprintf(buf, sizeof buf, "ffb probe: %s 0x%08lX", what, (unsigned long)hr);
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

static long current_delta(void) {
    double dt = tick_dt();
    uint32_t gen = g_map ? g_map->generation : 0;
    int flip = g_map && g_map->sign > 0;
    long delta;
    if (!g_gen_ready) {
        g_gen = gen;
        g_gen_ready = 1;
    } else if (gen != g_gen) {
        g_gen = gen;
        g_running = 1;
        g_t = 0.0;
        g_phase = FFB_PH_RUMBLE;
        return 0;
    }
    if (!g_running) {
        if (g_seen && InterlockedOr((LONG *)&g_driving, 0)) {
            g_wait += dt;
            if (g_wait >= 1.0) {
                g_running = 1;
                g_t = 0.0;
                g_phase = FFB_PH_RUMBLE;
            }
        }
        return 0;
    }
    g_t = ffb_advance(g_t, dt);
    g_phase = ffb_signal_phase(g_t);
    delta = ffb_round(ffb_signal_sample(g_t));
    if (flip) {
        delta = -delta;
    }
    return delta;
}

static HRESULT WINAPI hook_set(void *self, const DIEFFECT *eff, DWORD flags) {
    SetFn orig = orig_set(self);
    LONG game_mag;
    long delta;
    DIEFFECT copy;
    DICONSTANTFORCE cf;
    HRESULT hr;
    if (!orig) {
        return E_FAIL;
    }
    if (!g_live) {
        return orig(self, eff, flags);
    }
    if (InterlockedIncrement(&g_depth) != 1) {
        InterlockedDecrement(&g_depth);
        return orig(self, eff, flags);
    }
    if (!eff || eff->dwSize != sizeof(DIEFFECT) || !(flags & DIEP_TYPESPECIFICPARAMS)
        || eff->cbTypeSpecificParams != sizeof(DICONSTANTFORCE) || !eff->lpvTypeSpecificParams) {
        if (eff && eff->dwSize != sizeof(DIEFFECT) && eff->dwSize != 0 && !g_bad_size) {
            g_bad_size = (LONG)eff->dwSize;
        }
        InterlockedDecrement(&g_depth);
        return orig(self, eff, flags);
    }
    g_seen = 1;
    game_mag = ((const DICONSTANTFORCE *)eff->lpvTypeSpecificParams)->lMagnitude;
    if (InterlockedOr((LONG *)&g_driving, 0) && !g_have_mag) {
        g_first_mag = game_mag;
        g_have_mag = 1;
    }
    delta = current_delta();
    if (delta == 0) {
        InterlockedDecrement(&g_depth);
        return orig(self, eff, flags);
    }
    cf.lMagnitude = ffb_apply(game_mag, delta);
    copy = *eff;
    copy.lpvTypeSpecificParams = &cf;
    hr = orig(self, &copy, flags);
    InterlockedDecrement(&g_depth);
    return hr;
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
    GUID guid;
    GetGuidFn get_guid;
    (void)ctx;
    if (!fx) {
        return DIENUM_CONTINUE;
    }
    get_guid = (GetGuidFn)(*(void ***)fx)[VT_GET_EFFECT_GUID];
    if (in_dinput((void *)get_guid) && SUCCEEDED(get_guid(fx, &guid)) && guid_eq(&guid, &k_constant_force)) {
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
    if (g_live && !g_game_enum_done && InterlockedCompareExchange(&g_busy, 1, 0) == 0) {
        try_enum_game(self);
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
    if (SUCCEEDED(hr) && out && *out && guid && guid_eq(guid, &k_constant_force)) {
        hook_effect(*out);
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
        log_msg("ffb probe: create-effect slot missing");
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
        snprintf(buf, sizeof buf, "ffb probe: saw %s", utf);
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
        log_msg("ffb probe: no controller enumerated");
    }
}

static void open_map(void) {
    g_map_handle = CreateFileMappingW(INVALID_HANDLE_VALUE, NULL, PAGE_READWRITE, 0, sizeof(ProbeMap), L"Local\\MonoCruiseFfbProbe");
    if (g_map_handle) {
        g_map = (ProbeMap *)MapViewOfFile(g_map_handle, FILE_MAP_ALL_ACCESS, 0, 0, sizeof(ProbeMap));
    }
}

static void WINAPI on_frame(unsigned int event, const void *info, void *ctx) {
    static int logged_phase = FFB_PH_WAIT;
    static int logged_mag;
    static int logged_hook;
    int phase;
    (void)event;
    (void)info;
    (void)ctx;
    if (!g_log) {
        return;
    }
    if (g_bad_size) {
        char buf[80];
        snprintf(buf, sizeof buf, "ffb probe: unexpected effect size %ld", (long)g_bad_size);
        log_msg(buf);
        g_bad_size = 0;
    }
    if (!logged_hook && g_set_orig[0]) {
        log_msg("ffb probe: constant force hooked");
        logged_hook = 1;
    }
    if (!logged_mag && g_have_mag) {
        char buf[80];
        snprintf(buf, sizeof buf, "ffb probe: game force sample %ld", (long)g_first_mag);
        log_msg(buf);
        logged_mag = 1;
    }
    phase = (int)g_phase;
    if (phase == logged_phase) {
        return;
    }
    logged_phase = phase;
    if (phase == FFB_PH_PUSH) {
        log_msg("ffb probe: smooth push left");
    } else if (phase == FFB_PH_GAP) {
        log_msg("ffb probe: push released");
    } else if (phase == FFB_PH_RUMBLE) {
        log_msg("ffb probe: lane edge rumble");
    } else if (phase == FFB_PH_DONE) {
        log_msg("ffb probe: sequence finished");
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
    open_map();
    install_hook();
    if (init->register_for_event) {
        init->register_for_event(1, (void *)on_frame, NULL);
        init->register_for_event(3, (void *)on_drive_state, NULL);
        init->register_for_event(4, (void *)on_drive_state, NULL);
    }
    log_msg("ffb probe: waiting until you are driving");
    return 0;
}

__declspec(dllexport) void WINAPI scs_telemetry_shutdown(void) {
    unpatch_all();
    if (g_map) {
        UnmapViewOfFile(g_map);
        g_map = NULL;
    }
    if (g_map_handle) {
        CloseHandle(g_map_handle);
        g_map_handle = NULL;
    }
    g_log = NULL;
}
