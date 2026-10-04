#include <windows.h>
#include <TruckersMP/TruckersMP.hxx>
#include <memory>
#include <cstdint>

struct SharedState {
    volatile LONG sequence;
    std::uint32_t magic;
    std::uint32_t zone;
    std::uint32_t connected;
    std::uint64_t tick;
    std::uint64_t reserved;
};
static_assert(sizeof(SharedState) == 32);
static std::unique_ptr<TruckersMP::Session> session;
static HANDLE mapping = nullptr;
static SharedState *state = nullptr;
static std::uint32_t zone = 2;

static void publish() {
    if (!state || !session) return;
    InterlockedIncrement(&state->sequence);
    state->magic = 0x4e435a31;
    state->zone = zone;
    state->connected = session->Network().IsConnected().value_or(false) ? 1 : 0;
    state->tick = GetTickCount64();
    state->reserved = 0;
    InterlockedIncrement(&state->sequence);
}

TMP_EXPORT bool TMP_API truckersmp_init(const TruckersMP_Host *host, TruckersMP_PluginDesc *desc) {
    TruckersMP::PluginInfo info;
    info.m_name = "MonoCruise NCZ Bridge";
    info.m_author = "MonoCruise Personal Edition";
    info.m_version = "0.1.0";
    info.m_description = "Local no-collision-zone state for MonoCruise.";
    TruckersMP::FillPluginDesc(desc, info);
    session = TruckersMP::Session::Create(host);
    if (!session) return false;
    mapping = CreateFileMappingW(INVALID_HANDLE_VALUE, nullptr, PAGE_READWRITE, 0, 32, L"Local\\MonoCruise_TMP_NCZ_v1");
    if (mapping) state = static_cast<SharedState *>(MapViewOfFile(mapping, FILE_MAP_ALL_ACCESS, 0, 0, 32));
    if (!state) {
        if (mapping) CloseHandle(mapping);
        mapping = nullptr;
        session.reset();
        return false;
    }
    zone = 2;
    session->Gameplay().OnNoCollisionZone.Register([](TruckersMP::GameplayNoCollisionZoneEvent &event) {
        zone = event.GetEntered() ? 1 : 0;
        publish();
    });
    session->Network().OnConnected.Register([] { zone = 2; publish(); });
    session->Network().OnDisconnected.Register([] { zone = 2; publish(); });
    session->Render().OnPreRender.Register([] { publish(); });
    publish();
    return true;
}

TMP_EXPORT void TMP_API truckersmp_shutdown() {
    zone = 2;
    publish();
    session.reset();
    if (state) UnmapViewOfFile(state);
    if (mapping) CloseHandle(mapping);
    state = nullptr;
    mapping = nullptr;
}
