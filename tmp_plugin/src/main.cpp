// MonoCruise's TruckersMP plugin. Publishes the no-collision zone state for AEB; see tmp_plugin/README.md.
#include <TruckersMP/TruckersMP.hxx>

#include <windows.h>

#include <algorithm>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
#include <unordered_map>
#include <vector>

namespace
{
#pragma pack( push, 1 )
    // Local\MonoCruiseTmpState, decoded by core/radar/tmp_state.py. Bump kStateVersion on any change.
    struct StateData
    {
        uint32_t version;
        uint32_t heartbeat;
        uint8_t connected;
        uint8_t in_no_collision_zone;
        uint16_t players_streamed;
        uint16_t players_collidable;
        uint16_t reserved;
    };

    // Local\MonoCruiseTmpVehicles (experimental), decoded by tools/tmp_sdk_probe.py. Bump kVehiclesVersion on any change.
    struct VehiclesHeader
    {
        uint32_t version;
        uint32_t heartbeat;
        int64_t write_us;
        uint16_t own_ping_ms;
        uint16_t count;
        uint8_t connected;
        uint8_t reserved[ 3 ];
    };

    struct VehicleSlot
    {
        double position[ 3 ];
        float rotation[ 4 ];
        float linear_velocity[ 3 ];
        float angular_velocity[ 3 ];
        float sample_age_ms;
        uint32_t updates;
        float distance_m;
        uint8_t kind;
        uint8_t flags;
        uint16_t rig;
    };
#pragma pack( pop )
    static_assert( sizeof( StateData ) == 16 );
    static_assert( sizeof( VehiclesHeader ) == 24 );
    static_assert( sizeof( VehicleSlot ) == 80 );

    constexpr uint32_t kStateVersion = 1;
    constexpr wchar_t kStateName[]   = L"Local\\MonoCruiseTmpState";

    constexpr uint32_t kVehiclesVersion = 1;
    constexpr wchar_t kVehiclesName[]   = L"Local\\MonoCruiseTmpVehicles";
    constexpr size_t kMaxSlots          = 96;
    constexpr size_t kVehiclesSize      = sizeof( VehiclesHeader ) + kMaxSlots * sizeof( VehicleSlot );
    constexpr float kMaxRangeM          = 250.0f;

    constexpr uint8_t kKindTractor      = 0;
    constexpr uint8_t kKindTrailer      = 1;
    constexpr uint8_t kKindLocalTractor = 2;
    constexpr uint8_t kKindLocalTrailer = 3;

    constexpr uint8_t kFlagPlacement = 1;
    constexpr uint8_t kFlagLinear    = 2;
    constexpr uint8_t kFlagAngular   = 4;
    constexpr uint8_t kFlagSampled   = 8;

    std::unique_ptr< TruckersMP::Session > g_session;
    HANDLE g_mapping     = nullptr;
    StateData* g_state   = nullptr;
    uint32_t g_heartbeat = 0;

    std::atomic< bool > g_connected{ false };
    std::atomic< bool > g_in_zone{ false };
    std::atomic< int > g_streamed{ 0 };

    HANDLE g_vehicles_mapping = nullptr;
    uint8_t* g_vehicles       = nullptr;
    int64_t g_qpc_freq        = 1;
    bool g_vehicles_faulted   = false;

    // Arrival of each player's network samples, keyed by the opaque SDK handle. OnUpdate may run off the render thread.
    struct SampleArrival
    {
        int64_t last_qpc = 0;
        uint32_t count   = 0;
    };
    SRWLOCK g_arrivals_lock = SRWLOCK_INIT;
    std::unordered_map< uint64_t, SampleArrival > g_arrivals;

    int64_t qpc_now()
    {
        LARGE_INTEGER t;
        QueryPerformanceCounter( &t );
        return t.QuadPart;
    }

    void forget_all_arrivals()
    {
        AcquireSRWLockExclusive( &g_arrivals_lock );
        g_arrivals.clear();
        ReleaseSRWLockExclusive( &g_arrivals_lock );
    }

    void log( TruckersMP::LogLevel level, const char* text )
    {
        if ( g_session != nullptr )
        {
            g_session->Core().LogMessage( level, text );
        }
    }

    bool open_state()
    {
        g_mapping = CreateFileMappingW(
            INVALID_HANDLE_VALUE, nullptr, PAGE_READWRITE, 0, sizeof( StateData ), kStateName
        );
        if ( g_mapping == nullptr )
        {
            return false;
        }
        g_state = static_cast< StateData* >( MapViewOfFile( g_mapping, FILE_MAP_ALL_ACCESS, 0, 0, sizeof( StateData ) ) );
        if ( g_state == nullptr )
        {
            CloseHandle( g_mapping );
            g_mapping = nullptr;
            return false;
        }
        return true;
    }

    bool open_vehicles()
    {
        g_vehicles_mapping = CreateFileMappingW(
            INVALID_HANDLE_VALUE, nullptr, PAGE_READWRITE, 0, static_cast< DWORD >( kVehiclesSize ), kVehiclesName
        );
        if ( g_vehicles_mapping == nullptr )
        {
            return false;
        }
        g_vehicles = static_cast< uint8_t* >( MapViewOfFile( g_vehicles_mapping, FILE_MAP_ALL_ACCESS, 0, 0, kVehiclesSize ) );
        if ( g_vehicles == nullptr )
        {
            CloseHandle( g_vehicles_mapping );
            g_vehicles_mapping = nullptr;
            return false;
        }
        return true;
    }

    void close_state()
    {
        // Version 0 tells the reader there is no writer, without waiting for the heartbeat to go stale.
        if ( g_state != nullptr )
        {
            std::memset( g_state, 0, sizeof( StateData ) );
            UnmapViewOfFile( g_state );
            g_state = nullptr;
        }
        if ( g_mapping != nullptr )
        {
            CloseHandle( g_mapping );
            g_mapping = nullptr;
        }
    }

    void close_vehicles()
    {
        if ( g_vehicles != nullptr )
        {
            std::memset( g_vehicles, 0, kVehiclesSize );
            UnmapViewOfFile( g_vehicles );
            g_vehicles = nullptr;
        }
        if ( g_vehicles_mapping != nullptr )
        {
            CloseHandle( g_vehicles_mapping );
            g_vehicles_mapping = nullptr;
        }
        forget_all_arrivals();
    }

    void shutdown()
    {
        // Destroying the session unregisters every listener before the shared memory goes.
        g_session.reset();
        close_vehicles();
        close_state();
        g_vehicles_faulted = false;
        g_connected = false;
        g_in_zone   = false;
        g_streamed  = 0;
        g_heartbeat = 0;
    }

    void on_connection( bool connected )
    {
        // Cleared on disconnect only: spawning inside a zone may report it before OnConnected.
        g_connected = connected;
        g_streamed  = 0;
        if ( !connected )
        {
            g_in_zone = false;
        }
        forget_all_arrivals();
        log( TruckersMP::LogLevel::Info, connected ? "[MonoCruise] Connected." : "[MonoCruise] Disconnected." );
    }

    void on_zone( bool entered )
    {
        const bool was_in_zone = g_in_zone.exchange( entered );
        if ( entered )
        {
            log( TruckersMP::LogLevel::Info, g_connected ? "[MonoCruise] Entered a no-collision zone."
                                                         : "[MonoCruise] Entered a no-collision zone before connecting." );
        }
        else
        {
            log( TruckersMP::LogLevel::Info, was_in_zone ? "[MonoCruise] Left a no-collision zone."
                                                         : "[MonoCruise] Left a no-collision zone that was never reported entered." );
        }
    }

    // Network-rate callback: record the arrival only, the SDK forbids calling back into it from here.
    void on_player_update( uint64_t handle )
    {
        const int64_t now = qpc_now();
        AcquireSRWLockExclusive( &g_arrivals_lock );
        try
        {
            SampleArrival& arrival = g_arrivals[ handle ];
            arrival.last_qpc       = now;
            ++arrival.count;
        }
        catch ( ... )
        {
        }
        ReleaseSRWLockExclusive( &g_arrivals_lock );
    }

    void forget_player( uint64_t handle )
    {
        AcquireSRWLockExclusive( &g_arrivals_lock );
        g_arrivals.erase( handle );
        ReleaseSRWLockExclusive( &g_arrivals_lock );
    }

    void publish()
    {
        if ( g_state == nullptr )
        {
            return;
        }
        if ( ++g_heartbeat == 0 )
        {
            g_heartbeat = 1;
        }
        const int streamed = g_streamed.load();

        StateData data            = {};
        data.version              = kStateVersion;
        data.heartbeat            = g_heartbeat;
        data.connected            = g_connected ? 1 : 0;
        data.in_no_collision_zone = g_in_zone ? 1 : 0;
        data.players_streamed     = static_cast< uint16_t >( streamed < 0 ? 0 : ( streamed > 0xFFFF ? 0xFFFF : streamed ) );
        std::memcpy( g_state, &data, sizeof( StateData ) );
    }

    template < typename Body >
    void fill_motion( VehicleSlot& slot, const Body& body )
    {
        if ( const auto placement = body.GetPlacement() )
        {
            slot.position[ 0 ] = placement->position.x;
            slot.position[ 1 ] = placement->position.y;
            slot.position[ 2 ] = placement->position.z;
            slot.rotation[ 0 ] = placement->rotation.w;
            slot.rotation[ 1 ] = placement->rotation.x;
            slot.rotation[ 2 ] = placement->rotation.y;
            slot.rotation[ 3 ] = placement->rotation.z;
            slot.flags |= kFlagPlacement;
        }
        if ( const auto v = body.GetLinearVelocity() )
        {
            slot.linear_velocity[ 0 ] = v->x;
            slot.linear_velocity[ 1 ] = v->y;
            slot.linear_velocity[ 2 ] = v->z;
            slot.flags |= kFlagLinear;
        }
        if ( const auto w = body.GetAngularVelocity() )
        {
            slot.angular_velocity[ 0 ] = w->x;
            slot.angular_velocity[ 1 ] = w->y;
            slot.angular_velocity[ 2 ] = w->z;
            slot.flags |= kFlagAngular;
        }
    }

    struct Candidate
    {
        TruckersMP::Player player;
        float distance_m;
        bool local;
    };

    VehicleSlot base_slot( const Candidate& c, uint16_t rig, int64_t now )
    {
        SampleArrival arrival;
        bool sampled = false;
        AcquireSRWLockShared( &g_arrivals_lock );
        const auto found = g_arrivals.find( c.player.RawHandle().m_value );
        if ( found != g_arrivals.end() )
        {
            arrival = found->second;
            sampled = arrival.last_qpc != 0;
        }
        ReleaseSRWLockShared( &g_arrivals_lock );

        VehicleSlot base   = {};
        base.distance_m    = c.distance_m;
        base.rig           = rig;
        base.sample_age_ms = -1.0f;
        if ( sampled )
        {
            base.sample_age_ms = static_cast< float >( static_cast< double >( now - arrival.last_qpc ) * 1000.0 / g_qpc_freq );
            base.updates       = arrival.count;
            base.flags |= kFlagSampled;
        }
        return base;
    }

    // One rig (tractor, then its trailer) per candidate, nearest first, while two slots remain.
    size_t fill_rigs( const std::vector< Candidate >& candidates, VehicleSlot* slots, int64_t now )
    {
        size_t count = 0;
        uint16_t rig = 0;
        for ( const Candidate& c : candidates )
        {
            if ( count + 2 > kMaxSlots )
            {
                break;
            }
            const auto vehicle = c.player.GetVehicle();
            if ( !vehicle || !vehicle->IsValid() )
            {
                continue;
            }
            const VehicleSlot base = base_slot( c, rig, now );

            VehicleSlot slot = base;
            slot.kind        = c.local ? kKindLocalTractor : kKindTractor;
            fill_motion( slot, *vehicle );
            if ( slot.flags & kFlagPlacement )
            {
                slots[ count++ ] = slot;
            }
            if ( const auto trailer = vehicle->GetTrailer(); trailer && trailer->IsValid() )
            {
                VehicleSlot tslot = base;
                tslot.kind        = c.local ? kKindLocalTrailer : kKindTrailer;
                fill_motion( tslot, *trailer );
                if ( tslot.flags & kFlagPlacement )
                {
                    slots[ count++ ] = tslot;
                }
            }
            ++rig;
        }
        return count;
    }

    // Player queries run only on a server. Network().GetLastPing() is never called: off a server it
    // dereferences null inside client 0.7.7.11 and crashed the game at the main menu (2026-10-11).
    void publish_vehicles()
    {
        if ( g_vehicles == nullptr || g_session == nullptr )
        {
            return;
        }
        auto& session     = *g_session;
        const int64_t now = qpc_now();
        const bool online = g_connected.load();

        std::vector< Candidate > candidates;
        const auto local         = online ? session.Player().GetLocalPlayer() : std::nullopt;
        const uint64_t local_key = local ? local->RawHandle().m_value : 0;
        if ( local && local->IsValid() )
        {
            candidates.push_back( { *local, 0.0f, true } );
        }
        if ( const auto players = online ? session.Player().GetAllPlayers() : std::nullopt )
        {
            for ( const auto& player : *players )
            {
                if ( !player.IsValid() || player.RawHandle().m_value == local_key )
                {
                    continue;
                }
                const auto distance = player.GetDistanceFromLocalPlayer();
                if ( distance && *distance <= kMaxRangeM )
                {
                    candidates.push_back( { player, *distance, false } );
                }
            }
        }
        std::stable_sort( candidates.begin(), candidates.end(), []( const Candidate& a, const Candidate& b ) {
            return a.distance_m < b.distance_m;
        } );

        static VehicleSlot slots[ kMaxSlots ];
        const size_t count = fill_rigs( candidates, slots, now );

        VehiclesHeader header = {};
        header.version        = kVehiclesVersion;
        header.write_us       = static_cast< int64_t >( static_cast< double >( now ) * 1000000.0 / g_qpc_freq );
        header.own_ping_ms    = 0xFFFF;
        header.count          = static_cast< uint16_t >( count );
        header.connected      = online ? 1 : 0;

        // Heartbeat 0 while writing: a reader that sees one non-zero heartbeat before and after its copy got one write.
        auto* live = reinterpret_cast< volatile uint32_t* >( g_vehicles + offsetof( VehiclesHeader, heartbeat ) );
        *live      = 0;
        MemoryBarrier();
        uint8_t* body = g_vehicles + sizeof( VehiclesHeader );
        std::memcpy( body, slots, count * sizeof( VehicleSlot ) );
        std::memset( body + count * sizeof( VehicleSlot ), 0, ( kMaxSlots - count ) * sizeof( VehicleSlot ) );
        std::memcpy( g_vehicles, &header, sizeof( VehiclesHeader ) );
        MemoryBarrier();
        *live = g_heartbeat;
    }

    void report_vehicles_fault()
    {
        if ( g_vehicles != nullptr )
        {
            std::memset( g_vehicles, 0, sizeof( VehiclesHeader ) );
        }
        log( TruckersMP::LogLevel::Error, "[MonoCruise] A TruckersMP SDK call faulted; vehicle motion block off until restart." );
    }

    // Experimental: a fault inside the TruckersMP client switches the block off instead of taking the game down.
    void publish_vehicles_guarded()
    {
        if ( g_vehicles_faulted )
        {
            return;
        }
        __try
        {
            publish_vehicles();
        }
        __except ( EXCEPTION_EXECUTE_HANDLER )
        {
            g_vehicles_faulted = true;
            report_vehicles_fault();
        }
    }
}

TMP_EXPORT bool TMP_API truckersmp_init( const TruckersMP_Host* host, TruckersMP_PluginDesc* desc )
{
    TruckersMP::PluginInfo info;
    info.m_name        = "MonoCruise";
    info.m_author      = "LD-Tech";
    info.m_version     = "1.0.3";
    info.m_description = "No-collision zone state and nearby vehicle motion for MonoCruise. Reads no player identities.";
    TruckersMP::FillPluginDesc( desc, info );

    shutdown();

    // Returning false makes the client unload the DLL without truckersmp_shutdown, so clean up first.
    g_session = TruckersMP::Session::Create( host );
    if ( g_session == nullptr )
    {
        return false;
    }
    if ( !open_state() )
    {
        log( TruckersMP::LogLevel::Error, "[MonoCruise] Could not open the shared state." );
        shutdown();
        return false;
    }
    LARGE_INTEGER freq;
    QueryPerformanceFrequency( &freq );
    g_qpc_freq = freq.QuadPart > 0 ? freq.QuadPart : 1;
    if ( !open_vehicles() )
    {
        log( TruckersMP::LogLevel::Warning, "[MonoCruise] Could not open the vehicle motion block; the zone state is unaffected." );
    }

    auto& session = *g_session;
    session.Network().OnConnected.Register( [] { on_connection( true ); } );
    session.Network().OnDisconnected.Register( [] { on_connection( false ); } );
    session.Gameplay().OnNoCollisionZone.Register(
        []( TruckersMP::GameplayNoCollisionZoneEvent& e ) { on_zone( e.GetEntered() ); }
    );
    session.Player().OnStreamIn.Register( []( TruckersMP::PlayerStreamInEvent& ) { ++g_streamed; } );
    session.Player().OnStreamOut.Register( []( TruckersMP::PlayerStreamOutEvent& e ) {
        --g_streamed;
        forget_player( e.GetPlayer().RawHandle().m_value );
    } );
    session.Player().OnUpdate.Register(
        []( TruckersMP::PlayerUpdateEvent& e ) { on_player_update( e.GetPlayer().RawHandle().m_value ); }
    );

    if ( !session.Render().IsAvailable() )
    {
        log( TruckersMP::LogLevel::Warning, "[MonoCruise] Render module unavailable; the zone state will not update." );
    }
    session.Render().OnPreRender.Register( [] {
        publish();
        publish_vehicles_guarded();
    } );

    g_connected = session.Network().IsConnected().value_or( false );
    if ( const auto players = session.Player().GetAllPlayers() )
    {
        g_streamed = static_cast< int >( players->size() );
    }
    log( TruckersMP::LogLevel::Info, "[MonoCruise] No-collision zone state ready." );
    return true;
}

TMP_EXPORT void TMP_API truckersmp_shutdown( void ) { shutdown(); }
