// MonoCruise's TruckersMP plugin. Publishes the no-collision zone state for AEB; see tmp_plugin/README.md.
#include <TruckersMP/TruckersMP.hxx>

#include <windows.h>

#include <algorithm>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <new>
#include <string>
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
        uint16_t zone_events;  // +1 per OnNoCollisionZone event, wraps; clears the reader's teleport guard
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

    constexpr uint32_t kStateVersion = 2;
    constexpr wchar_t kStateName[]   = L"Local\\MonoCruiseTmpState";

    constexpr uint32_t kVehiclesVersion = 1;
    constexpr wchar_t kVehiclesName[]   = L"Local\\MonoCruiseTmpVehicles";
    constexpr size_t kMaxSlots          = 96;
    constexpr size_t kVehiclesSize      = sizeof( VehiclesHeader ) + kMaxSlots * sizeof( VehicleSlot );
    constexpr float kMaxRangeM          = 250.0f;

    // Samples arrive at about 4.5 Hz, so 20 Hz loses nothing. Each publish gets a time budget for
    // collecting players and another for reading rigs, nearest first, so one slow client cannot stall a frame.
    constexpr int64_t kVehiclesPeriodUs = 50000;
    constexpr int64_t kCollectBudgetUs  = 800;
    constexpr int64_t kFillBudgetUs     = 1200;
    constexpr size_t kMaxArrivals       = 4096;

    // Oldest client build the vehicle block was checked against. Older or unreadable builds run zone-state-only.
    constexpr uint32_t kMinClient[ 4 ] = { 0, 7, 7, 11 };

    constexpr uint8_t kKindTractor      = 0;
    constexpr uint8_t kKindTrailer      = 1;
    constexpr uint8_t kKindLocalTractor = 2;
    constexpr uint8_t kKindLocalTrailer = 3;

    constexpr uint8_t kFlagPlacement = 1;
    constexpr uint8_t kFlagLinear    = 2;
    constexpr uint8_t kFlagAngular   = 4;
    constexpr uint8_t kFlagSampled   = 8;

    // Arrival of each player's network samples, keyed by the opaque SDK handle. OnUpdate may run off the render thread.
    struct SampleArrival
    {
        int64_t last_qpc = 0;
        uint32_t count   = 0;
    };

    struct Arrivals
    {
        SRWLOCK lock = SRWLOCK_INIT;
        std::unordered_map< uint64_t, SampleArrival > map;
    };

    // The session, the arrivals and both mappings are leaked on purpose: nothing here has a destructor
    // that runs at DLL detach or process exit, because the client may be gone by then.
    TruckersMP::Session* g_session = nullptr;
    HANDLE g_mapping               = nullptr;
    StateData* g_state             = nullptr;
    uint32_t g_heartbeat           = 0;
    std::atomic< Arrivals* > g_arrivals{ nullptr };

    HANDLE g_vehicles_mapping = nullptr;
    uint8_t* g_vehicles       = nullptr;
    int64_t g_qpc_freq        = 1;
    int64_t g_next_vehicles   = 0;
    bool g_vehicles_enabled   = false;
    bool g_budget_warned      = false;

    std::atomic< bool > g_live{ false };
    std::atomic< bool > g_dead{ false };
    std::atomic< unsigned long > g_fault_code{ 0 };
    std::atomic< bool > g_connected{ false };
    std::atomic< bool > g_in_zone{ false };
    std::atomic< uint16_t > g_zone_events{ 0 };
    std::atomic< int > g_streamed{ 0 };

    int64_t qpc_now()
    {
        LARGE_INTEGER t;
        QueryPerformanceCounter( &t );
        return t.QuadPart;
    }

    // Blocking only where the plugin is live; shutdown asks for a try-lock so a lock left held by a fault cannot hang it.
    void forget_all_arrivals( bool wait )
    {
        Arrivals* arrivals = g_arrivals.load();
        if ( arrivals == nullptr )
        {
            return;
        }
        if ( wait )
        {
            AcquireSRWLockExclusive( &arrivals->lock );
        }
        else if ( !TryAcquireSRWLockExclusive( &arrivals->lock ) )
        {
            return;
        }
        arrivals->map.clear();
        ReleaseSRWLockExclusive( &arrivals->lock );
    }

    // The mappings stay mapped until process exit; only their contents are cleared.
    void zero_published()
    {
        if ( g_state != nullptr )
        {
            std::memset( g_state, 0, sizeof( StateData ) );
        }
        if ( g_vehicles != nullptr )
        {
            *reinterpret_cast< volatile uint32_t* >( g_vehicles + offsetof( VehiclesHeader, heartbeat ) ) = 0;
            MemoryBarrier();
            std::memset( g_vehicles, 0, kVehiclesSize );
        }
    }

    // Debugger and thread-naming exceptions are not faults. Kept tiny: it can run on a nearly full stack.
    int seh_filter( unsigned long code )
    {
        switch ( code )
        {
        case 0x40010006:
        case 0x4001000A:
        case 0x406D1388:
        case 0x80000003:
        case 0x80000004:
            return EXCEPTION_CONTINUE_SEARCH;
        default:
            g_fault_code = code;
            return EXCEPTION_EXECUTE_HANDLER;
        }
    }

    // No C++ objects may live in a function with __try, so this stays a bare trampoline.
    bool run_seh( void ( *fn )( void* ), void* ctx )
    {
        __try
        {
            fn( ctx );
            return true;
        }
        __except ( seh_filter( GetExceptionCode() ) )
        {
            return false;
        }
    }

    // True when f ran to the end: a hardware fault or a C++ exception inside it both read as false.
    template < typename F >
    bool run_guarded( F& f )
    {
        struct Call
        {
            F* f;
            bool threw;
        } call{ &f, false };
        auto thunk = []( void* ctx ) {
            Call* c = static_cast< Call* >( ctx );
            try
            {
                ( *c->f )();
            }
            catch ( ... )
            {
                c->threw = true;
                g_fault_code = 0xE06D7363;
            }
        };
        return run_seh( thunk, &call ) && !call.threw;
    }

    void log_after_fault( const char* text )
    {
        TruckersMP::Session* session = g_session;
        if ( session == nullptr )
        {
            return;
        }
        auto call = [ & ] { session->Core().LogMessage( TruckersMP::LogLevel::Error, text ); };
        run_guarded( call );
    }

    // The first fault silences the plugin for good: published memory is zeroed (readers see no writer) and
    // nothing calls the SDK again. Logged once, with a call that is itself guarded.
    void enter_dead( const char* where )
    {
        if ( g_dead.exchange( true ) )
        {
            return;
        }
        zero_published();
        char text[ 160 ];
        std::snprintf(
            text, sizeof text, "[MonoCruise] A TruckersMP call faulted (0x%08lX in %s); plugin off until restart.",
            g_fault_code.load(), where
        );
        log_after_fault( text );
    }

    template < typename F >
    bool guard( const char* where, F&& f )
    {
        if ( g_dead.load() )
        {
            return false;
        }
        if ( run_guarded( f ) )
        {
            return true;
        }
        enter_dead( where );
        return false;
    }

    template < typename F >
    void callback( const char* where, F&& f )
    {
        if ( g_live.load() )
        {
            guard( where, f );
        }
    }

    // Only inside a guard: it is an SDK call.
    void sdk_log( TruckersMP::LogLevel level, const char* text )
    {
        if ( g_session != nullptr )
        {
            g_session->Core().LogMessage( level, text );
        }
    }

    void log_guarded( TruckersMP::LogLevel level, const char* text )
    {
        guard( "log", [ & ] { sdk_log( level, text ); } );
    }

    bool open_state()
    {
        if ( g_state != nullptr )
        {
            return true;
        }
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
        if ( g_vehicles != nullptr )
        {
            return true;
        }
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

    // Unregistering keeps a client that unloads the DLL after shutdown from calling into freed code.
    void unregister_listeners()
    {
        if ( g_session == nullptr )
        {
            return;
        }
        TruckersMP::Session& session = *g_session;
        session.Network().OnConnected.UnregisterAll();
        session.Network().OnDisconnected.UnregisterAll();
        session.Gameplay().OnNoCollisionZone.UnregisterAll();
        session.Player().OnStreamIn.UnregisterAll();
        session.Player().OnStreamOut.UnregisterAll();
        session.Player().OnUpdate.UnregisterAll();
        session.Render().OnPreRender.UnregisterAll();
    }

    void shutdown()
    {
        g_live = false;
        guard( "shutdown", [] { unregister_listeners(); } );
        zero_published();
        forget_all_arrivals( false );
        g_session          = nullptr;
        g_vehicles_enabled = false;
        g_next_vehicles    = 0;
        g_connected        = false;
        g_in_zone          = false;
        g_zone_events      = 0;
        g_streamed         = 0;
        g_heartbeat        = 0;
    }

    void on_connection( bool connected )
    {
        callback( "connection", [ & ] {
            // Cleared on disconnect only: spawning inside a zone may report it before OnConnected.
            g_connected = connected;
            g_streamed  = 0;
            if ( !connected )
            {
                g_in_zone = false;
            }
            forget_all_arrivals( true );
            g_next_vehicles = 0;
            sdk_log( TruckersMP::LogLevel::Info, connected ? "[MonoCruise] Connected." : "[MonoCruise] Disconnected." );
        } );
    }

    void on_zone( bool entered )
    {
        callback( "zone", [ & ] {
            const bool was_in_zone = g_in_zone.exchange( entered );
            ++g_zone_events;
            if ( entered )
            {
                sdk_log(
                    TruckersMP::LogLevel::Info, g_connected ? "[MonoCruise] Entered a no-collision zone."
                                                            : "[MonoCruise] Entered a no-collision zone before connecting."
                );
            }
            else
            {
                sdk_log(
                    TruckersMP::LogLevel::Info, was_in_zone ? "[MonoCruise] Left a no-collision zone."
                                                            : "[MonoCruise] Left a no-collision zone that was never reported entered."
                );
            }
        } );
    }

    // Network-rate callback: record the arrival only, the SDK forbids calling back into it from here.
    void on_player_update( uint64_t handle )
    {
        Arrivals* arrivals = g_arrivals.load();
        if ( arrivals == nullptr )
        {
            return;
        }
        const int64_t now = qpc_now();
        AcquireSRWLockExclusive( &arrivals->lock );
        try
        {
            if ( arrivals->map.size() >= kMaxArrivals )
            {
                arrivals->map.clear();
            }
            SampleArrival& arrival = arrivals->map[ handle ];
            arrival.last_qpc       = now;
            ++arrival.count;
        }
        catch ( ... )
        {
        }
        ReleaseSRWLockExclusive( &arrivals->lock );
    }

    void forget_player( uint64_t handle )
    {
        Arrivals* arrivals = g_arrivals.load();
        if ( arrivals == nullptr )
        {
            return;
        }
        AcquireSRWLockExclusive( &arrivals->lock );
        arrivals->map.erase( handle );
        ReleaseSRWLockExclusive( &arrivals->lock );
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
        data.zone_events          = g_zone_events.load();
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

    struct Budget
    {
        int64_t start;
        int64_t limit;

        bool spent() const { return qpc_now() - start > limit; }
    };

    VehicleSlot base_slot( const Candidate& c, uint16_t rig, int64_t now )
    {
        SampleArrival arrival;
        bool sampled         = false;
        Arrivals* arrivals   = g_arrivals.load();
        if ( arrivals != nullptr )
        {
            AcquireSRWLockShared( &arrivals->lock );
            const auto found = arrivals->map.find( c.player.RawHandle().m_value );
            if ( found != arrivals->map.end() )
            {
                arrival = found->second;
                sampled = arrival.last_qpc != 0;
            }
            ReleaseSRWLockShared( &arrivals->lock );
        }

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

    // One rig (tractor, then its trailer) per candidate, nearest first, while two slots and budget remain.
    size_t fill_rigs( const std::vector< Candidate >& candidates, VehicleSlot* slots, int64_t now, const Budget& budget, bool& over )
    {
        size_t count = 0;
        uint16_t rig = 0;
        for ( const Candidate& c : candidates )
        {
            if ( count + 2 > kMaxSlots )
            {
                break;
            }
            if ( budget.spent() )
            {
                over = true;
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
    void publish_vehicles( int64_t now )
    {
        if ( g_vehicles == nullptr || g_session == nullptr )
        {
            return;
        }
        auto& session       = *g_session;
        const Budget collect{ now, g_qpc_freq * kCollectBudgetUs / 1000000 };
        const bool online   = g_connected.load();
        bool over         = false;

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
                if ( collect.spent() )
                {
                    over = true;
                    break;
                }
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
        const Budget fill{ qpc_now(), g_qpc_freq * kFillBudgetUs / 1000000 };
        const size_t count = fill_rigs( candidates, slots, now, fill, over );

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

        if ( over && !g_budget_warned )
        {
            g_budget_warned = true;
            sdk_log( TruckersMP::LogLevel::Warning, "[MonoCruise] Vehicle motion block over its time budget; the farthest rigs were left out." );
        }
    }

    void publish_vehicles_throttled()
    {
        if ( !g_vehicles_enabled || g_vehicles == nullptr )
        {
            return;
        }
        const int64_t now = qpc_now();
        if ( now < g_next_vehicles )
        {
            return;
        }
        g_next_vehicles = now + g_qpc_freq * kVehiclesPeriodUs / 1000000;
        publish_vehicles( now );
    }

    // Up to four dotted numbers after an optional leading v. Anything else is unreadable and counts as untested.
    bool client_is_tested( const std::string& version )
    {
        uint32_t parts[ 4 ] = { 0, 0, 0, 0 };
        size_t at           = 0;
        size_t count        = 0;
        while ( at < version.size() && ( version[ at ] == ' ' || version[ at ] == 'v' || version[ at ] == 'V' ) )
        {
            ++at;
        }
        while ( at < version.size() && count < 4 )
        {
            if ( version[ at ] < '0' || version[ at ] > '9' )
            {
                return false;
            }
            uint32_t value = 0;
            for ( ; at < version.size() && version[ at ] >= '0' && version[ at ] <= '9'; ++at )
            {
                value = value * 10 + static_cast< uint32_t >( version[ at ] - '0' );
                if ( value > 99999 )
                {
                    return false;
                }
            }
            parts[ count++ ] = value;
            if ( at < version.size() && version[ at ] == '.' )
            {
                ++at;
            }
            else
            {
                break;
            }
        }
        while ( at < version.size() && ( version[ at ] == ' ' || version[ at ] == '\r' || version[ at ] == '\n' ) )
        {
            ++at;
        }
        if ( count == 0 || at != version.size() )
        {
            return false;
        }
        for ( size_t i = 0; i < 4; ++i )
        {
            if ( parts[ i ] != kMinClient[ i ] )
            {
                return parts[ i ] > kMinClient[ i ];
            }
        }
        return true;
    }

    void register_listeners( TruckersMP::Session& session )
    {
        session.Network().OnConnected.Register( [] { on_connection( true ); } );
        session.Network().OnDisconnected.Register( [] { on_connection( false ); } );
        session.Gameplay().OnNoCollisionZone.Register(
            []( TruckersMP::GameplayNoCollisionZoneEvent& e ) { on_zone( e.GetEntered() ); }
        );
        session.Player().OnStreamIn.Register( []( TruckersMP::PlayerStreamInEvent& ) {
            callback( "stream-in", [] { ++g_streamed; } );
        } );
        session.Player().OnStreamOut.Register( []( TruckersMP::PlayerStreamOutEvent& e ) {
            callback( "stream-out", [ &e ] {
                --g_streamed;
                forget_player( e.GetPlayer().RawHandle().m_value );
            } );
        } );
        session.Player().OnUpdate.Register( []( TruckersMP::PlayerUpdateEvent& e ) {
            callback( "update", [ &e ] { on_player_update( e.GetPlayer().RawHandle().m_value ); } );
        } );

        if ( !session.Render().IsAvailable() )
        {
            sdk_log( TruckersMP::LogLevel::Warning, "[MonoCruise] Render module unavailable; the zone state will not update." );
        }
        session.Render().OnPreRender.Register( [] {
            callback( "render", [] {
                publish();
                publish_vehicles_throttled();
            } );
        } );
    }
}

TMP_EXPORT bool TMP_API truckersmp_init( const TruckersMP_Host* host, TruckersMP_PluginDesc* desc )
{
    TruckersMP::PluginInfo info;
    info.m_name        = "MonoCruise";
    info.m_author      = "LD-Tech";
    info.m_version     = "1.0.5";
    info.m_description = "No-collision zone state and nearby vehicle motion for MonoCruise. Reads no player identities.";

    shutdown();
    g_dead       = false;
    g_fault_code = 0;
    g_budget_warned = false;

    // Returning false makes the client unload the DLL without truckersmp_shutdown, so nothing may stay registered.
    TruckersMP::Session* session = nullptr;
    const bool created           = guard( "init", [ & ] {
        TruckersMP::FillPluginDesc( desc, info );
        session = TruckersMP::Session::Create( host ).release();
    } );
    if ( !created || session == nullptr )
    {
        return false;
    }
    g_session = session;

    if ( !open_state() )
    {
        log_guarded( TruckersMP::LogLevel::Error, "[MonoCruise] Could not open the shared state." );
        shutdown();
        return false;
    }
    LARGE_INTEGER freq;
    QueryPerformanceFrequency( &freq );
    g_qpc_freq = freq.QuadPart > 0 ? freq.QuadPart : 1;
    if ( g_arrivals.load() == nullptr )
    {
        g_arrivals = new ( std::nothrow ) Arrivals;
    }

    std::string client;
    bool have_client = false;
    const bool asked = guard( "init", [ & ] {
        if ( const auto version = session->Core().ClientVersion() )
        {
            client      = *version;
            have_client = true;
        }
    } );
    if ( !asked )
    {
        shutdown();
        return false;
    }
    g_vehicles_enabled = have_client && client_is_tested( client );
    if ( g_vehicles_enabled && !open_vehicles() )
    {
        g_vehicles_enabled = false;
        log_guarded( TruckersMP::LogLevel::Warning, "[MonoCruise] Could not open the vehicle motion block; the zone state is unaffected." );
    }
    if ( !g_vehicles_enabled )
    {
        std::string text = have_client ? "[MonoCruise] TruckersMP client " + client + " is not at or above the tested 0.7.7.11 or is unreadable; zone state only."
                                       : "[MonoCruise] TruckersMP client version unavailable; zone state only.";
        log_guarded( TruckersMP::LogLevel::Warning, text.c_str() );
    }

    // From here the client holds our callbacks, so a fault leaves the plugin loaded and inert instead of unloading it.
    g_live = true;
    if ( !guard( "init", [ & ] { register_listeners( *session ); } ) )
    {
        return true;
    }
    guard( "init", [ & ] {
        g_connected = session->Network().IsConnected().value_or( false );
        if ( g_connected.load() )
        {
            if ( const auto players = session->Player().GetAllPlayers() )
            {
                g_streamed = static_cast< int >( players->size() );
            }
        }
        sdk_log( TruckersMP::LogLevel::Info, "[MonoCruise] No-collision zone state ready." );
    } );
    return true;
}

TMP_EXPORT void TMP_API truckersmp_shutdown( void ) { shutdown(); }

// At detach only our own memory is touched: the client may already be gone, and the loader lock is held.
BOOL WINAPI DllMain( HINSTANCE, DWORD reason, LPVOID )
{
    if ( reason == DLL_PROCESS_DETACH )
    {
        g_live = false;
        g_dead = true;
        zero_published();
    }
    return TRUE;
}
