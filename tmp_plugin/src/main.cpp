// MonoCruise's TruckersMP plugin. Publishes the no-collision zone state for AEB; see tmp_plugin/README.md.
#include <TruckersMP/TruckersMP.hxx>

#include <windows.h>

#include <atomic>
#include <cstdint>
#include <cstring>
#include <memory>

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
#pragma pack( pop )
    static_assert( sizeof( StateData ) == 16 );

    constexpr uint32_t kStateVersion = 1;
    constexpr wchar_t kStateName[]   = L"Local\\MonoCruiseTmpState";

    std::unique_ptr< TruckersMP::Session > g_session;
    HANDLE g_mapping     = nullptr;
    StateData* g_state   = nullptr;
    uint32_t g_heartbeat = 0;

    std::atomic< bool > g_connected{ false };
    std::atomic< bool > g_in_zone{ false };
    std::atomic< int > g_streamed{ 0 };

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

    void shutdown()
    {
        // Destroying the session unregisters every listener before the shared memory goes.
        g_session.reset();
        close_state();
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
}

TMP_EXPORT bool TMP_API truckersmp_init( const TruckersMP_Host* host, TruckersMP_PluginDesc* desc )
{
    TruckersMP::PluginInfo info;
    info.m_name        = "MonoCruise";
    info.m_author      = "LD-Tech";
    info.m_version     = "1.0.1";
    info.m_description = "No-collision zone state for MonoCruise's emergency braking. Reads no player identities.";
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

    auto& session = *g_session;
    session.Network().OnConnected.Register( [] { on_connection( true ); } );
    session.Network().OnDisconnected.Register( [] { on_connection( false ); } );
    session.Gameplay().OnNoCollisionZone.Register(
        []( TruckersMP::GameplayNoCollisionZoneEvent& e ) { on_zone( e.GetEntered() ); }
    );
    session.Player().OnStreamIn.Register( []( TruckersMP::PlayerStreamInEvent& ) { ++g_streamed; } );
    session.Player().OnStreamOut.Register( []( TruckersMP::PlayerStreamOutEvent& ) { --g_streamed; } );

    if ( !session.Render().IsAvailable() )
    {
        log( TruckersMP::LogLevel::Warning, "[MonoCruise] Render module unavailable; the zone state will not update." );
    }
    session.Render().OnPreRender.Register( [] { publish(); } );

    g_connected = session.Network().IsConnected().value_or( false );
    if ( const auto players = session.Player().GetAllPlayers() )
    {
        g_streamed = static_cast< int >( players->size() );
    }
    log( TruckersMP::LogLevel::Info, "[MonoCruise] No-collision zone state ready." );
    return true;
}

TMP_EXPORT void TMP_API truckersmp_shutdown( void ) { shutdown(); }
