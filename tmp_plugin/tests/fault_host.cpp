// Fault-injection host for monocruise_tmp.dll: a fake TruckersMP client whose tables fault, return junk and call back late.
// Usage: fault_host <dll> <scenario>, or fault_host --list. Exit 0 means the scenario passed; see tmp_plugin/README.md.
#include <TruckersMP/TruckersMP.h>

#include <windows.h>

#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <map>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

namespace
{
    enum class Act
    {
        Normal,
        Fail,
        Crash,
        Throw
    };

    struct Listener
    {
        uint64_t id;
        TruckersMP_EventFn fn;
        void* ctx;
    };

    constexpr size_t kVehiclesSize = 24 + 96 * 80;

    std::vector< Listener > g_listeners;
    std::vector< Listener > g_remembered;
    std::map< std::string, Act > g_acts;
    std::map< std::string, int > g_hits;
    std::vector< std::string > g_logs;
    std::set< void* > g_allocs;
    std::atomic< long > g_calls{ 0 };
    std::atomic< bool > g_host_gone{ false };
    std::string g_version = "0.7.7.11";
    int g_version_mode    = 0;
    bool g_connected      = true;
    bool g_no_core        = false;
    int g_players         = 12;
    int g_list_mode       = 0;
    int g_burn_us         = 0;
    int g_failures        = 0;
    int g_ctx_token       = 0;

    [[noreturn]] void die( int code, const char* what )
    {
        std::printf( "FAIL: %s\n", what );
        std::fflush( stdout );
        TerminateProcess( GetCurrentProcess(), static_cast< UINT >( code ) );
        for ( ;; )
        {
            Sleep( 1000 );
        }
    }

    [[noreturn]] void forbidden( const char* name )
    {
        std::printf( "FAIL: the plugin called %s\n", name );
        std::fflush( stdout );
        TerminateProcess( GetCurrentProcess(), 91 );
        for ( ;; )
        {
            Sleep( 1000 );
        }
    }

#define CHECK( cond )                                                    \
    do                                                                   \
    {                                                                    \
        if ( !( cond ) )                                                 \
        {                                                                \
            std::printf( "FAIL: %s (line %d)\n", #cond, __LINE__ );      \
            ++g_failures;                                                \
        }                                                                \
    } while ( 0 )

    int64_t qpc_us()
    {
        LARGE_INTEGER t, f;
        QueryPerformanceCounter( &t );
        QueryPerformanceFrequency( &f );
        return t.QuadPart * 1000000 / f.QuadPart;
    }

    // True when the call must report failure. Crash and Throw do not return.
    bool injected( const char* name )
    {
        if ( g_host_gone )
        {
            die( 95, "the plugin called the client after it was gone" );
        }
        ++g_calls;
        ++g_hits[ name ];
        const auto found = g_acts.find( name );
        switch ( found == g_acts.end() ? Act::Normal : found->second )
        {
        case Act::Fail:
            return true;
        case Act::Crash:
            *reinterpret_cast< volatile int* >( 8 ) = 1;
            return true;
        case Act::Throw:
            throw std::runtime_error( name );
        default:
            return false;
        }
    }

    const char* dup_bytes( const std::string& text )
    {
        char* copy = static_cast< char* >( std::malloc( text.size() + 1 ) );
        std::memcpy( copy, text.c_str(), text.size() + 1 );
        g_allocs.insert( copy );
        return copy;
    }

    TruckersMP_EResult core_add( TruckersMP_PluginContext*, uint64_t id, TruckersMP_EventFn fn, void* ctx )
    {
        if ( injected( "Core.AddListener" ) )
        {
            return TruckersMP_Result_InternalError;
        }
        g_listeners.push_back( { id, fn, ctx } );
        return TruckersMP_Result_Ok;
    }

    TruckersMP_EResult core_remove( TruckersMP_PluginContext*, uint64_t id, TruckersMP_EventFn fn, void* ctx )
    {
        if ( g_host_gone )
        {
            die( 92, "the plugin called RemoveListener after the client was gone" );
        }
        if ( injected( "Core.RemoveListener" ) )
        {
            return TruckersMP_Result_InternalError;
        }
        for ( size_t i = 0; i < g_listeners.size(); ++i )
        {
            if ( g_listeners[ i ].id == id && g_listeners[ i ].fn == fn && g_listeners[ i ].ctx == ctx )
            {
                g_listeners.erase( g_listeners.begin() + static_cast< std::ptrdiff_t >( i ) );
                break;
            }
        }
        return TruckersMP_Result_Ok;
    }

    TruckersMP_EResult core_unlock( TruckersMP_PluginContext*, TruckersMP_String ) { forbidden( "Core.UnlockIntent" ); }

    TruckersMP_EResult core_log( TruckersMP_PluginContext*, TruckersMP_LogLevel, TruckersMP_String text )
    {
        if ( g_host_gone )
        {
            die( 93, "the plugin logged after the client was gone" );
        }
        if ( injected( "Core.LogMessage" ) )
        {
            return TruckersMP_Result_InternalError;
        }
        g_logs.emplace_back( text.m_ptr, text.m_size );
        return TruckersMP_Result_Ok;
    }

    bool core_game_id( TruckersMP_PluginContext*, TruckersMP_String* ) { forbidden( "Core.GameID" ); }

    bool core_client_version( TruckersMP_PluginContext*, TruckersMP_String* out )
    {
        if ( injected( "Core.ClientVersion" ) )
        {
            return false;
        }
        if ( g_version_mode == 1 )
        {
            out->m_ptr  = reinterpret_cast< const char* >( 0x1000 );
            out->m_size = 8;
            return true;
        }
        out->m_ptr  = dup_bytes( g_version );
        out->m_size = static_cast< uint32_t >( g_version.size() );
        return true;
    }

    bool net_connected( TruckersMP_PluginContext*, bool* out )
    {
        if ( injected( "Network.IsConnected" ) )
        {
            return false;
        }
        *out = g_connected;
        return true;
    }

    bool net_ping( TruckersMP_PluginContext*, uint16_t* ) { forbidden( "Network.GetLastPing" ); }

    bool net_server( TruckersMP_PluginContext*, uint64_t* ) { forbidden( "Network.GetGameServerID" ); }

    bool pl_local( TruckersMP_PluginContext*, TruckersMP_Player_Handle* out )
    {
        if ( injected( "Players.GetLocalPlayer" ) || !g_connected )
        {
            return false;
        }
        out->m_value = 1;
        return true;
    }

    bool pl_by_id( TruckersMP_PluginContext*, int32_t, TruckersMP_Player_Handle* ) { forbidden( "Players.GetPlayerByID" ); }

    bool pl_all( TruckersMP_PluginContext*, TruckersMP_List* out )
    {
        if ( injected( "Players.GetAllPlayers" ) )
        {
            return false;
        }
        if ( g_list_mode == 1 )
        {
            out->m_items  = reinterpret_cast< const void* >( 0x10 );
            out->m_count  = 0xFFFFFFFFu;
            out->m_stride = 8;
            return true;
        }
        if ( g_list_mode == 2 )
        {
            out->m_items  = nullptr;
            out->m_count  = 5;
            out->m_stride = 8;
            return true;
        }
        auto* items = static_cast< TruckersMP_Player_Handle* >( std::malloc( sizeof( TruckersMP_Player_Handle ) * static_cast< size_t >( g_players ) ) );
        g_allocs.insert( items );
        for ( int i = 0; i < g_players; ++i )
        {
            items[ i ].m_value = static_cast< uint64_t >( i + 2 );
        }
        out->m_items  = items;
        out->m_count  = static_cast< uint32_t >( g_players );
        out->m_stride = sizeof( TruckersMP_Player_Handle );
        return true;
    }

    // Identity, staff, patron, latency and collision queries: the plugin must never reach any of them.
#define FORBID_PLAYER( fn, type ) \
    bool fn( TruckersMP_PluginContext*, TruckersMP_Player_Handle, type* ) { forbidden( #fn ); }
    FORBID_PLAYER( ph_GetPlayerID, int32_t )
    FORBID_PLAYER( ph_GetAccountID, uint64_t )
    FORBID_PLAYER( ph_GetSteamID, uint64_t )
    FORBID_PLAYER( ph_GetUsername, TruckersMP_String )
    FORBID_PLAYER( ph_GetTagText, TruckersMP_String )
    FORBID_PLAYER( ph_GetTagColor, TruckersMP_Color )
    FORBID_PLAYER( ph_IsPatron, bool )
    FORBID_PLAYER( ph_IsGameModerator, bool )
    FORBID_PLAYER( ph_IsTeamMember, bool )
    FORBID_PLAYER( ph_IsManager, bool )
    FORBID_PLAYER( ph_CanStreamFromCamera, bool )
    FORBID_PLAYER( ph_StreamsFromCamera, bool )
    FORBID_PLAYER( ph_GetNetworkLatency, uint16_t )
    FORBID_PLAYER( ph_GetTrailer, TruckersMP_Trailer_Handle )
    FORBID_PLAYER( ph_GetDistanceToCamera, float )

    bool ph_CanCollideWith( TruckersMP_PluginContext*, TruckersMP_Player_Handle, TruckersMP_Player_Handle, bool* ) { forbidden( "ph_CanCollideWith" ); }

    bool ph_GetVehicle( TruckersMP_PluginContext*, TruckersMP_Player_Handle self, TruckersMP_Vehicle_Handle* out )
    {
        if ( injected( "PlayerHandle.GetVehicle" ) )
        {
            return false;
        }
        out->m_value = 100 + self.m_value;
        return true;
    }

    bool ph_GetDistanceFromLocalPlayer( TruckersMP_PluginContext*, TruckersMP_Player_Handle self, float* out )
    {
        if ( injected( "PlayerHandle.GetDistanceFromLocalPlayer" ) )
        {
            return false;
        }
        const int64_t until = qpc_us() + g_burn_us;
        while ( qpc_us() < until )
        {
        }
        *out = 1.0f + 0.5f * static_cast< float >( self.m_value );
        return true;
    }

    bool veh_GetTrailer( TruckersMP_PluginContext*, TruckersMP_Vehicle_Handle self, TruckersMP_Trailer_Handle* out )
    {
        if ( injected( "Vehicle.GetTrailer" ) || self.m_value % 2 != 0 )
        {
            return false;
        }
        out->m_value = 200 + self.m_value;
        return true;
    }

    bool veh_GetPlacement( TruckersMP_PluginContext*, TruckersMP_Vehicle_Handle self, TruckersMP_Placement* out )
    {
        if ( injected( "Vehicle.GetPlacement" ) )
        {
            return false;
        }
        *out = { { 10.0 * static_cast< double >( self.m_value ), 0.0, 5.0 }, { 1.0f, 0.0f, 0.0f, 0.0f } };
        return true;
    }

    bool veh_GetLinearVelocity( TruckersMP_PluginContext*, TruckersMP_Vehicle_Handle, TruckersMP_Float3* out )
    {
        if ( injected( "Vehicle.GetLinearVelocity" ) )
        {
            return false;
        }
        *out = { 0.0f, 0.0f, -20.0f };
        return true;
    }

    bool veh_GetAngularVelocity( TruckersMP_PluginContext*, TruckersMP_Vehicle_Handle, TruckersMP_Float3* out )
    {
        if ( injected( "Vehicle.GetAngularVelocity" ) )
        {
            return false;
        }
        *out = { 0.0f, 0.01f, 0.0f };
        return true;
    }

    bool veh_GetBoundingBox( TruckersMP_PluginContext*, TruckersMP_Vehicle_Handle, TruckersMP_Bounds* ) { forbidden( "Vehicle.GetBoundingBox" ); }

    bool trl_Exists( TruckersMP_PluginContext*, TruckersMP_Trailer_Handle, bool* ) { forbidden( "Trailer.Exists" ); }

    bool trl_GetPlacement( TruckersMP_PluginContext*, TruckersMP_Trailer_Handle self, TruckersMP_Placement* out )
    {
        if ( injected( "Trailer.GetPlacement" ) )
        {
            return false;
        }
        *out = { { 10.0 * static_cast< double >( self.m_value ), 0.0, 17.0 }, { 1.0f, 0.0f, 0.0f, 0.0f } };
        return true;
    }

    bool trl_GetLinearVelocity( TruckersMP_PluginContext*, TruckersMP_Trailer_Handle, TruckersMP_Float3* out )
    {
        if ( injected( "Trailer.GetLinearVelocity" ) )
        {
            return false;
        }
        *out = { 0.0f, 0.0f, -20.0f };
        return true;
    }

    bool trl_GetAngularVelocity( TruckersMP_PluginContext*, TruckersMP_Trailer_Handle, TruckersMP_Float3* out )
    {
        if ( injected( "Trailer.GetAngularVelocity" ) )
        {
            return false;
        }
        *out = { 0.0f, 0.0f, 0.0f };
        return true;
    }

    bool trl_GetBoundingBox( TruckersMP_PluginContext*, TruckersMP_Trailer_Handle, TruckersMP_Bounds* ) { forbidden( "Trailer.GetBoundingBox" ); }

    bool render_id( TruckersMP_PluginContext*, TruckersMP_RendererID* ) { forbidden( "Render.GetRendererID" ); }

    bool render_device( TruckersMP_PluginContext*, uint64_t* ) { forbidden( "Render.GetDeviceHandle" ); }

    bool gp_packages( TruckersMP_PluginContext*, TruckersMP_List* ) { forbidden( "Gameplay.GetMountedPackages" ); }

#define FORBID_ACCOUNT( fn, type ) \
    bool fn( TruckersMP_PluginContext*, type* ) { forbidden( #fn ); }
    FORBID_ACCOUNT( acct_GetAccountID, uint64_t )
    FORBID_ACCOUNT( acct_IsGameModerator, bool )
    FORBID_ACCOUNT( acct_IsTeamMember, bool )
    FORBID_ACCOUNT( acct_IsPatron, bool )
    FORBID_ACCOUNT( acct_GetCompanyID, uint64_t )

    const TruckersMP_Core g_core                 = { core_add, core_remove, core_unlock, core_log, core_game_id, core_client_version };
    const TruckersMP_Accounts g_accounts         = { acct_GetAccountID, acct_IsGameModerator, acct_IsTeamMember, acct_IsPatron, acct_GetCompanyID };
    const TruckersMP_Gameplay g_gameplay         = { gp_packages };
    const TruckersMP_Network g_network           = { net_connected, net_ping, net_server };
    const TruckersMP_Players g_players_table     = { pl_local, pl_by_id, pl_all };
    const TruckersMP_PlayerHandle g_player_handle = {
        ph_GetPlayerID,       ph_GetAccountID,         ph_GetSteamID,        ph_GetUsername,   ph_GetTagText,
        ph_GetTagColor,       ph_IsPatron,             ph_IsGameModerator,   ph_IsTeamMember,  ph_IsManager,
        ph_CanStreamFromCamera, ph_StreamsFromCamera,  ph_GetNetworkLatency, ph_GetVehicle,    ph_GetTrailer,
        ph_GetDistanceFromLocalPlayer, ph_GetDistanceToCamera, ph_CanCollideWith };
    const TruckersMP_Render g_render   = { render_id, render_device };
    const TruckersMP_Trailer g_trailer = { trl_Exists, trl_GetPlacement, trl_GetLinearVelocity, trl_GetAngularVelocity, trl_GetBoundingBox };
    const TruckersMP_Vehicle g_vehicle = { veh_GetTrailer, veh_GetPlacement, veh_GetLinearVelocity, veh_GetAngularVelocity, veh_GetBoundingBox };

    TruckersMP_EResult host_declare( TruckersMP_PluginContext*, uint32_t ) { return TruckersMP_Result_Ok; }

    const void* host_get_api( TruckersMP_PluginContext*, uint64_t id )
    {
        switch ( id )
        {
        case TRUCKERSMP_API_CORE:
            return g_no_core ? nullptr : &g_core;
        case TRUCKERSMP_API_ACCOUNTS:
            return &g_accounts;
        case TRUCKERSMP_API_GAMEPLAY:
            return &g_gameplay;
        case TRUCKERSMP_API_NETWORK:
            return &g_network;
        case TRUCKERSMP_API_PLAYERS:
            return &g_players_table;
        case TRUCKERSMP_API_PLAYER_HANDLE:
            return &g_player_handle;
        case TRUCKERSMP_API_RENDER:
            return &g_render;
        case TRUCKERSMP_API_TRAILER:
            return &g_trailer;
        case TRUCKERSMP_API_VEHICLE:
            return &g_vehicle;
        default:
            return nullptr;
        }
    }

    void host_free( TruckersMP_PluginContext*, void* ptr )
    {
        if ( g_host_gone )
        {
            die( 95, "the plugin called the client after it was gone" );
        }
        if ( g_allocs.erase( ptr ) != 0 )
        {
            std::free( ptr );
        }
    }

    TruckersMP_Host g_host = {
        sizeof( TruckersMP_Host ), reinterpret_cast< TruckersMP_PluginContext* >( &g_ctx_token ), host_declare, host_get_api, host_free
    };

    void fire( uint64_t id, void* payload, uint32_t size, bool stale = false )
    {
        const std::vector< Listener > copy = stale ? g_remembered : g_listeners;
        for ( const Listener& l : copy )
        {
            if ( l.id == id )
            {
                l.fn( l.ctx, id, payload, size );
            }
        }
    }

    void fire_connection( bool up, bool stale = false )
    {
        fire( up ? TRUCKERSMP_EVT_NETWORK_CONNECTED : TRUCKERSMP_EVT_NETWORK_DISCONNECTED, nullptr, 0, stale );
    }

    void fire_zone( bool entered, bool stale = false )
    {
        TruckersMP_Evt_GameplayNoCollisionZone payload{ entered };
        fire( TRUCKERSMP_EVT_GAMEPLAY_NO_COLLISION_ZONE, &payload, sizeof payload, stale );
    }

    void fire_player( uint64_t id, uint64_t handle, bool stale = false )
    {
        TruckersMP_Player_Handle payload{ handle };
        fire( id, &payload, sizeof payload, stale );
    }

    void render( bool stale = false ) { fire( TRUCKERSMP_EVT_RENDER_PRE_RENDER, nullptr, 0, stale ); }

    void fire_everything( bool stale = false )
    {
        fire_connection( true, stale );
        fire_connection( false, stale );
        fire_zone( true, stale );
        fire_zone( false, stale );
        fire_player( TRUCKERSMP_EVT_PLAYER_STREAM_IN, 2, stale );
        fire_player( TRUCKERSMP_EVT_PLAYER_STREAM_OUT, 2, stale );
        fire_player( TRUCKERSMP_EVT_PLAYER_UPDATE, 3, stale );
        for ( int i = 0; i < 3; ++i )
        {
            render( stale );
        }
    }

    struct View
    {
        HANDLE mapping       = nullptr;
        const uint8_t* bytes = nullptr;
        size_t size          = 0;
    };

    View g_state_view;
    View g_vehicles_view;

    // A test must never touch a running game's real shared memory, so every mapping name carries this pid.
    std::wstring test_name( std::wstring name )
    {
        const std::wstring real = L"MonoCruise";
        const size_t at         = name.find( real );
        if ( at == std::wstring::npos )
        {
            die( 94, "the plugin asked for a mapping this host does not know" );
        }
        return name.replace( at, real.size(), L"McTest" + std::to_wstring( GetCurrentProcessId() ) + L"_" );
    }

    HANDLE WINAPI redirected_create_mapping(
        HANDLE file, LPSECURITY_ATTRIBUTES attributes, DWORD protect, DWORD high, DWORD low, LPCWSTR name
    )
    {
        const std::wstring renamed = name != nullptr ? test_name( name ) : std::wstring();
        return CreateFileMappingW( file, attributes, protect, high, low, name != nullptr ? renamed.c_str() : nullptr );
    }

    // Patches the plugin's import of CreateFileMappingW, so the shipped binary is tested as it is.
    bool redirect_mappings( HMODULE module )
    {
        auto* base = reinterpret_cast< uint8_t* >( module );
        auto* dos  = reinterpret_cast< IMAGE_DOS_HEADER* >( base );
        auto* nt   = reinterpret_cast< IMAGE_NT_HEADERS* >( base + dos->e_lfanew );
        const IMAGE_DATA_DIRECTORY dir = nt->OptionalHeader.DataDirectory[ IMAGE_DIRECTORY_ENTRY_IMPORT ];
        for ( auto* desc = reinterpret_cast< IMAGE_IMPORT_DESCRIPTOR* >( base + dir.VirtualAddress ); desc->Name != 0; ++desc )
        {
            auto* thunk = reinterpret_cast< IMAGE_THUNK_DATA* >( base + desc->FirstThunk );
            auto* named = reinterpret_cast< IMAGE_THUNK_DATA* >( base + desc->OriginalFirstThunk );
            for ( ; named->u1.AddressOfData != 0; ++named, ++thunk )
            {
                if ( IMAGE_SNAP_BY_ORDINAL( named->u1.Ordinal ) )
                {
                    continue;
                }
                const auto* entry = reinterpret_cast< IMAGE_IMPORT_BY_NAME* >( base + named->u1.AddressOfData );
                if ( std::strcmp( entry->Name, "CreateFileMappingW" ) != 0 )
                {
                    continue;
                }
                DWORD old = 0;
                VirtualProtect( &thunk->u1.Function, sizeof thunk->u1.Function, PAGE_READWRITE, &old );
                thunk->u1.Function = reinterpret_cast< ULONGLONG >( &redirected_create_mapping );
                VirtualProtect( &thunk->u1.Function, sizeof thunk->u1.Function, old, &old );
                return true;
            }
        }
        return false;
    }

    View open_view( const wchar_t* name, size_t size )
    {
        View view;
        view.mapping = OpenFileMappingW( FILE_MAP_READ, FALSE, test_name( name ).c_str() );
        if ( view.mapping != nullptr )
        {
            view.bytes = static_cast< const uint8_t* >( MapViewOfFile( view.mapping, FILE_MAP_READ, 0, 0, size ) );
            view.size  = size;
        }
        return view;
    }

    void attach_views()
    {
        if ( g_state_view.bytes == nullptr )
        {
            g_state_view = open_view( L"Local\\MonoCruiseTmpState", 16 );
        }
        if ( g_vehicles_view.bytes == nullptr )
        {
            g_vehicles_view = open_view( L"Local\\MonoCruiseTmpVehicles", kVehiclesSize );
        }
    }

    bool all_zero( const View& view )
    {
        for ( size_t i = 0; view.bytes != nullptr && i < view.size; ++i )
        {
            if ( view.bytes[ i ] != 0 )
            {
                return false;
            }
        }
        return true;
    }

    struct StateRead
    {
        uint32_t version, heartbeat;
        int connected, zone;
    };

    StateRead read_state()
    {
        StateRead s{};
        if ( g_state_view.bytes != nullptr )
        {
            std::memcpy( &s.version, g_state_view.bytes, 4 );
            std::memcpy( &s.heartbeat, g_state_view.bytes + 4, 4 );
            s.connected = g_state_view.bytes[ 8 ];
            s.zone      = g_state_view.bytes[ 9 ];
        }
        return s;
    }

    struct VehiclesRead
    {
        uint32_t version, heartbeat;
        int count, connected;
        bool any_sampled;
    };

    VehiclesRead read_vehicles()
    {
        VehiclesRead v{};
        if ( g_vehicles_view.bytes == nullptr )
        {
            return v;
        }
        const uint8_t* b = g_vehicles_view.bytes;
        uint16_t count   = 0;
        std::memcpy( &v.version, b, 4 );
        std::memcpy( &v.heartbeat, b + 4, 4 );
        std::memcpy( &count, b + 18, 2 );
        v.count     = count;
        v.connected = b[ 20 ];
        for ( int i = 0; i < count && i < 96; ++i )
        {
            v.any_sampled = v.any_sampled || ( b[ 24 + i * 80 + 77 ] & 8 ) != 0;
        }
        return v;
    }

    struct Plugin
    {
        HMODULE dll                                                   = nullptr;
        bool( __cdecl* init )( const TruckersMP_Host*, TruckersMP_PluginDesc* ) = nullptr;
        void( __cdecl* shut )( void )                                 = nullptr;
    } g_plugin;

    bool init_plugin()
    {
        TruckersMP_PluginDesc desc = {};
        desc.m_size                = sizeof desc;
        const bool ok              = g_plugin.init( &g_host, &desc );
        attach_views();
        return ok;
    }

    int count_logs( const char* needle )
    {
        int n = 0;
        for ( const std::string& line : g_logs )
        {
            n += line.find( needle ) != std::string::npos ? 1 : 0;
        }
        return n;
    }

    // After a fault the plugin owes the client silence: no SDK call, no log, and nothing left published.
    void expect_inert()
    {
        const long calls  = g_calls;
        const size_t logs = g_logs.size();
        for ( int i = 0; i < 20; ++i )
        {
            fire_everything();
        }
        CHECK( g_calls == calls );
        CHECK( g_logs.size() == logs );
        CHECK( all_zero( g_state_view ) );
        CHECK( all_zero( g_vehicles_view ) );
        g_plugin.shut();
        CHECK( g_calls == calls );
        CHECK( all_zero( g_state_view ) );
    }

    void warm_up()
    {
        fire_connection( true );
        fire_player( TRUCKERSMP_EVT_PLAYER_UPDATE, 2 );
        fire_player( TRUCKERSMP_EVT_PLAYER_UPDATE, 2 );
        render();
    }

    void s_healthy()
    {
        CHECK( init_plugin() );
        CHECK( g_listeners.size() == 7 );
        warm_up();
        const StateRead s = read_state();
        CHECK( s.version == 1 && s.heartbeat != 0 && s.connected == 1 && s.zone == 0 );
        const VehiclesRead v = read_vehicles();
        CHECK( v.version == 1 && v.heartbeat != 0 && v.count > 0 && v.connected == 1 && v.any_sampled );

        fire_zone( true );
        render();
        CHECK( read_state().zone == 1 );
        fire_zone( false );
        render();
        CHECK( read_state().zone == 0 );

        const int hits        = g_hits[ "Players.GetAllPlayers" ];
        const uint32_t before = read_state().heartbeat;
        for ( int i = 0; i < 300; ++i )
        {
            render();
        }
        CHECK( g_hits[ "Players.GetAllPlayers" ] - hits <= 3 );
        CHECK( read_state().heartbeat - before == 300 );
        Sleep( 80 );
        render();
        CHECK( g_hits[ "Players.GetAllPlayers" ] > hits );

        g_remembered = g_listeners;
        g_plugin.shut();
        CHECK( all_zero( g_state_view ) && all_zero( g_vehicles_view ) );
        CHECK( g_listeners.empty() );
    }

    void zone_only_client( const char* version )
    {
        g_version = version;
        CHECK( init_plugin() );
        warm_up();
        for ( int i = 0; i < 5; ++i )
        {
            Sleep( 60 );
            render();
        }
        CHECK( read_state().heartbeat != 0 && read_state().connected == 1 );
        CHECK( g_hits[ "Players.GetLocalPlayer" ] == 0 );
        CHECK( g_hits[ "Vehicle.GetPlacement" ] == 0 );
        CHECK( all_zero( g_vehicles_view ) );
        CHECK( count_logs( "zone state only" ) == 1 );
        g_plugin.shut();
        CHECK( all_zero( g_state_view ) );
    }

    void s_old_client() { zone_only_client( "0.7.7.10" ); }

    void s_older_major_client() { zone_only_client( "0.6.99.99" ); }

    void s_unparseable_client() { zone_only_client( "banana" ); }

    void s_suffixed_client() { zone_only_client( "0.7.7.11-beta" ); }

    void newer_client( const char* version )
    {
        g_version = version;
        CHECK( init_plugin() );
        warm_up();
        CHECK( read_vehicles().version == 1 && read_vehicles().count > 0 );
        CHECK( count_logs( "zone state only" ) == 0 );
        g_plugin.shut();
        CHECK( all_zero( g_state_view ) && all_zero( g_vehicles_view ) );
    }

    void s_minimum_client() { newer_client( "0.7.7.11" ); }

    void s_newer_client() { newer_client( "v0.8.1.0" ); }

    void s_short_newer_client() { newer_client( "1.0" ); }

    void s_version_unavailable()
    {
        g_acts[ "Core.ClientVersion" ] = Act::Fail;
        CHECK( init_plugin() );
        warm_up();
        CHECK( read_state().heartbeat != 0 );
        CHECK( g_hits[ "Vehicle.GetPlacement" ] == 0 );
        CHECK( count_logs( "version unavailable" ) == 1 );
        g_plugin.shut();
    }

    void s_version_junk()
    {
        g_version_mode = 1;
        CHECK( !init_plugin() );
        CHECK( g_listeners.empty() );
        CHECK( all_zero( g_state_view ) );
        CHECK( count_logs( "faulted" ) == 1 );
    }

    void s_core_missing()
    {
        g_no_core = true;
        CHECK( !init_plugin() );
        CHECK( g_listeners.empty() );
    }

    void s_list_fails()
    {
        g_acts[ "Players.GetAllPlayers" ] = Act::Fail;
        CHECK( init_plugin() );
        warm_up();
        CHECK( read_state().heartbeat != 0 );
        CHECK( read_vehicles().count > 0 );
        g_plugin.shut();
    }

    void s_queries_fail()
    {
        for ( const char* name : { "Vehicle.GetTrailer", "Vehicle.GetPlacement", "Vehicle.GetLinearVelocity", "Vehicle.GetAngularVelocity",
                                   "Trailer.GetPlacement", "PlayerHandle.GetVehicle" } )
        {
            g_acts[ name ] = Act::Fail;
        }
        CHECK( init_plugin() );
        warm_up();
        CHECK( read_state().heartbeat != 0 );
        CHECK( read_vehicles().version == 1 && read_vehicles().count == 0 );
        g_plugin.shut();
    }

    void s_list_junk_huge()
    {
        g_list_mode = 1;
        CHECK( init_plugin() );
        warm_up();
        CHECK( count_logs( "faulted" ) == 1 );
        expect_inert();
    }

    void s_list_null_items()
    {
        g_list_mode = 2;
        CHECK( init_plugin() );
        warm_up();
        CHECK( count_logs( "faulted" ) == 1 );
        expect_inert();
    }

    void s_null_payloads()
    {
        CHECK( init_plugin() );
        warm_up();
        for ( const uint64_t id : { TRUCKERSMP_EVT_PLAYER_STREAM_IN, TRUCKERSMP_EVT_PLAYER_STREAM_OUT, TRUCKERSMP_EVT_PLAYER_UPDATE,
                                    TRUCKERSMP_EVT_GAMEPLAY_NO_COLLISION_ZONE } )
        {
            uint8_t one = 1;
            fire( id, nullptr, 0 );
            fire( id, &one, 1 );
        }
        const uint32_t before = read_state().heartbeat;
        render();
        CHECK( read_state().heartbeat != before );
        CHECK( count_logs( "faulted" ) == 0 );
        g_plugin.shut();
    }

    void s_bad_payload()
    {
        CHECK( init_plugin() );
        warm_up();
        fire( TRUCKERSMP_EVT_PLAYER_UPDATE, reinterpret_cast< void* >( 8 ), 8 );
        CHECK( count_logs( "faulted" ) == 1 );
        expect_inert();
    }

    void s_bad_payload_other_thread()
    {
        CHECK( init_plugin() );
        warm_up();
        HANDLE thread = CreateThread(
            nullptr, 0,
            []( LPVOID ) -> DWORD {
                fire( TRUCKERSMP_EVT_PLAYER_STREAM_OUT, reinterpret_cast< void* >( 8 ), 8 );
                return 0;
            },
            nullptr, 0, nullptr
        );
        WaitForSingleObject( thread, 5000 );
        CHECK( count_logs( "faulted" ) == 1 );
        expect_inert();
    }

    void s_after_shutdown()
    {
        CHECK( init_plugin() );
        warm_up();
        g_remembered = g_listeners;
        g_plugin.shut();
        CHECK( g_listeners.empty() );
        const long calls  = g_calls;
        const size_t logs = g_logs.size();
        for ( int i = 0; i < 20; ++i )
        {
            fire_everything( true );
        }
        CHECK( g_calls == calls );
        CHECK( g_logs.size() == logs );
        CHECK( all_zero( g_state_view ) && all_zero( g_vehicles_view ) );
    }

    void s_reinit()
    {
        CHECK( init_plugin() );
        warm_up();
        g_plugin.shut();
        CHECK( all_zero( g_state_view ) );
        CHECK( init_plugin() );
        CHECK( g_listeners.size() == 7 );
        warm_up();
        CHECK( read_state().heartbeat != 0 && read_state().connected == 1 );
        CHECK( read_vehicles().count > 0 );
        g_plugin.shut();
        CHECK( all_zero( g_state_view ) && all_zero( g_vehicles_view ) );
    }

    void s_unload_without_shutdown()
    {
        CHECK( init_plugin() );
        warm_up();
        CHECK( !all_zero( g_state_view ) );
        g_host_gone = true;
        FreeLibrary( g_plugin.dll );
        g_plugin.dll = nullptr;
        CHECK( all_zero( g_state_view ) && all_zero( g_vehicles_view ) );
    }

    void s_exit_without_shutdown()
    {
        CHECK( init_plugin() );
        warm_up();
        g_host_gone = true;
    }

    void s_slow_host()
    {
        g_players = 200;
        g_burn_us = 300;
        CHECK( init_plugin() );
        fire_connection( true );
        render();
        for ( int i = 0; i < 3; ++i )
        {
            Sleep( 60 );
            const int64_t start = qpc_us();
            render();
            CHECK( qpc_us() - start < 25000 );
        }
        CHECK( read_vehicles().count > 0 && read_vehicles().count < 40 );
        CHECK( count_logs( "time budget" ) == 1 );
        g_plugin.shut();
    }

    void s_update_race()
    {
        CHECK( init_plugin() );
        warm_up();
        static std::atomic< bool > stop{ false };
        HANDLE thread = CreateThread(
            nullptr, 0,
            []( LPVOID ) -> DWORD {
                for ( uint64_t h = 2; !stop; h = 2 + ( h + 1 ) % 40 )
                {
                    fire_player( TRUCKERSMP_EVT_PLAYER_UPDATE, h );
                }
                return 0;
            },
            nullptr, 0, nullptr
        );
        const int64_t end = qpc_us() + 400000;
        for ( int i = 0; qpc_us() < end; ++i )
        {
            render();
            if ( i % 25 == 0 )
            {
                fire_connection( i % 50 == 0 );
            }
            Sleep( 2 );
        }
        stop = true;
        WaitForSingleObject( thread, 5000 );
        CHECK( count_logs( "faulted" ) == 0 );
        CHECK( read_state().heartbeat != 0 );
        g_plugin.shut();
        CHECK( all_zero( g_state_view ) );
    }

    const char* const kFaultApis[] = {
        "Core.ClientVersion",
        "Core.AddListener",
        "Core.LogMessage",
        "Core.RemoveListener",
        "Network.IsConnected",
        "Players.GetLocalPlayer",
        "Players.GetAllPlayers",
        "PlayerHandle.GetVehicle",
        "PlayerHandle.GetDistanceFromLocalPlayer",
        "Vehicle.GetTrailer",
        "Vehicle.GetPlacement",
        "Vehicle.GetLinearVelocity",
        "Vehicle.GetAngularVelocity",
        "Trailer.GetPlacement",
        "Trailer.GetLinearVelocity",
        "Trailer.GetAngularVelocity",
    };

    void s_fault( const std::string& api, Act act )
    {
        g_acts[ api ]    = act;
        const bool shown = api != "Core.LogMessage";
        if ( api == "Core.RemoveListener" )
        {
            CHECK( init_plugin() );
            warm_up();
            g_plugin.shut();
            CHECK( all_zero( g_state_view ) && all_zero( g_vehicles_view ) );
            CHECK( count_logs( "faulted" ) == 1 );
            expect_inert();
            return;
        }
        const bool started = init_plugin();
        if ( api == "Core.ClientVersion" )
        {
            CHECK( !started );
            CHECK( g_listeners.empty() );
            CHECK( all_zero( g_state_view ) );
            CHECK( count_logs( "faulted" ) == 1 );
            return;
        }
        CHECK( started );
        warm_up();
        if ( shown )
        {
            CHECK( count_logs( "faulted" ) == 1 );
        }
        expect_inert();
    }

    std::map< std::string, std::function< void() > > build_scenarios()
    {
        std::map< std::string, std::function< void() > > all = {
            { "healthy", s_healthy },
            { "old_client", s_old_client },
            { "older_major_client", s_older_major_client },
            { "unparseable_client", s_unparseable_client },
            { "suffixed_client", s_suffixed_client },
            { "minimum_client", s_minimum_client },
            { "newer_client", s_newer_client },
            { "short_newer_client", s_short_newer_client },
            { "version_unavailable", s_version_unavailable },
            { "version_junk", s_version_junk },
            { "core_missing", s_core_missing },
            { "list_fails", s_list_fails },
            { "queries_fail", s_queries_fail },
            { "list_junk_huge", s_list_junk_huge },
            { "list_null_items", s_list_null_items },
            { "null_payloads", s_null_payloads },
            { "bad_payload", s_bad_payload },
            { "bad_payload_other_thread", s_bad_payload_other_thread },
            { "after_shutdown", s_after_shutdown },
            { "reinit", s_reinit },
            { "unload_without_shutdown", s_unload_without_shutdown },
            { "exit_without_shutdown", s_exit_without_shutdown },
            { "slow_host", s_slow_host },
            { "update_race", s_update_race },
        };
        for ( const char* api : kFaultApis )
        {
            all[ std::string( "fault." ) + api + ".crash" ] = [ api ] { s_fault( api, Act::Crash ); };
            all[ std::string( "fault." ) + api + ".throw" ] = [ api ] { s_fault( api, Act::Throw ); };
        }
        return all;
    }

    LONG WINAPI unhandled( EXCEPTION_POINTERS* info )
    {
        std::printf( "FAIL: a fault escaped the plugin (0x%08lX)\n", info->ExceptionRecord->ExceptionCode );
        std::fflush( stdout );
        TerminateProcess( GetCurrentProcess(), 99 );
        return EXCEPTION_EXECUTE_HANDLER;
    }

    DWORD WINAPI watchdog( LPVOID )
    {
        Sleep( 20000 );
        die( 98, "the scenario hung" );
    }
}

int main( int argc, char** argv )
{
    const auto scenarios = build_scenarios();
    if ( argc == 2 && std::string( argv[ 1 ] ) == "--list" )
    {
        for ( const auto& entry : scenarios )
        {
            std::printf( "%s\n", entry.first.c_str() );
        }
        return 0;
    }
    if ( argc != 3 || scenarios.find( argv[ 2 ] ) == scenarios.end() )
    {
        std::printf( "usage: fault_host <dll> <scenario> | --list\n" );
        return 2;
    }
    SetUnhandledExceptionFilter( unhandled );
    CreateThread( nullptr, 0, watchdog, nullptr, 0, nullptr );

    g_plugin.dll = LoadLibraryA( argv[ 1 ] );
    if ( g_plugin.dll == nullptr )
    {
        std::printf( "FAIL: cannot load %s\n", argv[ 1 ] );
        return 3;
    }
    if ( !redirect_mappings( g_plugin.dll ) )
    {
        std::printf( "FAIL: cannot redirect the plugin's shared memory away from a running game\n" );
        return 3;
    }
    g_plugin.init = reinterpret_cast< decltype( g_plugin.init ) >( GetProcAddress( g_plugin.dll, "truckersmp_init" ) );
    g_plugin.shut = reinterpret_cast< decltype( g_plugin.shut ) >( GetProcAddress( g_plugin.dll, "truckersmp_shutdown" ) );
    if ( g_plugin.init == nullptr || g_plugin.shut == nullptr )
    {
        std::printf( "FAIL: the DLL does not export the plugin entry points\n" );
        return 3;
    }

    scenarios.at( argv[ 2 ] )();
    g_host_gone = true;
    std::printf( g_failures == 0 ? "PASS %s\n" : "FAILED %s\n", argv[ 2 ] );
    std::fflush( stdout );
    return g_failures == 0 ? 0 : 1;
}
