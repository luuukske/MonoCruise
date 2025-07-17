import shutil
import winreg
import subprocess
import time
import psutil
from pathlib import Path
from CTkMessagebox import CTkMessagebox
# Common utility functions
def find_scs_game_path(game_type="ets2"):
    """Find SCS game installation directory"""
    game_configs = {
        "ets2": {
            "steam_id": "227300",
            "folder_name": "Euro Truck Simulator 2",
            "exe_name": "eurotrucks2.exe"
        },
        "ats": {
            "steam_id": "270880", 
            "folder_name": "American Truck Simulator",
            "exe_name": "amtrucks.exe"
        }
    }
    
    if game_type not in game_configs:
        return None
    
    config = game_configs[game_type]
    found_paths = []
    
    # Try Steam registry (main installation)
    try:
        key = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, 
                           rf"SOFTWARE\Microsoft\Windows\CurrentVersion\Uninstall\Steam App {config['steam_id']}")
        path = winreg.QueryValueEx(key, "InstallLocation")[0]
        winreg.CloseKey(key)
        if Path(path).exists():
            found_paths.append(Path(path))
    except:
        pass
    
    # Try to find Steam library folders
    steam_paths = []
    try:
        # Try Steam registry for Steam location
        key = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, 
                           r"SOFTWARE\WOW6432Node\Valve\Steam")
        steam_path = winreg.QueryValueEx(key, "InstallPath")[0]
        winreg.CloseKey(key)
        steam_paths.append(Path(steam_path))
    except:
        pass
    
    # Add common Steam locations
    steam_paths.extend([
        Path("C:/Program Files (x86)/Steam"),
        Path("C:/Program Files/Steam"),
    ])
    
    # Check for multiple Steam library folders
    for steam_path in steam_paths:
        if steam_path.exists():
            # Check libraryfolders.vdf for additional library locations
            library_vdf = steam_path / "steamapps" / "libraryfolders.vdf"
            if library_vdf.exists():
                try:
                    with open(library_vdf, 'r', encoding='utf-8') as f:
                        content = f.read()
                        import re
                        # Find all library paths
                        paths = re.findall(r'"path"\s*"([^"]+)"', content)
                        for lib_path in paths:
                            lib_path = lib_path.replace('\\\\', '/')
                            game_path = Path(lib_path) / "steamapps" / "common" / config['folder_name']
                            if game_path.exists():
                                found_paths.append(game_path)
                except:
                    pass
            
            # Also check the default location in this Steam installation
            default_game = steam_path / "steamapps" / "common" / config['folder_name']
            if default_game.exists():
                found_paths.append(default_game)
    
    # Check all drives for common Steam locations
    import string
    for drive in string.ascii_uppercase:
        drive_paths = [
            Path(f"{drive}:/Program Files (x86)/Steam/steamapps/common/{config['folder_name']}"),
            Path(f"{drive}:/Program Files/Steam/steamapps/common/{config['folder_name']}"),
            Path(f"{drive}:/Steam/steamapps/common/{config['folder_name']}"),
            Path(f"{drive}:/Games/Steam/steamapps/common/{config['folder_name']}"),
            Path(f"{drive}:/SteamLibrary/steamapps/common/{config['folder_name']}"),
        ]
        for p in drive_paths:
            if p.exists():
                found_paths.append(p)
    
    # Remove duplicates and validate installations
    unique_paths = []
    for path in found_paths:
        if path not in unique_paths and validate_scs_game_installation(path, game_type):
            unique_paths.append(path)
    
    # Return the first valid path, or None if none found
    return unique_paths[0] if unique_paths else None

def validate_scs_game_installation(game_path, game_type="ets2"):
    """Validate that the game exe exists in the expected location"""
    if not game_path:
        return False
    
    exe_names = {
        "ets2": "eurotrucks2.exe",
        "ats": "amtrucks.exe"
    }
    
    if game_type not in exe_names:
        return False
    
    exe_path = game_path / "bin" / "win_x64" / exe_names[game_type]
    return exe_path.exists()

def is_scs_game_running(game_type="ets2"):
    """Check if SCS game is running"""
    exe_names = {
        "ets2": "eurotrucks2.exe",
        "ats": "amtrucks.exe"
    }
    
    if game_type not in exe_names:
        return False
    
    target_exe = exe_names[game_type]
    for proc in psutil.process_iter(['name']):
        if proc.info['name'] and target_exe in proc.info['name'].lower():
            return True
    return False

def close_scs_game(game_type="ets2"):
    """Close SCS game if running"""
    exe_names = {
        "ets2": "eurotrucks2.exe",
        "ats": "amtrucks.exe"
    }
    
    if game_type not in exe_names:
        return
    
    target_exe = exe_names[game_type]
    for proc in psutil.process_iter(['name', 'pid']):
        if proc.info['name'] and target_exe in proc.info['name'].lower():
            try:
                proc.terminate()
                proc.wait(timeout=10)
            except:
                proc.kill()
    time.sleep(1)  # Wait for complete shutdown

def launch_scs_game(game_path, game_type="ets2"):
    """Launch SCS game"""
    exe_names = {
        "ets2": "eurotrucks2.exe",
        "ats": "amtrucks.exe"
    }
    
    if game_type not in exe_names:
        return
    
    exe_path = game_path / "bin" / "win_x64" / exe_names[game_type]
    if exe_path.exists():
        subprocess.Popen([str(exe_path)], cwd=str(exe_path.parent))

def get_sdk_dll_path(game_path):
    """Get the path to the SDK DLL"""
    return game_path / "bin" / "win_x64" / "plugins" / "scs-telemetry.dll"

def find_all_scs_game_installations(game_type="ets2"):
    """Find all SCS game installations on the system"""
    game_configs = {
        "ets2": {
            "steam_id": "227300",
            "folder_name": "Euro Truck Simulator 2",
            "exe_name": "eurotrucks2.exe"
        },
        "ats": {
            "steam_id": "270880", 
            "folder_name": "American Truck Simulator",
            "exe_name": "amtrucks.exe"
        }
    }
    
    if game_type not in game_configs:
        return []
    
    config = game_configs[game_type]
    found_paths = []
    
    # Try Steam registry (main installation)
    try:
        key = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, 
                           rf"SOFTWARE\Microsoft\Windows\CurrentVersion\Uninstall\Steam App {config['steam_id']}")
        path = winreg.QueryValueEx(key, "InstallLocation")[0]
        winreg.CloseKey(key)
        if Path(path).exists():
            found_paths.append(Path(path))
    except:
        pass
    
    # Try to find Steam library folders
    steam_paths = []
    try:
        # Try Steam registry for Steam location
        key = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, 
                           r"SOFTWARE\WOW6432Node\Valve\Steam")
        steam_path = winreg.QueryValueEx(key, "InstallPath")[0]
        winreg.CloseKey(key)
        steam_paths.append(Path(steam_path))
    except:
        pass
    
    # Add common Steam locations
    steam_paths.extend([
        Path("C:/Program Files (x86)/Steam"),
        Path("C:/Program Files/Steam"),
    ])
    
    # Check for multiple Steam library folders
    for steam_path in steam_paths:
        if steam_path.exists():
            # Check libraryfolders.vdf for additional library locations
            library_vdf = steam_path / "steamapps" / "libraryfolders.vdf"
            if library_vdf.exists():
                try:
                    with open(library_vdf, 'r', encoding='utf-8') as f:
                        content = f.read()
                        import re
                        # Find all library paths
                        paths = re.findall(r'"path"\s*"([^"]+)"', content)
                        for lib_path in paths:
                            lib_path = lib_path.replace('\\\\', '/')
                            game_path = Path(lib_path) / "steamapps" / "common" / config['folder_name']
                            if game_path.exists():
                                found_paths.append(game_path)
                except:
                    pass
            
            # Also check the default location in this Steam installation
            default_game = steam_path / "steamapps" / "common" / config['folder_name']
            if default_game.exists():
                found_paths.append(default_game)
    
    # Check all drives for common Steam locations
    import string
    for drive in string.ascii_uppercase:
        drive_paths = [
            Path(f"{drive}:/Program Files (x86)/Steam/steamapps/common/{config['folder_name']}"),
            Path(f"{drive}:/Program Files/Steam/steamapps/common/{config['folder_name']}"),
            Path(f"{drive}:/Steam/steamapps/common/{config['folder_name']}"),
            Path(f"{drive}:/Games/Steam/steamapps/common/{config['folder_name']}"),
            Path(f"{drive}:/SteamLibrary/steamapps/common/{config['folder_name']}"),
        ]
        for p in drive_paths:
            if p.exists():
                found_paths.append(p)
    
    # Remove duplicates and validate installations
    unique_paths = []
    for path in found_paths:
        if path not in unique_paths and validate_scs_game_installation(path, game_type):
            unique_paths.append(path)
    
    return unique_paths

# Main functions
def check_scs_sdk(game_type="ets2"):
    """
    Check if SCS SDK is installed and game installation is valid.
    Returns True if SDK is installed and game is valid, False otherwise.
    """
    try:
        # Find game installation
        game_path = find_scs_game_path(game_type)
        if not game_path:
            return False
        
        # Validate game installation
        if not validate_scs_game_installation(game_path, game_type):
            return False
        
        # Check if SDK is already installed
        sdk_dll = get_sdk_dll_path(game_path)
        return sdk_dll.exists()
        
    except Exception as e:
        return False

def check_all_scs_sdk_installations(game_type="ets2"):
    """
    Check SDK status for all SCS game installations found on the system.
    Returns list of tuples: (game_path, sdk_installed)
    """
    installations = []
    all_paths = find_all_scs_game_installations(game_type)
    
    for game_path in all_paths:
        sdk_dll = get_sdk_dll_path(game_path)
        sdk_installed = sdk_dll.exists()
        installations.append((game_path, sdk_installed))
    
    return installations

def install_scs_sdk(game_type="ets2", target_path=None):
    """
    Install SCS SDK if not already installed.
    Args:
        game_type: "ets2" or "ats"
        target_path: Optional specific game installation path to install to
    Returns (success, was_already_installed) tuple.
    - success: True if SDK is ready to use, False if failed
    - was_already_installed: True if SDK was already there, False if freshly installed
    """
    try:
        # Use specified path or find automatically
        if target_path:
            game_path = Path(target_path)
        else:
            game_path = find_scs_game_path(game_type)
            
        if not game_path:
            print(f"Error: {game_type.upper()} installation not found")
            return False, False
        
        # Validate game installation
        if not validate_scs_game_installation(game_path, game_type):
            exe_names = {"ets2": "eurotrucks2.exe", "ats": "amtrucks.exe"}
            print(f"Error: {exe_names[game_type]} not found in {game_path / 'bin' / 'win_x64'}")
            print(f"This may not be a valid {game_type.upper()} installation directory.")
            return False, False
        
        # Check if SDK is already installed
        sdk_dll = get_sdk_dll_path(game_path)
        
        if sdk_dll.exists():
            print(f"SDK already installed at: {sdk_dll}")
            return True, True

        msg = CTkMessagebox(title="SDK not isntalled", message=f'Should i automatically install the SDK for {game_type}? ETS2 will close automatically.',
                    icon="warning", option_1="Cancel", option_2="Install", wraplength=400, sound=True)

        if msg.get()=="Install":
            print("Installing SDK...")
            # Check if game is running and close it if needed
            game_was_running = is_scs_game_running(game_type)
            if game_was_running:
                print(f"Closing {game_type.upper()}...")
                close_scs_game(game_type)
            
            # Install SDK
            script_dir = Path(__file__).parent
            source_dll = script_dir / "scs-telemetry.dll"
            
            if not source_dll.exists():
                print("Error: scs-telemetry.dll not found in script directory")
                return False, False
            
            # Create plugins directory if needed
            plugins_dir = sdk_dll.parent
            plugins_dir.mkdir(parents=True, exist_ok=True)
            
            # Copy DLL
            shutil.copy2(source_dll, sdk_dll)
            print(f"SDK installed successfully to: {sdk_dll}")
            
            # Launch game to create shared memory folder
            if game_was_running:
                print(f"Relaunching {game_type.upper()}...")
                launch_scs_game(game_path, game_type)
            
            return True, False
        else:
            print("SDK installation cancelled by user.")
            return False, False
        
    except Exception as e:
        print(f"Error during SDK installation: {e}")
        return False, False

def install_scs_sdk_to_all(game_type="ets2"):
    """
    Install SCS SDK to all found game installations.
    Returns list of tuples: (game_path, success, was_already_installed)
    """
    results = []
    all_paths = find_all_scs_game_installations(game_type)
    
    if not all_paths:
        print(f"No {game_type.upper()} installations found")
        return results
    
    print(f"Found {len(all_paths)} {game_type.upper()} installation(s)")
    
    for game_path in all_paths:
        print(f"\nProcessing {game_type.upper()} installation at: {game_path}")
        success, was_installed = install_scs_sdk(game_type, game_path)
        results.append((game_path, success, was_installed))
    
    return results

def install_scs_sdk_to_both_games():
    """
    Install SCS SDK to all ETS2 and ATS installations found on the system.
    Returns dict with results for both games: {"ets2": [...], "ats": [...]}
    """
    results = {}
    
    print("=== Installing SDK to ETS2 ===")
    results["ets2"] = install_scs_sdk_to_all("ets2")
    
    print("\n=== Installing SDK to ATS ===")
    results["ats"] = install_scs_sdk_to_all("ats")
    
    return results

def check_and_install_scs_sdk():
    """
    Check if SCS SDK is installed in any SCS game (ETS2 or ATS) and install to all found games if needed.
    Only installs if valid game installations (with .exe files) are found.
    
    Returns:
        dict: {
            "found_games": ["ets2", "ats"],  # List of games found on system
            "sdk_already_installed": bool,   # True if SDK was found in any game
            "installation_results": {        # Results of installation attempts
                "ets2": [(path, success, was_installed), ...],
                "ats": [(path, success, was_installed), ...]
            },
            "summary": {
                "total_installations": int,
                "successful_installs": int,
                "already_had_sdk": int,
                "failed_installs": int
            }
        }
    """
    result = {
        "found_games": [],
        "sdk_already_installed": False,
        "installation_results": {"ets2": [], "ats": []},
        "summary": {
            "total_installations": 0,
            "successful_installs": 0,
            "already_had_sdk": 0,
            "failed_installs": 0
        }
    }
    
    print("Scanning for SCS games (ETS2 and ATS)...")
    
    # Check for ETS2 and ATS installations
    games_to_check = ["ets2", "ats"]
    
    for game_type in games_to_check:
        installations = find_all_scs_game_installations(game_type)
        if installations:
            result["found_games"].append(game_type)
            print(f"Found {len(installations)} {game_type.upper()} installation(s)")
            
            # Check if SDK is already installed in any installation of this game
            for game_path in installations:
                sdk_dll = get_sdk_dll_path(game_path)
                if sdk_dll.exists():
                    result["sdk_already_installed"] = True
                    print(f"SDK already installed in {game_type.upper()} at: {game_path}")
    
    # If no games found, return early
    if not result["found_games"]:
        print("No SCS games (ETS2 or ATS) found on this system.")
        print("Cannot install SDK - no valid game installations detected.")
        return result

    print(f"\nFound games: {', '.join([g.upper() for g in result['found_games']])}")
    
    if result["sdk_already_installed"]:
        print("SDK is already installed in at least one game.")
        print("Installing SDK to all found SCS game installations...")
    else:
        print("No SDK found in any SCS game.")
        print("Installing SDK to all found SCS game installations...")
    
    # Install SDK to all found games
    for game_type in result["found_games"]:
        print(f"\n=== Processing {game_type.upper()} installations ===")
        game_results = install_scs_sdk_to_all(game_type)
        result["installation_results"][game_type] = game_results
        
        # Update summary statistics
        for path, success, was_installed in game_results:
            result["summary"]["total_installations"] += 1
            if success:
                if was_installed:
                    result["summary"]["already_had_sdk"] += 1
                else:
                    result["summary"]["successful_installs"] += 1
            else:
                result["summary"]["failed_installs"] += 1
    
    # Print summary
    print("\n" + "="*50)
    print("INSTALLATION SUMMARY")
    print("="*50)
    summary = result["summary"]
    print(f"Total game installations found: {summary['total_installations']}")
    print(f"Already had SDK: {summary['already_had_sdk']}")
    print(f"Successfully installed SDK: {summary['successful_installs']}")
    print(f"Failed installations: {summary['failed_installs']}")
    
    if summary["failed_installs"] > 0:
        print(f"\nFailed installations:")
        for game_type, results in result["installation_results"].items():
            for path, success, was_installed in results:
                if not success:
                    print(f"  - {game_type.upper()}: {path}")
    
    if summary["successful_installs"] > 0 or summary["already_had_sdk"] > 0:
        print(f"\nSDK is now ready for use in {summary['successful_installs'] + summary['already_had_sdk']} installation(s)!")
    
    return result

# Legacy function names for backward compatibility
def find_ets2_path():
    return find_scs_game_path("ets2")

def validate_ets2_installation(ets2_path):
    return validate_scs_game_installation(ets2_path, "ets2")

def is_game_running():
    return is_scs_game_running("ets2")

def close_game():
    return close_scs_game("ets2")

def launch_game(ets2_path):
    return launch_scs_game(ets2_path, "ets2")

def check_ets2_sdk():
    return check_scs_sdk("ets2")

def check_ats_sdk():
    return check_scs_sdk("ats")

def install_ets2_sdk(target_path=None):
    return install_scs_sdk("ets2", target_path)