import shutil
import winreg
import subprocess
import time
import psutil
from pathlib import Path
from CTkMessagebox import CTkMessagebox

# DLLs to manage
SDK_DLLS = [
    "scs-telemetry.dll",
    "input_semantical.dll",
    "ets2_la_plugin.dll"
]

def is_steam_installed():
    """
    Checks if Steam is installed by looking for its registry key.
    Returns True if installed, False otherwise.
    """
    try:
        key = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"SOFTWARE\WOW6432Node\Valve\Steam")
        winreg.CloseKey(key)
        return True
    except FileNotFoundError:
        pass

    try:
        key = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"SOFTWARE\Valve\Steam")
        winreg.CloseKey(key)
        return True
    except FileNotFoundError:
        pass

    return False

def find_scs_game_path(game_type="ets2"):
    if not is_steam_installed():
        return "steam_not_installed"
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
    try:
        key = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, 
                           rf"SOFTWARE\Microsoft\Windows\CurrentVersion\Uninstall\Steam App {config['steam_id']}")
        path = winreg.QueryValueEx(key, "InstallLocation")[0]
        winreg.CloseKey(key)
        if Path(path).exists():
            found_paths.append(Path(path))
    except:
        pass
    steam_paths = []
    try:
        key = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, 
                           r"SOFTWARE\WOW6432Node\Valve\Steam")
        steam_path = winreg.QueryValueEx(key, "InstallPath")[0]
        winreg.CloseKey(key)
        steam_paths.append(Path(steam_path))
    except:
        pass
    steam_paths.extend([
        Path("C:/Program Files (x86)/Steam"),
        Path("C:/Program Files/Steam"),
    ])
    for steam_path in steam_paths:
        if steam_path.exists():
            library_vdf = steam_path / "steamapps" / "libraryfolders.vdf"
            if library_vdf.exists():
                try:
                    with open(library_vdf, 'r', encoding='utf-8') as f:
                        content = f.read()
                        import re
                        paths = re.findall(r'"path"\s*"([^"]+)"', content)
                        for lib_path in paths:
                            lib_path = lib_path.replace('\\\\', '/')
                            game_path = Path(lib_path) / "steamapps" / "common" / config['folder_name']
                            if game_path.exists():
                                found_paths.append(game_path)
                except:
                    pass
            default_game = steam_path / "steamapps" / "common" / config['folder_name']
            if default_game.exists():
                found_paths.append(default_game)
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
    unique_paths = []
    for path in found_paths:
        if path not in unique_paths and validate_scs_game_installation(path, game_type):
            unique_paths.append(path)
    return unique_paths[0] if unique_paths else None

def validate_scs_game_installation(game_path, game_type="ets2"):
    if not game_path or game_path == "steam_not_installed":
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
    time.sleep(1)

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

def get_sdk_dll_paths(game_path):
    dll_paths = {}
    for dll_name in SDK_DLLS:
        dll_paths[dll_name] = game_path / "bin" / "win_x64" / "plugins" / dll_name
    return dll_paths

def has_sdk_dlls(game_path):
    if game_path == "steam_not_installed":
        return False
    dll_paths = get_sdk_dll_paths(game_path)
    return all(path.exists() for path in dll_paths.values())

def find_all_scs_game_installations(game_type="ets2"):
    if not is_steam_installed():
        return "steam_not_installed"
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
    try:
        key = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, 
                           rf"SOFTWARE\Microsoft\Windows\CurrentVersion\Uninstall\Steam App {config['steam_id']}")
        path = winreg.QueryValueEx(key, "InstallLocation")[0]
        winreg.CloseKey(key)
        if Path(path).exists():
            found_paths.append(Path(path))
    except:
        pass
    steam_paths = []
    try:
        key = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, 
                           r"SOFTWARE\WOW6432Node\Valve\Steam")
        steam_path = winreg.QueryValueEx(key, "InstallPath")[0]
        winreg.CloseKey(key)
        steam_paths.append(Path(steam_path))
    except:
        pass
    steam_paths.extend([
        Path("C:/Program Files (x86)/Steam"),
        Path("C:/Program Files/Steam"),
    ])
    for steam_path in steam_paths:
        if steam_path.exists():
            library_vdf = steam_path / "steamapps" / "libraryfolders.vdf"
            if library_vdf.exists():
                try:
                    with open(library_vdf, 'r', encoding='utf-8') as f:
                        content = f.read()
                        import re
                        paths = re.findall(r'"path"\s*"([^"]+)"', content)
                        for lib_path in paths:
                            lib_path = lib_path.replace('\\\\', '/')
                            game_path = Path(lib_path) / "steamapps" / "common" / config['folder_name']
                            if game_path.exists():
                                found_paths.append(game_path)
                except:
                    pass
            default_game = steam_path / "steamapps" / "common" / config['folder_name']
            if default_game.exists():
                found_paths.append(default_game)
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
    unique_paths = []
    for path in found_paths:
        if path not in unique_paths and validate_scs_game_installation(path, game_type):
            unique_paths.append(path)
    return unique_paths

# Main functions
def check_scs_sdk(game_type="ets2"):
    """
    Check if all SDK DLLs are installed and game installation is valid.
    Returns True if all DLLs are installed and game is valid, False otherwise.
    """
    try:
        game_path = find_scs_game_path(game_type)
        if not game_path:
            return False
        if not validate_scs_game_installation(game_path, game_type):
            return False
        return has_sdk_dlls(game_path)
    except Exception:
        return False

def check_all_scs_sdk_installations(game_type="ets2"):
    """
    Check SDK status for all SCS game installations found on the system.
    Returns list of tuples: (game_path, sdk_installed)
    """
    installations = []
    all_paths = find_all_scs_game_installations(game_type)
    for game_path in all_paths:
        sdk_installed = has_sdk_dlls(game_path)
        installations.append((game_path, sdk_installed))
    return installations

def install_scs_sdk(game_type="ets2", target_path=None):
    """
    Install all required SDK DLLs if not already installed.
    Args:
        game_type: "ets2" or "ats"
        target_path: Optional specific game installation path to install to
    Returns (success, was_already_installed) tuple.
    - success: True if SDK is ready to use, False if failed
    - was_already_installed: True if SDK was already there, False if freshly installed
    """
    try:
        if target_path:
            game_path = Path(target_path)
        else:
            game_path = find_scs_game_path(game_type)
        if not game_path:
            print(f"Error: {game_type.upper()} installation not found")
            return False, False
        if not validate_scs_game_installation(game_path, game_type):
            exe_names = {"ets2": "eurotrucks2.exe", "ats": "amtrucks.exe"}
            print(f"Error: {exe_names[game_type]} not found in {game_path / 'bin' / 'win_x64'}")
            print(f"This may not be a valid {game_type.upper()} installation directory.")
            return False, False

        dll_paths = get_sdk_dll_paths(game_path)
        dlls_installed = {name: path.exists() for name, path in dll_paths.items()}
        was_already_installed = all(dlls_installed.values())
        if was_already_installed:
            print(f"SDK DLLs already installed at: {dll_paths}")
            return True, True

        missing_dlls = [name for name, installed in dlls_installed.items() if not installed]
        msg = CTkMessagebox(
            master=root,
            title="SDK not installed",
            message=f'Should I automatically install the SDK DLLs ({", ".join(missing_dlls)}) for {game_type.upper()}? The game will close automatically if running.',
            icon="warning",
            option_1="Cancel",
            option_2="Install",
            wraplength=400,
            sound=True
        )

        if msg.get() == "Install":
            print("Installing SDK DLLs...")
            game_was_running = is_scs_game_running(game_type)
            if game_was_running:
                print(f"Closing {game_type.upper()}...")
                close_scs_game(game_type)

            script_dir = Path(__file__).parent
            success = True
            for dll_name, dll_target_path in dll_paths.items():
                source_dll = script_dir / dll_name
                if not source_dll.exists():
                    print(f"Error: {dll_name} not found in script directory")
                    success = False
                    continue
                plugins_dir = dll_target_path.parent
                plugins_dir.mkdir(parents=True, exist_ok=True)
                try:
                    shutil.copy2(source_dll, dll_target_path)
                    print(f"Installed {dll_name} to: {dll_target_path}")
                except Exception as e:
                    print(f"Error copying {dll_name}: {e}")
                    success = False

            """ # temporarily disable because it's anoying...
            if game_was_running:
                print(f"Relaunching {game_type.upper()}...")
                launch_scs_game(game_path, game_type)
            """

            if success:
                print(f"SDK DLLs installed successfully to: {dll_paths}")
            else:
                print("Some DLLs failed to install.")

            return success, False
        else:
            print("SDK installation cancelled by user.")
            return False, False
    except Exception as e:
        print(f"Error during SDK installation: {e}")
        return False, False

def install_scs_sdk_to_all(game_type="ets2"):
    """
    Install all required SDK DLLs to all found game installations.
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

def check_and_install_scs_sdk(root):
    """
    Check if all SDK DLLs are installed in any SCS game (ETS2 or ATS) and install to all found games if needed.
    Only installs if valid game installations (with .exe files) are found.
    Returns:
        dict: {
            "found_games": ["ets2", "ats"],
            "sdk_already_installed": bool,
            "installation_results": {
                "ets2": [...],
                "ats": [...]
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
    games_to_check = ["ets2", "ats"]
    for game_type in games_to_check:
        installations = find_all_scs_game_installations(game_type)
        if installations:
            result["found_games"].append(game_type)
            print(f"Found {len(installations)} {game_type.upper()} installation(s)")
            for game_path in installations:
                if has_sdk_dlls(game_path):
                    result["sdk_already_installed"] = True
                    print(f"SDK DLLs already installed in {game_type.upper()} at: {game_path}")

    if not result["found_games"]:
        print("No SCS games (ETS2 or ATS) found on this system.")
        print("Cannot install SDK DLLs - no valid game installations detected.")
        return result

    print(f"\nFound games: {', '.join([g.upper() for g in result['found_games']])}")
    if result["sdk_already_installed"]:
        print("SDK DLLs are already installed in at least one game.")
    else:
        print("No SDK DLLs found in any SCS game.")
    print("Installing SDK DLLs to all found SCS game installations...")

    for game_type in result["found_games"]:
        print(f"\n=== Processing {game_type.upper()} installations ===")
        game_results = install_scs_sdk_to_all(game_type)
        result["installation_results"][game_type] = game_results
        for path, success, was_installed in game_results:
            result["summary"]["total_installations"] += 1
            if success:
                if was_installed:
                    result["summary"]["already_had_sdk"] += 1
                else:
                    result["summary"]["successful_installs"] += 1
            else:
                result["summary"]["failed_installs"] += 1

    print("\n" + "="*50)
    print("INSTALLATION SUMMARY")
    print("="*50)
    summary = result["summary"]
    print(f"Total game installations found: {summary['total_installations']}")
    print(f"Already had SDK DLLs: {summary['already_had_sdk']}")
    print(f"Successfully installed SDK DLLs: {summary['successful_installs']}")
    print(f"Failed installations: {summary['failed_installs']}")
    if summary["failed_installs"] > 0:
        print(f"\nFailed installations:")
        for game_type, results in result["installation_results"].items():
            for path, success, was_installed in results:
                if not success:
                    print(f"  - {game_type.upper()}: {path}")

    if summary["successful_installs"] > 0 or summary["already_had_sdk"] > 0:
        print(f"\nSDK DLLs are now ready for use in {summary['successful_installs'] + summary['already_had_sdk']} installation(s)!")
        msg = CTkMessagebox(
            master=root,
            title="SDK installed",
            message='SDK DLLs installed successfully. You can now open the game.',
            icon="warning",
            option_1="Okay",
            wraplength=300,
            sound=True
        )
        msg.get()
    if summary["failed_installs"] > 0:
        print("Some installations failed. Please check the logs for details.")
        msg = CTkMessagebox(
            master=root,
            title="SDK installation failed",
            message='Some installations failed. Please check the logs for details.',
            icon="warning",
            option_1="Okay",
            wraplength=300,
            sound=True
        )
        msg.get()
    return result

def update_sdk_dlls(root):
    """
    Update all installed SDK DLLs (scs-telemetry.dll, input_semantical.dll, ets2_la_plugin.dll)
    to all locations where ALL are already installed.
    If the game is running, ask the user for confirmation before closing and updating.
    Skips locations where not all DLLs are present.
    Only overwrites if source DLL exists.
    Returns a list of (game_path, updated_dlls, skipped_dlls, errors)
    Returns 'steam_not_installed' if Steam is missing.
    """
    if not is_steam_installed():
        print("Steam is not installed on this system.")
        return "steam_not_installed"

    script_dir = Path(__file__).parent
    results = []
    for game_type in ["ets2", "ats"]:
        all_paths = find_all_scs_game_installations(game_type)
        if all_paths == "steam_not_installed":
            return "steam_not_installed"
        for game_path in all_paths:
            dll_paths = get_sdk_dll_paths(game_path)
            if all(path.exists() for path in dll_paths.values()):
                updated_dlls = []
                skipped_dlls = []
                errors = []

                # Check if the game is running
                running = is_scs_game_running(game_type)
                if running:
                    msg = CTkMessagebox(
                        master=root,
                        title=f"{game_type.upper()} is running",
                        message=f"{game_type.upper()} is currently running at {game_path}.\nDo you want to close it automatically to update the SDK DLLs?",
                        icon="warning",
                        option_1="Cancel",
                        option_2="Close and Update",
                        wraplength=400,
                        sound=True
                    )
                    user_choice = msg.get()
                    if user_choice == "Close and Update":
                        close_scs_game(game_type)
                        print(f"Closed {game_type.upper()} for update at {game_path}")
                    else:
                        print(f"Skipped updating {game_path} for {game_type.upper()} because the user cancelled.")
                        results.append((game_path, [], SDK_DLLS, [("User cancelled update because game was running.", "")]))
                        continue

                for dll_name, target_path in dll_paths.items():
                    source_dll = script_dir / dll_name
                    if source_dll.exists():
                        try:
                            target_file = target_path / dll_name
                            if target_file.exists():
                                target_file.unlink()
                            shutil.copy2(source_dll, target_path)
                            updated_dlls.append(dll_name)
                            print(f"Updated {dll_name} in {target_path}")
                        except Exception as e:
                            errors.append((dll_name, str(e)))
                            print(f"Failed to update {dll_name} in {target_path}: {e}")
                    else:
                        skipped_dlls.append(dll_name)
                        print(f"Source DLL {dll_name} not found in script directory, skipped updating.")
                results.append((game_path, updated_dlls, skipped_dlls, errors))
            else:
                print(f"Skipped {game_path} for {game_type.upper()} - not all SDK DLLs are installed.")
    return results