import shutil
import winreg
import subprocess
import time
import psutil
from pathlib import Path

def setup_ets2_sdk():
    def is_game_running():
        """Check if ETS2 is running"""
        for proc in psutil.process_iter(['name']):
            if proc.info['name'] and 'eurotrucks2.exe' in proc.info['name'].lower():
                return True
        return False
    
    def close_game():
        """Close ETS2 if running"""
        for proc in psutil.process_iter(['name', 'pid']):
            if proc.info['name'] and 'eurotrucks2.exe' in proc.info['name'].lower():
                try:
                    proc.terminate()
                    proc.wait(timeout=10)
                except:
                    proc.kill()
        time.sleep(1)  # Wait for complete shutdown
    
    def launch_game(ets2_path):
        """Launch ETS2"""
        exe_path = ets2_path / "bin" / "win_x64" / "eurotrucks2.exe"
        if exe_path.exists():
            subprocess.Popen([str(exe_path)], cwd=str(exe_path.parent))
    """
    Check if ETS2 SDK is installed and install it if not.
    Returns True if SDK is ready, False if failed.
    """
    
    def find_ets2_path():
        """Find ETS2 installation directory"""
        try:
            # Try Steam registry
            key = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, 
                               r"SOFTWARE\Microsoft\Windows\CurrentVersion\Uninstall\Steam App 227300")
            path = winreg.QueryValueEx(key, "InstallLocation")[0]
            winreg.CloseKey(key)
            return Path(path)
        except:
            # Try common locations
            paths = [
                Path("C:/Program Files (x86)/Steam/steamapps/common/Euro Truck Simulator 2"),
                Path("C:/Program Files/Steam/steamapps/common/Euro Truck Simulator 2"),
            ]
            for p in paths:
                if p.exists():
                    return p
            return None
    
    try:
        # Find ETS2 installation
        ets2_path = find_ets2_path()
        if not ets2_path:
            return False, False
        
        # Check if SDK is already installed
        plugins_dir = ets2_path / "bin" / "win_x64" / "plugins"
        sdk_dll_1 = plugins_dir / "scs-telemetry.dll"
        sdk_dll_2 = plugins_dir / "TruckyTelemetry.dll"

        
        if sdk_dll_1.exists() and sdk_dll_2.exists():
            return True, True
        
        # Check if game is running and close it if needed
        game_was_running = is_game_running()
        if game_was_running:
            close_game()
        
        # Install SDK
        script_dir = Path(__file__).parent
        source_dll_1 = script_dir / "scs-telemetry.dll"
        source_dll_2 = script_dir / "TruckyTelemetry.dll"
        
        if not source_dll_1.exists():
            return False, False
        if not source_dll_2.exists():
            return False, False
        
        # Create plugins directory if needed
        plugins_dir.mkdir(parents=True, exist_ok=True)
        
        # Copy DLL
        shutil.copy2(source_dll_1, sdk_dll_1)
        shutil.copy2(source_dll_2, sdk_dll_2)
        
        # Launch game to create shared memory folder
        launch_game(ets2_path)
        
        return True, False
        
    except:
        return False, False

# Usage
if __name__ == "__main__":
    if setup_ets2_sdk():
        print("SDK ready")
    else:
        print("SDK setup failed")