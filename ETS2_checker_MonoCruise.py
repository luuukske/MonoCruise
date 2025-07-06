import truck_telemetry
import time
import json
import subprocess
import sys

CHECK_INTERVAL = 0.5  # seconds

i=0
launched_connected = True

def is_process_running(process_name):
    try:
        import psutil
        i=0
        # Iterate through all running processes
        for proc in psutil.process_iter(['name']):
            try:
                if proc.info['name'] == process_name:
                    i+=1
            except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                continue

        if process_name == "ETS2_checker_MonoCruise.exe":
            if i>=2:
                return True
            else: 
                return False
        elif i>=1:
            return True
        return False
    except ImportError:
        # If psutil is not available, return False
        return False

def start_program():
    try:
        #os.open(r"./MonoCruise.exe")
        if not is_process_running("MonoCruise.exe"):
            print("running MonoCruise")
            subprocess.Popen(["MonoCruise.exe"])
        
    except:
        raise Exception("MonoCruise could not be started")
            
def is_ETS2_running():
    global data
    global launched_connected
    try:
        truck_telemetry.init()
        data = truck_telemetry.get_data()
    except Exception as e:
        if isinstance(e, FileNotFoundError) or str(e) == "SDK_NOT_ACTIVE":
            launched_connected = False
            return False
    else:
        if data["sdkActive"]:  # Check if SDK is still active
            return True
    launched_connected = False
    return False

def main():
    global i
    global launched_connected
    print(f"Monitoring for ETS2 SDK...")
    if is_process_running("ETS2_checker_MonoCruise.exe"):
        sys.exit()
    while True:
        if is_ETS2_running():
            print("ets2 detected.")
            if not launched_connected:
                start_program()
            while is_ETS2_running():
                time.sleep(CHECK_INTERVAL)  # Small sleep to prevent CPU hogging
            i+=1
            if i >= 15:
                with open('saves.json', "w") as file:
                    json.dump({"hide_button_action":False}, file, indent=4)
        
        time.sleep(CHECK_INTERVAL)

if __name__ == "__main__":
    main()