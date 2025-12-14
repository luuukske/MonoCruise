from time import sleep
from multiprocessing.shared_memory import SharedMemory
from subprocess import Popen, DEVNULL, check_output, CalledProcessError

CHECK_INTERVAL = 1

_launched_connected = True

def is_active():
    try:
        mem = SharedMemory(name="Local\\SCSTelemetry", create=False)
        return bool(mem.buf[0])
    except (FileNotFoundError, ValueError, OSError):
        return False

def is_process_running(process_name, min_count=1):
    """Check if process is running using tasklist (lower memory than psutil)"""
    try:
        output = check_output(
            f'tasklist /FI "IMAGENAME eq {process_name}" /NH',
            shell=True, text=True, stderr=DEVNULL
        )
        count = output.lower().count(process_name.lower())
        return count >= min_count
    except CalledProcessError:
        return False

def start_program():
    if not is_process_running("MonoCruise.exe"):
        Popen(["MonoCruise.exe"])

def is_ETS2_running():
    global _launched_connected
    try:
        if is_active():
            return True
    except (FileNotFoundError, Exception) as e:
        if isinstance(e, FileNotFoundError) or str(e) == "SDK_NOT_ACTIVE":
            pass
    _launched_connected = False
    return False

def main():
    global _launched_connected
    _iteration_count = 0
    
    if is_process_running("ETS2_checker_MonoCruise.exe", min_count=2):
        exit()

    while True:
        if is_ETS2_running():
            if not _launched_connected:
                start_program()
                _launched_connected = True
            while is_ETS2_running():
                sleep(CHECK_INTERVAL)
            _iteration_count += 1
            if _iteration_count >= 10:
                with open('./_internal/saves.json', "w") as file:
                    from json import dump
                    dump({"hide_button_action":  False}, file)
                _iteration_count = 0
        sleep(CHECK_INTERVAL)

if __name__ == "__main__":
    main()