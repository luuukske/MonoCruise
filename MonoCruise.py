print("hello?")
import threading
import customtkinter as ctk
import tkinter as tk
from PIL import Image
import sys
import ctypes
import os
import pygame
import json
import time
from cv2 import line, putText, circle, resize, cvtColor, COLOR_BGR2RGB, FONT_HERSHEY_SIMPLEX, circle, putText
import numpy as np
import traceback
import inspect
from datetime import datetime
import psutil
import subprocess

try:
    import truck_telemetry
except:
    raise Exception("truck_telemetry is not installed")
try:
    sys.path.append('./_internal')
    from scscontroller import SCSController
except:
    raise Exception("scscontroller is not installed")

# Get system DPI scaling
try:
    user32 = ctypes.windll.user32
    user32.SetProcessDPIAware()
    scaling = user32.GetDpiForSystem() / 96.0
except:
    scaling = 1.0

# Configure default font
try:
    # Try to use Segoe UI on Windows
    default_font = ("Segoe UI", 13)
    default_font_bold = ("Segoe UI", 13, "bold")
    code_font = ("Courier New", 10)
    # Test if font exists
    tk.font.nametofont("TkDefaultFont").configure(family="Segoe UI")
except:
    # Fallback to Helvetica Neue or system default
    default_font = ("Helvetica Neue", 13)
    default_font_bold = ("Helvetica Neue", 13, "bold")
    code_font = ("Courier New", 10)

# Set default font for all CTk widgets
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

# Global event for thread control
exit_event = threading.Event()
close_bar_event = threading.Event()
ets2_detected = threading.Event()  # Event for ETS2 detection
ui_ready = threading.Event()       # Event to signal UI is ready

# Banner colors
WAITING_COLOR = "#1f538d"  # Blue
CONNECTED_COLOR = "#304230"  # Light grey
DEFAULT_COLOR = "#2B2B2B"  # Dark grey
SETTINGS_COLOR = "#454545"  # Dark grey
DISABLED_COLOR = "#404040"  # Dark grey
CMD_COLOR = "#808080"  # Light grey
TRANSITION_DURATION = 1  # seconds
TRANSITION_FRAMERATE = 30  # frames per second

current_fade_id = 0

_running_process = None
_data_cache = None

def serialize_joystick(joy):
    """
    Returns the joystick's UUID as a string.
    
    Args:
        joy (pygame.joystick.Joystick): The joystick object.
        
    Returns:
        str or None: The joystick's UUID as a string, or None if not available.
    """
    if joy is None:
        return None
    # Use get_guid() if available
    return str(joy.get_guid()) if hasattr(joy, "get_guid") else None

def deserialize_joystick(uuid_str):
    """
    Searches for a connected joystick whose UUID matches the provided string.
    
    Args:
        uuid_str (str): The UUID string saved in the JSON file.
        
    Returns:
        pygame.joystick.Joystick or None: The matching joystick object if found.
    """
    pygame.joystick.quit()
    pygame.joystick.init()
    
    count = pygame.joystick.get_count()
    for i in range(count):
        j = pygame.joystick.Joystick(i)
        j.init()
        if hasattr(j, "get_guid") and str(j.get_guid()) == uuid_str:
            return j
    return None


def load_variables(filename):
    """
    Reads saved variables from a JSON file.
    
    If a joystick UUID is stored, the global variable 'device' will be set
    to the corresponding pygame joystick object.
    
    Args:
        filename (str): The name of the JSON file.
        
    Returns:
        dict: A dictionary containing the saved variables.
    """
    global _data_cache, device, debug_mode
    if _data_cache is None or _data_cache == {}:
        if os.path.exists(filename):
            try:
                with open(filename, "r") as file:
                    _data_cache = json.load(file)
            except json.JSONDecodeError:
                print(f"Error: Could not decode JSON from '{filename}'.")
                _data_cache = {}
        else:
            _data_cache = {}

    # Set the global device variable if a UUID is found in the saved data.
    if "device" in _data_cache and isinstance(_data_cache["device"], str):
        recovered = deserialize_joystick(_data_cache["device"])
        device = recovered  # Set the global device variable
        if recovered:
            print(f"Recovered joystick: {recovered.get_name()}")
        else:
            print("Joystick with saved UUID not found among connected devices.")
    
    return _data_cache

def check_and_start_exe():
    """
    Checks if ETS2_checker_MonoCruise.exe should be running based on global autostart_variable.
    Handles starting and stopping the exe based on the current state.
    """
    global _running_process, autostart_variable
    
    # Check the global autostart_variable
    should_be_running = autostart_variable.get()
    
    exe_name = "ETS2_checker_MonoCruise.exe"
    exe_path = "./ETS2_checker_MonoCruise.exe"
    
    # Check if the process is already running
    is_running = False
    running_pid = None
    
    try:
        for proc in psutil.process_iter(['pid', 'name']):
            if proc.info['name'] and proc.info['name'].lower() == exe_name.lower():
                is_running = True
                running_pid = proc.info['pid']
                if _running_process is None:
                    _running_process = psutil.Process(running_pid)
                break
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        pass
    
    if should_be_running and not is_running:
        # Should be running but isn't - start it
        try:
            _running_process = subprocess.Popen([exe_path])
            cmd_print(f"Started {exe_name}", display_duration=2)
        except Exception as e:
            cmd_print(f"Failed to start {exe_name}: {e}", display_duration=3)
    
    elif not should_be_running and is_running:
        # Should not be running but is - stop it
        try:
            if _running_process is not None:
                # Kill the entire process tree (parent and all children)
                if hasattr(_running_process, 'pid'):
                    parent = psutil.Process(_running_process.pid)
                else:
                    parent = _running_process
                
                children = parent.children(recursive=True)
                
                # First try to terminate gracefully
                for child in children:
                    try:
                        child.terminate()
                    except psutil.NoSuchProcess:
                        pass
                parent.terminate()
                
                # Wait for processes to terminate
                gone, alive = psutil.wait_procs(children + [parent], timeout=5)
                
                # Force kill any remaining processes
                for p in alive:
                    try:
                        p.kill()
                    except psutil.NoSuchProcess:
                        pass
                
                cmd_print(f"Stopped {exe_name} and all child processes", display_duration=2)
        except Exception as e:
            cmd_print(f"Failed to stop {exe_name}: {e}", display_duration=3)
        finally:
            _running_process = None
    

def save_variables(filename, **kwargs): #for starting and stopping the watchdog on ETS2
    """
    Saves given variables to a JSON file while preserving any existing values.
    
    For the key "device", only the joystick's UUID (as a string) is saved.
    This function merges new key/value pairs with the previously cached data
    so that existing data is not lost, and it ensures that _data_cache["device"]
    remains a string.
    
    Args:
        filename (str): The name of the JSON file.
        **kwargs: Arbitrary keyword arguments representing variables.
    """
    global _data_cache
    if not kwargs:
        return load_variables(filename)
    
    if _data_cache is None:
        _data_cache = load_variables(filename)
    
    # If "device" exists in _data_cache but is not a string, convert it.
    if "device" in _data_cache and not isinstance(_data_cache["device"], str):
        _data_cache["device"] = serialize_joystick(_data_cache["device"])
    
    # Merge new values into a copy of the current data.
    merged_data = _data_cache.copy()
    for key, value in kwargs.items():
        if key == "device":
            merged_data[key] = serialize_joystick(value) if value is not None else None
        else:
            merged_data[key] = value

    # Only update the file if something has changed.
    if merged_data != _data_cache:
        _data_cache = merged_data  # Update the global cache.
        with open(filename, "w") as file:
            json.dump(merged_data, file, indent=4)
        cmd_print(f"Changes detected. Variables saved to '{filename}'.", display_duration=1)
        
        # Check if autostart_variable was changed and handle exe accordingly
        if "autostart_variable" in kwargs:
            check_and_start_exe()



def cmd_print(text, color=CMD_COLOR, display_duration=10):
    global current_fade_id, cmd_label

    # Immediately print the text to the console.
    print(text)

    # If there's no label, exit early.
    if cmd_label is None:
        return

    # Increment fade id. This "invalidates" any currently running fade transition.
    current_fade_id += 1
    local_fade_id = current_fade_id

    def fade_and_clear(local_id, txt):
        # Set your desired colors (adjust as needed for your theme).
        original_color = color     # initial text color
        background_color = "#202020"   # color to fade into (usually matching the background)

        # Immediately update the label with the new text and reset its text color.
        cmd_label.after(0, lambda: cmd_label.configure(text=txt, text_color=original_color))

        # Wait for 10 seconds while the text is fully visible.
        # Instead of one long sleep, we check in small intervals if the fade should be canceled.
        check_interval = 0.2   # seconds
        increments = int(display_duration / check_interval)
        for _ in range(increments):
            time.sleep(check_interval)
            if local_id != current_fade_id or exit_event.is_set():
                # A new cmd_print has been issued; cancel this fade process.
                return

        # Fade transition: over 2 seconds in 20 steps.
        fade_duration = 2      # seconds for fading transition
        steps = 20
        step_delay = fade_duration / steps
        for i in range(steps):
            time.sleep(step_delay)
            if local_id != current_fade_id or exit_event.is_set():
                # A new message arrived; cancel the fade.
                return
            factor = (i + 1) / steps
            intermediate_color = interpolate_color(original_color, background_color, factor)
            cmd_label.after(0, lambda nc=intermediate_color: cmd_label.configure(text_color=nc))

        # Once fading is complete, clear the text entirely.
        cmd_label.after(0, lambda: cmd_label.configure(text=""))

    # Start the fade routine in a daemon thread so that it doesn't block your main thread.
    threading.Thread(target=fade_and_clear, args=(local_fade_id, text), daemon=True).start()

def connect_joystick():
    """Main game logic"""
    global device
    global axis
    global gasaxis
    global brakeaxis
    global pygame
    global connected_joystick_label
    global connected_joystick_gas_axis_label
    global connected_joystick_brake_axis_label
    global restart_connection_label

    if gasaxis != 0 or brakeaxis != 0:
        restart_connection_button.configure(text="reconnect to pedals")

    gasaxis = 0
    brakeaxis = 0
    device = 0
    joysticks = {}
    pygame.joystick.quit()
    pygame.quit()

    connected_joystick_label.configure(text="None connected")
    connected_joystick_gas_axis_label.configure(text="None")
    connected_joystick_brake_axis_label.configure(text="None")
    # Initialize pygame for joystick handling
    pygame.init()
    pygame.joystick.init()
    
    try:
        # Wait for joystick input
        cmd_print("Waiting for joystick input...")
        restart_connection_label.configure(text="connecting...")
        while not exit_event.is_set():
            for event in pygame.event.get():
                if event.type == pygame.JOYDEVICEADDED:
                    joy = pygame.joystick.Joystick(event.device_index)
                    joysticks[joy.get_instance_id()] = joy
                    if joy.get_name() != "vJoy Device":
                        cmd_print(f"Joystick connected: {joy.get_name()}")
                    restart_connection_label.configure(text="tap the brake pedal")
                elif event.type == pygame.JOYAXISMOTION:
                    if event.instance_id in joysticks and joysticks[event.instance_id].get_name() != "vJoy Device":
                        device = joysticks[event.instance_id]
                        if brakeaxis == 0:
                            brakeaxis = event.axis
                            cmd_print(f"Brake axis set to {brakeaxis}")
                            connected_joystick_brake_axis_label.configure(text=f"axis {brakeaxis}")
                            restart_connection_label.configure(text="saving...")
                            time.sleep(0.5)
                            restart_connection_label.configure(text="tap the gas pedal")
                        elif event.axis != brakeaxis:
                            gasaxis = event.axis
                            cmd_print(f"Gas axis set to {gasaxis}")
                            connected_joystick_gas_axis_label.configure(text=f"axis {gasaxis}")
                            restart_connection_label.configure(text="saving...")
                            time.sleep(0.5)
                            restart_connection_label.configure(text="")
                            device.init()
                            break
                elif event.type == pygame.JOYDEVICEREMOVED:
                    if event.instance_id in joysticks:
                        cmd_print(f"Joystick disconnected: {joysticks[event.instance_id].get_name()}")
                        del joysticks[event.instance_id]
            
            if gasaxis != 0 and brakeaxis != 0:
                restart_connection_button.configure(text="reconnect to pedals")
                cmd_print("pedals connected")
                break
                
            time.sleep(0.1)  # Small sleep to prevent CPU hogging
        # give the name of the joystick
        #make the name end with ... if the pixels available is smaller than the text
        if len(device.get_name()) > 25:
            connected_joystick_label.configure(text=f"{device.get_name()[:25]}...")
        else:
            connected_joystick_label.configure(text=f"{device.get_name()}")

        #save variables to the file
        save_variables(os.path.join(os.path.dirname(os.path.abspath(__file__)), "saves.json"),
                       gasaxis = gasaxis,
                       brakeaxis = brakeaxis,
                       device = device
                       )
        
    except Exception as e:
        log_error(e)
        
class LoadingDots:
    def __init__(self, label):
        self.label = label
        self.is_playing = False
        self.dots = ["", ".", "..", "..."]
        self.current_dot = 1
        self.frame_duration = 333
        self._after_id = None
        self._last_update = 0  # Track last update time
    def start(self):
        if not self.is_playing:
            self.is_playing = True
            self._last_update = time.time()
            self.update_dots()
    
    def stop(self):
        self.is_playing = False
        if self._after_id is not None:
            self.label.after_cancel(self._after_id)
            self._after_id = None
    
    def update_dots(self):
        if not self.is_playing:
            return
        
        current_time = time.time()
        # Ensure minimum time between updates
        if current_time - self._last_update >= self.frame_duration / 1000:
            # Update the label with the current dots
            self.label.configure(text=f"Waiting for ETS2{self.dots[self.current_dot]}")
            
            # Move to next dot, ensuring we always have at least one dot
            self.current_dot = (self.current_dot + 1) % len(self.dots)
            if self.current_dot == 0:
                self.current_dot = 1
            
            self._last_update = current_time
        
        # Schedule next update with shorter interval for smoother updates
        self._after_id = self.label.after(50, self.update_dots)

def plot_onepedaldrive(return_result=False):
    background_color = hex_to_rgb(DEFAULT_COLOR)
    
    # Create the base canvas.
    height, width = 400, 400
    img = np.empty((height, width, 3), dtype=np.uint8)
    img[:] = background_color

    # Sample 300 points from -1 to 1; for positive x apply gas, for negative apply brake.
    num_points = 100
    x_values = np.linspace(-1, 1, num_points)
    y_values = []
    for x in x_values:
        if x >= 0:
            gas = x
            brake = 0
        else:
            gas = 0
            brake = -x  # Convert negative x (brake) to positive.
        gas_output, brake_output = onepedaldrive(gas, brake)
        y_values.append(gas_output - brake_output)

    # Determine the y-range for correct scaling.
    y_min, y_max = min(y_values), max(y_values)
    if abs(y_max - y_min) < 1e-6:
        y_max = y_min + 1e-6

    # Mapping function: converts (x, y) in data coordinates to image (pixel) coordinates.
    def to_img_coords(x, y):
        col = int((x - (-1)) / (1 - (-1)) * width)
        row = height - int((y - y_min) / (y_max - y_min) * height)
        return (col, row)

    # Draw guidelines for 0 (x-axis and y-axis) in light grey.
    guideline_color = (100, 100, 100)
    origin = to_img_coords(0, 0)
    line(img, (0, origin[1]), (width - 1, origin[1]), guideline_color, 2)
    line(img, (origin[0], 0), (origin[0], height - 1), guideline_color, 2)

    # Add axis labels.
    putText(img, "input", (5, origin[1] - 10), FONT_HERSHEY_SIMPLEX, 0.7, guideline_color, 2)
    putText(img, "outputs", (origin[0] - 90, 30), FONT_HERSHEY_SIMPLEX, 0.7, guideline_color, 2)

    # Draw the graph: blue for segments where output is non-negative, red for negative.
    # Split segments if the graph crosses y = 0.
    for i in range(len(x_values) - 1):
        x1, y1 = x_values[i], y_values[i]
        x2, y2 = x_values[i + 1], y_values[i + 1]
        pt1 = to_img_coords(x1, y1)
        pt2 = to_img_coords(x2, y2)
        if y1 >= 0 and y2 >= 0:
            line(img, pt1, pt2, (255, 50, 50), 2)
        elif y1 < 0 and y2 < 0:
            line(img, pt1, pt2, (0, 0, 225), 2)
        else:
            # Calculate the intersection point at y = 0.
            t = -y1 / (y2 - y1)
            x_int = x1 + t * (x2 - x1)
            pt_int = to_img_coords(x_int, 0)
            if y1 < 0:
                line(img, pt1, pt_int, (0, 0, 225), 2)
                line(img, pt_int, pt2, (255, 50, 50), 2)
            else:
                line(img, pt1, pt_int, (255, 50, 50), 2)
                line(img, pt_int, pt2, (0, 0, 225), 2)

    if return_result:
        # Return the image, the conversion function, y-range, and origin.
        return img, to_img_coords

def overlay_dot_layer(x_value, new_width=600, new_height=600, image=None, to_img_coords=None):
    global gas_output
    global brake_output
    """
    Overlays a dot on the graph at the specified x_value.
    The dot (diameter 10) changes color based on whether the y output is negative.
    The resulting image is resized to (new_width, new_height), converted to a CTkImage, and returned.
    """
    # Get the base graph image and helper data
    
    # Compute the output at the chosen x_value.
    if x_value >= 0:
        gas = x_value
        brake = 0
    else:
        gas = 0
        brake = -x_value
    y_val = gas_output - brake_output

    # Map the (x_value, y_val) to pixel coordinates.
    dot_center = to_img_coords(x_value, y_val)
    
    # Choose dot color: blue for non-negative, red for negative.
    dot_color = (255, 50, 50) if y_val >= 0 else (0, 0, 225)
    
    #create a duplicate of the image
    image_duplicate = image.copy()

    # Draw the filled circle (radius=5 for diameter 10) on the image.
    circle(image_duplicate, dot_center, 7, dot_color, -1)
    # add text to the dot saying the value of the output
    putText(image_duplicate, f"{round(y_val, 2)}", (dot_center[0] - 10, dot_center[1] - 10), FONT_HERSHEY_SIMPLEX, 0.7, dot_color, 2)
    
    # Resize the image to the dimensions specified by the function arguments.
    resized_img = resize(image_duplicate, (new_width, new_height))
    
    # Convert the OpenCV image from BGR to RGB, then create a PIL image.
    resized_img_rgb = cvtColor(resized_img, COLOR_BGR2RGB)
    pil_image = Image.fromarray(resized_img_rgb)
    
    # Convert the PIL image into a CTkImage with the new size.
    ctk_image = ctk.CTkImage(pil_image, size=(new_width, new_height))
    
    return ctk_image

def interpolate(a, b, x, ya, yb):
    return a + (b - a) * ((x - ya) / (yb - ya))

def onepedaldrive(gasval, brakeval):
    global opd_mode_variable
    global gas_exponent_variable
    global brake_exponent_variable
    global max_opd_brake_variable
    global stopped
    global gear
    global data
    global offset_variable

    offset = offset_variable.get()
    
    val1 = min(max(gasval, 0), 1)
    val2 = (min(max(brakeval, 0), 1)**1.5)*-1
    sum_values = val1+val2

    if opd_mode_variable.get() == True:
        if sum_values<=offset:
            value = ((1)/(1+offset))*sum_values-((offset)/(1+offset))
        else:
            value = ((1)/(1-offset))*sum_values-((offset)/(1-offset))
    else:
        value = sum_values

    gasval = max(0, value)
    brakeval = min(min(0, value),val2)*-1

    gasval = gasval**gas_exponent_variable.get()
    brakeval = brakeval**brake_exponent_variable.get()

    return gasval, brakeval

def send(a, b, controller):
    global bar_val

    bar_val = a-b
    a = a
    b = b
 
    setattr(controller, "aforward", float(a))
    setattr(controller, "abackward", float(b))

def get_error_context():
    """Get detailed context about where an error occurred"""
    frame = inspect.currentframe().f_back
    try:
        filename = frame.f_code.co_filename
        function = frame.f_code.co_name
        line_number = frame.f_lineno
        locals_dict = frame.f_locals
        # Get the line of code that caused the error
        with open(filename, 'r') as f:
            lines = f.readlines()
            code_line = lines[line_number - 1].strip() if 0 <= line_number - 1 < len(lines) else "Unknown line"
        
        # Format the context information
        context = {
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'file': os.path.basename(filename),
            'function': function,
            'line': line_number,
            'code': code_line,
            'locals': {k: str(v) for k, v in locals_dict.items() if not k.startswith('_')}
        }
        return context
    except Exception as e:
        return {'error': f"Failed to get error context: {str(e)}"}

def log_error(error, context=None):
    """Log detailed error information and append to a log file"""
    if context is None:
        context = get_error_context()
    
    error_details = {
        'error_type': type(error).__name__,
        'error_message': str(error),
        'traceback': traceback.format_exc(),
        'context': context
    }
    
    # Prepare the error log string
    log_lines = []
    log_lines.append("\n" + "="*80)
    log_lines.append(f"ERROR OCCURRED AT: {context.get('timestamp', 'N/A')}")
    log_lines.append(f"Location: {context.get('file', 'N/A')}:{context.get('line', 'N/A')} in {context.get('function', 'N/A')}")
    log_lines.append(f"Error Type: {error_details['error_type']}")
    log_lines.append(f"Error Message: {error_details['error_message']}")
    log_lines.append("\nCode Context:")
    log_lines.append(f"  {context.get('code', 'N/A')}")
    log_lines.append("\nLocal Variables:")
    for var, value in context.get('locals', {}).items():
        #limit the length of the value to 100 characters
        if len(value) > 100:
            value = value[:100] + "..."
        log_lines.append(f"  {var} = {value}")
    log_lines.append("\nFull Traceback:")
    log_lines.append(error_details['traceback'])
    log_lines.append("="*80 + "\n")
    log_text = "\n".join(log_lines)

    # Print detailed error information
    print(log_text)
    cmd_print(f"error: '{error_details['error_message']}' on line {context.get('line', 'N/A')}", "#FF2020", 10)

    # Check if error_log.txt exists, if not, create it
    if not os.path.exists('error_log.txt'):
        try:
            with open('error_log.txt', 'w', encoding='utf-8') as f:
                f.write("=== Error Log Created ===\n")
        except Exception as create_file_error:
            cmd_print(f"Failed to create error_log.txt: {create_file_error}")

    # Append the error log to a file
    try:
        with open('error_log.txt', 'a', encoding='utf-8') as f:
            f.write(log_text)
    except Exception as file_error:
        cmd_print(f"Failed to write to error_log.txt: {file_error}")
    
    # Optionally, you could also write to a log file here
    # with open('error_log.txt', 'a') as f:
    #     f.write(str(error_details) + '\n')

def is_process_running(exe_name):
    """
    Scan all visible processes for exe_name (case-insensitive).
    Ignore AccessDenied errors.
    """
    for proc in psutil.process_iter(["name"]):
        try:
            if proc.info["name"] and proc.info["name"].lower() == exe_name.lower():
                return True
        except psutil.AccessDenied:
            continue
        time.sleep(0.001)
    return False




def sdk_check_thread():
    global autostart_variable
    global root
    """Background thread to check for ETS2 SDK connection"""
    time.sleep(0.2)
    cmd_print("SDK check thread starting...")
    first = True
    manual_start = False
    while not exit_event.is_set():
        try:
            truck_telemetry.init()  # Signal that ETS2 has been detected
            while not exit_event.is_set():
                # Try to get data to check if SDK is still connected
                data = truck_telemetry.get_data()
                if not data["sdkActive"]:  # Check if SDK is still active
                    raise Exception("SDK_NOT_ACTIVE")
                ets2_detected.set()
                time.sleep(0.5)  # Small sleep to prevent CPU hogging
        except Exception as e:
            if isinstance(e, FileNotFoundError) or str(e) == "Not support this telemetry sdk version" or str(e) == "SDK_NOT_ACTIVE":
                print(isinstance(e, FileNotFoundError), str(e) == "Not support this telemetry sdk version", str(e) == "SDK_NOT_ACTIVE")
                #print("ETS2 not found, please start the game first.")
                ets2_detected.clear()
                game_running = is_process_running("eurotrucks2.exe")
                if first:
                    if not game_running:
                        manual_start = True
                    print(f"starting in {'manual' if manual_start else 'auto'} start mode")
                if autostart_variable.get()==True and not first and not game_running and not manual_start:
                    exit_event.set()
                    print("shutting down")
                time.sleep(0.2)
            else:
                print(e)
                context = get_error_context()
                log_error(e, context)
        finally:
            first=False

def main():
    global controller
    """Main game logic"""
    global gasaxis
    global brakeaxis
    global joy
    global device
    global axis
    global polling_rate
    global hazards_variable
    global autodisable_hazards
    global horn_variable
    global airhorn_variable
    global speed
    global stopped
    global gear
    global data
    global img
    global to_img_coords
    global gas_exponent_variable
    global brake_exponent_variable
    global max_opd_brake_variable
    global bar_variable
    global bar_val
    global em_stop
    global opdgasval
    global opdbrakeval
    global gas_output
    global brake_output
    global autostart_variable
    # Initialize pygame for joystick handling
    
    # Start SDK check thread
    sdk_thread = threading.Thread(target=sdk_check_thread, daemon=True)
    sdk_thread.start()
    
    try:
        while gasaxis == 0 or brakeaxis == 0:
            if exit_event.is_set():
                break
            time.sleep(0.5)
            save_variables(os.path.join(os.path.dirname(os.path.abspath(__file__)), "saves.json"),
                           bar_variable = bar_variable.get(),
                           gas_exponent_variable = gas_exponent_variable.get(),
                           brake_exponent_variable = brake_exponent_variable.get(),
                           max_opd_brake_variable = max_opd_brake_variable.get(),
                           offset_variable = offset_variable.get(),
                           gasaxis = gasaxis,
                           brakeaxis  =  brakeaxis,
                           polling_rate = polling_rate.get(),
                           opd_mode_variable = opd_mode_variable.get(),
                           hazards_variable = hazards_variable.get(),
                           autodisable_hazards = autodisable_hazards.get(),
                           horn_variable = horn_variable.get(),
                           airhorn_variable = airhorn_variable.get(),
                           autostart_variable = autostart_variable.get()
                           )
            
        check_and_start_exe()

        if exit_event.is_set():
            return
            
        # Main game loop
        prev_brakeval = 0
        # prev_gasval = 1   (not used)
        brakeval = 0
        gasval = 0
        prev_speed = 0
        opdgasval = 0
        opdbrakeval = 0
        arrived = False
        stopped = False
        horn = False
        em_stop = False
        latency_timestamp = time.time()-0.015
        latency = 0.015
        hazards_prompted = False

        # start the bar thread
        bar_var_update()

        img, to_img_coords = plot_onepedaldrive(return_result=True)

        while 1:

            # save variables to the file
            save_variables(os.path.join(os.path.dirname(os.path.abspath(__file__)), "saves.json"),
                           bar_variable = bar_variable.get(),
                           gas_exponent_variable = gas_exponent_variable.get(),
                           brake_exponent_variable = brake_exponent_variable.get(),
                           max_opd_brake_variable = max_opd_brake_variable.get(),
                           offset_variable = offset_variable.get(),
                           gasaxis = gasaxis,
                           brakeaxis  =  brakeaxis,
                           polling_rate = polling_rate.get(),
                           opd_mode_variable = opd_mode_variable.get(),
                           hazards_variable = hazards_variable.get(),
                           autodisable_hazards = autodisable_hazards.get(),
                           horn_variable = horn_variable.get(),
                           airhorn_variable = airhorn_variable.get(),
                           autostart_variable = autostart_variable.get()
                           )

            while gasaxis == 0 or brakeaxis == 0 or device == 0:
                if exit_event.is_set():
                    break
                time.sleep(0.5)
                opdgasval = 0
                opdbrakeval = 0

            if exit_event.is_set():
                break

            timestamp = time.time()

            latency = timestamp - latency_timestamp
            latency_timestamp = timestamp

            latency_multiplier = (latency / 0.015) * 2

            # get app settings
            hazards_variable_var = hazards_variable.get()
            autodisable_hazards_var = autodisable_hazards.get()
            horn_variable_var = horn_variable.get()
            airhorn_variable_var = airhorn_variable.get()

            # get input
            for event in pygame.event.get():
                if event.type == pygame.JOYAXISMOTION:
                    brakeval = round((device.get_axis(brakeaxis)*-1+1)/2,3)
                    gasval = round((device.get_axis(gasaxis)*-1+1)/2,3)


            if not ets2_detected.is_set():
                cmd_print("Waiting for ETS2 SDK connection...")
                opdgasval = 0
                opdbrakeval = 0
                time.sleep(0.5)
                continue

            data = truck_telemetry.get_data()
            if not data["sdkActive"]:
                continue

            if int(data["routeDistance"]) != 0:
                if int(data["routeDistance"]) < 50:
                    arrived = True
                elif int(data["routeDistance"]) > 1000:
                    arrived = False
            else:
                arrived = False

            speed = round(data["speed"] * 3.6,3)  # Convert speed from m/s to km/h
            gear = int(data["gearDashboard"])

            opdgasval, opdbrakeval = onepedaldrive(gasval, brakeval)
            gas_output = opdgasval
            brake_output = opdbrakeval

            
            if data["cruiseControl"] == True and data["cruiseControlSpeed"] > 0 and brakeval == 0:
                opdbrakeval = 0
            elif stopped == True and gasval > 0 and speed <= 1 and gear != 0:
                opdbrakeval = 0.05
            elif stopped == True:
                opdbrakeval = max(0, opdbrakeval)

            if speed <= 0.3 and speed >= -0.3 and gasval == 0 and gear != 0:
                stopped = True
            #if gear >= 0 and speed <= -0.01:
            #    stopped = False
            #    opdbrakeval = max(0.75 , opdbrakeval)
            elif stopped == True and (speed >= 0.5 or speed <= -0.5):
                stopped = False
            elif stopped == True and opdgasval > 0.75:
                stopped = False
            if data["parkBrake"] == True and speed <= 2 and speed >= -2:
                stopped = True

            if debug_mode.get() == True:
                print(f"gasvalue: {round(opdgasval,3)} \tbrakevalue: {round(opdbrakeval,3)} \tspeed: {round(speed,3)} \tprev_speed: {round(prev_speed,3)} \tstopped: {stopped} \tgasval: {round(gasval,3)} \tbrakeval: {round(brakeval,3)} \tdiff: {round(prev_brakeval-brakeval,3)} \tdiff2: {round(prev_speed-speed,3)} \tlatency: {round(latency,3)} \tdist: {round(data['routeDistance'],3)} \tarrived: {arrived} \thazards: {data['lightsHazards']} \thazards_var: {hazards_variable_var} \thazards_prompted: {hazards_prompted}")
            
            if (prev_brakeval-brakeval <= -0.07*latency_multiplier or brakeval >= 0.8) and stopped == False and speed > 10 and arrived == False:
                stopped = True
                em_stop = True
                send(0,1, controller)
                cmd_print("#####stopping#####", "#FF2020", 3)
                if prev_brakeval-brakeval <= -0.15*latency_multiplier and speed > 40:
                    horn = True
                    cmd_print("#####HONKING!#####", "#FF2020", 3)
            elif prev_speed-speed >= 5 and arrived == False:
                stopped = True
                em_stop = True
                send(0,1, controller)
                cmd_print("#####crash#####", "#FF2020", 10)
                if data["lightsHazards"] == False and hazards_variable_var == True:
                    setattr(controller, "accmode", True)
                    hazards_prompted = True
                    time.sleep(0.05)
                    setattr(controller, "accmode", False)

            else:
                send(opdgasval, opdbrakeval, controller)
            if em_stop == True:
                print(f"lightsHazards: {data['lightsHazards']} \tspeed: {speed} \thazards_variable_var: {hazards_variable_var} \thazards_prompted: {hazards_prompted}")
                if data["lightsHazards"] == False and speed > 30 and hazards_variable_var == True and hazards_prompted == False:
                    setattr(controller, "accmode", True)
                    if autodisable_hazards_var == True:
                        hazards_prompted = True
                if horn_variable_var == True and horn == True:
                    setattr(controller, "wipers4", True)
                if airhorn_variable_var == True and horn == True:
                    setattr(controller, "wipers3", True)
                time.sleep(0.1)
                setattr(controller, "accmode", False)
                while (brakeval > 0.8 or prev_brakeval-brakeval <= -0.03*latency_multiplier):
                    if prev_brakeval-brakeval <= -0.15*latency_multiplier and horn == False:
                        cmd_print("#####/HONKING!\#####", "#FF2020", 3)
                        horn = True
                        if horn_variable_var == True:
                            setattr(controller, "wipers3", True)
                        if airhorn_variable_var == True:
                            setattr(controller, "wipers4", True)
                    for event in pygame.event.get():
                        if event.type == pygame.JOYAXISMOTION:
                            brakeval = round((device.get_axis(brakeaxis)*-1+1)/2,3)
                    time.sleep(0.05)
                    prev_brakeval = brakeval
                    # prev_gasval = gasval
                    prev_speed = speed
                    data = truck_telemetry.get_data()
                    speed = round(data["speed"] * 3.6,3)
                    timestamp = time.time()
                    latency_timestamp = time.time()-0.005
                send(0,0, controller)
                em_stop = False

                """
                if data["lightsHazards"] == True and speed > 10 and hazards_variable_var == True:
                    setattr(controller, "accmode", True)
                    hazards_prompted = False
                    time.sleep(0.05)
                    setattr(controller, "accmode", False)
                """
                if horn_variable_var == True and horn == True:
                    setattr(controller, "wipers3", False)
                if airhorn_variable_var == True:
                    setattr(controller, "wipers4", False)
                horn = False
                
            """
            if gasval < 0.5 and hazards_variable_var == True and hazards_prompted == True:
                cmd_print("autodisabled hazards (for safety)")
                setattr(controller, "accmode", True)
                hazards_prompted = False
                time.sleep(0.05)
                setattr(controller, "accmode", False)
            """

            if autodisable_hazards_var == True and hazards_prompted == True and speed > 10 and data["lightsHazards"] == True and gasval > 0.5 and brakeval == 0:
                cmd_print("autodisabled hazards")
                setattr(controller, "accmode", True)
                hazards_prompted = False
                time.sleep(0.05)
                setattr(controller, "accmode", False)

            prev_brakeval = brakeval
            prev_speed = speed







            

            # update the live visualization frame and enlarges it to the available space of the frame with a min size of 200x200 and a max size of 1000x1000

            # calculate the size of the frame
            frame_width = live_visualization_frame.winfo_width()-100
            frame_height = live_visualization_frame.winfo_height()-100
            frame_size = min(min(max(200, frame_width), 1000), min(max(200, frame_height), 1000))
            image = overlay_dot_layer(gasval-brakeval, 200, 200, img, to_img_coords)
            live_visualization_frame.configure(image=image)
            live_visualization_frame.image = image












            # make the program run at the input polling rate
            try:
                polling_rate.set(max(10, min(100, polling_rate.get())))
                time.sleep(max(0.005, 1/polling_rate.get() - (time.time()-timestamp))) # 0.005 min is for stability
            except:
                if not exit_event.is_set():
                    cmd_print("unreliable input values!")
                    
    except Exception as e:
        context = get_error_context()
        log_error(e, context)
        raise
    finally:
        exit_event.set()
        pygame.quit()

def game_thread():
    cmd_print("Game thread starting...")
    try:
        # Wait for UI to be ready
        ui_ready.wait()
        cmd_print("UI is ready, game logic starting...")
        main()
    except Exception as e:
        context = get_error_context()
        log_error(e, context)
    finally:
        # Signal exit even if there was an exception
        exit_event.set()
        if pygame.get_init():
            pygame.quit()

class AnimatedBar:
    def __init__(self, root):
        self.root = root
        self.temp_gasval = 0
        self.temp_brakeval = 0
        self.bar_width = 7         # Height (in pixels) of the bar.
        self.transparent_color = "magenta"
        
        # Remove window decorations and force window always on top.
        self.root.overrideredirect(True)
        self.root.attributes("-topmost", True)
        
        # Set the window background (and canvas background) to the transparent key color.
        self.root.config(bg=self.transparent_color)
        self.root.wm_attributes("-transparentcolor", self.transparent_color)
        
        # Get screen dimensions.
        self.screen_width = self.root.winfo_screenwidth()
        self.screen_height = self.root.winfo_screenheight()
        
        # Position the window across the full screen width at the bottom.
        self.root.geometry(f"{self.screen_width}x{self.bar_width}+0+{self.screen_height - self.bar_width}")
        
        # Create a canvas that fills the window.
        self.canvas = tk.Canvas(
            self.root,
            width=self.screen_width,
            height=self.bar_width,
            bg=self.transparent_color,
            bd=0,
            highlightthickness=0
        )
        self.canvas.pack()
        
        # Define the horizontal center of the screen (used as the starting edge for both bars)
        self.center = self.screen_width // 2

        # Create two rectangles (initially with zero extension)
        # The gas bar will extend right from center and is drawn in blue.
        self.gas_rect = self.canvas.create_rectangle(
            self.center, 0, self.center, self.bar_width,
            fill="blue", outline="blue"
        )
        # The brake bar will extend left from center and is drawn in red.
        self.brake_rect = self.canvas.create_rectangle(
            self.center, 0, self.center, self.bar_width,
            fill="red", outline="red"
        )
        
        # Flicker state flag (used in emergency mode)
        self.flicker_state = True
        
        # Start animation.
        self.animate()

    def animate(self):
        global em_stop, opdgasval, opdbrakeval, bar_variable, close_bar_event

        #if the screen changes size, update the size of the bar
        if self.screen_width != self.root.winfo_screenwidth() or self.screen_height != self.root.winfo_screenheight():
            self.screen_width = self.root.winfo_screenwidth()
            self.screen_height = self.root.winfo_screenheight()
            self.root.geometry(f"{self.screen_width}x{self.bar_width}+0+{self.screen_height - self.bar_width}")

        if exit_event.is_set() or close_bar_event.is_set():
            self.root.destroy()
            return

        temp_gasval = (opdgasval+self.temp_gasval*5)/6
        temp_brakeval = (opdbrakeval+self.temp_brakeval*5)/6

        value = temp_gasval-temp_brakeval

        if em_stop:
            # Emergency mode: the entire bar flickers between red and transparent every 200ms.
            self.flicker_state = not self.flicker_state
            flicker_color = "red" if self.flicker_state else self.transparent_color
            
            # For emergency, override normal extents. We set the gas bar from center to the right edge
            # and the brake bar from the left edge to center so that together they cover the full width.
            self.canvas.coords(self.gas_rect, self.center, 0, self.screen_width, self.bar_width)
            self.canvas.itemconfig(self.gas_rect, fill=flicker_color, outline=flicker_color, state='normal')
            self.canvas.coords(self.brake_rect, 0, 0, self.center, self.bar_width)
            self.canvas.itemconfig(self.brake_rect, fill=flicker_color, outline=flicker_color, state='normal')
            
            self.root.after(125, self.animate)
            self.temp_gasval = 0
            self.temp_brakeval = 1
        else:
            # Normal operation: reset flicker state.
            self.flicker_state = True

            # Gas bar: extend to the right from center.
            # The maximum extension is from center to the right edge.
            gas_extension = int((self.screen_width - self.center) * min(max(value, 0), 1))
            if gas_extension == 0:
                self.canvas.itemconfig(self.gas_rect, state='hidden')
            else:
                # Make sure the gas bar is visible if it’s needed.
                self.canvas.coords(self.gas_rect, self.center, 0, self.center + gas_extension, self.bar_width)
                self.canvas.itemconfig(self.gas_rect, fill="blue", outline="blue", state='normal')

            # Brake bar: extend to the left from center.
            # The maximum extension is from center to the left edge.
            brake_extension = int(self.center * min(max(value, -1), 0) * -1)
            if brake_extension == 0:
                self.canvas.itemconfig(self.brake_rect, state='hidden')
            else:
                self.canvas.coords(self.brake_rect, self.center - brake_extension, 0, self.center, self.bar_width)
                self.canvas.itemconfig(self.brake_rect, fill="red", outline="red", state='normal')

            self.temp_gasval = temp_gasval
            self.temp_brakeval = temp_brakeval

            self.root.after(10, self.animate)


global device
global gasaxis
global brakeaxis
global polling_rate
global opd_mode_variable
global hazards_variable
global autodisable_hazards_variable
global horn_variable
global airhorn_variable
global cmd_label
global live_visualization_frame
global offset_variable
global gas_exponent_variable
global brake_exponent_variable
global max_opd_brake_variable
global bar_variable
global bar_val
global debug_mode

cmd_label = None
device = 0
global controller
global connected_joystick_label
cmd_print("Starting MonoCruise...")
try:

    # load from save file

    save_file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "saves.json")
    load_variables(save_file_path)

    # load from save file
    try:
        gasaxis = _data_cache["gasaxis"]
        brakeaxis = _data_cache["brakeaxis"]
    except:
        gasaxis = 0
        brakeaxis = 0

    pygame.init()

    controller = SCSController()
    
    # Create the main window
    root = ctk.CTk()
    root.title("MonoCruise")
    try:
        root.iconbitmap(os.path.join(os.path.dirname(os.path.abspath(__file__)), "icon.ico"))
    except:
        pass  # Ignore if icon file not found or on non-Windows platform
    
    # Apply scaling to window sizes
    base_width = 400
    base_height = 300
    root.geometry(f"{int(base_width * scaling)}x{int(base_height * scaling)}")
    root.minsize(int(base_width * scaling), int(base_height * scaling))
    
    # Configure root window for better performance
    root.update_idletasks()  # Process any pending events
    
    main_frame = ctk.CTkFrame(root)
    main_frame.pack(fill="both", expand=True)
    
    # Create loading banner frame with invisible border padding
    banner_frame = ctk.CTkFrame(main_frame, fg_color=WAITING_COLOR, border_width=0)
    banner_frame.pack(fill="x", padx=5, pady=(5,0))
    
    # Create the loading label with dots animation
    loading_label = ctk.CTkLabel(
        banner_frame, 
        text="Initializing",
        text_color="white",
        font=default_font_bold,
    )
    loading_label.pack(pady=1, padx=10)
    
    # Create dots animation
    dots_anim = LoadingDots(loading_label)

    # create a settings page with a scollable frame that stretches to the the support buttons
    settings_frame_width = 400
    settings_frame = ctk.CTkFrame(main_frame, width=settings_frame_width)
    settings_frame.pack(side="left", fill="y", expand=False, padx=(5,0), pady=5)
    settings_frame.pack_propagate(False)  # Prevent frame from shrinking to fit contents

    # Create button frame for the bottom of the settings frame
    button_frame = ctk.CTkFrame(settings_frame, fg_color="transparent")
    button_frame.pack(side="bottom", pady=8, padx=5)

    # Create a label for settings
    settings_label = ctk.CTkLabel(settings_frame, text="Settings", font=("Segoe UI", 17, "bold"), text_color="lightgrey", fg_color="transparent", corner_radius=5)
    settings_label.pack(pady=(5, 0), padx=5, fill="x")

    # create a scrollable frame for the settings that stretches from the left to the right of the settings frame
    scrollable_frame = ctk.CTkScrollableFrame(settings_frame, bg_color="transparent", fg_color="transparent", border_width=1, border_color="#404040", corner_radius=5)
    scrollable_frame.pack(side="top", fill="both", expand=True, padx=5, pady=(5,0))
    #make the grid expand to the side of the scrollable frame
    scrollable_frame.grid_columnconfigure(1, weight=1)
    scrollable_frame.pack_propagate(False)


    #start of the settings


    def refresh_live_visualization():
        global img
        global to_img_coords
        img, to_img_coords = plot_onepedaldrive(return_result=True)

    def new_checkbutton(master, row, column, variable, command=None):
        checkbutton = ctk.CTkCheckBox(master, text="", command=command, font=default_font, text_color="lightgrey", fg_color=SETTINGS_COLOR, corner_radius=5, variable=variable, checkbox_width=20, checkbox_height=20, width=24, height=24, border_color=SETTINGS_COLOR, border_width=1)
        checkbutton.grid(row=row, column=column, padx=5, pady=1, sticky="e")
        return checkbutton

    
    def new_label(master, row, column, text):
        label = ctk.CTkLabel(master, text=text, font=default_font, text_color="lightgrey")
        label.grid(row=row, column=column, padx=10, pady=1, sticky="w")
        return label
    
    def new_entry(master, row, column, textvariable, temp_textvariable, command=None, max_value=None, min_value=None):
        def entry_wait(*args):
            def validate():
                if command is not None:
                    command()
                try:
                    val = float(temp_textvariable.get())
                    if val < min_value:
                        temp_textvariable.set(min_value)  # Revert to 10 if too low
                    elif val > max_value:
                        temp_textvariable.set(max_value)  # Revert to 100 if too high
                    else:
                        textvariable.set(val)  # Apply value only if valid
                except ValueError:
                    temp_textvariable.set(textvariable.get())  # Revert if input isn't a number

            master.after(2000, validate)  # Wait 2 seconds before validating

        temp_textvariable.trace_add("write", entry_wait)  # Monitor changes in temp_textvariable

        entry = ctk.CTkEntry(
            master,
            textvariable=temp_textvariable,
            font=default_font,
            text_color="lightgrey",
            fg_color=DEFAULT_COLOR,
            corner_radius=5,
            width=50,
            border_width=1,
            border_color=SETTINGS_COLOR
        )
        entry.grid(row=row, column=column, padx=10, pady=1, sticky="e")
        
        return entry





    # Create a label for input settings
    settings_label_1 = ctk.CTkLabel(scrollable_frame, text="Inputs", font=default_font_bold, text_color="lightgrey", fg_color=SETTINGS_COLOR, corner_radius=5)
    settings_label_1.grid(row=0, column=0, padx=10, pady=1, columnspan=2, sticky="new")

    connected_joystick_label = new_label(scrollable_frame, 1, 0, "Connected pedals:")

    try:
        connected_joystick_label = ctk.CTkLabel(scrollable_frame, text=f"{device.get_name()[:25]}...", font=default_font, text_color="lightgrey", fg_color=DEFAULT_COLOR, corner_radius=5)
    except:
        connected_joystick_label = ctk.CTkLabel(scrollable_frame, text="None connected", font=default_font, text_color="lightgrey", fg_color=DEFAULT_COLOR, corner_radius=5)
    connected_joystick_label.grid(row=1, column=1, padx=10, pady=1, sticky="e")

    connected_joystick_gas_axis_label = new_label(scrollable_frame, 2, 0, "Gas axis:")

    connected_joystick_gas_axis_label = ctk.CTkLabel(scrollable_frame, text="None" if gasaxis is 0 and device is not 0 else f"axis {gasaxis}", font=default_font, text_color="lightgrey", fg_color=DEFAULT_COLOR, corner_radius=5)
    connected_joystick_gas_axis_label.grid(row=2, column=1, padx=10, pady=1, sticky="e")

    connected_joystick_brake_axis_label = new_label(scrollable_frame, 3, 0, "Brake axis:")

    connected_joystick_brake_axis_label = ctk.CTkLabel(scrollable_frame, text="None" if brakeaxis is 0 and device is not 0 else f"axis {brakeaxis}", font=default_font, text_color="lightgrey", fg_color=DEFAULT_COLOR, corner_radius=5)
    connected_joystick_brake_axis_label.grid(row=3, column=1, padx=10, pady=1, sticky="e")

    restart_connection_button = ctk.CTkButton(scrollable_frame, text="connect to pedals", font=default_font, text_color="lightgrey", fg_color=WAITING_COLOR, corner_radius=5, hover_color="#333366", command=lambda: threading.Thread(target=connect_joystick).start())
    restart_connection_button.grid(row=4, column=0, padx=40, pady=(1,0), columnspan=2, sticky="ew")

    restart_connection_label = ctk.CTkLabel(scrollable_frame, text="", font=default_font, text_color="darkred", corner_radius=5)
    restart_connection_label.grid(row=5, column=0, padx=10, pady=(0,1), columnspan=2, sticky="ew")




    # create a label for program settings
    settings_label_2 = ctk.CTkLabel(scrollable_frame, text="Program settings", font=default_font_bold, text_color="lightgrey", fg_color=SETTINGS_COLOR, corner_radius=5)
    settings_label_2.grid(row=6, column=0, padx=10, pady=1, columnspan=2, sticky="new")

    autostart_variable = ctk.BooleanVar(value=True)
    autostart_MonoCruise_label = new_label(scrollable_frame,7 ,0 , "Autostart MonoCruise:")
    autostart_MonoCruise_button = new_checkbutton(scrollable_frame, 7, 1, autostart_variable)

    input_polling_rate_label = new_label(scrollable_frame, 8, 0, "Target polling rate (Hz):")

    polling_rate = ctk.IntVar(value=50)
    temp_polling_rate = ctk.IntVar(value=50)
    input_polling_rate_label = new_entry(scrollable_frame, 8, 1, polling_rate, temp_polling_rate, max_value=100, min_value=10)

    description_label = ctk.CTkLabel(scrollable_frame, text="lower values mean more input lag, higher values mean more cpu usage", font=("Segoe UI", 11), text_color="#606060", fg_color="transparent", corner_radius=5, bg_color="transparent", anchor="e", wraplength=185, height=10)
    description_label.grid(row=9, column=0, padx=10, pady=(0,4), columnspan=2, sticky="nsew")

    def hazards_var_update():
        if hazards_variable.get() == True:
            autodisable_hazards_label.grid(row=11, column=0, padx=10, pady=1, sticky="w")
            autodisable_hazards_checkbutton.grid(row=11, column=1, padx=5, pady=1, sticky="e")
        else:
            autodisable_hazards_label.grid_forget()
            autodisable_hazards_checkbutton.grid_forget()

    hazards_variable = ctk.BooleanVar(value=True)
    hazards_label = new_label(scrollable_frame, 10, 0, "Hazards:")
    hazards_checkbutton = new_checkbutton(scrollable_frame, 10, 1, hazards_variable, command=hazards_var_update)

    autodisable_hazards = ctk.BooleanVar(value=True)
    autodisable_hazards_label = new_label(scrollable_frame, 11, 0, "  Autodisable hazards:")
    autodisable_hazards_checkbutton = new_checkbutton(scrollable_frame, 11, 1, autodisable_hazards)

    horn_variable = ctk.BooleanVar(value=True)
    horn_label = new_label(scrollable_frame, 12, 0, "Horn:")
    horn_checkbutton = new_checkbutton(scrollable_frame, 12, 1, horn_variable)

    airhorn_variable = ctk.BooleanVar(value=True)
    airhorn_label = new_label(scrollable_frame, 13, 0, "Airhorn:")
    airhorn_checkbutton = new_checkbutton(scrollable_frame, 13, 1, airhorn_variable)

    def start_bar_thread():
        bar_root = tk.Tk()
        AnimatedBar(bar_root)
        bar_root.mainloop()
        print("bar closed")

    bar_thread = None

    def bar_var_update():
        global bar_variable
        global bar_thread
        global close_bar_event
        global bar_thread

        if bar_variable.get() == True:
            close_bar_event.clear()
            if bar_thread is None:
                bar_thread = threading.Thread(target=lambda: start_bar_thread(), daemon=True)
            bar_thread.start()
        else:
            
            close_bar_event.set()
            if bar_thread is not None:
                bar_thread.join()
                bar_thread = None

    bar_variable = ctk.BooleanVar(value=True)
    bar_label = new_label(scrollable_frame, 16, 0, "Live bottom bar:")
    bar_checkbutton = new_checkbutton(scrollable_frame, 16, 1, bar_variable, command=bar_var_update)


    # create a title for the one-pedal-drive system
    opd_title = ctk.CTkLabel(scrollable_frame, text="One-Pedal-Drive", font=default_font_bold, text_color="lightgrey", fg_color=SETTINGS_COLOR, corner_radius=5)
    opd_title.grid(row=17, column=0, padx=10, pady=1, columnspan=2, sticky="new")

    # create a label for the one-pedal-drive system
    opd_mode_label = new_label(scrollable_frame, 18, 0, "One Pedal Drive mode:")

    def opd_mode_var_update():
        if opd_mode_variable.get() == True:
            offset_label.grid(row=19, column=0, padx=10, pady=1, sticky="w")
            offset_entry.grid(row=19, column=1, padx=10, pady=1, sticky="e")
            max_opd_brake_label.grid(row=20, column=0, padx=10, pady=1, sticky="w")
            max_opd_brake_entry.grid(row=20, column=1, padx=10, pady=1, sticky="e")
            refresh_live_visualization()
        else:
            offset_label.grid_forget()
            offset_entry.grid_forget()
            max_opd_brake_label.grid_forget()
            max_opd_brake_entry.grid_forget()
            refresh_live_visualization()

    opd_mode_variable = ctk.BooleanVar(value=True)
    opd_mode_checkbutton = new_checkbutton(scrollable_frame, 18, 1, opd_mode_variable, command=opd_mode_var_update)

    offset_label = new_label(scrollable_frame, 19, 0, "  Offset:")
    offset_variable = ctk.DoubleVar(value=0.2)
    temp_offset_variable = ctk.DoubleVar(value=0.2)
    offset_entry = new_entry(scrollable_frame, 19, 1, offset_variable, temp_offset_variable, command=refresh_live_visualization, max_value=0.5, min_value=0)

    max_opd_brake_label = new_label(scrollable_frame, 20, 0, "  Max OPD brake:")
    max_opd_brake_variable = ctk.DoubleVar(value=0.03)
    temp_max_opd_brake_variable = ctk.DoubleVar(value=0.03)
    max_opd_brake_entry = new_entry(scrollable_frame, 20, 1, max_opd_brake_variable, temp_max_opd_brake_variable, command=refresh_live_visualization, max_value=0.2, min_value=0)

    gas_exponent_label = new_label(scrollable_frame, 21, 0, "Gas exponent:")
    gas_exponent_variable = ctk.DoubleVar(value=2)
    temp_gas_exponent_variable = ctk.DoubleVar(value=2)
    gas_exponent_entry = new_entry(scrollable_frame, 21, 1, gas_exponent_variable, temp_gas_exponent_variable, command=refresh_live_visualization, max_value=2.5, min_value=0.5)

    brake_exponent_label = new_label(scrollable_frame, 22, 0, "Brake exponent:")
    brake_exponent_variable = ctk.DoubleVar(value=2)
    temp_brake_exponent_variable = ctk.DoubleVar(value=2)
    brake_exponent_entry = new_entry(scrollable_frame, 22, 1, brake_exponent_variable, temp_brake_exponent_variable, command=refresh_live_visualization, max_value=2.5, min_value=0.5)




    # list of implemented libraries shown as a discription
    implemented_libraries_label = ctk.CTkLabel(scrollable_frame, text="Implemented libraries:",  font=("Segoe UI", 11), text_color="#606060", fg_color="transparent", corner_radius=5, height=0)
    implemented_libraries_label.grid(row=46, column=0, padx=10, pady=(10,0), columnspan=2, sticky="new")

    SCSController_label = ctk.CTkLabel(scrollable_frame, text="SCSController - tumppi066",  font=("Segoe UI", 11), text_color="#606060", fg_color="transparent", corner_radius=5, height=0)
    SCSController_label.grid(row=47, column=0, padx=10, pady=0, columnspan=2, sticky="new")

    pygame_label = ctk.CTkLabel(scrollable_frame, text="pygame - pygame",  font=("Segoe UI", 11), text_color="#606060", fg_color="transparent", corner_radius=5, height=0)
    pygame_label.grid(row=48, column=0, padx=10, pady=0, columnspan=2, sticky="new")

    truck_telemetry_label = ctk.CTkLabel(scrollable_frame, text="Truck telemetry - Dreagonmon",  font=("Segoe UI", 11), text_color="#606060", fg_color="transparent", corner_radius=5, height=0)
    truck_telemetry_label.grid(row=49, column=0, padx=10, pady=0, columnspan=2, sticky="new")

    customtkinter_label = ctk.CTkLabel(scrollable_frame, text="customtkinter - csm10495",  font=("Segoe UI", 11), text_color="#606060", fg_color="transparent", corner_radius=5, height=0)
    customtkinter_label.grid(row=50, column=0, padx=10, pady=0, columnspan=2, sticky="new")




    confirmation = False
    def reset_button_action():
        global confirmation
        #deletes the save file for rebuild after a program restart after a confirmation
        #restarts the program
        if confirmation == True:
            try:
                os.remove(os.path.join(os.path.dirname(os.path.abspath(__file__)), "saves.json"))
                reset_button.configure(text="restarting program...", hover_color="#333366")
                cmd_print("settings reset, restarting program...")
                
                on_root_close()
                
            except:
                pass
            try:
                os.system("MonoCruise.exe")
            except:
                try:
                    os.system("MonoCruise.py")
                except:
                    pass
            os.system("exit")
        else:
            reset_button.configure(text="are you sure?", hover_color="#333366")
            confirmation = True

    # create a reset button
    reset_button = ctk.CTkButton(scrollable_frame, text="reset all settings", font=default_font, text_color="lightgrey", fg_color=WAITING_COLOR, corner_radius=5, hover_color="#333366", command=reset_button_action)
    reset_button.grid(row=50, column=0, padx=10, pady=(20,1), columnspan=2, sticky="ew")

    # discription for the reset button
    reset_button_description = ctk.CTkLabel(scrollable_frame, text="this requires a program restart", font=("Segoe UI", 11), text_color="#606060", fg_color="transparent", corner_radius=5, bg_color="transparent", anchor="center", wraplength=165, height=10)
    reset_button_description.grid(row=51, column=0, padx=10, pady=(0,4), columnspan=2, sticky="nsew")





    #end of the settings

    #create image for live visualization from the python code
    title_frame = ctk.CTkFrame(main_frame, fg_color=DEFAULT_COLOR)
    title_frame.pack(side="top", padx=0, pady=0, fill="x")

    live_title = ctk.CTkLabel(
        title_frame, text="Live visualization", font=("Segoe UI", 15, "bold"), 
        text_color="lightgrey", fg_color=DEFAULT_COLOR, corner_radius=5
    )
    live_title.pack(side="right", padx=10, pady=1, fill="x", expand=True)

    cmd_label = ctk.CTkLabel(main_frame, text="Command", font=code_font, text_color=CMD_COLOR, fg_color="#202020", corner_radius=5, anchor="w")
    cmd_label.pack(side="bottom", padx=5, pady=5, fill="x")


    live_visualization_frame = ctk.CTkLabel(main_frame, text="", fg_color="transparent", corner_radius=0, width=200, height=200)
    live_visualization_frame.pack(side="top", padx=30, pady=30, fill="x", anchor="n")


    # Load and create CTkImages for buttons
    patreon_img_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "patreon.png")
    youtube_img_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "youtube.png")
    gear_img_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "gear.png")
    
    support_img = ctk.CTkImage(
        light_image=Image.open(patreon_img_path),
        dark_image=Image.open(patreon_img_path),
        size=(int((Image.open(patreon_img_path).width / Image.open(patreon_img_path).height) * 15), 15)
    )

    youtube_img = ctk.CTkImage(
        light_image=Image.open(youtube_img_path),
        dark_image=Image.open(youtube_img_path),
        size=(int((Image.open(youtube_img_path).width / Image.open(youtube_img_path).height) * 15), 15)
    )

    gear_icon = ctk.CTkImage(
        light_image=Image.open(gear_img_path),
        dark_image=Image.open(gear_img_path),
        size=(int((Image.open(gear_img_path).width / Image.open(gear_img_path).height) * 25), 25)
    )

    support_button = ctk.CTkButton(
        button_frame, 
        text="Support", 
        image=support_img,
        font=("Segoe UI", 10),
        text_color="black", 
        fg_color="white", 
        hover_color="lightgrey", 
        command=lambda: os.system("start https://www.patreon.com/c/lukasdeschryver"), 
        compound="left", 
        height=20
    )
    support_button.pack(side="left", padx=5)

    youtube_button = ctk.CTkButton(
        button_frame, 
        text="YouTube", 
        image=youtube_img,
        font=("Segoe UI", 11),
        text_color="black", 
        fg_color="white", 
        hover_color="lightgrey",
        command=lambda: os.system("start https://www.youtube.com/@ld-tech_org"), 
        compound="left", 
        height=20
    )
    youtube_button.pack(side="left", padx=5)

    def hide_button_action():
        button_frame.pack_forget()
        save_variables(os.path.join(os.path.dirname(os.path.abspath(__file__)), "saves.json"), hide_button_action = True)

    hide_button = ctk.CTkButton(
        button_frame, 
        text="X", 
        font=("Segoe UI", 11),
        text_color="darkgray", 
        fg_color="transparent",
        command=hide_button_action,
        width=20, 
        height=20, 
        hover=False
    )
    hide_button.pack(side="left", padx=0, pady=0)

    def settings_icon_action():
        if settings_frame.winfo_width() <= 2:
            settings_frame.configure(width=settings_frame_width, fg_color="#333333")
            settings_frame.pack(padx=(5,0))
        else:
            settings_frame.configure(width=1, fg_color="transparent")
            settings_frame.pack(padx=0)

    #gear icon button next to the settings frame
    #for action: **hide** the settings frame if it's visible, otherwise **show** it. also print the width of the settings frame to the console
    gear_icon_button = ctk.CTkButton(
        title_frame,
        text="",
        font=("Segoe UI", 2),
        image=gear_icon,
        command=settings_icon_action,
        width=0,
        height=0,
        hover=False,
        corner_radius=0,
        fg_color="transparent",
        bg_color="transparent",
        anchor="w",
        border_width=0
    )
    gear_icon_button.pack(side="top", padx=0, pady=5, anchor="nw")




    # Create a class to hold UI state
    class UIState:
        def __init__(self):
            self.last_mode = None  # new attribute to track current mode ("setup", "waiting", "running")
            self.current_color = WAITING_COLOR  # start off with a waiting color (e.g. before setup is complete)
            self.last_ets2_state = False
            self.last_device = device
            self.transition_active = False
            self.transition_start_time = 0
            self.transition_progress = 0
            self.transition_from_color = WAITING_COLOR
            self.transition_to_color = CONNECTED_COLOR
            self.transition_after_id = None
            self.last_frame_time = 0

    ui_state = UIState()

    def hex_to_rgb(hex_color):
        """Convert hex color to RGB tuple"""
        hex_color = hex_color.lstrip('#')
        return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
            
    def rgb_to_hex(rgb):
        """Convert RGB tuple to hex color"""
        return '#{:02x}{:02x}{:02x}'.format(int(rgb[0]), int(rgb[1]), int(rgb[2]))
            
    def interpolate_color(color1, color2, factor):
        """Interpolate between two colors with a factor (0-1)"""
        rgb1 = hex_to_rgb(color1)
        rgb2 = hex_to_rgb(color2)
            
        result = tuple(rgb1[i] + factor * (rgb2[i] - rgb1[i]) for i in range(3))
        return rgb_to_hex(result)
            
    def update_transition():
        """Update the color transition animation"""
        if not ui_state.transition_active:
            return
            
        current_time = time.time()
        elapsed = current_time - ui_state.transition_start_time
        # Calculate frame time (if available)
        frame_time = current_time - ui_state.last_frame_time if hasattr(ui_state, 'last_frame_time') else 0
        ui_state.last_frame_time = current_time
            
        if elapsed >= TRANSITION_DURATION:
            # Transition complete
            banner_frame.configure(fg_color=ui_state.transition_to_color)
            ui_state.transition_active = False
            ui_state.transition_after_id = None
            return
            
        # Calculate progress (0 to 1)
        progress = elapsed / TRANSITION_DURATION
        ui_state.transition_progress = progress
            
        # Calculate current color via interpolation
        current_color = interpolate_color(
            ui_state.transition_from_color,
            ui_state.transition_to_color,
            progress
        )
            
        # Update banner color
        banner_frame.configure(fg_color=current_color)
            
        # Calculate adaptive delay to hit the target framerate
        target_frame_time = 1000 / TRANSITION_FRAMERATE  # milliseconds
        actual_frame_time = frame_time * 1000  # convert seconds to milliseconds
        adaptive_delay = max(1, int(target_frame_time - actual_frame_time))
            
        # Schedule the next frame update for the transition
        ui_state.transition_after_id = root.after(adaptive_delay, update_transition)
            
    def start_color_transition(from_color, to_color):
        """Start a color transition animation"""
        # Cancel any ongoing transition if necessary
        if ui_state.transition_active and ui_state.transition_after_id:
            root.after_cancel(ui_state.transition_after_id)
        ui_state.transition_active = True
        ui_state.transition_start_time = time.time()
        ui_state.last_frame_time = ui_state.transition_start_time
        ui_state.transition_progress = 0
        ui_state.transition_from_color = from_color
        ui_state.transition_to_color = to_color
            
        # Start update_cycle for the transition
        update_transition()
            
    def update_ui_state():
        if exit_event.is_set():
            on_root_close()

        """Update UI state based on device, gasaxis, brakeaxis, and ETS2 connection status"""
        global device, gasaxis, brakeaxis

        # ---------------------------
        # SETUP MODE: if either gasaxis or brakeaxis equals 0,
        # display "finish setup in settings" and use WAITING_COLOR.
        # ---------------------------
        if gasaxis == 0 or brakeaxis == 0 or device == 0:
            current_mode = "setup"
            target_color = WAITING_COLOR

            if ui_state.last_mode != current_mode:
                start_color_transition(ui_state.current_color, target_color)
                ui_state.current_color = target_color
                ui_state.last_mode = current_mode

            # Stop any dots animation in setup mode.
            if dots_anim.is_playing:
                dots_anim.stop()

            # Update the label text directly, as no animation is used in setup mode.
            loading_label.configure(
                text="finish setup in settings",
                font=default_font,
                text_color="white",
                bg_color="transparent"
            )
            root.after(30, update_ui_state)
            return

        # ---------------------------
        # WAITING vs RUNNING MODE:
        # When gasaxis and brakeaxis are nonzero, check ETS2 connection.
        # ---------------------------
        if ets2_detected.is_set():
            current_mode = "running"
            target_color = CONNECTED_COLOR
        else:
            current_mode = "waiting"
            target_color = WAITING_COLOR

        # Begin a forced color transition if the mode changed.
        if ui_state.last_mode != current_mode:
            start_color_transition(ui_state.current_color, target_color)
            ui_state.current_color = target_color
            ui_state.last_mode = current_mode

        if current_mode == "running":
            # In running mode, ensure the dots animation is stopped.
            if dots_anim.is_playing:
                dots_anim.stop()
            loading_label.configure(
                text="Connected and running",
                font=default_font_bold,
                text_color="darkgray",
                bg_color="transparent"
            )
        elif current_mode == "waiting":
            # In waiting mode, start the dots animation if it isn't running.
            if not dots_anim.is_playing:
                dots_anim.start()
            # Update the other label attributes without modifying the text,
            # so the dots animation can continue to update it.
            loading_label.configure(
                font=default_font,
                text_color="white",
                bg_color="transparent"
            )

        ui_state.last_device = device

        if not exit_event.is_set():
            root.after(30, update_ui_state)

    def on_root_close():

        cmd_print("UI closing, cleaning up threads...")

        # Stop the dots animation if it's still running
        if dots_anim and dots_anim.is_playing:
            dots_anim.stop()
        # Set the exit event to stop all threads
        exit_event.set()
        # Clean up pygame
        if pygame.get_init():
            pygame.quit()

        if ets2_detected.is_set():
            send(0,0, controller)
            setattr(controller, "wipers3", False)
            setattr(controller, "wipers4", False)
            setattr(controller, "accmode", False)

            controller.close()
        root.destroy()

    #set variables from the save file
    
    debug_mode = ctk.BooleanVar(value=False)
    try:
        bar_variable.set(_data_cache["bar_variable"])
    except Exception: pass
    try:
        gas_exponent_variable.set(_data_cache["gas_exponent_variable"])
        temp_gas_exponent_variable.set(_data_cache["gas_exponent_variable"])
    except Exception: pass
    try:
        brake_exponent_variable.set(_data_cache["brake_exponent_variable"])
        temp_brake_exponent_variable.set(_data_cache["brake_exponent_variable"])
    except Exception: pass
    try:
        max_opd_brake_variable.set(_data_cache["max_opd_brake_variable"])
        temp_max_opd_brake_variable.set(_data_cache["max_opd_brake_variable"])
    except Exception: pass
    try:
        offset_variable.set(_data_cache["offset_variable"])
        temp_offset_variable.set(_data_cache["offset_variable"])
    except Exception: pass
    try:
        polling_rate.set(_data_cache["polling_rate"])
        temp_polling_rate.set(_data_cache["polling_rate"])
    except Exception: pass
    try:
        opd_mode_variable.set(_data_cache["opd_mode_variable"])
    except Exception: pass
    try:
        hazards_variable.set(_data_cache["hazards_variable"])
    except Exception: pass
    try:
        autodisable_hazards.set(_data_cache["autodisable_hazards"])
    except Exception: pass
    try:
        horn_variable.set(_data_cache["horn_variable"])
    except Exception: pass
    try:
        airhorn_variable.set(_data_cache["airhorn_variable"])
    except Exception: pass
    try:
        debug_mode.set(_data_cache["debug_mode"])
    except Exception: pass
    try:
        autostart_variable.set(_data_cache["autostart_variable"])
    except Exception: pass

    try:
        if _data_cache["hide_button_action"] == True:
            button_frame.pack_forget()
    except Exception: pass

    # Bind the closing event
    root.protocol("WM_DELETE_WINDOW", on_root_close)
    
    # Start UI immediately
    loading_label.configure(text="Waiting for ETS2")
    dots_anim.start()
    
    # Start the game thread
    thread_game = threading.Thread(target=game_thread, daemon=True)
    thread_game.start()
    
    # Signal that UI is ready
    ui_ready.set()
    
    # Start UI update cycle - more frequent updates for responsiveness
    update_ui_state()
    opd_mode_var_update()

    settings_frame.configure(width=1, fg_color="transparent")
    settings_frame.pack(padx=0)
    
    cmd_print("Starting mainloop...")
    # Start the mainloop in the main thread
    root.mainloop()
    
    # When mainloop exits, set exit event
    exit_event.set()
    cmd_print("Mainloop exited, cleaning up...")
    
    # Clean up pygame
    if pygame.get_init():
        pygame.quit()

    controller.close()
        
    sys.exit(0)
except Exception as e:
    context = get_error_context()
    log_error(e, context)
    exit_event.set()
    try:
        if pygame.get_init():
            pygame.quit()
        controller.close()
        sys.exit(1)
    except:
        pass