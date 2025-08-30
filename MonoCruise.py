import threading
import customtkinter as ctk
import tkinter as tk
version = "v1.0.1"

# temporary for radar output
import cv2

from PIL import Image, ImageDraw, ImageFont, ImageTk
import sys
import ctypes
import os
import pygame
import json
import time
import keyboard
import numpy as np
import traceback
import inspect
from datetime import datetime
from functools import lru_cache
import psutil
import subprocess
import winreg
from CTkMessagebox import CTkMessagebox
sys.path.append('./_internal')
from scscontroller import SCSController
from ETS2radar.ETS2radar import ETS2Radar



from connect_SDK import check_ets2_sdk, check_ats_sdk, check_and_install_scs_sdk

import truck_telemetry
try:
    truck_telemetry.init()  # Signals if ETS2 SDK has been detected
except:
    if not check_ets2_sdk() and not check_ats_sdk():
        print("SDK not installed, trying to install it automatically.")
        msg = CTkMessagebox(title="SDK not installed", message='Do you want to install the SDK automatically?',
            icon="warning", option_1="Cancel",option_2="Okay", wraplength=300, sound=True)
        msg.get()
        if msg.get() == "Okay":
            try:
                # Attempt to install the SDK automatically
                print("installing SDK")
                result = check_and_install_scs_sdk()
                if result["summary"]["failed_installs"] > 0:
                    raise Exception("SDK installation failed")
                
                if result is None:
                    raise Exception("Error checking or installing SDK")
                print("installed SDK")
            except:
                raise Exception("Error installing SDK")
        else:
            exit()

# Get system DPI scaling
try:
    ctypes.windll.shcore.SetProcessDpiAwareness(1)
    scaling = ctypes.windll.user32.GetDpiForSystem() / 96.0
except Exception as e:
    scaling = 1
    print(e)
    
# Create the main window
root = ctk.CTk()

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
close_buttons_threads = threading.Event()  # Event to signal button threads to close
close_bar_event = threading.Event()
ets2_detected = threading.Event()  # Event for ETS2 detection
ui_ready = threading.Event()       # Event to signal UI is ready

# colors
WAITING_COLOR = "#1f538d"
KEY_ON_COLOR = "#1f53FF"
CONNECTED_COLOR = "#304230"
LOST_COLOR = "#FF0000"
DEFAULT_COLOR = "#2B2B2B"
VAR_LABEL_COLOR = "#2B2B2B"
SETTINGS_COLOR = "#454545"
SETTING_HEADERS_COLOR = "#454545"
DISABLED_COLOR = "#F1F1F1"
CMD_COLOR = "#808080"
SPEEDLIMITER_COLOR = "#008B00"
CRUISECONTROL_COLOR = "#4876FF"

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
        try:
            j = pygame.joystick.Joystick(i)
        except:
            continue
        j.init()
        if hasattr(j, "get_guid") and str(j.get_guid()) == uuid_str:
            return j
    return None

def add_to_startup(exe_path: str, app_name: str):
    """
    Adds the specified executable to Windows startup.

    Parameters:
    - exe_path: Full path to the .exe file
    - app_name: Name of the application (used as the registry key name)
    """
    if not os.path.isfile(exe_path):
        raise FileNotFoundError(f"The file '{exe_path}' does not exist.")

    try:
        # Access the registry key for the current user's startup
        key = winreg.OpenKey(
            winreg.HKEY_CURRENT_USER,
            r"Software\Microsoft\Windows\CurrentVersion\Run",
            0,
            winreg.KEY_SET_VALUE
        )
        # Set the value; this adds the app to startup
        winreg.SetValueEx(key, app_name, 0, winreg.REG_SZ, exe_path)
        winreg.CloseKey(key)
        print(f"'{app_name}' has been added to startup.")
    except Exception as e:
        print(f"Failed to add to startup: {e}")

def remove_from_startup(app_name: str):
    """
    Removes the specified application from Windows startup for the current user.

    Parameters:
    - app_name: The registry value name you used when adding the app (e.g. "MyCoolApp").
    """
    try:
        key = winreg.OpenKey(
            winreg.HKEY_CURRENT_USER,
            r"Software\Microsoft\Windows\CurrentVersion\Run",
            0,
            winreg.KEY_SET_VALUE
        )
        # Delete the registry value; raises FileNotFoundError if it doesn't exist
        winreg.DeleteValue(key, app_name)
        winreg.CloseKey(key)
        print(f"'{app_name}' has been removed from startup.")
    except FileNotFoundError:
        print(f"No startup entry named '{app_name}' was found.")
    except PermissionError:
        print("Permission denied: you might need to run this with elevated privileges.")
    except Exception as e:
        print(f"Failed to remove from startup: {e}")

def refresh_button_labels():
    global device
    global cc_dec_button
    global cc_inc_button
    global cc_start_button
    global cc_start_label
    global cc_inc_label
    global cc_dec_label

    if device is not None and device.get_init():
        if cc_start_button is not None:
            cc_dec_label.configure(text=format_button_text(cc_dec_button), text_color="lightgrey")
        else:
            cc_dec_label.configure(text="None", text_color=SETTINGS_COLOR)
        if cc_inc_button is not None:
            cc_inc_label.configure(text=format_button_text(cc_inc_button), text_color="lightgrey")
        else:
            cc_inc_label.configure(text="None", text_color=SETTINGS_COLOR)
        if cc_start_button is not None:
            cc_start_label.configure(text=format_button_text(cc_start_button), text_color="lightgrey")
        else:
            cc_start_label.configure(text="None", text_color=SETTINGS_COLOR)

def check_wheel_connected():
    global device, recovered

    """
    Checks if a wheel is connected and tries to init from the recovered joystick if available.
    
    Args:
        device (pygame.joystick.Joystick): The joystick object to check.
        recovered (pygame.joystick.Joystick, optional): A previously recovered joystick object from the save file.
        
    Returns:
        bool: True if the wheel is connected, False otherwise.
    """
    if device is not None:
        try:
            device.get_instance_id()
            return True
        except:
            if recovered is not None:
                device = None
            cmd_print("Error: couldn't find pedals", "#FF2020", 10)
            return False
    else:
        if recovered is not None:
            try:
                recovered.get_instance_id()
                device = recovered
                device.init()
                device.get_instance_id()
                refresh_button_labels()
                return True
            except:
                return False
        else:
            return False

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
    global _data_cache, device, device_instance_id, debug_mode, recovered, device_lost
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
        if not check_wheel_connected():
            print("Joystick with saved UUID not found among connected devices.")
            device_lost = True
        else:
            device_instance_id = device.get_instance_id()
            print(f"Recovered joystick: {device.get_name()}")
    else:
        recovered = None
        device = None
        device_instance_id = 0
        device_lost = False
        _data_cache["device"] = None  # Ensure device is None if not found to avoid KeyErrors
            
    
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
            time.sleep(0.0005)
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        pass
    
    if should_be_running and not is_running:
        # Should be running but isn't - start it
        try:
            _running_process = subprocess.Popen([exe_path])
            cmd_print(f"Started {exe_name}", display_duration=2)
        except Exception as e:
            cmd_print(f"Failed to start {exe_name}: {e}", display_duration=3)
        finally:
            add_to_startup(exe_path, "ETS2_checker_MonoCruise")
    
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
            remove_from_startup("ETS2_checker_MonoCruise")
    
def save_variables(filename, **kwargs):
    """
    Saves given variables to a JSON file while preserving any existing values.
    Runs check_and_start_exe() only when autostart_variable flips state.
    args:
        filename (str): The name of the JSON file to save variables to.
        **kwargs: Key-value pairs of variables to save. If a key already exists, its value will be updated.
    """
    global _data_cache

    # If no new data, just return existing cache
    if not kwargs:
        return load_variables(filename)

    # Load existing data on first call
    if _data_cache is None:
        _data_cache = load_variables(filename)

    # Normalize device entry
    if "device" in _data_cache and not isinstance(_data_cache["device"], str):
        _data_cache["device"] = serialize_joystick(_data_cache["device"])

    # Capture old autostart value
    old_autostart = _data_cache.get("autostart_variable")

    # Merge new values into a copy
    merged_data = _data_cache.copy()
    for key, value in kwargs.items():
        if key == "device":
            merged_data[key] = serialize_joystick(value) if value is not None else None
        else:
            merged_data[key] = value

    # Only write file and trigger exe logic when something actually changed
    if merged_data != _data_cache:
        _data_cache = merged_data
        with open(filename, "w") as file:
            json.dump(merged_data, file, indent=4)

        cmd_print(f"Changes detected. Variables saved to '{filename}'.", display_duration=1)

        # Compare old vs new autostart state
        if "autostart_variable" in kwargs:
            new_autostart = merged_data.get("autostart_variable")
            if old_autostart != new_autostart:
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
        #cmd_label.after(0, lambda: cmd_label.configure(text=txt, text_color=original_color))
        if exit_event.is_set():
            return
        try:
            cmd_label.configure(text=txt, text_color=original_color)
        except:
            pass

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
    threading.Thread(target=fade_and_clear, args=(local_fade_id, text), daemon=True, name="fade cmd_print()").start()

def get_button(button):
    """
    Returns the current state/value of the button stored in button index.
    For regular buttons: returns True if pressed, False if released.
    For hat directions: returns True if hat is in that direction, False otherwise.
    
    Returns:
        bool or None: Current state of the button at button index, or None if button is None or device unavailable
    """
    global device
    
    if isinstance(button, str):
        return keyboard.is_pressed(button)
    
    if button is None or device is None or not device.get_init() or device_lost:
        return None
    
    button_count = device.get_numbuttons()
    
    # Regular button
    if button < button_count:
        return device.get_button(button)
    
    # Hat direction - convert back to hat index and direction
    hat_button_index = button - button_count
    hat_index = hat_button_index // 4
    direction_index = hat_button_index % 4
    
    if hat_index >= device.get_numhats():
        return None
    
    hat_x, hat_y = device.get_hat(hat_index)
    
    # Check if hat is currently in the direction that button represents
    if direction_index == 0:    # Up
        return hat_y == 1
    elif direction_index == 1:  # Right
        return hat_x == 1
    elif direction_index == 2:  # Down
        return hat_y == -1
    elif direction_index == 3:  # Left
        return hat_x == -1
    
    return False

def detect_joystick_movement(button_type="None given"):
    """
    Detects any movement (state changes) on the global joystick device or keyboard presses.
    Triggers on button presses (0 -> 1) AND releases (1 -> 0).
    Triggers on any hat position changes.
    Triggers on keyboard key presses (character keys only, no function keys).
    Regular buttons are indexed normally (0, 1, 2, etc.)
    Hat directions are mapped as:
    - Hat Up: button_count + (hat_index * 4) + 0
    - Hat Right: button_count + (hat_index * 4) + 1  
    - Hat Down: button_count + (hat_index * 4) + 2
    - Hat Left: button_count + (hat_index * 4) + 3
    
    Keyboard keys are stored as their Keyboard name capitalized.
    
    Updates global variable given as input with the detected input.
    """
    global device, close_buttons_threads, unassign, exit_event, device_lost, SETTINGS_COLOR
    global unassign_button, cc_start_label, cc_inc_label, cc_dec_label
    global cc_start_button, cc_inc_button, cc_dec_button

    button = None
    input_type = None
    input_value = None
    _prev_button_states = []
    _prev_hat_states = []
    key_pressed = threading.Event()
    detected_key = None
    should_exit = threading.Event()

    # Configure UI based on button type
    if button_type == "start":
        cc_start_label.configure(border_color="green")
        cmd_print("Waiting for start button press...", display_duration=1)
    elif button_type == "inc":
        cc_inc_label.configure(border_color="green")
        cmd_print("Waiting for increase button press...", display_duration=1)
    elif button_type == "dec":
        cc_dec_label.configure(border_color="green")
        cmd_print("Waiting for decrease button press...", display_duration=1)
    else:
        raise ValueError(f"Unknown button type: {button_type}")

    def on_key_press(event):
        nonlocal detected_key, key_pressed
        # Check exit conditions
        if (close_buttons_threads.is_set() or exit_event.is_set() or 
            device_lost or unassign or should_exit.is_set()):
            return
            
        button = event.name.capitalize()
        detected_key = button
        if check_key(button, button_type):
            key_pressed.set()

    def check_key(button, button_type):
        global cc_start_button, cc_inc_button, cc_dec_button
        if ((button_type == "start" and (cc_inc_button == button or cc_dec_button == button)) or
            (button_type == "inc" and (cc_start_button == button or cc_dec_button == button)) or
            (button_type == "dec" and (cc_start_button == button or cc_inc_button == button))):
            cmd_print(f'"{format_button_text(button)}" button cannot be used twice.', "#FF2020", 10)
            return False
        return True
    
    # Set up keyboard listener
    keyboard_hook = None
    try:
        keyboard_hook = keyboard.on_press(on_key_press)
    except Exception:
        pass

    # Main detection loop
    while (button is None and not close_buttons_threads.is_set() and 
           not exit_event.is_set() and not device_lost and 
           not unassign and not should_exit.is_set()):
        
        if device is None or not device.get_init():
            break
        
        # Check for keyboard input
        if key_pressed.is_set():
            if detected_key in ("Backspace", "Delete"):
                cmd_print("Unassigning button...")
                unassign_button.configure(state="normal")
                unassign = True
                break
            elif detected_key not in ("Esc", "Escape", "Enter", "Return"):
                button = detected_key
                input_type = "key"
                input_value = button
                break
            else:
                close_buttons_threads.set()
                break
        
        # Check joystick input
        if device is not None and device.get_init():
            try:
                button_count = device.get_numbuttons()
                hat_count = device.get_numhats()
            except Exception:
                break
            
            # Initialize previous states if needed
            if len(_prev_button_states) != button_count:
                _prev_button_states = [False] * button_count
            if len(_prev_hat_states) != hat_count:
                _prev_hat_states = [(0, 0)] * hat_count
            
            # Check for button state changes
            for i in range(button_count):
                if (close_buttons_threads.is_set() or exit_event.is_set() or 
                    device_lost or unassign or should_exit.is_set()):
                    button = "EXIT"
                    break
                try:
                    current_state = device.get_button(i)
                    if current_state != _prev_button_states[i]:
                        if check_key(i, button_type):
                            button = i
                            input_type = "button"
                            input_value = i
                            _prev_button_states[i] = current_state
                            break
                except Exception:
                    break
            
            if button == "EXIT" or button is not None:
                break
            
            # Check for hat state changes
            for hat_index in range(hat_count):
                if (close_buttons_threads.is_set() or exit_event.is_set() or 
                    device_lost or unassign or should_exit.is_set()):
                    button = "EXIT"
                    break
                try:
                    current_hat = device.get_hat(hat_index)
                    prev_hat = _prev_hat_states[hat_index]
                    
                    if current_hat != prev_hat:
                        hat_x, hat_y = current_hat
                        hat_button_base = button_count + (hat_index * 4)
                        
                        # Map hat directions to button indices
                        direction_map = {
                            (0, 1): (hat_button_base + 0, "up"),
                            (1, 0): (hat_button_base + 1, "right"),
                            (0, -1): (hat_button_base + 2, "down"),
                            (-1, 0): (hat_button_base + 3, "left")
                        }
                        
                        if current_hat in direction_map:
                            temp_button, direction = direction_map[current_hat]
                            if check_key(temp_button, button_type):
                                button = temp_button
                                input_type = "hat"
                                input_value = direction
                        
                        _prev_hat_states[hat_index] = current_hat
                        if button is not None:
                            break
                except Exception:
                    break
            
            if button == "EXIT":
                break

        time.sleep(0.005)

    # Cleanup
    should_exit.set()
    time.sleep(0.01)
    
    try:
        if keyboard_hook is not None:
            keyboard.unhook(keyboard_hook)
        else:
            keyboard.unhook_all()
    except Exception:
        pass
    
    try:
        keyboard.unhook_all()
    except Exception:
        pass

    unassign = False

    # Handle early exit conditions
    if (button == "EXIT" or device is None or not device.get_init() or 
        exit_event.is_set() or close_buttons_threads.is_set() or device_lost):
        close_buttons_threads.clear()
        return

    # Format display text based on input type
    if input_type == "key" and len(input_value) <= 2:
        display_text = f"key {input_value.capitalize()}"
    elif input_type == "key":
        display_text = input_value.capitalize()
    elif input_type == "button":
        display_text = f"button {int(input_value)}"
    elif input_type == "hat":
        display_text = f"hat {input_value}"
    else:
        display_text = "None"

    # Update UI based on button type
    text_color = "lightgrey" if display_text != "None" else SETTINGS_COLOR
    
    if button_type == "start":
        cc_start_button = button
        cc_start_label.configure(border_color=SETTINGS_COLOR, text=display_text, text_color=text_color)
    elif button_type == "inc":
        cc_inc_button = button
        cc_inc_label.configure(border_color=SETTINGS_COLOR, text=display_text, text_color=text_color)
    elif button_type == "dec":
        cc_dec_button = button
        cc_dec_label.configure(border_color=SETTINGS_COLOR, text=display_text, text_color=text_color)
    
    unassign_button.configure(state="disabled")
    close_buttons_threads.clear()

def connect_joystick():
    def capture_default_positions(joystick):
        """Capture the default (resting) positions of all axes for a joystick"""
        positions = {}
        for axis in range(joystick.get_numaxes()):
            if joystick.get_axis(axis) < -0.5:
                positions[axis] = -1
            elif joystick.get_axis(axis) >= -0.5:
                positions[axis] = 0
        return positions

    def check_axis_inversion(joystick, axis, default_position, current_position, threshold=0.5):
        """
        Check if an axis is inverted based on movement direction
        Returns True if inverted, False if normal
        """
        movement = default_position - current_position
        return movement < -threshold

    """Main game logic"""
    global exit_event
    global recovered
    global device
    global device_lost
    global device_instance_id
    global axis
    global gasaxis
    global brakeaxis
    global gas_inverted
    global brake_inverted
    global pygame
    global connected_joystick_label
    global connected_joystick_gas_axis_label
    global connected_joystick_brake_axis_label
    global restart_connection_label
    global pauze_pedal_detection

    if gasaxis != 0 or brakeaxis != 0:
        restart_connection_button.configure(text="reconnect to pedals")

    pauze_pedal_detection =True
    gasaxis = 0
    brakeaxis = 0
    if device is not None:
        try:
            device.quit()
        except:
            pass
    device = None
    recovered = None
    device_instance_id = 0
    gas_inverted = False
    brake_inverted = False
    default_axis_positions = {}
    joysticks = {}
    pygame.joystick.quit()
    time.sleep(0.1)

    connected_joystick_label.configure(text="None connected")
    connected_joystick_gas_axis_label.configure(text="None")
    connected_joystick_brake_axis_label.configure(text="None")
    pygame.joystick.init()
    
    try:
        # Wait for joystick input
        cmd_print("Waiting for joystick input...")
        restart_connection_label.configure(text="connecting...")
        while not exit_event.is_set():
            while not pygame.get_init():
                pygame.init()
            for event in pygame.event.get():
                if event.type == pygame.JOYDEVICEADDED:
                    try:
                        time.sleep(0.1)
                        joy = pygame.joystick.Joystick(event.device_index)
                    except Exception as e:
                        pygame.quit()
                        cmd_print("Failed to initialize joystick", "#FF2020", 30)
                        pygame.init()
                        pygame.joystick.init()
                        continue
                    joysticks[joy.get_instance_id()] = joy
                    
                    # Capture default positions when joystick connects
                    default_axis_positions[joy.get_instance_id()] = capture_default_positions(joy)
                    
                    if joy.get_name() != "vJoy Device":
                        cmd_print(f"Joystick connected: {joy.get_name()}")
                        cmd_print(f"Default positions captured for {joy.get_numaxes()} axes")
                    restart_connection_label.configure(text="tap the brake pedal")
                    
                elif event.type == pygame.JOYAXISMOTION:
                    if event.instance_id in joysticks and joysticks[event.instance_id].get_name() != "vJoy Device":
                        device = joysticks[event.instance_id]
                        device_instance_id = device.get_instance_id()

                        # Get default position for this axis
                        default_pos = default_axis_positions.get(event.instance_id, {}).get(event.axis, 0)
                        current_pos = event.value
                        
                        # Check for significant movement (threshold to avoid noise)
                        if abs(current_pos - default_pos) > 0.3:
                            if brakeaxis == 0:
                                brakeaxis = event.axis
                                # Check if brake axis is inverted
                                brake_inverted = check_axis_inversion(device, event.axis, default_pos, current_pos)
                                
                                cmd_print(f"Brake axis set to {brakeaxis}")
                                cmd_print(f"Brake axis inverted: {brake_inverted}")
                                connected_joystick_brake_axis_label.configure(text=f"axis {brakeaxis}")
                                restart_connection_label.configure(text="saving...")
                                time.sleep(0.5)
                                restart_connection_label.configure(text="tap the gas pedal")
                                
                            elif event.axis != brakeaxis:
                                gasaxis = event.axis
                                # Check if gas axis is inverted
                                gas_inverted = check_axis_inversion(device, event.axis, default_pos, current_pos)
                                
                                cmd_print(f"Gas axis set to {gasaxis}")
                                cmd_print(f"Gas axis inverted: {gas_inverted}")
                                connected_joystick_gas_axis_label.configure(text=f"axis {gasaxis}")
                                restart_connection_label.configure(text="saving...")
                                time.sleep(0.5)
                                restart_connection_label.configure(text="")
                                device.init()
                                break
                                
                elif event.type == pygame.JOYDEVICEREMOVED:
                    if event.instance_id in joysticks:
                        cmd_print(f"Joystick disconnected: {joysticks[event.instance_id].get_name()}")
                        # Clean up default positions for disconnected joystick
                        if event.instance_id in default_axis_positions:
                            del default_axis_positions[event.instance_id]
                        del joysticks[event.instance_id]
            
            if gasaxis != 0 and brakeaxis != 0:
                restart_connection_button.configure(text="reconnect to pedals")
                cmd_print("pedals connected")
                cmd_print(f"Final configuration - Gas: axis {gasaxis} (inverted: {gas_inverted}), Brake: axis {brakeaxis} (inverted: {brake_inverted})")
                break
                
            time.sleep(0.1)  # Small sleep to prevent CPU hogging
        # give the name of the joystick
        #make the name end with ... if the pixels available is smaller than the text
        if device is not None:
            if len(device.get_name()) > 20:
                connected_joystick_label.configure(text=f"{device.get_name()[:20]}...")
            else:
                connected_joystick_label.configure(text=f"{device.get_name()}")
            device_lost = False
            pauze_pedal_detection = False

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
        if not self.is_playing or exit_event.is_set():
            self.stop()
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
    # Canvas setup
    bg_color = hex_to_rgb(DEFAULT_COLOR)
    width, height = 400, 400
    img_pil = Image.new("RGB", (width, height), bg_color)
    draw = ImageDraw.Draw(img_pil)
    font = ImageFont.load_default(20)

    # Sample & compute
    num_points = 100
    xs = np.linspace(-1, 1, num_points)
    ys = []
    for x in xs:
        gas, brake = (x, 0) if x >= 0 else (0, -x)
        g_out, b_out = onepedaldrive(gas, brake)
        ys.append(g_out - b_out)

    y_min, y_max = min(ys), max(ys)
    if abs(y_max - y_min) < 1e-6:
        y_max = y_min + 1e-6

    # Coordinate mapper
    def to_img_coords(x, y):
        col = int((x + 1) / 2 * width)
        row = height - int((y - y_min) / (y_max - y_min) * height)
        return col, row

    # Blend function to mix colors with grey
    def blend_with_grey(color, grey, ratio=0.25):
        return tuple(int(ratio * c + (1 - ratio) * g) for c, g in zip(color, grey))

    # Define base colors
    pos_color = (50, 50, 225)
    neg_color = (255, 50, 50)
    grey = (100, 100, 100)

    # Blended axis colors
    pos_axis_color = blend_with_grey(pos_color, grey)
    neg_axis_color = blend_with_grey(neg_color, grey)

    # Draw axes with color split
    gx, gy = to_img_coords(0, 0)

    # Horizontal axis (X)
    draw.line([(0, gy), (gx, gy)], fill=neg_axis_color, width=2)  # Negative X
    draw.line([(gx, gy), (width - 1, gy)], fill=pos_axis_color, width=2)  # Positive X

    # Vertical axis (Y)
    draw.line([(gx, height - 1), (gx, gy)], fill=neg_axis_color, width=2)  # Negative Y
    draw.line([(gx, gy), (gx, 0)], fill=pos_axis_color, width=2)  # Positive Y

    # Axis labels
    draw.text((5, gy - 30), "pedals", fill=grey, font=font)
    draw.text((gx - 110, 5), "game input", fill=grey, font=font)

    # Plot curve segments
    for i in range(len(xs) - 1):
        x1, y1 = xs[i], ys[i]
        x2, y2 = xs[i+1], ys[i+1]
        p1, p2 = to_img_coords(x1, y1), to_img_coords(x2, y2)

        # decide color and split at y=0 if needed
        def draw_seg(a, b, color):
            draw.line([a, b], fill=color, width=2)

        if y1 >= 0 and y2 >= 0:
            draw_seg(p1, p2, pos_color)
        elif y1 < 0 and y2 < 0:
            draw_seg(p1, p2, neg_color)
        else:
            t = -y1 / (y2 - y1)
            xi = x1 + t * (x2 - x1)
            pi = to_img_coords(xi, 0)
            if y1 < 0:
                draw_seg(p1, pi, neg_color)
                draw_seg(pi, p2, pos_color)
            else:
                draw_seg(p1, pi, pos_color)
                draw_seg(pi, p2, neg_color)

    if return_result:
        return img_pil, to_img_coords

def overlay_dot_layer(x_value,
                      new_width=600,
                      new_height=600,
                      image=None,
                      to_img_coords=None):
    """
    Overlays a dot + label at x_value onto a PIL image,
    then resizes and wraps it in a CTkImage.
    """
    # Recompute y_val for the chosen x
    gas, brake = (x_value, 0) if x_value >= 0 else (0, -x_value)
    g_out, b_out = onepedaldrive(gas, brake)
    y_val = g_out - b_out

    # Draw on a copy
    pil_copy = image.copy()
    draw = ImageDraw.Draw(pil_copy)
    dot = to_img_coords(x_value, y_val)
    dot_color = (50, 50, 225) if y_val >= 0 else (255, 50, 50)
    r = 7  # radius
    bbox = [dot[0] - r, dot[1] - r, dot[0] + r, dot[1] + r]
    draw.ellipse(bbox, fill=dot_color)

    # Label it
    font = ImageFont.load_default(25)
    text = f"{round(y_val, 2)}"
    if dot[0]<345:
        draw.text((dot[0] + 7, min(dot[1], 369) + 5), text, fill=dot_color, font=font)
    else:
        draw.text((dot[0] - 60, max(dot[1], 30) - 35), text, fill=dot_color, font=font)


    # Resize & convert to CTkImage
    resized = pil_copy.resize((new_width, new_height), Image.LANCZOS)
    return ctk.CTkImage(resized, size=(new_width, new_height))

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
    global max_opd_brake_variable

    offset = offset_variable.get()
    brake_exponent = brake_exponent_variable.get()
    gas_exponent = gas_exponent_variable.get()
    opd_mode = opd_mode_variable.get()
    
    val1 = min(max(gasval, 0), 1)
    val2 = (min(max(brakeval, 0), 1)**brake_exponent)*-1
    sum_values = val1+val2

    if opd_mode == True:
        if sum_values<=offset:
            value = ((1)/(1+offset))*sum_values-((offset)/(1+offset))
        else:
            value = ((1)/(1-offset))*sum_values-((offset)/(1-offset))
    else:
        value = sum_values

    a = -(max_opd_brake_variable.get())**(1/brake_exponent)

    if sum_values>=0 and sum_values<=offset and opd_mode:
        value = (a/(offset**brake_exponent))*((-sum_values+offset)**brake_exponent)
    if sum_values<0 and opd_mode:
        value = interpolate(-1,a,sum_values,-1,0)

    gasval = max(0, value)
    brakeval = min(min(0, value),val2)*-1

    gasval = gasval**gas_exponent
    brakeval = brakeval**brake_exponent

    return gasval, brakeval

def send(a, b, controller):
    global bar_val
    global brake_exponent_variable
    global gas_exponent_variable
    global brakeval
    global gasval
    global gear
    global total_weight_tons
    global weight_adjustment
    global gas_output
    global brake_output
    global exit_event
    global data

    if exit_event.is_set():
        setattr(controller, "aforward", 0.0)
        setattr(controller, "abackward", 0.0)
        return

    if weight_adjustment.get():
        try:
            wheight_exp = (0.27*((total_weight_tons-8.93)/(11.7))+1)
        except:
            wheight_exp = 1
            if not device_lost:
                cmd_print("Error calculating weight adjustment", "#FF2020", 10)
    else:
        wheight_exp = 1
    b = b*wheight_exp

    bar_val = gas_output-brake_output
    try:
        if gear == 0:
            a = gasval**gas_exponent_variable.get() #for those people that like revvvving that engine
    except:
        a = gasval
    b = max(b,brakeval**brake_exponent_variable.get())
 
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

def format_button_text(button_index):
    """
    Formats the display text for a button based on its index.
    
    Args:
        button_index: The button index as stored by detect_joystick_movement()
                     - Keyboard keys: 10000 + hash(key_name) (e.g., 10000+, stored with key name)
                     - Joystick buttons: 0, 1, 2, etc.
                     - Hat directions: button_count + (hat_index * 4) + direction_offset
    
    Returns:
        str: Formatted text in the format "type value"
             - Keys: "key a", "key space", etc.
             - Buttons: "button 0", "button 1", etc.
             - Hats: "hat up", "hat right", etc.
    """
    global device
    
    if button_index is None:
        return "None"
    
    # Check if it's a keyboard key (10000+ range)
    if isinstance(button_index, str) :
        # For keyboard keys, we need to store the key name separately
        # This is a limitation - we'll need to modify the storage system
        # For now, return a generic keyboard indicator
        if len(button_index) <= 2:
            return f"key {button_index}"
        else:
            return str(button_index)
    
    # If we have a joystick device, check if it's a hat or regular button
    if device is not None and device.get_init():
        try:
            button_count = device.get_numbuttons()
            
            # If the index is within regular button range
            if button_index < button_count:
                return f"button {button_index}"
            
            # If the index is beyond regular buttons, it's a hat direction
            elif button_index >= button_count:
                hat_count = device.get_numhats()
                if hat_count > 0:
                    # Calculate which hat and direction
                    hat_offset = button_index - button_count
                    hat_index = hat_offset // 4
                    direction_offset = hat_offset % 4
                    
                    # Make sure this is a valid hat index
                    if hat_index < hat_count:
                        directions = ["up", "right", "down", "left"]
                        direction = directions[direction_offset]
                        return f"hat {direction}"
            
        except Exception as e:
            print(f"Error formatting button text: {e}")
    
    # Fallback for when device is not available or index doesn't match expected patterns
    # Assume it's a regular button if it's a small number
    if button_index < 100:  # Arbitrary threshold for likely button indices
        return f"button {button_index}"
    else:
        return f"Unknown input ({button_index})"

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

def is_process_running(process_name):
    try:
        i=0
        # Iterate through all running processes
        for proc in psutil.process_iter(['name']):
            try:
                if proc.info['name'] == process_name:
                    i+=1
            except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                continue
            time.sleep(0.001)

        if process_name == "MonoCruise.exe":
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

def sdk_check_thread():
    global autostart_variable
    global root
    global device_lost
    if autostart_variable and not device_lost:
        root.iconify()
    """Background thread to check for ETS2 SDK connection"""
    time.sleep(0.2)
    cmd_print("SDK check thread starting...")
    first = True
    manual_start = False
    while not exit_event.is_set():
        try:
            if first:
                print(f"starting in {'manual' if manual_start else 'auto'} start mode")
            while not exit_event.is_set():
                # Try to get data to check if SDK is still connected
                truck_telemetry.init()
                data = truck_telemetry.get_data()
                if not data["sdkActive"]:  # Check if SDK is still active
                    raise Exception("SDK_NOT_ACTIVE")
                ets2_detected.set()
                time.sleep(0.5)  # Small sleep to prevent CPU hogging
                first = False
        except Exception as e:
            if isinstance(e, FileNotFoundError) or str(e) == "Not support this telemetry sdk version" or str(e) == "SDK_NOT_ACTIVE":
                #print("ETS2 not found, please start the game first.")
                ets2_detected.clear()
                if first:
                    if not ets2_detected.is_set():
                        manual_start = True
                        root.deiconify()

                if autostart_variable.get()==True and not first and not ets2_detected.is_set() and not manual_start and root.state() == 'iconic':
                    print("shutting down")
                    time.sleep(1)
                    exit_event.set()
                elif not manual_start:
                    manual_start = True

                time.sleep(0.2)
            else:
                print(e)
                context = get_error_context()
                log_error(e, context)
        if first:
            print(f"starting in {'manual' if manual_start else 'auto'} start mode")
            first=False

def calc_truck_weight(data):
    # Extract weight information
    trailer_mass = data.get('unitMass', 0)  # trailer weight in kg
    cargo_mass = data.get('cargoMass', 0)  # Cargo weight in kg
    
    # Get trailer information from trailer array
    trailers = data.get('trailer', [])
    trailer_attached = False
    unit_mass = 8500 + data.get('fuel', 300)
    
    if trailers and len(trailers) > 0:
        first_trailer = trailers[0]
        trailer_attached = first_trailer.get('attached', False)
    
    # Check if cargo is loaded
    is_cargo_loaded = data.get('isCargoLoaded', False)
    
    # Only include trailer and cargo weight if trailer is actually attached
    if trailer_attached:
        total_weight_kg = unit_mass + trailer_mass + cargo_mass*is_cargo_loaded
    else:
        total_weight_kg = unit_mass

    return total_weight_kg / 1000

def refresh_button_detection():
    global cc_dec_button
    global cc_inc_button
    global cc_start_button
    global cc_dec_label
    global cc_inc_label
    global cc_start_label
    global cc_dec
    global cc_inc
    global cc_start

    cc_dec = get_button(cc_dec_button)
    cc_inc = get_button(cc_inc_button)
    cc_start = get_button(cc_start_button)

    if cc_dec:
        if cc_dec_label.cget("border_color") != KEY_ON_COLOR:
            cc_dec_label.configure(border_color=KEY_ON_COLOR)
    else:
        if cc_dec_label.cget("border_color") != SETTINGS_COLOR:
            cc_dec_label.configure(border_color=SETTINGS_COLOR)
    if cc_inc:
        if cc_inc_label.cget("border_color") != KEY_ON_COLOR:
            cc_inc_label.configure(border_color=KEY_ON_COLOR)
    else:
        if cc_inc_label.cget("border_color") != SETTINGS_COLOR:
            cc_inc_label.configure(border_color=SETTINGS_COLOR)
    if cc_start:
        if cc_start_label.cget("border_color") != KEY_ON_COLOR:
            cc_start_label.configure(border_color=KEY_ON_COLOR)
    else:
        if cc_start_label.cget("border_color") != SETTINGS_COLOR:
            cc_start_label.configure(border_color=SETTINGS_COLOR)

class cc_panel:
    def __init__(self, text_content: str, cc_mode: str = "Cruise control", cc_enabled: bool = True, x_co = 100, y_co = 100):
        """
        Initialize the cruise control display panel.
        
        Args:
            text_content: Initial speed text to display (e.g., "100 km/h")
            cc_mode: Cruise control mode ("Speed limiter" or "Cruise control")
            cc_enabled: Whether the cruise control system is enabled
        """
        self.scale_mult = 1.2
        self.start_x = x_co
        self.start_y = y_co
        self.panel_x = int(300 * self.scale_mult)
        self.panel_y = int(100 * self.scale_mult)
        self.bg_color = "#D3D3D3"
        self.fg_color = "#000000"
        self.radius = int(30 * self.scale_mult)
        self.cc_mode = cc_mode
        self.cc_enabled = cc_enabled
        self.text_color = self._get_color_for_mode(cc_mode, cc_enabled)
        self.text_content = text_content
        self.icon_spacing = int(20 * self.scale_mult)
        self.opacity = 0.6
        
        self.gui_thread = None
        self.root1 = None
        self.root2 = None
        self.running = False
        
        self.tk_img1 = None
        self.tk_img2 = None

        # Cache expensive operations
        self.font = self._load_font()
        self._icon_cache = {}
        self._position_cache = {}
        self._bg_rgb = self._hex_to_rgb(self.bg_color)
        
        self.icon = self._load_icon()
        self.text_position, self.icon_position = self._calculate_positions()

        self.img1, self.img2 = self._create_images()
        self._start_gui_thread()

    @staticmethod
    @lru_cache(maxsize=32)
    def _hex_to_rgb(hex_color):
        """Convert hex color to RGB tuple (cached)"""
        if isinstance(hex_color, str) and hex_color.startswith('#'):
            hex_color = hex_color.lstrip('#')
            return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
        return hex_color

    def _start_gui_thread(self):
        """Start the GUI in a separate thread"""
        self.running = True
        self.gui_thread = threading.Thread(target=self._show_images, daemon=True)
        self.gui_thread.start()

    def show(self):
        """Show both windows (deiconify)."""
        if self.root1 and self.root2:
            self.root1.after(0, self.root1.deiconify)
            self.root2.after(0, self.root2.deiconify)

    def hide(self):
        """Hide both windows (withdraw)."""
        if self.root1 and self.root2:
            self.root1.after(0, self.root1.withdraw)
            self.root2.after(0, self.root2.withdraw)

    def stop(self):
        """Stop the GUI and close all windows."""
        self.running = False
        
        def close_window(root):
            if root and root.winfo_exists():
                try:
                    root.after(0, root.quit)
                except:
                    pass
        
        close_window(self.root1)
        close_window(self.root2)

    def move(self, x: int, y: int):
        """Move both windows to the specified position."""
        if self.root1 and self.root2:
            self.root1.geometry(f"+{x}+{y}")
            self.root2.geometry(f"+{x}+{y}")

    def update(self, new_text, cc_mode: str = None, cc_enabled: bool = None):
        """Update the display with new information."""
        needs_icon_reload = cc_mode is not None and cc_mode != self.cc_mode
        needs_color_update = cc_enabled is not None and cc_enabled != self.cc_enabled
        needs_text_update = new_text != self.text_content
        
        # Only update what actually changed
        if not (needs_icon_reload or needs_color_update or needs_text_update):
            return
            
        self.text_content = new_text
        
        if needs_icon_reload:
            self.cc_mode = cc_mode
            self.icon = self._load_icon()
        
        if needs_color_update:
            self.cc_enabled = cc_enabled
        
        if needs_color_update or needs_icon_reload:
            self.text_color = self._get_color_for_mode(self.cc_mode, self.cc_enabled)
        
        if needs_text_update or needs_icon_reload:
            self.text_position, self.icon_position = self._calculate_positions()
        
        self.img1, self.img2 = self._create_images()
        
        if self.root1 and self.root2:
            self.root1.after(0, self._update_gui_images)

    def _update_gui_images(self):
        """Update the GUI images (must be called from main thread)"""
        if not (self.root1 and self.root2):
            return
            
        try:
            # Reuse existing PhotoImage objects if possible
            if hasattr(self, 'tk_img1') and self.tk_img1:
                self.tk_img1.paste(self.img1)
            else:
                self.tk_img1 = ImageTk.PhotoImage(image=self.img1, master=self.root1)
                
            if hasattr(self, 'tk_img2') and self.tk_img2:
                self.tk_img2.paste(self.img2)
            else:
                self.tk_img2 = ImageTk.PhotoImage(image=self.img2, master=self.root2)
            
            self.root1.winfo_children()[0].configure(image=self.tk_img1)
            self.root2.winfo_children()[0].configure(image=self.tk_img2)
            self.root1.lift()
            self.root2.lift()
        except Exception as e:
            print(f"Error updating GUI images: {e}")

    @lru_cache(maxsize=4)
    def _load_font(self):
        """Load font (cached)"""
        try:
            return ImageFont.truetype("arialbd.ttf", int(40 * self.scale_mult))
        except IOError:
            return ImageFont.load_default()

    def _load_icon(self):
        """Load and resize the cruise control icon based on current mode (with caching)"""
        # Create cache key
        cache_key = (self.cc_mode, self.text_content, self.scale_mult)
        if cache_key in self._icon_cache:
            return self._icon_cache[cache_key]
            
        try:
            icon_file = "speed limiter.png" if self.cc_mode == "Speed limiter" else "cruise control.png"
            icon = Image.open(os.path.join(os.path.dirname(os.path.abspath(__file__)), icon_file))
            
            # Calculate icon size
            temp_draw = ImageDraw.Draw(Image.new("RGB", (1, 1)))
            bbox = temp_draw.textbbox((0, 0), self.text_content, font=self.font)
            text_height = bbox[3] - bbox[1]
            icon_size = int(text_height * 2)
            
            icon = icon.resize((icon_size, icon_size), Image.Resampling.LANCZOS)
            
        except (IOError, FileNotFoundError):
            # Create simple placeholder
            icon_size = int(80 * self.scale_mult)
            icon = Image.new("RGBA", (icon_size, icon_size), (255, 255, 255, 255))
            draw = ImageDraw.Draw(icon)
            draw.ellipse([10, 10, icon_size-10, icon_size-10], outline=(0, 0, 0), width=3)
            draw.text((icon_size//2-5, icon_size//2-10), "CC", fill=(0, 0, 0))
        
        # Cache the result
        self._icon_cache[cache_key] = icon
        return icon

    def _calculate_positions(self):
        """Calculate positions for both text and icon (with caching)"""
        cache_key = (self.text_content, self.cc_mode, self.scale_mult)
        if cache_key in self._position_cache:
            return self._position_cache[cache_key]
            
        # Use a minimal temporary draw object
        temp_draw = ImageDraw.Draw(Image.new("RGB", (1, 1)))
        
        # Get text dimensions
        text_bbox = temp_draw.textbbox((0, 0), self.text_content, font=self.font)
        text_w = text_bbox[2] - text_bbox[0]
        text_h = text_bbox[3] - text_bbox[1]
        
        # Get icon dimensions
        icon_w, icon_h = self.icon.size
        
        # Calculate positions
        right_margin = int(20 * self.scale_mult)
        icon_x = self.panel_x - icon_w - right_margin
        icon_y = (self.panel_y - icon_h) // 2
        text_x = icon_x - self.icon_spacing - text_w - text_bbox[0]
        text_y = (self.panel_y - text_h) // 2 - text_bbox[1]
        
        result = ((text_x, text_y), (icon_x, icon_y))
        self._position_cache[cache_key] = result
        return result

    def _multiply_text_color_numpy(self, image, multiply_color, exception_color):
        """
        Optimized color multiplication using NumPy for better performance.
        """
        multiply_rgb = self._hex_to_rgb(multiply_color)
        exception_rgb = self._hex_to_rgb(exception_color)
        
        # Convert PIL image to numpy array
        img_array = np.array(image.convert('RGB'), dtype=np.uint8)
        
        # Create mask for pixels that are NOT the exception color
        mask = ~np.all(img_array == exception_rgb, axis=2)
        
        # Apply multiplication only to non-exception pixels
        multiply_factors = np.array(multiply_rgb, dtype=np.float32) / 255.0
        
        # Vectorized multiplication
        img_float = img_array.astype(np.float32)
        img_float[mask] *= multiply_factors
        
        # Clip values and convert back
        img_array = np.clip(img_float, 0, 255).astype(np.uint8)
        
        return Image.fromarray(img_array)

    def _create_images(self):
        """Create the display images with optimizations"""
        # Create base image once
        base_shape = Image.new("RGB", (self.panel_x, self.panel_y), self.bg_color)
        draw_base = ImageDraw.Draw(base_shape)
        draw_base.rounded_rectangle(
            [(0, 0), (self.panel_x, self.panel_y)],
            radius=self.radius,
            fill=self.fg_color,
            outline="black",
        )
        
        # Image 1
        img1 = base_shape.copy()
        draw1 = ImageDraw.Draw(img1)
        draw1.text(self.text_position, self.text_content, font=self.font, fill=self.text_color)
        
        # Optimize icon pasting
        colored_icon = self._multiply_text_color_numpy(self.icon, self.text_color, self.bg_color)
        img1.paste(colored_icon, self.icon_position, 
                  self.icon if self.icon.mode == 'RGBA' else None)

        # Image 2 - optimized with numpy operations
        img2 = Image.new("RGB", (self.panel_x, self.panel_y), self.bg_color)
        draw2 = ImageDraw.Draw(img2)
        draw2.text(self.text_position, self.text_content, font=self.font, fill="white")
        img2.paste(self.icon, self.icon_position, 
                  self.icon if self.icon.mode == 'RGBA' else None)
        
        img2 = self._remove_anti_aliasing_numpy(img2, threshold=234)
        img2 = self._multiply_text_color_numpy(img2, self.text_color, self.bg_color)

        return img1, img2

    def _remove_anti_aliasing_numpy(self, image, threshold=234):
        """Optimized anti-aliasing removal using NumPy"""
        img_array = np.array(image.convert('RGB'), dtype=np.uint8)
        
        # Convert to grayscale for brightness check
        grayscale = np.dot(img_array[...,:3], [0.299, 0.587, 0.114])
        
        # Create mask for pixels above threshold
        mask = grayscale >= threshold
        
        # Create result array filled with background color
        bg_rgb = self._bg_rgb
        result = np.full_like(img_array, bg_rgb, dtype=np.uint8)
        
        # Keep original colors where mask is True
        result[mask] = img_array[mask]
        
        return Image.fromarray(result)

    def _make_draggable(self, window1, window2):
        """Make both windows draggable together (optimized to reduce redundant calculations)"""
        drag_data = {'start_x1': 0, 'start_y1': 0, 'start_x2': 0, 'start_y2': 0}
        
        def start_move(event):
            drag_data['start_x1'] = event.x_root - window1.winfo_x()
            drag_data['start_y1'] = event.y_root - window1.winfo_y()
            drag_data['start_x2'] = event.x_root - window2.winfo_x()
            drag_data['start_y2'] = event.y_root - window2.winfo_y()

        def stop_move(event):
            try:
                save_variables(os.path.join(os.path.dirname(os.path.abspath(__file__)), "saves.json"),
                            panel_x=window1.winfo_x(),
                            panel_y=window1.winfo_y())
            except:
                pass  # Ignore save errors

        def do_move(event):
            x1 = event.x_root - drag_data['start_x1']
            y1 = event.y_root - drag_data['start_y1']
            x2 = event.x_root - drag_data['start_x2']
            y2 = event.y_root - drag_data['start_y2']
            
            window1.geometry(f"+{x1}+{y1}")
            window2.geometry(f"+{x2}+{y2}")

        # Bind events to both windows
        for window in [window1, window2]:
            window.bind("<Button-1>", start_move)
            window.bind("<B1-Motion>", do_move)
            window.bind("<ButtonRelease-1>", stop_move)

    def _show_images(self):
        """Display the GUI windows"""
        if not all(hasattr(self, attr) for attr in ['img1', 'img2']):
            print("Images not loaded")
            return

        try:
            # Create first window
            root1 = tk.Tk()
            self._setup_window(root1)
            root1.attributes("-alpha", self.opacity)
            
            tk_img1 = ImageTk.PhotoImage(self.img1, master=root1)
            self.tk_img1 = tk_img1
            
            label1 = tk.Label(root1, image=tk_img1, bd=0, highlightthickness=0, bg=self.bg_color)
            label1.pack(padx=0, pady=0)

            # Create second window
            root2 = tk.Toplevel(root1)
            self._setup_window(root2)
            
            tk_img2 = ImageTk.PhotoImage(self.img2, master=root2)
            self.tk_img2 = tk_img2
            
            label2 = tk.Label(root2, image=tk_img2, bd=0, highlightthickness=0, bg=self.bg_color)
            label2.pack(padx=0, pady=0)

            # Position windows
            root1.update_idletasks()
            root1.geometry(f"+{root1.winfo_x()}+{root1.winfo_y()}")

            self._make_draggable(root1, root2)
            self.root1 = root1
            self.root2 = root2
            self.root1.mainloop()
            
        except Exception as e:
            print(f"Error in _show_images: {e}")

    def _setup_window(self, window):
        """Common window setup"""
        window.withdraw()  # Hide the window initially
        window.overrideredirect(True)
        window.attributes("-topmost", True)
        window.attributes("-transparentcolor", self.bg_color)
        window.configure(bg=self.bg_color)
        
        if self.start_x != None and self.start_y != None:
            window.geometry(f"+{self.start_x}+{self.start_y}")
            
        window.update_idletasks()

    def _get_color_for_mode(self, mode: str, enabled: bool):
        """Get the appropriate color based on the cruise control mode"""
        if not enabled:
            return DISABLED_COLOR
        return SPEEDLIMITER_COLOR if mode == "Speed limiter" else CRUISECONTROL_COLOR

def change_target_speed(increments, app=None):
    global target_speed

    if abs(increments) >= 5:
        if increments > 0:
            target_speed = ((target_speed // abs(increments)) + 1) * abs(increments)
        else:
            # Check if target_speed is already a multiple of abs(increments)
            if target_speed % abs(increments) == 0:
                # If it's already a multiple, go to the previous multiple
                target_speed = ((target_speed // abs(increments)) - 1) * abs(increments)
            else:
                # If it's not a multiple, round down to the nearest multiple
                target_speed = ((target_speed // abs(increments))) * abs(increments)
    else:
        target_speed += increments
    if target_speed < 30:
        target_speed = 30
    elif target_speed > 130:
        target_speed = 130
    if app is not None:
        if target_speed is not None:
            app.update(f"{int(target_speed)} km/h", cc_mode.get(), True)
        else:
            app.update("-- km/h", cc_mode.get(), True)

def main_cruise_control():
    global exit_event
    global cc_mode
    global cc_enabled
    global cc_inc_button
    global cc_dec_button
    global cc_start_button
    global cc_dec
    global cc_inc
    global cc_start
    global buttons_thread
    global brakeval
    global gasval
    global target_speed
    global speed
    global pauzed
    global panel
    global long_increments
    global short_increments
    global long_press_reset

    cc_target_speed_thread = threading.Thread(target=cc_target_speed_thread_func, daemon=True, name="CC Target Speed Thread")
    time_pressed_dec = None
    time_pressed_inc = None
    time_pressed_start = None
    target_speed = None
    long_press_dec = False
    long_press_inc = False
    long_press_start = False
    cc_enabled = False
    try:
        long_increment_int = int(long_increments.get().split()[0])
        short_increment_int = int(short_increments.get().split()[0])
    except ValueError:
        cmd_print("Invalid increment values in settings", "#FF2020", 10)
        long_increment_int = 5
        short_increment_int = 1
        long_increments.set(f"5 km/h")
        short_increments.set(f"1 km/h")
    if _data_cache and "panel_x" in _data_cache and "panel_y" in _data_cache:
        panel_x = _data_cache["panel_x"]
        panel_y = _data_cache["panel_y"]
    else:
        panel_x = 100
        panel_y = 100
    app = cc_panel("-- km/h", cc_mode.get(), False, panel_x, panel_y)
    #app.update(f"-- km/h", cc_mode.get(), False)


    while not exit_event.is_set() and not (cc_dec_button is None and cc_inc_button is None and cc_start_button is None):
        try:

            if buttons_thread is None or not buttons_thread.is_alive():
                refresh_button_detection()

            if (buttons_thread is not None and buttons_thread.is_alive()) or device_lost or not ets2_detected.is_set() or device is None or not device.get_init():
                app.hide()
                time.sleep(0.04)
                continue
            
            long_increment_int = int(long_increments.get().split()[0])
            short_increment_int = int(short_increments.get().split()[0])
            all_buttons_assigned = cc_dec_button is not None and cc_inc_button is not None and cc_start_button is not None

            if not all_buttons_assigned and (cc_dec or cc_inc or cc_start):
                cmd_print("Please assign all cruise control buttons in the settings", "#FF2020", 10)
                time.sleep(0.5)
                continue
            if cc_mode.get() == "Cruise control":
                if (cc_inc or cc_start) and data.get('parkBrake', False):
                    cmd_print("Cruise control cannot be used with parking brake engaged", "#FF2020", 10)
                    continue
                    
                if (cc_inc or cc_start) and data.get('gear', 0) <= 0:
                    cmd_print("Cruise control can only be used in drive", "#FF2020", 10)
                    continue

            if cc_dec and not cc_inc and not cc_start and all_buttons_assigned:
                if time_pressed_dec is None:
                    time_pressed_dec = time.time()
                delt_time_dec = (time.time() - time_pressed_dec)
                if (not long_press_dec and delt_time_dec > 0.3) or (long_press_dec and delt_time_dec > 0.6):
                    long_press_dec = True
                    time_pressed_dec = time.time()
                    if cc_dec:

                        # long press
                        print("long press on dec")
                        if target_speed != None:
                            change_target_speed(-long_increment_int, app)

            elif time_pressed_dec != None:
                if not long_press_dec:

                    # short press
                    print("short press dec")
                    if target_speed != None:
                        change_target_speed(-short_increment_int, app)

                else:
                    long_press_dec = False
                time_pressed_dec = None

            #########################################

            if cc_inc and not cc_dec and not cc_start and all_buttons_assigned:
                if time_pressed_inc is None:
                    time_pressed_inc = time.time()
                delt_time_inc = (time.time() - time_pressed_inc)
                if (not long_press_inc and delt_time_inc > 0.3) or (long_press_inc and delt_time_inc > 0.6):
                    long_press_inc = True
                    time_pressed_inc = time.time()
                    if cc_inc:

                        # long press
                        print("long press on inc")
                        if cc_enabled:
                            change_target_speed(long_increment_int, app)
                        elif target_speed is None or (speed > target_speed):
                            target_speed = max(min(int(round(speed)),130), 30)
                        if not cc_enabled:
                            cc_enabled = True
                            cmd_print("Cruise control enabled")

            elif time_pressed_inc != None:
                if not long_press_inc:

                    # short press
                    print("short press inc")
                    if cc_enabled:
                        change_target_speed(short_increment_int, app)
                    elif target_speed is None or speed > target_speed:
                        target_speed = max(min(int(round(speed)),130), 30)
                    if not cc_enabled:
                        cc_enabled = True
                        cmd_print("Cruise control enabled")

                else:
                    long_press_inc = False
                time_pressed_inc = None

            ########################################

            if cc_start and not cc_dec and not cc_inc and all_buttons_assigned:
                if time_pressed_start is None:
                    time_pressed_start = time.time()
                delt_time_start = (time.time() - time_pressed_start)
                if (not long_press_start and delt_time_start > 0.5):
                    long_press_start = True
                    if cc_start and long_press_reset.get():

                        # long press
                        print("long press on start")
                        target_speed = max(min(int(round(speed)),130), 30)
                        if not cc_enabled:
                            cc_enabled = True
                        app.update(f"{int(target_speed)} km/h", cc_mode.get(), True)
                    elif not long_press_reset.get():
                        cmd_print("Long press to reset is disabled")

            elif time_pressed_start != None:
                if not long_press_start:

                    # short press
                    print("short press start")
                    if cc_enabled:
                        cc_enabled = False
                        cmd_print("Cruise control disabled")
                    else:
                        cc_enabled = True
                        cmd_print("Cruise control enabled")

                    if target_speed == None:
                        target_speed = max(min(int(round(speed)),130), 30)
                    app.update(f"{int(target_speed)} km/h", cc_mode.get(), cc_enabled)

                else:
                    long_press_start = False
                time_pressed_start = None
            
            if cc_mode.get() == "Cruise control":
                if brakeval > 0.1 or em_stop or data.get('parkBrake', False):
                    if cc_enabled:
                        cc_enabled = False
                        app.update(f"{int(target_speed)} km/h", cc_mode.get(), cc_enabled)
                        cmd_print("Cruise control disabled due to brake input")

            if target_speed is not None:
                if cc_enabled and not cc_target_speed_thread.is_alive():
                    cc_target_speed_thread = threading.Thread(target=cc_target_speed_thread_func, daemon=True, name="CC Target Speed Thread")
                    cc_target_speed_thread.start()

            # update panel
            if not exit_event.is_set():
                try:
                    if show_cc_ui.get() and ets2_detected.is_set() and all_buttons_assigned:
                        if not app.root1.winfo_viewable() and not app.root2.winfo_viewable():
                            app.show()
                    else:
                        if app.root1.winfo_viewable() and app.root2.winfo_viewable():
                            app.hide()
                except:
                    pass

                if target_speed is not None:
                    app.update(f"{int(target_speed)} km/h", cc_mode.get(), cc_enabled)
                else:
                    app.update(f"-- km/h", cc_mode.get(), False)
            else:
                try:
                    app.stop()
                except:
                    pass
            time.sleep(0.02)
        except Exception as e:
            # close panel
            try:
                if app is not None and app.running:
                    app.stop()
            except:
                pass
            context = get_error_context()
            log_error(e, context)
            time.sleep(1)
    app.stop()



















































from collections import defaultdict, deque
import numpy as np

# memory of previous data for each lead vehicle (fixed-length queue)
_prev_data = defaultdict(lambda: deque(maxlen=10))
_current_lead_id = None

def adaptive_cruise_control(vehicles_in_lane, ego_speed, min_gap=10.0, acc_time_gap=1, debug=False):
    """
    Adaptive Cruise Control logic with lead vehicle acceleration anticipation.
    All input values are averaged over time. Queue resets when lead_id changes.
    
    Args:
        vehicles_in_lane: List of (vehicle_id, distance_m, speed_kmph) sorted by distance (closest first).
        ego_speed: Ego vehicle speed in km/h.
        min_gap: Minimum gap to lead vehicle in meters.
        acc_time_gap: Desired time gap to lead vehicle in seconds.
        debug: Enable detailed debug prints.
    Returns:
        acc_value: Float in [-1, 1]. Positive = accelerate, negative = brake.
    """
    global _current_lead_id

    if not vehicles_in_lane:
        return 1.0  # full accelerate if free road

    # Lead vehicle
    lead_id, lead_dist_raw, lead_speed_raw = vehicles_in_lane[0]

    # Check if lead vehicle changed - if so, reset the queue
    if _current_lead_id != lead_id:
        _prev_data[lead_id].clear()
        _current_lead_id = lead_id
        if debug:
            print(f"Lead vehicle changed to {lead_id}, queue reset")

    # Update deque with all input data (distance, speed, ego_speed)
    data_point = {
        'distance': lead_dist_raw,
        'speed': lead_speed_raw,
        'ego_speed': ego_speed
    }
    _prev_data[lead_id].append(data_point)

    # Calculate averaged values
    data_history = list(_prev_data[lead_id])
    
    lead_dist = np.mean([d['distance'] for d in data_history])
    lead_speed = np.mean([d['speed'] for d in data_history])
    avg_ego_speed = np.mean([d['ego_speed'] for d in data_history])

    # Calculate desired gap using averaged ego speed
    desired_gap = max(avg_ego_speed / 3.6 * acc_time_gap, min_gap)
    gap_error = lead_dist - desired_gap
    speed_error = lead_speed - avg_ego_speed

    # calculate closeness amplifier using averaged values
    actual_time_gap = max(min(lead_dist / (max(avg_ego_speed, 1) / 3.6), 10), 0.01)
    closeness_amp = pow(0.8, actual_time_gap*5-3) + 0.5
    if speed_error < 0:
        closeness_amp = closeness_amp ** 1.5
    else:
        closeness_amp = 1

    acceleration_amp = max(-(actual_time_gap/2)**3+1, 0.2)
    
    # calculate slow speed adjustment using averaged ego speed
    slow_speed_adj = pow(0.8, max(avg_ego_speed, 0.0))*2.5

    # Compute average delta_v using averaged speeds from first and last data points
    delta_v = 0.0
    if len(data_history) > 1:
        delta_v = data_history[-1]['speed'] - data_history[0]['speed']

    # Gains
    K_gap = 0.10 * closeness_amp * (slow_speed_adj/2+1)
    K_speed = 0.13 * closeness_amp * (slow_speed_adj/2+1)
    K_acc = 0.20 *acceleration_amp

    # Control law (sum of weighted errors)
    acc_raw = K_gap * np.sign(gap_error)*((abs(gap_error)/10)**0.7)*10 + K_speed * speed_error + K_acc * delta_v
    if acc_raw <= 0:
        acc_raw -= slow_speed_adj

    # disabled clamping to make it able to emergency brake
    acc_value = acc_raw / 1.5

    if debug:
        print(f"Lead id={lead_id} raw_dist={lead_dist_raw:.1f}m avg_dist={lead_dist:.1f}m")
        print(f"Gap error={gap_error:.2f} | Speed error={speed_error:.2f} | Delta_v={delta_v:.2f}")

    return acc_value


































































def cc_target_speed_thread_func():
    global exit_event
    global target_speed
    global speed
    global data
    global total_weight_tons
    global cc_mode
    global cc_enabled
    global cc_gas
    global cc_brake
    global cc_locked
    global cc_limiting
    global brake_exponent_variable
    global _data_cache
    global acc_data

    cc_locked = False
    cc_limiting = False
    prev_speed = speed
    prev_target_speed = target_speed
    prev_acc_dist = None
    prev_cc_gas = 0.0
    prev_cc_brake = 0.0
    ds = 0.0
    av_ds = 0.0
    integral_sum = 0.0
    ff_est = 0.0
    alpha  = 0.8
    P = 0.11
    I = 0.01
    D = 0.08
    max_integral = 0.2 / I
    max_proportional = 1.3

    prev_time = time.time()-0.1

    while not exit_event.is_set() and cc_enabled and not em_stop:
        if target_speed is not None and not pauzed:

            if cc_mode.get() != "Cruise control":
                acc_data = None

            slope = data['rotationY']
            weight_adjustment = (0.27*((total_weight_tons-8.93)/(8.5))+1)
            if cc_mode.get() == "Speed limiter":
                error = (target_speed-0.1) - speed
            else:
                error = target_speed - speed

            dt = time.time() - prev_time
            av_ds = (av_ds*4 + (prev_speed - speed)) / 5
            ds = prev_speed - speed
            prev_time = time.time()

            if cc_locked and not prev_cc_gas >= 0.9:
                if error < 0:
                    if cc_mode.get() == "Cruise control":
                        integral_sum += -(-error/10) * dt * 5 * min(abs(1/(max(av_ds*3, 0.01))), 5)
                    else:
                        integral_sum += -(-error/10) * dt * 10 * min(abs(1/(max(av_ds*3, 0.01))), 5)
                else:
                    integral_sum += (error/10) * dt * 5 * min(abs(1/(max(av_ds*3, 0.01))), 5)

            if ((error > 0 and (ds) > 0) or (error < 0 and (ds) < 0)) and abs(error) < 3 and abs(av_ds) < 0.05 and not (cc_mode.get() == "Speed limiter" and cc_limiting):
                cc_locked = True
            elif prev_target_speed != target_speed or abs(error) > 2 or (cc_mode.get() == "Speed limiter" and not cc_limiting):
                cc_locked = False

            integral_sum = max(-max_integral, min(max_integral, integral_sum))

            derivative = (av_ds) / dt

            if error < 0:
                proportional = -((-error)**0.8) * P
            else:
                proportional = (error**0.8) * P

            slow_speed_adjustment = (-(2**(-(max(target_speed, 30)*0.04)+0.3))+1)*1.3
            physics_adjustment = (speed * 0.0015 + slope * 18 * slow_speed_adjustment) * weight_adjustment

            base_val = (max(proportional, -max_proportional) * slow_speed_adjustment +
                        integral_sum * I +
                        derivative * D * slow_speed_adjustment +
                        physics_adjustment)
            
            acc_val = adaptive_cruise_control(acc_data, speed, 10, 1.5, data["accelerationZ"]) + physics_adjustment

            ff_est = alpha * ff_est + (1.0 - alpha) * base_val

            temp_val = min(base_val + ff_est, acc_val)

            cc_gas = (min(max(temp_val, 0), 1) + prev_cc_gas) / 2
            if temp_val > 0:
                cc_brake = 0.0
            else:
                cc_brake = min(max((-temp_val/20)**1.2, 0), 0.07)
            if acc_val < 0:
                cc_brake = max(cc_brake, max((min(abs(acc_val)/7, 2)**2),0.0))

        elif not pauzed:
            cc_locked = False
            cc_gas = 0.0
            cc_brake = 0.0

        prev_target_speed = target_speed
        prev_speed = speed
        prev_cc_gas = cc_gas
        prev_cc_brake = cc_brake
        time.sleep(0.1)




def main():
    global controller
    """Main game logic"""
    global gasaxis
    global brakeaxis
    global joy
    global device
    global device_instance_id
    global device_lost
    global recovered
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
    global pauzed
    global img
    global to_img_coords
    global img_copy
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
    global brakeval
    global gasval
    global total_weight_tons
    global cc_dec_label
    global cc_inc_label
    global cc_start_label
    global cc_dec
    global cc_inc
    global cc_start
    global cc_brake
    global cc_gas
    global cruise_control_thread
    global cc_locked
    global cc_limiting
    global acc_data
    # Initialize pygame for joystick handling
    
    # Start SDK check thread
    sdk_thread = threading.Thread(target=sdk_check_thread, daemon=True, name="SDK Check Thread")
    sdk_thread.start()

    cruise_control_thread = None
    
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
                            autostart_variable = autostart_variable.get(),
                            weight_adjustment = weight_adjustment.get(),
                            cc_dec_button = cc_dec_button,
                            cc_inc_button = cc_inc_button,
                            cc_start_button = cc_start_button,
                            cc_mode = cc_mode.get(),
                            long_increments = long_increments.get(),
                            short_increments = short_increments.get(),
                            long_press_reset = long_press_reset.get(),
                            show_cc_ui = show_cc_ui.get()
                            )

        if exit_event.is_set():
            return
            
        # Main game loop
        prev_brakeval = 0
        # prev_gasval = 1   (not used)
        
        brakeval = 0
        gasval = 0
        speed = 0
        prev_speed = 0
        gear = 0
        data = {}
        cc_start = False
        cc_dec = False
        cc_inc = False
        cc_gas = 0.0
        cc_brake = 0.0
        opdgasval = 0
        opdbrakeval = 0
        gas_output = 0
        brake_output = 0
        prev_opdbrakeval = 0
        gas_output = 0
        brake_output = 0
        arrived = False
        stopped = False
        pauzed = False
        horn = False
        em_stop = False
        latency_timestamp = time.time()-0.015
        latency = 0.015
        hazards_prompted = False
        offset = 0.5
        cv2.namedWindow("ETS2 Radar", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("ETS2 Radar", 600, 600)
        radar = ETS2Radar(show_window=True, fov_angle=40)
        prev_stop = time.time()

        # start the bar thread
        bar_var_update()
        
        check_and_start_exe()


        img, to_img_coords = plot_onepedaldrive(return_result=True)
        pil_copy = img.copy()
        new_width = 200
        new_height = 200
        img_copy = ctk.CTkImage(pil_copy.resize((new_width, new_height), Image.LANCZOS), size=(new_width, new_height))

        refresh_button_labels()


        while not exit_event.is_set():
            timestamp = time.time()

            # save variables to the file
            save_variables(os.path.join(os.path.dirname(os.path.abspath(__file__)), "saves.json"),
                            bar_variable = bar_variable.get(),
                            gas_exponent_variable = gas_exponent_variable.get(),
                            brake_exponent_variable = brake_exponent_variable.get(),
                            max_opd_brake_variable = max_opd_brake_variable.get(),
                            offset_variable = offset_variable.get(),
                            gasaxis = gasaxis,
                            brakeaxis  =  brakeaxis,
                            gas_inverted = gas_inverted,
                            brake_inverted = brake_inverted,
                            polling_rate = polling_rate.get(),
                            opd_mode_variable = opd_mode_variable.get(),
                            hazards_variable = hazards_variable.get(),
                            autodisable_hazards = autodisable_hazards.get(),
                            horn_variable = horn_variable.get(),
                            airhorn_variable = airhorn_variable.get(),
                            autostart_variable = autostart_variable.get(),
                            weight_adjustment = weight_adjustment.get(),
                            cc_dec_button = cc_dec_button,
                            cc_inc_button = cc_inc_button,
                            cc_start_button = cc_start_button,
                            cc_mode = cc_mode.get(),
                            long_increments = long_increments.get(),
                            short_increments = short_increments.get(),
                            long_press_reset = long_press_reset.get(),
                            show_cc_ui = show_cc_ui.get()
                            )

            if offset_variable.get() == 0:
                opd_mode_variable.set(False)
                temp_offset_variable.set(0.2)
                offset_variable.set(0.2)
                opd_mode_var_update()

            """ (old code, not used anymore)
            while (not isinstance(_data_cache["device"], str) or not _data_cache["device"] == "") and device_lost == True:
                if recovered is None:
                    recovered = deserialize_joystick(_data_cache["device"])
                    check_wheel_connected()
                if exit_event.is_set():
                    break
                time.sleep(0.5)
                opdgasval = 0
                opdbrakeval = 0
            """

            if exit_event.is_set():
                print("Mainloop exited, cleaning up...")
                return

            latency = timestamp - latency_timestamp
            latency_timestamp = timestamp

            latency_multiplier = (latency / 0.015) * 2

            # make the program run at the input polling rate
            try:
                polling_rate.set(max(10, min(100, polling_rate.get())))
                time.sleep(max(0.005, 1/polling_rate.get() - (time.time()-timestamp))) # 0.005 min is for stability
            except:
                cmd_print("unreliable input values!")

            timestamp = time.time()

            if exit_event.is_set():
                print("Mainloop exited, cleaning up...")
                return

            # get app settings
            hazards_variable_var = hazards_variable.get()
            autodisable_hazards_var = autodisable_hazards.get()
            horn_variable_var = horn_variable.get()
            airhorn_variable_var = airhorn_variable.get()

            if cc_dec_button is not None or cc_inc_button is not None or cc_start_button is not None:
                # start a background checker to prevent ignored button presses
                if cruise_control_thread is None or not cruise_control_thread.is_alive():
                    cruise_control_thread = threading.Thread(target=main_cruise_control, daemon=True, name="cruise control thread")
                    cruise_control_thread.start()

            # get input if pygame is initialized
            if pygame.get_init() and not pauze_pedal_detection and not exit_event.is_set():
                for event in pygame.event.get():
                    if event.type == pygame.JOYDEVICEREMOVED: # in development, this event is triggered when the joystick is disconnected
                        if event.instance_id == device_instance_id:
                            cmd_print("Your pedals disconnected", "#FF2020", 15)
                            # Handle the disconnection for your specific device
                            device_lost = True
                            device.quit()
                            device = None
                            #maximize root
                            root.deiconify()
                            root.focus_force()
                            root.lift()
                            root.attributes('-topmost', True)
                            root.attributes('-topmost', False)
                            
                    elif event.type == pygame.JOYDEVICEADDED:
                        # Optionally handle reconnection
                        if device_lost is True:  # If your device was previously disconnected
                            # You might want to check if this is the same type of device
                            i = 0
                            cmd_print("reconnecting to pedals", display_duration=63)
                            time.sleep(5)
                            while not exit_event.is_set():
                                try:
                                    recovered = deserialize_joystick(_data_cache["device"])
                                    if recovered is not None:
                                        recovered.init()
                                        device_instance_id = recovered.get_instance_id()
                                        device = recovered
                                        device.init()
                                        refresh_button_labels()
                                        break
                                    else:
                                        cmd_print("Failed to reconnect to pedals, reconfigure please", "#FF2020", 30)
                                        device_lost = False
                                        device = None
                                        connected_joystick_label.configure(text="None connected")
                                        gasaxis = 0
                                        brakeaxis = 0
                                        break
                                except Exception as e:
                                    print(f"Error reinitializing device: {e}")
                                    time.sleep(0.2)
                                    i+=1
                                    if i > 30:
                                        cmd_print("Failed to reconnect to pedals, reconfigure please", "#FF2020", 30)
                                        device_lost = True
                                        device = None
                                        connected_joystick_label.configure(text="None connected")
                                        break
                            if exit_event.is_set():
                                print("Mainloop exited, cleaning up...")
                                return
                            
                            if device is None:
                                continue
                            device.quit()
                            time.sleep(4) # a reinitialization is required to avoid inconsistent behavior
                            try:
                                device.init()
                                cmd_print("succesfully reconnected")
                                device_lost = False  # Reset the flag
                            except Exception as e:
                                cmd_print(f"Error reinitializing device: {e}", "#FF2020", 15)
                                device_lost = True
                            if device is not None:
                                if len(device.get_name()) > 20:
                                    connected_joystick_label.configure(text=f"{device.get_name()[:20]}...")
                                else:
                                    connected_joystick_label.configure(text=f"{device.get_name()}")

                                reset_operational_variables()  # Reset all operational variables
                    if event.type == pygame.JOYAXISMOTION and device is not None:
                        try:
                            if brake_inverted:
                                brakeval = round((device.get_axis(brakeaxis)*-1+1)/2,3)
                            else:
                                brakeval = round((device.get_axis(brakeaxis)+1)/2,3)
                            if gas_inverted:
                                gasval = round((device.get_axis(gasaxis)*-1+1)/2,3)
                            else:
                                gasval = round((device.get_axis(gasaxis)+1)/2,3)
                        except Exception as e:
                            cmd_print(f"E reading joy: {e}", "#FF2020", 15)
            else:
                pygame.init()
                pygame.joystick.init()


            if device_lost == True:
                gasval = 0
                brakeval = 0
                opdbrakeval = 0
                opdgasval = 0
                gas_output = 0
                brake_output = 0
                if em_stop == False:
                    send(0, 0.15, controller)
                else:
                    send(0, 1, controller)
                #activate hazards and horn if they are not already 
                if speed > 0.1:
                    setattr(controller, "wipers4", True)
                    setattr(controller, "wipers3", True)

                try:
                    if data["lightsHazards"] == False and not hazards_prompted:
                        setattr(controller, "accmode", True)
                        hazards_prompted = True
                        time.sleep(0.05)
                        setattr(controller, "accmode", False)
                except:
                    pass

                live_visualization_frame.configure(image=img_copy)
                live_visualization_frame.image = img_copy

                time.sleep(0.2)
                if data is not None or speed > 0.1:
                    setattr(controller, "steering", float(-data["gameSteer"]))
                else:
                    setattr(controller, "steering", 0.0)

                setattr(controller, "wipers4", False)
                setattr(controller, "wipers3", False)
                time.sleep(0.8)

            if not check_wheel_connected():
                continue

            if not ets2_detected.is_set():
                cmd_print("Waiting for ETS2 SDK connection...")
                opdgasval = 0
                opdbrakeval = 0
                gas_output = 0
                brake_output = 0
                live_visualization_frame.configure(image=img_copy)
                live_visualization_frame.image = img_copy
                time.sleep(0.05)
                continue

            try:
                data = truck_telemetry.get_data()
            except:
                ets2_detected.clear()
                continue

            pauzed = data['paused']

            # radar code
            acc_data = radar.update(data)

            slope = data['rotationY']

            if int(data["routeDistance"]) != 0:
                if int(data["routeDistance"]) < 150 and not arrived:
                    arrived = True
                    cmd_print("arrived at destination")
                elif int(data["routeDistance"]) > 1500:
                    arrived = False
            else:
                arrived = False

            speed = round(data["speed"] * 3.6, 4)  # Convert speed from m/s to km/h
            gear = int(data["gearDashboard"])

            total_weight_tons = calc_truck_weight(data)

            """
            text_file = open("Output.txt", "w")

            text_file.write(f"Data: {data}")

            text_file.close()
            """

            opdgasval, opdbrakeval = onepedaldrive(gasval, brakeval)

            '''
            if data["cruiseControl"] == True and data["cruiseControlSpeed"] > 0 and brakeval == 0:
                opdbrakeval = 0
            elif stopped == True and gasval > 0 and speed >= -0.3 and gear < 0:
                opdbrakeval = 0.1
            elif stopped == True and gasval > 0 and speed <= 0.3 and gear > 0:
                opdbrakeval = 0.1
            elif stopped == True:
                opdbrakeval = max(0, opdbrakeval)
            '''
            offset = offset_variable.get()
            a = (0.035)-slope/2
            if stopped:
                if gear > 0 and speed < 3 and gasval <= (0.7+offset*0.7) and gasval != 0:
                    opdbrakeval += min(0.03*(((-round(speed+0.8,1)+4)**5)/(4**5))+slope*2, 0.3)
                elif gear < 0 and speed > -3 and gasval <= (0.7+offset*0.7) and gasval != 0:
                    opdbrakeval += min(0.03*(((round(speed+0.8,1)+4)**5)/(4**5))-slope*2, 0.3)
                elif gasval == 0 and gear != 0:
                    opdbrakeval += 0.06
                delta_time = time.time()-prev_stop
                t = 0.5
                if prev_stop != 0 and delta_time < t:
                    opdbrakeval = opdbrakeval*(delta_time/t)+prev_opdbrakeval*(1-delta_time/t)
                else:
                    prev_stop = 0
            elif opdgasval == 0 and opd_mode_variable.get() and opdbrakeval < 0.3:
                if speed > 0:
                    b = max(opdbrakeval**0.8/2, 0.3)
                    opdbrakeval = max(opdbrakeval*((-1/(b*speed+1))+1)+a*(1-(-1/(b*speed+1)+1)),0)
                elif speed < 0:
                    b =  max(opdbrakeval**0.8/2,0.3)
                    opdbrakeval = max(opdbrakeval*((-1/(b*-speed+1))+1)+a*(1-(-1/(b*-speed+1)+1)),0)
            
            if data.get('userThrottle', 0.0) > 0.0:
                opdbrakeval = 0.0
            
            if data["cruiseControl"] and not cc_enabled:
                opdbrakeval = 0
            elif (cc_enabled and cc_mode.get() == "Cruise control"):
                if gasval <= 0.0:
                    opdbrakeval = cc_brake
                else:
                    opdbrakeval = 0.0
                opdgasval = max(cc_gas, opdgasval)
            elif (cc_enabled and cc_mode.get() == "Speed limiter"):
                if opdgasval > cc_gas and cc_gas == 1.0:
                    cc_limiting = True
                else:
                    cc_limiting = False
                opdbrakeval = max(cc_brake, opdbrakeval)
                opdgasval = min(cc_gas, opdgasval)
            
            if speed <= 0.1 and speed >= -0.1 and gasval == 0 and gear != 0 and not stopped:
                stopped = True
                prev_opdbrakeval = opdbrakeval
                prev_stop = time.time()
            elif stopped == True and (speed >= 4 and gear > 0 or speed <= -4 and gear < 0):
                stopped = False
                prev_stop = 0
            elif stopped and opdgasval > 0.75:
                stopped = False
                prev_stop = 0
            if data["parkBrake"] == True and speed <= 2 and speed >= -2 and not stopped:
                stopped = True
                prev_opdbrakeval = opdbrakeval
                prev_stop = time.time()

            gas_output = opdgasval
            brake_output = opdbrakeval
            # print(f"gas: {gas_output}\tbrake: {brake_output}\tstopped: {stopped}")

            #stopped = False

            if debug_mode.get() == True:
                print(f"gasvalue: {round(opdgasval,3)} \tbrakevalue: {round(opdbrakeval,3)} \tspeed: {round(speed,3)} \tstopped: {stopped} \tgasval: {round(gasval,3)} \tbrakeval: {round(brakeval,3)} \tdiff: {round(prev_brakeval-brakeval,3)} \tdiff2: {round(prev_speed-speed,3)} \thazards: {data['lightsHazards']} \thazards_var: {hazards_variable_var} \thazards_prompted: {hazards_prompted}")
            
            if ((prev_brakeval-brakeval <= -0.07*latency_multiplier or brakeval >= 0.8 or data["parkBrake"]) and stopped == False and speed > 10 and arrived == False and not pauzed):
                stopped = True
                em_stop = True
                send(0,1, controller)
                cmd_print("#####stopping#####", "#FF2020", 3)
                if prev_brakeval-brakeval <= -0.15*latency_multiplier and speed > 40:
                    horn = True
                    cmd_print("#####HONKING!#####", "#FF2020", 3)
            elif prev_speed-speed >= 5 and arrived == False and not pauzed:
                setattr(controller, "accmode", False)
                stopped = True
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
                while (brakeval > 0.8 or prev_brakeval-brakeval <= -0.03*latency_multiplier or data["parkBrake"]) and not exit_event.is_set():
                    if prev_brakeval-brakeval <= -0.15*latency_multiplier and horn == False:
                        cmd_print("#####/HONKING!\#####", "#FF2020", 3)
                        horn = True
                        if horn_variable_var == True:
                            setattr(controller, "wipers3", True)
                        if airhorn_variable_var == True:
                            setattr(controller, "wipers4", True)
                            
                    if check_wheel_connected() and device is not None:
                        if pygame.get_init():
                            pygame.init()
                            pygame.joystick.init()
                        for event in pygame.event.get():
                            if event.type == pygame.JOYAXISMOTION:
                                brakeval = round((device.get_axis(brakeaxis)*-1+1)/2,3)
                            if event.type == pygame.JOYDEVICEREMOVED:
                                if event.instance_id == device_instance_id:
                                    cmd_print("Your pedals disconnected", "#FF2020", 15)
                                    device_lost = True
                                    break
                    elif speed > 0.1:
                        brakeval = 1
                    else:
                        break
                    time.sleep(0.05)
                    prev_brakeval = brakeval
                    # prev_gasval = gasval
                    prev_speed = speed
                    data = truck_telemetry.get_data()
                    speed = round(data["speed"] * 3.6,3)
                    timestamp = time.time()
                    latency_timestamp = time.time()-0.005
                if device_lost == False:
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

            if autodisable_hazards_var == True and hazards_prompted == True and speed > 10 and data["lightsHazards"] == True and opdgasval > 0.5 and brakeval == 0:
                cmd_print("autodisabled hazards")
                setattr(controller, "accmode", True)
                hazards_prompted = False
                time.sleep(0.05)
                setattr(controller, "accmode", False)
            elif autodisable_hazards_var == False:
                hazards_prompted = False
            
            #added this to prevent the hazards_prompted from staying on and blocking the hazards activation
            if hazards_prompted == True and not data["lightsHazards"]:
                hazards_prompted = False

            prev_brakeval = brakeval
            prev_speed = speed

            # update the live visualization frame and enlarges it to the available space of the frame with a min size of 200x200 and a max size of 1000x1000

            # calculate the size of the frame
            '''
            frame_width = live_visualization_frame.winfo_width()-100
            frame_height = live_visualization_frame.winfo_height()-100
            frame_size = min(min(max(200, frame_width), 1000), min(max(200, frame_height), 1000)) '''
            image = overlay_dot_layer(gasval-brakeval, 200, 200, img, to_img_coords)
            live_visualization_frame.configure(image=image)
            live_visualization_frame.image = image

























        print("Main loop exited, cleaning up...")
        radar.cleanup()
        return
    except Exception as e:
        context = get_error_context()
        log_error(e, context)
        exit_event.set()
        radar.cleanup()
        raise e  # Re-raise the exception to be caught by the main thread

def game_thread():
    cmd_print("Game thread starting...")
    try:
        # Wait for UI to be ready
        ui_ready.wait()
        cmd_print("UI is ready, game logic starting...")
        main()
        print("Game thread finished execution.")
    except Exception as e:
        context = get_error_context()
        log_error(e, context)
        exit_event.set()

class AnimatedBar:
    global cc_enabled, cc_locked #################################################### still to be defined
    def __init__(self, root):
        self.root = root
        self.temp_gasval = 0
        self.temp_brakeval = 0
        self.bar_width = 7         # Height (in pixels) of the bar.
        self.transparent_color = "magenta"
        
        # Remove window decorations and force window always on top.
        self.root.overrideredirect(True)
        
        # Set the window background (and canvas background) to the transparent key color.
        self.root.config(bg=self.transparent_color)
        self.root.wm_attributes("-transparentcolor", self.transparent_color)
        self.root.wm_attributes("-toolwindow", True)
        self.root.wm_attributes("-topmost", True)  # Keep the window on top
        self.root.wm_attributes("-disabled", False)  # Disable user interaction
        self.root.wm_attributes("-alpha", 0.9) # Set the window transparency to 80%
        
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
        global em_stop, gas_output, brake_output, bar_variable, close_bar_event

        #if the screen changes size, update the size of the bar
        if self.screen_width != self.root.winfo_screenwidth() or self.screen_height != self.root.winfo_screenheight():
            self.screen_width = self.root.winfo_screenwidth()
            self.screen_height = self.root.winfo_screenheight()
            self.root.geometry(f"{self.screen_width}x{self.bar_width}+0+{self.screen_height - self.bar_width}")

        if exit_event.is_set() or close_bar_event.is_set():
            self.root.destroy()
            return
    
        
        self.root.lift()
        if cc_enabled and cc_locked:
            average = 20
        else:
            average = 10
        temp_gasval = (gas_output+self.temp_gasval*average)/(average+1)
        temp_brakeval = (brake_output+self.temp_brakeval*average)/(average+1)

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
                # Make sure the gas bar is visible if it's needed.
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

def reset_operational_variables():
    """
    Resets all operational variables to their initial state after device reconnection.
    This prevents issues with stale data from before the disconnection.
    """
    global prev_brakeval, prev_speed, speed, opdgasval, opdbrakeval, gas_output, brake_output
    global prev_opdbrakeval, arrived, stopped, horn, em_stop, latency_timestamp
    global latency, hazards_prompted, offset, prev_stop, gasval, brakeval, bar_val
    
    # Reset brake and speed tracking variables
    prev_brakeval = 0
    prev_speed = 0
    speed = 0
    prev_opdbrakeval = 0
    
    # Reset pedal values
    brakeval = 0
    gasval = 0
    
    # Reset one-pedal drive outputs
    opdgasval = 0
    opdbrakeval = 0
    gas_output = 0
    brake_output = 0
    prev_opdbrakeval = 0
    bar_val = 0
    
    # Reset game state variables
    arrived = False
    stopped = False
    horn = False
    em_stop = False
    
    # Reset timing variables
    latency_timestamp = time.time() - 0.015
    latency = 0.015
    prev_stop = time.time()

    setattr(controller, "steering", 0.0)
    setattr(controller, "accmode", False)
    setattr(controller, "wipers4", False)
    setattr(controller, "wipers3", False)


global device
global device_instance_id
global device_lost
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
global data
global pauze_pedal_detection
global cc_start_button
global cc_inc_button
global cc_dec_button
global unassign

cc_enabled = False
cc_locked = False
data = None
cmd_label = None
device = None
buttons_thread = None
device_instance_id = 0
device_lost = False
pauze_pedal_detection = False
unassign = False
global controller
global connected_joystick_label

#checking if MonoCruise is already running
if is_process_running("MonoCruise.exe"):
    sys.exit()

cmd_print("Starting MonoCruise...")
try:
    # load from save file
    save_file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "saves.json")
    load_variables(save_file_path)

    # load from save file
    try:
        gasaxis = _data_cache["gasaxis"]
    except:
        gasaxis = 0
    try:
        brakeaxis = _data_cache["brakeaxis"]
    except:
        brakeaxis = 0
    try:
        gas_inverted = _data_cache["gas_inverted"]
    except:
        gas_inverted = False
    try:
        brake_inverted = _data_cache["brake_inverted"]
    except:
        brake_inverted = False
    try:
        cc_start_button = _data_cache["cc_start_button"]
    except:
        cc_start_button = None
    try:
        cc_inc_button = _data_cache["cc_inc_button"]
    except:
        cc_inc_button = None
    try:
        cc_dec_button = _data_cache["cc_dec_button"]
    except:
        cc_dec_button = None

    pygame.init()
    pygame.joystick.init()

    controller = SCSController()

    #reset the all used sdk variables
    setattr(controller, "accmode", False)
    setattr(controller, "wipers4", False)
    setattr(controller, "wipers3", False)
    
    setattr(controller, "steering", 0.0)
    setattr(controller, "aforward", 0.0)
    setattr(controller, "abackward", 0.0)

    root.title("MonoCruise")
    try:
        root.iconbitmap(os.path.join(os.path.dirname(os.path.abspath(__file__)), "icon.ico"))
    except:
        pass  # Ignore if icon file not found or on non-Windows platform
    
    # Apply scaling to window sizes
    base_width = 700
    base_height = 500
    root.geometry(f"{int(base_width)}x{int(base_height)}")
    root.minsize(int(base_width), int(base_height))
    
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
    scrollable_frame.pack(side="top", fill="both", expand=True, padx=5, pady=(5,5))
    #make the grid expand to the side of the scrollable frame
    scrollable_frame.grid_columnconfigure(1, weight=1)
    scrollable_frame.pack_propagate(False)


    #start of the settings

    def refresh_live_visualization():
        global img
        global to_img_coords
        global img_copy
        global live_visualization_frame
        img, to_img_coords = plot_onepedaldrive(return_result=True)
        pil_copy = img.copy()
        new_width = 200
        new_height = 200
        img_copy = ctk.CTkImage(pil_copy.resize((new_width, new_height), Image.LANCZOS), size=(new_width, new_height))
        live_visualization_frame.configure(image=img_copy)
        live_visualization_frame.image = img_copy

    def new_checkbutton(master, row, column, variable, command=None):
        checkbutton = ctk.CTkCheckBox(master, text="", command=command, font=default_font, text_color="lightgrey", fg_color=SETTINGS_COLOR, corner_radius=5, variable=variable, checkbox_width=20, checkbox_height=20, width=24, height=24, border_color=SETTINGS_COLOR, border_width=1.5)
        checkbutton.grid(row=row, column=column, padx=5, pady=1, sticky="e")
        return checkbutton

    def new_label(master, row, column, text):
        label = ctk.CTkLabel(master, text=text, font=default_font, text_color="lightgrey")
        label.grid(row=row, column=column, padx=10, pady=1, sticky="w")
        return label
    
    def new_entry(master, row, column, textvariable, temp_textvariable, command=None, max_value=None, min_value=None):
        # Keep this variable accessible—can be a global or instance variable depending on context
        timer_id = None
        def remove_focus(*args):
            master.focus()

        def entry_wait(*args):
            nonlocal timer_id  # If inside a closure; or use global if defined globally

            def validate():
                try:
                    val = float(temp_textvariable.get())
                    if val < min_value:
                        temp_textvariable.set(min_value)  # Revert to min if too low
                    elif val > max_value:
                        temp_textvariable.set(max_value)  # Revert to max if too high
                    else:
                        textvariable.set(val)  # Apply value only if valid
                except ValueError:
                    temp_textvariable.set(textvariable.get())  # Revert if input isn't a number
                finally:
                    remove_focus() # remove focus on entry field
                    if command is not None:
                        command()

            # Cancel the previous timer if it exists
            if timer_id is not None:
                master.after_cancel(timer_id)

            # Set a new timer and save its ID
            timer_id = master.after(2000, validate)

        temp_textvariable.trace_add("write", entry_wait)  # Monitor changes in temp_textvariable

        entry = ctk.CTkEntry(
            master,
            textvariable=temp_textvariable,
            font=default_font,
            text_color="lightgrey",
            fg_color=DEFAULT_COLOR,
            corner_radius=5,
            width=50,
            border_width=1.5,
            border_color=SETTINGS_COLOR
        )
        entry.grid(row=row, column=column, padx=10, pady=1, sticky="e")
        entry.bind("<Return>", remove_focus)
        entry.bind("<Escape>", remove_focus)
        
        return entry

    def new_optionmenu(master, row, column, values, value, default_value=None, command=None):
        """
        Creates a pre-configured CTkOptionMenu with automatic value extraction and storage,
        wrapped in a bordered frame. Returns the numeric value directly.
        
        Args:
            master: Parent widget
            row: Grid row position
            column: Grid column position
            values: List of option values (e.g., ["1 km/h", "3 km/h", "5 km/h"])
            value: ctk.StringVar to hold the selected value
            default_value: values to set at start (optional)
            command: Optional callback function to execute after value change
        
        Returns:
            optionmenu_widget: The created CTkOptionMenu widget
        """
        def remove_focus(*args):
            master.focus()
        
        # Set default value if not provided
        if default_value is None:
            default_value = values[0]
        
        # Create StringVar for the option menu
        if value is not None:
            # If a value is provided, use it to set the default value
            if value.get() not in values:
                cmd_print(f"{value} not in allowed options. value set to default.", "#FF2020", 5)
                value.set(value=default_value)
        elif value is None:
            value = ctk.StringVar(value=default_value)
        
        def update_value(selected_value):
            # Execute additional command if provided
            if command is not None:
                command()
        
        # Create bordered frame
        frame = ctk.CTkFrame(
            master,
            border_width=1.5,
            border_color=SETTINGS_COLOR,
            fg_color="transparent",
            corner_radius=5
        )
        
        # Grid the frame
        frame.grid(row=row, column=column, padx=(0,8), pady=1, sticky="e")
        
        # Create the option menu inside the frame
        optionmenu = ctk.CTkOptionMenu(
            frame,
            values=values,
            variable=value,
            command=update_value,
            width=100, 
            fg_color=DEFAULT_COLOR,
            button_color=DEFAULT_COLOR,
            button_hover_color=DEFAULT_COLOR,
            corner_radius=3.5,
            bg_color=SETTINGS_COLOR
        )
        
        # Grid the option menu inside the frame with some padding
        optionmenu.grid(row=0, column=0, padx=1.5, pady=1.5)
        
        # Bind events
        optionmenu.bind("<Return>", remove_focus)
        optionmenu.bind("<Escape>", remove_focus)
        
        return optionmenu




    # Create a label for input settings
    settings_label_1 = ctk.CTkLabel(scrollable_frame, text="Inputs", font=default_font_bold, text_color="lightgrey", fg_color=SETTING_HEADERS_COLOR, corner_radius=5)
    settings_label_1.grid(row=0, column=0, padx=0, pady=1, columnspan=2, sticky="new")

    connected_joystick_label = new_label(scrollable_frame, 1, 0, "Connected pedals:")

    try:
        connected_joystick_label = ctk.CTkLabel(scrollable_frame, text=f"{device.get_name()[:20]}...", font=default_font, text_color="lightgrey", fg_color=VAR_LABEL_COLOR, corner_radius=5)
    except:
        connected_joystick_label = ctk.CTkLabel(scrollable_frame, text="None connected", font=default_font, text_color="lightgrey", fg_color=VAR_LABEL_COLOR, corner_radius=5)
    connected_joystick_label.grid(row=1, column=1, padx=10, pady=1, sticky="e")

    connected_joystick_gas_axis_label = new_label(scrollable_frame, 2, 0, "Gas axis:")

    connected_joystick_gas_axis_label = ctk.CTkLabel(scrollable_frame, text="None" if gasaxis == 0 and device != 0 else f"axis {gasaxis}", font=default_font, text_color="lightgrey", fg_color=VAR_LABEL_COLOR, corner_radius=5)
    connected_joystick_gas_axis_label.grid(row=2, column=1, padx=10, pady=1, sticky="e")

    connected_joystick_brake_axis_label = new_label(scrollable_frame, 3, 0, "Brake axis:")

    connected_joystick_brake_axis_label = ctk.CTkLabel(scrollable_frame, text="None" if brakeaxis == 0 and device != 0 else f"axis {brakeaxis}", font=default_font, text_color="lightgrey", fg_color=VAR_LABEL_COLOR, corner_radius=5)
    connected_joystick_brake_axis_label.grid(row=3, column=1, padx=10, pady=1, sticky="e")

    restart_connection_button = ctk.CTkButton(scrollable_frame, text="connect to pedals", font=default_font, text_color="lightgrey", fg_color=WAITING_COLOR, corner_radius=5, hover_color="#333366", command=lambda: threading.Thread(target=connect_joystick, name="connect joystick thread").start())
    restart_connection_button.grid(row=4, column=0, padx=40, pady=(1,0), columnspan=2, sticky="ew")

    restart_connection_label = ctk.CTkLabel(scrollable_frame, text="", font=default_font, text_color="darkred", corner_radius=5)
    restart_connection_label.grid(row=5, column=0, padx=10, pady=(0,1), columnspan=2, sticky="ew")




    # create a label for program settings
    settings_label_2 = ctk.CTkLabel(scrollable_frame, text="Program settings", font=default_font_bold, text_color="lightgrey", fg_color=SETTING_HEADERS_COLOR, corner_radius=5)
    settings_label_2.grid(row=6, column=0, padx=0, pady=1, columnspan=2, sticky="new")

    autostart_variable = ctk.BooleanVar(value=True)
    autostart_MonoCruise_label = new_label(scrollable_frame,7 ,0 , "Autostart MonoCruise:")
    autostart_MonoCruise_button = new_checkbutton(scrollable_frame, 7, 1, autostart_variable)

    input_polling_rate_label = new_label(scrollable_frame, 8, 0, "Target polling rate (Hz):")

    polling_rate = ctk.IntVar(value=50)
    temp_polling_rate = ctk.IntVar(value=50)
    input_polling_rate_label = new_entry(scrollable_frame, 8, 1, polling_rate, temp_polling_rate, max_value=100, min_value=10)

    description_label = ctk.CTkLabel(scrollable_frame, text="lower values mean more input lag, higher values mean more cpu usage", font=("Segoe UI", 11), text_color="#606060", fg_color="transparent", corner_radius=5, bg_color="transparent", anchor="e", wraplength=185, height=10, justify="right")
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
                bar_thread = threading.Thread(target=lambda: start_bar_thread(), daemon=True, name="bar Thread")
            bar_thread.start()
        else:
            
            close_bar_event.set()
            if bar_thread is not None:
                bar_thread.join(timeout=2)
                if bar_thread.is_alive():
                    raise RuntimeError("Failed to close the bar thread gracefully.")
                    
                bar_thread = None

    bar_variable = ctk.BooleanVar(value=True)
    bar_label = new_label(scrollable_frame, 16, 0, "Live bottom bar:")
    bar_checkbutton = new_checkbutton(scrollable_frame, 16, 1, bar_variable, command=bar_var_update)

    def detect_cc_button(button_type):
        global device
        global buttons_thread
        global close_buttons_threads
        global cc_start_button
        global cc_inc_button
        global cc_dec_button
        global cc_start_label
        global cc_inc_label
        global cc_dec_label

        if device is None:
            cmd_print("No joystick connected. Please connect your pedals first.", "#FF2020", 5)
            return

        if buttons_thread is not None and buttons_thread.is_alive():
            close_buttons_threads.set()
            cc_dec_label.configure(border_color=SETTINGS_COLOR)
            cc_inc_label.configure(border_color=SETTINGS_COLOR)
            cc_start_label.configure(border_color=SETTINGS_COLOR)
            
            # Force unhook keyboard listeners before joining
            try:
                keyboard.unhook_all()
            except:
                pass
                
            while close_buttons_threads.is_set():
                time.sleep(0.1)
        else:
            cc_dec_label.configure(border_color=SETTINGS_COLOR)
            cc_inc_label.configure(border_color=SETTINGS_COLOR)
            cc_start_label.configure(border_color=SETTINGS_COLOR)
            close_buttons_threads.clear()
            # Clean up any lingering keyboard hooks
            try:
                keyboard.unhook_all()
            except:
                pass
            
        buttons_thread = threading.Thread(target=detect_joystick_movement, daemon=True, args=(button_type,), name="buttons Thread")
        buttons_thread.start()
        unassign_button.configure(state="normal")


    cc_title = ctk.CTkLabel(scrollable_frame, text="Cruise Control", font=default_font_bold, text_color="lightgrey", fg_color=SETTING_HEADERS_COLOR, corner_radius=5)
    cc_title.grid(row=19, column=0, padx=0, pady=(20,3), columnspan=2, sticky="new")
    # create a label for the modes
    ctk.CTkLabel(scrollable_frame, text="Mode:", font=default_font, text_color="lightgrey").grid(row=21, column=0, padx=10, pady=0, sticky="w", columnspan=2)

    cc_mode = ctk.StringVar(value="Cruise Control")
    # Create border frame with 1.5px border
    cc_mode_border_frame = ctk.CTkFrame(scrollable_frame,
                                    fg_color=DEFAULT_COLOR,
                                    border_color=SETTINGS_COLOR,
                                    border_width=1.5,
                                    corner_radius=7)

    cc_mode_border_frame.grid(row=21, column=0, padx=10, pady=(8,8), columnspan=2, sticky="e")

    # Create the segmented button inside the border frame
    cc_mode_segmented_button = ctk.CTkSegmentedButton(cc_mode_border_frame, values=["Cruise control", "Speed limiter"],
                                                      variable=cc_mode, dynamic_resizing=False, width=220, selected_hover_color=WAITING_COLOR,
                                                      selected_color=WAITING_COLOR, text_color="lightgrey",bg_color="transparent", 
                                                      corner_radius=5, font=default_font, fg_color=DEFAULT_COLOR, unselected_color=DEFAULT_COLOR, border_width=1.5)
    cc_mode_segmented_button.grid(row=0, column=0, padx=3, pady=3, sticky="nsew")
    cc_mode.set("Cruise control")  # Set default value

    new_label(scrollable_frame, 22, 0, "Enable/Disable button:")

    cc_start_label = ctk.CTkButton(scrollable_frame, text="None", font=default_font, text_color=SETTINGS_COLOR, width=150, fg_color=VAR_LABEL_COLOR, corner_radius=5, command=lambda: detect_cc_button("start"), border_width=1.5, border_color=SETTINGS_COLOR)
    cc_start_label.grid(row=22, column=1, padx=10, pady=1, sticky="e")

    new_label(scrollable_frame, 23, 0, "Increase button:")

    cc_inc_label = ctk.CTkButton(scrollable_frame, text="None", font=default_font, text_color=SETTINGS_COLOR, width=150, fg_color=VAR_LABEL_COLOR, corner_radius=5, command=lambda: detect_cc_button("inc"), border_width=1.5, border_color=SETTINGS_COLOR)
    cc_inc_label.grid(row=23, column=1, padx=10, pady=1, sticky="e")

    new_label(scrollable_frame, 24, 0, "Decrease button:")

    cc_dec_label = ctk.CTkButton(scrollable_frame, text="None", font=default_font, text_color=SETTINGS_COLOR, width=150, fg_color=VAR_LABEL_COLOR, corner_radius=5, command=lambda: detect_cc_button("dec"), border_width=1.5, border_color=SETTINGS_COLOR)
    cc_dec_label.grid(row=24, column=1, padx=10, pady=1, sticky="e")

    def unassign_true():
        global unassign
        unassign = True

    unassign_button = ctk.CTkButton(scrollable_frame,width=150, text="Unassign", font=default_font, text_color="lightgrey", fg_color=WAITING_COLOR, corner_radius=5, hover_color="#333366", command=unassign_true, state="disabled")
    unassign_button.grid(row=25, column=0, padx=10, pady=(1,8), columnspan=2, sticky="e")

    short_increments = ctk.StringVar(value=_data_cache["short_increments"] if "short_increments" in _data_cache else "1 km/h")
    short_press_increments_label = new_label(scrollable_frame, 26, 0, "Short press increments:")
    short_press_increments_options = new_optionmenu(
        scrollable_frame, 26, 1, 
        values=["1 km/h", "2 km/h", "3 km/h", "5 km/h", "10 km/h"], 
        default_value="1 km/h",
        value=short_increments
    )

    long_increments = ctk.StringVar(value=_data_cache["long_increments"] if "long_increments" in _data_cache else "5 km/h")
    long_press_increments_label = new_label(scrollable_frame, 27, 0, "Long press increments:")
    long_press_increments_options = new_optionmenu(
        scrollable_frame, 27, 1, 
        values=["1 km/h", "2 km/h", "3 km/h", "5 km/h", "10 km/h"], 
        default_value="5 km/h",
        value=long_increments,
    )

    long_press_reset_label = new_label(scrollable_frame, 28, 0, "Hold enable to reset:")
    long_press_reset = ctk.BooleanVar(value=True)
    long_press_reset_button = new_checkbutton(
        scrollable_frame, 28, 1,
        long_press_reset)

    show_cc_ui_label = new_label(scrollable_frame, 29, 0, "Show set speed on screen:")
    show_cc_ui = ctk.BooleanVar(value=True)
    show_cc_ui_button = new_checkbutton(
        scrollable_frame, 29, 1,
        show_cc_ui)
    ctk.CTkLabel(scrollable_frame, text="just drag it across the screen to move", font=("Segoe UI", 11), text_color="#606060", fg_color="transparent", corner_radius=5, bg_color="transparent", anchor="e", wraplength=185, height=10, justify="right").grid(row=30, column=0, padx=10, pady=(0,8), columnspan=2, sticky="nsew")








    # create a title for the one-pedal-drive system
    opd_title = ctk.CTkLabel(scrollable_frame, text="One-Pedal-Drive", font=default_font_bold, text_color="lightgrey", fg_color=SETTING_HEADERS_COLOR, corner_radius=5)
    opd_title.grid(row=37, column=0, padx=0, pady=(20,3), columnspan=2, sticky="new")

    # create a label for the one-pedal-drive system
    opd_mode_label = new_label(scrollable_frame, 38, 0, "One Pedal Drive mode:")

    def opd_mode_var_update():
        if opd_mode_variable.get() == True:
            offset_label.grid(row=39, column=0, padx=10, pady=1, sticky="w")
            offset_entry.grid(row=39, column=1, padx=10, pady=1, sticky="e")
            description_offset_label.grid(row=40, column=0, padx=10, pady=(0,8), columnspan=2, sticky="nsew")
            max_opd_brake_label.grid(row=41, column=0, padx=10, pady=1, sticky="w")
            max_opd_brake_entry.grid(row=41, column=1, padx=10, pady=1, sticky="e")
            description_max_opd_label.grid(row=42, column=0, padx=10, pady=(0,8), columnspan=2, sticky="nsew")
            refresh_live_visualization()
        else:
            offset_label.grid_forget()
            offset_entry.grid_forget()
            description_offset_label.grid_forget()
            max_opd_brake_label.grid_forget()
            max_opd_brake_entry.grid_forget()
            description_max_opd_label.grid_forget()
            refresh_live_visualization()

    opd_mode_variable = ctk.BooleanVar(value=True)
    opd_mode_checkbutton = new_checkbutton(scrollable_frame, 38, 1, opd_mode_variable, command=opd_mode_var_update)

    offset_label = new_label(scrollable_frame, 39, 0, "  Offset:")
    offset_variable = ctk.DoubleVar(value=0.2)
    temp_offset_variable = ctk.DoubleVar(value=0.2)
    offset_entry = new_entry(scrollable_frame, 39, 1, offset_variable, temp_offset_variable, command=refresh_live_visualization, max_value=0.5, min_value=0)

    description_offset_label = ctk.CTkLabel(scrollable_frame, text="The amount you have to press the gas to not be braking or accelerating", font=("Segoe UI", 11), text_color="#606060", fg_color="transparent", corner_radius=5, bg_color="transparent", anchor="e", wraplength=185, height=10, justify="right")
    description_offset_label.grid(row=40, column=0, padx=10, pady=(0,8), columnspan=2, sticky="nsew")

    max_opd_brake_label = new_label(scrollable_frame, 41, 0, "  Max OPD brake:")
    max_opd_brake_variable = ctk.DoubleVar(value=0.03)
    temp_max_opd_brake_variable = ctk.DoubleVar(value=0.03)
    max_opd_brake_entry = new_entry(scrollable_frame, 41, 1, max_opd_brake_variable, temp_max_opd_brake_variable, command=refresh_live_visualization, max_value=0.2, min_value=0)

    description_max_opd_label = ctk.CTkLabel(scrollable_frame, text="The amount of braking when not touching the pedals", font=("Segoe UI", 11), text_color="#606060", fg_color="transparent", corner_radius=5, bg_color="transparent", anchor="e", wraplength=185, height=10, justify="right")
    description_max_opd_label.grid(row=42, column=0, padx=10, pady=(0,8), columnspan=2, sticky="nsew")

    gas_exponent_label = new_label(scrollable_frame, 43, 0, "Gas exponent:")
    gas_exponent_variable = ctk.DoubleVar(value=2)
    temp_gas_exponent_variable = ctk.DoubleVar(value=2)
    gas_exponent_entry = new_entry(scrollable_frame, 43, 1, gas_exponent_variable, temp_gas_exponent_variable, command=refresh_live_visualization, max_value=2.5, min_value=0.8)

    brake_exponent_label = new_label(scrollable_frame, 44, 0, "Brake exponent:")
    brake_exponent_variable = ctk.DoubleVar(value=2)
    temp_brake_exponent_variable = ctk.DoubleVar(value=2)
    brake_exponent_entry = new_entry(scrollable_frame, 44, 1, brake_exponent_variable, temp_brake_exponent_variable, command=refresh_live_visualization, max_value=2.5, min_value=0.8)

    new_label(scrollable_frame, 45, 0, "Weight adjustment brake:")
    weight_adjustment = ctk.BooleanVar(value=True)
    opd_mode_checkbutton = new_checkbutton(scrollable_frame, 45, 1, weight_adjustment)


    # list of implemented libraries shown as a discription
    implemented_libraries_label = ctk.CTkLabel(scrollable_frame, text="Implemented libraries:",  font=("Segoe UI", 11), text_color="#606060", fg_color="transparent", corner_radius=5, height=0)
    implemented_libraries_label.grid(row=49, column=0, padx=10, pady=(10,0), columnspan=2, sticky="new")

    SCSController_label = ctk.CTkLabel(scrollable_frame, text="SCSController - tumppi066",  font=("Segoe UI", 11), text_color="#606060", fg_color="transparent", corner_radius=5, height=0)
    SCSController_label.grid(row=50, column=0, padx=10, pady=0, columnspan=2, sticky="new")

    pygame_label = ctk.CTkLabel(scrollable_frame, text="pygame - pygame",  font=("Segoe UI", 11), text_color="#606060", fg_color="transparent", corner_radius=5, height=0)
    pygame_label.grid(row=51, column=0, padx=10, pady=0, columnspan=2, sticky="new")

    truck_telemetry_label = ctk.CTkLabel(scrollable_frame, text="Truck telemetry - Dreagonmon",  font=("Segoe UI", 11), text_color="#606060", fg_color="transparent", corner_radius=5, height=0)
    truck_telemetry_label.grid(row=52, column=0, padx=10, pady=0, columnspan=2, sticky="new")

    customtkinter_label = ctk.CTkLabel(scrollable_frame, text="customtkinter - csm10495",  font=("Segoe UI", 11), text_color="#606060", fg_color="transparent", corner_radius=5, height=0)
    customtkinter_label.grid(row=53, column=0, padx=10, pady=0, columnspan=2, sticky="new")

    #create a button to reinstall the SDK
    reinstall_SDK_button = ctk.CTkButton(
        scrollable_frame,
        text="reinstall SDK",
        font=default_font,
        text_color="lightgrey",
        fg_color=WAITING_COLOR,
        corner_radius=5,
        hover_color="#333366",
        command=lambda: threading.Thread(target=check_and_install_scs_sdk, daemon=True, name="reintall SDK").start()
    )
    reinstall_SDK_button.grid(row=54, column=0, padx=10, pady=(20,1), columnspan=2, sticky="ew")


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
    reset_button.grid(row=55, column=0, padx=10, pady=(5,1), columnspan=2, sticky="ew")

    # discription for the reset button
    reset_button_description = ctk.CTkLabel(scrollable_frame, text="this requires a program restart", font=("Segoe UI", 11), text_color="#606060", fg_color="transparent", corner_radius=5, bg_color="transparent", anchor="center", wraplength=165, height=10)
    reset_button_description.grid(row=56, column=0, padx=10, pady=(0,4), columnspan=2, sticky="nsew")





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

    ctk.CTkLabel(main_frame, text=version, font=("Segoe UI", 11), text_color="#505050", fg_color="transparent", corner_radius=5, bg_color="transparent", anchor="e", wraplength=185, height=10, justify="right").pack(side="bottom", padx=0, pady=5, anchor="se")


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
        if not ui_state.transition_active or exit_event.is_set():
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
        if not isinstance(_data_cache["device"], str) or _data_cache["device"] == "" or gasaxis == 0 or brakeaxis == 0:
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
        # WAITING vs RUNNING MODE vs LOST MODE:
        # When gasaxis and brakeaxis are nonzero, check ETS2 connection. If the pedals are disconnected, set the mode to lost.
        # ---------------------------
        if ets2_detected.is_set() and device_lost == False:
            current_mode = "running"
            target_color = CONNECTED_COLOR
        elif device_lost == True:
            current_mode = "lost"
            target_color = LOST_COLOR
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
        elif current_mode == "lost":
            if dots_anim.is_playing:
                dots_anim.stop()
            loading_label.configure(
                text="Pedals disconnected",
                font=default_font_bold,
                text_color="white",
                bg_color="transparent"
            )

        ui_state.last_device = device

        if not exit_event.is_set():
            root.after(30, update_ui_state)

    def on_root_close():

        cmd_print("UI closing, cleaning up threads...")

        # Stop the dots animation if it's still running
        if dots_anim.is_playing: # and dots_anim (testing)
            dots_anim.stop()
        # Set the exit event to stop all threads
        print("UI closed, quitting MonoCruise")
        exit_event.set()

        time.sleep(0.1) # Give threads time to finish

        # Clean up pygame
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
        debug_mode.set(_data_cache["debug_mode"])
    except Exception: pass
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
        weight_adjustment.set(_data_cache["weight_adjustment"])
    except Exception: pass
    try:
        cc_mode.set(_data_cache["cc_mode"])
    except Exception: pass
    try:
        short_increments.set(_data_cache["short_increments"])
    except Exception: pass
    try:
        long_increments.set(_data_cache["long_increments"])
    except Exception: pass
    try:
        long_press_reset.set(_data_cache["long_press_reset"])
    except Exception: pass
    try:
        show_cc_ui.set(_data_cache["show_cc_ui"])
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
    thread_game = threading.Thread(target=game_thread, daemon=True, name="MonoCruise Game Thread")
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
    
    print("Mainloop exited")
    # When mainloop exits, set exit event
    exit_event.set()
    
    # Clean up pygame
    if pygame.get_init():
        pygame.quit()

    print("Cleaning up threads...")

    # join all threads to ensure they are cleaned up
    if thread_game is not None and thread_game.is_alive():
        print("Joining game thread...")
        thread_game.join(timeout=1)  # Ensure the game thread is cleaned up
        if thread_game.is_alive():
            print("Game thread did not finish in time, forcing exit.")
        else:
            print("Game thread joined.")
        thread_game = None
    if bar_thread is not None and bar_thread.is_alive():
        print("Joining bar thread...")
        bar_thread.join(timeout=1)  # Ensure the bar thread is cleaned up
    if dots_anim.is_playing:
        dots_anim.stop()
    if bar_thread is not None and bar_thread.is_alive():
        bar_thread.join(timeout=1)  # Ensure the bar thread is cleaned up
    # Close the controller
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