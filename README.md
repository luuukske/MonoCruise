<a href="https://sourceforge.net/p/monocruise/"><img alt="Download MonoCruise" src="https://sourceforge.net/sflogo.php?type=18&amp;group_id=3904914" width=150></a>
[![Download MonoCruise](https://img.shields.io/sourceforge/dw/monocruise.svg)](https://sourceforge.net/projects/monocruise/files/latest/download)
[![Download MonoCruise](https://img.shields.io/sourceforge/dt/monocruise.svg)](https://sourceforge.net/projects/monocruise/files/latest/download)

## Visit the official site for accurate and up-to-date showcases and information: https://ld-tech.org/projects/monocruise/

> [!WARNING]
> v1.1.0 is a major architecture rewrite and is still being stabilized.
> Some features may behave differently from v1.0.x. Use with care.

# MonoCruise
MonoCruise is a third-party software that sits in between ETS2/ATS and your pedals.
MonoCruise has a ton of quality of life features, like a better Adaptive Cruise Control, Automatic Emergency Braking, or a One-Pedal Driving system for heavy traffic.
Every feature (including ACC and AEB) works in TruckersMP and singleplayer ETS2/ATS.

![image_2025-07-02_202137925](https://github.com/user-attachments/assets/0b35aa19-340f-44a9-8e8b-0493c9cd30ca)

### features

**Cruise & speed control**
- Adaptive Cruise Control (ACC): holds a safe following distance from the lead vehicle using an intelligent driver model
- ACC gap level adjustment: 4 gap levels, assignable to buttons
- Traditional Cruise Control with speed limiter mode
- Short and long speed increment/decrement buttons (configurable step sizes)

**Safety**
- Automatic Emergency Braking (AEB): detects imminent collisions using arc-trajectory geometry and applies a two-phase brake sequence
- Emergency stop detection: full brake lock on sudden pedal slam or crash

**Pedal & driving feel**
- One-Pedal Driving system: combined throttle/brake on a single axis
- Exponential braking and accelerating: configurable non-linear pedal curves
- Adaptive pedal capacity learning: calibrates to your actual pedal hardware over time
- Smooth unified pedal output with road-load feedforward and gearshift freeze

**Comfort & automation**
- Auto start and stop for non-intrusive UX
- Automatically horn when braking hard
- Auto enable hazard lights when braking for traffic, auto disable on acceleration
- Live braking and accelerating bar on the bottom of the screen

**Input**
- Multi-device button support: joystick buttons, hat directions, and keyboard keys
- Automatic pedal reconnect: recovers gracefully if your pedals disconnect mid-drive
## .exe install

1. Download and run "MonoCruise installer.exe".

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<a href="https://sourceforge.net/projects/monocruise/files/latest/download"><img alt="Download MonoCruise" src="https://a.fsdn.com/con/app/sf-download-button" width=276 height=48 srcset="https://a.fsdn.com/con/app/sf-download-button?button_size=2x 2x"></a>

2. Run MonoCruise.
3. Follow the on-screen instructions.

   this will install the required SDK files to communicate with the game code.
4. Press ok or enter when asked for SDK confirmation.

   <img src="https://github.com/user-attachments/assets/76c706de-60b6-457c-ae78-0dc6185810df" alt="Alt text" width="400"/>

5. Wait for MonoCruise to connect.
6. Open the settings tab on the MonoCruise window.
7. Press "Connect to pedals".
   
   <img src="https://github.com/user-attachments/assets/b4b010d3-e3b6-4abf-a29a-a1a9fa72668c" alt="Alt text" width="400"/>

8. Press your brake pedal.
9. Press your gas pedal.

### set up the cruise control (optional):
10. scroll down to the cruise control settings
11. press the button next to the button you want to assign

       <img src="https://github.com/user-attachments/assets/e38a6fc1-2ce7-4cd7-8b48-d0e6aba333e6" alt="Alt text" width="400"/>

12. press your key/button you want to be assigned

Now you're done and can use MonoCruise in ETS2.

MonoCruise will automatically start together with ETS2. you can disable this in the settings.

> [!IMPORTANT]
> The MonoCruise window should remain open if you want to use it

## Adaptive Cruise Control (v1.1.0 and above):
Enable the ACC and it will hold a safe following distance from the lead vehicle in singleplayer or TruckersMP. The ACC uses an intelligent driver model (IIDM) to select and track the closest in-lane vehicle, and scores multiple candidates to pick the correct lead. The following gap has 4 levels, set from "Following distance" under Adaptive Cruise Control in the settings, or changed while driving with the optional ACC distance buttons (bind just one and it cycles through all four).

> [!CAUTION]
> The ACC is EXTREMELY experimental.
> The ACC has a tendency to brakecheck. BE MINDFUL WHEN TURNING ON!

## Automatic Emergency Braking (v1.1.0 and above):
The AEB monitors traffic using arc-based trajectory geometry and applies a staged brake intervention if a collision is imminent. Phase 1 applies ~50% deceleration for 0.5 s as a warning; Phase 2 applies ~90% deceleration if the threat persists. An audible beep plays at the start of each intervention. You can override AEB at any time by pressing the brake pedal yourself.

> [!CAUTION]
> AEB is EXTREMELY experimental.
> False positives can occur at intersections and during lane changes. BE MINDFUL WHEN TURNING ON!

## .py install
Not supported yet, but you can try it.

## uses:
- [ETS2LA plugin](https://gitlab.com/ETS2LA/ets2la_plugin): used for getting AI/MP vehicle data for ACC and AEB.
- [Truck_Telemetry](https://github.com/dreagonmon/truck_telemetry): used to get telemetry data from the game.
- [scscontroller](https://github.com/ETS2LA/scs-sdk-controller/tree/main): used to send commands to the game like braking, gas, hazards, etc.
- [PySide6](https://doc.qt.io/qtforpython-6/): used as the UI framework.
- [pygame](https://github.com/pygame/pygame): used to get pedal values and to play sounds.
- [Shapely](https://shapely.readthedocs.io/): used for arc-trajectory collision geometry in AEB.

This project is licensed under the MIT License.
It includes third-party code under the CC0-1.0, MIT, and BSD 3-Clause licenses.

See [MonoCruise/THIRD_PARTY_LICENSES/](https://github.com/luuukske/MonoCruise/tree/main/THIRD_PARTY_LICENSES) for details.


