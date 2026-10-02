<a href="https://sourceforge.net/p/monocruise/"><img alt="Download MonoCruise" src="https://sourceforge.net/sflogo.php?type=18&amp;group_id=3904914" width=150></a>
[![Download MonoCruise](https://img.shields.io/sourceforge/dw/monocruise.svg)](https://sourceforge.net/projects/monocruise/files/latest/download)
[![Download MonoCruise](https://img.shields.io/sourceforge/dt/monocruise.svg)](https://sourceforge.net/projects/monocruise/files/latest/download)

## Visit the official site for accurate and up-to-date showcases and information: https://ld-tech.org/projects/monocruise/

# MonoCruise
MonoCruise is a third-party software that sits in between ETS2/ATS and your pedals.
MonoCruise has a ton of quality of life features, like a better Adaptive Cruise Control, Automatic Emergency Braking, or a One-Pedal Driving system for heavy traffic.
Every feature (including ACC and AEB) works in TruckersMP and singleplayer ETS2/ATS, and every feature can be turned off.

![image_2025-07-02_202137925](https://github.com/user-attachments/assets/0b35aa19-340f-44a9-8e8b-0493c9cd30ca)

## Requirements
- Windows 10 or 11 (64-bit)
- Euro Truck Simulator 2 or American Truck Simulator
- Physical pedals (USB or DirectInput, for example the pedals of a wheel set). MonoCruise sits between your pedals and the game, so driving with only a keyboard or controller is not supported.

### features

**Cruise & speed control**
- Adaptive Cruise Control (ACC): holds a safe following distance from the vehicle ahead, slows down with traffic and pulls away with it again
- ACC gap level adjustment: 4 gap levels, assignable to buttons
- Traditional Cruise Control with speed limiter mode, plus an optional global speed limit
- Short and long speed increment/decrement buttons (configurable step sizes)
- km/h in ETS2, mph in ATS

**Safety**
- Automatic Emergency Braking (AEB): predicts where your truck is going and brakes for vehicles you would otherwise hit
- Emergency stop detection: full brake lock on sudden pedal slam or crash

**Pedal & driving feel**
- One-Pedal Driving system: press the gas to speed up, ease off to brake progressively
- Exponential braking and accelerating: configurable non-linear pedal curves
- Learns how hard your truck can accelerate and brake, so cruise and ACC stay accurate with any load
- Hill compensation, and follows the in-game Braking intensity slider

**Comfort & automation**
- Auto start and stop together with the game, for non-intrusive UX
- Automatically horn when braking hard
- Auto enable hazard lights when braking for traffic, auto disable on acceleration
- Live braking and accelerating bar on the bottom of the screen

**Input**
- Multi-device button support: joystick buttons, hat directions, keyboard keys and USB button devices
- Automatic pedal reconnect: recovers gracefully if your pedals disconnect mid-drive

## .exe install

1. Download "MonoCruise.installer.exe" from SourceForge, or from the [latest release on GitHub](https://github.com/luuukske/MonoCruise/releases/latest).

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<a href="https://sourceforge.net/projects/monocruise/files/latest/download"><img alt="Download MonoCruise" src="https://a.fsdn.com/con/app/sf-download-button" width=276 height=48 srcset="https://a.fsdn.com/con/app/sf-download-button?button_size=2x 2x"></a>

2. Run the installer.

   MonoCruise is not code-signed yet, so Windows may show "Windows protected your PC". Click **More info**, then **Run anyway**. Some antivirus programs also flag unsigned apps; that is a false positive, and the full source code is on this page.

   Leave "Start MonoCruise automatically when ETS2/ATS is running" ticked if you want MonoCruise to open together with the game.
3. Run MonoCruise. It installs the SDK files the game needs to communicate with MonoCruise.
4. Start the game, and press ok or enter when asked for SDK confirmation.

   <img src="https://github.com/user-attachments/assets/76c706de-60b6-457c-ae78-0dc6185810df" alt="In-game SDK confirmation dialog" width="400"/>

5. Wait for MonoCruise to connect.
6. Open the settings on the MonoCruise window.
7. Press "Connect to pedals".
8. Tap your brake pedal.
9. Tap your gas pedal.

### set up the cruise control (optional):
10. Scroll down to the Cruise Control settings.
11. Press the button next to the one you want to assign (Enable/Disable, Increase, Decrease, and optionally the ACC distance buttons).
12. Press the key or wheel button you want to use.

Now you're done and can use MonoCruise in ETS2 or ATS.

> [!IMPORTANT]
> MonoCruise has to keep running while you drive. When it was started together with the game, it stays minimised and closes again when you quit the game.

To stop MonoCruise starting with the game, untick "Autostart MonoCruise" in the settings. If that setting is greyed out, autostart was not chosen in the installer: reinstall with it ticked, or enable "MonoCruiseChecker" in Task Manager under Startup apps.

### Updating
MonoCruise tells you when an update is out. Install it with the "Update" button in the MonoCruise settings; your settings are kept.

### Coming from v1.0.x?
v1.1.0 is a full rewrite. Uninstall v1.0.x first (Windows Settings > Apps), then install v1.1.0 and connect your pedals and buttons again.

## Adaptive Cruise Control (v1.1.0 and above):
Turn on Adaptive Cruise Control in the settings. With cruise control on, it holds a safe following distance from the vehicle ahead in singleplayer or TruckersMP, slows down with traffic, and speeds back up to your set speed when the road clears. The following gap has 4 levels, set from "Following distance" under Adaptive Cruise Control in the settings, or changed while driving with the optional ACC distance buttons (bind just one and it cycles through all four).

Pressing the brake turns cruise control off.

> [!CAUTION]
> ACC is still experimental and can brake harder than needed in some situations. Stay ready to take over.

## Automatic Emergency Braking (v1.1.0 and above):
AEB is off by default. Turn it on with "Emergency Braking" in the settings. It watches the traffic around you along the path your truck is taking. When a collision is coming it beeps, flashes the cruise panel and brakes as hard as needed to stop behind the vehicle. It works with or without cruise control.

To override AEB, press the gas pedal fully. Pressing the brake yourself does not cancel it.

> [!CAUTION]
> AEB is experimental. It can brake when there is no real danger, for example at intersections or when traffic changes lanes next to you. Keep your eyes on the road: AEB is a backup, not a replacement for braking yourself.

## Privacy
MonoCruise runs entirely on your own PC: no account, no telemetry, no ads. It only goes online to check for updates and for a newer SDK plugin.

The one exception is opt-in: "Help improve AEB and ACC" in the settings (off by default) shares short, anonymous recordings of AEB events so AEB and ACC can be tested against real driving. The prompt shows exactly what is shared before you tick it, and unticking stops it.

## Support
- Questions and help: [Discord](https://discord.gg/MM9eV4CSxt)
- Bugs: [open an issue](https://github.com/luuukske/MonoCruise/issues). Attach `monocruise.log`, and `monocruise.prev.log` if you restarted MonoCruise since the problem. Both are in the MonoCruise install folder, by default `%LOCALAPPDATA%\Programs\MonoCruise`.

## .py install
Not supported yet, but you can try it.

## uses:
- [ETS2LA plugin](https://gitlab.com/ETS2LA/ets2la_plugin): used for getting AI/MP vehicle data for ACC and AEB.
- [Truck_Telemetry](https://github.com/dreagonmon/truck_telemetry): used to get telemetry data from the game.
- [scscontroller](https://github.com/ETS2LA/scs-sdk-controller/tree/main): used to send commands to the game like braking, gas, hazards, etc.
- [PySide6](https://doc.qt.io/qtforpython-6/): used as the UI framework.
- [pygame](https://github.com/pygame/pygame): used to get pedal values and to play sounds.
- [hidapi](https://github.com/trezor/cython-hidapi): used to read USB button devices.

This project is licensed under the MIT License.
It includes third-party code under the CC0-1.0, MIT, and BSD 3-Clause licenses.

See [THIRD_PARTY_LICENSES/](THIRD_PARTY_LICENSES/) for details.
