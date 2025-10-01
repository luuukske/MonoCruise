# MonoCruise
[![Download MonoCruise](https://img.shields.io/sourceforge/dw/monocruise.svg)](https://sourceforge.net/projects/monocruise/files/latest/download)
[![Download MonoCruise](https://img.shields.io/sourceforge/dt/monocruise.svg)](https://sourceforge.net/projects/monocruise/files/latest/download)

MonoCruise is a third-party software that sits in between ETS2/ATS and your pedals. 
MonoCruise has a ton of quality of life features, like a better Adaptive Cruise Controll or a One-Pedal Driving system for heavy traffic.
every feature (including the ACC) works in TruckersMP and singleplayer ETS2/ATS.

![image_2025-07-02_202137925](https://github.com/user-attachments/assets/0b35aa19-340f-44a9-8e8b-0493c9cd30ca)

### features
This also includes quality of life features like:
- smoother Adaptive Cruise Control (releases with v1.0.1)
- better Cruise Control
- One-Pedal driving system
- live braking and accelerating bar on the bottom of the screen
- automatically horn when braking hard
- auto enable hazard lights when braking for traffic
- auto disable hazards
- exponential braking and accelerating
- auto start and stop for non-intrucive UX
## .exe install

1. Download and run "MonoCruise installer.exe".
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

## Adaptive Cruise Control (v1.0.1 only):
you can enable the experemental ACC and it will hold a safe distance from the lead vehicle (singleplayer or TruckersMP). you currently cannot change the following distance, but that's comming.
   
> [!CAUTION]
> The ACC is EXTREMELY experemental. 
> The ACC has a tendency to brakecheck, BE MINDFULL WHEN TURNING ON!

## .py install
Not supported yet, but you can try it.

## uses:
- [ETS2LA plugin](https://gitlab.com/ETS2LA/ets2la_plugin): used for getting ai/MP vehicles data for Adaptive Cruise Control.
- [Truck_Telemetry](https://github.com/dreagonmon/truck_telemetry): used to get data from the game. 
- [scscontroller](https://github.com/ETS2LA/scs-sdk-controller/tree/main): used to send commands to the game like braking, gas, hazards, etc..
- [CustomTkinter](https://github.com/TomSchimansky/CustomTkinter): used as a modern UI for Python.
- [pygame](https://github.com/pygame/pygame): used to get pedal values and to play sounds.

This project is licensed under the MIT License.
It includes third-party code under the CC0-1.0, MIT, and BSD 3-Clause licenses.

See [MonoCruise/THIRD_PARTY_LICENSES/](https://github.com/luuukske/MonoCruise/tree/main/THIRD_PARTY_LICENSES) for details.

