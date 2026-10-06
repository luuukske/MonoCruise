# Main pedal thread

> Reads physical pedals and buttons; `sending_thread` writes game outputs.
> Cruise buttons are published here for `cruise_control_thread`.

## Scope

- Pygame pedal device (critical) plus extra joysticks for bindings.
- Raw axes each tick; One-Pedal Drive when cruise is not commanding.
- Weight-based brake adjustment; legacy hold-brake / park detection.
- Emergency stop on sudden brake / crash until user releases (`em_stop`).
- Button bindings → `cc_*_held` for cruise; `joystick_button_states` for `input_bindings`.
- Settings capture APIs for button assign and pedal connect flows.

Does not send to the game, hazard/horn, cruise logic, AEB, or keyboard hook lifecycle.

## Pedal axes come from their own events

`gasval` and `brakeval` move only on a `JOYAXISMOTION` event of that exact axis
on the pedal device (`pedal_axes.PedalAxes`). Never re-read the other pedal with
`get_axis()` when one axis reports.

SDL 2.28 opens DirectInput joysticks in buffered mode and only learns an axis's
value when a report changes it. Pedals that report only on change (no idle
stream) leave every untouched axis at SDL's initial 0.0 after open, which
normalises to half travel. The old handler read both axes on any motion event,
so the first brake press after launch published 50% gas until the gas pedal
was touched once (issue #12, Moza SRP). A MOZA R3 streams at 1 kHz and never
showed it, which is why it looked user-specific.

So an axis that has not reported reads as released, after setup, a reconnect,
or a finished pedal configuration. The same rule covers the reconnect FSM,
which used to seed both values with `get_axis()` straight after opening.

Once an axis has reported it is live, and from then on SDL's value is real, so
live axes are re-read with `get_axis()` every tick (`PedalAxes.refresh`). A
missed event cannot leave a pedal stuck, which the old re-read-on-any-event
handler also guaranteed. Pedal-device events keep feeding `PedalAxes` during
pedal configuration, or a cancel would leave the pre-config values.

The brake's first report after a reset jumps from 0 to wherever the foot is.
`take_brake_went_live()` makes `loop()` skip the stomp test for that one tick,
or a brake held through a reconnect at speed would trip `em_stop`. A real
stomp is caught a tick later, and `brakeval >= 0.8` is never skipped.

## Pedal configuration

`start_pedal_config()` opens every joystick (vJoy excluded) and logs what it
found: name, vid:pid from the GUID, axis count, or the open error. A device
another program holds exclusively fails to open there; SDL 2.28 opens every
DirectInput joystick `DISCL_EXCLUSIVE | DISCL_BACKGROUND`.

Taps are measured by `pedal_axes.TapDetector` from motion events, never from
`get_axis()` at open. A baseline read at open is SDL's 0.0 for a change-only
device: a pedal resting at -1.0 then looked pressed and inverted the moment it
reported.

- An axis that has sat within 0.2 of -1 or +1 rests at that end (sticky: a full
  stomp reaching the other end does not move it). Its tap is 0.15 from that
  end, and the end decides inversion (resting at +1 means inverted). The low
  threshold is for load-cell brakes, where a tap moves the output little.
- Any other axis (steering, odd calibration) keeps the old rule: 0.3 from its
  first event, inverted when it moved down.
- Within a tick a pedal-like axis beats any other, then the larger move wins.
  The gas stage only accepts the brake's device and never the brake axis.

Known limit: a pedal already pressed when the flow starts has its first event
mid-travel, so its release can still read as an inverted tap.

A cancel logs, per device, how many axis events arrived and the largest move,
so a "pedals not detected" report says whether pygame saw the device at all,
whether it reported, or whether the tap was too small.

## Opening joysticks

pygame's `Joystick(i)` blocks every thread for the whole first open of a
device: 75 to 150 ms for DirectInput, measured on a MOZA R3, since SDL sleeps
50 ms per open. Reopening an open device is free, and pygame never closes one
when the object is dropped, so a device stays open for the session once opened.
Button capture and pedal configuration used to open every joystick in one
tick, stalling the sending thread for the sum.

Both now share `joystick_pool.JoystickPool`: it walks the device indices and
starts no further open once 20 ms of the tick are spent, so already-open
devices finish at once and each slow open gets a tick of its own. A device
count change restarts the walk, and starting either flow requests a restart
(`_joy_pool_restart`, applied by the loop thread) so a failed open is retried.
`joystick_capture_ready` is only published once the walk is complete, so the
HID scan never raw-scans a joystick pygame is still opening.

Opening a force-feedback wheelbase through SDL resets its effects and takes
exclusive access (Trello card 141), so every open avoided matters.

## Finding a device whose GUID changed

An SDL GUID is the bus, a CRC16 of the device name, vid, pid and version. A
firmware update or a rename changes it, and an exact match then reported
"pedals not found" and dropped joystick button bindings until the driver
reconfigured. `_find_joystick` tries the exact GUID first and stops opening at
the first match. Failing that it accepts the only connected joystick with the
same vid:pid, logging that the GUID changed. Two matches is ambiguous and finds
nothing. For the pedals the match must also have enough axes for the
configured gas and brake axes, so a changed device is never driven blind.
Settings keep the old GUID; every lookup goes through this fallback.

## Pre-override intent fields

`opdgasval` and `opdbrakeval` publish the OPD-mapped user demand as computed at
`gas_output` / `brake_output` assignment time, **before** the AEB override,
sudden-slam, crash, and `em_stop` branches rewrite the outputs. `gas_output` /
`brake_output` are the post-override values actually sent onward.

Consumers wanting the driver's intent must read the `opd*` pair; reading
`brake_output` instead feeds AEB's own slam back in. Keep any new override
branch below the `opd*` snapshot.

`opdbrakeval` is recorded in AEB clips. Any value above zero silences AEB warn
(along with `mapper_command_brake`). See `core/aeb/README.md` section 3 item 6.
`brake_output` is still unusable for this: AEB writes it.

## Binding capture runs with the game closed

`loop()` returns early when telemetry is not connected, but button assignment
must still work, so that branch keeps `_ensure_button_devices()` and
`_update_joystick_states()`. Those publish `joystick_button_states` (the pressed
highlight in settings), detect the joystick capture press, and publish
`joystick_capture_ready`, which gates the HID capture scan in
`button_device_thread`. With the calls behind the telemetry gate, both joystick
and HID assignment were dead whenever the game was not running, while keyboard
assignment kept working because it captures from an OS hook.

Nothing can act on a press there: `_read_cc_button_states` stays below the gate,
so `cc_*_held` remains False while the game is closed.

## Hat virtual buttons

`virtual_code = button_count + hat_index * 4 + direction_index`

Direction index: 0=up, 1=right, 2=down, 3=left (pygame hat xy: (0,1), (1,0), (0,-1), (-1,0)).
