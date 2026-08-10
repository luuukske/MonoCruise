# button_device_thread

Reads buttons from HID devices that pygame/SDL does not expose as joysticks, so
they can be bound to cruise control. Binding format lives in
`core/INPUT_BINDINGS.md`; a `button_device` binding is a `vid_pid` plus a
`button_id`, where `button_id = byte_index * 8 + bit_index` into the raw report.

Publishes `data.button_states` as `{vid_pid: {button_id: bool}}`. `resolve_held()`
in `core/input_bindings.py` reads it, `main_pedal_thread` republishes the bound
buttons, and the press FSM in `core/cruise_control_thread` acts on the edges.

## Report reading

Every queued report is drained each tick, not one per tick. This matters more
than it looks. These devices report on change plus a slow keepalive, so a burst
only ever arrives during a state change, which is exactly the moment the reader
must not fall behind. Consuming one report per tick republished each report as a
full tick of state, so a burst of `set, clear, set` became three separate ticks
of button state regardless of how fast the real transitions were.

The drain is bounded by `_MAX_REPORTS_PER_TICK` so a runaway device cannot stall
the loop, and it checks `self.running` like any other inner loop.

A transient read error keeps the last settled state rather than publishing an
empty dict. Publishing empty would read as a release to the CC button FSM and
fire a spurious press. Only a real disconnect (`OSError`) clears the state, via
`_reset_button_state`, so a reconnect cannot inherit a button that was held when
the device went away.

`_settle_buttons` returns an entry for every bit the device has ever reported,
including bits that are false. `binding_state()` treats an empty dict as "device
has not reported yet" and returns `None`, which the capture guard in
`main_pedal_thread` distinguishes from a genuine `False`.

## Debounce

`_DEBOUNCE_S` (20 ms) is the window a bit must hold a new value before it is
published. Without it, switch contact bounce turns one physical press into two.

Measured on a MOZA Multi-function Stalk (346e:0024): the device sends 8-byte
reports, a keepalive every ~200 ms, and an immediate report on every change.
Two of six presses bounced, with glitches of 1.5 ms to 7.2 ms:

```
12.167 down -> 12.174 up (7.2 ms) -> 12.176 down (1.6 ms) -> 12.338 up
```

20 ms clears the worst observed bounce with margin while staying well under a
humanly producible press. The cost is 20 ms of press latency, which replaces the
coarser quantization the old 20 Hz tick already imposed.

Debounce covers the HID path only. Joystick and keyboard bindings resolve
through `main_pedal_thread._read_cc_button_states`, which is the choke point to
extend if those sources ever show the same symptom.

## Loop rate

100 Hz. Reads are non-blocking against devices that report at a few Hz, so the
cost is a wakeup. The rate is not free to change: it sets how promptly a settled
edge reaches the 100 Hz consumer chain, and a slower tick widens the window in
which a whole press can collapse into a single drain burst and be missed.

## Capture

`start_capture()` opens every non-tracked HID device (skipping mouse/keyboard
usages and anything pygame already owns by vid:pid), watches for a 0 to 1
transition, and hold-confirms it for `_CAPTURE_CONFIRM_S` before publishing
`capture_event`. The confirm is a duration, not a tick count, so changing
`loop_interval` cannot silently change how long a user must hold a button. Bits
that change during warm-up are marked noisy and ignored for the rest of the
capture session.

Capture must not open before `main_pedal_thread` publishes
`joystick_capture_ready`, or pygame-owned devices get raw-scanned and report
phantom buttons.
