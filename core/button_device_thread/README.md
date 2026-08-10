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

## Press counts

`data.button_press_counts` is a monotonic count of published press edges per bit,
alongside the level in `data.button_states`. Consumers poll far slower than a tap
lasts (`main_pedal_thread` runs at `polling_rate`, as low as 10 Hz), so watching
the level for an edge loses presses; they read the count instead. This thread can
count exactly because it runs at 100 Hz and drains every report. Counts reset with
the device on disconnect, and consumers treat a count going backwards as a resync
rather than replaying presses. See `core/INPUT_BINDINGS.md`.

`_settle_buttons` returns an entry for every bit the device has ever reported,
including bits that are false. `binding_state()` treats an empty dict as "device
has not reported yet" and returns `None`, which the capture guard in
`main_pedal_thread` distinguishes from a genuine `False`.

## Debounce

Switch contact bounce turns one physical press into two if it is not filtered.
Measured on a MOZA Multi-function Stalk (346e:0024): the device sends 8-byte
reports, a keepalive every ~200 ms, and an immediate report on every change.
Two of six presses bounced, with glitches of 1.5 ms to 7.2 ms:

```
12.167 down -> 12.174 up (7.2 ms) -> 12.176 down (1.6 ms) -> 12.338 up
```

Filtering this by requiring every level to hold for a fixed window does work,
but it also throws away genuine fast taps, because a short tap and a bounce
glitch are both just short. The two are told apart by structure instead:
**bounce is a dip inside a press, never a standalone pulse from idle.** So the
filter runs in two stages, each with its own job.

**Stage 1, bounce to logical** (`_apply_release_hold`, and the edge handling in
`_drain_reports`). A press promotes immediately, so no tap is ever too short to
count and press latency stays at zero. A release only promotes once the bit has
stayed low for `_RELEASE_HOLD_S` (15 ms); a re-press inside that window cancels
it, because that dip was bounce. This absorbs the glitch above without touching
the surrounding press.

**Stage 2, logical to published** (`_settle_buttons`). Stage 1 can move faster
than the consumer samples, so a release that matures and a press that lands in
the same tick would otherwise merge two taps into one hold. Logical edges are
queued and published at most one per tick, each level held for `_MIN_DWELL_S`
(20 ms), which is above Windows timer granularity so a consumer cannot step over
an edge. No edge is dropped, only delayed. Beyond `_MAX_PENDING_EDGES` the queue
resyncs to the current level rather than lagging further behind.

Measured behaviour: every tap down to 2 ms registers, sustained tapping is exact
through 25 taps/s and degrades gracefully above it (7/10 at 30 Hz, 2/10 at
40 Hz), and bounce glitches up to 14 ms still read as a single press. Human
tapping tops out near 8 to 14 taps/s.

Both stages cover the HID path only. Joystick and keyboard bindings resolve
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
