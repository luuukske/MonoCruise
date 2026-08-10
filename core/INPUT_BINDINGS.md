# Input binding format

Bindings live in `config.json` / `Settings`. `migrate_binding()` upgrades legacy
values; `resolve_held()` / `binding_state()` read live state.

| Form | Meaning |
| --- | --- |
| `null` | Unassigned |
| int (legacy) | Joystick button on configured pedal device |
| str (legacy) | Keyboard key name (e.g. `"A"`) |
| `{"source":"joystick","device_guid", "code", ...}` | Button or hat: hat virtual index = `button_count + hat_idx*4 + dir` (dir 0..3 = up/right/down/left) |
| `{"source":"keyboard","code"}` | Key name |
| `{"source":"button_device","vid_pid","button_id"}` | HID report bit: `button_id = byte_index*8 + bit_index` |

Public helpers: `binding_display_name`, `keyboard_is_pressed`, `resolve_press_count`.

## Press counts, not levels

`resolve_held()` returns a level. Every consumer of it polls on its own clock,
and at `polling_rate` 10 that clock is 100 ms, which is longer than a tap. The
level alone therefore loses fast presses.

`resolve_press_count(binding)` returns a monotonic count of presses from the
source that actually sees the edges, or `None` for sources that cannot count:

| Source | Counter | Why it is exact |
| --- | --- | --- |
| `button_device` | `ButtonDeviceThread.data.button_press_counts` | Ticks at 100 Hz and drains every HID report; counts debounced press edges |
| `keyboard` | `KeyboardThread.data.key_press_counts` | Counts from the OS hook, so no poll rate is involved. Auto-repeat is ignored, and only keys currently bound to a CC button are watched |
| `joystick` | none (`None`) | pygame exposes a level; edges would need SDL button events, and hats have none |

`main_pedal_thread` consumes these deltas into `cc_button_press_counts`, keyed by
binding name, and falls back to its own edge detection when the source returns
`None`. `cruise_control_thread` fires one short press per counted press. A count
that goes backwards means the source restarted and consumers resync rather than
replay.
