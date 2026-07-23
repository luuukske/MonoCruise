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

Public helpers: `binding_display_name`, `keyboard_is_pressed`.
