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

## Hat virtual buttons

`virtual_code = button_count + hat_index * 4 + direction_index`

Direction index: 0=up, 1=right, 2=down, 3=left (pygame hat xy: (0,1), (1,0), (0,-1), (-1,0)).
