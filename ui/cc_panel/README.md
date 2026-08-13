# CC Panel (`ui/cc_panel`)

## Purpose

`ui/cc_panel/main.py` implements the **Cruise Control Panel**: a small, always-on-top, translucent PySide6 window that renders (via `QPainter`) the most valuable driving-assistance state at a glance:

- **Set speed / display text** (e.g. `80 km/h`, `-- km/h`)
- **CC mode and enabled state**
  - `"Cruise control"` vs `"Speed limiter"` (affects icon + color)
  - enabled/disabled (affects color)
- **ACC status visualization** (when enabled)
  - whether ACC is enabled
  - whether ACC is *locked* onto a lead vehicle
  - current **distance setting** (1–4) shown as distance “lines”
  - whether the lead vehicle is considered a **truck** (affects icon/spacing)
- **AEB warning**
  - when `AEB_warn=True`, the panel starts a **blink animation** and uses the AEB color
  - after `AEB_warn` turns off, the blink can continue briefly for visibility (cooldown)
  - the main window samples `AEB_warn` every 100 ms. AEB sound waits a second
    warn tick and hard-stops on user-brake suppression, so a one-tick pulse
    cannot beep without a panel flash.

This panel is intended for **user-facing safety/status feedback**: it should remain responsive even when worker threads are busy and should not require those threads to touch Qt directly.

## Architecture (how it works)

- **Rendering**
  - A single internal widget (`_PanelWidget`) paints everything in `paintEvent()` using `QPainter`.
  - Icons are loaded from `ui/cc_panel/assets/` and tinted at paint time. Some pixmaps are cached to reduce redraw cost.

- **Threading model**
  - The public class `cc_panel` is a **thread-safe facade**.
  - Worker threads call `cc_panel.update(...)` freely. Each call merges fields into a pending dict under a lock and schedules **at most one** queued flush (`_flush_coalesced_sig` → apply + repaint). Other actions still use dedicated signals (`_show_sig`, `_move_sig`, …).
  - **Important**: the `QWidget` must still be **created and owned by the Qt main thread** (the thread running the `QApplication` event loop). Do not instantiate `cc_panel` inside a worker thread.

- **High-frequency `update()`**
  - Calling `update()` continuously (e.g. every telemetry tick) is supported: callers only pay a short critical section and dict merge; the GUI thread applies **one combined delta** per burst instead of one queued payload per call.
  - **Semantics**: last write wins per field among calls that are still pending before the next flush. If any call in a batch used `complete_update=True`, the merged batch carries `_complete_update` until applied.

- **State persistence**
  - The panel is draggable; on mouse release it persists its position via `Settings.save(values={"panel_x": ..., "panel_y": ...})`.
  - Use `cc_panel.ensure_on_screen()` to recover from saved coordinates that are off-screen (e.g. monitor layout changed).

## Public API (`cc_panel`)

All methods below are safe to call from any thread **unless noted otherwise**.

- **Construction (Qt main thread only)**
  - `cc_panel(text_content, cc_mode="Cruise control", cc_enabled=True, x_co=100, y_co=100, acc_enabled=False, scale_mult=1)`

- **Update state**
  - `update(new_text=None, cc_mode=None, cc_enabled=None, acc_locked=None, distance_to_lead=None, AEB_warn=None, complete_update=False, acc_enabled=None, acc_truck=None)`
  - Only provided fields are changed; omitted fields keep their previous values.
  - `complete_update=True` forces a full recalculation (useful if you changed multiple coupled fields or want to be extra safe after a state resync).
  - Coalesced: safe to invoke on every loop iteration; pending keys merge until the next GUI flush.

- **Visibility / lifecycle**
  - `show()` / `hide()`
  - `stop()` closes the window and stops the panel (intended for shutdown).

- **Position / appearance**
  - `move(x, y)`
  - `update_scaling(scale_mult)` rescales layout, font, icons, and caches.
  - `set_background_opacity(opacity)` sets only the background opacity in \([0.0, 1.0]\). Text and icons remain fully opaque.
  - `ensure_on_screen()` (call from the **Qt main thread**) clamps the window back onto a visible screen and persists the corrected position.

## Typical usage

### Create in `main.py` (Qt main thread)

Create one instance after `QApplication` exists and keep it for the whole app lifetime.

```python
from ui.cc_panel.main import cc_panel
from core.settings import Settings

panel = cc_panel(
    text_content="-- km/h",
    cc_mode="Cruise control",
    cc_enabled=False,
    x_co=Settings.panel_x,
    y_co=Settings.panel_y,
    acc_enabled=False,
    scale_mult=getattr(Settings, "panel_scale", 1.0),
)
panel.show()
panel.ensure_on_screen()  # Qt thread only
```

### Update from worker threads (thread-safe)

Any thread can push partial updates without coordinating with the GUI thread.

```python
# e.g. inside a BaseThread.loop() or other worker logic
panel.update(
    new_text="80 km/h",
    cc_enabled=True,
    cc_mode="Cruise control",
)

# ACC enabled with distance visualization:
panel.update(
    acc_enabled=True,
    distance_to_lead=3,  # 1..4
)

# ACC lock acquired:
panel.update(
    acc_locked=True,
    acc_truck=False,      # True if lead vehicle should be treated as a truck
)

# AEB warning (triggers blinking):
panel.update(AEB_warn=True)
panel.update(AEB_warn=False)  # blink may continue briefly (cooldown)
```

## Implementation notes / conventions

- **Valid `cc_mode` strings**
  - The implementation currently special-cases `"Speed limiter"` and `"Cruise control"` for icon + color selection. Other values fall back to a placeholder icon and default cruise-control color when enabled.

- **ACC “distance lines”**
  - The panel clamps `distance_to_lead` to 1–4 for rendering.
  - When ACC is enabled, the distance lines are shown even before lock, to give immediate feedback when the user changes the distance setting.
  - When locked, the vehicle icon is shown alongside the lines; before lock, only the lines may be shown (depending on current logic).

- **AEB blinking**
  - Blinking is driven by a `QTimer` on the GUI thread.
  - Timing is expressed in frames at 60 Hz to keep a stable cadence.

- **Don’t block the GUI thread**
  - Keep updates small; do heavy work in worker threads and only publish the final UI state through `update()`.

## ACC distance lines and vehicle cutout

Lines are cached per distance setting. Vehicle cutout uses DestinationOut on an offscreen buffer each frame; pixmaps at max scale, drawn at animated scale. Bbox from settled targets keeps cache keys stable.

## Lead speed indicator

Text rasterized for fractional blit; Gaussian eraser matches HTML prototype. Visibility follows lead_vehicle_speed is not None, not ACC lock.
