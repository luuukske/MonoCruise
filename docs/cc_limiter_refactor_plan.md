# CC / Speed Limiter Refactor

## Context

Today the Speed Limiter is implemented as a side-object owned by `SendingThread` (`core/sending_thread/thread.py`). Each sending tick calls `self._limiter.step_wanted(spd_ms)` and `min()`-merges the result with CC's published bid before the mapper. The limiter shares all PID gains with CC (`cc_kp`, `cc_ki`, `cc_kd`, `cc_integral_clamp`, `cc_accel_min_ms2`), so they cannot be tuned independently. `Settings.cc_mode` (`"Cruise control"` vs `"Speed limiter"`) only flips the post-mapper user-pedal merge (`max` vs `min`): both PIDs bid simultaneously.

The user wants the limiter to be able to react more aggressively than CC. The blocker is **shared tuning** and the **entanglement between two PIDs and the cc_mode flag** at the SendingThread layer. The fix is to make CC and Limiter **mutually exclusive sibling controllers**, dispatched by the existing orchestrator (`CruiseControlThread`) based on `Settings.cc_mode`, with the limiter getting its own `limiter_*` settings.

User decisions (final):
1. CC and Limiter are **mutually exclusive**: `cc_mode` selects which steps each tick.
2. The CC class keeps the button FSM and target ownership in **both** modes. In limiter mode, CC's set-speed value is routed into the limiter as its cap. Limiter gets **no buttons of its own**.
3. Both run at the same tick rate (`Settings.polling_rate`).
4. Independent tuning via new `limiter_*` settings.
5. **Disengage conditions (brake/park/gear/stop) apply to CC only**: the limiter does not disable on these events, matching current behaviour.
6. **Global limiter is always active when `global_speed_limit_kmh` is not null**: regardless of CC engagement state.

Outcome: limiter becomes a first-class peer of CC under a single orchestrator. `SendingThread` returns to being a pure mapper consumer of one upstream longitudinal bid.

---

## Architecture After

```
CruiseControlThread.loop()
├─ button FSM → updates self._cc_ctrl.enable/disable/target (both modes)
├─ CC-only disengage: user-brake, park/neutral/reverse, disarm-on-stop
│    (limiter is unaffected by any of these: matches current behaviour)
├─ mode-flip handover → reset()s the inactive controller's PID state
├─ dispatch by cc_mode:
│   ├─ "Cruise control" → cc_ctrl.step(ctx); if active, acc_ctrl.step(ctx); min-merge
│   └─ "Speed limiter"  → resolve target; limiter always on when global limit set;
│                         limiter_ctrl.step(ctx); acc_ctrl.reset()
├─ publish wanted_accel → telemetry.commanded_accel_ms2 (single bid)
└─ publish self.data    → active, cc_enabled, target_speed_kmh, wanted_accel_ms2,
                          active_controller

SendingThread.loop()
├─ read tel.commanded_accel_ms2 (single bid; no limiter side-call)
├─ accel_mapper.step(...)
└─ post-mapper user-pedal merge by cc_mode (unchanged behaviour)
```

---

## File-by-file changes

### `core/longitudinal/limiter.py`: rewrite in place

Replace the `step_wanted(speed_ms)` + self-clock design with a standard `LongitudinalController` subclass:

- Drop `step_wanted`, drop `_prev_mono` self-clock, drop `is_constraining`.
- Use `ctx.dt` from `LongCtx`.
- Add `enable()`, `disable()`, `set_target_kmh(v)`, `enabled` and `target_speed_kmh` properties, `active` (= `enabled and target is not None`), `reset()`.
- PID uses new gains: `Settings.limiter_kp`, `limiter_ki`, `limiter_kd`, `limiter_integral_clamp`, `limiter_accel_min_ms2`.
- Asymmetric clamp: lower side only (continuous-tracker invariant: must allow positive bids below the cap so the mapper engages and user gas still works).
- `step(ctx)` returns `LongOutput(wanted, True)` when active (every tick, regardless of whether currently constraining). See AGENTS.md continuous-tracker rule.
- **No disengage logic of any kind inside this class**: the orchestrator owns CC disengage; the limiter has none.

### `core/longitudinal/cc.py`: surgical trims

- Remove the user-brake disengage block (~lines 148–156). Lifted to orchestrator (CC-only path).
- Remove the park/neutral/reverse disengage block (~lines 159–169). Lifted to orchestrator (CC-only path).
- Remove the disarm-on-stop block (~lines 174–194). Lifted to orchestrator (CC-only path).
- `step()` becomes pure PID + smoothing + the target/active gates that depend on its own enable/target state.
- Keep `_clamp_target_kmh` against `global_speed_limit_kmh`: CC target is still capped by the global limit in both modes (CC owns the target value in limiter mode too).

### `core/longitudinal/base.py`: no change

Interface (`LongCtx`, `LongOutput`, `LongitudinalController`) is already correct.

### `core/longitudinal/acc.py`: no change

`active` already requires `Settings.cc_mode == "Cruise control"`. Orchestrator additionally calls `acc_ctrl.reset()` in limiter mode for cleanliness.

### `core/cruise_control_thread/thread.py`: main rewrite

Constructor:
```python
self._cc_ctrl = CruiseController()
self._limiter_ctrl = SpeedLimiter()      # NEW
self._acc_ctrl = AdaptiveCruiseController()
self._prev_cc_mode: str | None = None    # NEW: track mode for handover
```

`loop()` body: replace the current `cc_out → acc_out → arbitrate` block (~lines 194–210) with:

```python
mode = Settings.cc_mode

# Mode-flip handover: reset the now-inactive controller's PID state.
if mode != self._prev_cc_mode:
    if mode == "Speed limiter":
        self._cc_ctrl.reset()
        self._acc_ctrl.reset()
    else:
        self._limiter_ctrl.reset()
    self._prev_cc_mode = mode

# CC-only disengage: user brake, park/gear, disarm-on-stop.
# Limiter is intentionally excluded: it persists through brake presses and gear changes.
if mode == "Cruise control":
    self._handle_cc_disengage_conditions(ctx)

# Dispatch.
if mode == "Speed limiter":
    # Resolve target for limiter.
    # Priority: global speed limit (always-on) > CC's engaged target.
    if Settings.global_speed_limit_kmh is not None:
        # Global limit always active regardless of CC state.
        target = float(Settings.global_speed_limit_kmh)
        self._limiter_ctrl.set_target_kmh(target)
        self._limiter_ctrl.enable()
    elif self._cc_ctrl.enabled and self._cc_ctrl.target_speed_kmh is not None:
        # No global limit; use CC's manually-set target.
        self._limiter_ctrl.set_target_kmh(self._cc_ctrl.target_speed_kmh)
        self._limiter_ctrl.enable()
    else:
        self._limiter_ctrl.disable()

    long_out = self._limiter_ctrl.step(ctx)
    self._acc_ctrl.reset()
    acc_out = LongOutput(None, False)
else:
    long_out = self._cc_ctrl.step(ctx)
    if long_out.active:
        acc_out = self._acc_ctrl.step(ctx)
    else:
        self._acc_ctrl.reset()
        acc_out = LongOutput(None, False)

wanted_accel, commanding = self._arbitrate(long_out, acc_out)
self._publish_telemetry_command(wanted_accel if commanding else 0.0)
self._publish_data(commanding, wanted_accel if commanding else 0.0, mode)
self._maybe_reset_mapper_on_commanding_end(commanding)
```

`_handle_cc_disengage_conditions(ctx)` is the extracted method containing the three blocks stripped from `cc.py`. It calls `self._cc_ctrl.disable()` only. It does not touch `self._limiter_ctrl`.

Button FSM (`_tick_button_fsm` ~lines 318–414): no logic changes. The park-brake and non-drive-gear **engagement** guards remain (user cannot start a new engagement with park on or in neutral/reverse), but they gate **new engagements only**: they do not continuously disable the limiter mid-drive.

### `core/sending_thread/thread.py`: delete limiter integration

Delete:
- Line 27: `from core.longitudinal.limiter import SpeedLimiter`
- Line ~208: `self._limiter = SpeedLimiter()`
- Lines ~575–581 (`limiter_wanted = ...` and pre-mapper `min()`-merge)
- Lines ~843–844 (post-mapper `if limiter_active and ...` cap)

Update:
- Line ~587: `mapper_engaged = cruise_active or _aeb_active` (drop `or limiter_active`)
- Refresh the comment block at lines ~202–207 to describe the new single-bid input.

Keep untouched:
- `cruise_active` read (~lines 484–491): its semantics now are "the orchestrator is bidding (CC or limiter)", which is what we want.
- The cc_mode-driven post-mapper user-pedal merge (~lines 824–829): unchanged. In CC mode `max(a, mapper_gas)`; in limiter mode `min(a, mapper_gas)`.
- AEB controller, brake hysteresis, coast-down logger, capacity tracker.

### `core/settings.py`: add `limiter_*` keys

Insert near the existing `cc_kp` block (~line 108):

```python
limiter_kp: float = 0.5
limiter_ki: float = 0.0
limiter_kd: float = 0.0
limiter_integral_clamp: float = 3.0
limiter_accel_min_ms2: float = -1.0
```

Defaults mirror current `cc_*` values so behaviour is identical until the user tunes them. `Settings.load()` auto-merges missing keys (~line 256–260), so existing `config.json` files gain the new keys on first run: no migration code.

Keep `cc_mode` and `global_speed_limit_kmh` as-is. Document `global_speed_limit_kmh`'s dual role (CC target clamp + limiter always-on target when set) in a docstring.

### `monocruise.py`: no change

Thread registration unchanged; `CruiseControlThread()` handles the new limiter internally.

### `CruiseControlThreadData`: add one field

```python
@dataclass
class CruiseControlThreadData(ThreadData):
    active: bool = False                # any controller is bidding
    cc_enabled: bool = False            # CC FSM enable (mirrored into limiter in limiter mode)
    target_speed_kmh: float | None = None
    wanted_accel_ms2: float = 0.0
    active_controller: str = "none"     # NEW: "cc" | "limiter" | "none": debugging/telemetry
    _lock: threading.Lock = field(...)
```

UI (`cc_panel`) reads `target_speed_kmh` and `cc_enabled`: both keep their current meaning, so no UI changes needed beyond message text.

### Popups: mode-aware text

Mode-aware popup labels where the controller name is relevant:
- "CC disabled: brake pressed" (CC mode; limiter-mode never fires this)
- "Cannot engage with parking brake on" (mode-agnostic)
- "Can only engage in drive" (mode-agnostic)
- "Cruise control enabled" / "Speed limiter enabled": pick by `Settings.cc_mode`
- "Cruise control disabled" / "Speed limiter disabled"
- "Cruise target reset to current speed" / "Speed limit reset to current speed"

No new popup categories; existing `logger.info(..., extra={"popup": True})` pattern.

### `AGENTS.md`: update longitudinal section

- Document CC + Limiter as mutually exclusive sibling controllers selected by `cc_mode`.
- Document that disengage conditions (brake, park, neutral/reverse, disarm-on-stop) are CC-only: the limiter remains active through these events.
- Document that `global_speed_limit_kmh` keeps the limiter always-on regardless of CC FSM state.
- Restate the continuous-tracker invariant in the new context (limiter PID runs every tick when active, regardless of whether ego is over the cap).
- Restate the single-mapper invariant (one `AccelToPedals` instance in SendingThread, consuming one published bid).
- Document new `limiter_*` settings and `global_speed_limit_kmh`'s dual role.

---

## Order of implementation

Each step independently testable and revertible.

1. Add `limiter_*` settings keys. No behaviour change. Restart, confirm `config.json` gains them.
2. Reshape `SpeedLimiter` in place (still owned by SendingThread). Replace `step_wanted` with `step(ctx)`, switch to `limiter_*` settings, add lifecycle methods. SendingThread builds a local `LongCtx` and calls `step()`. Pre-mapper merge stays. Verify: CC identical; limiter now tunes independently.
3. Move ownership to `CruiseControlThread`. Add the dispatch block, mode-handover, CC-only disengage. Delete the limiter from SendingThread. Verify: full matrix below.
4. Add `active_controller` field to `CruiseControlThreadData`. Pure additive.
5. Mode-aware popup polish.
6. `AGENTS.md` update.

---

## Critical files

- `core/cruise_control_thread/thread.py`: orchestrator rewrite, CC-only disengage
- `core/longitudinal/limiter.py`: rewrite as proper `LongitudinalController` subclass
- `core/longitudinal/cc.py`: strip disengage logic (now owned by orchestrator, CC path)
- `core/sending_thread/thread.py`: delete limiter integration
- `core/settings.py`: add `limiter_*` keys
- `AGENTS.md`: document new architecture and invariants

---

## Risks / invariants to preserve

- **Continuous-tracker** (AGENTS.md): limiter PID runs every tick while active, regardless of whether currently constraining. Preserved by `SpeedLimiter.step()` always returning `LongOutput(wanted, True)` when active. Add a code comment citing AGENTS.md so future maintainers don't add a `if wanted_ms2 < 0` gate.
- **Single-mapper** (AGENTS.md): only one `AccelToPedals` instance stays in SendingThread, consuming one published bid. Refactor removes a path, doesn't add one.
- **Button FSM**: unchanged; FSM still drives `self._cc_ctrl` in both modes. Only the downstream consumer of target/enable changes.
- **Mode-flip handover**: brief discontinuity in commanded accel is acceptable for a manual toggle. Reset-on-flip ensures no stale integrator state from the inactive controller.
- **Limiter disengage immunity**: brake-tap, park brake, neutral/reverse, disarm-on-stop do NOT disable the limiter. `_handle_cc_disengage_conditions` is called only inside `if mode == "Cruise control"`.
- **Global limit always-on**: when `global_speed_limit_kmh is not None`, the limiter enable/target block runs unconditionally: it does not check `cc_ctrl.enabled`.
- **Watchdog timing**: one extra `step()` per tick is negligible at 60–100 Hz; well within the ≤0.5 s loop budget.

---

## Verification

**A. CC mode regression (no change expected)**
- `cc_mode = "Cruise control"`. Engage CC at 80 km/h, dec/inc buttons short + long, brake-tap disengage. Compare overshoot/settling against current build via `accel_to_pedals_debug.csv`.
- Enable ACC, follow a lead: confirm `min()` merge wins when ACC bids lower than CC.

**B. Limiter independent tuning**
- `cc_mode = "Speed limiter"`, `global_speed_limit_kmh = 100`, no CC engaged: drive past 90 → cap engages continuously (smooth tightening, AGENTS.md continuous-tracker rule).
- Set `limiter_kp = 1.0` (config.json hot-reloads in debug), `cc_kp` untouched. Same scenario shows tighter response: confirms independence.
- Engage CC via + button at 70: limiter target = 70 (mirrored from CC). Truck stabilises at 70 using limiter PID. Vary `limiter_kp` vs `cc_kp` to observe.

**C. No double-PID engagement**
- Add one debug log in the dispatch: `logger.debug("dispatch mode=%s active=%s", mode, long_out.active)`.
- Tail `monocruise.log`: exactly one bid per tick.

**D. Mode flip mid-engagement**
- Engage CC at 80, stabilise. Flip `cc_mode` to "Speed limiter" via UI. CC PID resets; limiter takes over with target=80. Truck switches from "drive to 80" to "cap at 80". Flip back: limiter resets, CC resumes.

**E. ACC gating**
- In limiter mode with a lead vehicle: ACC must NOT engage. Flip to CC: ACC engages.

**F. Continuous-tracker invariant (must not regress)**
- Limiter mode, `global_speed_limit_kmh = 80`, CC disengaged (fallback target). Coast up to 75 with gas held: `wanted_accel_ms2` should be positive but tightening, not zero. Pedal cap softens as ego nears 80: no overshoot at the boundary.

**G. Limiter disengage immunity (critical: current behaviour preserved)**
- Limiter mode active (either via global limit or CC target). Press brake to full stop, release: limiter resumes immediately without re-engagement.
- Shift to neutral mid-drive: limiter remains active (no popup, no disable).
- Apply park brake while moving: limiter remains active.
- Compare to CC mode: same brake press disables CC and shows popup. Confirms the disengage paths are independent.

**H. Global limit always-on**
- `global_speed_limit_kmh = 90`, `cc_mode = "Speed limiter"`, CC FSM not engaged. Boot truck: limiter active immediately without pressing any button. Drive past 85: cap engages.
- Clear `global_speed_limit_kmh` (set null): limiter deactivates unless CC FSM is engaged.

**I. Restart safety**
- Trigger a watchdog restart of `CruiseControlThread`. After restart: `cc_enabled=False`, `target=None`, both PIDs zeroed. Global limit re-enables limiter on first tick if set.

