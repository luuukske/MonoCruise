# SCS profile settings (ETS2 / ATS)

Read the **currently selected** profile's transmission type and braking
intensity from disk (and live telemetry when the SDK is up). File I/O and the
existing telemetry mapping only. No process injection, no memory scans, no
SII decryptor binaries.

## Why this is easy to get wrong

Steam Cloud splits one profile across two trees, and a user can have several
profiles. Picking "the newest folder under Documents" is not enough.

| Tree | Typical contents |
| --- | --- |
| `<user>/steam_profiles/<hex>/` | `config_local.cfg` (has `g_trans`, `g_brake_intensity`), `controls.sii` |
| `<user>/profiles/<hex>/` | Same layout for profiles that are **not** on Steam Cloud |
| Steam `userdata/<id>/<appid>/remote/profiles/<hex>/` | Cloud `config.cfg` (has `g_adaptive_shift` and the rest of gameplay) |

`<user>` is the game user directory: Documents `Euro Truck Simulator 2` /
`American Truck Simulator` on Windows, `~/.local/share/...` on Linux, or
whatever `-homedir` pointed at. Hex folder names are UTF-8 profile names.

Ignore `steam_profiles(<version>).bak`. Those are update snapshots, not the
live store.

## Selected profile

While a session exists, `game.log.txt` is the source of truth:

- `New profile selected: 'Name'`
- a path containing `steam_profiles/<hex>` or `steam/profiles/<hex>`

The last of each in the current log is the loaded profile. Folder mtime is the
fallback when there is no log (never launched, log rotated away).

`config_local.cfg` is rewritten on profile load and when the settings menu
closes. Mid-menu edits are still in memory only. The sequential/automatic
hotkey can also move transmission before disk catches up; `shifterType` from
the already-mapped `Local\SCSTelemetry` block is the live check for that.

## Keys

| Setting | Key | Where |
| --- | --- | --- |
| Transmission | `g_trans` | profile `config_local.cfg` |
| Adaptive gearbox | `g_adaptive_shift` | Steam Cloud `config.cfg` (or local `config.cfg`) |
| Braking intensity | `g_brake_intensity` | profile `config_local.cfg` |
| Brake analog deadzone | `c_brake_dz` | `controls.sii` |

`g_trans`: `0` arcade / simple automatic, `1` sequential, `2` H-shifter, `3`
automatic / realistic automatic.

Telemetry `shifterType` strings: `arcade`, `manual` (sequential), `hshifter`,
`automatic`.

`g_adaptive_shift` (automatic only): `0` off, `1.66` power, `3` normal, `10`
eco. Other values are custom.

`g_brake_intensity` is the Gameplay **Braking intensity** slider, not a
hidden analog curve. The cvar makes the same pedal brake harder: about `1/3`
left, `1.0` centre, `3` right. The UI labels that **50% / 100% / 150%**; those
labels are not the gain. 150% stores `3.0`, so light braking is about 3x as
strong as centre, while full brake rises only ~1.3x, most likely because the
tyres reach their grip limit and ABS caps the decel. Analog "sensitivity" in
Controls is only the deadzone (`c_brake_dz`) plus whatever the `abackward` mix
does; stock mixes are linear.

`g_intelligent_transmission` lives in the **global** `config.cfg` and is not
the Controls transmission dropdown.

## Send remap

Mapper, AEB and ACC were tuned at cvar `1.1`. The sending thread inverts the
live cvar `I` as the last step before `SCSController.abackward`:

`sent = min(1, logical * 1.1 / I)`

So a given pedal brakes the same at every slider setting, the driver's own
pedal included. That is the requirement, confirmed by feel: keep it.

Capacity learning undoes the same remap with `effective_brake_pedal` and never
scales decel. AEB counts on `tune_max * min(1, I / 1.1)`: the traction-limited
extra a high slider adds at full pedal is left unused. AEB and a manual
emergency-stop slam use `full_authority` and write the logical pedal. Do not go
back to `p ** (I / 1.1)` or to UI%/100 as the gain (150% UI is `I = 3`, not 1.5).
Measurements and the reasoning are in `core/sending_thread/README.md` under
**Brake intensity**.

At `I = 1.1` this is identity. At `I = 1.0` (100% UI) cruise is `* 1.1`, the
linear stand-in for the old `b ** 0.91`. Unreadable files behave as `I = 1.0`.
A weak slider (`I < 1.0`) cannot be fully recovered: pedal 1.0 only reaches
`I / 1.1` of the tune's travel. If AEB is enabled, warn once an hour.

## AV

Read existing text files and the telemetry shared-memory mapping MonoCruise
already uses. Do not add process listing, `shell=True`, decryptor downloads,
or writes into the game folder. Steam `userdata` is read, never written.

## Probe

```bash
python tools/read_scs_profile.py
python tools/read_scs_profile.py --json
```
