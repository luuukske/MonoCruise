# Boot-time update check

Mirrors `core.sdk_installer.start_boot_check`: a one-shot daemon thread runs the
(possible) GitHub round-trip off the Qt main thread. Failures are logged and
swallowed; boot is unchanged when offline.

Once per boot (subject to `THROTTLE_SECONDS`) the module compares the newest
release on the user's channel (stable vs preview) to the running build. Results
are cached in `Settings` so the UI can show a pending update without another
network call.

## UI surfaces

- **Popup (opt-in):** when `Settings.notify_for_updates` is true, a newer build
  is known, and `popup_throttled()` is false. Gated on `last_update_popup`, not
  `last_update_check`, so a skipped popup does not delay the next prompt by a
  full throttle window.
- **Banner + update-button tint:** `update_is_pending()` on the main thread;
  always reflects cache, including throttled boots.

## Settings cache fields

`last_update_check`, `latest_known_version`, and `last_update_popup` in `config.json`
are written by this module (not user knobs). See field comments in `core/settings.py`.


When the checker-launched session is minimized and the game disconnects,
`monocruise.py` may auto-close after the viz bar settles. Auto-close is skipped
while `update_is_pending()` is true so banner, tint, and popup stay available.
