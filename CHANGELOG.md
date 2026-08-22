# Changelog

All notable changes to MonoCruise are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

Sections under each release (in order): `Security`, `Added`, `Changed`, `Fixed`, `Removed`, `Known`.

The `[Unreleased]` block accumulates changes between releases. `tools/release.py bump`
renames it to the released version and starts a fresh `[Unreleased]` above it.

## [Unreleased]
### Added
- **Pick how eagerly cruise control accelerates**: a new Acceleration style setting under Cruise Control offers Efficiency, Normal and Sport. Normal is the default. Efficiency eases off soon after pulling away and stays gentle; Sport uses as much of the engine as your truck and load will give, so on a steep hill it is the truck holding you back rather than the setting.

- **Set how far ACC follows**: there is now a following-distance setting under Adaptive Cruise Control with four steps, from closest to farthest. You can also bind buttons to change it while driving, under the cruise control buttons in the settings. Bind only one and it cycles through all four.

### Changed
- **Cruise control pulled away too slowly and pushed too hard at speed**: it used to aim for the same acceleration at 15 km/h as at 85 km/h. It now accelerates noticeably harder from low speed and eases off as you get faster, and it no longer asks for more than your truck and load can actually deliver.

- **ACC follows a little further back by default**: the starting following distance moved up one step.
- **ACC's following distance setting now sets how relaxed it drives**: a farther setting is calmer about the gap, a closer one stays eager and pulls up to the vehicle ahead sooner, and neither goes slack just because the vehicle ahead is a long way off. A vehicle pulling away from you barely counts any more, since the gap is opening on its own, unless you are already close behind it. The gap you settle at is unchanged on every setting.

### Fixed
- **AEB braked without making a sound**: when a vehicle cut in and then moved away again, AEB kept braking but the alert never played, so the first thing you noticed was the truck slowing on its own. The alert now sounds for as long as AEB is braking.
- **ACC braked far too late and far too hard for stopped vehicles**: it would coast in with a light brake and then slam on hard at the last moment. It now starts slowing early and stops with roughly half the braking force, no jolt. In a test where the stopped vehicle only came into view at 110 m, ACC used to hit it.
- **Wheel and stalk buttons could not be assigned with the game closed**: clicking a cruise control button in the settings and pressing a button on your wheel, stalk or gamepad did nothing unless the game was running, while keyboard keys assigned fine. Assignment now works with the game closed, and buttons you already assigned show up as pressed there too.
- **AEB braking for traffic under a bridge**: MonoCruise used to see cars under bridges as on your road due to a straight-ahead check. It now follows the road's shape and checks each vehicle's tilt to tell if it's actually on your route. Cars under bridges within 30 m are now ignored.
- **ACC losing the vehicle ahead on hills at speed**: that same straight line went badly wrong over crests and dips, and the faster you went the further ahead your lead was, so it fell outside the slack and ACC let go of it. Above 85 km/h it was dropping 4 % of the traffic it should have been following, and over 11 % on steep grades. Both are now effectively zero.

## [1.1.0-preview.19] - 2026-08-13
### Changed
- **AEB intervention popup**: the intervention popup now a notice. a long time ask from a community member, i forgot about...

### Fixed
- **AEB beep cut off the moment the warning ended**: the alert stopped mid-chirp as soon as AEB let go, often while you were still braking or coasting. It now finishes the current beep and plays one more after the warning clears.

## [1.1.0-preview.18] - 2026-08-13
### Changed
- **AEB holds its braking force instead of easing off near the end**: the head start it reserves for the brakes to build up was being re-charged every moment, so as you slowed it quietly handed that distance back and the braking faded away while the warning was still going. It is now paid once, when AEB commits. Stops are shorter, and hold a steady force the whole way.


### Fixed
- **Cruise late on hills**: the pedals barely moved with the slope, so speed ran away downhill before cruise braked. Real grades now move them sooner.
- **AEB beeped with no warning on the cruise panel**: a warning that appeared and vanished in a moment could still play the full sound after the icon had already missed it. The beep now waits until the warning holds, and braking to dismiss it cuts the sound at once.
- **AEB still beeped during light cruise or one-pedal braking**: a small ACC or OPD brake did not count as braking, so the warning kept sounding while MonoCruise was already slowing you. Any of that brake now silences it. Hard automatic braking still warns you.
- **No screenshot on TruckersMP clips**: clips sent to help improve AEB/ACC from multiplayer had no picture of the road. They include one again.
- **AEB braking with trailers**: fixed early/late AEB triggers and slow stops after hooking up, plus trailer brake delay is now realistic for all rigs; braking adjusts instantly to your current setup.

## [1.1.0-preview.17] - 2026-08-11
### Changed
- **ego's collision box**: changed the collision box to be more accurate to the game (Volvo FH6 780 6x4).
- **AEB steps in slightly later on traffic in your own lane**: preview.12 made it engage earlier for vehicles squarely in front of you, which turned out to feel over-eager now that braking eases off properly. It waits as long as it does for everything else again, and brakes harder when it does commit.

### Fixed
- **ACC randomly losing tracking**: ACC was randomly losing tracking of the lead vehicle. this was caused by the new road prediction model, which was (still is really) not fully implemented. this has been fixed by adding a failsafe to the tracking system.
- **AEB grabbed the brakes early at low speed**: a timer forced it to engage whenever a collision was about a second and a half away, even when barely any braking was needed. At crawl speed that stopped you a metre or two short of the vehicle in front; it now waits until real braking is actually called for.
- **AEB braking was all-or-nothing**: every trigger went straight to the floor instead of braking as hard as the situation needed, so a 65 km/h stop finished around 5 m short of the vehicle ahead. It now works out what your rig can really do from its axles and weight, eases off as the gap closes, and still uses everything when a crash can't be avoided.
- **Weight was wrong after dropping a trailer**: with a job still assigned, MonoCruise kept counting cargo you were no longer pulling and read an empty cab as nearly four times its real weight. That threw off braking, throttle response and creep until you hooked up again.

## [1.1.0-preview.16] - 2026-08-10
### Fixed
- **MAJOR: preview.15 closed instantly on startup**: the build was missing one of its own files, so MonoCruise quit before the window ever appeared. It was broken for about an hour after release, and this build fixes it. If you are on preview.15, update.

## [1.1.0-preview.15] - 2026-08-10
### Added
- **Help improve AEB/ACC**: added a feature to send clips to help me improve the AEB/ACC system filtering system. this does not store any personal information or data, it is only used to help me improve MonoCruise's behavior. **Please turn this on so i can improve AEB/ACC!**

### Changed
- **Bold text whighter**: the bold text is now lighter to give a higher contrast withe the rest of the text.
- **General AEB filtering improvements**
### Fixed
- **Cruise control detected double presses**: a quick press could change the set speed twice, or switch cruise on and straight back off. One press is now one press.
- **Confirmation card layout**: the confirmation card now has a more consistent layout and is now alwayscentered.

## [1.1.0-preview.12] - 2026-08-07
### Added
- **Road prediction model**: ACC used to work out your lane from your own steering wheel, which only ever describes the road *behind* you. It now predicts the road ahead from the paths the traffic in front is actually driving, and knows how far out that prediction is still worth trusting (not used by AEB yet). Since this is what picks your lead, it settles a family of long-standing complaints:
  - **Parked and shoulder vehicles grabbed from far away**: one standing off to the side could become your lead from 60 m back.
  - **Lead dropped halfway through a curve**: the car ahead stopped counting well before it left your lane, so you accelerated into the bend and then braked late.
  - **Your lane placed to the outside of tight bends**: sharp corners read as gentler than they are, by over a metre at 60 m ahead. Motorway curves were never affected.
  - **Seconds of hesitation on winding roads**: the collision warning could speak up before ACC had accepted a car that was simply slower than you.
  - **Standing vehicles treated as the most certain target on the road**: they now have to earn it, and one you watched drive into place before it stopped still counts properly.
  - **The predicted lane jumped around in traffic and through corners**: both are damped now, without making it sluggish.
  - **The prediction faded in and out with traffic all around**: more traffic holding the same road now makes it steadier rather than shakier.
  - **Oncoming traffic was ignored entirely**: it now counts too, which is most of the difference on two-way roads.
  - **Slow to pick up a lead and slow to let one go**: roughly a fifth off both.
  - **The prediction gave up in tight, low-speed corners**: it cut off partway round junctions, roundabouts and hairpins, and tracking went with it. It now follows a corner the whole way round.
  - **Slow traffic read as standing still and took too long to pick up**: it kept too little of its own path to place in a lane. Slow vehicles directly ahead are now picked up in about a third of the time.
  - **Queued traffic did not help ACC recognise itself**: stopped vehicles lined up along the road now back each other up. One standing on its own is treated exactly as before, and a line that does not match the road you are on counts for nothing.

- **ACC intelligently handles lane changes**: ACC now looks at if you are passing, getting passed or just have a vehicle you don't want to follow next to you. This new system is also able to keep the vehicle you are trying to pass into account while doing the lane change.

### Fixed
- **ACC braked hardest for the targets it was least sure about**: full emergency braking could fire on a vehicle it had barely started tracking, usually something parked near the road. Maximum braking now needs the same confidence the rest of ACC uses, and close-range emergency braking is unchanged.
- **AEB braked late for traffic stopping ahead (TMP)**: a vehicle braking to a standstill in front of you could be mistaken for a stalled connection, so MonoCruise kept reading it as still moving for up to a second and a half and left the emergency brake far too late.
- **AEB warned after a crash with nothing in front of you**: being flipped, launched, or left sitting at a steep angle could set off the collision warning and add brake help on its own, with no vehicle anywhere near. A slope alone no longer counts as a hazard.
- **AEB braked for oncoming traffic on gentle bends**: on a long motorway curve a car coming the other way lined up with your bonnet from 40-90 m out and read as head-on, even though it passed cleanly. MonoCruise now checks whether the two of you are actually converging before braking, so traffic measured to pass clear is left alone. Genuine wrong-way drivers are unaffected.
- **AEB braked while turning at junctions and roundabouts**: holding a tight steering angle projected your path across the road you were turning onto, so traffic already on it looked like a collision from 30-60 m away. Far-off crossings found this way no longer trigger the brake; anything close still does.
- **AEB braked for vehicles driving alongside you at the same speed**: a neighbour in the next lane you were neither catching nor being passed by could set off a hard brake mid-turn. Braking cannot avoid a sideways contact, so it no longer tries, unless the other vehicle is measurably drifting into you.
- **AEB missed a vehicle pulling out in front of you and stopping**: something merging in from a side road at a shallow angle was treated as uncertain cross-traffic and had to prove itself for too long, so the brake came late or not at all. Traffic heading the same way as you is now recognised straight away.
- **AEB left it too late on stopped and slow traffic in your own lane**: it warned about the vehicle ahead but waited until the situation needed almost everything the brakes had before stepping in. For traffic squarely in your lane it now steps in earlier, while everything it is less sure about is unchanged.
- **AEB stayed quiet about oncoming traffic that was actually coming at you**: a vehicle far enough to the side on paper was written off as a safe pass even when its measured path was aimed straight at you. A measured head-on course is no longer waved through.

## [1.1.0-preview.11] - 2026-07-24
### Changed
- **Prevent duplicate popups**: duplicate popups are now prevented by checking the dedup_key of the popup message.

### Fixed
- **MAJOR: pedals dead after updating from 1.0**: after an update the game could load the old plugin beside the new one and swallow gas and brake entirely while the UI still looked fine. The leftover plugin is now disabled automatically at startup; restart the game once after the message appears.
- **AEB still beeped while you were braking**: the warning kept sounding for a moment after every alert, and light braking did not count as braking at all. A gentle dab on the pedal now silences it, and so does ACC slowing for the hazard. Hard automatic braking still warns you.
- **AEB brake help arrived too late when you braked gently**: the extra braking force AEB adds on top of yours only switched on once you were already braking hard enough not to need it. It now fades in smoothly from a light dab, so gentle braking into a hazard gets help instead of nothing.
- **Speed limiter fighting ACC at the limit**: with the global limit on, ACC holding right at the cap could get brake stabs from the limiter's overshoot protection over tiny speed drifts. Overshoot protection now stays out until you are properly over the limit, so ACC has room to work.

### Known
- **Brake capacity vs vehicle weight**: after a loaded job the learned max brake can stay low (about right for ~28 t cargo, far too low once empty), so AEB times stops as if the truck still brakes weakly. Ceiling policy was loosened so hard settled braking can raise it again; that is only a workaround. The mass-adjusted brake baseline underneath is still wrong and needs a real fix.

## [1.1.0-preview.10] - 2026-07-23
### Added
- **AEB intervention popup**: a warning popup now confirms when Automatic Emergency Braking holds long enough to count as a real intervention, not a flicker.

### Changed
- **internal docs reorganized into README.md and AGENTS.md files**: the internal docs were scattered across the codebase, making it difficult to find the information i needed. they have been organized into README.md and AGENTS.md files to make it easier to find the information i need.
- **Smarter speed-limiter braking on hills**: the limiter now brakes progressively harder the further you are over the limit, and eases off when the truck is already slowing on its own, so downhill and crest overshoots come back sooner without surprise brake piles. I would even say, it is better than most irl limiters. I outsmarted them.

### Fixed
- **Loaded truck slow to come back down to the set speed**: after a long climb the truck could sit above the set speed or the speed limit for tens of seconds with the throttle still feeding in. It now drops the throttle right away and brakes when it needs to.
- **False emergency stop with foot off the brake**: on Windows the pedal reader could mis-time a resting brake and slam full emergency stop. Timing and slam detection now ignore an untouched pedal.
- **AEB warning sound silent**: the warning beep failed to load its sound file; it plays again.

## [1.1.0-preview.9] - 2026-07-22
### Changed
- **ACC speed filtering smoothed out (TMP)**: ACC now reads its own filter chain instead of sharing one with AEB, so AEB's hard-brake responsiveness no longer leaks jitter into ACC's throttle/brake behavior. AEB's crash and lag detection are unaffected.

### Fixed
- **ACC disengages when accelerating**: auto-neutral now owns the gearbox when shifting to neutral. ACC now ignores neutral when auto-neutral owns the gearbox.
- **ACC hugs the leading vehicle on hard braking**: a side effect of the smoothing above, follow gap on a hard stop was tighter than intended. ACC now backs off sooner while staying just as smooth.
- **AEB braked late for crashed traffic (TMP)**: a vehicle crashing ahead could be mistaken for network lag, and network lag for a crash, delaying the emergency brake by up to a second. Crash and lag are now told apart reliably and a confirmed crash gets AEB's fastest response.

## [1.1.0-preview.8] - 2026-07-21
### Added
- **Autostart / auto-close**: starting with a game already running opens MonoCruise minimized and auto-quits it after the game closes; without a game everything behaves as before.
- **Game plugin auto-install**: a missing or outdated game plugin is now installed automatically at startup, with a reminder to restart the game.

### Fixed
- **AEB slow to recognize a stopped lead**: hard-braking traffic now switches to a responsive speed estimate, while steady driving keeps the existing smooth filtering.
- **AEB taps skewed learned brake strength**: short brake pulses taught from mid-transient readings, throwing off AEB timing. Learning now waits for braking to stabilize; hard AEB stops still teach it fast.
- **Hill starts blocked on steep grades**: a small rollback made the hill-hold keep braking against the gas. It now releases as soon as the truck stops rolling back.
- **Anti-creep too strong at launch**: weak engines couldn't overcome the creep-cancel brake; it now releases much earlier on the gas pedal.
- **Live pedal bar drops on pause**: pausing the game no longer makes the pedal bar look like cruise control disengaged.
- **AEB warn beeps while ACC brakes**: ACC follow-braking no longer triggers the AEB warning sound.
- **Updater install icon brightens with progress**: stage icons no longer fade in with install percent; progress stays on the connecting lines only.
- **Limiter brake lights fade**: Releasing brake with speed limiter active now keeps the smooth timeout fade, instead of instantly cutting lights if still on the gas.
- **Update popup delay fixed**: spacing now tracks when the popup is actually shown, not just when update checks occur.

### Known
- **ACC disengages at standstill**: when using auto-neutral, ACC disengages when starting again. this will be fixed in preview.9.

## [1.1.0-preview.7] - 2026-07-20
### Added
- **Auto-neutral at stops** (opt-in): shifts to neutral at low speed whenever the brake is on and the gas is off; off by default in settings.
- **Gear-engage creep cushion** (OPD or auto-neutral only): mapper creep cancel while D/R is closed; 100% on OPD brake, faded out with OPD gas so launch is smooth without killing reverse lights.
### Changed
- **AEB brakes at the last moment**: engagement threshold raised to 85% of usable capacity, standoff buffer reduced 1.6 m -> 0.2 m, and the speed-proportional response margin cut 0.45 s -> 0.10 s (physical actuator lag). AEB now waits for the last-point envelope instead of braking seconds early; the clip corpus keeps arbitrating WHICH targets are threats, not when to brake.
- **ACC follows AI vehicles at the true gap**: lead distance came from the same asymmetric bodies, so ACC held ~1.5-4 m more real gap than commanded behind AI traffic. Same gap setting now means the same physical gap.
- **Smoother low-speed braking**: creep compensation on the user brake path, proportional brake-hold release, and OPD pedal-cliff smoothing near standstill.
- **Speed limiter fixes**: stale-target re-clamp and no more surge when engaging at the cap.
### Fixed
- **USB button / stalk assignment**: HID capture no longer waits a half-second confirmation window that marked early presses as noise and made buttons look dead. Also fixed joystick-class devices pygame skips (e.g. MOZA Multi-function Stalk) so they are scanned via HID, opened by the correct interface path, and covered by declaring `hidapi` as a runtime dependency.
- **AEB braking for air behind AI traffic**: the collision model placed every AI vehicle's body asymmetrically around its position, extending it 1/3 of its length past the real rear bumper (~1.6 m on cars, ~4 m on trucks). AEB braked for that phantom, felt as a constant "invisible wall" behind SP traffic. Bodies are now symmetric, matching where vehicles actually are (validated against the AEB debug radar and live standoff measurements).
- **AEB brake pumping**: one approach could engage, release mid-stop while still closing, and re-engage up to 3 times. Braking itself pushed the internal "required decel" under the release threshold. An active event now holds until the threat actually resolves (target clears, pulls away, or you stop): one continuous brake per event.
- **Brake capacity estimate rotting**: the learned max-brake estimate drifted from ~9 down to ~4 m/s2 during normal driving (gentle presses extrapolate badly on the game's progressive brake curve), making AEB believe the truck brakes 3x worse than it does and fire at 2-3x the needed distance. Normal driving now only drifts the estimate slowly; AEB braking events (deep, honest presses) re-teach it fast.

## [1.1.0-preview.6] - 2026-07-16
### Changed
- **Release workflow**: creator of the version now correctly mentioned on the release instead of `github-actions[bot]` and added minor changes to release script. this is basically a test run.

## [1.1.0-preview.5] - 2026-07-16
### Added
- **Pedals lost banner**: banner now shows your pedals disconnected, just like v1.0.4.
- **Update available popup**: a popup to notify the user of an update. this can be turned off in the setting.
### Fixed
- **Connect pedals actually work**: wired up the connect button to be more reliable and actually work.
- **AEB/ACC speeds after pause**: vehicle speeds collapsed to ~0 on unpause (and similar hitches) because kinematics used wall-clock time across the gap; radar now integrates on the game's simulatedTime and holds filters while sim time is frozen.
- **AEB general improvements**: capsule bodies + ego-arc steering wiggle were beeping on adjacent-lane overtakes and passes; parallel-margin scaling, rear-overtaker filter suppress, and braking-worsens for cleared rear overtakers kill the class-A/B phantoms (clips f0b2ace6 etc.).
- **ACC oscilate at full throttle**: ACC would be switching between 95% throttle and 100% caused by the downshifting.
### Changed
- **ACC follow distance**: made the distance more realistic. the original values were placeholders.
- **AEB less sensitive**: changed back the AEB brake latch from 70% max brake to 80% max brake.
- **ACC smoother in SP**: SP now uses the same filtering from TMP as it showed so much success, i wanted to move the two systems together. only crash detection and lag detection stays for TMP only.
- **Less AEB shadow_near**: changed the shadow near to only clip about 6/h.
- **Limiters more responsive**: limiters react faster to speed changes and hold their speed more accuratly. 

## [1.1.0-preview.4] - 2026-07-11
### Fixed
- **AEB corner filtering**: filtering for AEB improved **MASIVELY** using clips gathered from AEB clipper (testers only), mostly focusing on corners, but general improvements can be seen.
- **AEB improved brake capacity**: improved the stability of the brake capacity estimation for correct AEB triggering. this prevents late reaction from the AEB system previously seen in one of the clips.
- **Highway slow queue FN**: fixed a (really really bad) bug in the AEB filtering causing slow moving traffic to be ignored when above certain high speed. also found thanks to the clipping.
- AEB clipping irregularities
- **OPD known issues**: reverted to legacy code. sending_thread caps opdgasval now, not just raw gas input. opd offset actually effects the gas output now. hard to get moving at slow speeds.

## [1.1.0-preview.3] - 2026-07-09
### Added

- **AEB clip capture** (debug mode only): when AEB triggers, MonoCruise saves a short replay clip plus a screenshot thumbnail to `%LocalAppData%/MonoCruise/`. Intended for testers to report false positives or missed detections. Send clips manually (no automatic upload). You'll get an on-screen notification when a clip is saved.

### Changed

- **Debug mode off by default**: developer tools (AEB radar view, clip capture) now require enabling debug in settings. Preview builds previously had this on.
- **Updater hands off after installing**: once an update completes, the updater shows the finished state for a moment, starts MonoCruise and closes itself.

### Fixed

- **AEB false triggers in corners**: cross-traffic that sweeps clear at intersections no longer fools the threat filter, and AEB won't engage when the target's movement already shows it will pass beside you. (thanks to eary AEB clips captured by me)
- **Updater self-updates now actually apply**: the updater's own new version used to be staged but never swapped in. A file the running updater keeps open blocks the swap, and the staged update was silently discarded afterwards. MonoCruise now applies the staged updater files once the updater has closed. This also removes the broken in-place swap that could leave an old updater unable to reach GitHub.

## [1.1.0-preview.2] - 2026-07-06

### Added

- **Updater closes MonoCruise for you**: clicking Update while MonoCruise is running now asks the app to shut down cleanly (settings saved, pedals released) instead of showing an error. If it will not close within 15 seconds, the old "Close MonoCruise before updating" message still appears.

### Changed

- **Failed updates show in red**: when an update fails, the stage it failed on (download/install) turns red in the updater's progress column.

### Fixed

- **Updater window icon**: the updater now shows the MonoCruise icon in its title bar and on the taskbar instead of the default icon.
- **AEB debug view no longer opens on every start**: the developer radar view now only appears in debug mode.

## [1.1.0-preview.1] - 2026-07-06

A ground-up rewrite focused on **stability** and **performance**, with a more reliable take on every existing feature: plus a built-in updater and on-screen notifications.

### Added

- **In-app updater**: installs new releases from GitHub without reinstalling; your config and logs are kept.
- **Stable / Preview update channels**: pick your channel in settings — Preview builds are released earlier and may contain bugs.
- **On-screen notifications**: always-on-top popups for updates, errors, and onboarding tips.
- **Lead-vehicle speed readout**: the lead truck's speed now shows on the CC panel above your set speed.
- **Multi-device button assignment**: cruise-control buttons can be assigned to any joystick button, hat direction, keyboard key, or USB button device (e.g. a button stalk) — click a configure button in settings and press the input. Assigned buttons light up while pressed, and an Unassign button clears a single binding.
- **ETS2 v1.60 support** (thanks to the automatic SDK fetcher).
- **Rewritten auto-start checker**: now simpler and antivirus-friendly — uses telemetry for game detection (no process or registry scanning), with installer-based startup opt-in and a plain-language log. Details in `checker/README.md`.
- **Global speed limiter**: a highly accurate global limiter so your truck never exceeds that set speed (useful for Trucky, for example).

### Changed

- **Keeps running when something breaks**: rewritten to one independent thread per subsystem, with a watchdog that detects a crashed or frozen part and restarts it automatically.
- **Faster, smoother UI**: switched from CustomTkinter to GPU-accelerated PySide6, lowering CPU usage and clearing a class of visual bugs.
- **More reliable ACC**: reworked lead-vehicle selection (arc-based in-lane scoring) sharply reduces the brake-checking the old ACC was prone to, and now accounts for road-trains (trailers-of-trailers).
- **Reworked AEB**: arc-trajectory geometry with staged braking (warning brake, then full brake) in place of the old straight-line check.
- **Self-tuning pedals**: gas/brake output now calibrates to your hardware over time (with per-gear learning), plus road-load/hill feedforward and a gearshift hold for smoother, more consistent control.
- **Automatic SDK installer**: automatically fetches the latest SDK for the latest ETS2/ATS version.
- **Thread-safe settings & logging**: settings save atomically with no global variables (faster and race-free), and errors can still surface via popup even after a worker thread crashes.
- **Much smaller downloads**: the installer and update packages no longer bundle unused Qt components (roughly 75% smaller).

### Fixed

- **Hazard flickering** during AEB braking.
- **CC vs. brake conflicts**: game braking now reliably disengages CC, and CC / limiter / user-pedal priority no longer fight each other.
- **CC panel on every display**: scale changes apply live (no restart), and the panel stays put on 4K and across mixed-DPI monitors.
- **Pedal bar misplacement** after waking from sleep.
- **One-Pedal braking** under speed-limiter mode.
- **Hazards** sometimes not switching off on acceleration.
- **Popup crash** on early `getattr()` calls.
- **Single-instance check**: MonoCruise now uses a named mutex instead of a process-name scan (which matched the app's own process in packaged builds); a second copy exits cleanly.
- **Release builds on PyInstaller 6**: the updater spec's script path stopped resolving under PyInstaller 6's spec-relative path rule.

**Known issues**

- ACC gap level can't be changed while actively following a lead vehicle: runtime adjustment is coming in a future update.
- AEB is experimental and can false-trigger in corners and during lane changes, so it is **disabled by default** — enable it in settings. **Use with care**.

