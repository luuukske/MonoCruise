Temporary probe. ETS2 loads it from `bin/win_x64/plugins`. Delete the DLL to remove it. It is not part of the SDK installer.

One second after you start driving, it plays a zero-mean rumble on the game's existing constant-force effect, then waits 10 seconds and does it again. Steering is not written. While the rumble is silent the game's force is passed through unchanged. The rumble peaks at 1600 of the 10000 full-scale units, and a stall cannot skip the fade-in.

`Local\MonoCruiseFfbProbe` is 8 bytes: a `uint32` generation, then an `int32` sign. Bump generation to play the sequence again. Set sign above zero if the push came out to the right.
