"""Pedal axis state from SDL motion events, and the pedal-connect tap detector. See README.md."""

from __future__ import annotations

from dataclasses import dataclass

# Tap size on the [-1, 1] scale. End-rest pedals use the smaller one.
# A middle-rest axis, including a shared gas and brake, uses the larger.
PEDAL_TAP_THRESHOLD = 0.15
TAP_THRESHOLD = 0.3
# Within this of -1 or +1, an axis is resting at that end.
_END_BAND = 0.2
# First report within this of 0: one axis, gas one way and brake the other.
_CENTER_BAND = 0.1


def normalise_axis(raw: float, inverted: bool) -> float:
    """Map a [-1, 1] SDL axis value to [0, 1] pedal travel."""
    if inverted:
        raw = -raw
    return round((raw + 1) / 2, 3)


def split_pedal(raw: float, negative: bool) -> float:
    """One half of a center-rest axis. 0 from the middle through the other pedal, 1 at the stop."""
    side = -raw if negative else raw
    if side <= 0.0:
        return 0.0
    return round(min(side, 1.0), 3)


def _shared_axis(shared: bool, gas_axis: int, brake_axis: int) -> bool:
    """True only when the flag is on and both pedals are that one index."""
    return bool(shared) and gas_axis == brake_axis


class PedalAxes:
    """Gas and brake, each moved only by a motion event of its own axis. See the README."""

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.gas = 0.0
        self.brake = 0.0
        # An axis is live once it has reported; from then on SDL's value is real.
        self.gas_live = False
        self.brake_live = False
        self._brake_went_live = False

    def on_motion(
        self,
        axis: int,
        value: float,
        *,
        gas_axis: int,
        brake_axis: int,
        gas_inverted: bool,
        brake_inverted: bool,
        shared: bool = False,
    ) -> None:
        if _shared_axis(shared, gas_axis, brake_axis):
            if axis != gas_axis:
                return
            if not self.brake_live:
                self._brake_went_live = True
            self.brake_live = True
            self.gas_live = True
            self.gas = split_pedal(value, gas_inverted)
            self.brake = split_pedal(value, brake_inverted)
            return
        if axis == brake_axis:
            if not self.brake_live:
                self._brake_went_live = True
            self.brake_live = True
            self.brake = normalise_axis(value, brake_inverted)
        if axis == gas_axis:
            self.gas_live = True
            self.gas = normalise_axis(value, gas_inverted)

    def refresh(
        self,
        read_axis,
        *,
        gas_axis: int,
        brake_axis: int,
        gas_inverted: bool,
        brake_inverted: bool,
        shared: bool = False,
    ) -> None:
        """Re-read live axes each tick, so a missed event cannot leave a pedal stuck."""
        if _shared_axis(shared, gas_axis, brake_axis):
            if self.gas_live or self.brake_live:
                raw = read_axis(gas_axis)
                self.gas = split_pedal(raw, gas_inverted)
                self.brake = split_pedal(raw, brake_inverted)
            return
        if self.brake_live:
            self.brake = normalise_axis(read_axis(brake_axis), brake_inverted)
        if self.gas_live:
            self.gas = normalise_axis(read_axis(gas_axis), gas_inverted)

    def take_brake_went_live(self) -> bool:
        """True once after the brake's first report: its jump from 0 is not a stomp."""
        went, self._brake_went_live = self._brake_went_live, False
        return went


@dataclass(frozen=True)
class TapHit:
    guid: str
    axis: int
    inverted: bool
    magnitude: float
    pedal_like: bool = True
    combined: bool = False
    negative_half: bool = False
    shared_gas: bool = False


def _outranks(hit: TapHit, current: TapHit) -> bool:
    # End-rest pedal, then the other half of a middle-rest brake, then the bigger move.
    return (hit.pedal_like, hit.shared_gas, hit.magnitude) > (
        current.pedal_like, current.shared_gas, current.magnitude,
    )


class TapDetector:
    """Finds the axis a pedal tap moved, measured from that axis's first event. See the README."""

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self._rest: dict[tuple[str, int], float] = {}
        self._end: dict[tuple[str, int], float] = {}
        self._center: set[tuple[str, int]] = set()
        self._half_peak: dict[tuple[str, int, bool], float] = {}
        self._events: dict[str, int] = {}
        self._peak: dict[str, tuple[int, float]] = {}
        self._only_guid: str | None = None
        self._skip_axis: int | None = None
        self._split_axis: int | None = None
        self._gas_negative: bool | None = None
        self._hit: TapHit | None = None

    def restrict(self, guid: str, skip_axis: int, *, gas_negative: bool | None = None) -> None:
        """Gas stage: same device. Skip the brake axis, or take its other half."""
        self._only_guid = guid
        self._skip_axis = None if gas_negative is not None else skip_axis
        self._split_axis = skip_axis if gas_negative is not None else None
        self._gas_negative = gas_negative
        self._hit = None
        if gas_negative is None:
            return
        magnitude = self._half_peak.get((guid, skip_axis, gas_negative), 0.0)
        if magnitude <= TAP_THRESHOLD:
            return
        self._hit = TapHit(
            guid, skip_axis, gas_negative, magnitude,
            pedal_like=False, combined=True, negative_half=gas_negative, shared_gas=True,
        )

    def on_motion(self, guid: str, axis: int, value: float) -> None:
        key = (guid, axis)
        first = self._rest.setdefault(key, value)
        if key not in self._center and key not in self._end and abs(first) <= _CENTER_BAND:
            # First report in the middle: gas one way, brake the other. See the README.
            self._center.add(key)
        end = self._end.get(key)
        if key not in self._center and end is None and abs(value) >= 1.0 - _END_BAND:
            # First end this axis sits at is its rest. A middle-rest axis never
            # adopts one: a press to the stop is not where it rests.
            end = self._end[key] = 1.0 if value > 0 else -1.0
        negative_half = value < 0.0
        if key in self._center:
            magnitude, inverted, threshold = abs(value), negative_half, TAP_THRESHOLD
            pedal_like, combined = False, True
            peak = self._half_peak.get((guid, axis, negative_half), 0.0)
            if magnitude > peak:
                self._half_peak[(guid, axis, negative_half)] = magnitude
        elif end is not None:
            magnitude, inverted, threshold = abs(value - end), end > 0, PEDAL_TAP_THRESHOLD
            pedal_like, combined = True, False
        else:
            magnitude, inverted, threshold = abs(value - first), value < first, TAP_THRESHOLD
            pedal_like, combined = False, False

        self._events[guid] = self._events.get(guid, 0) + 1
        peak_move = self._peak.get(guid)
        if peak_move is None or magnitude > peak_move[1]:
            self._peak[guid] = (axis, magnitude)

        shared_gas = False
        if self._only_guid is not None and guid != self._only_guid:
            return
        if self._gas_negative is not None and axis == self._split_axis:
            if negative_half != self._gas_negative or abs(value) <= TAP_THRESHOLD:
                return
            magnitude = abs(value)
            inverted = negative_half
            pedal_like, combined, shared_gas = False, True, True
        elif (
            self._only_guid is not None
            and self._skip_axis is not None
            and axis == self._skip_axis
        ):
            return
        elif magnitude <= threshold:
            return

        hit = TapHit(
            guid, axis, inverted, magnitude,
            pedal_like=pedal_like, combined=combined,
            negative_half=negative_half, shared_gas=shared_gas,
        )
        if (
            self._gas_negative is None
            and self._hit is not None
            and self._hit.combined
            and hit.combined
            and self._hit.guid == hit.guid
            and self._hit.axis == hit.axis
            and self._hit.negative_half != hit.negative_half
        ):
            # The first direction is the brake. The other half is the gas tap.
            return
        if self._hit is None or _outranks(hit, self._hit):
            self._hit = hit

    def take_hit(self) -> TapHit | None:
        hit, self._hit = self._hit, None
        return hit

    def summary(self, names: dict[str, str]) -> str:
        """One log line: what each device did while the flow waited for a tap."""
        parts = []
        for guid, name in names.items():
            events = self._events.get(guid, 0)
            if not events:
                parts.append(f"{name!r}: no axis events")
                continue
            axis, peak = self._peak[guid]
            parts.append(f"{name!r}: {events} events, max {peak:.2f} on axis {axis}")
        return "; ".join(parts) if parts else "no joysticks"
