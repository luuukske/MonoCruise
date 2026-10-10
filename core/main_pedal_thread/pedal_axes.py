"""Pedal axis state from SDL motion events, and the pedal-connect tap detector. See README.md."""

from __future__ import annotations

from dataclasses import dataclass

# Tap size on the [-1, 1] axis scale: from the end a pedal rests at, or from
# the first value of an axis that never sat near an end (steering, odd calibration).
PEDAL_TAP_THRESHOLD = 0.15
TAP_THRESHOLD = 0.3
# Within this of -1 or +1, an axis is resting at that end.
_END_BAND = 0.2


def normalise_axis(raw: float, inverted: bool) -> float:
    """Map a [-1, 1] SDL axis value to [0, 1] pedal travel."""
    if inverted:
        raw = -raw
    return round((raw + 1) / 2, 3)


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
    ) -> None:
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
    ) -> None:
        """Re-read live axes each tick, so a missed event cannot leave a pedal stuck."""
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


class TapDetector:
    """Finds the axis a pedal tap moved, measured from that axis's first event. See the README."""

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self._rest: dict[tuple[str, int], float] = {}
        self._end: dict[tuple[str, int], float] = {}
        self._events: dict[str, int] = {}
        self._peak: dict[str, tuple[int, float]] = {}
        self._only_guid: str | None = None
        self._skip_axis: int | None = None
        self._hit: TapHit | None = None

    def restrict(self, guid: str, skip_axis: int) -> None:
        """Gas stage: same device as the brake, any axis but the brake's."""
        self._only_guid = guid
        self._skip_axis = skip_axis
        self._hit = None

    def on_motion(self, guid: str, axis: int, value: float) -> None:
        key = (guid, axis)
        first = self._rest.setdefault(key, value)
        end = self._end.get(key)
        if end is None and abs(value) >= 1.0 - _END_BAND:
            # The first end an axis sits at is where it rests; sticky, so a
            # full stomp reaching the other end does not move it.
            end = self._end[key] = 1.0 if value > 0 else -1.0
        if end is not None:
            magnitude, inverted, threshold = abs(value - end), end > 0, PEDAL_TAP_THRESHOLD
        else:
            magnitude, inverted, threshold = abs(value - first), value < first, TAP_THRESHOLD

        self._events[guid] = self._events.get(guid, 0) + 1
        peak = self._peak.get(guid)
        if peak is None or magnitude > peak[1]:
            self._peak[guid] = (axis, magnitude)

        if self._only_guid is not None and (guid != self._only_guid or axis == self._skip_axis):
            return
        if magnitude <= threshold:
            return
        hit = TapHit(guid, axis, inverted, magnitude, pedal_like=end is not None)
        # A pedal outranks any other axis, however far that one moved.
        if self._hit is None or (hit.pedal_like, hit.magnitude) > (
            self._hit.pedal_like, self._hit.magnitude,
        ):
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
