"""Shared-memory command channel to the game-side FFB plugin (plugins/ffb).

The plugin creates ``Local\\MonoCruiseFfbCmd`` and owns the waveform, the torque ramp and the steering
servo. This side sends intent only and bumps a heartbeat every call, so a stalled app lets the plugin
fade out on its own. Steering values are ``truck.input.steering`` units, counterclockwise positive.
"""

from __future__ import annotations

import logging
import math
import mmap
import struct
import sys
import time
from dataclasses import dataclass

logger = logging.getLogger(__name__)

TAG = r"Local\MonoCruiseFfbCmd"
SIZE = 128
V4_SIZE = 96
SERVO3_SIZE = 64
LEGACY_SIZE = 32
VERSION = 4
AHEAD_VERSION = 5
LEAD_MAX_S = 0.08
SERVO3_VERSION = 3
TORQUE_VERSION = 2
RUMBLE_ONLY_VERSION = 1
PEAK_UNITS = 1600
PEAK_CAP_UNITS = 3000
TORQUE_CAP_UNITS = 3000
SERVO_CAP_UNITS = 6000
CMD_NONE = 0
CMD_RUMBLE = 1
_RETRY_S = 2.0

_U32 = struct.Struct("=I")
_CMD = struct.Struct("=BBH")
_I16 = struct.Struct("=h")
_F32 = struct.Struct("=f")
_U8 = struct.Struct("=B")
_SERVO = struct.Struct("=fffHbb")
_SERVO4 = struct.Struct("=fff")
_AHEAD = struct.Struct("=fffff")
_PLAN = struct.Struct("=ff")
_OFF_VERSION = 0
_OFF_HEARTBEAT = 4
_OFF_CMD = 8
_OFF_SEQ = 12
_OFF_TORQUE = 16
_OFF_GAME_FORCE = 18
_OFF_STATUS = 20
_OFF_FRAMES = 24
_OFF_APPLIED = 28
_OFF_SERVO_APPLIED = 30
_OFF_SERVO = 32
_OFF_STEER = 48
_OFF_STEER_RATE = 52
_OFF_STEER_FRAMES = 56
_OFF_SERVO_SCALE = 60
_OFF_SERVO4 = 64
_OFF_STAMP = 76
_OFF_INTEG = 80
_OFF_FRAME_HZ = 84
_OFF_CANCEL_APPLIED = 88
_OFF_SERVO_FLAGS = 90
_OFF_AHEAD = 92
_OFF_FF_APPLIED = 112
_OFF_PLAN = 116

STATUS_HOOKED = 1
STATUS_FORCE_SEEN = 2
STATUS_RUMBLING = 4
STATUS_DEAD = 8
STATUS_TORQUE = 16
STATUS_SERVO = 32
STATUS_STEER = 64
STATUS_ONE_SIDED = 128
STATUS_DAMP_FIRST = 256
STATUS_PUSH_AHEAD = 512
STATUS_FF_PLAN = 1024
SERVO_DAMP_FIRST = 1
SERVO_FF_PLAN = 2


@dataclass(frozen=True)
class ServoCommand:
    """Hold the wheel on ``target``, moving at ``rate``: a PID with ``kp`` per unit of error, ``kd`` per unit of rate
    error and ``ki`` per unit of error and second, capped. ``cancel`` is the share of the game's slow force replaced.

    ``direction`` is the torque sign that turns the wheel counterclockwise; 0 is off. ``only`` +1 or -1: the servo
    pushes only that way (counterclockwise positive), never back; 0 both ways. ``damping_first`` gives the damping its
    room under the cap before P and I get theirs, so a servo pinned at its cap still brakes a swinging wheel.
    ``accel`` is the target's acceleration; ``ff_*`` the rim's inertia, damping and friction the plugin pushes ahead
    from the target's motion, and ``lead_s`` how far ahead it reads the wheel (layout 5; 0 is none). With ``ff_plan``
    the push ahead uses ``plan_rate`` and ``plan_accel`` instead: the target's rate and acceleration a wheel delay
    from now, as the commanding feature plans them, so the rim starts and stops on time.
    """
    target: float = 0.0
    kp: float = 0.0
    ki: float = 0.0
    cap: int = 0
    direction: int = 0
    rate: float = 0.0
    kd: float = 0.0
    cancel: float = 0.0
    only: int = 0
    damping_first: bool = False
    accel: float = 0.0
    ff_inertia: float = 0.0
    ff_damping: float = 0.0
    ff_friction: float = 0.0
    lead_s: float = 0.0
    ff_plan: bool = False
    plan_rate: float = 0.0
    plan_accel: float = 0.0


SERVO_OFF = ServoCommand()


@dataclass(frozen=True)
class PluginStatus:
    """What the plugin reports back. ``present`` is False until it has stamped the layout version.

    Version 1 only rumbles, version 2 adds a torque, version 3 the wheel's own angle and a servo, version 4 the
    damped servo that cancels the game's slow force, version 5 the push ahead and the lead. Versions 4 and 5 steer.
    """

    present: bool = False
    hooked: bool = False
    force_seen: bool = False
    rumbling: bool = False
    dead: bool = False
    frames: int = 0
    version: int = 0
    game_force: int = 0
    applied: int = 0
    torquing: bool = False
    servoing: bool = False
    steer_ok: bool = False
    one_sided: bool = False
    damping_first: bool = False
    push_ahead: bool = False
    ff_plan: bool = False
    steer: float = 0.0
    steer_rate: float = 0.0
    steer_frames: int = 0
    servo_applied: int = 0
    servo_scale: float = 1.0
    servo_integ: float = 0.0
    frame_hz: float = 0.0
    cancel_applied: int = 0
    ff_applied: int = 0

    @property
    def torque_capable(self) -> bool:
        return self.present and self.version >= TORQUE_VERSION and self.hooked and not self.dead

    @property
    def servo_capable(self) -> bool:
        return self.torque_capable and self.version >= VERSION and self.steer_ok


def decode_status(raw: bytes) -> PluginStatus:
    version = _U32.unpack_from(raw, _OFF_VERSION)[0]
    sizes = {RUMBLE_ONLY_VERSION: LEGACY_SIZE, TORQUE_VERSION: LEGACY_SIZE, SERVO3_VERSION: SERVO3_SIZE,
             VERSION: V4_SIZE, AHEAD_VERSION: SIZE}
    if version not in sizes or len(raw) < sizes[version]:
        return PluginStatus()
    bits = _U32.unpack_from(raw, _OFF_STATUS)[0]
    base = dict(
        present=True,
        hooked=bool(bits & STATUS_HOOKED),
        force_seen=bool(bits & STATUS_FORCE_SEEN),
        rumbling=bool(bits & STATUS_RUMBLING),
        dead=bool(bits & STATUS_DEAD),
        frames=_U32.unpack_from(raw, _OFF_FRAMES)[0],
        version=version,
    )
    if version >= TORQUE_VERSION:
        base.update(
            game_force=_I16.unpack_from(raw, _OFF_GAME_FORCE)[0],
            applied=_I16.unpack_from(raw, _OFF_APPLIED)[0],
            torquing=bool(bits & STATUS_TORQUE),
        )
    if version >= SERVO3_VERSION:
        base.update(
            servoing=bool(bits & STATUS_SERVO),
            steer_ok=bool(bits & STATUS_STEER),
            one_sided=bool(bits & STATUS_ONE_SIDED),
            steer=_F32.unpack_from(raw, _OFF_STEER)[0],
            steer_rate=_F32.unpack_from(raw, _OFF_STEER_RATE)[0],
            steer_frames=_U32.unpack_from(raw, _OFF_STEER_FRAMES)[0],
            servo_applied=_I16.unpack_from(raw, _OFF_SERVO_APPLIED)[0],
            servo_scale=_F32.unpack_from(raw, _OFF_SERVO_SCALE)[0],
        )
    if version >= VERSION:
        base.update(
            servo_integ=_F32.unpack_from(raw, _OFF_INTEG)[0],
            frame_hz=_F32.unpack_from(raw, _OFF_FRAME_HZ)[0],
            cancel_applied=_I16.unpack_from(raw, _OFF_CANCEL_APPLIED)[0],
            damping_first=bool(bits & STATUS_DAMP_FIRST),
        )
    if version >= AHEAD_VERSION:
        base.update(
            push_ahead=bool(bits & STATUS_PUSH_AHEAD),
            ff_plan=bool(bits & STATUS_FF_PLAN),
            ff_applied=_I16.unpack_from(raw, _OFF_FF_APPLIED)[0],
        )
    return PluginStatus(**base)


def intensity_units(scale: float) -> int:
    """Peak magnitude out of 10000 for a user scale, never above the cap the plugin also enforces."""
    return max(0, min(PEAK_CAP_UNITS, round(PEAK_UNITS * scale)))


def torque_units(torque: float) -> int:
    """Assist torque out of 10000, signed as the game's own force, clamped to the plugin's cap."""
    if not math.isfinite(torque):
        return 0
    return max(-TORQUE_CAP_UNITS, min(TORQUE_CAP_UNITS, round(torque)))


def servo_fields(cmd: ServoCommand) -> tuple:
    """The servo request as written, ``(target, kp, ki, cap, direction, rate, kd, cancel, only)``.

    Anything non-finite, a negative gain or no direction is off.
    """
    values = (cmd.target, cmd.kp, cmd.ki, cmd.rate, cmd.kd, cmd.cancel)
    gains = (cmd.kp, cmd.ki, cmd.kd)
    if cmd.direction not in (-1, 1) or not all(math.isfinite(v) for v in values) or min(gains) < 0.0:
        return (0.0, 0.0, 0.0, 0, 0, 0.0, 0.0, 0.0, 0)
    target = max(-1.0, min(1.0, cmd.target))
    cap = max(0, min(SERVO_CAP_UNITS, int(cmd.cap)))
    only = (cmd.only > 0) - (cmd.only < 0)
    return (target, cmd.kp, cmd.ki, cap, cmd.direction, cmd.rate, cmd.kd, max(0.0, min(1.0, cmd.cancel)), only)


def servo_ahead(cmd: ServoCommand, fields: tuple) -> tuple:
    """``(accel, ff_inertia, ff_damping, ff_friction, lead_s)`` as written at offset 92: zeros while the servo is off,
    a bad number or a negative gain is none, the lead at most 0.08 s.
    """
    values = (cmd.accel, cmd.ff_inertia, cmd.ff_damping, cmd.ff_friction, cmd.lead_s)
    if not fields[4] or not all(math.isfinite(v) for v in values):
        return (0.0, 0.0, 0.0, 0.0, 0.0)
    accel, inertia, damping, friction, lead = values
    return (accel, max(0.0, inertia), max(0.0, damping), max(0.0, friction), max(0.0, min(LEAD_MAX_S, lead)))


def servo_plan(cmd: ServoCommand, fields: tuple) -> tuple:
    """``(plan_rate, plan_accel)`` as written at offset 116: zeros unless the servo is on and ``ff_plan`` asks."""
    values = (cmd.plan_rate, cmd.plan_accel)
    if not fields[4] or not cmd.ff_plan or not all(math.isfinite(v) for v in values):
        return (0.0, 0.0)
    return values


def servo_flags(cmd: ServoCommand, fields: tuple) -> int:
    """The flag byte at offset 90: damping first, the planned push ahead; only while the servo is on."""
    if not fields[4]:
        return 0
    plan = cmd.ff_plan and all(math.isfinite(v) for v in (cmd.plan_rate, cmd.plan_accel))
    return (SERVO_DAMP_FIRST if cmd.damping_first else 0) | (SERVO_FF_PLAN if plan else 0)


class RumbleChannel:
    """Writer for the command block. Absent plugin, non-Windows or any failure reads as ``PluginStatus()``."""

    def __init__(self) -> None:
        self._buf: mmap.mmap | None = None
        self._size = SIZE
        self._next_open = 0.0
        self._heartbeat = 0
        self._seq = 0
        self._sent: tuple[int, int, int] | None = None
        self._sent_torque: int | None = None
        self._sent_servo: tuple | None = None
        self._stamp = 0

    @property
    def is_open(self) -> bool:
        return self._buf is not None

    def _open(self, now: float) -> bool:
        if self._buf is not None:
            return True
        if sys.platform != "win32" or now < self._next_open:
            return False
        self._next_open = now + _RETRY_S
        for size in (SIZE, V4_SIZE, SERVO3_SIZE, LEGACY_SIZE):
            try:
                self._buf = mmap.mmap(0, size, TAG)
            except Exception:
                logger.debug("FFB command block unavailable at %d bytes", size, exc_info=True)
                continue
            self._size = size
            return True
        return False

    def publish(
        self, active: bool, side: int, units: int, now: float | None = None, torque: float = 0,
        servo: ServoCommand = SERVO_OFF,
    ) -> PluginStatus:
        """Send this tick's request. Nothing is written until the plugin has stamped its layout version.

        Rumble, torque and servo are written only when they change, so a second writer (the test tools)
        is not cancelled by this one repeating "nothing"; the heartbeat is bumped every call.
        """
        now = time.monotonic() if now is None else now
        if not self._open(now):
            return PluginStatus()
        try:
            status = decode_status(bytes(self._buf[:self._size]))
            if not status.present:
                self._sent = self._sent_torque = self._sent_servo = None
                return status
            if status.version >= TORQUE_VERSION and torque_units(torque) != self._sent_torque:
                self._sent_torque = torque_units(torque)
                _I16.pack_into(self._buf, _OFF_TORQUE, self._sent_torque)
            fields = servo_fields(servo)
            wanted = (fields, servo_flags(servo, fields), servo_ahead(servo, fields), servo_plan(servo, fields))
            if status.version >= VERSION and wanted != self._sent_servo:
                self._sent_servo = wanted
                _U8.pack_into(self._buf, _OFF_SERVO_FLAGS, wanted[1])
                if status.version >= AHEAD_VERSION:
                    _AHEAD.pack_into(self._buf, _OFF_AHEAD, *wanted[2])
                    _PLAN.pack_into(self._buf, _OFF_PLAN, *wanted[3])
                _SERVO4.pack_into(self._buf, _OFF_SERVO4, *fields[5:8])
                _SERVO.pack_into(self._buf, _OFF_SERVO, *fields[:5], fields[8])
                self._stamp = (self._stamp + 1) & 0xFFFFFFFF
                _U32.pack_into(self._buf, _OFF_STAMP, self._stamp)
            want = (CMD_RUMBLE, side, units) if active else (CMD_NONE, 0, 0)
            if want != self._sent:
                if active and (self._sent is None or self._sent[0] != CMD_RUMBLE):
                    self._seq = (self._seq + 1) & 0xFFFFFFFF
                    _U32.pack_into(self._buf, _OFF_SEQ, self._seq)
                _CMD.pack_into(self._buf, _OFF_CMD, *want)
                self._sent = want
            self._heartbeat = (self._heartbeat + 1) & 0xFFFFFFFF or 1
            _U32.pack_into(self._buf, _OFF_HEARTBEAT, self._heartbeat)
            return status
        except Exception:
            logger.debug("FFB command write failed", exc_info=True)
            self.close()
            return PluginStatus()

    def stop(self) -> None:
        """Ask for silence, no torque and no servo once; used when the feature is switched off or the thread ends."""
        if self._buf is not None:
            self.publish(False, 0, 0, torque=0, servo=SERVO_OFF)

    def close(self) -> None:
        if self._buf is not None:
            try:
                self._buf.close()
            except Exception:
                logger.debug("FFB command block close failed", exc_info=True)
        self._buf = None
        self._sent = self._sent_torque = self._sent_servo = None
