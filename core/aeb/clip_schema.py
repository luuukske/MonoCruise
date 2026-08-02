"""AEB clip schema: radar + AEB streams, metadata, JSON; raw buffers base64 at serialize only."""

from __future__ import annotations

import base64
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone

# Bump when the on-disk shape changes; the server rejects unknown versions.
# v2 adds EgoTelemetry.estimated_total_mass_kg (None on v1 clips).
SCHEMA_VERSION: int = 2

# Sentinel used by the AEB thread for "no threat this tick".
_INF: float = 1e9


def new_clip_id() -> str:
    return str(uuid.uuid4())


def utc_now_iso() -> str:
    """ISO-8601 UTC timestamp with a trailing Z, e.g. ``2026-06-17T12:00:00Z``."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _client_version() -> str:
    """Read the running client version lazily to keep this module import-light."""
    try:
        from core.version import __version__

        return str(__version__)
    except Exception:
        return "unknown"


def _b64(buf: bytes | None) -> str | None:
    return base64.b64encode(buf).decode("ascii") if buf is not None else None


def _unb64(s: str | None) -> bytes | None:
    return base64.b64decode(s) if s is not None else None


@dataclass
class EgoTelemetry:
    """Raw ego telemetry as read by ``RadarThread._read_ego`` (see radar README.md)."""

    coordinateX: float = 0.0
    coordinateY: float = 0.0
    coordinateZ: float = 0.0
    rotationX: float = 0.0        # yaw, normalised 0..1
    rotationY: float = 0.0        # pitch, raw (inverted downstream)
    speed: float = 0.0            # m/s
    userSteer: float = 0.0
    paused: bool = False
    ego_has_trailer: bool = False
    # None means the clip predates schema v2, not a massless truck.
    estimated_total_mass_kg: float | None = None

    def to_json(self) -> dict:
        return {
            "coordinateX": self.coordinateX,
            "coordinateY": self.coordinateY,
            "coordinateZ": self.coordinateZ,
            "rotationX": self.rotationX,
            "rotationY": self.rotationY,
            "speed": self.speed,
            "userSteer": self.userSteer,
            "paused": self.paused,
            "ego_has_trailer": self.ego_has_trailer,
            "estimated_total_mass_kg": self.estimated_total_mass_kg,
        }

    @classmethod
    def from_json(cls, d: dict) -> "EgoTelemetry":
        return cls(
            coordinateX=float(d.get("coordinateX", 0.0)),
            coordinateY=float(d.get("coordinateY", 0.0)),
            coordinateZ=float(d.get("coordinateZ", 0.0)),
            rotationX=float(d.get("rotationX", 0.0)),
            rotationY=float(d.get("rotationY", 0.0)),
            speed=float(d.get("speed", 0.0)),
            userSteer=float(d.get("userSteer", 0.0)),
            paused=bool(d.get("paused", False)),
            ego_has_trailer=bool(d.get("ego_has_trailer", False)),
            estimated_total_mass_kg=_opt_float(d.get("estimated_total_mass_kg")),
        )


@dataclass
class RadarFrameRecord:
    """One radar frame: traffic/parked bytes + ego telemetry (base64 at JSON only)."""

    t_wall: float                  # time.time() at frame: smoothing clock
    t_mono: float                  # time.monotonic() at frame: snapshot clock
    ego: EgoTelemetry = field(default_factory=EgoTelemetry)
    traffic_buf: bytes | None = None
    parked_buf: bytes | None = None

    def to_json(self) -> dict:
        return {
            "t_wall": self.t_wall,
            "t_mono": self.t_mono,
            "ego": self.ego.to_json(),
            "traffic_buf": _b64(self.traffic_buf),
            "parked_buf": _b64(self.parked_buf),
        }

    @classmethod
    def from_json(cls, d: dict) -> "RadarFrameRecord":
        return cls(
            t_wall=float(d.get("t_wall", 0.0)),
            t_mono=float(d.get("t_mono", 0.0)),
            ego=EgoTelemetry.from_json(d.get("ego", {})),
            traffic_buf=_unb64(d.get("traffic_buf")),
            parked_buf=_unb64(d.get("parked_buf")),
        )


@dataclass
class ConsumedContext:
    """Decision-relevant context the AEB tick actually read (see plan 4.2)."""

    max_brake_ms2: float = 7.8     # sending_thread capacity estimate
    brakeval: float = 0.0          # main_pedal_thread user brake
    gasval: float = 0.0            # gas override gate
    opdbrakeval: float = 0.0       # OPD gas-lift brake demand, pre-AEB-override
    program_brake: float = 0.0     # mapper_command_brake (CC/ACC/limiter), AEB-free
    aeb_enabled: bool = True

    def to_json(self) -> dict:
        return {
            "max_brake_ms2": self.max_brake_ms2,
            "brakeval": self.brakeval,
            "gasval": self.gasval,
            "opdbrakeval": self.opdbrakeval,
            "program_brake": self.program_brake,
            "aeb_enabled": self.aeb_enabled,
        }

    @classmethod
    def from_json(cls, d: dict) -> "ConsumedContext":
        return cls(
            max_brake_ms2=float(d.get("max_brake_ms2", 7.8)),
            brakeval=float(d.get("brakeval", 0.0)),
            gasval=float(d.get("gasval", 0.0)),
            opdbrakeval=float(d.get("opdbrakeval", 0.0)),
            program_brake=float(d.get("program_brake", 0.0)),
            aeb_enabled=bool(d.get("aeb_enabled", True)),
        )


@dataclass
class LiveAEB:
    """Recorded live decision for parity replay; id sets localize filter divergences."""

    aeb_warn: bool = False
    aeb_brake: bool = False
    engaged: bool = False
    target_decel_ms2: float = 0.0
    ff_decel_ms2: float = 0.0
    required_decel_ms2: float = 0.0
    effective_max_decel_ms2: float = 7.8
    time_to_brake: float = _INF
    time_to_collision: float = _INF
    colliding_ids: list[int] = field(default_factory=list)
    suppressed_ids: list[int] = field(default_factory=list)
    braking_worsens_ids: list[int] = field(default_factory=list)
    suppression_reasons: dict[str, list[str]] = field(default_factory=dict)

    def to_json(self) -> dict:
        return {
            "aeb_warn": self.aeb_warn,
            "aeb_brake": self.aeb_brake,
            "engaged": self.engaged,
            "target_decel_ms2": self.target_decel_ms2,
            "ff_decel_ms2": self.ff_decel_ms2,
            "required_decel_ms2": self.required_decel_ms2,
            "effective_max_decel_ms2": self.effective_max_decel_ms2,
            "time_to_brake": self.time_to_brake,
            "time_to_collision": self.time_to_collision,
            "colliding_ids": list(self.colliding_ids),
            "suppressed_ids": list(self.suppressed_ids),
            "braking_worsens_ids": list(self.braking_worsens_ids),
            "suppression_reasons": {
                str(k): list(v) for k, v in self.suppression_reasons.items()
            },
        }

    @classmethod
    def from_json(cls, d: dict) -> "LiveAEB":
        return cls(
            aeb_warn=bool(d.get("aeb_warn", False)),
            aeb_brake=bool(d.get("aeb_brake", False)),
            engaged=bool(d.get("engaged", False)),
            target_decel_ms2=float(d.get("target_decel_ms2", 0.0)),
            ff_decel_ms2=float(d.get("ff_decel_ms2", 0.0)),
            required_decel_ms2=float(d.get("required_decel_ms2", 0.0)),
            effective_max_decel_ms2=float(d.get("effective_max_decel_ms2", 7.8)),
            time_to_brake=float(d.get("time_to_brake", _INF)),
            time_to_collision=float(d.get("time_to_collision", _INF)),
            colliding_ids=[int(x) for x in d.get("colliding_ids", [])],
            suppressed_ids=[int(x) for x in d.get("suppressed_ids", [])],
            braking_worsens_ids=[int(x) for x in d.get("braking_worsens_ids", [])],
            suppression_reasons={
                str(k): [str(s) for s in v]
                for k, v in d.get("suppression_reasons", {}).items()
            },
        )


@dataclass
class AEBTickRecord:
    """One AEB-thread tick: consumed context + the live decision it produced."""

    t_mono: float                  # AEB thread clock at this tick
    radar_t_mono: float            # t_mono of the radar snapshot consumed
    consumed: ConsumedContext = field(default_factory=ConsumedContext)
    live_aeb: LiveAEB = field(default_factory=LiveAEB)

    def to_json(self) -> dict:
        return {
            "t_mono": self.t_mono,
            "radar_t_mono": self.radar_t_mono,
            "consumed": self.consumed.to_json(),
            "live_aeb": self.live_aeb.to_json(),
        }

    @classmethod
    def from_json(cls, d: dict) -> "AEBTickRecord":
        return cls(
            t_mono=float(d.get("t_mono", 0.0)),
            radar_t_mono=float(d.get("radar_t_mono", 0.0)),
            consumed=ConsumedContext.from_json(d.get("consumed", {})),
            live_aeb=LiveAEB.from_json(d.get("live_aeb", {})),
        )


@dataclass
class AEBWarmState:
    """Discrete AEB state applied before first replay tick (see clip_eval warm start)."""

    engaged: bool = False
    latched_threat_ids: list[int] = field(default_factory=list)
    latched_scope_ok_mono: dict[int, float] = field(default_factory=dict)
    latched_filter_ego_kmh: float | None = None
    warn_hold_until_mono: float | None = None
    brake_hold_until_mono: float | None = None
    target_decel_ms2: float = 0.0

    def to_json(self) -> dict:
        return {
            "engaged": self.engaged,
            "latched_threat_ids": list(self.latched_threat_ids),
            "latched_scope_ok_mono": {
                str(k): v for k, v in self.latched_scope_ok_mono.items()
            },
            "latched_filter_ego_kmh": self.latched_filter_ego_kmh,
            "warn_hold_until_mono": self.warn_hold_until_mono,
            "brake_hold_until_mono": self.brake_hold_until_mono,
            "target_decel_ms2": self.target_decel_ms2,
        }

    @classmethod
    def from_json(cls, d: dict) -> "AEBWarmState":
        return cls(
            engaged=bool(d.get("engaged", False)),
            latched_threat_ids=[int(x) for x in d.get("latched_threat_ids", [])],
            latched_scope_ok_mono={
                int(k): float(v)
                for k, v in d.get("latched_scope_ok_mono", {}).items()
            },
            latched_filter_ego_kmh=_opt_float(d.get("latched_filter_ego_kmh")),
            warn_hold_until_mono=_opt_float(d.get("warn_hold_until_mono")),
            brake_hold_until_mono=_opt_float(d.get("brake_hold_until_mono")),
            target_decel_ms2=float(d.get("target_decel_ms2", 0.0)),
        )


@dataclass
class Label:
    """Assigned offline in the labelling tool, never by the tester (plan 4.4)."""

    class_: str = "ignore"          # fp | fn | tp | good_intervention | ignore
    severity: int = 0               # 1..5, human-assigned
    should_trigger: dict | None = None   # {"from_t":.., "to_t":..}; None = must NOT trigger
    target_vid: int | None = None
    desired_peak_decel_ms2: float | None = None
    notes: str = ""

    def to_json(self) -> dict:
        return {
            "class": self.class_,
            "severity": self.severity,
            "should_trigger": self.should_trigger,
            "target_vid": self.target_vid,
            "desired_peak_decel_ms2": self.desired_peak_decel_ms2,
            "notes": self.notes,
        }

    @classmethod
    def from_json(cls, d: dict) -> "Label":
        return cls(
            class_=str(d.get("class", "ignore")),
            severity=int(d.get("severity", 0)),
            should_trigger=d.get("should_trigger"),
            target_vid=_opt_int(d.get("target_vid")),
            desired_peak_decel_ms2=_opt_float(d.get("desired_peak_decel_ms2")),
            notes=str(d.get("notes", "")),
        )


@dataclass
class ClipMetadata:
    """Clip-level metadata (plan 4.3). ``label`` stays None at capture time."""

    schema_version: int = SCHEMA_VERSION
    clip_id: str = ""
    client_version: str = ""
    trigger_source: str = "manual"
    also_triggered: list[str] = field(default_factory=list)
    brake_reached: bool = False
    captured_at: str = ""
    session_kind: str = "SP"          # or "TMP"
    session_kind_changed: bool = False
    calibration_id: str = ""
    calibration: dict = field(default_factory=dict)
    pre_s: float = 8.0
    post_s: float = 3.0
    burn_in_s: float = 2.0
    frame_count: int = 0
    tick_count: int = 0
    aeb_warm_state: AEBWarmState = field(default_factory=AEBWarmState)
    client_id: str | None = None      # attached at submission, not capture
    # Downscaled JPEG (base64) of the game screen at the trigger moment, for
    # human context while tagging. Embedded in the clip so it travels on share.
    thumbnail_jpeg: str | None = None
    label: Label | None = None

    @classmethod
    def create(cls, **overrides) -> "ClipMetadata":
        """Build metadata, filling clip_id / captured_at / client_version."""
        base = dict(
            schema_version=SCHEMA_VERSION,
            clip_id=new_clip_id(),
            client_version=_client_version(),
            captured_at=utc_now_iso(),
        )
        base.update(overrides)
        return cls(**base)

    def to_json(self) -> dict:
        return {
            "schema_version": self.schema_version,
            "clip_id": self.clip_id,
            "client_version": self.client_version,
            "trigger_source": self.trigger_source,
            "also_triggered": list(self.also_triggered),
            "brake_reached": self.brake_reached,
            "captured_at": self.captured_at,
            "session_kind": self.session_kind,
            "session_kind_changed": self.session_kind_changed,
            "calibration_id": self.calibration_id,
            "calibration": dict(self.calibration),
            "pre_s": self.pre_s,
            "post_s": self.post_s,
            "burn_in_s": self.burn_in_s,
            "frame_count": self.frame_count,
            "tick_count": self.tick_count,
            "aeb_warm_state": self.aeb_warm_state.to_json(),
            "client_id": self.client_id,
            "thumbnail_jpeg": self.thumbnail_jpeg,
            "label": self.label.to_json() if self.label is not None else None,
        }

    @classmethod
    def from_json(cls, d: dict) -> "ClipMetadata":
        label_raw = d.get("label")
        return cls(
            schema_version=int(d.get("schema_version", SCHEMA_VERSION)),
            clip_id=str(d.get("clip_id", "")),
            client_version=str(d.get("client_version", "")),
            trigger_source=str(d.get("trigger_source", "manual")),
            also_triggered=[str(x) for x in d.get("also_triggered", [])],
            brake_reached=bool(d.get("brake_reached", False)),
            captured_at=str(d.get("captured_at", "")),
            session_kind=str(d.get("session_kind", "SP")),
            session_kind_changed=bool(d.get("session_kind_changed", False)),
            calibration_id=str(d.get("calibration_id", "")),
            calibration=dict(d.get("calibration", {})),
            pre_s=float(d.get("pre_s", 8.0)),
            post_s=float(d.get("post_s", 3.0)),
            burn_in_s=float(d.get("burn_in_s", 2.0)),
            frame_count=int(d.get("frame_count", 0)),
            tick_count=int(d.get("tick_count", 0)),
            aeb_warm_state=AEBWarmState.from_json(d.get("aeb_warm_state", {})),
            client_id=d.get("client_id"),
            thumbnail_jpeg=d.get("thumbnail_jpeg"),
            label=Label.from_json(label_raw) if label_raw else None,
        )


@dataclass
class Clip:
    """Full capture: metadata plus radar_frames and aeb_ticks arrays in JSON."""

    metadata: ClipMetadata = field(default_factory=ClipMetadata)
    radar_frames: list[RadarFrameRecord] = field(default_factory=list)
    aeb_ticks: list[AEBTickRecord] = field(default_factory=list)

    def to_json_dict(self) -> dict:
        d = self.metadata.to_json()
        d["radar_frames"] = [f.to_json() for f in self.radar_frames]
        d["aeb_ticks"] = [t.to_json() for t in self.aeb_ticks]
        return d

    @classmethod
    def from_json_dict(cls, d: dict) -> "Clip":
        return cls(
            metadata=ClipMetadata.from_json(d),
            radar_frames=[RadarFrameRecord.from_json(x) for x in d.get("radar_frames", [])],
            aeb_ticks=[AEBTickRecord.from_json(x) for x in d.get("aeb_ticks", [])],
        )


def _opt_float(v: object) -> float | None:
    return None if v is None else float(v)


def _opt_int(v: object) -> int | None:
    return None if v is None else int(v)
