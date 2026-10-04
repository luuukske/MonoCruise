"""Bounded, telemetry-confirmed hazard-toggle experiment."""
from dataclasses import dataclass


@dataclass
class HazardFlash:
    end: float
    restore: bool
    target: bool
    changed: float
    confirmed_at: float | None = None
    active: bool = True

    @classmethod
    def begin(cls, now: float, restore: bool, current: bool):
        return cls(now + 5.0, restore, not current, now)

    def step(self, now: float, current: bool, idle: bool, enabled: bool) -> bool | None:
        if not self.active:
            return None
        if not enabled or now >= self.end or now - self.changed >= 1.0:
            self.active = False
            return self.restore
        if current != self.target or not idle:
            return None
        if self.confirmed_at is None:
            self.confirmed_at = now
        if now - self.confirmed_at < 0.10:
            return None
        self.target = not self.target
        self.changed = now
        self.confirmed_at = None
        return self.target
