from __future__ import annotations

import logging
from dataclasses import dataclass

from base_thread import BaseThread, SnapshotMixin
from registry import Event, Registry
from cruise_thread import CruiseData, CruiseThread
from updater_thread import UpdaterData, UpdaterThread
from ui.hud import HUD
from ui.popup import PopupManager


logger = logging.getLogger("monocruise.ui")


@dataclass
class UiData(SnapshotMixin):
    """
    Lightweight snapshot of what the UI is currently showing.

    The actual rendering objects live in ui.hud and ui.popup; this
    dataclass only exposes high-level state to other threads.
    """

    cruise_engaged: bool = False
    cruise_target_speed_kmh: float = 0.0
    last_popup_message: str | None = None


class UiThread(BaseThread[UiData]):
    """
    Owns HUD and popup queue; reacts to published events.

    Nothing else touches rendering or the HUD/popup classes directly.
    """

    def __init__(
        self,
        registry: Registry,
        *,
        name: str = "ui",
        loop_interval: float = 0.05,
    ) -> None:
        self._registry = registry
        self._hud: HUD | None = None
        self._popup_mgr: PopupManager | None = None
        super().__init__(name=name, loop_interval=loop_interval)

    def create_initial_data(self) -> UiData:
        return UiData()

    def on_startup(self) -> None:
        logger.info("UiThread starting")
        self._hud = HUD()
        self._popup_mgr = PopupManager()

        # Subscribe to events of interest.
        self._registry.subscribe("cruise:engaged", self.name)
        self._registry.subscribe("cruise:disengaged", self.name)
        self._registry.subscribe("updater:update_available", self.name)

    def loop(self) -> None:
        # Drain any queued events for this UI thread.
        events = self._registry.poll_events(self.name, max_events=20)
        for ev in events:
            self._handle_event(ev)

        # In a real implementation this is where you would also pump a
        # GUI event loop or repaint hints if needed.

    def on_shutdown(self) -> None:
        logger.info("UiThread stopping")
        self._registry.unsubscribe("cruise:engaged", self.name)
        self._registry.unsubscribe("cruise:disengaged", self.name)
        self._registry.unsubscribe("updater:update_available", self.name)

        if self._hud is not None:
            self._hud.shutdown()
            self._hud = None
        if self._popup_mgr is not None:
            self._popup_mgr.shutdown()
            self._popup_mgr = None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _handle_event(self, event: Event) -> None:
        if event.name == "cruise:engaged":
            self._on_cruise_engaged(event)
        elif event.name == "cruise:disengaged":
            self._on_cruise_disengaged(event)
        elif event.name == "updater:update_available":
            self._on_update_available(event)

    def _on_cruise_engaged(self, event: Event) -> None:
        target = float(event.payload.get("target_speed_kmh", 0.0)) if isinstance(event.payload, dict) else 0.0

        # Read full cruise state if available.
        cruise = self._registry.get_thread("cruise")
        if isinstance(cruise, CruiseThread):
            cruise_data: CruiseData = cruise.data
            target = cruise_data.target_speed_kmh or target

        self.data.update(cruise_engaged=True, cruise_target_speed_kmh=target)

        if self._hud is not None:
            self._hud.set_cruise_state(True, target)
        if self._popup_mgr is not None:
            msg = f"Cruise engaged at {target:.0f} km/h"
            self.data.update(last_popup_message=msg)
            self._popup_mgr.show_info("Cruise", msg)

    def _on_cruise_disengaged(self, event: Event) -> None:
        self.data.update(cruise_engaged=False)

        if self._hud is not None:
            self._hud.set_cruise_state(False, self.data.cruise_target_speed_kmh)
        if self._popup_mgr is not None:
            msg = "Cruise disengaged"
            self.data.update(last_popup_message=msg)
            self._popup_mgr.show_notice("Cruise", msg)

    def _on_update_available(self, event: Event) -> None:
        version = None
        if isinstance(event.payload, dict):
            version = event.payload.get("version")

        # Optionally read the full updater data for richer messaging.
        updater = self._registry.get_thread("updater")
        if isinstance(updater, UpdaterThread):
            udata: UpdaterData = updater.data
            version = udata.latest_version or version

        if version is None:
            version = "unknown"

        msg = f"Update available: {version}"
        self.data.update(last_popup_message=msg)

        if self._popup_mgr is not None:
            self._popup_mgr.show_info("Update", msg)


__all__ = ["UiThread", "UiData"]

