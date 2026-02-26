from __future__ import annotations

import logging


logger = logging.getLogger("monocruise.ui.popup")


class PopupManager:
    """
    Plain, non-threaded popup façade.

    UiThread owns a single instance of this class and calls its methods
    in response to events.  You can later adapt this to drive the
    existing core.popup.PopupWindow implementation.
    """

    def show_info(self, title: str, message: str) -> None:
        logger.info("[POPUP][INFO] %s: %s", title, message)

    def show_notice(self, title: str, message: str) -> None:
        logger.info("[POPUP][NOTICE] %s: %s", title, message)

    def show_warning(self, title: str, message: str) -> None:
        logger.warning("[POPUP][WARN] %s: %s", title, message)

    def show_error(self, title: str, message: str) -> None:
        logger.error("[POPUP][ERROR] %s: %s", title, message)

    def shutdown(self) -> None:
        logger.info("PopupManager shutdown")


__all__ = ["PopupManager"]

