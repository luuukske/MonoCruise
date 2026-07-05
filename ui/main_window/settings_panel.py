"""
MonoCruise – Settings panel (left‑side drawer).

All five sections:  Inputs · Program Settings · Cruise Control ·
One‑Pedal‑Drive · Footer/Credits.

Reads and writes the shared ``core.settings.Settings`` instance directly.
"""

from __future__ import annotations

import logging
import os
import re
import subprocess
import sys
import webbrowser
from typing import TYPE_CHECKING, Any, Callable

from PySide6.QtCore import QSize, Qt, QTimer
from PySide6.QtGui import QIcon, QPixmap
from PySide6.QtWidgets import (
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from core.input_bindings import binding_display_name, migrate_binding, resolve_held
from core.thread_management.registry import registry
from ui.main_window.constants import (
    FIELD_ROW_HEIGHT,
    RADIUS_SETTINGS_PANEL,
    SETTINGS_PANEL_WIDTH,
    SUBTEXT_GAP_TOP,
)
from ui.main_window.widgets import (
    BindButton,
    CheckBox,
    new_beta_pill,
    new_checkbutton,
    new_entry,
    new_label,
    new_optionmenu,
    new_section_header,
    new_spacer,
    new_subtext,
)
from ui.popup.popup_window import PopupWindow

if TYPE_CHECKING:
    from core.settings import Settings
    from ui.main_window.window import MonoCruiseWindow

logger = logging.getLogger(__name__)

# Settings fields that hold input bindings. The first three have configure
# buttons in the panel; the ACC gap fields are included so duplicate-steal
# checks cover every binding.
_BIND_KEYS = (
    "cc_start_button", "cc_inc_button", "cc_dec_button",
    "acc_dist_inc_button", "acc_dist_dec_button",
)

_BIND_BUTTON_SIZE = (150, 30)
_BIND_POLL_MS = 33.333

# Extra breathing room under the Unassign button, before the next setting.
_UNASSIGN_GAP = 10

# Resolve project‑root path for assets (gear.png, patreon.png, youtube.png
# live at the project root in the original repo).
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class SettingsPanel(QWidget):
    """Slide‑in settings drawer."""

    def __init__(
        self,
        parent: "MonoCruiseWindow",
        settings: "Settings",
        *,
        on_save: Callable[[], None],
        on_reset: Callable[[], None],
        show_confirm: Callable[..., Any],
    ) -> None:
        super().__init__(parent)
        self._settings = settings
        self._on_save = on_save
        self._show_confirm = show_confirm
        self._on_reset = on_reset
        self._reset_armed = False
        # Row min-heights stashed while a conditional row is hidden (see
        # _set_row_visible).
        self._hidden_row_heights: dict[int, int] = {}

        # Button-binding capture state
        self._bind_buttons: dict[str, BindButton] = {}
        self._configuring_key: str | None = None
        self._kb_capture_started = False
        self._unassign_armed = False
        # Keys whose freshly captured input is still held: blue "pressed"
        # highlight stays off until the input is released once.
        self._glow_suppress: set[str] = set()

        self.setMaximumWidth(SETTINGS_PANEL_WIDTH)
        self.setMinimumWidth(0)
        self.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Expanding)
        # Scope to this widget only. A selector-less stylesheet acts as
        # `* { ... }` for the whole subtree and would override every
        # background rule from the window stylesheet (buttons, inputs,
        # section headers) inside the panel.
        self.setObjectName("settingsPanel")
        self.setStyleSheet("QWidget#settingsPanel { background-color: transparent; }")

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        # Card: single rounded container for title + scroll + bar
        card = QWidget()
        card.setObjectName("settingsCard")
        card.setStyleSheet(
            f"QWidget#settingsCard {{ background-color: #333333; "
            f"border-radius: {RADIUS_SETTINGS_PANEL}px; }}"
        )
        card_lay = QVBoxLayout(card)
        card_lay.setContentsMargins(6, 6, 6, 6)
        card_lay.setSpacing(4)
        root.addWidget(card, 1)

        # Title row: "Settings" centred across the full container, with an
        # update button anchored near the right edge that opens the standalone
        # updater exe.
        header = QHBoxLayout()
        header.setContentsMargins(0, 0, 0, 0)
        header.setSpacing(0)

        # Parented now so it inherits STYLESHEET before its styled height is
        # read below (an unparented label measures at its unstyled height).
        title = QLabel("Settings", self)
        title.setObjectName("settingsTitle")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self._btn_update = QPushButton()
        self._btn_update.setObjectName("updateButton")
        self._btn_update.setCursor(Qt.CursorShape.PointingHandCursor)
        self._btn_update.setToolTip("Check for updates")
        # Manual icon+label layout: QPushButton's icon/text gap isn't QSS-
        # settable. Labels ignore mouse events so clicks reach the button.
        btn_lay = QHBoxLayout(self._btn_update)
        btn_lay.setContentsMargins(8, 5, 8, 5)
        btn_lay.setSpacing(4)
        icon_lbl = QLabel()
        icon_lbl.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        update_icon = os.path.join(_PROJECT_ROOT, "ui/main_window/assets/cloud-download.svg")
        if os.path.exists(update_icon):
            icon_lbl.setPixmap(QIcon(update_icon).pixmap(QSize(16, 16)))
        text_lbl = QLabel("Update")
        text_lbl.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        text_lbl.setStyleSheet("color: white; background: transparent; font-size: 12px;")
        btn_lay.addWidget(icon_lbl)
        btn_lay.addWidget(text_lbl)
        self._btn_update.clicked.connect(self._launch_updater)

        # A QPushButton ignores its child layout when sizing, so pin it.
        title.ensurePolished()
        text_lbl.ensurePolished()
        self._btn_update.ensurePolished()
        self._btn_update.setFixedSize(btn_lay.sizeHint())

        # Right margin = whitespace above the button; a left spacer mirrors
        # button width + margin so "Settings" stays centred in the card.
        right_margin = max(
            0,
            (title.sizeHint().height() - self._btn_update.height()) // 2,
        )
        left_spacer = QWidget()
        left_spacer.setStyleSheet("background: transparent;")
        left_spacer.setFixedWidth(self._btn_update.width() + right_margin)

        header.addWidget(left_spacer)
        header.addStretch(1)
        header.addWidget(title)
        header.addStretch(1)
        header.addWidget(self._btn_update)
        header.addSpacing(right_margin)
        card_lay.addLayout(header)

        # Scroll area
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        # No separate border/radius: the card is the visual container
        scroll.setStyleSheet(
            "QScrollArea { border: none; border-radius: 0px; background-color: transparent; }"
        )
        # Smooth scrolling (replicates yscrollincrement=20)
        scroll.verticalScrollBar().setSingleStep(20)

        inner = QWidget()
        inner.setObjectName("settingsInner")
        # Scoped for the same reason as the panel stylesheet above.
        inner.setStyleSheet("QWidget#settingsInner { background-color: transparent; }")
        self._grid = QGridLayout(inner)
        self._grid.setContentsMargins(10, 8, 10, 10)
        # Small vertical spacing for rows—so subtext stays visually tied to its setting
        self._grid.setHorizontalSpacing(4)
        self._grid.setVerticalSpacing(4)
        self._grid.setColumnStretch(0, 1)
        self._grid.setColumnMinimumWidth(1, 120)
        scroll.setWidget(inner)
        card_lay.addWidget(scroll, 1)

        self._row = 0
        self._inner = inner

        # Build each section
        self._build_inputs()
        self._build_program_settings()
        self._build_cruise_control()
        self._build_one_pedal_drive()
        self._build_footer()

        # Bottom button bar (Patreon · YouTube · Hide X)
        bar = QHBoxLayout()
        bar.setContentsMargins(2, 2, 2, 2)
        bar.setSpacing(5)

        self._btn_patreon = QPushButton("  Patreon")
        self._btn_patreon.setObjectName("supportButton")
        patreon_path = os.path.join(_PROJECT_ROOT, "ui/main_window/assets/patreon.png")
        if os.path.exists(patreon_path):
            self._btn_patreon.setIcon(QIcon(QPixmap(patreon_path)))
        self._btn_patreon.clicked.connect(
            lambda: webbrowser.open("https://www.patreon.com/")
        )
        bar.addWidget(self._btn_patreon)

        self._btn_youtube = QPushButton("  YouTube")
        self._btn_youtube.setObjectName("supportButton")
        youtube_path = os.path.join(_PROJECT_ROOT, "ui/main_window/assets/youtube.png")
        if os.path.exists(youtube_path):
            self._btn_youtube.setIcon(QIcon(QPixmap(youtube_path)))
        self._btn_youtube.clicked.connect(
            lambda: webbrowser.open("https://www.youtube.com/@ld-tech_org")
        )
        bar.addWidget(self._btn_youtube)

        bar.addStretch()

        self._hide_btn = QPushButton("X")
        self._hide_btn.setObjectName("hideButton")
        self._hide_btn.clicked.connect(self._on_hide_links)
        bar.addWidget(self._hide_btn)

        card_lay.addLayout(bar)

        # Restore hidden state from settings
        if settings.hide_button_action:
            self._btn_patreon.hide()
            self._btn_youtube.hide()
            self._hide_btn.hide()

        # Poll capture results and pressed-state highlights.
        self._bind_timer = QTimer(self)
        self._bind_timer.setInterval(_BIND_POLL_MS)
        self._bind_timer.timeout.connect(self._poll_bindings)
        self._bind_timer.start()

    # Row counter

    def _r(self, advance: int = 1) -> int:
        r = self._row
        self._row += advance
        return r

    def _spacer(self, height: int) -> None:
        """Insert an invisible full-width row to open up vertical breathing
        room beyond the grid's tight base spacing (see __init__)."""
        new_spacer(self._inner, self._r(), height)

    def _field_with_subtext(
        self,
        label_text: str,
        build_field: Callable[[QWidget, int, int], Any],
        subtext_text: str,
    ) -> tuple[Any, QLabel, int]:
        """One outer grid row containing a label+field pair with its
        description tucked tightly underneath, independent of the outer
        grid's row-to-row spacing -- which governs gaps to unrelated
        settings and is otherwise too wide for subtext to read as attached
        to the field above it. The gap above the subtext is SUBTEXT_GAP_TOP
        (below); the gap below it, before the next setting, is
        SUBTEXT_GAP_BOTTOM (QLabel#subtext's margin-bottom in the
        stylesheet) -- tune each independently.

        Returns (field, subtext_label, row). Most callers only need
        *field*; *subtext_label* is for fields whose subtext toggles
        on its own (e.g. update channel), and *row* for fields whose whole
        row toggles together (see _set_row_visible).
        """
        container = QWidget()
        # Scoped ID selector -- a selector-less stylesheet acts as `* { ... }`
        # for the whole subtree and would override the window stylesheet's
        # `QLineEdit`/Dropdown background rules for the field nested inside
        # (see the identical note on settingsPanel's stylesheet above).
        container.setObjectName("fieldWithSubtext")
        container.setStyleSheet("QWidget#fieldWithSubtext { background: transparent; }")
        inner = QGridLayout(container)
        inner.setContentsMargins(0, 0, 0, 0)
        inner.setHorizontalSpacing(4)
        inner.setVerticalSpacing(SUBTEXT_GAP_TOP)
        inner.setColumnStretch(0, 1)
        inner.setColumnMinimumWidth(1, 120)

        new_label(container, 0, 0, label_text)
        field = build_field(container, 0, 1)
        subtext = new_subtext(container, 1, 0, subtext_text, col_span=2)

        row = self._r()
        self._grid.addWidget(container, row, 0, 1, 2)
        return field, subtext, row

    # Section 1 – Inputs

    def _build_inputs(self) -> None:
        s = self._settings
        p = self._inner

        new_section_header(p, self._r(), "Inputs")

        new_label(p, self._r(0), 0, "Connected pedals:")
        self.lbl_pedals = new_label(
            p, self._r(), 1, "None",
            alignment=Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter,
        )

        new_label(p, self._r(0), 0, "Gas axis:")
        self.lbl_gas = new_label(
            p, self._r(), 1, str(s.gasaxis) if s.gasaxis else "—",
            alignment=Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter,
        )

        new_label(p, self._r(0), 0, "Brake axis:")
        self.lbl_brake = new_label(
            p, self._r(), 1, str(s.brakeaxis) if s.brakeaxis else "—",
            alignment=Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter,
        )

        self.btn_connect = QPushButton("Connect to pedals")
        self.btn_connect.clicked.connect(self._on_connect_pedals)
        self._grid.addWidget(self.btn_connect, self._r(), 0, 1, 2)

        self.lbl_conn_error = QLabel("")
        self.lbl_conn_error.setObjectName("errorLabel")
        self._grid.addWidget(self.lbl_conn_error, self._r(), 0, 1, 2)

    def _on_connect_pedals(self) -> None:
        # TODO: trigger joystick connection via registry / JoystickThread
        pass

    # Section 2 – Program Settings

    def _build_program_settings(self) -> None:
        s = self._settings
        p = self._inner

        new_section_header(p, self._r(), "Program settings")

        new_label(p, self._r(0), 0, "Autostart MonoCruise:")
        self.chk_autostart = new_checkbutton(
            p, self._r(), 1, s.autostart_variable,
            callback=lambda v: self._set("autostart_variable", v),
        )

        self.ent_polling, _, _ = self._field_with_subtext(
            "Target polling rate (Hz):",
            lambda c, r, col: new_entry(
                c, r, col,
                value=s.polling_rate, value_type=int,
                minimum=10, maximum=100,
                callback=lambda v: self._set("polling_rate", v),
            ),
            "How often inputs are read per second (10–100).",
        )

        new_label(p, self._r(0), 0, "Hazards:")
        self.chk_hazards = new_checkbutton(
            p, self._r(), 1, s.hazards_variable,
            callback=self._on_hazards_toggled,
        )

        # Autodisable hazards (conditionally visible)
        r_auto = self._r()
        new_label(p, r_auto, 0, "  Autodisable hazards:")
        self.chk_autodisable = new_checkbutton(
            p, r_auto, 1, s.autodisable_hazards,
            callback=lambda v: self._set("autodisable_hazards", v),
        )
        self._hazard_auto_row = r_auto
        self._set_row_visible(r_auto, s.hazards_variable)

        new_label(p, self._r(0), 0, "Horn:")
        self.chk_horn = new_checkbutton(
            p, self._r(), 1, s.horn_variable,
            callback=lambda v: self._set("horn_variable", v),
        )

        new_label(p, self._r(0), 0, "Airhorn:")
        self.chk_airhorn = new_checkbutton(
            p, self._r(), 1, s.airhorn_variable,
            callback=lambda v: self._set("airhorn_variable", v),
        )

        new_label(p, self._r(0), 0, "Live bottom bar:")
        self.chk_live_bar = new_checkbutton(
            p, self._r(), 1, s.bar_variable,
            callback=lambda v: self._set("bar_variable", v),
        )

        self.opt_channel, self._preview_subtext, _ = self._field_with_subtext(
            "Update channel:",
            lambda c, r, col: new_optionmenu(
                c, r, col,
                values=["Stable", "Preview"],
                default=s.update_channel.capitalize(),
                callback=self._on_update_channel_changed,
            ),
            "Preview builds are released earlier and may contain bugs.",
        )
        # Only the subtext toggles here -- the dropdown itself always stays
        # visible, unlike the whole-row conditionals (see _set_row_visible).
        self._preview_subtext.setVisible(s.update_channel.lower() == "preview")

    def _on_update_channel_changed(self, value: str) -> None:
        self._set("update_channel", value.lower())
        self._preview_subtext.setVisible(value.lower() == "preview")

    def _on_hazards_toggled(self, checked: bool) -> None:
        self._set("hazards_variable", checked)
        self._set_row_visible(self._hazard_auto_row, checked)

    # Section 3 – Cruise Control

    def _build_cruise_control(self) -> None:
        s = self._settings
        p = self._inner

        self._spacer(8)
        new_section_header(p, self._r(), "Cruise Control")

        # Mode label row
        r_mode = self._r(0)
        new_label(p, r_mode, 0, "Mode:")

        # Segmented button pair
        seg_frame = QFrame()
        seg_frame.setObjectName("segFrame")
        seg_lay = QHBoxLayout(seg_frame)
        seg_lay.setContentsMargins(2, 2, 2, 2)
        seg_lay.setSpacing(2)

        self._seg_cc = QPushButton("Cruise control")
        self._seg_sl = QPushButton("Speed limiter")
        for btn, mode in ((self._seg_cc, "cc"), (self._seg_sl, "limiter")):
            btn.setObjectName("segButton")
            btn.setProperty("segMode", mode)
            btn.setFixedHeight(22)
        # 22px pills + 2px inset + 2px frame border = 30px, same as the
        # bind buttons.
        seg_frame.setFixedHeight(30)
        self._seg_cc.clicked.connect(lambda: self._set_cruise_mode("Cruise control"))
        self._seg_sl.clicked.connect(lambda: self._set_cruise_mode("Speed limiter"))
        seg_lay.addWidget(self._seg_cc)
        seg_lay.addWidget(self._seg_sl)
        self._grid.addWidget(
            seg_frame, r_mode, 0, 1, 2,
            Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter,
        )
        self._grid.setRowMinimumHeight(r_mode, FIELD_ROW_HEIGHT)
        self._row += 1  # advance past the shared label/segmented-control row
        self._update_seg_style(s.cc_mode)

        # Global speed limiter (empty → None disables both CC clamp and
        # always-on limiter — see AGENTS.md global_speed_limit_kmh).
        self.ent_global_limit, _, _ = self._field_with_subtext(
            "Global speed limiter:",
            lambda c, r, col: new_entry(
                c, r, col,
                value=s.global_speed_limit_kmh, value_type=int,
                minimum=60, maximum=130, optional=True,
                suffix="km/h",
                callback=lambda v: self._set("global_speed_limit_kmh", v),
            ),
            "Empty to disable.",
        )

        # Button configure rows
        new_label(p, self._r(0), 0, "Enable/Disable button:")
        self._add_bind_button(self._r(), "cc_start_button")

        new_label(p, self._r(0), 0, "Increase button:")
        self._add_bind_button(self._r(), "cc_inc_button")

        new_label(p, self._r(0), 0, "Decrease button:")
        self._add_bind_button(self._r(), "cc_dec_button")

        # Unassign button: arms unassign mode, or clears the binding
        # currently being configured.
        self._unassign_btn = QPushButton("Unassign")
        self._unassign_btn.setObjectName("unassignButton")
        self._unassign_btn.setFixedSize(*_BIND_BUTTON_SIZE)
        self._unassign_btn.setProperty("armed", False)
        self._unassign_btn.clicked.connect(self._on_unassign_clicked)
        self._grid.addWidget(
            self._unassign_btn, self._r(), 0, 1, 2,
            Qt.AlignmentFlag.AlignRight,
        )
        self._spacer(_UNASSIGN_GAP)

        # Increments
        new_label(p, self._r(0), 0, "Short press increments:")
        increment_values = self._increment_display_values()
        self.opt_short = new_optionmenu(
            p, self._r(), 1,
            values=increment_values,
            default=self._format_increment_value(s.short_increments),
            callback=lambda v: self._set("short_increments", self._parse_increment_value(v)),
        )

        new_label(p, self._r(0), 0, "Long press increments:")
        self.opt_long = new_optionmenu(
            p, self._r(), 1,
            values=increment_values,
            default=self._format_increment_value(s.long_increments),
            callback=lambda v: self._set("long_increments", self._parse_increment_value(v)),
        )

        # Checkboxes
        new_label(p, self._r(0), 0, "Hold enable to reset:")
        self.chk_hold_reset = new_checkbutton(
            p, self._r(), 1, s.long_press_reset,
            callback=lambda v: self._set("long_press_reset", v),
        )

        self.chk_show_speed, _, _ = self._field_with_subtext(
            "Show set speed on screen:",
            lambda c, r, col: new_checkbutton(
                c, r, col, s.show_cc_ui,
                callback=lambda v: self._set("show_cc_ui", v),
            ),
            "just drag it across the screen to move",
        )

        # CC UI scaling
        new_label(p, self._r(0), 0, "Cruise Control UI scaling:")
        self.opt_scaling = new_optionmenu(
            p, self._r(), 1,
            values=["25%", "50%", "75%", "100%", "150%", "200%"],
            default=str(s.cc_panel_scaling) if s.cc_panel_scaling else "100%",
            callback=lambda v: self._set("cc_panel_scaling", v),
        )

        # ACC (BETA)
        r_acc = self._r()
        new_label(p, r_acc, 0, "Adaptive Cruise Control:")
        acc_widget = QWidget()
        acc_widget.setObjectName("transparentRow")
        acc_widget.setStyleSheet("QWidget#transparentRow { background: transparent; }")
        acc_lay = QHBoxLayout(acc_widget)
        acc_lay.setContentsMargins(0, 0, 0, 0)
        acc_lay.setSpacing(4)
        acc_pill = new_beta_pill()
        acc_lay.addStretch()
        acc_lay.addWidget(acc_pill)
        self.chk_acc = CheckBox()
        self.chk_acc.setFixedSize(24, 24)
        self.chk_acc.setChecked(bool(s.acc_enabled))
        self.chk_acc.toggled.connect(self._on_acc_toggled)
        acc_lay.addWidget(self.chk_acc)
        self._grid.addWidget(acc_widget, r_acc, 1)
        self._grid.setRowMinimumHeight(r_acc, FIELD_ROW_HEIGHT)

        # AEB (BETA)
        r_aeb = self._r()
        new_label(p, r_aeb, 0, "Emergency Braking:")
        aeb_widget = QWidget()
        aeb_widget.setObjectName("transparentRow")
        aeb_widget.setStyleSheet("QWidget#transparentRow { background: transparent; }")
        aeb_lay = QHBoxLayout(aeb_widget)
        aeb_lay.setContentsMargins(0, 0, 0, 0)
        aeb_lay.setSpacing(4)
        aeb_pill = new_beta_pill()
        aeb_lay.addStretch()
        aeb_lay.addWidget(aeb_pill)
        self.chk_aeb = CheckBox()
        self.chk_aeb.setFixedSize(24, 24)
        self.chk_aeb.setChecked(s.AEB_enabled)
        self.chk_aeb.toggled.connect(self._on_aeb_toggled)
        aeb_lay.addWidget(self.chk_aeb)
        self._grid.addWidget(aeb_widget, r_aeb, 1)
        self._grid.setRowMinimumHeight(r_aeb, FIELD_ROW_HEIGHT)

    # Cruise helpers

    def _speed_unit(self) -> str:
        # ATS uses mph, ETS2 uses km/h.
        return "mph" if self._settings.last_game == 2 else "km/h"

    def _increment_display_values(self) -> list[str]:
        unit = self._speed_unit()
        return [f"{value} {unit}" for value in (1, 2, 3, 5, 10)]

    def _parse_increment_value(self, value: Any) -> int:
        if isinstance(value, int):
            return value
        match = re.search(r"\d+", str(value))
        if match:
            return int(match.group(0))
        return 1

    def _format_increment_value(self, value: Any) -> str:
        return f"{self._parse_increment_value(value)} {self._speed_unit()}"

    def _set_cruise_mode(self, mode: str) -> None:
        self._set("cc_mode", mode)
        self._update_seg_style(mode)

    def _update_seg_style(self, mode: str) -> None:
        cc_active = mode in ("Cruise control", "ACC")
        for btn, active in ((self._seg_cc, cc_active), (self._seg_sl, not cc_active)):
            btn.setProperty("active", active)
            style = btn.style()
            style.unpolish(btn)
            style.polish(btn)
            btn.update()

    # Button binding: widgets

    def _add_bind_button(self, row: int, key: str) -> BindButton:
        btn = BindButton()
        btn.setFixedSize(*_BIND_BUTTON_SIZE)
        btn.clicked.connect(lambda _=False, k=key: self._on_bind_clicked(k))
        self._grid.addWidget(
            btn, row, 1,
            alignment=Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter,
        )
        self._grid.setRowMinimumHeight(row, FIELD_ROW_HEIGHT)
        self._bind_buttons[key] = btn
        self._refresh_bind_button(key)
        return btn

    def _refresh_bind_button(self, key: str) -> None:
        btn = self._bind_buttons[key]
        raw = getattr(self._settings, key)
        b = migrate_binding(raw)
        btn.set_binding_text(binding_display_name(raw), is_none=b is None)
        btn.setToolTip(str(b.get("device_name") or "") if b else "")

    def _set_cmd_hint(self, text: str) -> None:
        set_cmd = getattr(self.window(), "set_cmd", None)
        if callable(set_cmd):
            set_cmd(text)

    # Button binding: click handling

    def _on_bind_clicked(self, key: str) -> None:
        if self._unassign_armed:
            self._disarm_unassign()
            self._clear_binding(key)
            return
        if self._configuring_key == key:
            self._stop_configuring()
            self._set_cmd_hint("Assignment cancelled.")
            return
        self._start_configuring(key)

    def _on_unassign_clicked(self) -> None:
        if self._configuring_key is not None:
            key = self._configuring_key
            self._stop_configuring()
            self._clear_binding(key)
            return
        if self._unassign_armed:
            self._disarm_unassign()
            self._set_cmd_hint("Unassign cancelled.")
            return
        self._unassign_armed = True
        self._set_unassign_armed_style(True)
        self._set_cmd_hint("Click a configure button to unassign it.")

    def _disarm_unassign(self) -> None:
        self._unassign_armed = False
        self._set_unassign_armed_style(False)

    def _set_unassign_armed_style(self, armed: bool) -> None:
        self._unassign_btn.setProperty("armed", armed)
        style = self._unassign_btn.style()
        style.unpolish(self._unassign_btn)
        style.polish(self._unassign_btn)
        self._unassign_btn.update()

    def _clear_binding(self, key: str) -> None:
        self._set(key, None)
        self._glow_suppress.discard(key)
        self._refresh_bind_button(key)
        self._set_cmd_hint("Button unassigned.")

    # Button binding: capture lifecycle

    def _start_configuring(self, key: str) -> None:
        if self._configuring_key is not None:
            self._stop_configuring()
        self._configuring_key = key
        self._kb_capture_started = False

        for name in ("main_pedal_thread", "button_device_thread"):
            t = self._get_thread(name)
            if t is not None:
                try:
                    t.start_capture()
                except Exception:
                    logger.debug("failed to start capture on %s", name, exc_info=True)

        kb = self._get_thread("keyboard_thread")
        if kb is not None:
            try:
                kb.start_capture()
                with kb.data._lock:
                    self._kb_capture_started = bool(kb.data.capture_active)
            except Exception:
                logger.debug("failed to start keyboard capture", exc_info=True)

        self._bind_buttons[key].set_configuring(True)
        self._set_cmd_hint(
            "Press a button, hat direction or key to assign. "
            "Esc or click again to cancel."
        )

    def _stop_configuring(self) -> None:
        key = self._configuring_key
        self._configuring_key = None
        self._kb_capture_started = False
        for name in ("main_pedal_thread", "keyboard_thread", "button_device_thread"):
            t = self._get_thread(name)
            if t is not None:
                try:
                    t.cancel_capture()
                except Exception:
                    logger.debug("failed to cancel capture on %s", name, exc_info=True)
        if key is not None and key in self._bind_buttons:
            self._bind_buttons[key].set_configuring(False)

    def cancel_configuring(self) -> None:
        """Abort any active capture / armed unassign (e.g. drawer closed)."""
        if self._configuring_key is not None:
            self._stop_configuring()
            self._set_cmd_hint("Assignment cancelled.")
        if self._unassign_armed:
            self._disarm_unassign()

    # Button binding: polling (QTimer, main thread)

    def _poll_bindings(self) -> None:
        try:
            if self._configuring_key is not None:
                self._poll_capture()
            self._poll_held_glow()
        except Exception:
            logger.debug("binding poll failed", exc_info=True)

    def _poll_capture(self) -> None:
        key = self._configuring_key
        binding: dict | None = None

        pt = self._get_thread("main_pedal_thread")
        if pt is not None:
            with pt.data._lock:
                ev = pt.data.capture_event
            if ev is not None:
                ev = pt.consume_capture()
            if isinstance(ev, tuple) and len(ev) >= 3 and ev[0] == "joystick":
                _, guid, code = ev[:3]
                label = ev[3] if len(ev) > 3 else f"button {code}"
                device_name = ev[4] if len(ev) > 4 else ""
                binding = {
                    "source": "joystick",
                    "device_guid": guid,
                    "device_name": device_name,
                    "label": label,
                    "code": code,
                }

        if binding is None:
            bt = self._get_thread("button_device_thread")
            if bt is not None:
                with bt.data._lock:
                    ev = bt.data.capture_event
                if ev is not None:
                    ev = bt.consume_capture()
                if isinstance(ev, tuple) and len(ev) >= 3 and ev[0] == "button_device":
                    _, vid_pid, button_id = ev[:3]
                    label = ev[3] if len(ev) > 3 else f"button {button_id}"
                    device_name = ev[4] if len(ev) > 4 else ""
                    binding = {
                        "source": "button_device",
                        "vid_pid": vid_pid,
                        "device_name": device_name,
                        "label": label,
                        "button_id": button_id,
                    }

        if binding is None:
            kb = self._get_thread("keyboard_thread")
            if kb is not None:
                with kb.data._lock:
                    ev = kb.data.capture_event
                    kb_active = kb.data.capture_active
                if ev is not None:
                    key_name = kb.consume_capture()
                    if key_name:
                        binding = {"source": "keyboard", "code": key_name}
                elif self._kb_capture_started and not kb_active:
                    # Esc pressed: keyboard hook cleared its capture flag.
                    self._stop_configuring()
                    self._set_cmd_hint("Assignment cancelled.")
                    return

        if binding is not None and key is not None:
            self._finish_capture(key, binding)

    def _finish_capture(self, key: str, binding: dict) -> None:
        self._stop_configuring()
        self._steal_duplicates(binding, except_key=key)
        self._set(key, binding)
        # Keep the blue highlight off, and CC actions suppressed, until the
        # freshly assigned input is physically released.
        self._glow_suppress.add(key)
        pt = self._get_thread("main_pedal_thread")
        if pt is not None:
            try:
                pt.set_capture_guard(binding)
            except Exception:
                logger.debug("failed to set capture guard", exc_info=True)
        self._refresh_bind_button(key)
        self._set_cmd_hint(f"Assigned {binding_display_name(binding)}.")

    def _steal_duplicates(self, binding: dict, *, except_key: str) -> None:
        """Move the input if it was bound to another action: one input, one action."""
        sig = self._binding_signature(binding)
        if sig is None:
            return
        for other in _BIND_KEYS:
            if other == except_key:
                continue
            if self._binding_signature(migrate_binding(getattr(self._settings, other))) == sig:
                setattr(self._settings, other, None)
                logger.info("binding moved: %s unassigned", other)
                if other in self._bind_buttons:
                    self._refresh_bind_button(other)

    @staticmethod
    def _binding_signature(b: dict | None) -> tuple | None:
        if not b:
            return None
        source = b.get("source")
        if source == "joystick":
            return (source, b.get("device_guid"), b.get("code"))
        if source == "keyboard":
            return (source, b.get("code"))
        if source == "button_device":
            return (source, b.get("vid_pid"), b.get("button_id"))
        return None

    def _poll_held_glow(self) -> None:
        # Skip while the drawer is collapsed or the window is hidden.
        if not self.isVisible() or self.width() < 10:
            return
        for key, btn in self._bind_buttons.items():
            if key == self._configuring_key:
                continue
            raw = getattr(self._settings, key)
            if migrate_binding(raw) is None:
                btn.set_held(False)
                continue
            held = resolve_held(raw)
            if key in self._glow_suppress:
                if held:
                    btn.set_held(False)
                    continue
                self._glow_suppress.discard(key)
            btn.set_held(held)

    @staticmethod
    def _get_thread(name: str):
        try:
            return registry.get_thread(name)
        except KeyError:
            return None
        except Exception:
            logger.debug("registry lookup failed for %s", name, exc_info=True)
            return None

    def _on_acc_toggled(self, checked: bool) -> None:
        if checked:
            self.chk_acc.blockSignals(True)
            self.chk_acc.setChecked(False)
            self.chk_acc.blockSignals(False)
            self._show_confirm(
                "Enable Adaptive Cruise Control?",
                "This is a BETA feature. It may behave unexpectedly and "
                "could cause unintended braking or acceleration.\n\n"
                "Are you sure you want to enable it?",
                on_confirm=lambda: (
                    self.chk_acc.blockSignals(True),
                    self.chk_acc.setChecked(True),
                    self.chk_acc.blockSignals(False),
                    self._set("acc_enabled", True),
                ),
            )
        else:
            self._set("acc_enabled", False)

    def _on_aeb_toggled(self, checked: bool) -> None:
        if checked:
            self.chk_aeb.blockSignals(True)
            self.chk_aeb.setChecked(False)
            self.chk_aeb.blockSignals(False)
            self._show_confirm(
                "Enable Emergency Braking?",
                "This is a BETA feature. It may trigger unexpected hard "
                "braking.\n\nAre you sure you want to enable it?",
                on_confirm=lambda: (
                    self.chk_aeb.blockSignals(True),
                    self.chk_aeb.setChecked(True),
                    self.chk_aeb.blockSignals(False),
                    self._set("AEB_enabled", True),
                ),
            )
        else:
            self._set("AEB_enabled", False)

    # Section 4 – One‑Pedal‑Drive

    def _build_one_pedal_drive(self) -> None:
        s = self._settings
        p = self._inner

        self._spacer(8)
        new_section_header(p, self._r(), "One-Pedal-Drive")

        new_label(p, self._r(0), 0, "One Pedal Drive mode:")
        self.chk_opd = new_checkbutton(
            p, self._r(), 1, bool(s.opd_mode_variable),
            callback=self._on_opd_toggled,
        )

        # Conditional rows (visible only when OPD is on) ---
        self._opd_cond_start = self._row

        self.ent_offset, _, _ = self._field_with_subtext(
            "  Offset:",
            lambda c, r, col: new_entry(
                c, r, col,
                value=s.offset_variable, value_type=float,
                minimum=0.0, maximum=0.5,
                callback=lambda v: self._set("offset_variable", v),
            ),
            "The amount you have to press the gas to not be braking or accelerating",
        )

        self.ent_max_brake, _, _ = self._field_with_subtext(
            "  Max OPD brake:",
            lambda c, r, col: new_entry(
                c, r, col,
                value=s.max_opd_brake_variable, value_type=float,
                minimum=0.0, maximum=0.5,
                callback=lambda v: self._set("max_opd_brake_variable", v),
            ),
            "The amount of braking when not touching the pedals",
        )

        self._opd_cond_end = self._row

        # Always‑visible rows ---
        new_label(p, self._r(0), 0, "Gas exponent:")
        self.ent_gas_exp = new_entry(
            p, self._r(), 1,
            value=s.gas_exponent_variable if s.gas_exponent_variable else 2.0,
            value_type=float, minimum=0.8, maximum=2.5,
            callback=lambda v: self._set("gas_exponent_variable", v),
        )

        new_label(p, self._r(0), 0, "Brake exponent:")
        self.ent_brake_exp = new_entry(
            p, self._r(), 1,
            value=s.brake_exponent_variable if s.brake_exponent_variable else 2.0,
            value_type=float, minimum=0.8, maximum=2.5,
            callback=lambda v: self._set("brake_exponent_variable", v),
        )

        new_label(p, self._r(0), 0, "Weight adjustment brake:")
        self.chk_weight_adj = new_checkbutton(
            p, self._r(), 1, s.weight_adjustment,
            callback=lambda v: self._set("weight_adjustment", v),
        )

        # Apply conditional visibility
        for r in range(self._opd_cond_start, self._opd_cond_end):
            self._set_row_visible(r, bool(s.opd_mode_variable))

    def _on_opd_toggled(self, checked: bool) -> None:
        self._set("opd_mode_variable", int(checked))
        for r in range(self._opd_cond_start, self._opd_cond_end):
            self._set_row_visible(r, checked)

    # Section 5 – Footer / Credits

    def _build_footer(self) -> None:
        p = self._inner

        self._spacer(8)

        cred_header = QLabel("Implemented libraries:")
        cred_header.setObjectName("creditLabel")
        cred_header.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._grid.addWidget(cred_header, self._r(), 0, 1, 2)

        for lib in [
            "SCSController - mogaika",
            "pygame - pygame",
            "Truck telemetry - Dreagonmon",
        ]:
            lbl = QLabel(lib)
            lbl.setObjectName("creditLabel")
            lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
            self._grid.addWidget(lbl, self._r(), 0, 1, 2)

        # Reinstall SDK
        btn_sdk = QPushButton("reinstall SDK")
        btn_sdk.setObjectName("reinstallButton")
        btn_sdk.clicked.connect(self._reinstall_sdk)
        self._grid.addWidget(btn_sdk, self._r(), 0, 1, 2)

        # Reset all settings -- button + subtext share one container so the
        # subtext hugs the button, same as _field_with_subtext (no label
        # column needed here, so it's built directly rather than through it).
        reset_container = QWidget()
        reset_container.setObjectName("resetContainer")
        reset_container.setStyleSheet("QWidget#resetContainer { background: transparent; }")
        reset_grid = QGridLayout(reset_container)
        reset_grid.setContentsMargins(0, 0, 0, 0)
        reset_grid.setVerticalSpacing(SUBTEXT_GAP_TOP)

        self._reset_btn = QPushButton("reset all settings")
        self._reset_btn.setObjectName("dangerButton")
        self._reset_btn.clicked.connect(self._on_reset_click)
        reset_grid.addWidget(self._reset_btn, 0, 0, 1, 2)

        new_subtext(reset_container, 1, 0, "this requires a program restart", col_span=2)

        self._grid.addWidget(reset_container, self._r(), 0, 1, 2)

    def _reinstall_sdk(self) -> None:
        """Confirm, then force a fresh re-download of every SDK DLL."""
        self._show_confirm(
            "Reinstall SDK?",
            "This re-downloads and reinstalls every plugin DLL for all detected "
            "ETS2/ATS installations. Any running game will be closed.\n\n"
            "Continue?",
            on_confirm=self._do_reinstall_sdk,
        )

    def _do_reinstall_sdk(self) -> None:
        from core.sdk_installer import (
            SUPPORTED_GAME_VERSION,
            GameApplyResult,
            start_reinstall,
        )

        # Runs on the SDK worker thread. PopupWindow.emit is thread-safe (it
        # marshals onto the GUI thread), so the outcome is surfaced with a
        # styled check / warning popup straight from here.
        def _on_done(results: list["GameApplyResult"]) -> None:
            if not results:
                PopupWindow.emit(
                    "SDK reinstall",
                    "No ETS2 or ATS installation was found.",
                    "n",
                    duration_ms=5000,
                )
                return
            unsupported = [r for r in results if r.unsupported]
            if unsupported:
                games = ", ".join(r.game_type.upper() for r in unsupported)
                PopupWindow.emit(
                    "Unsupported game version",
                    f"{games} {SUPPORTED_GAME_VERSION} is not supported yet.",
                    "e",
                    duration_ms=8000,
                )
                return
            failed = [r for r in results if not r.success]
            if failed:
                games = ", ".join(r.game_type.upper() for r in failed)
                PopupWindow.emit(
                    "SDK reinstall failed",
                    f"Could not reinstall the SDK for {games}. See the log for details.",
                    "w",
                    duration_ms=7000,
                )
            else:
                games = ", ".join(r.game_type.upper() for r in results)
                PopupWindow.emit(
                    "SDK reinstalled",
                    f"All plugins were reinstalled for {games}.",
                    "c",
                    duration_ms=5000,
                )

        self._set_cmd_hint("Reinstalling SDK...")
        try:
            start_reinstall(_on_done)
        except Exception:
            logger.exception("failed to start SDK reinstall")
            PopupWindow.emit(
                "SDK reinstall failed", "Could not start the reinstall.", "e", duration_ms=10000
            )

    def _on_reset_click(self) -> None:
        if self._reset_armed:
            self._reset_armed = False
            self._reset_btn.setText("reset all settings")
            self._on_reset()
        else:
            self._reset_armed = True
            self._reset_btn.setText("Are you sure? Click again to confirm")
            QTimer.singleShot(3000, self._disarm_reset)

    def _disarm_reset(self) -> None:
        self._reset_armed = False
        self._reset_btn.setText("reset all settings")

    # Utilities

    def _on_hide_links(self) -> None:
        """Hide the Patreon/YouTube buttons and persist the preference."""
        self._btn_patreon.hide()
        self._btn_youtube.hide()
        self._hide_btn.hide()
        self._set("hide_button_action", True)

    def _launch_updater(self) -> None:
        """Open the standalone updater exe.

        The updater is a separate program installed at
        ``<install root>/updater/updater.exe`` (frozen builds); the install
        root is the directory holding MonoCruise.exe. In a dev tree the exe
        isn't built, so fall back to running the updater source module with the
        current interpreter. Best-effort: never let a failure crash the panel.
        """
        if getattr(sys, "frozen", False):
            install_root = os.path.dirname(os.path.abspath(sys.executable))
        else:
            install_root = _PROJECT_ROOT
        exe = os.path.join(install_root, "updater", "updater.exe")
        script = os.path.join(install_root, "updater", "updater.py")
        try:
            if os.path.exists(exe):
                subprocess.Popen([exe], cwd=install_root)
            elif os.path.exists(script):
                subprocess.Popen([sys.executable, script], cwd=install_root)
            else:
                logger.warning("updater not found at %s", exe)
        except Exception:
            logger.exception("failed to launch updater")

    def _set(self, key: str, value: Any) -> None:
        """Update a settings field and persist."""
        setattr(self._settings, key, value)
        self._on_save()

    def _set_row_visible(self, row: int, visible: bool) -> None:
        for col in range(self._grid.columnCount()):
            item = self._grid.itemAtPosition(row, col)
            if item and item.widget():
                item.widget().setVisible(visible)
        # A hidden row's widgets are invisible, but setRowMinimumHeight (used
        # to give field rows a uniform height) still reserves that space
        # unless cleared, leaving a blank gap. Stash the row's real minimum
        # while hidden and restore it on show; a row that was never hidden
        # isn't in the map, so showing it again is a no-op.
        if visible:
            if row in self._hidden_row_heights:
                self._grid.setRowMinimumHeight(row, self._hidden_row_heights.pop(row))
        else:
            self._hidden_row_heights.setdefault(row, self._grid.rowMinimumHeight(row))
            self._grid.setRowMinimumHeight(row, 0)

    # Bulk‑load  (called once after config load to sync widgets → values)

    def apply_settings(self, s: "Settings") -> None:
        """Push every settings value into the corresponding widget."""
        self._settings = s

        # Inputs
        self.lbl_gas.setText(str(s.gasaxis) if s.gasaxis else "—")
        self.lbl_brake.setText(str(s.brakeaxis) if s.brakeaxis else "—")

        # Program settings
        self.chk_autostart.setChecked(s.autostart_variable)
        self.ent_polling.setText(str(s.polling_rate))
        self.chk_hazards.setChecked(s.hazards_variable)
        self.chk_autodisable.setChecked(s.autodisable_hazards)
        self._set_row_visible(self._hazard_auto_row, s.hazards_variable)
        self.chk_horn.setChecked(s.horn_variable)
        self.chk_airhorn.setChecked(s.airhorn_variable)
        self.chk_live_bar.setChecked(s.bar_variable)
        self.opt_channel.setCurrentText(s.update_channel.capitalize())
        self._preview_subtext.setVisible(s.update_channel.lower() == "preview")

        # Cruise control
        self._update_seg_style(s.cc_mode)
        self.ent_global_limit.setText(
            "" if s.global_speed_limit_kmh is None else str(s.global_speed_limit_kmh)
        )
        if self._configuring_key is not None:
            self._stop_configuring()
        self._glow_suppress.clear()
        for key in self._bind_buttons:
            self._refresh_bind_button(key)
        # Keep persisted values numeric; add units only in UI display.
        self.opt_short.blockSignals(True)
        self.opt_long.blockSignals(True)
        self.opt_short.clear()
        self.opt_long.clear()
        increment_values = self._increment_display_values()
        self.opt_short.addItems(increment_values)
        self.opt_long.addItems(increment_values)
        short_val = self._format_increment_value(s.short_increments)
        long_val = self._format_increment_value(s.long_increments)
        self.opt_short.setCurrentText(short_val)
        self.opt_long.setCurrentText(long_val)
        self.opt_short.blockSignals(False)
        self.opt_long.blockSignals(False)
        self.chk_hold_reset.setChecked(s.long_press_reset)
        self.chk_show_speed.setChecked(s.show_cc_ui)
        self.opt_scaling.setCurrentText(str(s.cc_panel_scaling) if s.cc_panel_scaling else "100%")

        self.chk_acc.blockSignals(True)
        self.chk_acc.setChecked(bool(s.acc_enabled))
        self.chk_acc.blockSignals(False)

        self.chk_aeb.blockSignals(True)
        self.chk_aeb.setChecked(s.AEB_enabled)
        self.chk_aeb.blockSignals(False)

        # OPD
        self.chk_opd.setChecked(bool(s.opd_mode_variable))
        self.ent_offset.setText(str(s.offset_variable))
        self.ent_max_brake.setText(str(s.max_opd_brake_variable))
        self.ent_gas_exp.setText(str(s.gas_exponent_variable if s.gas_exponent_variable else 2.0))
        self.ent_brake_exp.setText(str(s.brake_exponent_variable if s.brake_exponent_variable else 2.0))
        self.chk_weight_adj.setChecked(s.weight_adjustment)

        for r in range(self._opd_cond_start, self._opd_cond_end):
            self._set_row_visible(r, bool(s.opd_mode_variable))
