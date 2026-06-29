"""
MonoCruise – Settings panel (left‑side drawer).

All five sections:  Inputs · Program Settings · Cruise Control ·
One‑Pedal‑Drive · Footer/Credits.

Reads and writes the shared ``core.settings.Settings`` instance directly.
"""

from __future__ import annotations

import os
import re
import webbrowser
from typing import TYPE_CHECKING, Any, Callable

from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QIcon, QPixmap
from PySide6.QtWidgets import (
    QCheckBox,
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

from ui.main_window.constants import (
    BG_COLOR,
    HEADER_BG,
    RADIUS_SETTINGS_PANEL,
    SETTINGS_COLOR,
    SETTINGS_PANEL_WIDTH,
    WAITING_COLOR,
)
from ui.main_window.widgets import (
    ClickableLabel,
    new_beta_pill,
    new_checkbutton,
    new_clickable_label,
    new_entry,
    new_label,
    new_optionmenu,
    new_section_header,
    new_subtext,
)

if TYPE_CHECKING:
    from core.settings import Settings
    from ui.main_window.window import MonoCruiseWindow

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

        self.setMaximumWidth(SETTINGS_PANEL_WIDTH)
        self.setMinimumWidth(0)
        self.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Expanding)
        self.setStyleSheet("background-color: transparent;")

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

        # Title
        title = QLabel("Settings")
        title.setObjectName("settingsTitle")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        card_lay.addWidget(title)

        # Scroll area
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        # No separate border/radius: the card is the visual container
        scroll.setStyleSheet(
            "QScrollArea { border: none; border-radius: 0px; background-color: transparent; }"
        )
        # Smooth scrolling (replicates yscrollincrement=3)
        scroll.verticalScrollBar().setSingleStep(3)

        inner = QWidget()
        inner.setStyleSheet("background-color: transparent;")
        self._grid = QGridLayout(inner)
        self._grid.setContentsMargins(10, 8, 10, 10)
        self._grid.setSpacing(4)
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

    # Row counter

    def _r(self, advance: int = 1) -> int:
        r = self._row
        self._row += advance
        return r

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
        self.btn_connect.setStyleSheet(f"background-color: {WAITING_COLOR};")
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

        new_label(p, self._r(0), 0, "Target polling rate (Hz):")
        self.ent_polling = new_entry(
            p, self._r(), 1,
            value=s.polling_rate, value_type=int,
            minimum=10, maximum=100,
            callback=lambda v: self._set("polling_rate", v),
        )
        new_subtext(p, self._r(), 0, "How often inputs are read per second (10–100).", col_span=2)

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

        new_label(p, self._r(0), 0, "Update channel:")
        self.opt_channel = new_optionmenu(
            p, self._r(), 1,
            values=["Stable", "Preview"],
            default=s.update_channel.capitalize(),
            callback=lambda v: self._set("update_channel", v.lower()),
        )
        new_subtext(
            p, self._r(), 0,
            "Preview builds are released earlier and may contain bugs.",
            col_span=2,
        )

    def _on_hazards_toggled(self, checked: bool) -> None:
        self._set("hazards_variable", checked)
        self._set_row_visible(self._hazard_auto_row, checked)

    # Section 3 – Cruise Control

    def _build_cruise_control(self) -> None:
        s = self._settings
        p = self._inner

        # Spacer
        spacer = QWidget()
        spacer.setFixedHeight(8)
        spacer.setStyleSheet("background: transparent;")
        self._grid.addWidget(spacer, self._r(), 0, 1, 2)

        new_section_header(p, self._r(), "Cruise Control")

        # Mode label row
        new_label(p, self._r(0), 0, "Mode:")

        # Segmented button pair
        r_mode = self._row - 1  # reuse same row
        seg_frame = QFrame()
        seg_frame.setStyleSheet(
            f"QFrame {{ border: 1.5px solid {SETTINGS_COLOR}; "
            f"border-radius: 7px; background-color: {BG_COLOR}; }}"
        )
        seg_lay = QHBoxLayout(seg_frame)
        seg_lay.setContentsMargins(0, 0, 0, 0)
        seg_lay.setSpacing(0)

        self._seg_cc = QPushButton("Cruise control")
        self._seg_sl = QPushButton("Speed limiter")
        for btn in (self._seg_cc, self._seg_sl):
            btn.setStyleSheet("border-radius: 0px; padding: 5px 12px;")
        self._seg_cc.clicked.connect(lambda: self._set_cruise_mode("Cruise control"))
        self._seg_sl.clicked.connect(lambda: self._set_cruise_mode("Speed limiter"))
        seg_lay.addWidget(self._seg_cc)
        seg_lay.addWidget(self._seg_sl)
        self._grid.addWidget(
            seg_frame, self._r(), 0, 1, 2,
            Qt.AlignmentFlag.AlignRight,
        )
        self._update_seg_style(s.cc_mode)

        # Button detection rows
        new_label(p, self._r(0), 0, "Enable/Disable button:")
        self.lbl_enable_btn = new_clickable_label(
            p, self._r(), 1,
            self._format_btn(s.cc_start_button),
            callback=self._detect_enable_btn,
        )

        new_label(p, self._r(0), 0, "Increase button:")
        self.lbl_increase_btn = new_clickable_label(
            p, self._r(), 1,
            self._format_btn(s.cc_inc_button),
            callback=self._detect_increase_btn,
        )

        new_label(p, self._r(0), 0, "Decrease button:")
        self.lbl_decrease_btn = new_clickable_label(
            p, self._r(), 1,
            self._format_btn(s.cc_dec_button),
            callback=self._detect_decrease_btn,
        )

        # Unassign button
        unassign_btn = QPushButton("Unassign")
        unassign_btn.setFixedWidth(150)
        unassign_btn.clicked.connect(self._unassign_buttons)
        self._grid.addWidget(
            unassign_btn, self._r(), 0, 1, 2,
            Qt.AlignmentFlag.AlignRight,
        )

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

        new_label(p, self._r(0), 0, "Show set speed on screen:")
        self.chk_show_speed = new_checkbutton(
            p, self._r(), 1, s.show_cc_ui,
            callback=lambda v: self._set("show_cc_ui", v),
        )
        new_subtext(
            p, self._r(), 0,
            "just drag it across the screen to move",
            col_span=2,
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
        acc_widget.setStyleSheet("background: transparent;")
        acc_lay = QHBoxLayout(acc_widget)
        acc_lay.setContentsMargins(0, 0, 0, 0)
        acc_lay.setSpacing(4)
        acc_pill = new_beta_pill()
        acc_lay.addStretch()
        acc_lay.addWidget(acc_pill)
        self.chk_acc = QCheckBox()
        self.chk_acc.setFixedSize(24, 24)
        self.chk_acc.setChecked(bool(s.acc_enabled))
        self.chk_acc.toggled.connect(self._on_acc_toggled)
        acc_lay.addWidget(self.chk_acc)
        self._grid.addWidget(acc_widget, r_acc, 1)

        # AEB (BETA)
        r_aeb = self._r()
        new_label(p, r_aeb, 0, "Emergency Braking:")
        aeb_widget = QWidget()
        aeb_widget.setStyleSheet("background: transparent;")
        aeb_lay = QHBoxLayout(aeb_widget)
        aeb_lay.setContentsMargins(0, 0, 0, 0)
        aeb_lay.setSpacing(4)
        aeb_pill = new_beta_pill()
        aeb_lay.addStretch()
        aeb_lay.addWidget(aeb_pill)
        self.chk_aeb = QCheckBox()
        self.chk_aeb.setFixedSize(24, 24)
        self.chk_aeb.setChecked(s.AEB_enabled)
        self.chk_aeb.toggled.connect(self._on_aeb_toggled)
        aeb_lay.addWidget(self.chk_aeb)
        self._grid.addWidget(aeb_widget, r_aeb, 1)

    # Cruise helpers

    @staticmethod
    def _format_btn(value: Any) -> str:
        if value is None or value == "":
            return "Click to assign"
        return str(value)

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
        if mode == "Cruise control" or mode == "ACC":
            self._seg_cc.setStyleSheet(
                f"background-color: {WAITING_COLOR}; border-radius: 0px; padding: 5px 12px;"
            )
            self._seg_sl.setStyleSheet(
                f"background-color: {SETTINGS_COLOR}; border-radius: 0px; padding: 5px 12px;"
            )
        else:
            self._seg_cc.setStyleSheet(
                f"background-color: {SETTINGS_COLOR}; border-radius: 0px; padding: 5px 12px;"
            )
            self._seg_sl.setStyleSheet(
                f"background-color: {WAITING_COLOR}; border-radius: 0px; padding: 5px 12px;"
            )

    def _detect_enable_btn(self) -> None:
        # TODO: enter button‑detection mode for enable/disable
        pass

    def _detect_increase_btn(self) -> None:
        # TODO: enter button‑detection mode for increase
        pass

    def _detect_decrease_btn(self) -> None:
        # TODO: enter button‑detection mode for decrease
        pass

    def _unassign_buttons(self) -> None:
        self._settings.cc_start_button = None
        self._settings.cc_inc_button = None
        self._settings.cc_dec_button = None
        self.lbl_enable_btn.setText("Click to assign")
        self.lbl_increase_btn.setText("Click to assign")
        self.lbl_decrease_btn.setText("Click to assign")
        self._on_save()

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

        spacer = QWidget()
        spacer.setFixedHeight(8)
        spacer.setStyleSheet("background: transparent;")
        self._grid.addWidget(spacer, self._r(), 0, 1, 2)

        new_section_header(p, self._r(), "One-Pedal-Drive")

        new_label(p, self._r(0), 0, "One Pedal Drive mode:")
        self.chk_opd = new_checkbutton(
            p, self._r(), 1, bool(s.opd_mode_variable),
            callback=self._on_opd_toggled,
        )

        # Conditional rows (visible only when OPD is on) ---
        self._opd_cond_start = self._row

        r_off = self._r()
        new_label(p, r_off, 0, "  Offset:")
        self.ent_offset = new_entry(
            p, r_off, 1,
            value=s.offset_variable, value_type=float,
            minimum=0.0, maximum=0.5,
            callback=lambda v: self._set("offset_variable", v),
        )
        new_subtext(
            p, self._r(), 0,
            "The amount you have to press the gas to not be braking or accelerating",
            col_span=2,
        )

        r_mb = self._r()
        new_label(p, r_mb, 0, "  Max OPD brake:")
        self.ent_max_brake = new_entry(
            p, r_mb, 1,
            value=s.max_opd_brake_variable, value_type=float,
            minimum=0.0, maximum=0.5,
            callback=lambda v: self._set("max_opd_brake_variable", v),
        )
        new_subtext(
            p, self._r(), 0,
            "The amount of braking when not touching the pedals",
            col_span=2,
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

        spacer = QWidget()
        spacer.setFixedHeight(8)
        spacer.setStyleSheet("background: transparent;")
        self._grid.addWidget(spacer, self._r(), 0, 1, 2)

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

        # Reset all settings
        self._reset_btn = QPushButton("reset all settings")
        self._reset_btn.setObjectName("dangerButton")
        self._reset_btn.clicked.connect(self._on_reset_click)
        self._grid.addWidget(self._reset_btn, self._r(), 0, 1, 2)

        new_subtext(p, self._r(), 0, "this requires a program restart", col_span=2)

    def _reinstall_sdk(self) -> None:
        # TODO: reinstall ETS2 telemetry SDK files
        pass

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

    def _set(self, key: str, value: Any) -> None:
        """Update a settings field and persist."""
        setattr(self._settings, key, value)
        self._on_save()

    def _set_row_visible(self, row: int, visible: bool) -> None:
        for col in range(self._grid.columnCount()):
            item = self._grid.itemAtPosition(row, col)
            if item and item.widget():
                item.widget().setVisible(visible)

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

        # Cruise control
        self._update_seg_style(s.cc_mode)
        self.lbl_enable_btn.setText(self._format_btn(s.cc_start_button))
        self.lbl_increase_btn.setText(self._format_btn(s.cc_inc_button))
        self.lbl_decrease_btn.setText(self._format_btn(s.cc_dec_button))
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
