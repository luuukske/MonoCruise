"""
MonoCruise – Design tokens, QSS stylesheet, and UI constants.

Colour values are taken directly from the original MonoCruise.py source.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Application metadata
# ---------------------------------------------------------------------------
APP_NAME = "MonoCruise"
WINDOW_WIDTH = 700
WINDOW_HEIGHT = 500
SETTINGS_PANEL_WIDTH = 400

# ---------------------------------------------------------------------------
# Colour palette
# ---------------------------------------------------------------------------
BG_COLOR       = "#2B2B2B"
BG_DARKER      = "#242424"
SETTINGS_COLOR = "#454545"
WAITING_COLOR  = "#1f538d"
CONNECTED_COLOR = "#304230"
LOST_COLOR     = "#FF0000"
HEADER_BG      = "#454545"
CMD_COLOR      = "#808080"
CMD_BG_COLOR   = "#202020"
TEXT_COLOR     = "#d3d3d3"
SUBTEXT_COLOR  = "#606060"
PILL_RED       = "#FF0000"
BUTTON_HOVER   = "#333366"
DANGER_TEXT    = "#8b0000"
DISABLED_COLOR = "#F1F1F1"
KEY_ON_COLOR   = "#1f53FF"
SPEEDLIMITER_COLOR  = "#008B00"
CRUISECONTROL_COLOR = "#4876FF"

# Action-button accent colours
REINSTALL_COLOR       = "#8B5000"
REINSTALL_HOVER       = "#A86200"
DANGER_BUTTON_COLOR   = "#8B1A1A"
DANGER_BUTTON_HOVER   = "#A82020"

# ---------------------------------------------------------------------------
# Border radii  (change individually as desired)
# ---------------------------------------------------------------------------
RADIUS_BANNER         = 5   # BannerWidget (top status bar)
RADIUS_SETTINGS_PANEL = 5   # SettingsPanel slide-in drawer
RADIUS_SCROLL         = 5   # QScrollArea (settings scroll frame)
RADIUS_BUTTON         = 5   # All QPushButton (factory standard across type)
RADIUS_INPUT          = 5   # QLineEdit and QComboBox

# ---------------------------------------------------------------------------
# Font families
# ---------------------------------------------------------------------------
FONT_FAMILY = "Segoe UI"
FONT_FAMILY_FALLBACK = "Helvetica Neue"
FONT_SIZE = 13
MONO_FAMILY = "Courier New"
MONO_SIZE = 10

# ---------------------------------------------------------------------------
# Centralised QSS
# ---------------------------------------------------------------------------
STYLESHEET = f"""
/* ── Global ─────────────────────────────────────────────────────────── */
QWidget {{
    background-color: {BG_COLOR};
    color: {TEXT_COLOR};
    font-family: "{FONT_FAMILY}", "{FONT_FAMILY_FALLBACK}";
    font-size: {FONT_SIZE}px;
}}

QLabel {{
    background-color: transparent;
    padding: 0px;
}}

QLabel#subtext {{
    color: {SUBTEXT_COLOR};
    font-size: 11px;
}}

QLabel#sectionHeader {{
    background-color: {HEADER_BG};
    font-weight: bold;
    padding: 6px 8px;
    border-radius: 5px;
}}

QLabel#cmdLabel {{
    color: {CMD_COLOR};
    font-family: "{MONO_FAMILY}";
    font-size: {MONO_SIZE}px;
    background-color: {CMD_BG_COLOR};
    padding: 4px 8px;
    border-radius: 5px;
}}

QLabel#versionLabel {{
    color: #505050;
    font-size: 11px;
    background-color: transparent;
}}

QLabel#pillBeta {{
    color: white;
    background-color: {PILL_RED};
    font-size: 11px;
    font-weight: bold;
    padding: 1px 5px;
    border-radius: 15px;
}}

QLabel#settingsTitle {{
    font-size: 17px;
    font-weight: bold;
    color: {TEXT_COLOR};
    padding: 5px 5px;
}}

QLabel#errorLabel {{
    color: {DANGER_TEXT};
    font-size: {FONT_SIZE}px;
}}

QLabel#creditLabel {{
    color: {SUBTEXT_COLOR};
    font-size: 11px;
}}

QLabel#clickable {{
    color: {WAITING_COLOR};
    font-size: {FONT_SIZE}px;
}}
QLabel#clickable:hover {{
    text-decoration: underline;
}}

QPushButton {{
    background-color: {WAITING_COLOR};
    color: white;
    border: none;
    border-radius: {RADIUS_BUTTON}px;
    padding: 6px 14px;
    font-weight: normal;
    font-size: {FONT_SIZE}px;
}}
QPushButton:hover {{
    background-color: {BUTTON_HOVER};
}}
QPushButton:pressed {{
    background-color: #292952;
}}

QPushButton#gearButton {{
    background-color: transparent;
    border: none;
    padding: 0px;
}}
QPushButton#gearButton:hover {{
    background-color: transparent;
}}

QPushButton#hideButton {{
    background-color: transparent;
    color: darkgray;
    font-size: 11px;
    padding: 0px;
    min-width: 20px; max-width: 20px;
    min-height: 20px; max-height: 20px;
}}
QPushButton#hideButton:hover {{
    color: {TEXT_COLOR};
}}

QPushButton#dangerButton {{
    background-color: {DANGER_BUTTON_COLOR};
}}
QPushButton#dangerButton:hover {{
    background-color: {DANGER_BUTTON_HOVER};
}}

QPushButton#supportButton {{
    background-color: white;
    color: black;
    font-size: 11px;
    font-weight: normal;
    height: 20px;
    border-radius: 5px;
}}
QPushButton#supportButton:hover {{
    background-color: lightgrey;
}}

QPushButton#reinstallButton {{
    background-color: {REINSTALL_COLOR};
}}
QPushButton#reinstallButton:hover {{
    background-color: {REINSTALL_HOVER};
}}

QCheckBox {{
    spacing: 0px;
    background-color: transparent;
}}
QCheckBox::indicator {{
    width: 20px; height: 20px;
    border: 2px solid {SETTINGS_COLOR};
    border-radius: 5px;
    background-color: {BG_COLOR};
}}
QCheckBox::indicator:checked {{
    background-color: {SETTINGS_COLOR};
    border-color: {SETTINGS_COLOR};
}}
QCheckBox::indicator:hover {{
    border-color: {WAITING_COLOR};
}}

QLineEdit {{
    background-color: {BG_COLOR};
    color: {TEXT_COLOR};
    border: 1.5px solid {SETTINGS_COLOR};
    border-radius: {RADIUS_INPUT}px;
    padding: 3px 6px;
    font-size: {FONT_SIZE}px;
    selection-background-color: {WAITING_COLOR};
}}
QLineEdit:focus {{
    border-color: {WAITING_COLOR};
}}

QComboBox {{
    background-color: {BG_COLOR};
    color: {TEXT_COLOR};
    border: 1.5px solid {SETTINGS_COLOR};
    border-radius: {RADIUS_INPUT}px;
    padding: 3px 8px;
    min-width: 80px;
    font-size: {FONT_SIZE}px;
}}
QComboBox:hover {{
    border-color: {WAITING_COLOR};
}}
QComboBox::drop-down {{
    border: none; width: 20px;
}}
QComboBox::down-arrow {{
    image: none;
    border-left: 4px solid transparent;
    border-right: 4px solid transparent;
    border-top: 5px solid {TEXT_COLOR};
    margin-right: 6px;
}}
QComboBox QAbstractItemView {{
    background-color: {BG_COLOR};
    color: {TEXT_COLOR};
    selection-background-color: {WAITING_COLOR};
    border: 1px solid {SETTINGS_COLOR};
    outline: 0;
}}

QPushButton#segActive {{
    border-radius: 0px; padding: 5px 12px;
}}
QPushButton#segInactive {{
    background-color: {SETTINGS_COLOR};
    border-radius: 0px; padding: 5px 12px;
}}

QScrollArea {{
    border: 1px solid {SETTINGS_COLOR};
    border-radius: {RADIUS_SCROLL}px;
    background-color: transparent;
}}
QScrollBar:vertical {{
    background-color: {BG_COLOR}; width: 8px; margin: 0;
}}
QScrollBar::handle:vertical {{
    background-color: {SETTINGS_COLOR};
    min-height: 30px; border-radius: 4px;
}}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{ height: 0; }}
QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical {{ background: none; }}

QWidget#overlayBg {{
    background-color: rgba(0, 0, 0, 160);
}}
QFrame#dialogCard {{
    background-color: #252525;
    border: 1px solid {SETTINGS_COLOR};
    border-radius: 8px;
}}
"""