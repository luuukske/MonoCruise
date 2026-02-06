# Background colors
BG_MAIN = "#2b2b2b"           # Main dark grey background
BG_SECTION = "#1e1e1e"        # Darker section background
BG_SECTION_BORDER = "#3a3a3a" # Subtle border color

# Accent colors
COLOR_INACTIVE = "#4a4a4a"    # Inactive icons/progress
COLOR_ACTIVE = "#4caf50"      # Green for completed/progress
COLOR_BUTTON_BLUE = "#2196f3" # Blue button
COLOR_BUTTON_BLUE_HOVER = "#1976d2"
COLOR_BUTTON_UPDATE = "#4caf50"
COLOR_BUTTON_UPDATE_HOVER = "#388e3c"
COLOR_PRERELEASE = "#f44336"  # Red for pre-release badge

# Alert colors (GitHub-style)
COLOR_NOTE = "#0969da"        # Blue for NOTE alerts
COLOR_TIP = "#1a7f37"         # Green for TIP alerts
COLOR_IMPORTANT = "#8250df"   # Purple for IMPORTANT alerts
COLOR_WARNING = "#9a6700"     # Orange for WARNING alerts
COLOR_CAUTION = "#cf222e"     # Red for CAUTION alerts

# Text colors
TEXT_PRIMARY = "#ffffff"
TEXT_SECONDARY = "#b0b0b0"

# Font
FONT_FAMILY = "Inter, Sans-serif"

STYLESHEET = f"""
    * {{
        font-family: {FONT_FAMILY};
    }}

    QMainWindow, QWidget {{
        background-color: {BG_MAIN};
        color: {TEXT_PRIMARY};
    }}
    
    QComboBox {{
        background-color: {BG_SECTION};
        border: 1px solid {BG_SECTION_BORDER};
        border-radius: 5px;
        padding: 8px 12px;
        color: {TEXT_PRIMARY};
        min-width: 20px;
    }}
    
    QComboBox:hover {{
        border-color: {COLOR_BUTTON_BLUE};
    }}
    
    QComboBox:on {{
        border-color: {COLOR_BUTTON_BLUE};
    }}
    
    QComboBox::drop-down {{
        border: none;
        width: 24px;
        subcontrol-origin: padding;
        subcontrol-position: center right;
    }}
    
    QComboBox QAbstractItemView {{
        background-color: {BG_SECTION};
        color: {TEXT_PRIMARY};
        selection-background-color: {COLOR_BUTTON_BLUE};
        border: 1px solid {BG_SECTION_BORDER};
        border-radius: 5px;
        padding: 4px;
    }}
    
    QPushButton {{
        background-color: {BG_SECTION};
        border-radius: 5px;
        padding: 10px 20px;
        font-weight: bold;
        border: none;
        color: {TEXT_PRIMARY};
    }}
    
    QPushButton#blueButton {{
        background-color: {COLOR_BUTTON_BLUE};
        color: {TEXT_PRIMARY};
    }}
    
    QPushButton#blueButton:hover {{
        background-color: {COLOR_BUTTON_BLUE_HOVER};
    }}
    
    QPushButton#blueButton:pressed {{
        background-color: {COLOR_BUTTON_BLUE_HOVER};
    }}
    
    QPushButton#updateButton {{
        background-color: {COLOR_BUTTON_BLUE};
        color: {TEXT_PRIMARY};
    }}
    
    QPushButton#updateButton:hover {{
        background-color: {COLOR_BUTTON_BLUE_HOVER};
    }}
    
    QPushButton#updateButton:pressed {{
        background-color: {COLOR_BUTTON_BLUE_HOVER};
    }}
    
    QPushButton#updateButton:disabled {{
        background-color: {COLOR_INACTIVE};
        color: {TEXT_SECONDARY};
    }}
    
    QLabel#releaseTitle {{
        font-size: 18px;
        font-weight: bold;
    }}
    
    QLabel#prereleaseBadge {{
        background-color: {COLOR_PRERELEASE};
        color: white;
        padding: 4px 10px;
        border-radius: 10px;
        font-size: 12px;
        font-weight: bold;
    }}
    
    QLabel {{
        color: {TEXT_PRIMARY};
    }}
    
    QScrollArea {{
        border: none;
        background-color: transparent;
    }}
    
    QScrollBar:vertical {{
        background: transparent;
        width: 8px;
        margin: 0;
    }}
    
    QScrollBar::handle:vertical {{
        background: {TEXT_SECONDARY};
        border-radius: 4px;
        min-height: 30px;
        margin: 2px;
    }}
    
    QScrollBar::handle:vertical:hover {{
        background: {TEXT_PRIMARY};
    }}
    
    QScrollBar::add-line:vertical,
    QScrollBar::sub-line:vertical {{
        height: 0px;
    }}
    
    QScrollBar::add-page:vertical,
    QScrollBar::sub-page:vertical {{
        background: transparent;
    }}
    
    QFrame {{
        background-color: transparent;
    }}
    
    QSlider::groove:horizontal {{
        border: none;
        height: 6px;
        background: {COLOR_INACTIVE};
        border-radius: 3px;
    }}
    
    QSlider::handle:horizontal {{
        background: {COLOR_BUTTON_BLUE};
        width: 14px;
        height: 14px;
        margin: -4px 0;
        border-radius: 7px;
    }}
    
    QSlider::sub-page:horizontal {{
        background: {COLOR_BUTTON_BLUE};
        border-radius: 3px;
    }}
    
    /* Markdown styles for QLabel/QTextBrowser */
    QLabel[objectName="markdownContent"] {{
        color: {TEXT_PRIMARY};
        line-height: 1.6;
    }}
    
    QTextBrowser {{
        background-color: transparent;
        border: none;
        color: {TEXT_PRIMARY};
        font-family: {FONT_FAMILY};
        line-height: 1.6;
    }}
"""

# Individual widget styles (to be applied directly to widgets)
BLUE_BUTTON_STYLE = f"""
    QPushButton#blueButton {{
        background-color: {COLOR_BUTTON_BLUE};
        color: {TEXT_PRIMARY};
        border-radius: 7px;
        padding: 8px 40px;
        font-weight: bold;
        border: none;
    }}
    QPushButton#blueButton:hover {{
        background-color: {COLOR_BUTTON_BLUE_HOVER};
    }}
    QPushButton#blueButton:pressed {{
        background-color: {COLOR_BUTTON_BLUE_HOVER};
    }}
"""

UPDATE_BUTTON_STYLE = f"""
    QPushButton#updateButton {{
        background-color: {COLOR_BUTTON_BLUE};
        color: {TEXT_PRIMARY};
        border-radius: 7px;
        padding: 8px 40px;
        font-weight: bold;
        border: none;
    }}
    QPushButton#updateButton:hover {{
        background-color: {COLOR_BUTTON_BLUE_HOVER};
    }}
    QPushButton#updateButton:pressed {{
        background-color: {COLOR_BUTTON_BLUE_HOVER};
    }}
    QPushButton#updateButton:disabled {{
        background-color: {COLOR_INACTIVE};
        color: {TEXT_SECONDARY};
    }}
"""

RELEASE_TITLE_STYLE = f"""
    QLabel#releaseTitle {{
        font-size: 18px;
        font-weight: bold;
    }}
"""

PRERELEASE_BADGE_STYLE = f"""
    QLabel#prereleaseBadge {{
        background-color: {COLOR_PRERELEASE};
        color: white;
        padding: 2px 8px;
        border-radius: 12px;
        font-size: 11px;
        font-weight: bold;
    }}
"""