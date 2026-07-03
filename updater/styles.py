import os
import tempfile

# Background colors
BG_MAIN = "#2b2b2b"           # Main dark grey background
BG_SECTION = "#1e1e1e"        # Darker section background
BG_SECTION_BORDER = "#4a4a4a" # Subtle border color

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
COLOR_QUOTE = "#5a5a5a"       # Grey for blockquotes

# Text colors
TEXT_PRIMARY = "#ffffff"
TEXT_SECONDARY = "#b0b0b0"

# Font
FONT_FAMILY = "Inter, Sans-serif"

# Ordered font fallback list for widgets we paint ourselves (QPainter.drawText
# ignores CSS-style comma lists, so it needs an explicit QFont.setFamilies list).
# Inter is not installed/bundled, so it resolves to the first available fallback.
FONT_FALLBACKS = ["Inter", "Segoe UI", "sans-serif"]

# Custom animated channel/version dropdown.
#
# The dropdown is a fully custom-painted widget
# (styles.py QComboBox rules do not apply to it) because QSS cannot reproduce
# box-shadow, transitions, transform/rotate or an animated border-radius.

# Palette (monochrome, no accent colour)
DROPDOWN_FIELD_BG = "#242424"       # field + popup background
DROPDOWN_BORDER = "#3a3a3a"         # 1px border
DROPDOWN_BORDER_HOVER = "#555555"   # field border on hover (#555)
DROPDOWN_DIVIDER = "#383838"        # inset divider under the field
DROPDOWN_TEXT = "#e8e8e8"           # field + row text
DROPDOWN_CHEVRON = "#8a8a8a"        # chevron stroke
DROPDOWN_SELECTION_BAR = "#d8d8d8"  # 3x15 selection marker bar
DROPDOWN_ROW_HOVER_RGBA = (255, 255, 255, 13)  # rgba(255,255,255,.05) -> a=round(.05*255)=13
DROPDOWN_SHADOW_RGBA = (0, 0, 0, 115)          # box-shadow rgba(0,0,0,.45) -> a=round(.45*255)=115

# Box model (logical px, matching the CSS)
DROPDOWN_RADIUS = 8                 # field/popup corner radius
DROPDOWN_ROW_RADIUS = 6             # row hover/selection radius
DROPDOWN_ROW_MARGIN_Y = 3           # vertical inset of the highlight box within
                                    # each row -> a 2x gap between adjacent boxes
                                    # (row pitch/text position unchanged)
DROPDOWN_BORDER_W = 1
DROPDOWN_FIELD_PAD_X = 13           # field padding: 9px 13px
DROPDOWN_FIELD_PAD_Y = 9
DROPDOWN_FIELD_GAP = 10             # gap between text and chevron
DROPDOWN_CHEVRON_SIZE = 14          # svg 14x14 (viewBox 24, stroke-width 2.5)
DROPDOWN_CHEVRON_STROKE = 2.5
DROPDOWN_POPUP_PAD = 5              # popup padding when open
DROPDOWN_DIVIDER_INSET = 8          # divider margin: 0 8px 6px
DROPDOWN_DIVIDER_GAP = 6
DROPDOWN_ROW_PAD_X = 10             # row padding: 9px 10px
DROPDOWN_ROW_PAD_Y = 9
DROPDOWN_BAR_W = 3                  # selection bar 3x15, radius 2
DROPDOWN_BAR_H = 15
DROPDOWN_BAR_RADIUS = 2
DROPDOWN_BAR_GAP = 10               # gap between bar and text
DROPDOWN_FONT_PX = 14               # font: 500 14px/1 Inter
DROPDOWN_FONT_WEIGHT = 500          # QFont.Weight.Medium

# Field/popup min widths (reference: version 190, channel 120)
DROPDOWN_MIN_WIDTH_VERSION = 190
DROPDOWN_MIN_WIDTH_CHANNEL = 120

# Open-height length limit: the popup shows at most this many rows, longer lists
# scroll. A thin thumb appears on the right edge when the list overflows.
DROPDOWN_MAX_VISIBLE_ROWS = 6
DROPDOWN_SCROLLBAR_W = 3
DROPDOWN_SCROLLBAR_MARGIN = 3       # inset from the card's right/top/bottom edges
DROPDOWN_SCROLLBAR_GAP = 2         # clearance between the highlight box and the thumb
DROPDOWN_SCROLLBAR_RGBA = (255, 255, 255, 46)

# Animation (reference data-props defaults: dur=250ms, stagger=40ms)
DROPDOWN_DURATION_MS = 250
DROPDOWN_STAGGER_MS = 30
DROPDOWN_CASCADE_DELAY_MS = 100     # rows wait this long before sliding in, so the
                                    # slide is visible after the popup has expanded
DROPDOWN_OPACITY_MS = 175           # round(dur * 0.7)
DROPDOWN_SLIDE_PX = 4               # popup translateY(-4px) -> translateY(0)
DROPDOWN_ROW_OFFSET_PX = 7          # row translateY(-7px) on enter
# Easing: field/popup/chevron use cubic-bezier(.22,1,.36,1); rows + opacity use
# the CSS default "ease" = cubic-bezier(.25,.1,.25,1).
DROPDOWN_EASE_SNAP = (0.22, 1.0, 0.36, 1.0)
DROPDOWN_EASE_STD = (0.25, 0.1, 0.25, 1.0)


def _combo_arrow_rule() -> str:
    """Render a down-arrow chevron to a temp PNG and return its stylesheet rule.

    Qt stylesheets can't draw CSS-triangle borders (a ``border-top`` trick
    renders as a solid block, not a triangle), so we rasterise an SVG chevron
    to a PNG and point ``image:`` at it. Returns "" on failure, in which case
    the combo just shows no custom arrow.
    """
    try:
        from PySide6.QtSvg import QSvgRenderer
        from PySide6.QtGui import QImage, QPainter
        from PySide6.QtCore import QByteArray

        svg = (
            '<svg xmlns="http://www.w3.org/2000/svg" width="24" height="16">'
            f'<path d="M5 5.5 L12 12 L19 5.5" fill="none" stroke="{TEXT_SECONDARY}" '
            'stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"/></svg>'
        )
        renderer = QSvgRenderer(QByteArray(svg.encode("utf-8")))
        img = QImage(24, 16, QImage.Format.Format_ARGB32)
        img.fill(0)
        painter = QPainter(img)
        renderer.render(painter)
        painter.end()
        path = os.path.join(tempfile.gettempdir(), "mc_combo_arrow.png")
        if not img.save(path, "PNG"):
            return ""
        url = path.replace("\\", "/")
        return (
            f'QComboBox::down-arrow {{ image: url("{url}"); '
            'width: 12px; height: 8px; margin-right: 10px; }'
        )
    except Exception:
        return ""


_ARROW_RULE = _combo_arrow_rule()

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

    {_ARROW_RULE}

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
        font-size: 22px;
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