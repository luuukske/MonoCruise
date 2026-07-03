"""
Shared visual theme for MonoCruise shared factories.

A single ``Theme`` dataclass carries the colours/fonts the shared renderer needs
so it never imports a specific app/updater palette module. Each caller builds a
``Theme`` from its own constants and injects it.

The defaults reproduce the updater's markdown palette, so a bare ``Theme()`` is a
sensible standalone fallback.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Theme:
    # ── Markdown renderer ───────────────────────────────────────────────
    md_text_primary: str = "#ffffff"
    md_text_secondary: str = "#b0b0b0"
    md_section_border: str = "#4a4a4a"
    md_font_family: str = "Inter, Sans-serif"
    md_color_note: str = "#0969da"
    md_color_tip: str = "#1a7f37"
    md_color_important: str = "#8250df"
    md_color_warning: str = "#9a6700"
    md_color_caution: str = "#cf222e"
    md_color_quote: str = "#5a5a5a"
