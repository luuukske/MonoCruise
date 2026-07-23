# Shared UI (`shared/`)

## Dropdown (`dropdown.py`)

Custom animated dropdown used by the MonoCruise app and the updater. Pixel-faithful port of the reference HTML/CSS/JS (`MonoCruise Dropdown.dc.html`). Fully custom-painted (not a styled `QComboBox`) because Qt Style Sheets cannot reproduce the reference box-shadow, CSS transitions, chevron rotation, or animated field border-radius.

**Behaviour (mirrors the reference):**

- Field with current value + chevron rotating 180° on open.
- Popup extends down from the field, sharing background so field and list fuse (field bottom corners square when open).
- On open: height and opacity grow together; rows cascade with stagger. On close: roll-up without re-stagger.
- Selection: 3×15 bar at row left (no checkmark).
- Dismisses on outside click; closes if the owning window moves or resizes.

**API:** `Dropdown` exposes the `QComboBox` subset MonoCruise uses (`addItem`, `addItems`, `clear`, `count`, `itemData`, `currentText`, `currentIndex`, `setCurrentIndex`, `setCurrentText`, `currentTextChanged`, `currentIndexChanged`).

**Theming:** `DROPDOWN_*` constants are the widget design (from the reference), not app theming. Callers may override `field_bg`, `border_w`, `border_color`, `border_hover`, `radius`, `text_color`, `font_px`, `pad_y`.

**Implementation notes:**

- Popup is a child overlay of the field's top-level window (shared coordinates and DPR; avoids fractional-DPI misalignment).
- Drop shadow is seam-clipped so blur does not paint above the field bottom edge.
- Open height clamped by natural height, `DROPDOWN_MAX_VISIBLE_ROWS`, and window cap; overflow scrolls with a thin thumb.
- Cascade anchor: first on-screen row when scrolled open; rows above anchor render static.

## Markdown renderer (`markdown_renderer.py`)

GitHub-flavoured markdown to HTML for release notes (updater + app). Alert blocks, lists, and `_style_tag()` embed multi-line CSS in triple-quoted strings; those are stylesheet literals, not documentation comments.
