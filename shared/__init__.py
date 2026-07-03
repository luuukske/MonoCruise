"""
MonoCruise shared UI library.

Dependency-free (PySide6 only) helpers shared between the main MonoCruise app and
the standalone updater exe. Keep this package free of any imports from ``ui`` or
``updater`` so the updater can bundle it in isolation.
"""

from shared.theme import Theme
from shared.markdown_renderer import GitHubMarkdownRenderer

# The animated dropdown widget is NOT re-exported here: importing it pulls in
# PySide6.QtWidgets, which non-GUI consumers of this package shouldn't pay for.
# Use `from shared.dropdown import Dropdown` directly.

__all__ = ["Theme", "GitHubMarkdownRenderer"]
