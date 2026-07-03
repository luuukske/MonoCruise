"""
MonoCruise shared UI library.

Dependency-free (PySide6 only) helpers shared between the main MonoCruise app and
the standalone updater exe. Keep this package free of any imports from ``ui`` or
``updater`` so the updater can bundle it in isolation.
"""

from shared.theme import Theme
from shared.markdown_renderer import GitHubMarkdownRenderer

__all__ = ["Theme", "GitHubMarkdownRenderer"]
