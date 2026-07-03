import sys
import os
import json
import re
import tempfile
import zipfile
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QHBoxLayout, QVBoxLayout,
    QLabel, QComboBox, QPushButton, QFrame, QScrollArea, QSpacerItem, QSizePolicy
)
from PySide6.QtSvg import QSvgRenderer
from PySide6.QtCore import Qt, QThread, Signal, QByteArray
from PySide6.QtGui import QPixmap, QFontDatabase, QPainter, QCursor

import styles
from styles import (
    STYLESHEET, BG_SECTION, BG_SECTION_BORDER, COLOR_INACTIVE,
    COLOR_ACTIVE, COLOR_PRERELEASE, TEXT_SECONDARY,
    UPDATE_BUTTON_STYLE, RELEASE_TITLE_STYLE, PRERELEASE_BADGE_STYLE
)
from packaging.version import Version, InvalidVersion

from github_api import GitHubAPI
from video_player import VideoPlayer

# The shared markdown renderer lives at the repo root (bundled into the updater
# exe via updater.spec). Put the repo root on sys.path so `import shared` works
# both when running from source and from the frozen build.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try:
    from shared import GitHubMarkdownRenderer, Theme
    _SHARED_OK = True
except Exception:  # pragma: no cover - defensive: keep the updater usable
    _SHARED_OK = False
    Theme = None

    class GitHubMarkdownRenderer:  # minimal fallback if shared can't load
        def __init__(self, *_a, **_k):
            self.video_url = None

        def render(self, markdown_text: str) -> str:
            import html as _html
            return f"<pre>{_html.escape(markdown_text or '')}</pre>"

        def get_video_url(self):
            return None


def _shared_theme():
    """Build the shared Theme (markdown colours) from the updater's styles palette."""
    if not _SHARED_OK:
        return None
    return Theme(
        md_text_primary=styles.TEXT_PRIMARY,
        md_text_secondary=styles.TEXT_SECONDARY,
        md_section_border=styles.BG_SECTION_BORDER,
        md_font_family=styles.FONT_FAMILY,
        md_color_note=styles.COLOR_NOTE,
        md_color_tip=styles.COLOR_TIP,
        md_color_important=styles.COLOR_IMPORTANT,
        md_color_warning=styles.COLOR_WARNING,
        md_color_caution=styles.COLOR_CAUTION,
        md_color_quote=styles.COLOR_QUOTE,
    )


THEME = _shared_theme()

REPO_OWNER = "luuukske"
REPO_NAME = "MonoCruise"

# The updater is a separate exe that can't import the app's modules, so it reads
# install-dir state (channel + installed version) from plain files written by the
# running app: config.json (settings) and installed_version.txt (version marker).
DEFAULT_CHANNEL = "stable"


def install_dir() -> str:
    """Directory the updater (and the app it updates) live in."""
    return os.path.dirname(os.path.abspath(__file__))


def read_channel(directory: str) -> str:
    """Read the user's update channel from config.json. Defaults to 'stable'."""
    try:
        with open(os.path.join(directory, "config.json"), encoding="utf-8") as fh:
            value = json.load(fh).get("update_channel")
        if isinstance(value, str) and value.lower() in {"stable", "preview"}:
            return value.lower()
    except (OSError, ValueError):
        pass
    return DEFAULT_CHANNEL


def installed_version_text(directory: str) -> str:
    """Raw installed-version string as written by the app, or '' if unknown."""
    try:
        with open(os.path.join(directory, "installed_version.txt"), encoding="utf-8") as fh:
            return fh.read().strip()
    except OSError:
        return ""


def installed_version(directory: str) -> Version | None:
    """Parsed installed version for comparison, or None if missing/unparseable."""
    try:
        return Version(installed_version_text(directory))
    except (InvalidVersion, ValueError):
        return None


def release_version(release: dict) -> Version | None:
    """Parse a release's tag (e.g. 'v1.1.0-beta.1') into a Version, or None."""
    tag = (release or {}).get("tag_name", "")
    try:
        return Version(tag[1:] if tag.startswith("v") else tag)
    except (InvalidVersion, ValueError):
        return None

# Embedded SVG icons
SVG_ICONS = { # credits to https://lucide.dev/
    'download': '''<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M12 13v8l-4-4"/><path d="m12 21 4-4"/><path d="M4.393 15.269A7 7 0 1 1 15.71 8h1.79a4.5 4.5 0 0 1 2.436 8.284"/></svg>''',
    'install': '''<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M12 15V3"/><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><path d="m7 10 5 5 5-5"/></svg>''',
    'check': '''<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"/><path d="m9 12 2 2 4-4"/></svg>'''
}


class UpdateWorker(QThread):
    """Background worker for downloading and installing updates."""
    download_progress = Signal(float)
    install_progress = Signal(float)
    stage_changed = Signal(str)  # 'download', 'install', 'finished'
    error = Signal(str)
    finished_signal = Signal()

    # Paths inside the install dir that must never be overwritten or deleted by
    # an update. Matched as posix-style path prefixes (relative to install_dir).
    PRESERVE_PATHS = (
        'config.json',
        'config.json.bak',
        'logs/',
    )
    # The updater can't replace itself while running; skip its own files too.
    UPDATER_FILES = ('updater.exe', 'updater.py')

    def __init__(self, api: GitHubAPI, release: dict, install_dir: str):
        super().__init__()
        self.api = api
        self.release = release
        self.install_dir = install_dir

    def run(self):
        try:
            asset_url = self.api.get_release_asset_url(self.release)
            if not asset_url:
                self.error.emit("No update package found in release")
                return

            self.stage_changed.emit('download')
            temp_zip = os.path.join(tempfile.gettempdir(), 'update.zip')
            self.api.download_asset(
                asset_url, temp_zip,
                progress_callback=lambda p: self.download_progress.emit(p)
            )

            self.stage_changed.emit('install')
            self._extract_update(temp_zip)

            os.remove(temp_zip)
            self.stage_changed.emit('finished')
            self.finished_signal.emit()

        except Exception as e:
            self.error.emit(str(e))

    def _should_skip(self, member: str) -> bool:
        """Return True if a zip member should not be extracted."""
        rel = member.replace('\\', '/')
        rel_lc = rel.lower()
        # Skip the updater's own files (locked while it runs).
        for name in self.UPDATER_FILES:
            if rel_lc == name or rel_lc.endswith('/' + name):
                return True
        # Skip anything in the preserve list: never clobber user state.
        for path in self.PRESERVE_PATHS:
            if rel_lc == path.lower().rstrip('/'):
                return True
            if path.endswith('/') and rel_lc.startswith(path.lower()):
                return True
        return False

    def _extract_update(self, zip_path: str):
        with zipfile.ZipFile(zip_path, 'r') as zf:
            bad = zf.testzip()
            if bad is not None:
                raise ValueError(f"Corrupt entry in update package: {bad}")
            members = [m for m in zf.namelist() if not self._should_skip(m)]

            install_real = os.path.realpath(self.install_dir)
            for i, member in enumerate(members):
                # Guard against Zip Slip: reject entries that escape install_dir
                target = os.path.realpath(os.path.join(self.install_dir, member))
                if not target.startswith(install_real + os.sep) and target != install_real:
                    raise ValueError(f"Blocked malicious zip entry: {member}")
                zf.extract(member, self.install_dir)
                self.install_progress.emit((i + 1) / len(members))


class ProgressLine(QWidget):
    """Vertical progress line with rounded corners - fills top to bottom."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedWidth(8)
        self.setMinimumHeight(20)
        self._progress = 0.0
        self.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Expanding)
        
        # 5px margin top and bottom to protect icons
        self.setContentsMargins(0, 20, 0, 20)
        
    def set_progress(self, value: float):
        self._progress = max(0.0, min(1.0, value))
        self.update()
    
    def set_complete(self):
        self._progress = 1.0
        self.update()
        
    def paintEvent(self, event):
        from PySide6.QtGui import QPainter, QBrush, QColor, QPainterPath
        
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        
        margins = self.contentsMargins()
        width = self.width()
        height = self.height() - margins.top() - margins.bottom()
        y_offset = margins.top()
        radius = width / 2
        
        # Background (inactive)
        path_bg = QPainterPath()
        path_bg.addRoundedRect(0, y_offset, width, height, radius, radius)
        painter.fillPath(path_bg, QBrush(QColor(COLOR_INACTIVE)))
        
        # Progress (active) - fills from TOP to BOTTOM
        if self._progress > 0:
            progress_height = int(height * self._progress)
            path_progress = QPainterPath()
            path_progress.addRoundedRect(0, y_offset, width, progress_height, radius, radius)
            painter.fillPath(path_progress, QBrush(QColor(COLOR_ACTIVE)))

        painter.end()


class IconWidget(QLabel):
    """Widget to display SVG icons with currentColor replacement."""
    
    def __init__(self, icon_key: str, size: int = 48, parent=None):
        super().__init__(parent)
        self.svg_content = SVG_ICONS.get(icon_key, '')
        self.icon_size = size
        self.setFixedSize(size, size)
        self._active = False
        self._progress = 0.0
        self._render_icon()
    
    def _render_icon(self):
        """Render SVG with current color applied."""
        
        color = COLOR_ACTIVE if self._active else COLOR_INACTIVE
        
        # Replace currentColor with actual color
        svg_colored = self.svg_content.replace('currentColor', color)
        
        # Convert to bytes for QSvgRenderer
        svg_bytes = QByteArray(svg_colored.encode('utf-8'))
        renderer = QSvgRenderer(svg_bytes)
        
        if not renderer.isValid():
            print(f"Invalid SVG: {self.icon_path}")
            return
        
        # Render to pixmap
        pixmap = QPixmap(self.icon_size, self.icon_size)
        pixmap.fill(Qt.GlobalColor.transparent)
        
        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        
        # Apply opacity based on progress if active
        if self._active and self._progress > 0:
            opacity = 0.5 + (self._progress * 0.5)  # Range from 0.5 to 1.0
            painter.setOpacity(opacity)
        
        renderer.render(painter)
        painter.end()
        
        self.setPixmap(pixmap)
    
    def set_active(self, active: bool):
        self._active = active
        self._render_icon()
    
    def set_progress(self, progress: float):
        """Set progress value (0.0 to 1.0) to make icon dynamic."""
        self._progress = max(0.0, min(1.0, progress))
        if self._active:
            self._render_icon()


class ProgressPanel(QFrame):
    """Left panel showing update progress with icons and lines."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setStyleSheet(f"background-color: {BG_SECTION}; border-radius: 10px;")
        
        layout = QHBoxLayout(self)  # Horizontal to center the vertical content
        layout.setContentsMargins(20, 20, 20, 20)
        
        # Vertical container for icons and lines
        v_container = QVBoxLayout()
        v_container.setSpacing(0)
        v_container.setAlignment(Qt.AlignmentFlag.AlignHCenter)
        
        # Create icons
        self.download_icon = IconWidget('download')
        self.install_icon = IconWidget('install')
        self.finished_icon = IconWidget('check')
        
        # Create progress lines
        self.line_download_to_install = ProgressLine()
        self.line_install_to_finished = ProgressLine()
        
        # Add widgets
        v_container.addWidget(self.download_icon, alignment=Qt.AlignmentFlag.AlignHCenter)
        v_container.addWidget(self.line_download_to_install, stretch=2, alignment=Qt.AlignmentFlag.AlignHCenter)
        v_container.addWidget(self.install_icon, alignment=Qt.AlignmentFlag.AlignHCenter)
        v_container.addWidget(self.line_install_to_finished, stretch=1, alignment=Qt.AlignmentFlag.AlignHCenter)
        v_container.addWidget(self.finished_icon, alignment=Qt.AlignmentFlag.AlignHCenter)
        
        layout.addLayout(v_container)
    
    def set_download_progress(self, progress: float):
        self.download_icon.set_active(True)
        # Apply subtle easing (progress^1.6) to make it appear faster
        eased_progress = progress ** 0.83
        self.line_download_to_install.set_progress(eased_progress)
    
    def set_install_progress(self, progress: float):
        self.line_download_to_install.set_complete()
        self.install_icon.set_active(True)
        self.install_icon.set_progress(progress)
        self.line_install_to_finished.set_progress(progress)
    
    def set_finished(self):
        self.line_install_to_finished.set_complete()
        self.finished_icon.set_active(True)
    
    def reset(self):
        self.download_icon.set_active(False)
        self.install_icon.set_active(False)
        self.install_icon.set_progress(0)
        self.finished_icon.set_active(False)
        self.line_download_to_install.set_progress(0)
        self.line_install_to_finished.set_progress(0)


class SelectorSection(QFrame):
    """Top section: Branch and version selectors."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setStyleSheet(f"background-color: {BG_SECTION}; border-radius: 10px;")
        
        layout = QHBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(10)
        
        self.channel_combo = QComboBox()
        self.channel_combo.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))
        self.channel_combo.addItems(["Stable", "Preview"])
        layout.addWidget(self.channel_combo)

        layout.addStretch()

        self.installed_label = QLabel("")
        self.installed_label.setStyleSheet(f"color: {TEXT_SECONDARY};")
        self.installed_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.installed_label)

        layout.addStretch()

        self.version_combo = QComboBox()
        self.version_combo.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))
        self.version_combo.setMinimumWidth(200)
        # Grow to fit the longest tag so pre-release names aren't elided ("v1.1.0-b...").
        self.version_combo.setSizeAdjustPolicy(
            QComboBox.SizeAdjustPolicy.AdjustToContents
        )
        layout.addWidget(self.version_combo)


class DetailsSection(QFrame):
    """Bottom section: Release details and update button."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setStyleSheet(f"background-color: {BG_SECTION}; border-radius: 10px;")
        
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(10)
        
        # Title row
        title_row = QHBoxLayout()
        title_row.setSpacing(10)
        
        self.release_title = QLabel("Select a version")
        self.release_title.setObjectName("releaseTitle")
        self.release_title.setStyleSheet(RELEASE_TITLE_STYLE)
        title_row.addWidget(self.release_title)
        
        self.prerelease_badge = QLabel("pre-release")
        self.prerelease_badge.setObjectName("prereleaseBadge")
        self.prerelease_badge.setStyleSheet(PRERELEASE_BADGE_STYLE)
        self.prerelease_badge.hide()
        title_row.addWidget(self.prerelease_badge)
        
        title_row.addStretch()
        
        self.update_btn = QPushButton("Update")
        self.update_btn.setObjectName("updateButton")
        self.update_btn.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))
        self.update_btn.setStyleSheet(UPDATE_BUTTON_STYLE)
        self.update_btn.setEnabled(False)
        title_row.addWidget(self.update_btn)
        
        layout.addLayout(title_row)
        
        # Scroll area for video player and release body
        scroll = QScrollArea()
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setWidgetResizable(True)
        
        # Container widget for scroll content
        scroll_content = QWidget()
        scroll_content.setStyleSheet(f"background-color: {BG_SECTION};")
        self.scroll_layout = QVBoxLayout(scroll_content)
        self.scroll_layout.setContentsMargins(0, 0, 0, 0)
        self.scroll_layout.setSpacing(10)
        
        # Video player placeholder (created lazily to avoid GPU/RAM cost)
        self.video_container = QHBoxLayout()
        self.video_container.setContentsMargins(0, 0, 0, 0)
        self.video_player = None  # Created on demand
        
        self.scroll_layout.addLayout(self.video_container)
        
        # Release body (supports HTML for markdown rendering)
        self.release_body = QLabel()
        self.release_body.setWordWrap(True)
        self.release_body.setAlignment(Qt.AlignmentFlag.AlignTop)
        self.release_body.setTextFormat(Qt.TextFormat.RichText)
        self.release_body.setOpenExternalLinks(True)
        self.release_body.setStyleSheet(f"""
            QLabel {{
                color: {TEXT_SECONDARY};
                font-family: Inter, Sans-serif;
                font-size: 14px;
                line-height: 1.6;
            }}
        """)
        self.scroll_layout.addWidget(self.release_body)
        
        self.scroll_layout.addStretch()
        
        scroll.setWidget(scroll_content)
        layout.addWidget(scroll, stretch=1)


    def ensure_video_player(self):
        """Create the VideoPlayer lazily when a video URL is found."""
        if self.video_player is not None:
            return self.video_player
        self.video_player = VideoPlayer()
        self.video_container.addStretch()
        self.video_container.addWidget(self.video_player)
        self.video_container.addStretch()
        return self.video_player

    def destroy_video_player(self):
        """Fully destroy the VideoPlayer to release GPU/RAM resources."""
        if self.video_player is None:
            return
        self.video_player.clear()
        self.video_player.setParent(None)
        self.video_player.deleteLater()
        self.video_player = None
        # Clear the stretch items from the container layout
        while self.video_container.count():
            item = self.video_container.takeAt(0)
            if item.widget():
                item.widget().deleteLater()


class SelectorPanel(QWidget):
    """Right panel containing selector and details as separate sections."""
    
    def __init__(self, api: GitHubAPI, parent=None):
        super().__init__(parent)
        self.api = api
        self.releases = []
        self.current_release = None

        self.install_dir = install_dir()
        self.channel = read_channel(self.install_dir)
        self.installed_version = installed_version(self.install_dir)
        self.installed_version_text = installed_version_text(self.install_dir)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(10)  # Gap shows window background
        
        # Selector section (top)
        self.selector = SelectorSection()
        self.selector.installed_label.setText(self._installed_text())
        self.selector.channel_combo.setCurrentText(self.channel.capitalize())
        self.selector.channel_combo.currentTextChanged.connect(self._on_channel_changed)
        self.selector.version_combo.currentIndexChanged.connect(self._on_version_changed)
        layout.addWidget(self.selector)

        # Details section (bottom)
        self.details = DetailsSection()
        layout.addWidget(self.details, stretch=1)

        # Expose for external access
        self.update_btn = self.details.update_btn

        self._load_channel(self.channel)

    def _installed_text(self) -> str:
        if not self.installed_version_text:
            return "Installed: unknown"
        return f"Installed: v{self.installed_version_text}"

    def _load_channel(self, channel: str):
        """Populate the version list with the releases visible on *channel*."""
        if not channel:
            return
        self.channel = channel.lower()
        self.api.invalidate_cache()
        self.releases = self.api.get_releases_for_channel(self.channel)
        self.selector.version_combo.clear()

        for release in self.releases:
            tag = release['tag_name']
            if release.get('prerelease'):
                tag += " (pre-release)"
            self.selector.version_combo.addItem(tag, release)

        self._select_latest()

    def _on_channel_changed(self, channel: str):
        self._load_channel(channel)
    
    def _on_version_changed(self, index: int):
        if index < 0 or index >= len(self.releases):
            self.current_release = None
            self.details.update_btn.setEnabled(False)
            self.details.destroy_video_player()
            return
        
        self.current_release = self.releases[index]
        self.details.release_title.setText(
            self.current_release['name'] or self.current_release['tag_name']
        )
        
        # Render markdown to HTML
        markdown_text = self.current_release.get('body', 'No description')
        renderer = GitHubMarkdownRenderer(THEME)
        html_content = renderer.render(markdown_text)
        
        # Load video if found: create player lazily, destroy when not needed
        video_url = renderer.get_video_url()
        if video_url:
            player = self.details.ensure_video_player()
            player.load_video(video_url)
        else:
            self.details.destroy_video_player()
        
        # Extract body content for QLabel
        body_match = re.search(r'<body[^>]*>(.*?)</body>', html_content, re.DOTALL)
        style_match = re.search(r'<style[^>]*>(.*?)</style>', html_content, re.DOTALL)
        
        if body_match:
            body_content = body_match.group(1)
            if style_match:
                styles = style_match.group(1)
                body_content = f'<style>{styles}</style>{body_content}'
            self.details.release_body.setText(body_content)
        else:
            self.details.release_body.setText(html_content)
        
        if self.current_release.get('prerelease'):
            self.details.prerelease_badge.show()
        else:
            self.details.prerelease_badge.hide()

        self._apply_install_state()

    def _apply_install_state(self):
        """Reflect the installed version on the update button.

        If the selected release is already the installed version there is
        nothing to do; otherwise it can be installed (newer = upgrade, older =
        downgrade). When the installed version is unknown, always allow it.
        """
        selected = release_version(self.current_release)
        is_installed = (
            self.installed_version is not None
            and selected is not None
            and selected == self.installed_version
        )
        self.details.update_btn.setEnabled(not is_installed)
        self.details.update_btn.setText("Up to date" if is_installed else "Update")

    def _select_latest(self):
        latest = self.api.get_latest_release_for_channel(self.channel)

        if latest:
            for i in range(self.selector.version_combo.count()):
                if self.selector.version_combo.itemData(i)['id'] == latest['id']:
                    self.selector.version_combo.setCurrentIndex(i)
                    break


class UpdaterWindow(QMainWindow):
    """Main application window."""
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("MonoCruise Updater")
        self.setMinimumSize(900, 500)
        
        self.api = GitHubAPI(REPO_OWNER, REPO_NAME)
        self.worker = None
        
        central = QWidget()
        self.setCentralWidget(central)
        
        layout = QHBoxLayout(central)
        layout.setSpacing(10)
        layout.setContentsMargins(10, 10, 10, 10)
        
        # Left: Progress panel
        self.progress_panel = ProgressPanel()
        self.progress_panel.setFixedWidth(100)
        layout.addWidget(self.progress_panel)
        
        # Right: Selector panel
        self.selector_panel = SelectorPanel(self.api)
        self.selector_panel.update_btn.clicked.connect(self._start_update)
        layout.addWidget(self.selector_panel, stretch=1)
    
    def _start_update(self):
        if not self.selector_panel.current_release:
            return
        
        # Prevent launching a second worker while one is still running
        if self.worker is not None and self.worker.isRunning():
            return
        
        self.progress_panel.reset()
        self.selector_panel.update_btn.setEnabled(False)
        
        install_dir = os.path.dirname(os.path.abspath(__file__))
        
        self.worker = UpdateWorker(
            self.api,
            self.selector_panel.current_release,
            install_dir
        )
        self.worker.download_progress.connect(self.progress_panel.set_download_progress)
        self.worker.install_progress.connect(self.progress_panel.set_install_progress)
        self.worker.stage_changed.connect(self._on_stage_changed)
        self.worker.finished_signal.connect(self._on_update_finished)
        self.worker.error.connect(self._on_error)
        self.worker.start()
    
    def _on_stage_changed(self, stage: str):
        if stage == 'finished':
            self.progress_panel.set_finished()
    
    def _on_update_finished(self):
        self.selector_panel.update_btn.setEnabled(True)
        self._cleanup_worker()
    
    def _on_error(self, message: str):
        self.selector_panel.update_btn.setEnabled(True)
        self.selector_panel.details.release_title.setText(f"Error: {message}")
        self._cleanup_worker()
    
    def _cleanup_worker(self):
        if self.worker is not None:
            # Wait for the thread to fully exit before destroying the object.
            # The custom finished_signal fires from inside run(), so the
            # thread is still alive at that point.
            self.worker.wait()
            self.worker.deleteLater()
            self.worker = None


def main():
    app = QApplication(sys.argv)
    app.setStyleSheet(STYLESHEET)
    
    window = UpdaterWindow()
    window.show()
    
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
