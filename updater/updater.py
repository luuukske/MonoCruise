import sys
import os
import json
import math
import shutil
import subprocess
import tempfile
import zipfile
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QHBoxLayout, QVBoxLayout,
    QLabel, QPushButton, QFrame, QScrollArea, QSpacerItem, QSizePolicy,
    QMessageBox, QStackedWidget, QGraphicsEffect
)
from PySide6.QtSvg import QSvgRenderer
from PySide6.QtCore import (
    Qt, QThread, Signal, QByteArray, QTimer, QElapsedTimer, QPoint, QPointF,
    Property, QPropertyAnimation, QEasingCurve,
)
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
from dropdown import Dropdown, EASE_STD

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

        def render_blocks(self, markdown_text: str) -> list[str]:
            rendered = self.render(markdown_text)
            return [rendered] if rendered else []

        def style_css(self) -> str:
            return ""

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
# install-root state (channel + installed version) from plain files written by the
# running app: config.json (settings) and installed_version.txt (version marker).
DEFAULT_CHANNEL = "stable"

# Self-update layout, all directly under the install root:
#   updater/          the updater's own exe + support files (this program)
#   updater_pending/  new updater files staged by an update, swapped in at the
#                     NEXT updater launch (sentinel written last marks it complete)
#   updater_old/      previous updater files parked by a swap; locked while that
#                     process ran, deleted at the launch after
#   update_staging.tmp/  scratch dir an update extracts into before any live
#                     file is touched
# The running updater can't overwrite its own exe/DLLs, but Windows does allow
# *renaming* open files — the swap is pure renames, done in-process at startup.
# Deliberately no helper script or watcher process: an unsigned exe spawning a
# script that rewrites exes is a classic antivirus false-positive pattern.
PENDING_DIR = 'updater_pending'
OLD_DIR = 'updater_old'
STAGING_DIR = 'update_staging.tmp'
PENDING_SENTINEL = '.complete'


def updater_dir() -> str:
    """Directory the updater itself lives in.

    Frozen (PyInstaller), __file__ points inside the bundle's _internal dir,
    so use the exe path: {install root}/updater.
    """
    if getattr(sys, 'frozen', False):
        return os.path.dirname(os.path.abspath(sys.executable))
    return os.path.dirname(os.path.abspath(__file__))


def install_root() -> str:
    """The install root: MonoCruise.exe, config.json and updater/ live here."""
    return os.path.dirname(updater_dir())


def read_channel(root: str) -> str:
    """Read the user's update channel from config.json in the install root.
    Defaults to 'stable'."""
    try:
        with open(os.path.join(root, "config.json"), encoding="utf-8") as fh:
            value = json.load(fh).get("update_channel")
        if isinstance(value, str) and value.lower() in {"stable", "preview"}:
            return value.lower()
    except (OSError, ValueError):
        pass
    return DEFAULT_CHANNEL


def installed_version_text(root: str) -> str:
    """Raw installed-version string as written by the app to the install root,
    or '' if unknown."""
    try:
        path = os.path.join(root, "installed_version.txt")
        with open(path, encoding="utf-8") as fh:
            return fh.read().strip()
    except OSError:
        return ""


def installed_version(root: str) -> Version | None:
    """Parsed installed version for comparison, or None if missing/unparseable."""
    try:
        return Version(installed_version_text(root))
    except (InvalidVersion, ValueError):
        return None


def _move_tree(src: str, dst: str) -> None:
    """Move every file under src into dst via per-file renames.

    A directory rename fails while any file inside it is open, but renaming
    the individual files succeeds even for a running exe or loaded DLLs —
    that's what makes the self-update swap possible.
    """
    for dirpath, _dirnames, filenames in os.walk(src):
        rel = os.path.relpath(dirpath, src)
        target_dir = dst if rel == os.curdir else os.path.join(dst, rel)
        os.makedirs(target_dir, exist_ok=True)
        for name in filenames:
            os.replace(os.path.join(dirpath, name), os.path.join(target_dir, name))


def apply_pending_self_update() -> None:
    """Swap in updater files staged by a previous run. Call before Qt starts."""
    if not getattr(sys, 'frozen', False):
        return  # dev runs from source: nothing to swap
    _apply_pending_self_update(updater_dir(), install_root())


def _apply_pending_self_update(udir: str, root: str) -> None:
    """Testable core of the startup swap. Must never raise: whatever happens
    to the pending files, the updater has to stay launchable.

    The swap only affects files on disk; the current process keeps running the
    already-loaded old code and the new updater takes effect next launch.
    """
    # Leftovers from previous runs: updater_old was locked by the process that
    # created it, and a staging dir means an update crashed mid-extract.
    shutil.rmtree(os.path.join(root, OLD_DIR), ignore_errors=True)
    shutil.rmtree(os.path.join(root, STAGING_DIR), ignore_errors=True)

    pending = os.path.join(root, PENDING_DIR)
    old = os.path.join(root, OLD_DIR)
    if not os.path.isdir(pending):
        return
    sentinel = os.path.join(pending, PENDING_SENTINEL)
    if not os.path.isfile(sentinel):
        # Staging never completed; discard it. The update can simply be re-run.
        shutil.rmtree(pending, ignore_errors=True)
        return
    try:
        os.remove(sentinel)
        _move_tree(udir, old)      # park current files (renames work while running)
        _move_tree(pending, udir)  # staged files in
        shutil.rmtree(pending, ignore_errors=True)
    except OSError:
        # Roll back whatever moved out. os.replace restores over any half-moved
        # new files, so this converges on the intact old updater.
        try:
            _move_tree(old, udir)
        except OSError:
            pass


def release_version(release: dict) -> Version | None:
    """Parse a release's tag (e.g. 'v1.1.0-preview.1') into a Version, or None."""
    tag = (release or {}).get("tag_name", "")
    try:
        return Version(tag[1:] if tag.startswith("v") else tag)
    except (InvalidVersion, ValueError):
        return None

# Embedded SVG icons
SVG_ICONS = { # credits to https://lucide.dev/
    'download': '''<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M12 13v8l-4-4"/><path d="m12 21 4-4"/><path d="M4.393 15.269A7 7 0 1 1 15.71 8h1.79a4.5 4.5 0 0 1 2.436 8.284"/></svg>''',
    'install': '''<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M12 15V3"/><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><path d="m7 10 5 5 5-5"/></svg>''',
    'check': '''<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"/><path d="m9 12 2 2 4-4"/></svg>''',
    'wifi-off': '''<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="lucide lucide-wifi-off-icon lucide-wifi-off"><path d="M12 20h.01"/><path d="M8.5 16.429a5 5 0 0 1 7 0"/><path d="M5 12.859a10 10 0 0 1 5.17-2.69"/><path d="M19 12.859a10 10 0 0 0-2.007-1.523"/><path d="M2 8.82a15 15 0 0 1 4.177-2.643"/><path d="M22 8.82a15 15 0 0 0-11.288-3.764"/><path d="m2 2 20 20"/></svg>'''
}


class UpdateWorker(QThread):
    """Background worker for downloading and installing updates."""
    download_progress = Signal(float)
    install_progress = Signal(float)
    stage_changed = Signal(str)  # 'download', 'install', 'finished'
    error = Signal(str)
    finished_signal = Signal()

    # Paths inside the install root that must never be overwritten or deleted by
    # an update. Matched as posix-style path prefixes (relative to install root).
    PRESERVE_PATHS = (
        'config.json',
        'config.json.bak',
        'logs/',
    )
    # Root-level updater files from the old flat zip layout: can't be replaced
    # while the updater runs, and there's no pending-swap path for them. Current
    # zips ship the updater under updater/ instead, which is staged to
    # updater_pending/ by _move_into_place.
    UPDATER_FILES = ('updater.exe', 'updater.py')

    def __init__(self, api: GitHubAPI, release: dict, install_root: str):
        super().__init__()
        self.api = api
        self.release = release
        self.install_root = install_root

    def run(self):
        try:
            asset_url = self.api.get_release_asset_url(self.release)
            if not asset_url:
                self.error.emit("No update package found in release")
                return

            self.stage_changed.emit('download')
            # Unique temp name: a second updater instance or a stale leftover
            # must never share the download file.
            fd, temp_zip = tempfile.mkstemp(prefix='MonoCruise-update-', suffix='.zip')
            os.close(fd)
            try:
                self.api.download_asset(
                    asset_url, temp_zip,
                    progress_callback=lambda p: self.download_progress.emit(p)
                )

                self.stage_changed.emit('install')
                self._install_update(temp_zip)
            finally:
                try:
                    os.remove(temp_zip)
                except OSError:
                    pass

            self.stage_changed.emit('finished')
            self.finished_signal.emit()

        except Exception as e:
            self.error.emit(str(e))

    def _should_skip(self, member: str) -> bool:
        """Return True if a zip member should not be extracted."""
        rel = member.replace('\\', '/')
        rel_lc = rel.lower()
        # Root-level only: updater/updater.exe from the current zip layout must
        # go through the pending-swap path, not be skipped.
        if rel_lc in self.UPDATER_FILES:
            return True
        # Skip anything in the preserve list: never clobber user state.
        for path in self.PRESERVE_PATHS:
            if rel_lc == path.lower().rstrip('/'):
                return True
            if path.endswith('/') and rel_lc.startswith(path.lower()):
                return True
        return False

    def _install_update(self, zip_path: str):
        """Extract to a staging dir inside the install root, validate, then
        move files into place.

        Staging first means a corrupt archive, a full disk or an unwritable
        install dir is caught before a single live file is touched; the move
        phase that follows is same-volume renames only — the fastest and least
        interruptible part. Staging inside the install root (not %TEMP%)
        guarantees the renames stay on one volume.
        """
        staging = os.path.join(self.install_root, STAGING_DIR)
        shutil.rmtree(staging, ignore_errors=True)
        os.makedirs(staging)
        try:
            with zipfile.ZipFile(zip_path, 'r') as zf:
                bad = zf.testzip()
                if bad is not None:
                    raise ValueError(f"Corrupt entry in update package: {bad}")
                members = [m for m in zf.namelist() if not self._should_skip(m)]

                # Guard against Zip Slip before extracting anything.
                staging_real = os.path.realpath(staging)
                for member in members:
                    target = os.path.realpath(os.path.join(staging, member))
                    if not target.startswith(staging_real + os.sep) and target != staging_real:
                        raise ValueError(f"Blocked malicious zip entry: {member}")

                for i, member in enumerate(members):
                    zf.extract(member, staging)
                    self.install_progress.emit((i + 1) / len(members))

            if not os.path.isfile(os.path.join(staging, 'MonoCruise.exe')):
                raise ValueError("Update package is missing MonoCruise.exe")
            self._ensure_app_not_running()
            self._move_into_place(staging)
        finally:
            shutil.rmtree(staging, ignore_errors=True)

    def _ensure_app_not_running(self):
        """A running MonoCruise.exe locks the files about to be replaced.
        Failing here, before any file has moved, beats failing halfway through.
        """
        exe = os.path.join(self.install_root, 'MonoCruise.exe')
        if not os.path.exists(exe):
            return
        try:
            with open(exe, 'r+b'):
                pass
        except PermissionError:
            raise RuntimeError("Close MonoCruise before updating") from None

    def _move_into_place(self, staging: str):
        """Move validated staged files into the install tree.

        The zip's updater/ subtree goes to updater_pending/ (the running
        updater can't replace its own files; the next launch swaps them in —
        see _apply_pending_self_update). Everything else renames straight into
        the install root. Pending is staged first so the app-file loop below
        never touches the live updater/ dir.
        """
        pending = os.path.join(self.install_root, PENDING_DIR)
        staged_updater = os.path.join(staging, 'updater')
        if os.path.isdir(staged_updater):
            shutil.rmtree(pending, ignore_errors=True)
            _move_tree(staged_updater, pending)
            shutil.rmtree(staged_updater, ignore_errors=True)
            # Sentinel written last: a partial pending dir (crash mid-move) is
            # discarded at the next launch instead of being swapped in.
            with open(os.path.join(pending, PENDING_SENTINEL), 'w', encoding="utf-8") as fh:
                fh.write('')
        _move_tree(staging, self.install_root)


# How long the no-connection splash waits before silently retrying the fetch.
OFFLINE_RETRY_MS = 10_000


class ReleaseFetchWorker(QThread):
    """Fetch the release list off the UI thread so the loading splash can
    animate while GitHub is queried."""
    loaded = Signal(list, object)  # releases, latest release (dict | None)
    failed = Signal(str)

    def __init__(self, api: GitHubAPI, channel: str):
        super().__init__()
        self.api = api
        self.channel = channel

    def run(self):
        try:
            releases = self.api.get_releases_for_channel(self.channel)
            latest = self.api.get_latest_release_for_channel(self.channel)
        except Exception as e:
            self.failed.emit(str(e))
            return
        self.loaded.emit(releases, latest)


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


class LoadingBar(QWidget):
    """Indeterminate horizontal bar: a rounded segment sweeps left-right-left.

    The repaint timer only runs while the widget is visible, so a hidden
    splash page costs nothing.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedSize(styles.SPLASH_BAR_WIDTH, styles.SPLASH_BAR_HEIGHT)
        self._clock = QElapsedTimer()
        self._timer = QTimer(self)
        self._timer.setInterval(styles.SPLASH_BAR_FRAME_MS)
        self._timer.timeout.connect(self.update)

    def showEvent(self, event):
        self._clock.start()
        self._timer.start()
        super().showEvent(event)

    def hideEvent(self, event):
        self._timer.stop()
        super().hideEvent(event)

    def paintEvent(self, event):
        from PySide6.QtGui import QBrush, QColor, QPainterPath

        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        w, h = self.width(), self.height()
        radius = h / 2
        track = QPainterPath()
        track.addRoundedRect(0, 0, w, h, radius, radius)
        painter.fillPath(track, QBrush(QColor(COLOR_INACTIVE)))

        # Cosine-eased sweep position: 0 -> 1 -> 0 over one period.
        period = styles.SPLASH_BAR_PERIOD_MS
        t = (self._clock.elapsed() % period) / period
        pos = 0.5 - 0.5 * math.cos(2 * math.pi * t)
        seg_w = w * styles.SPLASH_BAR_SEGMENT
        painter.setClipPath(track)
        segment = QPainterPath()
        segment.addRoundedRect(pos * (w - seg_w), 0, seg_w, h, radius, radius)
        painter.fillPath(segment, QBrush(QColor(COLOR_ACTIVE)))
        painter.end()


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
        
        self.channel_combo = Dropdown(styles.DROPDOWN_MIN_WIDTH_CHANNEL)
        self.channel_combo.addItems(["Stable", "Preview"])
        layout.addWidget(self.channel_combo)

        layout.addStretch()

        self.installed_label = QLabel("")
        self.installed_label.setStyleSheet(f"color: {TEXT_SECONDARY};")
        self.installed_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.installed_label)

        layout.addStretch()

        # Grows to fit the longest tag so pre-release names aren't elided.
        self.version_combo = Dropdown(styles.DROPDOWN_MIN_WIDTH_VERSION)
        layout.addWidget(self.version_combo)


class _BlockEffect(QGraphicsEffect):
    """Fade + slide-down-from-above for one changelog block, as a paint-time
    graphics effect.

    Animating at paint time (instead of e.g. animating margins) keeps every
    label's geometry fixed for the whole cascade: the scroll-area layout never
    reflows mid-animation, so the content height stays stable (no jitter when
    scrolled to the bottom) and there is no snap when the animation lands.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self._progress = 0.0

    def set_progress(self, value: float):
        if value != self._progress:
            self._progress = value
            self.update()

    def boundingRectFor(self, rect):
        # While sliding, the block paints up to ROW_OFFSET_PX above its rest
        # position; the effect's bounds must cover that band.
        return rect.adjusted(0, -styles.CHANGELOG_ROW_OFFSET_PX, 0, 0)

    def draw(self, painter):
        if self._progress >= 1.0:
            self.drawSource(painter)  # at rest: skip the pixmap round-trip
            return
        if self._progress <= 0.0:
            return
        offset = QPoint()  # filled in by sourcePixmap
        pixmap = self.sourcePixmap(
            Qt.CoordinateSystem.LogicalCoordinates, offset,
            QGraphicsEffect.PixmapPadMode.NoPad,
        )
        if pixmap.isNull():
            return
        dy = -styles.CHANGELOG_ROW_OFFSET_PX * (1.0 - self._progress)
        painter.save()
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, True)
        painter.setOpacity(self._progress)
        painter.drawPixmap(QPointF(offset.x(), offset.y() + dy), pixmap)
        painter.restore()


class ChangelogCascade(QWidget):
    """Stacks the release notes as one QLabel per markdown block (heading/
    paragraph/list item/alert/...) and, on a version change, cascades them in
    with the same fade + slide stagger as the Dropdown's rows.

    A single ``cascade`` clock drives every row's progress (mirroring
    ``_PopupCard.cascade`` in dropdown.py) instead of one animation per row.
    Blank-line spacer blocks are laid out but hold no stagger slot, so the
    visible lines cascade back to back like the dropdown's rows do.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self._layout = QVBoxLayout(self)
        self._layout.setContentsMargins(0, 0, 0, 0)
        self._layout.setSpacing(0)
        self._rows: list[tuple[_BlockEffect, int]] = []  # (effect, stagger slot)
        self._cascade = 0.0
        self._anim = QPropertyAnimation(self, b"cascade", self)
        self._anim.setEasingCurve(QEasingCurve(QEasingCurve.Type.Linear))

    def _get_cascade(self) -> float:
        return self._cascade

    def _set_cascade(self, v: float):
        self._cascade = v
        self._apply_cascade()

    cascade = Property(float, _get_cascade, _set_cascade)

    def _apply_cascade(self):
        dur = styles.CHANGELOG_DURATION_MS
        for effect, slot in self._rows:
            local = self._cascade - slot * styles.CHANGELOG_STAGGER_MS
            t = 0.0 if local <= 0 else (1.0 if local >= dur else local / dur)
            effect.set_progress(EASE_STD.valueForProgress(t))

    def set_blocks(self, style_tag: str, blocks: list[str], animate: bool):
        """Replace the shown blocks. Cascades them in only when *animate* is
        set -- callers pass False when the version selection hasn't actually
        changed, so re-rendering the same release doesn't replay the intro."""
        self._anim.stop()
        while self._layout.count():
            item = self._layout.takeAt(0)
            if item.widget():
                item.widget().hide()  # gone this frame, not at deleteLater time
                item.widget().deleteLater()
        self._rows = []

        slots = 0
        for block_html in blocks:
            label = QLabel()
            label.setWordWrap(True)
            label.setAlignment(Qt.AlignmentFlag.AlignTop)
            label.setTextFormat(Qt.TextFormat.RichText)
            label.setOpenExternalLinks(True)
            label.setStyleSheet(f"""
                QLabel {{
                    color: {TEXT_SECONDARY};
                    font-family: Inter, Sans-serif;
                    font-size: 14px;
                    line-height: 1.6;
                }}
            """)
            label.setText(style_tag + block_html)
            self._layout.addWidget(label)
            if block_html.strip() == '<p></p>':
                continue  # blank-line spacer: static, no stagger slot
            effect = _BlockEffect(label)
            effect.set_progress(1.0)
            label.setGraphicsEffect(effect)
            self._rows.append((effect, slots))
            slots += 1

        if not animate or not self._rows:
            return  # effects already at rest (progress 1.0)

        total = styles.CHANGELOG_STAGGER_MS * (slots - 1) + styles.CHANGELOG_DURATION_MS
        self._cascade = 0.0
        self._apply_cascade()  # progress 0 everywhere before the first paint
        self._anim.setStartValue(0.0)
        self._anim.setEndValue(float(total))
        self._anim.setDuration(total)
        self._anim.start()


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
        self.scroll = QScrollArea()
        self.scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.scroll.setWidgetResizable(True)
        
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
        
        # Release body: one animated block per changelog line/section.
        self.release_body = ChangelogCascade()
        self.scroll_layout.addWidget(self.release_body)
        
        self.scroll_layout.addStretch()
        
        self.scroll.setWidget(scroll_content)

        # Stack: page 0 = release notes, 1 = loading splash, 2 = no connection.
        # Splash pages sit where the notes normally render, content centered.
        self.stack = QStackedWidget()
        self.stack.setStyleSheet(f"background-color: {BG_SECTION};")
        self.stack.addWidget(self.scroll)
        self.stack.addWidget(self._build_loading_page())
        self.stack.addWidget(self._build_offline_page())
        layout.addWidget(self.stack, stretch=1)

    def _build_splash_page(self, widgets: list[QWidget]) -> QWidget:
        """A page with its widgets centered both ways in the notes area."""
        page = QWidget()
        page.setStyleSheet(f"background-color: {BG_SECTION};")
        v = QVBoxLayout(page)
        v.setSpacing(styles.SPLASH_SPACING)
        v.addStretch()
        for widget in widgets:
            v.addWidget(widget, alignment=Qt.AlignmentFlag.AlignHCenter)
        v.addStretch()
        return page

    def _build_loading_page(self) -> QWidget:
        self.loading_bar = LoadingBar()
        label = QLabel("Loading releases…")
        label.setStyleSheet(styles.SPLASH_SUBTITLE_STYLE)
        return self._build_splash_page([self.loading_bar, label])

    def _build_offline_page(self) -> QWidget:
        icon = IconWidget('wifi-off', styles.SPLASH_ICON_SIZE)
        title = QLabel("No internet connection")
        title.setStyleSheet(styles.SPLASH_TITLE_STYLE)
        subtitle = QLabel("Can't reach GitHub — retrying automatically")
        subtitle.setStyleSheet(styles.SPLASH_SUBTITLE_STYLE)
        return self._build_splash_page([icon, title, subtitle])

    def show_content(self):
        self.stack.setCurrentIndex(0)

    def show_loading(self):
        self.stack.setCurrentIndex(1)

    def show_offline(self):
        self.stack.setCurrentIndex(2)

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
        self._last_version_key = None  # last tag_name shown, for cascade-on-change

        self.install_root = install_root()
        self.channel = read_channel(self.install_root)
        self.installed_version = installed_version(self.install_root)
        self.installed_version_text = installed_version_text(self.install_root)

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

        self._fetch_worker = None
        self._pending_fetch = False  # a channel change arrived mid-fetch
        self._retry_timer = QTimer(self)
        self._retry_timer.setSingleShot(True)
        self._retry_timer.setInterval(OFFLINE_RETRY_MS)
        self._retry_timer.timeout.connect(self._retry_fetch)

        self._load_channel(self.channel)

    def _installed_text(self) -> str:
        if not self.installed_version_text:
            return "Installed: unknown"
        return f"Installed: v{self.installed_version_text}"

    def _load_channel(self, channel: str):
        """Fetch the releases visible on *channel* in the background and show
        the loading splash meanwhile."""
        if not channel:
            return
        self.channel = channel.lower()
        self._start_fetch(silent=False)

    def _retry_fetch(self):
        # Silent: keep the no-connection splash up instead of flashing the
        # loader for every automatic retry.
        self._start_fetch(silent=True)

    def _start_fetch(self, silent: bool):
        self._retry_timer.stop()
        if self._fetch_worker is not None and self._fetch_worker.isRunning():
            # A fetch is in flight (e.g. rapid channel switching); remember to
            # refetch for the current channel once it lands.
            self._pending_fetch = True
            return
        self._pending_fetch = False

        self.releases = []
        self.selector.version_combo.clear()
        self.details.release_title.setText("Select a version")
        self.details.prerelease_badge.hide()
        if not silent:
            self.details.show_loading()

        self.api.invalidate_cache()
        worker = ReleaseFetchWorker(self.api, self.channel)
        worker.loaded.connect(lambda releases, latest, w=worker: self._on_fetch_loaded(w, releases, latest))
        worker.failed.connect(lambda _msg, w=worker: self._on_fetch_failed(w))
        self._fetch_worker = worker
        worker.start()

    def _cleanup_fetch_worker(self, worker: ReleaseFetchWorker):
        worker.wait()
        worker.deleteLater()
        if self._fetch_worker is worker:
            self._fetch_worker = None

    def _resume_pending_fetch(self) -> bool:
        """Restart the fetch for the current channel if one was queued while
        the finished worker ran; its results are stale either way."""
        if not self._pending_fetch:
            return False
        self._start_fetch(silent=False)
        return True

    def _on_fetch_loaded(self, worker: ReleaseFetchWorker, releases: list, latest):
        self._cleanup_fetch_worker(worker)
        if self._resume_pending_fetch() or worker.channel != self.channel:
            return
        self.releases = releases
        for release in releases:
            self.selector.version_combo.addItem(release['tag_name'], release)
        self.details.show_content()
        self._select_latest(latest)

    def _on_fetch_failed(self, worker: ReleaseFetchWorker):
        self._cleanup_fetch_worker(worker)
        if self._resume_pending_fetch() or worker.channel != self.channel:
            return
        self.details.show_offline()
        self._retry_timer.start()

    def orphan_fetch_worker(self):
        """Detach a still-running fetch so the window can close immediately.
        Returns the worker (for the caller to wait() on after the UI is gone)
        or None. A blocking wait here would freeze the close for up to the
        15s request timeout."""
        self._retry_timer.stop()
        worker = self._fetch_worker
        self._fetch_worker = None
        if worker is not None and worker.isRunning():
            try:
                worker.loaded.disconnect()
                worker.failed.disconnect()
            except (RuntimeError, TypeError):
                pass
            return worker
        return None

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
        
        # Render markdown to one HTML block per changelog line/section
        markdown_text = self.current_release.get('body', 'No description')
        renderer = GitHubMarkdownRenderer(THEME)
        blocks = renderer.render_blocks(markdown_text)
        style_tag = renderer.style_css()

        # Load video if found: create player lazily, destroy when not needed
        video_url = renderer.get_video_url()
        if video_url:
            player = self.details.ensure_video_player()
            player.load_video(video_url)
        else:
            self.details.destroy_video_player()

        # Cascade the changelog in only when the selected version actually
        # changed (not on a redundant re-render of the same release). A new
        # version starts reading from the top, so drop any old scroll offset
        # before the new (differently sized) content lands.
        version_key = self.current_release.get('tag_name')
        animate = version_key != self._last_version_key
        self._last_version_key = version_key
        if animate:
            self.details.scroll.verticalScrollBar().setValue(0)
        self.details.release_body.set_blocks(style_tag, blocks, animate)

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

    def _select_latest(self, latest):
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

        self.worker = UpdateWorker(
            self.api,
            self.selector_panel.current_release,
            install_root()
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
        self._launch_app()

    def _launch_app(self):
        """Auto-start the freshly installed app. Must run only after the
        install's file moves: once running, MonoCruise locks its files again."""
        exe = os.path.join(install_root(), 'MonoCruise.exe')
        if os.path.isfile(exe):
            subprocess.Popen([exe], cwd=install_root())

    def closeEvent(self, event):
        # Killing the worker mid-move can leave a half-updated install; refuse
        # to close while an update is running.
        if self.worker is not None and self.worker.isRunning():
            event.ignore()
            return
        # A release-list fetch is harmless to abandon, but its QThread must
        # outlive the event loop; main() waits on it after exec() returns.
        self.orphan_fetch = self.selector_panel.orphan_fetch_worker()
        super().closeEvent(event)

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


_INSTANCE_MUTEX = None


def _other_instance_running() -> bool:
    """True if another updater instance already holds the single-instance mutex.

    Two concurrent updaters installing into the same directory would corrupt
    each other (shared staging dir, competing file moves, double self-swap).
    """
    try:
        import ctypes
        kernel32 = ctypes.windll.kernel32
        global _INSTANCE_MUTEX  # keep the handle alive for the process lifetime
        _INSTANCE_MUTEX = kernel32.CreateMutexW(None, False, "MonoCruiseUpdaterSingleInstance")
        return kernel32.GetLastError() == 183  # ERROR_ALREADY_EXISTS
    except Exception:
        return False  # non-Windows dev run: no guard needed


def main():
    duplicate = _other_instance_running()
    if not duplicate:
        # Finish a self-update staged by the previous run. Must happen before
        # QApplication: Qt lazy-loads plugin DLLs from the updater dir, and
        # loading them mid-session from a half-swapped tree could mix versions.
        apply_pending_self_update()

    app = QApplication(sys.argv)
    app.setStyleSheet(STYLESHEET)

    if duplicate:
        QMessageBox.warning(None, "MonoCruise Updater",
                            "The MonoCruise updater is already running.")
        sys.exit(0)

    window = UpdaterWindow()
    window.orphan_fetch = None
    window.show()

    code = app.exec()
    # Window is gone; joining an abandoned fetch thread here is invisible to
    # the user and avoids destroying a running QThread at interpreter exit.
    if window.orphan_fetch is not None:
        window.orphan_fetch.wait()
    sys.exit(code)


if __name__ == "__main__":
    main()
