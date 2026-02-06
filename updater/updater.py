import sys
import os
import tempfile
import zipfile
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QHBoxLayout, QVBoxLayout,
    QLabel, QComboBox, QPushButton, QFrame, QScrollArea, QSpacerItem, QSizePolicy
)
from PyQt6.QtSvg import QSvgRenderer
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QByteArray
from PyQt6.QtGui import QPixmap, QFontDatabase, QPainter, QCursor

from styles import (
    STYLESHEET, BG_SECTION, BG_SECTION_BORDER, COLOR_INACTIVE, 
    COLOR_ACTIVE, COLOR_PRERELEASE, TEXT_SECONDARY,
    BLUE_BUTTON_STYLE, UPDATE_BUTTON_STYLE, RELEASE_TITLE_STYLE, PRERELEASE_BADGE_STYLE
)
from github_api import GitHubAPI

REPO_OWNER = "luuukske"
REPO_NAME = "test-updater"

# Embedded SVG icons
SVG_ICONS = {
    'download': '''<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M12 13v8l-4-4"/><path d="m12 21 4-4"/><path d="M4.393 15.269A7 7 0 1 1 15.71 8h1.79a4.5 4.5 0 0 1 2.436 8.284"/></svg>''',
    'install': '''<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M12 15V3"/><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><path d="m7 10 5 5 5-5"/></svg>''',
    'check': '''<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"/><path d="m9 12 2 2 4-4"/></svg>'''
}


class UpdateWorker(QThread):
    """Background worker for downloading and installing updates."""
    download_progress = pyqtSignal(float)
    install_progress = pyqtSignal(float)
    stage_changed = pyqtSignal(str)  # 'download', 'install', 'finished'
    error = pyqtSignal(str)
    finished_signal = pyqtSignal()

    def __init__(self, api: GitHubAPI, release: dict, install_dir: str):
        super().__init__()
        self.api = api
        self.release = release
        self.install_dir = install_dir
        self.updater_files = ['updater.exe', 'updater.py']

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

    def _extract_update(self, zip_path: str):
        with zipfile.ZipFile(zip_path, 'r') as zf:
            members = [m for m in zf.namelist()
                       if not any(skip in m for skip in self.updater_files)]
            
            for i, member in enumerate(members):
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
        from PyQt6.QtGui import QPainter, QBrush, QColor, QPainterPath
        
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
        
        # Download progress smoothing
        self._download_smoothed = 0.0
        self._download_raw = 0.0
        self._smoothing_factor = 0.002  # Lower = smoother (0.0 to 1.0)
    
    def set_download_progress(self, progress: float):
        self.download_icon.set_active(True)
        
        # Store raw progress
        self._download_raw = max(0.0, min(1.0, progress))
        
        # If download is actually complete, set to 100% regardless of smoothing
        if self._download_raw >= 1.0:
            self._download_smoothed = 1.0
        else:
            # Apply exponential moving average for smoothing
            # EMA: smoothed = smoothed + alpha * (raw - smoothed)
            self._download_smoothed = self._download_smoothed + self._smoothing_factor * (self._download_raw - self._download_smoothed)
        
        # Apply subtle easing (progress^0.83) to make it appear faster
        eased_progress = self._download_smoothed ** 0.83
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
        # Reset smoothing state
        self._download_smoothed = 0.0
        self._download_raw = 0.0


class SelectorSection(QFrame):
    """Top section: Branch and version selectors."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setStyleSheet(f"background-color: {BG_SECTION}; border-radius: 10px;")
        
        layout = QHBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(10)
        
        self.branch_combo = QComboBox()
        self.branch_combo.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))
        layout.addWidget(self.branch_combo)
        
        self.latest_btn = QPushButton("Select Latest")
        self.latest_btn.setObjectName("blueButton")
        self.latest_btn.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))
        self.latest_btn.setStyleSheet(BLUE_BUTTON_STYLE)
        layout.addWidget(self.latest_btn)
        
        layout.addStretch()
        
        self.version_combo = QComboBox()
        self.version_combo.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))
        self.version_combo.setMinimumWidth(200)
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
        
        # Release body
        self.release_body = QLabel()
        self.release_body.setWordWrap(True)
        self.release_body.setStyleSheet(f"color: {TEXT_SECONDARY};")
        self.release_body.setAlignment(Qt.AlignmentFlag.AlignTop)
        
        scroll = QScrollArea()
        scroll.setWidget(self.release_body)
        scroll.setWidgetResizable(True)
        layout.addWidget(scroll, stretch=1)


class SelectorPanel(QWidget):
    """Right panel containing selector and details as separate sections."""
    
    def __init__(self, api: GitHubAPI, parent=None):
        super().__init__(parent)
        self.api = api
        self.releases = []
        self.current_release = None
        
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(10)  # Gap shows window background
        
        # Selector section (top)
        self.selector = SelectorSection()
        self.selector.branch_combo.currentTextChanged.connect(self._on_branch_changed)
        self.selector.latest_btn.clicked.connect(self._select_latest)
        self.selector.version_combo.currentIndexChanged.connect(self._on_version_changed)
        layout.addWidget(self.selector)
        
        # Details section (bottom)
        self.details = DetailsSection()
        layout.addWidget(self.details, stretch=1)
        
        # Expose for external access
        self.update_btn = self.details.update_btn
        
        self._load_branches()
    
    def _load_branches(self):
        branches = self.api.get_branches()
        self.selector.branch_combo.clear()
        
        branch_names = [b['name'] for b in branches]
        
        for default in ['main', 'master']:
            if default in branch_names:
                branch_names.remove(default)
                branch_names.insert(0, default)
                break
        
        self.selector.branch_combo.addItems(branch_names)
    
    def _on_branch_changed(self, branch: str):
        if not branch:
            return
        self.releases = self.api.get_releases_for_branch(branch)
        self.selector.version_combo.clear()
        
        for release in self.releases:
            tag = release['tag_name']
            if release.get('prerelease'):
                tag += " (pre-release)"
            self.selector.version_combo.addItem(tag, release)
    
    def _on_version_changed(self, index: int):
        if index < 0 or index >= len(self.releases):
            self.current_release = None
            self.details.update_btn.setEnabled(False)
            return
        
        self.current_release = self.releases[index]
        self.details.release_title.setText(
            self.current_release['name'] or self.current_release['tag_name']
        )
        self.details.release_body.setText(
            self.current_release.get('body', 'No description')
        )
        
        if self.current_release.get('prerelease'):
            self.details.prerelease_badge.show()
        else:
            self.details.prerelease_badge.hide()
        
        self.details.update_btn.setEnabled(True)
    
    def _select_latest(self):
        branch = self.selector.branch_combo.currentText()
        latest = self.api.get_latest_release_for_branch(branch)
        
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
    
    def _on_error(self, message: str):
        self.selector_panel.update_btn.setEnabled(True)
        self.selector_panel.details.release_title.setText(f"Error: {message}")


def main():
    app = QApplication(sys.argv)
    app.setStyleSheet(STYLESHEET)
    
    window = UpdaterWindow()
    window.show()
    
    sys.exit(app.exec())


if __name__ == "__main__":
    main()