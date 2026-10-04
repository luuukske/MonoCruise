"""Run the existing Windows map exporter without blocking the GUI."""
from pathlib import Path
import sys

from PySide6.QtCore import QProcess
from PySide6.QtWidgets import QPushButton

from core.road_map.service import service


class MapPreparationButton(QPushButton):
    def __init__(self, parent=None):
        super().__init__('Prepare map validation data', parent)
        self.setToolTip('Select the ETS2 folder to export missing variant and placement evidence.')
        self.setEnabled(sys.platform == 'win32')
        self.process = QProcess(self)
        self.process.setProcessChannelMode(QProcess.ProcessChannelMode.MergedChannels)
        self.process.readyReadStandardOutput.connect(self._drain)
        self.process.finished.connect(self._finished)
        self.process.errorOccurred.connect(self._failed)
        self.clicked.connect(self.prepare)

    def prepare(self):
        if sys.platform != 'win32' or self.process.state() != QProcess.ProcessState.NotRunning:
            return
        script = Path(__file__).resolve().parents[2] / 'tools' / 'prepare_trucklib_map.ps1'
        if not script.is_file():
            self.setText('Preparation tool missing; install the latest package')
            return
        self.setEnabled(False)
        self.setText('Preparing map; select the ETS2 folder...')
        self.process.start('powershell.exe', ['-NoProfile', '-STA', '-ExecutionPolicy', 'Bypass',
                                              '-File', str(script)])

    def _drain(self):
        self.process.readAllStandardOutput()

    def _finished(self, code, status):
        self.setEnabled(sys.platform == 'win32')
        if code == 0 and status == QProcess.ExitStatus.NormalExit:
            self.setText('Prepare map validation data')
            service.load()
        else:
            self.setText('Map preparation failed; try again')

    def _failed(self, error):
        self.setEnabled(sys.platform == 'win32')
        self.setText('Preparation tool failed to start; try again')
