"""Transport bar: play/stop/seek, position display, BPM."""
from __future__ import annotations
from PyQt6.QtCore import pyqtSignal, Qt
from PyQt6.QtWidgets import (QWidget, QHBoxLayout, QPushButton, QLabel,
                              QSlider, QDoubleSpinBox)
from ui import styles


class TransportBar(QWidget):
    play_clicked = pyqtSignal()
    stop_clicked = pyqtSignal()
    seek_requested = pyqtSignal(float)   # seconds
    bpm_changed = pyqtSignal(float)      # new BPM value

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedHeight(48)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(8, 4, 8, 4)
        layout.setSpacing(8)

        self._play_btn = QPushButton("▶ 播放")
        self._play_btn.setFixedWidth(80)
        self._stop_btn = QPushButton("■ 停止")
        self._stop_btn.setFixedWidth(80)

        self._pos_label = QLabel("0:00.000")
        self._pos_label.setFixedWidth(80)
        self._pos_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._pos_label.setStyleSheet(f"font-family: Consolas; color: {styles.THEME};")

        self._seek_slider = QSlider(Qt.Orientation.Horizontal)
        self._seek_slider.setRange(0, 10000)
        self._seek_slider.setValue(0)

        # BPM spinbox
        bpm_label = QLabel("BPM:")
        bpm_label.setFixedWidth(36)
        self._bpm_spin = QDoubleSpinBox()
        self._bpm_spin.setRange(20.0, 300.0)
        self._bpm_spin.setValue(120.0)
        self._bpm_spin.setDecimals(1)
        self._bpm_spin.setSingleStep(1.0)
        self._bpm_spin.setFixedWidth(72)
        self._bpm_spin.setToolTip("BPM（每分钟节拍数）")

        self._duration = 0.0
        self._dragging = False

        layout.addWidget(self._play_btn)
        layout.addWidget(self._stop_btn)
        layout.addWidget(self._pos_label)
        layout.addWidget(self._seek_slider, stretch=1)
        layout.addWidget(bpm_label)
        layout.addWidget(self._bpm_spin)

        self._play_btn.clicked.connect(self.play_clicked)
        self._stop_btn.clicked.connect(self.stop_clicked)
        self._seek_slider.sliderPressed.connect(self._on_slider_press)
        self._seek_slider.sliderReleased.connect(self._on_slider_release)
        self._bpm_spin.valueChanged.connect(self.bpm_changed)

    def set_duration(self, sec: float):
        self._duration = sec

    def set_position(self, sec: float):
        if not self._dragging:
            if self._duration > 0:
                self._seek_slider.setValue(int(sec / self._duration * 10000))
            self._pos_label.setText(_fmt_time(sec))

    def set_playing(self, playing: bool):
        self._play_btn.setText("⏸ 暂停" if playing else "▶ 播放")

    def set_bpm(self, bpm: float):
        self._bpm_spin.blockSignals(True)
        self._bpm_spin.setValue(round(bpm, 1))
        self._bpm_spin.blockSignals(False)

    def get_bpm(self) -> float:
        return self._bpm_spin.value()

    def _on_slider_press(self):
        self._dragging = True

    def _on_slider_release(self):
        self._dragging = False
        if self._duration > 0:
            sec = self._seek_slider.value() / 10000.0 * self._duration
            self.seek_requested.emit(sec)


def _fmt_time(sec: float) -> str:
    m = int(sec // 60)
    s = sec - m * 60
    return f"{m}:{s:06.3f}"
