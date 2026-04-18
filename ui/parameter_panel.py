"""Parameter panel: retune speed, key, scale."""
from __future__ import annotations
from PyQt6.QtCore import pyqtSignal
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QSlider, QComboBox, QGroupBox, QPushButton
)
from PyQt6.QtCore import Qt
from ui import styles

KEYS = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
SCALES = ["major", "minor", "harmonic_minor", "pentatonic", "chromatic"]


class LabeledSlider(QWidget):
    value_changed = pyqtSignal(float)

    def __init__(self, label: str, min_val: float, max_val: float,
                 default: float, fmt: str = "{:.2f}", parent=None):
        super().__init__(parent)
        self._min = min_val
        self._max = max_val
        self._fmt = fmt

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)

        self._label = QLabel(label)
        self._label.setFixedWidth(72)
        layout.addWidget(self._label)

        self._slider = QSlider(Qt.Orientation.Horizontal)
        self._slider.setRange(0, 1000)
        self._slider.setMinimumWidth(140)
        self._slider.setTracking(False)
        self._slider.setValue(int((default - min_val) / (max_val - min_val) * 1000))
        layout.addWidget(self._slider, stretch=1)

        self._value_label = QLabel(fmt.format(default))
        self._value_label.setFixedWidth(52)
        self._value_label.setAlignment(Qt.AlignmentFlag.AlignRight)
        layout.addWidget(self._value_label)

        self._slider.sliderMoved.connect(self._on_preview)
        self._slider.valueChanged.connect(self._on_change)

    def _raw_to_value(self, raw: int) -> float:
        return self._min + raw / 1000.0 * (self._max - self._min)

    def _update_value_label(self, raw: int):
        self._value_label.setText(self._fmt.format(self._raw_to_value(raw)))

    def _on_preview(self, raw: int):
        self._update_value_label(raw)

    def _on_change(self, raw: int):
        self._update_value_label(raw)
        self.value_changed.emit(self._raw_to_value(raw))

    def get_value(self) -> float:
        raw = self._slider.value()
        return self._raw_to_value(raw)

    def set_value(self, val: float):
        raw = int((val - self._min) / (self._max - self._min) * 1000)
        self._slider.setValue(raw)
        self._update_value_label(raw)


class ParameterPanel(QWidget):
    params_changed = pyqtSignal(dict)   # emits full param dict on any change

    def __init__(self, parent=None):
        super().__init__(parent)
        self._suspend_emit = False
        self.setMinimumWidth(320)
        self.setMaximumWidth(420)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(12)

        # Pitch correction group
        pitch_group = QGroupBox("音高修正")
        pg_layout = QVBoxLayout(pitch_group)
        pg_layout.setSpacing(8)

        # Key selector
        key_row = QHBoxLayout()
        key_row.addWidget(QLabel("调式"))
        self._key_combo = QComboBox()
        self._key_combo.addItems(KEYS)
        self._key_combo.setToolTip("选择根音（调式中心）")
        key_row.addWidget(self._key_combo)
        pg_layout.addLayout(key_row)

        # Scale selector
        scale_row = QHBoxLayout()
        scale_row.addWidget(QLabel("音阶"))
        self._scale_combo = QComboBox()
        self._scale_combo.addItems(SCALES)
        self._scale_combo.setToolTip("选择音阶类型")
        scale_row.addWidget(self._scale_combo)
        pg_layout.addLayout(scale_row)

        layout.addWidget(pitch_group)

        # Reset button
        reset_row = QHBoxLayout()
        reset_row.addStretch()
        reset_btn = QPushButton("重置参数")
        reset_btn.setToolTip("重置所有参数到默认值")
        reset_btn.clicked.connect(self._reset_params)
        reset_row.addWidget(reset_btn)
        reset_row.addStretch()
        layout.addLayout(reset_row)
        layout.addStretch()

        # Connect signals
        self._key_combo.currentTextChanged.connect(self._emit)
        self._scale_combo.currentTextChanged.connect(self._emit)

    def _emit(self, *_):
        if self._suspend_emit:
            return
        self.params_changed.emit(self.get_params())

    def _reset_params(self):
        """Reset all parameters to default values."""
        self._suspend_emit = True
        try:
            self._key_combo.setCurrentIndex(0)  # C
            self._scale_combo.setCurrentIndex(0)  # major
        finally:
            self._suspend_emit = False
            self.params_changed.emit(self.get_params())

    def get_params(self) -> dict:
        return {
            "key": self._key_combo.currentText(),
            "scale": self._scale_combo.currentText(),
            "retune_speed": 0.0,  # Default: preserve original pitch (no auto-tune)
        }

    def set_params(self, params: dict):
        self._suspend_emit = True
        try:
            if "key" in params:
                idx = self._key_combo.findText(params["key"])
                if idx >= 0:
                    self._key_combo.setCurrentIndex(idx)
            if "scale" in params:
                idx = self._scale_combo.findText(params["scale"])
                if idx >= 0:
                    self._scale_combo.setCurrentIndex(idx)
        finally:
            self._suspend_emit = False
