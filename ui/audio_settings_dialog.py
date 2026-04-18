"""Audio settings dialog: device selection, buffer quality, export options."""
from __future__ import annotations
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QGroupBox,
    QLabel, QComboBox, QSlider, QPushButton, QCheckBox,
    QSpinBox, QWidget, QFormLayout
)
from ui.parameter_panel import LabeledSlider
from ui import styles


class AudioSettingsDialog(QDialog):
    """Comprehensive audio I/O settings dialog."""

    settings_applied = pyqtSignal(dict)

    def __init__(self, engine, parent=None):
        super().__init__(parent)
        self._engine = engine
        self.setWindowTitle("⚙️ 音频设置")
        self.setMinimumWidth(500)
        self.setModal(True)

        self._build_ui()
        self._load_current_settings()

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(12)

        # ── Playback Settings Group ──────────────────────────────────────
        playback_group = QGroupBox("播放设置")
        pb_layout = QFormLayout(playback_group)
        pb_layout.setSpacing(8)

        # Device selection
        device_row = QHBoxLayout()
        self._device_combo = QComboBox()
        self._device_combo.setToolTip("选择音频输出设备")
        device_row.addWidget(self._device_combo)

        refresh_btn = QPushButton("刷新")
        refresh_btn.setFixedWidth(60)
        refresh_btn.clicked.connect(self._refresh_devices)
        device_row.addWidget(refresh_btn)
        pb_layout.addRow("输出设备:", device_row)

        # Buffer quality
        self._buffer_combo = QComboBox()
        self._buffer_combo.addItems(["低延迟 (低稳定性)", "中等延迟 (推荐)", "高延迟 (高稳定性)"])
        self._buffer_combo.setToolTip("缓冲区大小：低延迟=响应快但可能爆音，高延迟=稳定但延迟大")
        pb_layout.addRow("缓冲质量:", self._buffer_combo)

        # Volume control
        self._volume_slider = LabeledSlider("音量", 0.0, 2.0, 1.0, "{:.2f}")
        self._volume_slider.setToolTip("播放音量：0=静音，1.0=原始音量，2.0=+6dB增益")
        pb_layout.addRow("", self._volume_slider)

        layout.addWidget(playback_group)

        # ── Export Settings Group ────────────────────────────────────────
        export_group = QGroupBox("导出设置")
        ex_layout = QFormLayout(export_group)
        ex_layout.setSpacing(8)

        # Format selection
        format_row = QHBoxLayout()
        self._format_combo = QComboBox()
        from utils.audio_utils import EXPORT_FORMATS
        for fmt_name, fmt_info in EXPORT_FORMATS.items():
            self._format_combo.addItem(f"{fmt_name} - {fmt_info['description']}", fmt_name)
        self._format_combo.currentTextChanged.connect(self._on_format_changed)
        format_row.addWidget(self._format_combo)
        ex_layout.addRow("导出格式:", format_row)

        # Bit depth / quality
        self._subtype_combo = QComboBox()
        self._subtype_combo.setToolTip("位深度/编码格式")
        ex_layout.addRow("位深度:", self._subtype_combo)

        # Quality slider for lossy formats
        self._quality_widget = QWidget()
        qual_layout = QHBoxLayout(self._quality_widget)
        qual_layout.setContentsMargins(0, 0, 0, 0)
        self._quality_slider = LabeledSlider("质量", 0.0, 1.0, 0.8, "{:.2f}")
        self._quality_slider.setToolTip("有损格式的压缩质量（仅OGG等有效）")
        qual_layout.addWidget(self._quality_slider)
        ex_layout.addRow("", self._quality_widget)

        # Options
        self._fade_check = QCheckBox("应用淡入淡出（防止爆音）")
        self._fade_check.setChecked(False)
        self._fade_check.setToolTip(
            "⚠️ 通常保持关闭\n"
            "vocoder 输出已经处理好边界\n"
            "仅在导出原始音频（未处理）时开启"
        )
        ex_layout.addRow("", self._fade_check)

        self._normalize_check = QCheckBox("归一化音量")
        self._normalize_check.setChecked(False)
        self._normalize_check.setToolTip(
            "⚠️ 通常保持关闭\n"
            "vocoder 已自动匹配原始音频 RMS 响度\n"
            "仅在需要统一多文件音量时开启"
        )
        ex_layout.addRow("", self._normalize_check)

        layout.addWidget(export_group)

        # ── Audio Info Display ──────────────────────────────────────────
        info_group = QGroupBox("当前设备信息")
        info_layout = QVBoxLayout(info_group)
        self._device_info_label = QLabel("未连接设备")
        self._device_info_label.setWordWrap(True)
        info_layout.addWidget(self._device_info_label)
        layout.addWidget(info_group)

        # ── Buttons ──────────────────────────────────────────────────────
        btn_layout = QHBoxLayout()
        btn_layout.addStretch()

        apply_btn = QPushButton("✓ 应用")
        apply_btn.setFixedWidth(100)
        apply_btn.clicked.connect(self._apply_and_close)
        btn_layout.addWidget(apply_btn)

        cancel_btn = QPushButton("✗ 取消")
        cancel_btn.setFixedWidth(100)
        cancel_btn.clicked.connect(self.reject)
        btn_layout.addWidget(cancel_btn)

        layout.addLayout(btn_layout)

        # Initialize subtypes for default format
        self._on_format_changed(self._format_combo.currentText())

    def _refresh_devices(self):
        """Refresh the list of available audio devices."""
        self._device_combo.clear()
        devices = self._engine.get_output_devices()

        if not devices:
            self._device_combo.addItem("未找到音频设备", None)
            return

        self._device_combo.addItem("默认设备", None)  # None = system default
        for dev in devices:
            label = f"{dev['name']}"
            if dev.get('is_default'):
                label += " [默认]"
            label += f" ({dev['channels']}ch, {int(dev['default_sr'])}Hz)"
            self._device_combo.addItem(label, dev['id'])

    def _on_format_changed(self, text: str):
        """Update subtype combo when format changes."""
        from utils.audio_utils import EXPORT_FORMATS, FORMAT_QUALITY_PRESETS

        self._subtype_combo.clear()

        # Get selected format name from data
        idx = self._format_combo.currentIndex()
        fmt_name = self._format_combo.itemData(idx)
        if fmt_name is None or fmt_name not in EXPORT_FORMATS:
            return

        fmt_info = EXPORT_FORMATS[fmt_name]
        for subtype in fmt_info['subtypes']:
            preset = FORMAT_QUALITY_PRESETS.get(subtype, {})
            display = subtype
            if 'use_case' in preset:
                display += f" ({preset['use_case']})"
            elif 'bitrate' in preset:
                display += f" ({preset['bitrate']})"
            self._subtype_combo.addItem(display, subtype)

        # Show/hide quality slider for lossy formats
        is_lossy = fmt_name == "OGG"
        self._quality_widget.setVisible(is_lossy)

    def _load_current_settings(self):
        """Load current settings into UI."""
        from core.audio_engine import AudioSettings

        settings = AudioSettings()

        # Refresh and set device
        self._refresh_devices()

        device_id = settings.get_device_id()
        for i in range(self._device_combo.count()):
            if self._device_combo.itemData(i) == device_id:
                self._device_combo.setCurrentIndex(i)
                break

        # Buffer quality
        buf_quality = settings.get_buffer_quality()
        buf_idx = {"low": 0, "medium": 1, "high": 2}.get(buf_quality, 1)
        self._buffer_combo.setCurrentIndex(buf_idx)

        # Volume
        vol = settings.get_volume()
        self._volume_slider.set_value(vol)

        # Export format
        fmt = settings.get_export_format()
        for i in range(self._format_combo.count()):
            if self._format_combo.itemData(i) == fmt:
                self._format_combo.setCurrentIndex(i)
                break

        # Subtype
        subtype = settings.get_export_subtype()
        for i in range(self._subtype_combo.count()):
            if self._subtype_combo.itemData(i) == subtype:
                self._subtype_combo.setCurrentIndex(i)
                break

        # Quality (for lossy formats)
        # Note: Could store this in settings if needed

        # Checkboxes
        self._fade_check.setChecked(settings.get_apply_fade())
        self._normalize_check.setChecked(settings.get_normalize())

        # Update device info
        self._update_device_info()

    def _update_device_info(self):
        """Display current device information."""
        try:
            current_device = self._engine.current_device
            sr = self._engine._sr
            blocksize = self._engine._blocksize
            latency_ms = blocksize / sr * 1000

            info_text = (
                f"<b>当前设备:</b> {current_device}<br>"
                f"<b>采样率:</b> {sr} Hz<br>"
                f"<b>缓冲区:</b> {blocksize} samples (~{latency_ms:.1f}ms)<br>"
                f"<b>声道数:</b> {self._engine._channels}"
            )
            self._device_info_label.setText(info_text)
        except Exception as e:
            self._device_info_label.setText(f"获取设备信息失败: {e}")

    def _apply_and_close(self):
        """Apply all settings and close dialog."""
        from core.audio_engine import AudioSettings

        settings = AudioSettings()

        # Apply device setting
        device_id = self._device_combo.currentData()
        self._engine.set_device(device_id)
        settings.set_device_id(device_id)

        # Apply buffer quality
        buf_map = {0: "low", 1: "medium", 2: "high"}
        buf_quality = buf_map.get(self._buffer_combo.currentIndex(), "medium")
        self._engine.set_buffer_quality(buf_quality)
        settings.set_buffer_quality(buf_quality)

        # Apply volume
        vol = self._volume_slider.get_value()
        self._engine.set_volume(vol)
        settings.set_volume(vol)

        # Apply export settings
        fmt = self._format_combo.currentData()
        if fmt:
            settings.set_export_format(fmt)

        subtype = self._subtype_combo.currentData()
        if subtype:
            settings.set_export_subtype(subtype)

        settings.set_apply_fade(self._fade_check.isChecked())
        settings.set_normalize(self._normalize_check.isChecked())

        # Emit signal with all settings
        all_settings = {
            'device_id': device_id,
            'buffer_quality': buf_quality,
            'volume': vol,
            'export_format': fmt,
            'export_subtype': subtype,
            'apply_fade': self._fade_check.isChecked(),
            'normalize': self._normalize_check.isChecked(),
        }
        self.settings_applied.emit(all_settings)

        self.accept()


class VolumeControl(QWidget):
    """Compact volume control widget for transport bar."""

    volume_changed = pyqtSignal(float)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedHeight(32)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(4, 0, 4, 0)
        layout.setSpacing(4)

        self._mute_btn = QPushButton("🔊")
        self._mute_btn.setFixedWidth(32)
        self._mute_btn.setCheckable(True)
        self._mute_btn.setToolTip("静音切换")
        self._mute_btn.clicked.connect(self._toggle_mute)
        layout.addWidget(self._mute_btn)

        self._slider = QSlider(Qt.Orientation.Horizontal)
        self._slider.setRange(0, 200)
        self._slider.setValue(100)
        self._slider.setMaximumWidth(120)
        self._slider.valueChanged.connect(self._on_slider_change)
        layout.addWidget(self._slider)

        self._value_label = QLabel("100%")
        self._value_label.setFixedWidth(40)
        self._value_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self._value_label)

        self._pre_mute_value = 100

    def _toggle_mute(self, checked: bool):
        if checked:
            self._pre_mute_value = self._slider.value()
            self._slider.setValue(0)
            self._mute_btn.setText("🔇")
        else:
            self._slider.setValue(self._pre_mute_value)
            self._mute_btn.setText("🔊")

    def _on_slider_change(self, value: int):
        vol = value / 100.0
        self._value_label.setText(f"{value}%")
        self._mute_btn.setChecked(value == 0)
        self.volume_changed.emit(vol)

    def set_volume(self, vol: float):
        self._slider.setValue(int(vol * 100))

    @property
    def volume(self) -> float:
        return self._slider.value() / 100.0
