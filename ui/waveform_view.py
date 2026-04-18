"""Waveform view using pyqtgraph."""
from __future__ import annotations
import numpy as np
import pyqtgraph as pg
from PyQt6.QtCore import pyqtSignal
from ui import styles


class WaveformView(pg.PlotWidget):
    seek_requested = pyqtSignal(float)   # seconds

    def __init__(self, parent=None):
        super().__init__(parent)
        self._sr = 44100
        self._duration = 0.0

        self.setMinimumHeight(80)
        self.setMaximumHeight(120)
        self.showGrid(x=False, y=False)
        self.getAxis("left").hide()

        # Apply initial theme
        self._apply_theme()

        self._curve = self.plot(pen=pg.mkPen(styles.WAVEFORM, width=1))
        self._playhead = pg.InfiniteLine(
            pos=0, angle=90, pen=pg.mkPen("white", width=1, style=pg.QtCore.Qt.PenStyle.DashLine)
        )
        self.addItem(self._playhead)

        # Click to seek
        self.scene().sigMouseClicked.connect(self._on_click)

    def _apply_theme(self):
        """Apply current theme colors to the waveform view."""
        self.setBackground(styles.BG)
        self.getAxis("bottom").setTextPen(pg.mkPen(styles.TEXT_DIM))
        if hasattr(self, '_curve'):
            self._curve.setPen(pg.mkPen(styles.WAVEFORM, width=1))

    def update_theme(self):
        """Update theme colors when theme changes."""
        self._apply_theme()

    def set_audio(self, audio: np.ndarray, sr: int = 44100):
        import hashlib
        self._sr = sr
        self._duration = len(audio) / sr
        # Downsample for display
        max_pts = 4000
        step = max(1, len(audio) // max_pts)
        t = np.arange(0, len(audio), step) / sr
        a = audio[::step]

        # Calculate hash to verify data is changing
        hash_val = hashlib.md5(a[:100].tobytes()).hexdigest()[:8]
        print(f"[WAVEFORM] set_audio: hash={hash_val}, points={len(t)}, range=[{a.min():.3f}, {a.max():.3f}]")

        self._curve.setData(t, a)
        # Don't reset X range here - preserve user's zoom/scroll position
        # setXRange(0, self._duration, padding=0)
        self.setYRange(-1, 1, padding=0.05)
    
    def update_waveform_data(self, audio: np.ndarray, sr: int = 44100):
        """Update waveform data without resetting view range.
        
        This is for real-time preview during drag operations.
        """
        self._sr = sr
        self._duration = len(audio) / sr
        # Downsample for display
        max_pts = 4000
        step = max(1, len(audio) // max_pts)
        t = np.arange(0, len(audio), step) / sr
        a = audio[::step]
        self._curve.setData(t, a)
        # Only update Y range, preserve X range (zoom/scroll)
        self.setYRange(-1, 1, padding=0.05)

    def set_playhead(self, sec: float):
        self._playhead.setValue(sec)

    def zoom_to_fit(self):
        """Zoom the waveform to fit the full duration."""
        if self._duration > 0:
            self.setXRange(0, self._duration, padding=0)

    def _on_click(self, event):
        if event.button() == 1:
            pos = self.plotItem.vb.mapSceneToView(event.scenePos())
            sec = max(0.0, min(pos.x(), self._duration))
            self.seek_requested.emit(sec)
