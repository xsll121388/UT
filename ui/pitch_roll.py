"""Pitch roll: custom QWidget showing F0 curves + MIDI notes.

Coordinate system:
  X axis: time (seconds)
  Y axis: MIDI note number (MIDI_MIN-MIDI_MAX), higher = top

Modes:
  "draw_f0"    — mouse draws/edits the F0 target curve (original behaviour)
  "edit_notes" — mouse selects, moves, resizes MIDI notes
"""
from __future__ import annotations
import numpy as np
from PyQt6.QtCore import Qt, QRect, QPoint, QRectF, pyqtSignal, QSize
from PyQt6.QtGui import (QPainter, QColor, QPen, QBrush, QFont, QPixmap,
                          QWheelEvent, QMouseEvent, QKeyEvent, QCursor)
from PyQt6.QtWidgets import QWidget, QSizePolicy, QRubberBand, QScrollBar, QMenu

from ui import styles
from utils.audio_utils import MidiNote, StretchPoint

PIANO_WIDTH = 48      # px for piano key labels
NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
MIDI_MIN = 36         # C2
MIDI_MAX = 96         # C7
RESIZE_HANDLE_PX = 6  # px width of the resize grab zone on note right edge
STRETCH_PT_RADIUS = 5 # px radius for stretch control points


class PitchRoll(QWidget):
    f0_edited = pyqtSignal(np.ndarray)       # emitted when user draws new F0
    midi_notes_changed = pyqtSignal(list)    # emitted when notes are edited
    selection_changed = pyqtSignal(set)      # emitted when selected note set changes
    note_resized = pyqtSignal(int, float, float)  # idx, old_end_sec, new_end_sec
    note_cut = pyqtSignal(int, float)             # original note idx, cut_sec
    seek_requested = pyqtSignal(float)            # seconds - emitted when clicking time ruler
    stretch_changed = pyqtSignal(bool)            # emitted when stretch points change (is_timing_edit)
    stretch_preview = pyqtSignal()                # emitted during drag for real-time waveform preview (lightweight)
    history_restored = pyqtSignal(str, str)       # direction, action_type
    
    # Grid snap settings
    grid_snap_enabled = False  # Whether grid snapping is enabled
    snap_resolution = 0.25     # Grid resolution in seconds (0.25 = 1/16 note at 60 BPM)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setMinimumHeight(200)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)

        self._f0_original: np.ndarray = np.array([])
        self._f0_target: np.ndarray = np.array([])
        self._sr: int = 44100
        self._fps: int = 100
        self._total_samples: int = 0

        # View state
        self._view_start: float = 0.0    # seconds
        self._view_end: float = 10.0
        self._midi_min: float = float(MIDI_MIN)
        self._midi_max: float = float(MIDI_MAX)

        # Playhead
        self._playhead_sec: float = 0.0

        # F0 drawing state
        self._drawing = False
        self._last_draw_point: tuple[int, float] | None = None
        self._f0_draw_snapshot: np.ndarray | None = None
        self._f0_draw_changed: bool = False

        # MIDI note layer
        self._midi_notes: list[MidiNote] = []
        self._selected: set[int] = set()

        # Edit mode
        self._mode: str = "draw_f0"

        # Note drag state
        self._drag_op: str | None = None          # "move" | "resize" | "rubberband" | "stretch_pt"
        self._drag_start: QPoint | None = None
        self._drag_note_idx: int | None = None
        self._resize_old_end: float = 0.0         # end_sec snapshot at resize press
        # snapshot: list of (idx, start_sec, end_sec, pitch) for all selected notes
        self._drag_snapshots: list[tuple] = []

        # Cut mode state
        self._cut_hover_note_idx: int | None = None
        self._cut_hover_sec: float | None = None

        # Stretch point drag state
        self._stretch_note_idx: int | None = None
        self._stretch_pt_idx: int | None = None
        self._stretch_pt_start_pos: float = 0.0
        self._last_stretch_emit_time: float = 0.0  # Throttle stretch updates

        # Waveform background
        self._waveform_audio: np.ndarray = np.array([], dtype=np.float32)
        self._waveform_sr: int = 44100

        # Rubber-band widget
        self._rubberband: QRubberBand | None = None

        # Undo / redo stacks — each entry is (f0_target_copy, midi_notes_copy, action_type)
        self._undo_stack: list[tuple] = []
        self._redo_stack: list[tuple] = []

        # Scrollbars (children, positioned in resizeEvent)
        SB = QScrollBar
        self._hbar = SB(Qt.Orientation.Horizontal, self)
        self._vbar = SB(Qt.Orientation.Vertical,   self)
        self._hbar.setRange(0, 1000)
        self._vbar.setRange(0, 1000)
        self._hbar.setSingleStep(20)
        self._vbar.setSingleStep(20)
        self._hbar.setPageStep(200)
        self._vbar.setPageStep(200)
        self._sb_updating = False   # guard against recursive updates
        self._hbar.valueChanged.connect(self._on_hbar)
        self._vbar.valueChanged.connect(self._on_vbar)

        self.setMouseTracking(True)

        # Performance optimization: cache for background layers
        self._bg_cache_pixmap = None
        self._bg_cache_valid = False
        self._last_view_state = None
        self._last_stretch_state = None  # Track stretch points state for cache invalidation

    # ── public API ────────────────────────────────────────────────────────

    def set_data(self, f0_original: np.ndarray, f0_target: np.ndarray,
                 total_samples: int, sr: int = 44100):
        self._f0_original = f0_original.astype(np.float32)
        self._f0_target = f0_target.astype(np.float32)
        self._total_samples = total_samples
        self._sr = sr
        duration = total_samples / sr
        self._view_start = 0.0
        self._view_end = min(duration, 30.0)
        self.update()

    def set_playhead(self, sec: float):
        self._playhead_sec = sec
        self.update()

    def update_f0_target(self, f0_target: np.ndarray):
        self._f0_target = f0_target.astype(np.float32)
        self.update()

    def set_midi_notes(self, notes: list[MidiNote]) -> None:
        self._midi_notes = list(notes)
        self._selected.clear()
        self.update()
        self.selection_changed.emit(set())
        # Emit midi_notes_changed to trigger real-time rendering
        self.midi_notes_changed.emit(self._midi_notes)

    def get_midi_notes(self) -> list[MidiNote]:
        return list(self._midi_notes)

    def set_mode(self, mode: str) -> None:
        """Switch between 'draw_f0', 'edit_notes', and 'cut'."""
        old_mode = self._mode
        self._mode = mode
        self._selected.clear()
        if mode == "cut":
            self.setCursor(Qt.CursorShape.CrossCursor)
        else:
            self.unsetCursor()
        self.update()
        self.selection_changed.emit(set())
        
        # Show mode switch hint
        mode_names = {
            "edit_notes": "音符编辑",
            "cut": "切割模式",
            "draw_f0": "绘制音高",
        }
        mode_name = mode_names.get(mode, mode)
        if hasattr(self, '_status'):
            self._status.showMessage(f"已切换到 {mode_name} 模式", 2000)

    def set_waveform(self, audio: np.ndarray, sr: int) -> None:
        """Set audio waveform to display as background in the pitch roll."""
        self._waveform_audio = audio
        self._waveform_sr = sr
        self._total_samples = len(audio)
        self._sr = sr
        self.update()

    # ── coordinate helpers ────────────────────────────────────────────────

    def _sec_to_x(self, sec: float) -> float:
        roll_w = self._canvas_w() - PIANO_WIDTH
        t_range = self._view_end - self._view_start
        if t_range <= 0:
            return float(PIANO_WIDTH)
        return PIANO_WIDTH + (sec - self._view_start) / t_range * roll_w

    def _x_to_sec(self, x: float) -> float:
        roll_w = self._canvas_w() - PIANO_WIDTH
        t_range = self._view_end - self._view_start
        if roll_w <= 0:
            return self._view_start
        return self._view_start + (x - PIANO_WIDTH) / roll_w * t_range

    def _midi_to_y(self, midi: float) -> float:
        h = self._canvas_h()
        m_range = self._midi_max - self._midi_min
        if m_range <= 0:
            return float(h)
        return h - (midi - self._midi_min) / m_range * h

    def _y_to_midi(self, y: float) -> int:
        h = self._canvas_h()
        m_range = self._midi_max - self._midi_min
        if h <= 0:
            return int(self._midi_min)
        return int(round(self._midi_min + (h - y) / h * m_range))

    def _note_rect(self, note: MidiNote) -> QRect:
        RULER_HEIGHT = 20
        x1 = int(self._sec_to_x(note.start_sec))
        x2 = int(self._sec_to_x(note.end_sec))
        y  = RULER_HEIGHT + int(self._midi_to_y(note.pitch + 1))
        nh = int(self._midi_to_y(note.pitch)) - int(self._midi_to_y(note.pitch + 1))
        return QRect(x1, y, max(x2 - x1, 4), max(nh, 4))

    _SB_SIZE = 12  # scrollbar thickness px

    def _canvas_w(self) -> int:
        return self.width() - self._SB_SIZE

    def _canvas_h(self) -> int:
        return self.height() - self._SB_SIZE

    # ── scrollbars ────────────────────────────────────────────────────────

    _SB_SCALE = 1000  # scrollbar integer range

    def _sync_scrollbars(self):
        """Push current view state into the scrollbars (no signal loop)."""
        if self._sb_updating:
            return
        self._sb_updating = True

        # Use audio duration if available, otherwise derive from MIDI notes
        total = self._total_duration()

        # Horizontal — handle size reflects zoom level, always visible
        view_span = self._view_end - self._view_start
        h_ratio = max(0.01, min(view_span / total, 0.99))   # 1%–99%
        h_page  = max(10, int(h_ratio * self._SB_SCALE))
        h_max   = self._SB_SCALE - h_page                   # always >= 1
        h_val   = int(self._view_start / total * self._SB_SCALE)
        h_val   = max(0, min(h_val, h_max))
        self._hbar.setPageStep(h_page)
        self._hbar.setMaximum(h_max)
        self._hbar.setValue(h_val)

        # Vertical
        midi_range = 127.0
        v_ratio = max(0.01, min((self._midi_max - self._midi_min) / midi_range, 0.99))
        v_page  = max(10, int(v_ratio * self._SB_SCALE))
        v_max   = self._SB_SCALE - v_page
        v_val   = int((127.0 - self._midi_max) / midi_range * self._SB_SCALE)
        v_val   = max(0, min(v_val, v_max))
        self._vbar.setPageStep(v_page)
        self._vbar.setMaximum(v_max)
        self._vbar.setValue(v_val)

        self._sb_updating = False

    def _total_duration(self) -> float:
        if self._total_samples > 0 and self._sr > 0:
            return self._total_samples / self._sr
        if self._midi_notes:
            return max(n.end_sec for n in self._midi_notes) * 1.1
        return max(self._view_end, 10.0)

    def _on_hbar(self, value: int):
        if self._sb_updating:
            return
        total = self._total_duration()
        span  = self._view_end - self._view_start
        self._view_start = value / self._SB_SCALE * total
        self._view_end   = self._view_start + span
        super().update()

    def _on_vbar(self, value: int):
        if self._sb_updating:
            return
        span = self._midi_max - self._midi_min
        # value=0 → top (midi_max=127), value=max → bottom
        self._midi_max = 127.0 - value / self._SB_SCALE * 127.0
        self._midi_min = self._midi_max - span
        super().update()

    def resizeEvent(self, event):
        sb = self._SB_SIZE
        w, h = self.width(), self.height()
        self._hbar.setGeometry(PIANO_WIDTH, h - sb, w - PIANO_WIDTH - sb, sb)
        self._vbar.setGeometry(w - sb, 0, sb, h - sb)
        super().resizeEvent(event)

    def update(self):
        self._sync_scrollbars()
        super().update()

    # ── painting ──────────────────────────────────────────────────────────

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        w, h = self._canvas_w(), self._canvas_h()
        roll_w = w - PIANO_WIDTH

        # Check if view state changed (zoom/pan)
        current_view_state = (self._view_start, self._view_end, self._midi_min, self._midi_max, w, h)
        view_changed = (self._last_view_state != current_view_state)
        
        # Check if stretch state changed (for real-time waveform preview)
        current_stretch_state = tuple(
            (i, [(sp.position, sp.orig_position) for sp in note.stretch_points])
            for i, note in enumerate(self._midi_notes)
            if note.stretch_points
        )
        stretch_changed = (self._last_stretch_state != current_stretch_state)

        # Use cached background if view hasn't changed and stretch hasn't changed
        if self._bg_cache_valid and not view_changed and not stretch_changed and self._bg_cache_pixmap:
            p.drawPixmap(0, 0, self._bg_cache_pixmap)
        else:
            # Render background layers to cache
            self._bg_cache_pixmap = QPixmap(self.width(), self.height())
            cache_painter = QPainter(self._bg_cache_pixmap)
            cache_painter.setRenderHint(QPainter.RenderHint.Antialiasing)

            # Background (full widget including scrollbar corner) - use dynamic theme color
            from ui import styles
            cache_painter.fillRect(0, 0, self.width(), self.height(), QColor(styles.BG))

            # Time ruler at top (FL Studio style)
            self._draw_time_ruler(cache_painter, roll_w)

            # Piano keys (left strip)
            self._draw_piano(cache_painter, h)

            # Grid lines (semitone rows)
            self._draw_grid(cache_painter, roll_w, h)

            # Waveform background (before notes)
            self._draw_waveform(cache_painter, roll_w, h)

            cache_painter.end()

            # Draw cached background
            p.drawPixmap(0, 0, self._bg_cache_pixmap)

            # Mark cache as valid
            self._bg_cache_valid = True
            self._last_view_state = current_view_state
            self._last_stretch_state = current_stretch_state

        # Draw dynamic layers (MIDI notes, F0 curves, playhead) directly
        # MIDI notes layer
        self._draw_midi_notes(p)

        # F0 curves
        self._draw_f0_curves(p, roll_w, h)
        RULER_HEIGHT = 20
        px = int(self._sec_to_x(self._playhead_sec))
        if PIANO_WIDTH <= px <= w:
            p.setPen(QPen(QColor("#ffffff"), 1))
            p.drawLine(px, RULER_HEIGHT, px, h)

        # Cut mode preview line
        if self._mode == "cut" and self._cut_hover_sec is not None:
            cx = int(self._sec_to_x(self._cut_hover_sec))
            if PIANO_WIDTH <= cx <= w:
                valid_cut = False
                if self._cut_hover_note_idx is not None and 0 <= self._cut_hover_note_idx < len(self._midi_notes):
                    note = self._midi_notes[self._cut_hover_note_idx]
                    valid_cut = note.start_sec + 0.05 < self._cut_hover_sec < note.end_sec - 0.05

                if valid_cut:
                    # Valid cut: bright red dashed line
                    p.setPen(QPen(QColor("#ff4444"), 2, Qt.PenStyle.DashLine))
                else:
                    # Invalid cut: semi-transparent gray with pattern
                    p.setPen(QPen(QColor("#888888"), 1, Qt.PenStyle.DotLine))
                p.drawLine(cx, RULER_HEIGHT, cx, h)
                
                # Draw cut icon at top only for valid cuts
                if valid_cut:
                    p.setPen(QPen(QColor("#ff4444")))
                    font = QFont()
                    font.setPointSize(10)
                    font.setBold(True)
                    p.setFont(font)
                    # Use Unicode scissors symbol (more reliable than emoji)
                    p.drawText(cx - 5, RULER_HEIGHT - 2, "✄")

        # Grid snap indicator
        if self.grid_snap_enabled:
            p.setPen(QPen(QColor("#00ff00"), 2))
            font = QFont()
            font.setPointSize(8)
            font.setBold(True)
            p.setFont(font)
            p.drawText(PIANO_WIDTH + 10, RULER_HEIGHT + 15, f"SNAP: {self.snap_resolution}s")

        # Current mode indicator
        mode_colors = {
            "edit_notes": QColor("#4ade80"),  # Green
            "cut": QColor("#f87171"),         # Red
            "draw_f0": QColor("#60a5fa"),     # Blue
        }
        mode_labels = {
            "edit_notes": "音符编辑",
            "cut": "切割模式",
            "draw_f0": "绘制音高",
        }
        mode = getattr(self, '_mode', 'edit_notes')
        mode_color = mode_colors.get(mode, QColor("#ffffff"))
        mode_label = mode_labels.get(mode, "未知模式")
        
        # Draw mode badge in top-right corner
        badge_text = f"模式：{mode_label}"
        p.setPen(QPen(mode_color, 2))
        p.setBrush(QColor(mode_color.lighter(120)))
        font = QFont()
        font.setPointSize(9)
        font.setBold(True)
        p.setFont(font)
        text_rect = p.fontMetrics().boundingRect(badge_text)
        badge_x = w - text_rect.width() - 15
        badge_y = RULER_HEIGHT + 5
        p.drawRoundedRect(badge_x, badge_y, text_rect.width() + 16, text_rect.height() + 8, 4, 4)
        p.setPen(QColor("#000000"))
        p.drawText(badge_x + 8, badge_y + text_rect.height() + 2, badge_text)

    def _draw_waveform(self, p: QPainter, roll_w: int, h: int):
        """Draw audio waveform as a subtle background behind MIDI notes.
        
        When stretch control points exist on MIDI notes, the waveform
        is redrawn in real-time to reflect the stretched segments.
        """
        if len(self._waveform_audio) == 0 or roll_w <= 0:
            return
        RULER_HEIGHT = 20
        canvas_h = h - RULER_HEIGHT
        if canvas_h <= 0:
            return

        audio = self._waveform_audio
        sr = self._waveform_sr
        total_samples = len(audio)

        # Build a map of note regions with stretch points for real-time preview
        stretch_regions = []
        for note in self._midi_notes:
            if note.stretch_points:
                stretch_regions.append((note.start_sec, note.end_sec, note))

        waveform_color = QColor("#4a4a7a")
        stretch_color = QColor("#7a7aaa")

        view_start = self._view_start
        view_end = self._view_end
        view_range = view_end - view_start

        half = canvas_h // 2
        mid_y = RULER_HEIGHT + half

        for px in range(roll_w):
            x = PIANO_WIDTH + px
            t_start = view_start + px / roll_w * view_range
            t_end   = view_start + (px + 1) / roll_w * view_range
            
            # Check if this pixel falls within a stretched note region
            in_stretch = False
            remap_start = t_start
            remap_end = t_end
            for ns, ne, note in stretch_regions:
                if ns <= t_start <= ne:
                    in_stretch = True
                    remap_start = self._remap_time_for_stretch(t_start, note)
                    remap_end = self._remap_time_for_stretch(min(t_end, ne), note)
                    break
            
            s0 = int(remap_start * sr)
            s1 = int(remap_end   * sr)
            s0 = max(0, min(s0, total_samples - 1))
            s1 = max(s0 + 1, min(s1, total_samples))
            chunk = audio[s0:s1]
            if len(chunk) == 0:
                continue
            amp = float(np.max(np.abs(chunk)))
            extent = int(amp * half)
            
            if in_stretch:
                p.setPen(QPen(stretch_color, 1))
            else:
                p.setPen(QPen(waveform_color, 1))
            p.drawLine(x, mid_y - extent, x, mid_y + extent)

    def _remap_time_for_stretch(self, t: float, note: MidiNote) -> float:
        """Remap display time to original audio time based on stretch points.
        
        When a stretch point is dragged, the display space changes:
        - Left segment: compressed/stretched based on position shift
        - Right segment: inversely stretched/compressed to preserve total duration
        
        This maps from the current display position back to the original
        audio position so the waveform preview shows the stretched result.
        """
        if not note.stretch_points:
            return t
        
        note_dur = note.end_sec - note.start_sec
        if note_dur <= 0:
            return t
        
        rel_pos = (t - note.start_sec) / note_dur
        rel_pos = max(0.0, min(1.0, rel_pos))
        
        # Build current segment boundaries
        cur_boundaries = [0.0]
        for sp in note.stretch_points:
            cur_boundaries.append(sp.position)
        cur_boundaries.append(1.0)
        
        # Build original segment boundaries
        orig_boundaries = [0.0]
        for sp in note.stretch_points:
            orig_boundaries.append(sp.orig_position)
        orig_boundaries.append(1.0)
        
        # Find which current segment this position falls in
        for i in range(len(cur_boundaries) - 1):
            cur_start = cur_boundaries[i]
            cur_end = cur_boundaries[i + 1]
            
            if cur_start <= rel_pos <= cur_end:
                cur_dur = cur_end - cur_start
                orig_dur = orig_boundaries[i + 1] - orig_boundaries[i]
                
                if cur_dur > 0:
                    pos_in_seg = (rel_pos - cur_start) / cur_dur
                else:
                    pos_in_seg = 0.0
                
                # Map back to original position
                orig_pos = orig_boundaries[i] + pos_in_seg * orig_dur
                return note.start_sec + orig_pos * note_dur
        
        return t

    def _draw_time_ruler(self, p: QPainter, roll_w: int):
        """
        Draw time ruler at the top (FL Studio style).
        
        Shows time markers with:
        - Major grid lines every 1 second (with labels)
        - Minor grid lines every 0.5 seconds
        - Subtle grid lines every 0.25 seconds
        """
        RULER_HEIGHT = 20
        start_sec = self._view_start
        end_sec = self._view_end
        
        # Calculate visible time range
        duration = end_sec - start_sec
        if duration <= 0:
            return
        
        # Determine grid spacing based on zoom level
        if duration < 1.0:
            major_step = 0.25
            minor_step = 0.1
        elif duration < 2.0:
            major_step = 0.5
            minor_step = 0.25
        elif duration < 5.0:
            major_step = 1.0
            minor_step = 0.5
        else:
            major_step = 2.0
            minor_step = 1.0
        
        # Find start position (snap to grid)
        first_major = int(start_sec / major_step) * major_step
        
        font = QFont()
        font.setPointSize(7)
        p.setFont(font)
        
        # Draw vertical grid lines and labels
        current = first_major
        while current < end_sec:
            x = int(self._sec_to_x(current))
            if x < PIANO_WIDTH:
                current += major_step
                continue
            
            # Determine line type
            is_major = (abs(current % major_step) < 0.001)
            is_minor = (abs(current % minor_step) < 0.001) and not is_major
            
            if is_major:
                # Major grid line (1 second)
                color = QColor("#4a4a8a")
                line_width = 2
                
                # Draw line
                p.fillRect(x - line_width // 2, 0, line_width, RULER_HEIGHT, color)
                
                # Draw time label
                p.setPen(QColor("#8a8aaa"))
                font.setBold(True)
                p.setFont(font)
                time_label = f"{current:.1f}s"
                p.drawText(x + 3, RULER_HEIGHT - 3, time_label)
                
            elif is_minor:
                # Minor grid line (0.5 seconds)
                color = QColor("#2a2a4a")
                line_width = 1
                p.fillRect(x - line_width // 2, 0, line_width, RULER_HEIGHT // 2, color)
            else:
                # Subtle grid line (0.25 seconds)
                color = QColor("#1a1a2e")
                line_width = 1
                p.fillRect(x - line_width // 2, 0, line_width, RULER_HEIGHT // 4, color)
            
            current += minor_step
        
        # Draw ruler bottom border with improved contrast
        p.setPen(QPen(QColor("#5a5a7e"), 1))  # Brighter border
        p.drawLine(PIANO_WIDTH, RULER_HEIGHT, PIANO_WIDTH + roll_w, RULER_HEIGHT)

    def _draw_piano(self, p: QPainter, h: int):
        """
        Draw the piano keyboard on the left side.

        Visual design:
        - White keys: lighter rectangles
        - Black keys: darker rectangles
        - C notes: labeled with octave number
        """
        from ui import styles
        RULER_HEIGHT = 20
        m_range = self._midi_max - self._midi_min
        font = QFont()
        font.setPointSize(7)
        p.setFont(font)

        for midi in range(int(self._midi_min), int(self._midi_max) + 1):
            # Calculate y-positions to match grid lines exactly
            y_top = int(self._midi_to_y(midi + 1))
            y_bot = int(self._midi_to_y(midi))
            note_in_oct = midi % 12
            is_black = note_in_oct in (1, 3, 6, 8, 10)

            # Piano key colors - use dynamic theme
            if is_black:
                color = QColor(styles.PIANO_BLACK)
            else:
                color = QColor(styles.PIANO_WHITE)

            # Draw piano key - ensure height is at least 1 pixel
            # Use RULER_HEIGHT offset to align with grid lines
            key_height = max(y_bot - y_top, 1)
            p.fillRect(0, RULER_HEIGHT + y_top, PIANO_WIDTH - 2, key_height, color)

            # Label C notes
            if note_in_oct == 0:  # C notes
                octave = midi // 12 - 1
                label = f"C{octave}"

                # Draw label with better visibility
                p.setPen(QColor(styles.TEXT))
                font.setBold(True)
                p.setFont(font)
                p.drawText(2, RULER_HEIGHT + y_top, PIANO_WIDTH - 4, key_height,
                           Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft,
                           label)

    def _draw_grid(self, p: QPainter, roll_w: int, h: int):
        """
        Draw the pitch grid with improved visual hierarchy.

        Grid lines are drawn for each MIDI note with:
        - C notes (every 12 semitones): brighter, thicker line
        - Other notes: standard grid lines
        - Black key positions: slightly dimmer

        Grid lines are aligned with piano key centers.
        """
        from ui import styles
        RULER_HEIGHT = 20

        # Draw grid line for each MIDI note, aligned with piano key CENTER
        for midi in range(int(self._midi_min), int(self._midi_max) + 1):
            # Calculate y-position for piano key CENTER
            y_top = self._midi_to_y(midi + 1)
            y_bottom = self._midi_to_y(midi)
            y_center = int((y_top + y_bottom) / 2)

            note_in_oct = midi % 12
            octave = midi // 12 - 1

            # Visual hierarchy for grid lines - use dynamic theme
            is_c_note = (note_in_oct == 0)
            is_black_key = note_in_oct in (1, 3, 6, 8, 10)

            if is_c_note:
                # C notes: prominent grid line (brighter, thicker)
                color = QColor(styles.GRID_MAJOR)
                line_width = 2
            elif is_black_key:
                # Black key positions: subtle grid line
                color = QColor(styles.GRID_SUBTLE)
                line_width = 1
            else:
                # White key positions (non-C): standard grid line
                color = QColor(styles.GRID_MINOR)
                line_width = 1

            # Draw grid line aligned with piano key CENTER
            p.fillRect(PIANO_WIDTH, RULER_HEIGHT + y_center, roll_w, line_width, color)

            # Draw note label for C notes
            if is_c_note:
                p.setPen(QColor(styles.TEXT_DIM))
                font = QFont()
                font.setPointSize(7)
                font.setBold(True)
                p.setFont(font)
                label = f"C{octave}"
                p.drawText(PIANO_WIDTH + 5, RULER_HEIGHT + int(y_center) + 5, label)

    def _draw_midi_notes(self, p: QPainter):
        if not self._midi_notes:
            return
        font = QFont()
        font.setPointSize(9)
        font.setBold(True)
        p.setFont(font)

        # Modern color scheme with better contrast
        base_color = QColor("#10b981")   # Emerald green
        sel_color  = QColor("#3b82f6")   # Bright blue for selected
        lyric_color = QColor("#ffffff")  # White text for better readability

        for i, note in enumerate(self._midi_notes):
            rect = self._note_rect(note)
            if rect.right() < PIANO_WIDTH or rect.left() > self.width():
                continue  # outside view

            selected = i in self._selected
            fill = sel_color if selected else base_color
            fill_alpha = QColor(fill)
            fill_alpha.setAlpha(255)  # Fully opaque

            # Draw note segments with stretch visualization
            if note.stretch_points:
                self._draw_stretched_note(p, note, rect, fill_alpha, selected, lyric_color)
            else:
                # Draw subtle shadow for depth
                shadow_rect = rect.adjusted(2, 2, 2, 2)
                p.fillRect(shadow_rect, QColor(0, 0, 0, 30))

                # Draw main note body
                p.fillRect(rect, fill_alpha)

                # Border with better visibility
                border_color = QColor("#60a5fa") if selected else QColor("#059669")
                border_pen = QPen(border_color, 2)
                p.setPen(border_pen)
                p.drawRect(rect)

                # Lyric text with shadow for readability
                if note.lyric:
                    # Text shadow
                    p.setPen(QColor(0, 0, 0, 100))
                    p.drawText(rect.adjusted(1, 1, 1, 1), Qt.AlignmentFlag.AlignCenter, note.lyric)
                    # Main text
                    p.setPen(lyric_color)
                    p.drawText(rect, Qt.AlignmentFlag.AlignCenter, note.lyric)

            # Draw stretch control points (confined to note boundaries)
            # Save current brush state
            old_brush = p.brush()
            for sp in note.stretch_points:
                sp_x = rect.left() + int(sp.position * rect.width())
                orig_x = rect.left() + int(sp.orig_position * rect.width())
                sp_y = rect.top()

                # Draw original position marker (faint vertical line, confined to note)
                if abs(sp.position - sp.orig_position) > 0.01:
                    p.setPen(QPen(QColor("#777799"), 1, Qt.PenStyle.DotLine))  # Brighter
                    p.drawLine(orig_x, rect.top(), orig_x, rect.bottom())

                # Draw current position line (subtle white color, confined to note)
                p.setPen(QPen(QColor("#dddddd"), 1, Qt.PenStyle.DashLine))  # Brighter
                p.drawLine(sp_x, rect.top(), sp_x, rect.bottom())

                # Draw control point circle at top (subtle white color)
                p.setPen(QPen(QColor("#ffffff"), 2))  # Thicker for better visibility
                p.setBrush(QBrush(QColor("#ffffff")))  # White: uniform color
                p.drawEllipse(QPoint(sp_x, sp_y), STRETCH_PT_RADIUS, STRETCH_PT_RADIUS)
            # Restore brush state
            p.setBrush(old_brush)

    def _draw_stretched_note(self, p: QPainter, note: MidiNote, rect: QRect,
                              fill_color: QColor, selected: bool, lyric_color: QColor):
        """Draw a note with stretch segments using uniform color.

        Each stretch point divides the note into segments, but all segments
        use the same fill color for a clean, consistent appearance.
        """
        if not note.stretch_points:
            return

        # Draw subtle shadow for depth
        shadow_rect = rect.adjusted(2, 2, 2, 2)
        p.fillRect(shadow_rect, QColor(0, 0, 0, 30))

        # Build segment boundaries from current positions
        boundaries = [0.0]
        for sp in note.stretch_points:
            boundaries.append(sp.position)
        boundaries.append(1.0)

        # Draw all segments with uniform color (no stretch-based coloring)
        for seg_i in range(len(boundaries) - 1):
            seg_start = boundaries[seg_i]
            seg_end = boundaries[seg_i + 1]

            # Calculate segment rect
            seg_x1 = rect.left() + int(seg_start * rect.width())
            seg_x2 = rect.left() + int(seg_end * rect.width())
            seg_rect = QRect(seg_x1, rect.top(), max(seg_x2 - seg_x1, 1), rect.height())

            # Use uniform fill color for all segments
            p.fillRect(seg_rect, fill_color)

        # Border with better visibility
        border_color = QColor("#60a5fa") if selected else QColor("#059669")
        border_pen = QPen(border_color, 2)
        p.setPen(border_pen)
        p.setBrush(Qt.BrushStyle.NoBrush)
        p.drawRect(rect)

        # Lyric text with shadow for readability
        if note.lyric:
            # Text shadow
            p.setPen(QColor(0, 0, 0, 100))
            p.drawText(rect.adjusted(1, 1, 1, 1), Qt.AlignmentFlag.AlignCenter, note.lyric)
            # Main text
            p.setPen(lyric_color)
            p.drawText(rect, Qt.AlignmentFlag.AlignCenter, note.lyric)

        # Reset brush to ensure it doesn't affect next note
        p.setBrush(Qt.BrushStyle.NoBrush)

    def _draw_f0_curves(self, p: QPainter, roll_w: int, h: int):
        RULER_HEIGHT = 20
        
        def curve(f0_arr, color):
            if len(f0_arr) == 0:
                return
            pen = QPen(QColor(color), 2)
            pen.setCapStyle(Qt.PenCapStyle.RoundCap)
            p.setPen(pen)
            duration = self._total_samples / self._sr if self._sr > 0 else 1.0
            prev = None
            for fi, hz in enumerate(f0_arr):
                if hz <= 0:
                    prev = None
                    continue
                t = fi / self._fps
                if t < self._view_start or t > self._view_end:
                    prev = None
                    continue
                midi = 12 * np.log2(hz / 440.0) + 69
                x = int(self._sec_to_x(t))
                # Align to MIDI note center (grid line position)
                y_top = self._midi_to_y(midi + 1)
                y_bottom = self._midi_to_y(midi)
                y_center = (y_top + y_bottom) / 2
                y = RULER_HEIGHT + int(y_center)
                if prev:
                    p.drawLine(prev[0], prev[1], x, y)
                prev = (x, y)

        curve(self._f0_original, styles.F0_ORIGINAL)
        curve(self._f0_target,   styles.F0_TARGET)

    # ── mouse events ──────────────────────────────────────────────────────

    def mousePressEvent(self, event: QMouseEvent):
        RULER_HEIGHT = 20
        
        # Check if clicking in time ruler area (top 20px)
        if event.position().y() <= RULER_HEIGHT and event.position().x() >= PIANO_WIDTH:
            # Clicked in time ruler - seek to that position
            sec = self._x_to_sec(event.position().x())
            self.seek_requested.emit(sec)
            return
        
        # Handle right-click context menu
        if event.button() == Qt.MouseButton.RightButton:
            if self._mode == "edit_notes":
                self._show_note_context_menu(event)
            elif self._mode == "cut":
                # Show cut mode help menu
                self._show_cut_help_menu(event)
            elif self._mode == "draw_f0":
                # Show F0 draw mode help menu
                self._show_f0_help_menu(event)
            return
        
        if self._mode == "edit_notes":
            self._notes_press(event)
        elif self._mode == "cut":
            self._cut_press(event)
        else:
            self._f0_press(event)

    def mouseMoveEvent(self, event: QMouseEvent):
        if self._mode == "edit_notes":
            self._notes_move(event)
        elif self._mode == "cut":
            self._cut_move(event)
        else:
            self._f0_move(event)

    def mouseReleaseEvent(self, event: QMouseEvent):
        if self._mode == "edit_notes":
            self._notes_release(event)
        elif self._mode == "cut":
            self._cut_release(event)
        else:
            self._f0_release(event)

    def wheelEvent(self, event: QWheelEvent):
        delta = event.angleDelta().y()
        mods  = event.modifiers()
        factor = 1.15 if delta > 0 else 1 / 1.15

        ctrl  = bool(mods & Qt.KeyboardModifier.ControlModifier)
        shift = bool(mods & Qt.KeyboardModifier.ShiftModifier)
        alt   = bool(mods & Qt.KeyboardModifier.AltModifier)

        if alt:
            # Alt+scroll: Adjust time ruler height (future feature)
            # Show hint to user if status bar is available
            if hasattr(self, '_status'):
                self._status.showMessage("Alt+滚轮：调整时间标尺高度（功能开发中）", 2000)
            event.ignore()
            return
        if ctrl:
            # Ctrl + scroll → vertical (pitch) zoom
            self._zoom_midi(factor)
            if hasattr(self, '_status'):
                self._status.showMessage("Ctrl+滚轮：垂直缩放（音高范围）", 1500)
        elif shift:
            # Shift + scroll → horizontal pan
            step = (self._view_end - self._view_start) * (0.15 if delta > 0 else -0.15)
            self._pan_time(-step)
            if hasattr(self, '_status'):
                self._status.showMessage("Shift+滚轮：水平平移", 1500)
        else:
            # Plain scroll → horizontal (time) zoom around cursor
            roll_w = self._canvas_w() - PIANO_WIDTH
            if roll_w > 0:
                mx = event.position().x() - PIANO_WIDTH
                pivot = self._view_start + mx / roll_w * (self._view_end - self._view_start)
            else:
                pivot = (self._view_start + self._view_end) / 2
            total = self._total_duration()
            span  = (self._view_end - self._view_start) / factor
            span  = max(0.5, span)          # never collapse
            new_start = max(0.0, pivot - (pivot - self._view_start) / factor)
            new_end   = new_start + span
            if total > 0 and new_end > total:
                new_end   = total
                new_start = max(0.0, total - span)
            self._view_start = new_start
            self._view_end   = new_end
            if hasattr(self, '_status'):
                self._status.showMessage("滚轮：水平缩放（时间范围）", 1500)
        self.update()

    def keyPressEvent(self, event: QKeyEvent):
        key  = event.key()
        mods = event.modifiers()

        # ── Delete selected notes ──────────────────────────────────────────
        if self._mode == "edit_notes" and key == Qt.Key.Key_Delete:
            if self._selected:
                self._push_undo("delete_notes")
                self._midi_notes = [n for i, n in enumerate(self._midi_notes)
                                    if i not in self._selected]
                self._selected.clear()
                self.update()
                self.midi_notes_changed.emit(self._midi_notes)
                self.selection_changed.emit(set())
            return
        
        # Select All (Ctrl+A)
        if self._mode == "edit_notes" and (mods & Qt.KeyboardModifier.ControlModifier) and key == Qt.Key.Key_A:
            self._selected = set(range(len(self._midi_notes)))
            self.update()
            self.selection_changed.emit(self._selected.copy())
            return

        # Undo / redo are handled by `MainWindow` global shortcuts so one key press
        # maps to exactly one history step even when `PitchRoll` has focus.

        # ── Zoom shortcuts ─────────────────────────────────────────────────
        total = self._total_duration()

        if key in (Qt.Key.Key_Equal, Qt.Key.Key_Plus):          # + : zoom in (time)
            self._zoom_time(1.25)
        elif key == Qt.Key.Key_Minus:                            # - : zoom out (time)
            self._zoom_time(1 / 1.25)
        elif key == Qt.Key.Key_BracketRight:                     # ] : zoom in (pitch)
            self._zoom_midi(1.25)
        elif key == Qt.Key.Key_BracketLeft:                      # [ : zoom out (pitch)
            self._zoom_midi(1 / 1.25)

        # ── Pan shortcuts ──────────────────────────────────────────────────
        elif key == Qt.Key.Key_Left:
            step = (self._view_end - self._view_start) * 0.1
            self._pan_time(-step)
        elif key == Qt.Key.Key_Right:
            step = (self._view_end - self._view_start) * 0.1
            self._pan_time(step)
        elif key == Qt.Key.Key_Up:
            step = (self._midi_max - self._midi_min) * 0.1
            self._pan_midi(step)
        elif key == Qt.Key.Key_Down:
            step = (self._midi_max - self._midi_min) * 0.1
            self._pan_midi(-step)

        # ── Grid snap toggle ───────────────────────────────────────────────
        elif key == Qt.Key.Key_G:
            # Toggle grid snap
            self.grid_snap_enabled = not self.grid_snap_enabled
            status = "ON" if self.grid_snap_enabled else "OFF"
            print(f"Grid snap: {status} (resolution: {self.snap_resolution}s)")
            return

        # ── Reset view ─────────────────────────────────────────────────────
        elif key == Qt.Key.Key_Home:
            self._view_start = 0.0
            self._view_end   = total if total > 0 else 10.0
            self._midi_min   = float(MIDI_MIN)
            self._midi_max   = float(MIDI_MAX)

        else:
            super().keyPressEvent(event)
            return

        self.update()

    # ── view helpers ──────────────────────────────────────────────────────

    def _zoom_time(self, factor: float):
        total = self._total_duration()
        pivot = (self._view_start + self._view_end) / 2
        span  = (self._view_end - self._view_start) / factor
        span  = max(0.5, span)
        new_start = max(0.0, pivot - span / 2)
        new_end   = new_start + span
        if total > 0 and new_end > total:
            new_end   = total
            new_start = max(0.0, total - span)
        self._view_start = new_start
        self._view_end   = new_end

    def _zoom_midi(self, factor: float):
        mid  = (self._midi_min + self._midi_max) / 2
        span = (self._midi_max - self._midi_min) / factor
        span = max(2.0, span)          # never collapse to nothing
        span = min(span, 127.0)        # never exceed full MIDI range
        new_min = mid - span / 2
        new_max = mid + span / 2
        # Shift window to stay within [0, 127] without distorting span
        if new_min < 0:
            new_min, new_max = 0.0, span
        if new_max > 127:
            new_min, new_max = 127.0 - span, 127.0
        self._midi_min = new_min
        self._midi_max = new_max

    def _pan_time(self, delta: float):
        total = self._total_duration()
        span  = self._view_end - self._view_start
        self._view_start = max(0.0, min(self._view_start + delta, total - span))
        self._view_end   = self._view_start + span

    def _pan_midi(self, delta: float):
        span = self._midi_max - self._midi_min
        new_min = max(0, self._midi_min + delta)
        new_max = min(127, new_min + span)
        # Ensure new_max doesn't go below new_min
        if new_max < new_min:
            new_max = new_min
        self._midi_min = new_min
        self._midi_max = new_max

    def set_grid_snap(self, enabled: bool):
        """Enable or disable grid snapping."""
        self.grid_snap_enabled = enabled

    def set_snap_resolution(self, resolution: float):
        """Set the grid snap resolution in seconds."""
        self.snap_resolution = resolution
        print(f"Grid snap resolution: {resolution}s")

    def _snap_to_grid(self, time_sec: float) -> float:
        """Snap a time value to the nearest grid line."""
        if not self.grid_snap_enabled:
            return time_sec
        
        # Round to nearest grid resolution
        snapped = round(time_sec / self.snap_resolution) * self.snap_resolution
        return max(0.0, snapped)  # Ensure non-negative

    # ── undo / redo ───────────────────────────────────────────────────────

    def _push_undo(
        self,
        action_type: str = "generic",
        f0_snapshot: np.ndarray | None = None,
        midi_notes_snapshot: list[MidiNote] | None = None,
    ):
        import copy
        self._undo_stack.append((
            f0_snapshot.copy() if f0_snapshot is not None else (
                self._f0_target.copy() if len(self._f0_target) else np.array([])
            ),
            copy.deepcopy(midi_notes_snapshot) if midi_notes_snapshot is not None else copy.deepcopy(self._midi_notes),
            self._selected.copy(),  # Save selection state
            action_type,
        ))
        self._redo_stack.clear()
        # Cap stack depth to prevent memory issues
        max_depth = 50
        if len(self._undo_stack) > max_depth:
            # Remove oldest entries
            removed_count = len(self._undo_stack) - max_depth
            self._undo_stack = self._undo_stack[removed_count:]
            # Show hint when stack is full
            if hasattr(self, '_status'):
                self._status.showMessage(f"撤销历史记录已达上限 ({max_depth} 条)", 2000)

    def _undo(self):
        if not self._undo_stack:
            return
        import copy
        f0, notes, selected, action_type = self._undo_stack.pop()
        self._redo_stack.append((
            self._f0_target.copy() if len(self._f0_target) else np.array([]),
            copy.deepcopy(self._midi_notes),
            self._selected.copy(),  # Save selection state
            action_type,
        ))
        self._f0_target  = f0
        self._midi_notes = notes
        # Don't clear selection - restore previous state or keep current selection if it makes sense
        # Only clear selection if the undone action would invalidate it
        if action_type in ("delete_notes", "cut"):
            self._selected.clear()
        else:
            self._selected = selected  # Restore selection state
        self.update()
        self.selection_changed.emit(self._selected.copy())
        self.history_restored.emit("undo", action_type)
        # Emit f0_edited to update display, but main_window should not change f0_manually_edited flag
        self.f0_edited.emit(self._f0_target.copy())
        if action_type != "f0_draw":
            self.midi_notes_changed.emit(self._midi_notes)

    def _redo(self):
        if not self._redo_stack:
            return
        import copy
        f0, notes, selected, action_type = self._redo_stack.pop()
        self._undo_stack.append((
            self._f0_target.copy() if len(self._f0_target) else np.array([]),
            copy.deepcopy(self._midi_notes),
            self._selected.copy(),  # Save selection state
            action_type,
        ))
        self._f0_target  = f0
        self._midi_notes = notes
        self._selected = selected  # Restore selection state
        # Don't clear selection - restore previous state
        self.update()
        self.selection_changed.emit(self._selected.copy())
        self.history_restored.emit("redo", action_type)
        # Emit f0_edited to update display, but main_window should not change f0_manually_edited flag
        self.f0_edited.emit(self._f0_target.copy())
        if action_type != "f0_draw":
            self.midi_notes_changed.emit(self._midi_notes)

    # ── F0 draw mode ──────────────────────────────────────────────────────

    def _f0_press(self, event: QMouseEvent):
        if event.button() == Qt.MouseButton.LeftButton:
            if len(self._f0_target) == 0:
                return
            self._drawing = True
            self._last_draw_point = None
            self._f0_draw_snapshot = self._f0_target.copy()
            self._f0_draw_changed = False
            self._record_draw_point(event.pos())

    def _f0_move(self, event: QMouseEvent):
        if self._drawing:
            self._record_draw_point(event.pos())

    def _f0_release(self, event: QMouseEvent):
        if event.button() == Qt.MouseButton.LeftButton and self._drawing:
            self._drawing = False
            self._last_draw_point = None
            if self._f0_draw_changed:
                self.f0_edited.emit(self._f0_target.copy())
            self._f0_draw_snapshot = None
            self._f0_draw_changed = False

    def _record_draw_point(self, pos: QPoint):
        if len(self._f0_target) == 0:
            return
        RULER_HEIGHT = 20
        roll_w = self._canvas_w() - PIANO_WIDTH
        h = self._canvas_h()
        x = pos.x() - PIANO_WIDTH
        y = pos.y() - RULER_HEIGHT  # Offset for time ruler
        if x < 0 or x > roll_w:
            return
        time_range = self._view_end - self._view_start
        midi_range = self._midi_max - self._midi_min
        t = self._view_start + x / roll_w * time_range
        midi = float(np.clip(self._midi_min + (h - y) / h * midi_range,
                             self._midi_min, self._midi_max)) - 0.5
        midi = np.clip(midi, self._midi_min, self._midi_max)
        frame_idx = int(t * self._fps)

        def apply_frame(fi: int, midi_value: float) -> None:
            if not (0 <= fi < len(self._f0_target)):
                return
            new_f0 = 440.0 * (2.0 ** ((midi_value - 69) / 12.0))
            if np.isclose(self._f0_target[fi], new_f0, rtol=1e-5, atol=1e-5):
                return
            if not self._f0_draw_changed and self._f0_draw_snapshot is not None:
                self._push_undo("f0_draw", f0_snapshot=self._f0_draw_snapshot)
                self._f0_draw_changed = True
            self._f0_target[fi] = new_f0

        if self._last_draw_point is not None:
            last_fi, last_midi = self._last_draw_point
            fi_min = min(last_fi, frame_idx)
            fi_max = max(last_fi, frame_idx)
            if fi_max > fi_min:
                for fi in range(fi_min, fi_max + 1):
                    alpha = (fi - fi_min) / (fi_max - fi_min)
                    m = last_midi + alpha * (midi - last_midi) if last_fi < frame_idx \
                        else midi + alpha * (last_midi - midi)
                    apply_frame(fi, m)
            else:
                apply_frame(frame_idx, midi)
        else:
            apply_frame(frame_idx, midi)

        self._last_draw_point = (frame_idx, midi)
        self.update()

    # ── note edit mode ────────────────────────────────────────────────────

    def _hit_test(self, pos: QPoint) -> tuple[int | None, str]:
        """Return (note_index, zone) where zone is 'resize', 'body', 'stretch_pt', or 'none'."""
        for i, note in enumerate(self._midi_notes):
            rect = self._note_rect(note)
            if not rect.contains(pos):
                continue
            # Check stretch control points first (highest priority)
            for sp_idx, sp in enumerate(note.stretch_points):
                sp_x = rect.left() + int(sp.position * rect.width())
                sp_y = rect.top()
                dist = ((pos.x() - sp_x) ** 2 + (pos.y() - sp_y) ** 2) ** 0.5
                if dist <= STRETCH_PT_RADIUS + 2:
                    return i, "stretch_pt"
            # resize zone: right RESIZE_HANDLE_PX pixels
            if pos.x() >= rect.right() - RESIZE_HANDLE_PX:
                return i, "resize"
            return i, "body"
        return None, "none"

    def _notes_press(self, event: QMouseEvent):
        if event.button() != Qt.MouseButton.LeftButton:
            return
        pos = event.pos()
        idx, zone = self._hit_test(pos)

        if idx is not None:
            # Select note (add to selection if Ctrl held)
            if not (event.modifiers() & Qt.KeyboardModifier.ControlModifier):
                if idx not in self._selected:
                    self._selected = {idx}
            else:
                if idx in self._selected:
                    self._selected.discard(idx)
                else:
                    self._selected.add(idx)

            if zone == "stretch_pt":
                self._push_undo()
                self._drag_op = "stretch_pt"
                self._stretch_note_idx = idx
                note = self._midi_notes[idx]
                rect = self._note_rect(note)
                rel_x = (pos.x() - rect.left()) / max(rect.width(), 1)
                for sp_i, sp in enumerate(note.stretch_points):
                    if abs(sp.position - rel_x) < 0.05:
                        self._stretch_pt_idx = sp_i
                        self._stretch_pt_start_pos = sp.position
                        break
            elif zone == "resize":
                self._push_undo("resize")
                self._drag_op = "resize"
                self._drag_note_idx = idx
                self._resize_old_end = self._midi_notes[idx].end_sec
            else:
                self._push_undo()
                self._drag_op = "move"
                # Snapshot all selected notes
                self._drag_snapshots = [
                    (i, self._midi_notes[i].start_sec,
                        self._midi_notes[i].end_sec,
                        self._midi_notes[i].pitch)
                    for i in self._selected
                ]
            self._drag_start = pos
        else:
            # Start rubber-band selection
            self._selected.clear()
            self._drag_op = "rubberband"
            self._drag_start = pos
            if self._rubberband is None:
                self._rubberband = QRubberBand(QRubberBand.Shape.Rectangle, self)
            self._rubberband.setGeometry(QRect(pos, QSize()))
            self._rubberband.show()

        self.update()
        self.selection_changed.emit(set(self._selected))

    def _notes_move(self, event: QMouseEvent):
        if self._drag_op is None or self._drag_start is None:
            return
        pos = event.pos()
        dx = pos.x() - self._drag_start.x()
        dy = pos.y() - self._drag_start.y()

        if self._drag_op == "stretch_pt" and self._stretch_note_idx is not None and self._stretch_pt_idx is not None:
            note = self._midi_notes[self._stretch_note_idx]
            sp = note.stretch_points[self._stretch_pt_idx]
            rect = self._note_rect(note)
            new_rel_x = (pos.x() - rect.left()) / max(rect.width(), 1)
            
            # Clamp: keep distance from neighbors and edges
            min_pos = 0.05
            max_pos = 0.95
            # Also clamp relative to other stretch points
            for i, other_sp in enumerate(note.stretch_points):
                if i != self._stretch_pt_idx:
                    if other_sp.position < sp.orig_position:
                        min_pos = max(min_pos, other_sp.position + 0.03)
                    else:
                        max_pos = min(max_pos, other_sp.position - 0.03)
            
            new_rel_x = max(min_pos, min(max_pos, new_rel_x))
            sp.position = new_rel_x
            # Update pitch_roll during drag to show real-time waveform preview
            self.update()
            # Emit preview signal for real-time waveform update
            self.stretch_preview.emit()

        elif self._drag_op == "move" and self._drag_snapshots:
            dt = dx / max(self._canvas_w() - PIANO_WIDTH, 1) * (self._view_end - self._view_start)
            dm = -dy / max(self._canvas_h(), 1) * (self._midi_max - self._midi_min)
            for snap_idx, snap_start, snap_end, snap_pitch in self._drag_snapshots:
                note = self._midi_notes[snap_idx]
                dur = snap_end - snap_start
                # Apply grid snap to start time
                new_start_unsnapped = max(0.0, snap_start + dt)
                new_start = self._snap_to_grid(new_start_unsnapped)
                note.start_sec = new_start
                note.end_sec   = new_start + dur
                note.pitch = int(np.clip(round(snap_pitch + dm), MIDI_MIN, MIDI_MAX))
            self.update()
            # Don't emit during drag to avoid performance issues
            # Only emit on release (handled in _notes_release)

        elif self._drag_op == "resize" and self._drag_note_idx is not None:
            note = self._midi_notes[self._drag_note_idx]
            new_end_unsnapped = self._x_to_sec(pos.x())
            # Apply grid snap to resize end time
            new_end = self._snap_to_grid(new_end_unsnapped)
            note.end_sec = max(note.start_sec + 0.05, new_end)
            self.update()
            # Don't emit during drag to avoid performance issues
            # Only emit on release (handled in _notes_release)

        elif self._drag_op == "rubberband" and self._rubberband is not None:
            rb_rect = QRect(self._drag_start, pos).normalized()
            self._rubberband.setGeometry(rb_rect)
            # Highlight notes that intersect
            self._selected = {
                i for i, note in enumerate(self._midi_notes)
                if self._note_rect(note).intersects(rb_rect)
            }
            self.update()

    def _notes_release(self, event: QMouseEvent):
        if event.button() != Qt.MouseButton.LeftButton:
            return
        if self._rubberband is not None:
            self._rubberband.hide()
        if self._drag_op in ("move", "resize", "stretch_pt"):
            # Only emit midi_notes_changed if there was actual movement
            # Check if mouse moved more than a small threshold (5 pixels)
            pos = event.pos()
            if self._drag_start is not None:
                dx = abs(pos.x() - self._drag_start.x())
                dy = abs(pos.y() - self._drag_start.y())
                if dx > 5 or dy > 5:
                    # Update pitch_roll display now that drag is complete
                    if self._drag_op == "stretch_pt":
                        self.update()
                    self.midi_notes_changed.emit(self._midi_notes)
                    if self._drag_op == "resize" and self._drag_note_idx is not None:
                        note = self._midi_notes[self._drag_note_idx]
                        self.note_resized.emit(self._drag_note_idx, self._resize_old_end, note.end_sec)
                    if self._drag_op == "stretch_pt":
                        self.stretch_changed.emit(True)  # True = timing edit, preserve pitch
        elif self._drag_op == "rubberband":
            self.selection_changed.emit(set(self._selected))
        self._drag_op = None
        self._drag_start = None
        self._drag_note_idx = None
        self._drag_snapshots = []
        self._stretch_note_idx = None
        self._stretch_pt_idx = None
        self._stretch_pt_start_pos = 0.0
        self.update()

    # ── right-click context menu ──────────────────────────────────────────

    def _show_cut_help_menu(self, event: QMouseEvent):
        """Show help menu for cut mode."""
        menu = QMenu(self)
        menu.setTitle("切割模式帮助")
        
        help_action = menu.addAction("✂ 切割音符")
        help_action.setEnabled(False)
        menu.addSeparator()
        
        info1 = menu.addAction("在音符上点击即可切割")
        info1.setEnabled(False)
        info2 = menu.addAction("红色虚线表示切割位置")
        info2.setEnabled(False)
        info3 = menu.addAction("按 Esc 或点击其他工具退出切割模式")
        info3.setEnabled(False)
        
        menu.exec(QCursor.pos())

    def _show_f0_help_menu(self, event: QMouseEvent):
        """Show help menu for F0 draw mode."""
        menu = QMenu(self)
        menu.setTitle("绘制音高模式帮助")
        
        help_action = menu.addAction("✏️ 绘制音高曲线")
        help_action.setEnabled(False)
        menu.addSeparator()
        
        info1 = menu.addAction("按住左键拖动绘制音高")
        info1.setEnabled(False)
        info2 = menu.addAction("绿色线：原始音高")
        info2.setEnabled(False)
        info3 = menu.addAction("蓝色线：目标音高")
        info3.setEnabled(False)
        info4 = menu.addAction("按 Esc 或点击其他工具退出绘制模式")
        info4.setEnabled(False)
        
        menu.exec(QCursor.pos())

    def _show_note_context_menu(self, event: QMouseEvent):
        pos = event.pos()
        idx, zone = self._hit_test(pos)

        menu = QMenu(self)

        if idx is not None:
            note = self._midi_notes[idx]
            rect = self._note_rect(note)
            rel_pos = (pos.x() - rect.left()) / max(rect.width(), 1)
            rel_pos = max(0.05, min(0.95, rel_pos))

            menu.addSeparator()

            add_pt_action = menu.addAction("添加拉伸控制点")
            add_pt_action.setData(("add_stretch", idx, rel_pos))

            # Check if clicking near an existing stretch point
            near_sp = None
            for sp_i, sp in enumerate(note.stretch_points):
                if abs(sp.position - rel_pos) < 0.05:
                    near_sp = sp_i
                    break

            if near_sp is not None:
                remove_pt_action = menu.addAction("删除此控制点")
                remove_pt_action.setData(("remove_stretch", idx, near_sp))
            else:
                remove_pt_action = None

            if note.stretch_points:
                clear_pts_action = menu.addAction("清除所有控制点")
                clear_pts_action.setData(("clear_stretch", idx))
            else:
                clear_pts_action = None

            menu.addSeparator()

            # Show stretch ratios for existing points
            if note.stretch_points:
                for sp_i, sp in enumerate(note.stretch_points):
                    left_pct = int(sp.left_ratio * 100)
                    right_pct = int(sp.right_ratio * 100)
                    moved = abs(sp.position - sp.orig_position) > 0.01
                    if moved:
                        label = f"控制点 {sp_i+1}: 左{left_pct}% 右{right_pct}%"
                    else:
                        label = f"控制点 {sp_i+1}: 未移动"
                    info_action = menu.addAction(label)
                    info_action.setEnabled(False)
                menu.addSeparator()

            action = menu.exec(QCursor.pos())

            if action is None:
                return

            action_data = action.data()
            if not action_data:
                return

            if isinstance(action_data, tuple):
                action_type = action_data[0]

                if action_type == "add_stretch":
                    _, note_idx, position = action_data
                    self._push_undo()
                    self._midi_notes[note_idx].stretch_points.append(
                        StretchPoint(orig_position=position, position=position)
                    )
                    self._midi_notes[note_idx].stretch_points.sort(key=lambda sp: sp.position)
                    self.update()
                    self.midi_notes_changed.emit(self._midi_notes)
                    self.stretch_changed.emit(True)

                elif action_type == "remove_stretch":
                    _, note_idx, sp_idx = action_data
                    self._push_undo()
                    self._midi_notes[note_idx].stretch_points.pop(sp_idx)
                    self.update()
                    self.midi_notes_changed.emit(self._midi_notes)
                    self.stretch_changed.emit(True)

                elif action_type == "clear_stretch":
                    _, note_idx = action_data
                    self._push_undo()
                    self._midi_notes[note_idx].stretch_points.clear()
                    self.update()
                    self.midi_notes_changed.emit(self._midi_notes)
                    self.stretch_changed.emit(True)
        else:
            info_action = menu.addAction("(未选中音符)")
            info_action.setEnabled(False)
            menu.exec(QCursor.pos())

    # ── cut mode ──────────────────────────────────────────────────────────

    def _cut_press(self, event: QMouseEvent):
        if event.button() != Qt.MouseButton.LeftButton:
            return
        pos = event.pos()
        if pos.x() < PIANO_WIDTH:
            return
        cut_sec = self._x_to_sec(pos.x())
        idx, _ = self._hit_test(pos)
        if idx is None:
            self._cut_hover_note_idx = None
            self._cut_hover_sec = cut_sec
            self.update()
            return
        note = self._midi_notes[idx]
        # Don't cut too close to edges
        if cut_sec <= note.start_sec + 0.05 or cut_sec >= note.end_sec - 0.05:
            return
        self._cut_hover_note_idx = idx
        self._cut_hover_sec = cut_sec
        self.update()

    def _cut_move(self, event: QMouseEvent):
        pos = event.pos()
        if pos.x() < PIANO_WIDTH:
            self._cut_hover_note_idx = None
            self._cut_hover_sec = None
            self.update()
            return
        cut_sec = self._x_to_sec(pos.x())
        idx, _ = self._hit_test(pos)
        self._cut_hover_note_idx = idx
        self._cut_hover_sec = cut_sec
        self.update()

    def _cut_release(self, event: QMouseEvent):
        if event.button() != Qt.MouseButton.LeftButton:
            return
        if self._cut_hover_note_idx is None or self._cut_hover_sec is None:
            self._cut_hover_note_idx = None
            self._cut_hover_sec = None
            self.update()
            return
        idx = self._cut_hover_note_idx
        cut_sec = self._cut_hover_sec
        self._cut_hover_note_idx = None
        self._cut_hover_sec = None

        if idx >= len(self._midi_notes):
            self.update()
            return
        note = self._midi_notes[idx]
        # Apply grid snap
        cut_sec = self._snap_to_grid(cut_sec)
        # Don't cut too close to edges
        if cut_sec <= note.start_sec + 0.05 or cut_sec >= note.end_sec - 0.05:
            self.update()
            return
        
        self._push_undo("cut")
        left = MidiNote(
            note.start_sec, cut_sec, note.pitch, note.lyric,
            stretch_points=list(note.stretch_points)  # Copy stretch points to left part
        )
        right = MidiNote(
            cut_sec, note.end_sec, note.pitch, "",
            stretch_points=[]  # Right part starts with no stretch points
        )
        self._midi_notes[idx:idx + 1] = [left, right]
        self.update()
        self.midi_notes_changed.emit(self._midi_notes)
        self.note_cut.emit(idx, cut_sec)
