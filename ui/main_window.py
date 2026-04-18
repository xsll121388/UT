"""Main application window."""
from __future__ import annotations
import os
import numpy as np
from PyQt6.QtCore import Qt, QTimer, pyqtSlot
from PyQt6.QtGui import QAction, QKeySequence, QDragEnterEvent, QDropEvent, QShortcut
from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QSplitter, QFileDialog, QMessageBox, QLabel, QStatusBar, QPushButton, QProgressBar,
    QApplication
)

from core.project import ProjectState, ClipState, TuningData, compute_tuning_id, compute_midi_hash
from core.render_cache import RenderCache
from core.render_worker import RenderWorker
from core.audio_engine import AudioEngine
from core.pitch_tracker import PitchTracker
from core.pitch_corrector import snap_f0_to_scale, smooth_f0
from utils.audio_utils import split_into_chunks, load_audio, load_midi_notes, time_stretch_to_duration
from utils.performance import get_monitor, time_operation
from utils.error_handler import show_error, show_friendly_exception, show_info, confirm_action
from ui.waveform_view import WaveformView
from ui.pitch_roll import PitchRoll
from ui.lyrics_bar import LyricsBar
from ui.parameter_panel import ParameterPanel
from ui.transport_bar import TransportBar
from ui.audio_settings_dialog import AudioSettingsDialog, VolumeControl
from ui.batch_dialog import BatchDialog
from ui import styles


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("UT")
        self.resize(1200, 700)
        self.setMinimumSize(900, 500)

        self._project = ProjectState()
        self._project_path: str | None = None
        self._audio: np.ndarray = np.array([], dtype=np.float32)
        self._audio_raw: np.ndarray = np.array([], dtype=np.float32)
        self._sr = 44100
        self._midi_notes: list = []
        self._sample_folder: str | None = None   # persists across sessions
        self._timeline_undo_stack: list[dict] = []
        self._timeline_redo_stack: list[dict] = []

        self._cache = RenderCache()
        self._worker = RenderWorker(self._cache, self)
        self._engine = AudioEngine(self._cache, self)
        self._tracker: PitchTracker | None = None  # lazy init

        self._build_ui()
        self._connect_signals()
        self._worker.start()

        # Enable drag & drop
        self.setAcceptDrops(True)

        # Space key shortcut for play/pause
        shortcut_space = QShortcut(QKeySequence(Qt.Key.Key_Space), self)
        shortcut_space.activated.connect(self._toggle_play)

        # Global Undo/Redo shortcuts
        shortcut_undo = QShortcut(QKeySequence(QKeySequence.StandardKey.Undo), self)
        shortcut_undo.activated.connect(self._undo)
        
        shortcut_redo = QShortcut(QKeySequence(QKeySequence.StandardKey.Redo), self)
        shortcut_redo.activated.connect(self._redo)

        # Transport shortcuts
        shortcut_stop = QShortcut(QKeySequence(Qt.Key.Key_S), self)
        shortcut_stop.activated.connect(self._stop)
        
        shortcut_home = QShortcut(QKeySequence(Qt.Key.Key_Home), self)
        shortcut_home.activated.connect(self._go_to_start)
        
        shortcut_end = QShortcut(QKeySequence(Qt.Key.Key_End), self)
        shortcut_end.activated.connect(self._go_to_end)
        
        shortcut_seek_back = QShortcut(QKeySequence(Qt.Key.Key_Left), self)
        shortcut_seek_back.activated.connect(lambda: self._seek_relative(-5.0))
        
        shortcut_seek_fwd = QShortcut(QKeySequence(Qt.Key.Key_Right), self)
        shortcut_seek_fwd.activated.connect(lambda: self._seek_relative(5.0))
        
        shortcut_seek_back_fine = QShortcut(QKeySequence(Qt.KeyboardModifier.ShiftModifier | Qt.Key.Key_Left), self)
        shortcut_seek_back_fine.activated.connect(lambda: self._seek_relative(-1.0))
        
        shortcut_seek_fwd_fine = QShortcut(QKeySequence(Qt.KeyboardModifier.ShiftModifier | Qt.Key.Key_Right), self)
        shortcut_seek_fwd_fine.activated.connect(lambda: self._seek_relative(1.0))

        # Mode switching shortcuts
        shortcut_draw_mode = QShortcut(QKeySequence(Qt.Key.Key_F1), self)
        shortcut_draw_mode.activated.connect(self._on_mode_draw)
        
        shortcut_edit_mode = QShortcut(QKeySequence(Qt.Key.Key_F2), self)
        shortcut_edit_mode.activated.connect(self._on_mode_edit)
        
        shortcut_midi_snap = QShortcut(QKeySequence(Qt.Key.Key_F3), self)
        shortcut_midi_snap.activated.connect(self._toggle_midi_snap)

        # Zoom shortcuts (global)
        shortcut_zoom_fit = QShortcut(QKeySequence(Qt.Key.Key_0), self)
        shortcut_zoom_fit.activated.connect(self._zoom_to_fit)
        
        shortcut_zoom_fit_f = QShortcut(QKeySequence(Qt.Key.Key_F), self)
        shortcut_zoom_fit_f.activated.connect(self._zoom_to_fit)

        # Pitch correction shortcuts
        shortcut_reextract = QShortcut(QKeySequence("Ctrl+P"), self)
        shortcut_reextract.activated.connect(self._reextract_pitch)
        
        shortcut_apply_corr = QShortcut(QKeySequence("Ctrl+K"), self)
        shortcut_apply_corr.activated.connect(self._apply_correction)
        
        shortcut_reset_target = QShortcut(QKeySequence("Ctrl+T"), self)
        shortcut_reset_target.activated.connect(self._reset_target)

        # Render shortcuts
        shortcut_render_all = QShortcut(QKeySequence(Qt.Key.Key_F5), self)
        shortcut_render_all.activated.connect(self._render_all)

        # Edit shortcuts (placeholder for future implementation)
        shortcut_select_all = QShortcut(QKeySequence(QKeySequence.StandardKey.SelectAll), self)
        shortcut_select_all.activated.connect(self._on_select_all)
        
        shortcut_copy = QShortcut(QKeySequence(QKeySequence.StandardKey.Copy), self)
        shortcut_copy.activated.connect(self._on_copy)
        
        shortcut_paste = QShortcut(QKeySequence(QKeySequence.StandardKey.Paste), self)
        shortcut_paste.activated.connect(self._on_paste)

        # Position update timer
        self._pos_timer = QTimer(self)
        self._pos_timer.setInterval(50)
        self._pos_timer.timeout.connect(self._update_position)
        self._pos_timer.start()

    def _tuning(self) -> TuningData:
        if not self._project.clips:
            return TuningData()
        return self._project.get_clip_tuning(0)

    # ── UI construction ───────────────────────────────────────────────────

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        # Transport bar
        self._transport = TransportBar()
        root.addWidget(self._transport)

        # Main splitter: left=editor, right=params
        splitter = QSplitter(Qt.Orientation.Horizontal)
        root.addWidget(splitter, stretch=1)

        # Editor area (waveform + mode bar + pitch roll + lyrics bar)
        editor = QWidget()
        editor_layout = QVBoxLayout(editor)
        editor_layout.setContentsMargins(4, 4, 4, 4)
        editor_layout.setSpacing(4)

        self._waveform = WaveformView()
        editor_layout.addWidget(self._waveform)

        # Mode toggle bar
        mode_bar = QHBoxLayout()
        mode_bar.setSpacing(4)
        self._btn_draw = QPushButton("✏ 绘制 (F1)")
        self._btn_edit = QPushButton("♪ 编辑 (F2)")
        self._btn_cut  = QPushButton("✂ 切割 (F3)")
        for btn in (self._btn_draw, self._btn_edit, self._btn_cut):
            btn.setCheckable(True)
            btn.setFixedHeight(26)
            btn.setFixedWidth(110)
        self._btn_draw.setChecked(True)
        mode_bar.addWidget(self._btn_draw)
        mode_bar.addWidget(self._btn_edit)
        mode_bar.addWidget(self._btn_cut)

        # MIDI pitch snap toggle
        self._btn_midi_snap = QPushButton("🎵 对齐 MIDI")
        self._btn_midi_snap.setObjectName("midi_snap")
        self._btn_midi_snap.setCheckable(True)
        self._btn_midi_snap.setChecked(False)
        self._btn_midi_snap.setFixedHeight(26)
        self._btn_midi_snap.setFixedWidth(110)
        self._btn_midi_snap.setToolTip("开启后，音频音高将自动拉平到 MIDI 音符\n快捷键：F3")
        mode_bar.addWidget(self._btn_midi_snap)

        mode_bar.addStretch()
        editor_layout.addLayout(mode_bar)

        self._pitch_roll = PitchRoll()
        editor_layout.addWidget(self._pitch_roll, stretch=1)

        self._lyrics_bar = LyricsBar()
        editor_layout.addWidget(self._lyrics_bar)

        splitter.addWidget(editor)

        # Parameter panel
        self._params = ParameterPanel()
        splitter.addWidget(self._params)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 0)
        splitter.setSizes([880, 320])

        # Status bar
        self._status = QStatusBar()
        self.setStatusBar(self._status)
        
        # Render status label
        self._render_label = QLabel("待机")
        self._status.addPermanentWidget(self._render_label)
        
        # Progress bar for rendering
        self._render_progress = QProgressBar()
        self._render_progress.setVisible(False)
        self._render_progress.setMaximumWidth(200)
        self._render_progress.setRange(0, 100)
        self._render_progress.setValue(0)
        self._status.addPermanentWidget(self._render_progress)

        # Menu bar
        self._build_menu()

    def _build_menu(self):
        mb = self.menuBar()

        def act(text, slot, shortcut=None):
            a = QAction(text, self)
            a.triggered.connect(slot)
            if shortcut:
                a.setShortcut(QKeySequence(shortcut))
            return a

        file_menu = mb.addMenu("文件")
        file_menu.addAction(act("打开音频...", self._open_audio, "Ctrl+O"))
        file_menu.addAction(act("导入 MIDI...", self._import_midi, "Ctrl+M"))
        file_menu.addAction(act("导入音源文件夹...", self._import_sample_folder, "Ctrl+Shift+M"))
        file_menu.addAction(act("重新生成音频（用当前音源文件夹）", self._rebuild_from_sample_folder, "Ctrl+R"))
        file_menu.addSeparator()
        file_menu.addAction(act("保存工程", self._save_project, "Ctrl+S"))
        file_menu.addAction(act("另存为...", self._save_project_as, "Ctrl+Shift+S"))
        file_menu.addAction(act("打开工程...", self._open_project, "Ctrl+Shift+O"))
        file_menu.addSeparator()
        file_menu.addAction(act("导出音频...", self._export_audio, "Ctrl+E"))
        file_menu.addSeparator()
        file_menu.addAction(act("批量处理...", self._open_batch))
        file_menu.addSeparator()
        file_menu.addAction(act("退出", self.close, "Ctrl+Q"))

        settings_menu = mb.addMenu("设置")
        settings_menu.addAction(act("音频设置...", self._open_audio_settings))

        # Theme submenu
        theme_menu = settings_menu.addMenu("主题")
        self._theme_dark_action = theme_menu.addAction("暗色主题")
        self._theme_dark_action.setCheckable(True)
        self._theme_dark_action.setChecked(True)
        self._theme_dark_action.triggered.connect(lambda: self._set_theme("dark"))

        self._theme_light_action = theme_menu.addAction("亮色主题")
        self._theme_light_action.setCheckable(True)
        self._theme_light_action.setChecked(False)
        self._theme_light_action.triggered.connect(lambda: self._set_theme("light"))

        edit_menu = mb.addMenu("编辑")
        edit_menu.addAction(act("重新提取音高", self._reextract_pitch))
        edit_menu.addAction(act("应用修正", self._apply_correction))
        edit_menu.addAction(act("重置目标音高", self._reset_target))
        edit_menu.addSeparator()
        
        # Grid snap submenu
        snap_menu = edit_menu.addMenu("网格吸附")
        self._snap_action = snap_menu.addAction("启用网格吸附")
        self._snap_action.setCheckable(True)
        self._snap_action.setChecked(False)
        self._snap_action.triggered.connect(self._toggle_grid_snap)
        
        snap_resolution_menu = snap_menu.addMenu("吸附分辨率")
        snap_resolution_menu.addAction("1/4 拍 (0.25s)", lambda: self._set_snap_resolution(0.25))
        snap_resolution_menu.addAction("1/8 拍 (0.125s)", lambda: self._set_snap_resolution(0.125))
        snap_resolution_menu.addAction("1/16 拍 (0.0625s)", lambda: self._set_snap_resolution(0.0625))
        snap_resolution_menu.addAction("1/32 拍 (0.03125s)", lambda: self._set_snap_resolution(0.03125))

        tools_menu = mb.addMenu("工具")
        tools_menu.addAction(act("模型诊断...", self._open_model_docs))
        tools_menu.addAction(act("性能报告...", self._show_performance_report))

        help_menu = mb.addMenu("帮助")
        help_menu.addAction(act("关于", self._show_about))

    # ── signal wiring ─────────────────────────────────────────────────────

    def _connect_signals(self):
        self._transport.play_clicked.connect(self._toggle_play)
        self._transport.stop_clicked.connect(self._stop)
        self._transport.seek_requested.connect(self._seek)
        self._waveform.seek_requested.connect(self._seek)
        self._pitch_roll.seek_requested.connect(self._seek)
        self._pitch_roll.history_restored.connect(self._on_pitch_history_restored)
        self._pitch_roll.stretch_changed.connect(self._on_stretch_changed)
        self._pitch_roll.stretch_preview.connect(self._on_stretch_preview_direct)
        self._params.params_changed.connect(self._on_params_changed)
        self._pitch_roll.f0_edited.connect(self._on_f0_edited)
        self._pitch_roll.midi_notes_changed.connect(self._on_midi_notes_changed)
        self._pitch_roll.selection_changed.connect(self._lyrics_bar.set_selected)
        self._lyrics_bar.lyrics_changed.connect(self._on_lyrics_changed)
        self._btn_draw.clicked.connect(self._on_mode_draw)
        self._btn_edit.clicked.connect(self._on_mode_edit)
        self._btn_cut.clicked.connect(self._on_mode_cut)
        self._pitch_roll.note_resized.connect(self._on_note_resized)
        self._pitch_roll.note_cut.connect(self._on_note_cut)
        self._btn_midi_snap.toggled.connect(self._on_midi_snap_toggled)
        self._worker.chunk_done.connect(self._on_chunk_done)
        self._worker.error.connect(self._on_worker_error)
        self._engine.error_occurred.connect(self._on_audio_error)
        # playback_stopped is now polled via _update_position timer (no cross-thread signal)

    def _capture_timeline_state(self) -> dict | None:
        if len(self._audio) == 0:
            return None

        state = {
            "audio": self._audio.copy(),
            "audio_raw": self._audio_raw.copy(),
            "sr": self._sr,
            "f0_original": np.array([], dtype=np.float32),
        }
        tuning = self._tuning()
        if len(tuning.f0_original) > 0:
            state["f0_original"] = tuning.f0_original.copy()
        return state

    def _restore_timeline_state(self, state: dict | None) -> None:
        if not state:
            return

        audio = state["audio"].copy()
        audio_raw = state["audio_raw"].copy()
        sr = int(state["sr"])

        self._audio = audio
        self._audio_raw = audio_raw
        self._sr = sr

        intervals = split_into_chunks(audio, sr)
        self._cache.reset(audio, intervals, sr)
        self._waveform.set_audio(audio, sr)
        self._pitch_roll.set_waveform(audio, sr)
        self._transport.set_duration(len(audio) / sr if sr > 0 else 0.0)

        if self._project.clips:
            self._tuning().f0_original = state["f0_original"].copy()

    @pyqtSlot(str, str)
    def _on_pitch_history_restored(self, direction: str, action_type: str):
        # Set flag to prevent f0_manually_edited from being changed during undo/redo
        self._is_restoring_history = True
        QTimer.singleShot(100, lambda: setattr(self, '_is_restoring_history', False))
        
        if action_type != "resize":
            return

        current_state = self._capture_timeline_state()
        if direction == "undo":
            if not self._timeline_undo_stack:
                return
            state = self._timeline_undo_stack.pop()
            if current_state is not None:
                self._timeline_redo_stack.append(current_state)
            self._restore_timeline_state(state)
        elif direction == "redo":
            if not self._timeline_redo_stack:
                return
            state = self._timeline_redo_stack.pop()
            if current_state is not None:
                self._timeline_undo_stack.append(current_state)
            self._restore_timeline_state(state)

    # ── file operations ───────────────────────────────────────────────────

    def _open_audio(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "打开音频", "", "音频文件 (*.wav *.flac *.mp3 *.ogg *.aiff)"
        )
        if not path:
            return
        self._load_audio_file(path)

    def _import_midi(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "导入 MIDI", "", "MIDI 文件 (*.mid *.midi)"
        )
        if not path:
            return
        try:
            notes, bpm = load_midi_notes(path)
        except Exception as e:
            show_friendly_exception(self, e, context="读取 MIDI 文件时发生错误")
            return
        if not notes:
            show_info(self, "提示", "MIDI 文件中未找到音符。")
            return
        self._midi_notes = notes
        self._pitch_roll.set_midi_notes(notes)
        self._lyrics_bar.set_notes(notes)
        self._transport.set_bpm(bpm)

        # If audio already loaded, re-stretch from original to new MIDI duration
        if len(self._audio_raw) > 0:
            midi_dur = max(n.end_sec for n in notes)
            stretched = time_stretch_to_duration(self._audio_raw, self._sr, midi_dur)
            self._audio_raw = stretched.copy()
            self._audio = stretched
            intervals = split_into_chunks(stretched, self._sr)
            self._cache.reset(stretched, intervals, self._sr)
            self._waveform.set_audio(stretched, self._sr)
            self._pitch_roll.set_waveform(stretched, self._sr)
            self._transport.set_duration(midi_dur)
            self._extract_pitch()
            self._status.showMessage(
                f"已导入 {len(notes)} 个音符，BPM={bpm:.1f}，音频已拉伸到 {midi_dur:.1f}s"
            )
            return

        self._status.showMessage(
            f"已导入 {len(notes)} 个音符，BPM={bpm:.1f}: {os.path.basename(path)}"
        )
        # If audio is already loaded and snap is enabled, apply immediately
        if self._btn_midi_snap.isChecked() and len(self._audio) > 0 and self._project.clips:
            self._snap_f0_to_midi()

    def _import_sample_folder(self):
        """Import a folder of audio samples named by lyric character.

        Each file like 啊.wav / 哦.flac is matched to MIDI notes whose lyric
        equals the filename stem. The samples are placed at the MIDI note
        start times on a silent timeline and the result is loaded as audio.
        The folder path is remembered — re-importing uses the same folder
        without prompting if called again.
        """
        if not self._midi_notes:
            show_info(self, "请先导入 MIDI 文件", "请先导入 MIDI 文件，再导入音源文件夹。")
            return

        # Use remembered folder, or prompt for a new one
        start_dir = self._sample_folder or ""
        folder = QFileDialog.getExistingDirectory(self, "选择音源文件夹", start_dir)
        if not folder:
            return
        self._sample_folder = folder   # remember for next time
        self._import_sample_folder_from(folder)

    def _rebuild_from_sample_folder(self):
        """Re-run sample folder assembly using the remembered folder path."""
        if not self._sample_folder:
            show_info(self, "需要音源文件夹", '尚未选择音源文件夹，请先使用"导入音源文件夹"。')
            return
        if not self._midi_notes:
            show_info(self, "需要 MIDI 文件", "请先导入 MIDI 文件。")
            return
        self._import_sample_folder_from(self._sample_folder)

    def _import_sample_folder_from(self, folder: str, force_reextract: bool = False):
        """Core logic shared by _import_sample_folder and _rebuild_from_sample_folder."""
        from utils.audio_utils import apply_stretch_points

        SR = 44100
        AUDIO_EXTS = {'.wav', '.flac', '.mp3', '.ogg', '.aiff'}

        sample_map: dict[str, str] = {}
        for fname in os.listdir(folder):
            stem, ext = os.path.splitext(fname)
            if ext.lower() in AUDIO_EXTS:
                sample_map[stem] = os.path.join(folder, fname)

        if not sample_map:
            show_error(self, "sample_folder_empty", details="文件夹中未找到音频文件")
            return

        total_sec = max(n.end_sec for n in self._midi_notes) + 0.5
        total_samples = int(total_sec * SR)
        timeline = np.zeros(total_samples, dtype=np.float32)

        placed = 0
        missing_lyrics: set[str] = set()

        for note in self._midi_notes:
            lyric = note.lyric.strip()
            if not lyric:
                continue
            if lyric not in sample_map:
                missing_lyrics.add(lyric)
                continue
            try:
                seg, seg_sr = load_audio(sample_map[lyric], target_sr=SR)
            except Exception:
                missing_lyrics.add(lyric)
                continue

            note_dur_sec = note.end_sec - note.start_sec

            # Apply stretch control points if present
            if note.stretch_points:
                has_moved = any(abs(sp.position - sp.orig_position) > 0.01 for sp in note.stretch_points)
                if has_moved:
                    # First stretch to note duration, then apply stretch points
                    seg = time_stretch_to_duration(seg, SR, note_dur_sec)
                    seg = apply_stretch_points(seg, SR, note)
                else:
                    seg = time_stretch_to_duration(seg, SR, note_dur_sec)
            else:
                seg = time_stretch_to_duration(seg, SR, note_dur_sec)

            start_sample = int(note.start_sec * SR)
            end_sample = min(start_sample + len(seg), total_samples)
            timeline[start_sample:end_sample] += seg[:end_sample - start_sample]
            placed += 1

        if placed == 0:
            show_error(self, "sample_place_failed", 
                      details=f"没有音符被放置。\n缺少音源：{', '.join(sorted(missing_lyrics)) or '（所有音符均无歌词）'}")
            return

        peak = np.max(np.abs(timeline))
        if peak > 1.0:
            timeline /= peak

        msg = f"已放置 {placed} 个音符"
        if missing_lyrics:
            msg += f"，缺少音源: {', '.join(sorted(missing_lyrics))}"
        self._status.showMessage(msg)
        self._load_audio_array(timeline, SR, label=f"音源文件夹: {os.path.basename(folder)}", force_reextract=force_reextract)

    def _load_audio_file(self, path: str):
        self._status.showMessage(f"加载：{os.path.basename(path)}")
        try:
            audio, sr = load_audio(path, target_sr=44100)
            if len(audio) == 0:
                show_error(self, "audio_load", details="音频文件为空或无法读取")
                return
        except Exception as e:
            show_friendly_exception(self, e, context="加载音频文件时发生错误")
            return
        self._load_audio_array(audio, sr, label=os.path.basename(path), path=path)

    def _load_audio_array(self, audio: np.ndarray, sr: int,
                          label: str = "", path: str = "", force_reextract: bool = False):
        self._audio_raw = audio.copy()
        if self._midi_notes:
            midi_dur = max(n.end_sec for n in self._midi_notes)
            audio_dur = len(audio) / sr
            if abs(audio_dur - midi_dur) > 0.5:
                self._status.showMessage(f"正在调整音频时长以匹配MIDI...")
                QApplication.processEvents()
                try:
                    audio = time_stretch_to_duration(audio, sr, midi_dur)
                except Exception as e:
                    print(f"[stretch] Error: {e}")
        self._audio_raw = audio.copy()
        self._audio = audio
        self._sr = sr

        intervals = split_into_chunks(audio, sr)
        self._cache.reset(audio, intervals, sr)

        self._waveform.set_audio(audio, sr)
        self._pitch_roll.set_waveform(audio, sr)
        self._transport.set_duration(len(audio) / sr)

        tuning_id = compute_tuning_id(path or label)

        if not self._project.clips:
            self._project.clips.append(ClipState(audio_path=path or label, tuning_id=tuning_id))
        else:
            self._project.clips[0].audio_path = path or label
            self._project.clips[0].tuning_id = tuning_id

        if label:
            self._status.showMessage(f"已加载: {label}")

        tuning = self._tuning()

        if len(tuning.f0_target) > 0 and not force_reextract:
            saved_f0_target = tuning.f0_target.copy()
            saved_manually_edited = tuning.f0_manually_edited

            try:
                if self._tracker is None:
                    self._tracker = PitchTracker()
                f0 = self._tracker.extract_from_44k(self._audio)
                tuning.f0_original = f0
            except Exception:
                pass

            new_f0_len = len(self._audio) // (self._sr // 100)
            if len(saved_f0_target) != new_f0_len and new_f0_len > 0:
                old_indices = np.linspace(0, len(saved_f0_target) - 1, len(saved_f0_target))
                new_indices = np.linspace(0, len(saved_f0_target) - 1, new_f0_len)
                saved_f0_target = np.interp(new_indices, old_indices, saved_f0_target).astype(np.float32)

            tuning.f0_target = saved_f0_target
            tuning.f0_manually_edited = saved_manually_edited

            self._pitch_roll.set_data(tuning.f0_original, tuning.f0_target, len(self._audio), self._sr)
            self._worker.set_params(self._audio, tuning.f0_target, self._sr,
                                    enable_advanced_f0=True)
            self._cache.invalidate_all()
            self._worker.wake()
            self._render_label.setText("渲染中")
        elif len(tuning.f0_target) > 0 and force_reextract:
            old_f0_target = tuning.f0_target.copy()
            old_f0_original = tuning.f0_original.copy() if len(tuning.f0_original) > 0 else np.array([], dtype=np.float32)

            self._extract_pitch()

            if len(old_f0_target) > 0 and len(old_f0_original) > 0 and len(tuning.f0_target) > 0:
                min_old = min(len(old_f0_target), len(old_f0_original))
                edited_mask = np.abs(old_f0_target[:min_old] - old_f0_original[:min_old]) > 1.0

                if len(edited_mask) != len(tuning.f0_target):
                    old_idx = np.linspace(0, len(edited_mask) - 1, len(edited_mask))
                    new_idx = np.linspace(0, len(edited_mask) - 1, len(tuning.f0_target))
                    edited_mask = np.interp(new_idx, old_idx, edited_mask.astype(np.float32)) > 0.5

                resampled_old_target = old_f0_target
                if len(old_f0_target) != len(tuning.f0_target):
                    old_indices = np.linspace(0, len(old_f0_target) - 1, len(old_f0_target))
                    new_indices = np.linspace(0, len(old_f0_target) - 1, len(tuning.f0_target))
                    resampled_old_target = np.interp(new_indices, old_indices, old_f0_target).astype(np.float32)

                tuning.f0_target[edited_mask] = resampled_old_target[edited_mask]
                tuning.f0_manually_edited = True

                self._pitch_roll.update_f0_target(tuning.f0_target)
                self._worker.set_params(self._audio, tuning.f0_target, self._sr,
                                        enable_advanced_f0=True)
                self._cache.invalidate_all()
                self._worker.wake()
        else:
            self._extract_pitch()

    def _extract_pitch(self):
        self._status.showMessage("提取音高中...")
        self._render_label.setText("提取音高...")
        self._render_progress.setVisible(False)
        try:
            if self._tracker is None:
                self._tracker = PitchTracker()
            f0 = self._tracker.extract_from_44k(self._audio)
        except Exception as e:
            show_friendly_exception(self, e, context="音高提取过程中发生错误")
            self._render_label.setText("错误")
            return

        tuning = self._tuning()
        tuning.f0_original = f0

        if self._btn_midi_snap.isChecked() and self._midi_notes:
            tuning.f0_target = self._build_midi_f0(f0)
        else:
            params = self._params.get_params()
            tuning.f0_target = snap_f0_to_scale(
                f0, root=params["key"], scale=params["scale"],
                retune_speed=params["retune_speed"]
            )

        if tuning.f0_original is None or len(tuning.f0_original) == 0:
            self._pitch_roll.set_data(tuning.f0_original, tuning.f0_target, len(self._audio), self._sr)
        else:
            self._pitch_roll._f0_original = tuning.f0_original.astype(np.float32)
            self._pitch_roll._f0_target = tuning.f0_target.astype(np.float32)
            self._pitch_roll.update()

        self._worker.set_params(self._audio, tuning.f0_target, self._sr,
                                enable_advanced_f0=True)
        self._cache.invalidate_all()
        self._worker.wake()
        self._render_label.setText(f"渲染中 (0/{len(self._cache.chunks)})")
        self._render_progress.setVisible(True)
        self._render_progress.setValue(0)
        self._status.showMessage("音高提取完成，渲染中...")

    def _build_midi_f0(self, f0_original: np.ndarray) -> np.ndarray:
        """Build f0_target snapped to MIDI notes with smooth transitions.

        - Inside a note: blend original F0 toward the MIDI target Hz using
          retune_speed (0=no change, 1=hard snap).
        - At note boundaries: short portamento ramp (up to 50ms) to avoid
          abrupt pitch jumps.
        - Outside any note: keep original F0 unchanged.
        """
        FPS = 100
        PORTAMENTO_FRAMES = 5   # ~50ms ramp at note start
        retune_speed = 1.0  # MIDI snap always uses full correction to target notes

        f0_target = f0_original.copy()

        for note in self._midi_notes:
            hz_target = 440.0 * (2.0 ** ((note.pitch - 69) / 12.0))
            start_frame = int(note.start_sec * FPS)
            end_frame   = min(int(note.end_sec * FPS), len(f0_target))

            for fi in range(start_frame, end_frame):
                if f0_original[fi] <= 0:
                    continue
                # Portamento: ramp blend factor from 0→retune_speed over first N frames
                ramp_pos = fi - start_frame
                if ramp_pos < PORTAMENTO_FRAMES:
                    blend = retune_speed * (ramp_pos / PORTAMENTO_FRAMES)
                else:
                    blend = retune_speed
                f0_target[fi] = f0_original[fi] + blend * (hz_target - f0_original[fi])

        return f0_target

    def _snap_f0_to_midi(self):
        if not self._project.clips or not self._midi_notes:
            return
        tuning = self._tuning()
        if len(tuning.f0_original) == 0:
            return
        tuning.f0_target = self._build_midi_f0(tuning.f0_original)
        self._pitch_roll.update_f0_target(tuning.f0_target)
        self._worker.set_params(self._audio, tuning.f0_target, self._sr,
                                enable_advanced_f0=True)
        self._cache.invalidate_all()
        self._worker.wake()
        self._status.showMessage("已将音高拉平到 MIDI 音符，渲染中...")

    def _save_project(self):
        # Sync current state to project before saving
        self._project.midi_notes = list(self._midi_notes)
        self._project.sample_rate = self._sr
        self._project.sample_folder = self._sample_folder or ""

        # Ensure current tuning data exists in tuning_map
        if self._project.clips:
            self._tuning()  # This ensures tuning data is in tuning_map

        if self._project_path:
            self._project.save(self._project_path)
            self._status.showMessage(f"已保存: {self._project_path}")
        else:
            self._save_project_as()

    def _save_project_as(self):
        # Sync current state to project before saving
        self._project.midi_notes = list(self._midi_notes)
        self._project.sample_rate = self._sr
        self._project.sample_folder = self._sample_folder or ""

        # Ensure current tuning data exists in tuning_map
        if self._project.clips:
            self._tuning()  # This ensures tuning data is in tuning_map

        path, _ = QFileDialog.getSaveFileName(
            self, "保存工程", self._project.name + ".ut",
            "UT 工程 (*.ut)"
        )
        if path:
            self._project_path = path
            self._project.save(path)
            self._status.showMessage(f"已保存: {path}")

    def _open_project(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "打开工程", "", "UT 工程 (*.ut)"
        )
        if not path:
            return
        try:
            self._project = ProjectState.load(path)
            self._project_path = path
        except Exception as e:
            show_friendly_exception(self, e, context="打开工程文件时发生错误")
            return

        # Load MIDI notes first (always available)
        self._midi_notes = self._project.midi_notes
        self._sr = self._project.sample_rate

        # Restore sample folder path
        if self._project.sample_folder:
            self._sample_folder = self._project.sample_folder

        if self._project.clips:
            clip = self._project.clips[0]
            audio_loaded = False

            # Try to load audio file if path exists
            if os.path.exists(clip.audio_path):
                try:
                    audio, sr = load_audio(clip.audio_path, target_sr=44100)
                    self._audio_raw = audio.copy()
                    self._audio = audio
                    self._sr = sr
                    intervals = split_into_chunks(audio, sr)
                    self._cache.reset(audio, intervals, sr)
                    self._waveform.set_audio(audio, sr)
                    self._pitch_roll.set_waveform(audio, sr)
                    self._transport.set_duration(len(audio) / sr)
                    audio_loaded = True
                except Exception as e:
                    show_friendly_exception(self, e, context="加载音频文件时发生错误")

            # Load tuning data and MIDI notes even if audio is missing
            tuning = self._tuning()
            self._params.set_params({
                "key": tuning.key, "scale": tuning.scale,
                "retune_speed": tuning.retune_speed,
            })

            # Set MIDI notes
            self._pitch_roll.set_midi_notes(self._midi_notes)
            self._lyrics_bar.set_notes(self._midi_notes)

            # Set pitch data if available
            if len(tuning.f0_original) > 0:
                audio_len = len(self._audio) if audio_loaded else len(tuning.f0_original) * 512
                self._pitch_roll.set_data(
                    tuning.f0_original, tuning.f0_target, audio_len, self._sr
                )

            # Start rendering if audio is loaded
            if audio_loaded:
                self._worker.set_params(self._audio, tuning.f0_target, self._sr,
                                        enable_advanced_f0=True)
                self._cache.invalidate_all()
                self._worker.wake()
            else:
                # Show warning if audio path doesn't exist
                if clip.audio_path.startswith("音源文件夹:"):
                    folder_name = clip.audio_path.replace("音源文件夹: ", "")
                    msg = f"工程使用了样本文件夹 '{folder_name}'，但音频未加载。\n\nMIDI 音符和音高数据已加载。"
                    if self._sample_folder and os.path.isdir(self._sample_folder):
                        msg += f"\n\n检测到样本文件夹路径：\n{self._sample_folder}\n\n是否立即重新导入？"
                        if confirm_action(self, "重新导入样本文件夹？", msg):
                            self._import_sample_folder_from(self._sample_folder)
                    else:
                        msg += "\n\n请使用 '文件 → 导入样本文件夹' 重新导入音源。"
                        show_info(self, "音频文件缺失", msg)
                else:
                    show_info(self, "音频文件缺失",
                             f"音频文件不存在: {clip.audio_path}\n"
                             f"MIDI 音符和音高数据已加载。")

        self._status.showMessage(f"已打开: {path}")

    def _export_audio(self):
        """Enhanced export with format selection."""
        if len(self._audio) == 0:
            show_info(self, "需要音频", "请先加载或生成音频文件")
            return

        from utils.audio_utils import EXPORT_FORMATS
        from core.audio_engine import AudioSettings

        settings = AudioSettings()

        # Build file dialog filter
        filters = []
        for fmt_name, fmt_info in EXPORT_FORMATS.items():
            ext = fmt_info['ext']
            desc = fmt_info['description']
            filters.append(f"{desc} (*{ext})")

        all_filter = ";;".join(filters)
        default_fmt = settings.get_export_format()
        filter_idx = list(EXPORT_FORMATS.keys()).index(default_fmt) if default_fmt in EXPORT_FORMATS else 0
        selected_filter = filters[filter_idx] if filter_idx < len(filters) else filters[0]

        path, selected = QFileDialog.getSaveFileName(
            self, "导出音频", "", all_filter, selected_filter
        )
        if not path:
            return

        # Auto-add extension if missing
        _, ext = os.path.splitext(path)
        if not ext:
            # Determine format from selected filter
            for fmt_name, fmt_info in EXPORT_FORMATS.items():
                if fmt_info['description'] in selected:
                    path += fmt_info['ext']
                    break

        # Get format settings
        format_name = settings.get_export_format()
        subtype = settings.get_export_subtype()
        apply_fade = settings.get_apply_fade()
        normalize = settings.get_normalize()

        try:
            # Stitch all rendered chunks
            out = self._cache.get_audio_at(0, self._cache.total_samples)

            # Apply stretch control points from MIDI notes
            from utils.audio_utils import apply_stretch_points
            for note in self._midi_notes:
                if note.stretch_points:
                    seg_start = int(note.start_sec * self._sr)
                    seg_end = int(note.end_sec * self._sr)
                    seg_end = min(seg_end, len(out))
                    if seg_start < seg_end:
                        seg_audio = out[seg_start:seg_end]
                        stretched = apply_stretch_points(seg_audio, self._sr, note)
                        out[seg_start:seg_start + len(stretched)] = stretched

            # Export with current settings
            self._engine.export(
                path,
                out,
                sr=self._sr,
                format_name=format_name,
                subtype=subtype,
                apply_fade=apply_fade,
                normalize=normalize,
            )

            file_size_mb = os.path.getsize(path) / (1024 * 1024)
            duration_sec = len(self._audio) / self._sr

            self._status.showMessage(f"已导出：{path} ({file_size_mb:.2f} MB, {duration_sec:.1f}s)")
            show_info(
                self, "导出成功",
                f"文件已保存至:\n{path}\n\n"
                f"格式：{format_name} ({subtype})\n"
                f"大小：{file_size_mb:.2f} MB\n"
                f"时长：{duration_sec:.1f} 秒\n"
                f"采样率：{self._sr} Hz"
            )

        except Exception as e:
            show_friendly_exception(self, e, context="导出音频时发生错误")
            self._status.showMessage("导出失败")

    def _open_audio_settings(self):
        """Open audio settings dialog."""
        dlg = AudioSettingsDialog(self._engine, self)
        dlg.settings_applied.connect(self._on_audio_settings_changed)
        dlg.exec()

    def _on_audio_settings_changed(self, settings_dict: dict):
        """Handle updated audio settings."""
        vol = settings_dict.get('volume', 1.0)
        self._status.showMessage(f"音量: {vol*100:.0f}% | 设备: {self._engine.current_device}")

    def _open_batch(self):
        dlg = BatchDialog(self)
        dlg.exec()

    # ── playback ──────────────────────────────────────────────────────────

    def _toggle_play(self):
        if self._engine.is_playing:
            self._engine.stop()
            self._transport.set_playing(False)
        else:
            if len(self._audio) == 0:
                return
            self._engine.play(self._engine.position_sec)
            self._transport.set_playing(True)

    def _stop(self):
        self._engine.stop()
        self._engine.seek(0.0)
        self._transport.set_playing(False)
        self._transport.set_position(0.0)
        self._pitch_roll.set_playhead(0.0)
        self._waveform.set_playhead(0.0)

    def _go_to_start(self):
        """Go to the start of the timeline."""
        self._seek(0.0)

    def _go_to_end(self):
        """Go to the end of the timeline."""
        if len(self._audio) > 0:
            self._seek(len(self._audio) / self._sr)

    def _seek_relative(self, delta_sec: float):
        """Seek relative to current position."""
        new_pos = self._engine.position_sec + delta_sec
        new_pos = max(0.0, min(new_pos, len(self._audio) / self._sr if len(self._audio) > 0 else 0.0))
        self._seek(new_pos)

    def _toggle_midi_snap(self):
        """Toggle MIDI pitch snap."""
        self._btn_midi_snap.setChecked(not self._btn_midi_snap.isChecked())
        self._on_midi_snap_toggled(self._btn_midi_snap.isChecked())

    def _zoom_to_fit(self):
        """Zoom waveform and pitch roll to fit all content."""
        self._waveform.zoom_to_fit()
        self._pitch_roll.reset_view()
        self._status.showMessage("已缩放至适合视图")

    def _undo(self):
        """Undo the last operation (global)."""
        # Delegate to pitch_roll if it has undo capability
        if hasattr(self._pitch_roll, '_undo'):
            self._pitch_roll._undo()
            self._status.showMessage("已撤销")
        else:
            self._status.showMessage("暂无可撤销的操作")

    def _redo(self):
        """Redo the last undone operation (global)."""
        # Delegate to pitch_roll if it has redo capability
        if hasattr(self._pitch_roll, '_redo'):
            self._pitch_roll._redo()
            self._status.showMessage("已重做")
        else:
            self._status.showMessage("暂无可重做的操作")

    def _render_all(self):
        """Force re-render all chunks."""
        if len(self._audio) > 0 and self._project.clips:
            self._cache.invalidate_all()
            self._worker.wake()
            self._status.showMessage("重新渲染中...")

    def _seek(self, sec: float):
        was_playing = self._engine.is_playing
        if was_playing:
            self._engine.stop()
        self._engine.seek(sec)
        self._transport.set_position(sec)
        self._pitch_roll.set_playhead(sec)
        self._waveform.set_playhead(sec)
        if was_playing:
            self._engine.play(sec)
            self._transport.set_playing(True)

    def _update_position(self):
        # Check stopped flag (set by audio thread, consumed here on main thread)
        if self._engine._stopped_flag:
            self._engine._stopped_flag = False
            self._on_playback_stopped()
        if self._engine.is_playing:
            sec = self._engine.position_sec
            self._transport.set_position(sec)
            self._pitch_roll.set_playhead(sec)
            self._waveform.set_playhead(sec)

    @pyqtSlot(str)
    def _on_audio_error(self, msg: str):
        """Handle audio engine errors."""
        self._status.showMessage(f"音频错误：{msg}")
        show_error(self, "audio_playback", details=msg)

    # ── parameter / pitch editing ─────────────────────────────────────────

    def _on_mode_draw(self):
        self._btn_edit.setChecked(False)
        self._btn_cut.setChecked(False)
        self._btn_draw.setChecked(True)
        self._pitch_roll.set_mode("draw_f0")

    def _on_mode_edit(self):
        self._btn_draw.setChecked(False)
        self._btn_cut.setChecked(False)
        self._btn_edit.setChecked(True)
        self._pitch_roll.set_mode("edit_notes")

    def _on_mode_cut(self):
        self._btn_draw.setChecked(False)
        self._btn_edit.setChecked(False)
        self._btn_cut.setChecked(True)
        self._pitch_roll.set_mode("cut")

    def _on_note_resized(self, idx: int, old_end: float, new_end: float):
        """Stretch the audio segment corresponding to the resized note."""
        if len(self._audio) == 0:
            return
        # Don't re-fetch notes from pitch_roll - they are already updated via signal
        # Just use the current self._midi_notes to avoid circular updates
        if idx >= len(self._midi_notes):
            return

        snapshot = self._capture_timeline_state()
        if snapshot is not None:
            self._timeline_undo_stack.append(snapshot)
            self._timeline_redo_stack.clear()

        note = self._midi_notes[idx]
        sr = self._sr
        seg_start = int(note.start_sec * sr)
        seg_end_old = int(old_end * sr)
        seg_end_old = min(seg_end_old, len(self._audio))
        segment = self._audio[seg_start:seg_end_old]
        if len(segment) == 0:
            return
        target_sec = new_end - note.start_sec
        stretched = time_stretch_to_duration(segment, sr, target_sec)
        delta = new_end - old_end
        before = self._audio[:seg_start]
        after  = self._audio[seg_end_old:]
        new_audio = np.concatenate([before, stretched, after])
        # Only shift subsequent notes when stretching (delta > 0).
        # When shrinking, leave other notes in place — they are independent.
        if delta > 0:
            for n in self._midi_notes:
                if n.start_sec >= old_end - 0.001:
                    n.start_sec += delta
                    n.end_sec   += delta
        self._audio = new_audio
        self._audio_raw = new_audio.copy()
        intervals = split_into_chunks(new_audio, sr)
        self._cache.reset(new_audio, intervals, sr)
        self._waveform.set_audio(new_audio, sr)
        self._pitch_roll.set_waveform(new_audio, sr)
        # Update pitch_roll's internal note list to match
        self._pitch_roll._midi_notes = self._midi_notes
        self._transport.set_duration(len(new_audio) / sr)
        # DO NOT re-extract pitch — preserve user-drawn F0 curve
        # Instead, just update the worker with the existing F0 target
        tuning = self._tuning()
        self._worker.set_params(new_audio, tuning.f0_target, self._sr,
                                enable_advanced_f0=True)
        self._cache.invalidate_all()
        self._worker.wake()

    def _on_note_cut(self, _idx: int, _cut_sec: float):
        """Sync note list after a cut operation and trigger audio rendering."""
        # Don't re-fetch notes from pitch_roll - they are already updated via signal
        # Just update the lyrics bar and trigger rendering
        self._lyrics_bar.set_notes(self._midi_notes)
        
        # Trigger real-time rendering to apply the cut
        if len(self._audio) > 0 and self._project.clips:
            self._apply_midi_notes_realtime()

    def _toggle_grid_snap(self):
        """Toggle grid snap on/off."""
        enabled = self._snap_action.isChecked()
        self._pitch_roll.set_grid_snap(enabled)
        status = "已启用" if enabled else "已禁用"
        self._status.showMessage(f"网格吸附：{status}")

    def _set_snap_resolution(self, resolution: float):
        """Set the grid snap resolution."""
        self._pitch_roll.set_snap_resolution(resolution)
        self._status.showMessage(f"网格吸附分辨率：{resolution}s")

    @pyqtSlot(bool)
    def _on_stretch_changed(self, is_timing_edit: bool = False):
        """Apply stretch control points to the render cache in real-time.

        Args:
            is_timing_edit: True if this is a timing edit (preserve pitch), False otherwise
        """
        # Drag is complete, apply final changes
        self._apply_midi_notes_realtime(preserve_pitch=is_timing_edit)
    
    @pyqtSlot()
    def _on_stretch_preview_direct(self):
        """Direct real-time preview during stretch point drag.

        Uses QTimer.singleShot to defer update after mouse event completes.
        """
        from PyQt6.QtCore import QTimer
        # Defer the actual update to let the mouse event complete first
        QTimer.singleShot(0, self._do_stretch_preview_update)

    def _do_stretch_preview_update(self):
        """Actually perform the stretch preview update."""
        from utils.audio_utils import apply_stretch_points
        from PyQt6.QtWidgets import QApplication
        import hashlib

        if not self._midi_notes or len(self._audio_raw) == 0:
            return

        # Quick rebuild for visual preview only
        audio = self._audio_raw.copy()
        sr = self._sr

        # Calculate hash before stretch
        hash_before = hashlib.md5(audio[:10000].tobytes()).hexdigest()[:8]

        # Apply stretch control points for preview
        stretch_applied = False
        for note in self._midi_notes:
            if note.stretch_points:
                has_moved = any(abs(sp.position - sp.orig_position) > 0.01 for sp in note.stretch_points)
                if has_moved:
                    seg_start = int(note.start_sec * sr)
                    seg_end = int(note.end_sec * sr)
                    seg_end = min(seg_end, len(audio))
                    if seg_start >= seg_end:
                        continue

                    seg_audio = audio[seg_start:seg_end]
                    try:
                        stretched = apply_stretch_points(seg_audio, sr, note)
                        replace_len = min(len(stretched), seg_end - seg_start)
                        audio[seg_start:seg_start + replace_len] = stretched[:replace_len]
                        stretch_applied = True
                    except Exception as e:
                        print(f"[PREVIEW] Error applying stretch: {e}")
                        # Continue processing other notes even if one fails
                        pass

        # Calculate hash after stretch
        hash_after = hashlib.md5(audio[:10000].tobytes()).hexdigest()[:8]

        if stretch_applied:
            print(f"[PREVIEW] Audio changed: {hash_before} -> {hash_after}")
        else:
            print(f"[PREVIEW] No stretch applied")

        # Update waveform display
        self._waveform.set_audio(audio, sr)
        QApplication.processEvents()

    def _apply_midi_notes_realtime(self, preserve_pitch: bool = False):
        from utils.audio_utils import apply_stretch_points

        if not self._midi_notes or len(self._audio) == 0:
            return

        if not self._project.clips:
            return

        tuning = self._tuning()

        f0_original = tuning.f0_target.copy() if len(tuning.f0_target) > 0 else tuning.f0_target
        f0_to_use = f0_original

        audio = self._audio_raw.copy()
        sr = self._sr
        has_changes = False

        for note in self._midi_notes:
            if note.stretch_points:
                has_moved = any(abs(sp.position - sp.orig_position) > 0.01 for sp in note.stretch_points)
                if has_moved:
                    has_changes = True
                    seg_start = int(note.start_sec * sr)
                    seg_end = int(note.end_sec * sr)
                    seg_end = min(seg_end, len(audio))
                    if seg_start >= seg_end:
                        continue

                    seg_audio = audio[seg_start:seg_end]
                    try:
                        stretched = apply_stretch_points(seg_audio, sr, note)
                        replace_len = min(len(stretched), seg_end - seg_start)
                        audio[seg_start:seg_start + replace_len] = stretched[:replace_len]
                    except Exception as e:
                        print(f"[realtime] Error applying stretch for note {note.pitch}: {e}")
                        self._status.showMessage(f"音符 {note.pitch} 拉伸失败，请检查控制点设置", 3000)

        if not has_changes:
            audio = self._audio.copy()

        if len(f0_original) > 0:
            old_dur = len(self._audio) / sr
            new_dur = len(audio) / sr

            if abs(new_dur - old_dur) > 0.001:
                fps = 100
                old_f0_len = len(f0_original)
                new_f0_len = int(new_dur * fps)

                if new_f0_len > 0 and new_f0_len != old_f0_len:
                    import numpy as np
                    old_indices = np.linspace(0, old_f0_len - 1, old_f0_len)
                    new_indices = np.linspace(0, old_f0_len - 1, new_f0_len)
                    f0_to_use = np.interp(new_indices, old_indices, f0_original).astype(np.float32)
                    tuning.f0_target = f0_to_use.copy()
                    self._pitch_roll.update_f0_target(tuning.f0_target)
            elif has_changes:
                f0_to_use = f0_original
        else:
            f0_to_use = tuning.f0_target

        if len(tuning.f0_target) > 0:
            self._pitch_roll.update_f0_target(tuning.f0_target)

        self._worker.set_params(audio, f0_to_use, self._sr)
        self._cache.invalidate_all()
        self._worker.wake()

        if has_changes:
            self._audio = audio
            self._waveform.set_audio(audio, sr)
            self._waveform.update()
            self._pitch_roll.set_waveform(audio, sr)
            self._pitch_roll.update()
            self._transport.set_duration(len(audio) / sr)
            self._cache.replace_dry_region(0, audio)

        self._status.showMessage("实时预览中...")

    def _on_midi_snap_toggled(self, checked: bool):
        if checked:
            if self._midi_notes and self._project.clips and len(self._tuning().f0_original) > 0:
                self._snap_f0_to_midi()
        else:
            if self._project.clips and len(self._tuning().f0_original) > 0:
                tuning = self._tuning()

                has_manual_edit = (
                    tuning.f0_target is not None and
                    len(tuning.f0_target) > 0 and
                    not np.allclose(tuning.f0_target, tuning.f0_original, equal_nan=True)
                )

                if has_manual_edit:
                    self._worker.set_params(self._audio, tuning.f0_target, self._sr,
                                            enable_advanced_f0=True)
                    self._cache.invalidate_all()
                    self._worker.wake()
                    self._status.showMessage("已恢复音阶校正模式（保留手动编辑），渲染中...")
                else:
                    params = self._params.get_params()
                    tuning.f0_target = snap_f0_to_scale(
                        tuning.f0_original, root=params["key"], scale=params["scale"],
                        retune_speed=params["retune_speed"]
                    )
                    self._pitch_roll.update_f0_target(tuning.f0_target)
                    self._worker.set_params(self._audio, tuning.f0_target, self._sr,
                                            enable_advanced_f0=True)
                    self._cache.invalidate_all()
                    self._worker.wake()
                    self._status.showMessage("已恢复音阶校正模式，渲染中...")

    def _on_lyrics_changed(self, notes: list):
        old_midi_notes = list(self._midi_notes)
        old_map = {}
        for i, n in enumerate(old_midi_notes):
            if n.stretch_points:
                key = (round(n.start_sec, 3), round(n.end_sec, 3), n.pitch)
                old_map[key] = n.stretch_points

        self._midi_notes = list(notes)

        for i, n in enumerate(self._midi_notes):
            key = (round(n.start_sec, 3), round(n.end_sec, 3), n.pitch)
            if key in old_map and not n.stretch_points:
                n.stretch_points = old_map[key]

        self._is_handling_lyrics_change = True
        try:
            self._pitch_roll.set_midi_notes(self._midi_notes)
        finally:
            self._is_handling_lyrics_change = False

        self._lyrics_bar.set_notes(self._midi_notes)

        if self._sample_folder and os.path.isdir(self._sample_folder):
            self._import_sample_folder_from(self._sample_folder, force_reextract=True)
        elif len(self._audio) > 0:
            self._apply_midi_notes_realtime()

    def _on_midi_notes_changed(self, notes: list):
        if getattr(self, '_is_handling_lyrics_change', False):
            return

        if len(self._audio) > 0 and self._project.clips and len(self._midi_notes) > 0:
            import copy
            self._midi_notes_before_edit = copy.deepcopy(self._midi_notes)

        self._midi_notes = list(notes)

        if len(self._audio) > 0 and self._project.clips:
            self._apply_midi_notes_realtime()

    @pyqtSlot(dict)
    def _on_params_changed(self, params: dict):
        if not self._project.clips:
            return
        tuning = self._tuning()
        tuning.key = params["key"]
        tuning.scale = params["scale"]
        tuning.retune_speed = params["retune_speed"]

        if len(tuning.f0_original) > 0:
            if tuning.f0_manually_edited:
                self._worker.set_params(self._audio, tuning.f0_target, self._sr)
            else:
                tuning.f0_target = snap_f0_to_scale(
                    tuning.f0_original, root=tuning.key, scale=tuning.scale,
                    retune_speed=tuning.retune_speed
                )
                self._pitch_roll.update_f0_target(tuning.f0_target)
                self._worker.set_params(self._audio, tuning.f0_target, self._sr)

            self._cache.invalidate_all()
            self._worker.wake()

    @pyqtSlot(np.ndarray)
    def _on_f0_edited(self, f0: np.ndarray):
        if not self._project.clips:
            return
        tuning = self._tuning()
        tuning.f0_target = f0
        if not hasattr(self, '_is_restoring_history') or not self._is_restoring_history:
            tuning.f0_manually_edited = True
        self._pitch_roll.update_f0_target(tuning.f0_target)
        self._worker.set_params(self._audio, f0, self._sr)
        self._cache.invalidate_all()
        self._worker.wake()

    def _reextract_pitch(self):
        if len(self._audio) > 0:
            self._extract_pitch()

    def _reextract_pitch_without_reset_view(self):
        if len(self._audio) == 0:
            return
        self._status.showMessage("提取音高中...")
        self._render_label.setText("提取音高...")
        try:
            if self._tracker is None:
                self._tracker = PitchTracker()
            f0 = self._tracker.extract_from_44k(self._audio)
        except Exception as e:
            show_friendly_exception(self, e, context="音高提取失败")
            self._render_label.setText("错误")
            return

        tuning = self._tuning()
        tuning.f0_original = f0

        if self._btn_midi_snap.isChecked() and self._midi_notes:
            tuning.f0_target = self._build_midi_f0(f0)
        else:
            params = self._params.get_params()
            tuning.f0_target = snap_f0_to_scale(
                f0, root=params["key"], scale=params["scale"],
                retune_speed=params["retune_speed"]
            )

        self._pitch_roll.update_f0_target(tuning.f0_target)
        self._worker.set_params(self._audio, tuning.f0_target, self._sr,
                                enable_advanced_f0=True)
        self._cache.invalidate_all()
        self._worker.wake()
        self._render_label.setText(f"渲染中 (0/{len(self._cache.chunks)})")

    def _apply_correction(self):
        self._on_params_changed(self._params.get_params())

    def _reset_target(self):
        if not self._project.clips:
            return
        tuning = self._tuning()
        if len(tuning.f0_original) > 0:
            tuning.f0_target = tuning.f0_original.copy()
            self._pitch_roll.update_f0_target(tuning.f0_target)
            self._worker.set_params(self._audio, tuning.f0_target, self._sr)
            self._cache.invalidate_all()
            self._worker.wake()

    # ── worker callbacks ──────────────────────────────────────────────────

    @pyqtSlot(int)
    def _on_chunk_done(self, idx: int):
        done = sum(1 for c in self._cache.chunks if c.status.name == "SUCCEEDED")
        total = len(self._cache.chunks)
        self._render_label.setText(f"渲染 {done}/{total}")
        progress_pct = int((done / total) * 100) if total > 0 else 0
        self._render_progress.setValue(progress_pct)
        if done == total:
            self._render_label.setText("渲染完成 ✓")
            # Auto-hide progress bar after 2 seconds
            QTimer.singleShot(2000, lambda: self._render_progress.setVisible(False))

    @pyqtSlot(str)
    def _on_worker_error(self, msg: str):
        self._status.showMessage(f"渲染错误: {msg}")

    @pyqtSlot()
    def _on_playback_stopped(self):
        self._transport.set_playing(False)
    
    def _on_select_all(self):
        """Handle Ctrl+A - delegate to pitch_roll if in edit mode."""
        if self._pitch_roll:
            # PitchRoll handles Ctrl+A internally in edit_notes mode
            pass
    
    def _on_copy(self):
        """Handle Ctrl+C - placeholder for future copy functionality."""
        show_info(self, "功能开发中", "复制功能正在开发中，敬请期待！")
    
    def _on_paste(self):
        """Handle Ctrl+V - placeholder for future paste functionality."""
        show_info(self, "功能开发中", "粘贴功能正在开发中，敬请期待！")

    def _open_model_docs(self):
        from PyQt6.QtWidgets import QDialog, QTextEdit, QVBoxLayout, QPushButton
        import io, model_docs_generator as mdg

        buf = io.StringIO()
        import sys
        old_stdout = sys.stdout
        sys.stdout = buf
        try:
            mdg.inspect_onnx_model(
                mdg.os.path.join(mdg.MODEL_DIR, "rmvpe.onnx"), "RMVPE Pitch Extractor"
            )
            mdg.inspect_onnx_model(
                mdg.os.path.join(mdg.MODEL_DIR, "hifigan.onnx"), "NSF-HiFiGAN Vocoder"
            )
            buf.write(mdg.generate_complete_documentation())
        except Exception as e:
            buf.write(f"\n错误：{e}\n")
        finally:
            sys.stdout = old_stdout

        dlg = QDialog(self)
        dlg.setWindowTitle("模型诊断")
        dlg.resize(800, 600)
        layout = QVBoxLayout(dlg)
        text = QTextEdit()
        text.setReadOnly(True)
        text.setFontFamily("Courier New")
        text.setPlainText(buf.getvalue())
        layout.addWidget(text)
        btn = QPushButton("关闭")
        btn.clicked.connect(dlg.accept)
        layout.addWidget(btn)
        dlg.exec()

    def _show_performance_report(self):
        """Show performance monitoring report."""
        from PyQt6.QtWidgets import QDialog, QTextEdit, QVBoxLayout, QPushButton, QHBoxLayout
        
        monitor = get_monitor()
        report = monitor.report()
        
        # Add mel cache stats
        try:
            from utils.mel_cache import get_mel_cache
            mel_cache = get_mel_cache()
            report += "\n\n" + mel_cache.report()
        except:
            pass
        
        dlg = QDialog(self)
        dlg.setWindowTitle("性能报告")
        dlg.resize(800, 600)
        layout = QVBoxLayout(dlg)
        
        text = QTextEdit()
        text.setReadOnly(True)
        text.setFontFamily("Courier New")
        text.setPlainText(report)
        layout.addWidget(text)
        
        # Buttons
        btn_layout = QHBoxLayout()
        btn_copy = QPushButton("复制报告")
        btn_copy.clicked.connect(lambda: text.copy())
        btn_layout.addWidget(btn_copy)
        
        btn_reset = QPushButton("重置统计")
        btn_reset.clicked.connect(lambda: self._reset_performance_stats())
        btn_layout.addWidget(btn_reset)
        
        btn_layout.addStretch()
        
        btn_close = QPushButton("关闭")
        btn_close.clicked.connect(dlg.accept)
        btn_layout.addWidget(btn_close)
        
        layout.addLayout(btn_layout)
        dlg.exec()
    
    def _reset_performance_stats(self):
        """Reset performance monitoring statistics."""
        monitor = get_monitor()
        monitor.reset()
        try:
            from utils.mel_cache import get_mel_cache
            mel_cache = get_mel_cache()
            mel_cache.clear()
        except:
            pass
        self._status.showMessage("性能统计已重置")

    def _show_about(self):
        QMessageBox.about(
            self,
            "关于 UT",
            "<h3>UT V1.1</h3>"
            "<p>专业音高修正工具</p>"
            "<p>基于 RMVPE 音高提取 + NSF-HiFiGAN 声码器</p>"
            "<p>支持实时预览、批量处理、自定义音阶</p>"
            "<hr>"
            "<p><b>常用快捷键：</b></p>"
            "<p><b>播放控制：</b></p>"
            "<ul>"
            "<li><b>空格</b> - 播放/暂停</li>"
            "<li><b>S</b> - 停止</li>"
            "<li><b>Home</b> - 跳到开始</li>"
            "<li><b>End</b> - 跳到结尾</li>"
            "<li><b>←/→</b> - 快退/快进 5 秒</li>"
            "<li><b>Shift+←/→</b> - 快退/快进 1 秒</li>"
            "</ul>"
            "<p><b>编辑操作：</b></p>"
            "<ul>"
            "<li><b>Ctrl+Z</b> - 撤销</li>"
            "<li><b>Ctrl+Y</b> - 重做</li>"
            "<li><b>F1</b> - 绘制音高模式</li>"
            "<li><b>F2</b> - 编辑音符模式</li>"
            "<li><b>F3</b> - 切换 MIDI 对齐</li>"
            "<li><b>Delete</b> - 删除选中音符</li>"
            "</ul>"
            "<p><b>视图导航：</b></p>"
            "<ul>"
            "<li><b>+/-</b> - 时间轴缩放</li>"
            "<li><b>[ ]</b> - 音高轴缩放</li>"
            "<li><b>↑↓←→</b> - 平移视图</li>"
            "<li><b>0 或 F</b> - 缩放至适合</li>"
            "<li><b>Home</b> - 重置视图</li>"
            "</ul>"
            "<p><b>文件操作：</b></p>"
            "<ul>"
            "<li><b>Ctrl+O</b> - 打开音频</li>"
            "<li><b>Ctrl+M</b> - 导入 MIDI</li>"
            "<li><b>Ctrl+S</b> - 保存工程</li>"
            "<li><b>Ctrl+E</b> - 导出音频</li>"
            "</ul>"
            "<p><b>音高修正：</b></p>"
            "<ul>"
            "<li><b>Ctrl+P</b> - 重新提取音高</li>"
            "<li><b>Ctrl+K</b> - 应用修正</li>"
            "<li><b>Ctrl+T</b> - 重置目标音高</li>"
            "<li><b>F5</b> - 重新渲染</li>"
            "</ul>"
        )

    # ── cleanup ───────────────────────────────────────────────────────────

    def _set_theme(self, theme_name):
        """Switch between light and dark themes."""
        from ui import styles
        styles.set_theme(theme_name)

        # Update stylesheet
        QApplication.instance().setStyleSheet(styles.get_stylesheet())

        # Update theme action checkboxes
        self._theme_dark_action.setChecked(theme_name == "dark")
        self._theme_light_action.setChecked(theme_name == "light")

        # Update waveform view theme
        self._waveform.update_theme()

        # Force repaint of all widgets
        self._pitch_roll.update()
        self._waveform.update()
        self.update()

        self._status.showMessage(f"已切换到{'暗色' if theme_name == 'dark' else '亮色'}主题")

    def closeEvent(self, event):
        self._engine.stop()
        self._worker.stop()
        self._worker.wait(3000)
        event.accept()

    # ── drag & drop ──────────────────────────────────────────────────────

    def dragEnterEvent(self, event: QDragEnterEvent):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event: QDropEvent):
        urls = event.mimeData().urls()
        if urls:
            path = urls[0].toLocalFile()
            if path and os.path.exists(path):
                ext = os.path.splitext(path)[1].lower()
                if ext in ['.wav', '.flac', '.mp3', '.ogg', '.aiff']:
                    self._load_audio_file(path)
