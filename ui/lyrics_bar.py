"""Lyrics input bar — sits below the pitch roll.

User types space-separated syllables (e.g. "shi ni wo"), selects notes in
the pitch roll, then clicks Apply to assign syllables to the selected notes
in order. Unselected notes are not affected.
"""
from __future__ import annotations
from PyQt6.QtCore import pyqtSignal
from PyQt6.QtWidgets import QWidget, QHBoxLayout, QLabel, QLineEdit, QPushButton

from utils.audio_utils import MidiNote
from ui import styles


class LyricsBar(QWidget):
    """Bottom bar for lyric input.

    Signals:
        lyrics_changed(list[MidiNote]) — emitted after Apply is clicked with
            the full note list (selected notes updated, others unchanged).
    """
    lyrics_changed = pyqtSignal(list)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedHeight(36)
        self._notes: list[MidiNote] = []
        self._selected: set[int] = set()   # indices of currently selected notes

        layout = QHBoxLayout(self)
        layout.setContentsMargins(8, 4, 8, 4)
        layout.setSpacing(8)

        label = QLabel("歌词:")
        label.setFixedWidth(40)
        layout.addWidget(label)

        self._edit = QLineEdit()
        self._edit.setPlaceholderText("输入音节，空格分隔，如: shi ni wo")
        self._edit.textChanged.connect(self._update_button_state)
        layout.addWidget(self._edit)

        self._count_label = QLabel("选中 0 个音符")
        self._count_label.setFixedWidth(90)
        self._count_label.setStyleSheet(f"color: {styles.TEXT_DIM};")
        layout.addWidget(self._count_label)

        self._apply_btn = QPushButton("Apply")
        self._apply_btn.setFixedWidth(60)
        self._apply_btn.setFixedHeight(26)
        self._apply_btn.setEnabled(False)
        self._apply_btn.clicked.connect(self._apply)
        layout.addWidget(self._apply_btn)

    def set_notes(self, notes: list[MidiNote]) -> None:
        self._notes = list(notes)
        self._update_button_state()

    def set_selected(self, selected: set[int]) -> None:
        """Called by main window whenever pitch roll selection changes."""
        self._selected = set(selected)
        n = len(self._selected)
        self._count_label.setText(f"选中 {n} 个音符")
        self._update_button_state()

    def get_text(self) -> str:
        return self._edit.text()

    def set_text(self, text: str) -> None:
        self._edit.setText(text)

    def _syllables(self) -> list[str]:
        text = self._edit.text().strip()
        return [s for s in text.split() if s]

    def _update_button_state(self) -> None:
        has_selection = bool(self._selected)
        has_text = bool(self._syllables())
        self._apply_btn.setEnabled(has_selection and has_text)

    def _apply(self) -> None:
        syllables = self._syllables()
        if not syllables or not self._selected:
            return

        # Selected note indices sorted by start time
        ordered = sorted(self._selected,
                         key=lambda i: self._notes[i].start_sec
                         if i < len(self._notes) else 0)

        for slot, note_idx in enumerate(ordered):
            if note_idx < len(self._notes):
                self._notes[note_idx].lyric = syllables[slot] if slot < len(syllables) else ""

        self.lyrics_changed.emit(self._notes)
