"""QSS themes and color constants."""

# Theme management
_current_theme = "dark"  # default theme

# Dark theme colors
DARK_THEME = {
    "BG": "#1a1a2e",
    "PANEL": "#16213e",
    "ACCENT": "#0f3460",
    "THEME": "#e94560",
    "TEXT": "#eaeaea",
    "TEXT_DIM": "#a0a0a0",
    "F0_ORIGINAL": "#ef4444",
    "F0_TARGET": "#3b82f6",
    "MIDI_NOTE": "#22c55e",
    "WAVEFORM": "#06b6d4",
    "LYRIC": "#fbbf24",
    "GRID_MAJOR": "#4a4a8a",
    "GRID_MINOR": "#4a4a6e",
    "GRID_SUBTLE": "#222240",
    "PIANO_BLACK": "#2a2a4e",
    "PIANO_WHITE": "#4a4a6e",
}

# Light theme colors
LIGHT_THEME = {
    "BG": "#f5f5f5",
    "PANEL": "#ffffff",
    "ACCENT": "#e0e0e0",
    "THEME": "#2196f3",
    "TEXT": "#212121",
    "TEXT_DIM": "#757575",
    "F0_ORIGINAL": "#f44336",
    "F0_TARGET": "#2196f3",
    "MIDI_NOTE": "#4caf50",
    "WAVEFORM": "#00bcd4",
    "LYRIC": "#ff9800",
    "GRID_MAJOR": "#bdbdbd",
    "GRID_MINOR": "#e0e0e0",
    "GRID_SUBTLE": "#f5f5f5",
    "PIANO_BLACK": "#e0e0e0",
    "PIANO_WHITE": "#fafafa",
}

# Current theme colors (will be updated by set_theme)
_colors = DARK_THEME.copy()

# Export color constants
BG = _colors["BG"]
PANEL = _colors["PANEL"]
ACCENT = _colors["ACCENT"]
THEME = _colors["THEME"]
TEXT = _colors["TEXT"]
TEXT_DIM = _colors["TEXT_DIM"]
F0_ORIGINAL = _colors["F0_ORIGINAL"]
F0_TARGET = _colors["F0_TARGET"]
MIDI_NOTE = _colors["MIDI_NOTE"]
WAVEFORM = _colors["WAVEFORM"]
LYRIC = _colors["LYRIC"]
GRID_MAJOR = _colors["GRID_MAJOR"]
GRID_MINOR = _colors["GRID_MINOR"]
GRID_SUBTLE = _colors["GRID_SUBTLE"]
PIANO_BLACK = _colors["PIANO_BLACK"]
PIANO_WHITE = _colors["PIANO_WHITE"]


def get_current_theme():
    """Get current theme name."""
    return _current_theme


def set_theme(theme_name):
    """Set current theme and update color constants."""
    global _current_theme, _colors
    global BG, PANEL, ACCENT, THEME, TEXT, TEXT_DIM
    global F0_ORIGINAL, F0_TARGET, MIDI_NOTE, WAVEFORM, LYRIC
    global GRID_MAJOR, GRID_MINOR, GRID_SUBTLE, PIANO_BLACK, PIANO_WHITE

    _current_theme = theme_name
    _colors = LIGHT_THEME.copy() if theme_name == "light" else DARK_THEME.copy()

    # Update module-level constants
    BG = _colors["BG"]
    PANEL = _colors["PANEL"]
    ACCENT = _colors["ACCENT"]
    THEME = _colors["THEME"]
    TEXT = _colors["TEXT"]
    TEXT_DIM = _colors["TEXT_DIM"]
    F0_ORIGINAL = _colors["F0_ORIGINAL"]
    F0_TARGET = _colors["F0_TARGET"]
    MIDI_NOTE = _colors["MIDI_NOTE"]
    WAVEFORM = _colors["WAVEFORM"]
    LYRIC = _colors["LYRIC"]
    GRID_MAJOR = _colors["GRID_MAJOR"]
    GRID_MINOR = _colors["GRID_MINOR"]
    GRID_SUBTLE = _colors["GRID_SUBTLE"]
    PIANO_BLACK = _colors["PIANO_BLACK"]
    PIANO_WHITE = _colors["PIANO_WHITE"]


def get_stylesheet():
    """Generate stylesheet for current theme."""
    return f"""
QMainWindow, QDialog {{
    background-color: {BG};
    color: {TEXT};
}}
QWidget {{
    background-color: {BG};
    color: {TEXT};
    font-family: "Microsoft YaHei", "PingFang SC", Arial, sans-serif;
    font-size: 10pt;
}}
QMenuBar {{
    background-color: {PANEL};
    color: {TEXT};
    border-bottom: 1px solid {ACCENT};
}}
QMenuBar::item:selected {{
    background-color: {ACCENT};
}}
QMenu {{
    background-color: {PANEL};
    color: {TEXT};
    border: 1px solid {ACCENT};
}}
QMenu::item:selected {{
    background-color: {THEME};
}}
QPushButton {{
    background-color: {ACCENT};
    color: {TEXT};
    border: none;
    border-radius: 4px;
    padding: 6px 14px;
    font-size: 10pt;
}}
QPushButton:hover {{
    background-color: {THEME};
}}
QPushButton:pressed {{
    background-color: {THEME};
    opacity: 0.8;
}}
QPushButton:checked {{
    background-color: {THEME};
    color: white;
    border: 1px solid {THEME};
}}
QPushButton:checked:hover {{
    background-color: {THEME};
    opacity: 0.9;
}}
QPushButton#midi_snap:checked {{
    background-color: #16a34a;
    border: 1px solid #4ade80;
}}
QPushButton#midi_snap:checked:hover {{
    background-color: #15803d;
}}
QSlider::groove:horizontal {{
    height: 4px;
    background: {ACCENT};
    border-radius: 2px;
}}
QSlider::sub-page:horizontal {{
    background: {THEME};
    border-radius: 2px;
}}
QSlider::handle:horizontal {{
    width: 16px;
    height: 16px;
    margin: -6px 0;
    background: white;
    border-radius: 8px;
    border: 2px solid {THEME};
}}
QComboBox {{
    background-color: {ACCENT};
    color: {TEXT};
    border: 1px solid {THEME};
    border-radius: 4px;
    padding: 4px 8px;
}}
QComboBox::drop-down {{
    border: none;
}}
QComboBox QAbstractItemView {{
    background-color: {PANEL};
    color: {TEXT};
    selection-background-color: {THEME};
}}
QLabel {{
    color: {TEXT};
    background: transparent;
}}
QSplitter::handle {{
    background-color: {ACCENT};
}}
QScrollBar:vertical {{
    background: {PANEL};
    width: 8px;
}}
QScrollBar::handle:vertical {{
    background: {ACCENT};
    border-radius: 4px;
    min-height: 20px;
}}
QScrollBar::handle:vertical:hover {{
    background: {THEME};
}}
QScrollBar:horizontal {{
    background: {PANEL};
    height: 8px;
}}
QScrollBar::handle:horizontal {{
    background: {ACCENT};
    border-radius: 4px;
    min-width: 20px;
}}
QScrollBar::handle:horizontal:hover {{
    background: {THEME};
}}
QListWidget {{
    background-color: {PANEL};
    border: 1px solid {ACCENT};
    border-radius: 4px;
}}
QListWidget::item:selected {{
    background-color: {THEME};
}}
QProgressBar {{
    background-color: {PANEL};
    border: 1px solid {ACCENT};
    border-radius: 4px;
    text-align: center;
    color: {TEXT};
}}
QProgressBar::chunk {{
    background-color: {THEME};
    border-radius: 3px;
}}
QLineEdit {{
    background-color: {PANEL};
    color: {TEXT};
    border: 1px solid {ACCENT};
    border-radius: 4px;
    padding: 4px 8px;
}}
QLineEdit:focus {{
    border-color: {THEME};
}}
QGroupBox {{
    color: {TEXT_DIM};
    border: 1px solid {ACCENT};
    border-radius: 4px;
    margin-top: 8px;
    padding-top: 8px;
}}
QGroupBox::title {{
    subcontrol-origin: margin;
    left: 8px;
    color: {TEXT_DIM};
}}
QToolTip {{
    background-color: {PANEL};
    color: {TEXT};
    border: 1px solid {THEME};
    border-radius: 4px;
    padding: 4px;
}}
"""

# Legacy constant for backward compatibility
STYLESHEET = get_stylesheet()

