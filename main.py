"""UT - entry point."""
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from PyQt6.QtWidgets import QApplication
from PyQt6.QtCore import Qt
from ui.main_window import MainWindow
from ui.styles import STYLESHEET


def main():
    app = QApplication(sys.argv)
    app.setApplicationName("UT")
    app.setOrganizationName("UT")
    app.setStyleSheet(STYLESHEET)

    window = MainWindow()
    window.show()

    # If a file was passed as argument, open it
    if len(sys.argv) > 1 and os.path.exists(sys.argv[1]):
        window._load_audio_file(sys.argv[1])

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
