from PyQt6.QtCore import QSettings

_settings = QSettings("OpenTunePro", "OpenTunePro")


def get(key: str, default=None):
    return _settings.value(key, default)


def set(key: str, value) -> None:
    _settings.setValue(key, value)


def get_model_dir() -> str:
    import os
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return get("model_dir", os.path.join(here, "models"))
