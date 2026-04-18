"""Enhanced audio engine with device selection, volume control, and buffer optimization."""
from __future__ import annotations
import threading
import numpy as np
import sounddevice as sd
import soundfile as sf
from PyQt6.QtCore import QObject, pyqtSignal
from typing import Optional

from core.render_cache import RenderCache
from utils.audio_utils import load_audio, save_audio


class AudioEngine(QObject):
    position_changed = pyqtSignal(float)
    playback_stopped = pyqtSignal()
    error_occurred = pyqtSignal(str)

    def __init__(self, cache: RenderCache, parent=None):
        super().__init__(parent)
        self._cache = cache

        # Audio configuration
        self._sr = 44100
        self._channels = 1
        self._dtype = "float32"

        # Playback state
        self._pos = 0
        self._playing = False
        self._stopped_flag = False
        self._stream: Optional[sd.OutputStream] = None
        self._lock = threading.Lock()

        # Volume control (0.0 - 2.0, where 1.0 = unity gain)
        self._volume = 1.0

        # Buffer settings (tune for latency vs stability)
        self._blocksize = 2048       # samples per callback
        self._latency = "low"         # low/medium/high
        self._buffer_config = {
            "low": {"blocksize": 1024, "latency": "low"},
            "medium": {"blocksize": 2048, "latency": "medium"},
            "high": {"blocksize": 4096, "latency": "high"},
        }

        # Device selection (None = default device)
        self._device_id: Optional[int] = None
        self._device_name: str = ""

    def get_output_devices(self) -> list[dict]:
        """Get list of available output audio devices."""
        devices = []
        try:
            device_list = sd.query_devices()
            for i, dev in enumerate(device_list):
                if dev['max_output_channels'] > 0:
                    devices.append({
                        'id': i,
                        'name': dev['name'],
                        'channels': dev['max_output_channels'],
                        'default_sr': dev['default_samplerate'],
                        'is_default': i == sd.default.device[1],
                    })
        except Exception as e:
            self.error_occurred.emit(f"Failed to query audio devices: {e}")
        return devices

    def set_device(self, device_id: Optional[int]):
        """Set output device by ID."""
        if self._playing:
            self.stop()
        self._device_id = device_id
        if device_id is not None:
            try:
                info = sd.query_devices(device_id)
                self._device_name = info['name']
            except Exception:
                self._device_name = f"Device {device_id}"
        else:
            self._device_name = "Default Device"

    def set_buffer_quality(self, quality: str):
        """Set buffer size preset: 'low', 'medium', or 'high' latency."""
        if quality in self._buffer_config:
            config = self._buffer_config[quality]
            self._blocksize = config['blocksize']
            self._latency = config['latency']

    def set_volume(self, volume: float):
        """
        Set playback volume.

        Args:
            volume: Volume level (0.0 = mute, 1.0 = normal, 2.0 = +6dB boost)
        """
        self._volume = max(0.0, min(2.0, volume))

    @property
    def volume(self) -> float:
        return self._volume

    def load(self, path: str) -> tuple[np.ndarray, int]:
        """Load audio file and prepare for playback."""
        audio, sr = load_audio(path, target_sr=self._sr)
        self._sr = sr
        self._pos = 0
        return audio, sr

    def play(self, start_sec: float = 0.0):
        """Start or resume playback from position."""
        self.stop()
        with self._lock:
            self._pos = int(start_sec * self._sr)
            self._playing = True
            self._stopped_flag = False

        # Build stream kwargs with optional device selection
        stream_kwargs = dict(
            samplerate=self._sr,
            channels=self._channels,
            dtype=self._dtype,
            blocksize=self._blocksize,
            callback=self._callback,
            finished_callback=self._on_finished,
        )

        if self._device_id is not None:
            stream_kwargs['device'] = self._device_id

        try:
            self._stream = sd.OutputStream(**stream_kwargs)
            self._stream.start()
        except Exception as e:
            self.error_occurred.emit(f"Failed to start playback: {e}")
            with self._lock:
                self._playing = False
                self._stopped_flag = True

    def stop(self):
        """Stop playback immediately."""
        with self._lock:
            self._playing = False
        if self._stream is not None:
            try:
                self._stream.stop()
                self._stream.close()
            except Exception:
                pass
            finally:
                self._stream = None

    def seek(self, sec: float):
        """Seek to position in seconds."""
        with self._lock:
            self._pos = int(sec * self._sr)

    @property
    def position_sec(self) -> float:
        """Current playback position in seconds."""
        return self._pos / self._sr

    @property
    def is_playing(self) -> bool:
        """Check if currently playing."""
        return self._playing

    @property
    def current_device(self) -> str:
        """Get current device name."""
        return self._device_name or "Default Device"

    def export(
        self,
        path: str,
        audio: np.ndarray,
        sr: int = 44100,
        format_name: str = "WAV",
        subtype: str = "FLOAT",
        apply_fade: bool = False,
        normalize: bool = False,
    ):
        """
        Export audio with format options (as per OpenTune SPEC v1.1).

        SPEC Requirement:
          - Format: WAV
          - Sample Rate: 44.1kHz
          - Bit Depth: 32-bit Float

        Args:
            path: Output file path
            audio: Audio data array (should be float32 from vocoder)
            sr: Sample rate (default 44100 per SPEC)
            format_name: Format (default WAV per SPEC)
            subtype: Bit depth/codec (default FLOAT = 32-bit per SPEC)
            apply_fade: Apply short fade-in/out (default False)
            normalize: Normalize before export (default False)
        """
        from utils.audio_utils import save_audio as _save, apply_fade as _fade

        out_audio = audio.copy()

        if apply_fade:
            out_audio = _fade(out_audio, fade_in_ms=5.0, fade_out_ms=5.0, sr=sr)

        if normalize:
            peak = np.max(np.abs(out_audio))
            if peak > 1e-8:
                out_audio = out_audio / peak

        _save(path, out_audio, sr, format_name=format_name, subtype=subtype)

    def _callback(self, outdata: np.ndarray, frames: int, time, status):
        """Audio callback - runs in C audio thread, NO Qt calls allowed!"""
        with self._lock:
            if not self._playing:
                outdata[:] = 0
                raise sd.CallbackStop()

            chunk = self._cache.get_audio_at(self._pos, frames)
            n = len(chunk)

            # Apply volume
            if self._volume != 1.0:
                chunk = chunk * self._volume

            outdata[:n, 0] = chunk
            if n < frames:
                outdata[n:, 0] = 0
            self._pos += frames

            if self._pos >= self._cache.total_samples:
                self._playing = False
                self._stopped_flag = True
                raise sd.CallbackStop()

    def _on_finished(self):
        """Called when playback reaches end - runs in audio thread."""
        self._stopped_flag = True


class AudioSettings:
    """Persistent audio settings storage."""

    DEFAULTS = {
        'sample_rate': 44100,
        'buffer_quality': 'medium',
        'volume': 1.0,
        'export_format': 'WAV',
        'export_subtype': 'FLOAT',           # 默认使用 FLOAT 保持最佳音质（与原版一致）
        'apply_fade_on_export': False,       # 默认关闭（vocoder 输出已处理好）
        'normalize_on_export': False,        # 默认关闭（vocoder 已做 RMS 匹配）
        'output_device_id': None,
    }

    def __init__(self):
        from utils.config import get, set as _set
        self._get = get
        self._set = _set

    def get_sample_rate(self) -> int:
        return int(self._get('audio_sample_rate', self.DEFAULTS['sample_rate']))

    def set_sample_rate(self, sr: int):
        from utils.audio_utils import SUPPORTED_SAMPLE_RATES
        if sr in SUPPORTED_SAMPLE_RATES:
            self._set('audio_sample_rate', sr)

    def get_buffer_quality(self) -> str:
        return str(self._get('audio_buffer_quality', self.DEFAULTS['buffer_quality']))

    def set_buffer_quality(self, quality: str):
        if quality in ['low', 'medium', 'high']:
            self._set('audio_buffer_quality', quality)

    def get_volume(self) -> float:
        return float(self._get('audio_volume', self.DEFAULTS['volume']))

    def set_volume(self, vol: float):
        self._set('audio_volume', max(0.0, min(2.0, vol)))

    def get_export_format(self) -> str:
        return str(self._get('export_format', self.DEFAULTS['export_format']))

    def set_export_format(self, fmt: str):
        from utils.audio_utils import EXPORT_FORMATS
        if fmt in EXPORT_FORMATS:
            self._set('export_format', fmt)

    def get_export_subtype(self) -> str:
        return str(self._get('export_subtype', self.DEFAULTS['export_subtype']))

    def set_export_subtype(self, subtype: str):
        self._set('export_subtype', subtype)

    def get_apply_fade(self) -> bool:
        return bool(self._get('apply_fade_on_export', self.DEFAULTS['apply_fade_on_export']))

    def set_apply_fade(self, value: bool):
        self._set('apply_fade_on_export', value)

    def get_normalize(self) -> bool:
        return bool(self._get('normalize_on_export', self.DEFAULTS['normalize_on_export']))

    def set_normalize(self, value: bool):
        self._set('normalize_on_export', value)

    def get_device_id(self) -> Optional[int]:
        val = self._get('output_device_id')
        return int(val) if val is not None else None

    def set_device_id(self, device_id: Optional[int]):
        self._set('output_device_id', device_id)
