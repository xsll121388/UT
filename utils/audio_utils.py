"""Enhanced audio I/O utilities with format support and quality options."""
import numpy as np
import os
import soundfile as sf
import librosa
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class StretchPoint:
    """A stretch control point inside a MidiNote.
    
    Acts as a movable boundary that divides the note into segments.
    Dragging the point left/right shifts the boundary, causing the
    left segment to compress and the right to stretch (or vice versa).
    The total note duration is always preserved.
    
    orig_position: 0.0-1.0, the ORIGINAL position when the point was created
    position: 0.0-1.0, the CURRENT position (moved by dragging)
    """
    orig_position: float = 0.5
    position: float = 0.5

    @property
    def left_ratio(self) -> float:
        """Stretch ratio for the segment to the LEFT of this point."""
        if self.orig_position <= 0:
            return 1.0
        return self.position / self.orig_position

    @property
    def right_ratio(self) -> float:
        """Stretch ratio for the segment to the RIGHT of this point."""
        if self.orig_position >= 1.0:
            return 1.0
        remaining_orig = 1.0 - self.orig_position
        remaining_now = 1.0 - self.position
        if remaining_orig <= 0:
            return 1.0
        return remaining_now / remaining_orig


@dataclass
class MidiNote:
    """A single MIDI note with optional lyric and stretch control points."""
    start_sec: float
    end_sec: float
    pitch: int        # MIDI note number (0-127)
    lyric: str = ""
    stretch_points: list = field(default_factory=list)


def midi_note_to_dict(note: MidiNote) -> dict:
    """Serialize MidiNote to dictionary."""
    return {
        "start_sec": note.start_sec,
        "end_sec": note.end_sec,
        "pitch": note.pitch,
        "lyric": note.lyric,
        "stretch_points": [
            {"orig_position": sp.orig_position, "position": sp.position}
            for sp in note.stretch_points
        ],
    }


def midi_note_from_dict(d: dict) -> MidiNote:
    """Deserialize MidiNote from dictionary."""
    note = MidiNote(
        start_sec=d["start_sec"],
        end_sec=d["end_sec"],
        pitch=d["pitch"],
        lyric=d.get("lyric", ""),
    )
    for spd in d.get("stretch_points", []):
        note.stretch_points.append(StretchPoint(
            orig_position=spd.get("orig_position", 0.5),
            position=spd.get("position", 0.5),
        ))
    return note


def load_midi_notes(path: str) -> tuple:
    """Parse a MIDI file and return (notes, bpm).

    notes: list[MidiNote] sorted by start time
    bpm: float — first tempo found in the file (default 120.0)

    Builds a global tempo map first (required for Type 1 MIDI files where
    tempo events live in track 0 but notes live in track 1+).
    """
    import mido
    mid = mido.MidiFile(path)
    tpb = mid.ticks_per_beat

    # --- Pass 1: build global tempo map as list of (abs_tick, tempo_us) ---
    tempo_map: list[tuple[int, int]] = [(0, 500000)]  # default 120 BPM
    for track in mid.tracks:
        abs_tick = 0
        for msg in track:
            abs_tick += msg.time
            if msg.type == 'set_tempo':
                tempo_map.append((abs_tick, msg.tempo))
    tempo_map.sort(key=lambda x: x[0])
    # deduplicate: keep last tempo at each tick
    seen: dict[int, int] = {}
    for tick, us in tempo_map:
        seen[tick] = us
    tempo_map = sorted(seen.items())

    first_bpm = 60_000_000 / tempo_map[0][1] if tempo_map else 120.0
    for _, us in tempo_map:
        first_bpm = 60_000_000 / us
        break
    # first explicit non-default tempo
    for tick, us in tempo_map:
        if us != 500000:
            first_bpm = 60_000_000 / us
            break

    def ticks_to_sec(abs_tick: int) -> float:
        """Convert absolute tick position to seconds using the tempo map."""
        sec = 0.0
        prev_tick = 0
        prev_tempo = 500000
        for map_tick, map_tempo in tempo_map:
            if map_tick >= abs_tick:
                break
            sec += mido.tick2second(map_tick - prev_tick, tpb, prev_tempo)
            prev_tick = map_tick
            prev_tempo = map_tempo
        sec += mido.tick2second(abs_tick - prev_tick, tpb, prev_tempo)
        return sec

    # --- Pass 2: collect notes using global tick positions ---
    notes = []
    for track in mid.tracks:
        abs_tick = 0
        pending: dict[int, int] = {}  # pitch -> start_tick
        for msg in track:
            abs_tick += msg.time
            if msg.type == 'note_on' and msg.velocity > 0:
                pending[msg.note] = abs_tick
            elif msg.type == 'note_off' or (msg.type == 'note_on' and msg.velocity == 0):
                if msg.note in pending:
                    start_tick = pending.pop(msg.note)
                    notes.append(MidiNote(
                        ticks_to_sec(start_tick),
                        ticks_to_sec(abs_tick),
                        msg.note,
                    ))

    return sorted(notes, key=lambda n: n.start_sec), first_bpm


@dataclass
class AudioInfo:
    """Audio file metadata."""
    path: str
    sample_rate: int
    channels: int
    duration_sec: float
    frames: int
    format: str
    subtype: str


SUPPORTED_SAMPLE_RATES = [22050, 32000, 44100, 48000, 88200, 96000]

EXPORT_FORMATS = {
    "WAV": {"ext": ".wav", "description": "WAV (无损)", "subtypes": ["PCM_16", "PCM_24", "PCM_32", "FLOAT", "DOUBLE"]},
    "FLAC": {"ext": ".flac", "description": "FLAC (无损压缩)", "subtypes": ["PCM_16", "PCM_24"]},
    "AIFF": {"ext": ".aiff", "description": "AIFF (无损)", "subtypes": ["PCM_16", "PCM_24", "FLOAT"]},
    "OGG": {"ext": ".ogg", "description": "OGG Vorbis (有损压缩)", "subtypes": ["VORBIS"]},
}

FORMAT_QUALITY_PRESETS = {
    "PCM_16": {"bits": 16, "dynamic_range": "96 dB", "use_case": "标准CD质量（兼容性最好）"},
    "PCM_24": {"bits": 24, "dynamic_range": "144 dB", "use_case": "专业录音"},
    "PCM_32": {"bits": 32, "dynamic_range": "192 dB", "use_case": "高精度归档"},
    "FLOAT": {"bits": 32, "dynamic_range": "∞ dB", "use_case": "✅ 推荐：编辑/处理（最佳音质，与原版一致）"},
    "DOUBLE": {"bits": 64, "dynamic_range": "∞ dB", "use_case": "超高精度（文件较大）"},
    "VORBIS": {"quality": "0.0-1.0", "bitrate": "64-500 kbps", "use_case": "网络传输（有损压缩）"},
}


def get_audio_info(path: str) -> AudioInfo:
    """Get audio file metadata without loading the full file."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Audio file not found: {path}")

    info = sf.info(path)
    return AudioInfo(
        path=path,
        sample_rate=int(info.samplerate),
        channels=int(info.channels),
        duration_sec=float(info.duration),
        frames=int(info.frames),
        format=info.format,
        subtype=info.subtype,
    )


def load_audio(
    path: str,
    target_sr: int = 44100,
    mono: bool = True,
    normalize: bool = True,
) -> tuple[np.ndarray, int]:
    """
    Load audio file with enhanced options.

    Args:
        path: File path
        target_sr: Target sample rate (resample if needed)
        mono: Convert to mono if True
        normalize: Normalize to [-1, 1] range

    Returns:
        (audio_array, sample_rate)
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Audio file not found: {path}")

    audio, sr = sf.read(path, dtype="float32", always_2d=False)

    original_channels = audio.ndim

    # Channel handling
    if mono and audio.ndim == 2:
        audio = audio.mean(axis=1)

    # Resample if needed
    if sr != target_sr:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
        sr = target_sr

    # Normalize if requested
    if normalize and len(audio) > 0:
        peak = np.max(np.abs(audio))
        if peak > 1e-8:
            audio = audio / peak

    return audio, sr


def save_audio(
    path: str,
    audio: np.ndarray,
    sr: int = 44100,
    format_name: str = "WAV",
    subtype: str = "PCM_16",
    quality: Optional[float] = None,
    normalize: bool = True,
    clip: bool = True,
) -> None:
    """
    Save audio with format and quality options.

    Args:
        path: Output file path
        audio: Audio data
        sr: Sample rate
        format_name: Format name (WAV, FLAC, AIFF, OGG)
        subtype: Bit depth / codec subtype
        quality: Quality for lossy formats (0.0-1.0)
        normalize: Normalize before saving
        clip: Clip to [-1, 1] to prevent distortion
    """
    if len(audio) == 0:
        raise ValueError("Cannot save empty audio array")

    audio_out = audio.copy()

    # Ensure correct dtype based on subtype
    if subtype in ["FLOAT", "DOUBLE"]:
        audio_out = audio_out.astype(np.float32 if subtype == "FLOAT" else np.float64)
    else:
        audio_out = np.clip(audio_out, -1.0, 1.0) if clip else audio_out
        if subtype == "PCM_16":
            audio_out = (audio_out * 32767).astype(np.int16)
        elif subtype == "PCM_24":
            audio_out = (audio_out * 8388607).astype(np.int32)
        elif subtype == "PCM_32":
            audio_out = (audio_out * 2147483647).astype(np.int32)

    # Auto-detect format from extension if needed
    _, ext = os.path.splitext(path)
    ext = ext.lower()
    if ext in [".wav"]:
        format_name = "WAV"
    elif ext in [".flac"]:
        format_name = "FLAC"
    elif ext in [".aiff", ".aif"]:
        format_name = "AIFF"
    elif ext in [".ogg"]:
        format_name = "OGG"

    # Set quality for lossy formats
    kwargs = {}
    if quality is not None and format_name == "OGG":
        kwargs["quality"] = quality

    sf.write(path, audio_out, sr, format=format_name.lower(), subtype=subtype, **kwargs)


def time_stretch_to_duration(audio: np.ndarray, sr: int, target_sec: float) -> np.ndarray:
    """Uniformly time-stretch audio to exactly target_sec (pitch-preserving).

    Uses librosa's phase-vocoder. rate = current/target:
      >1 → compress (speed up), <1 → stretch (slow down).
    """
    current_sec = len(audio) / sr
    if abs(current_sec - target_sec) < 0.01:
        return audio
    rate = current_sec / target_sec
    stretched = librosa.effects.time_stretch(audio, rate=rate)
    target_samples = int(target_sec * sr)
    if len(stretched) > target_samples:
        return stretched[:target_samples].astype(np.float32)
    return np.pad(stretched, (0, target_samples - len(stretched))).astype(np.float32)


def apply_stretch_points(audio: np.ndarray, sr: int, note: MidiNote) -> np.ndarray:
    """Apply stretch control points to audio within a note.
    
    Each stretch point acts as a movable boundary. When the boundary is
    shifted, the left segment is compressed/stretched and the right segment
    is inversely stretched/compressed, preserving total duration.
    
    Args:
        audio: Audio samples for this note
        sr: Sample rate
        note: MidiNote with stretch_points
    
    Returns:
        Time-stretched audio with total duration preserved.
    """
    if not note.stretch_points or len(audio) == 0:
        return audio
    
    total_samples = len(audio)
    total_sec = total_samples / sr
    
    # Build current and original segment boundaries
    cur_boundaries = [0.0]
    orig_boundaries = [0.0]
    for sp in note.stretch_points:
        cur_boundaries.append(sp.position)
        orig_boundaries.append(sp.orig_position)
    cur_boundaries.append(1.0)
    orig_boundaries.append(1.0)
    
    # Process each segment
    result_parts = []
    for i in range(len(cur_boundaries) - 1):
        cur_start = cur_boundaries[i]
        cur_end = cur_boundaries[i + 1]
        orig_start = orig_boundaries[i]
        orig_end = orig_boundaries[i + 1]
        
        orig_dur_frac = orig_end - orig_start
        cur_dur_frac = cur_end - cur_start
        
        # Calculate target duration for this segment
        target_sec = cur_dur_frac * total_sec
        orig_sec = orig_dur_frac * total_sec
        
        # Get original audio for this segment
        orig_s0 = int(orig_start * total_samples)
        orig_s1 = int(orig_end * total_samples)
        orig_s0 = max(0, min(orig_s0, total_samples))
        orig_s1 = max(orig_s0, min(orig_s1, total_samples))
        
        seg_audio = audio[orig_s0:orig_s1]
        if len(seg_audio) == 0:
            continue
        
        # Calculate stretch ratio
        ratio = cur_dur_frac / orig_dur_frac if orig_dur_frac > 0 else 1.0
        
        if abs(ratio - 1.0) < 0.02:
            # No significant stretch
            result_parts.append(seg_audio)
        else:
            # Apply time stretch to match target duration
            stretched = time_stretch_to_duration(seg_audio, sr, target_sec)
            result_parts.append(stretched)
    
    if not result_parts:
        return audio
    
    # Concatenate all segments
    result = np.concatenate(result_parts)
    
    # Ensure total length matches original (preserve duration)
    if len(result) > total_samples:
        result = result[:total_samples]
    elif len(result) < total_samples:
        result = np.pad(result, (0, total_samples - len(result)))
    
    return result.astype(np.float32)


def split_into_chunks(audio: np.ndarray, sr: int, chunk_sec: float = 5.0) -> list[tuple[int, int]]:
    """Split audio into fixed-size chunks. No gaps, full coverage."""
    chunk_len = int(sr * chunk_sec)
    total = len(audio)
    intervals = []
    pos = 0
    while pos < total:
        end = min(pos + chunk_len, total)
        intervals.append((pos, end))
        pos = end
    return intervals


def hz_to_midi(hz: np.ndarray) -> np.ndarray:
    """Convert Hz to MIDI note numbers. 0 Hz stays 0."""
    out = np.zeros_like(hz)
    voiced = hz > 0
    out[voiced] = 12 * np.log2(hz[voiced] / 440.0) + 69
    return out


def midi_to_hz(midi: np.ndarray) -> np.ndarray:
    out = np.zeros_like(midi, dtype=np.float32)
    voiced = midi > 0
    out[voiced] = 440.0 * (2.0 ** ((midi[voiced] - 69) / 12.0))
    return out


def normalize(audio: np.ndarray, target_db: float = -3.0) -> np.ndarray:
    """Normalize audio to target dB level."""
    peak = np.max(np.abs(audio))
    if peak < 1e-8:
        return audio
    target_linear = 10 ** (target_db / 20.0)
    return audio * (target_linear / peak)


def apply_fade(
    audio: np.ndarray,
    fade_in_ms: float = 5.0,
    fade_out_ms: float = 5.0,
    sr: int = 44100,
) -> np.ndarray:
    """Apply linear fade-in/fade-out to prevent clicks."""
    result = audio.copy()

    if fade_in_ms > 0:
        fade_in_samples = int(sr * fade_in_ms / 1000.0)
        fade_in_samples = min(fade_in_samples, len(result) // 2)
        if fade_in_samples > 0:
            fade_curve = np.linspace(0.0, 1.0, fade_in_samples, dtype=np.float32)
            result[:fade_in_samples] *= fade_curve

    if fade_out_ms > 0:
        fade_out_samples = int(sr * fade_out_ms / 1000.0)
        fade_out_samples = min(fade_out_samples, len(result) // 2)
        if fade_out_samples > 0:
            fade_curve = np.linspace(1.0, 0.0, fade_out_samples, dtype=np.float32)
            result[-fade_out_samples:] *= fade_curve

    return result


def detect_silence(
    audio: np.ndarray,
    threshold_db: float = -40.0,
    min_silence_ms: float = 100.0,
    sr: int = 44100,
) -> list[tuple[float, float]]:
    """Detect silent regions in audio."""
    threshold_linear = 10 ** (threshold_db / 20.0)
    min_samples = int(sr * min_silence_ms / 1000.0)

    is_silent = np.abs(audio) < threshold_linear

    silence_regions = []
    start = None
    for i, silent in enumerate(is_silent):
        if silent and start is None:
            start = i
        elif not silent and start is not None:
            if i - start >= min_samples:
                silence_regions.append((start / sr, i / sr))
            start = None

    # Check trailing silence
    if start is not None and len(audio) - start >= min_samples:
        silence_regions.append((start / sr, len(audio) / sr))

    return silence_regions


def mix_audio(
    audio1: np.ndarray,
    audio2: np.ndarray,
    gain1: float = 1.0,
    gain2: float = 1.0,
    mode: str = "add",
) -> np.ndarray:
    """Mix two audio arrays together."""
    len_max = max(len(audio1), len(audio2))

    a1 = np.pad(audio1, (0, len_max - len(audio1))) * gain1
    a2 = np.pad(audio2, (0, len_max - len(audio2))) * gain2

    if mode == "add":
        result = a1 + a2
    elif mode == "multiply":
        result = a1 * a2
    else:
        raise ValueError(f"Unknown mixing mode: {mode}")

    # Prevent clipping
    peak = np.max(np.abs(result))
    if peak > 1.0:
        result = result / peak

    return result
