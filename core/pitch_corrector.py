"""AutoTune pitch correction: snap F0 to nearest scale degree."""
from __future__ import annotations
import numpy as np

# Semitone offsets for each scale type relative to root
SCALES = {
    "major":       [0, 2, 4, 5, 7, 9, 11],
    "minor":       [0, 2, 3, 5, 7, 8, 10],
    "harmonic_minor": [0, 2, 3, 5, 7, 8, 11],
    "pentatonic":  [0, 2, 4, 7, 9],
    "chromatic":   list(range(12)),
}

NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]


def note_to_semitone(note: str) -> int:
    note = note.strip().upper().replace("♭", "b").replace("♯", "#")
    note = note.replace("Db", "C#").replace("Eb", "D#").replace("Gb", "F#") \
               .replace("Ab", "G#").replace("Bb", "A#")
    return NOTE_NAMES.index(note)


def build_scale_midi(root: str, scale: str) -> np.ndarray:
    """Return all MIDI note numbers in the given scale across full range."""
    root_semi = note_to_semitone(root)
    offsets = SCALES.get(scale, SCALES["major"])
    notes = []
    for octave in range(0, 10):
        for off in offsets:
            midi = octave * 12 + root_semi + off
            if 0 <= midi <= 127:
                notes.append(midi)
    return np.array(sorted(set(notes)), dtype=np.float32)


def snap_f0_to_scale(
    f0: np.ndarray,
    root: str = "C",
    scale: str = "major",
    retune_speed: float = 0.5,
) -> np.ndarray:
    """
    Snap F0 (Hz) toward nearest scale note.
    retune_speed: 0.0 = no correction, 1.0 = full snap.
    Returns corrected F0 in Hz.
    """
    scale_midi = build_scale_midi(root, scale)
    out = f0.copy()
    voiced = f0 > 0

    if not voiced.any():
        return out

    # Convert voiced frames to MIDI
    midi_in = 12 * np.log2(f0[voiced] / 440.0) + 69

    # Find nearest scale note for each frame
    diffs = np.abs(midi_in[:, np.newaxis] - scale_midi[np.newaxis, :])
    nearest_idx = np.argmin(diffs, axis=1)
    midi_target = scale_midi[nearest_idx]

    # Blend: lerp between original and target
    midi_corrected = midi_in + retune_speed * (midi_target - midi_in)

    # Convert back to Hz
    out[voiced] = 440.0 * (2.0 ** ((midi_corrected - 69) / 12.0))
    return out.astype(np.float32)


def smooth_f0(f0: np.ndarray, window: int = 5) -> np.ndarray:
    """Simple moving average smoothing for F0 transitions."""
    if window <= 1:
        return f0
    from scipy.ndimage import uniform_filter1d
    voiced = f0 > 0
    out = f0.copy()
    if voiced.any():
        out[voiced] = uniform_filter1d(f0[voiced].astype(np.float64), size=window).astype(np.float32)
    return out
