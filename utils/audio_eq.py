"""Audio EQ utilities for frequency shaping."""
from __future__ import annotations
import numpy as np
from scipy import signal


def apply_midrange_boost(
    audio: np.ndarray,
    sr: int = 44100,
    center_freq: float = 1000.0,
    gain_db: float = 3.0,
    q_factor: float = 1.0
) -> np.ndarray:
    """
    Apply parametric EQ boost to midrange frequencies.

    Args:
        audio: Input audio signal
        sr: Sample rate
        center_freq: Center frequency in Hz (500-2000 Hz for vocal presence)
        gain_db: Boost amount in dB (positive = boost, negative = cut)
        q_factor: Q factor (bandwidth control, higher = narrower)

    Returns:
        EQ-processed audio
    """
    if abs(gain_db) < 0.1:
        return audio

    # Design peaking EQ filter
    w0 = 2 * np.pi * center_freq / sr
    alpha = np.sin(w0) / (2 * q_factor)
    A = 10 ** (gain_db / 40)

    # Biquad coefficients for peaking EQ
    b0 = 1 + alpha * A
    b1 = -2 * np.cos(w0)
    b2 = 1 - alpha * A
    a0 = 1 + alpha / A
    a1 = -2 * np.cos(w0)
    a2 = 1 - alpha / A

    # Normalize
    b = np.array([b0, b1, b2]) / a0
    a = np.array([a0, a1, a2]) / a0

    # Apply filter
    result = signal.lfilter(b, a, audio).astype(np.float32)

    return result


def apply_multiband_eq(
    audio: np.ndarray,
    sr: int = 44100,
    low_gain_db: float = 0.0,      # 100-300 Hz
    mid_gain_db: float = 0.0,      # 800-1500 Hz
    high_gain_db: float = 0.0,     # 3000-8000 Hz
) -> np.ndarray:
    """
    Apply 3-band EQ for frequency balance correction.

    Args:
        audio: Input audio signal
        sr: Sample rate
        low_gain_db: Low frequency boost/cut
        mid_gain_db: Mid frequency boost/cut (vocal presence)
        high_gain_db: High frequency boost/cut (air/brightness)

    Returns:
        EQ-processed audio
    """
    result = audio.copy()

    # Low shelf (200 Hz)
    if abs(low_gain_db) > 0.1:
        result = apply_shelf_filter(result, sr, 200, low_gain_db, 'low')

    # Mid peaking (1000 Hz, Q=1.5 for vocal presence)
    if abs(mid_gain_db) > 0.1:
        result = apply_midrange_boost(result, sr, 1000, mid_gain_db, q_factor=1.5)

    # High shelf (5000 Hz)
    if abs(high_gain_db) > 0.1:
        result = apply_shelf_filter(result, sr, 5000, high_gain_db, 'high')

    return result


def apply_shelf_filter(
    audio: np.ndarray,
    sr: int,
    freq: float,
    gain_db: float,
    shelf_type: str = 'low'
) -> np.ndarray:
    """
    Apply low-shelf or high-shelf filter.

    Args:
        audio: Input audio
        sr: Sample rate
        freq: Cutoff frequency
        gain_db: Gain in dB
        shelf_type: 'low' or 'high'

    Returns:
        Filtered audio
    """
    w0 = 2 * np.pi * freq / sr
    A = 10 ** (gain_db / 40)
    alpha = np.sin(w0) / 2 * np.sqrt((A + 1/A) * (1/0.707 - 1) + 2)

    cos_w0 = np.cos(w0)

    if shelf_type == 'low':
        # Low shelf
        b0 = A * ((A + 1) - (A - 1) * cos_w0 + 2 * np.sqrt(A) * alpha)
        b1 = 2 * A * ((A - 1) - (A + 1) * cos_w0)
        b2 = A * ((A + 1) - (A - 1) * cos_w0 - 2 * np.sqrt(A) * alpha)
        a0 = (A + 1) + (A - 1) * cos_w0 + 2 * np.sqrt(A) * alpha
        a1 = -2 * ((A - 1) + (A + 1) * cos_w0)
        a2 = (A + 1) + (A - 1) * cos_w0 - 2 * np.sqrt(A) * alpha
    else:
        # High shelf
        b0 = A * ((A + 1) + (A - 1) * cos_w0 + 2 * np.sqrt(A) * alpha)
        b1 = -2 * A * ((A - 1) + (A + 1) * cos_w0)
        b2 = A * ((A + 1) + (A - 1) * cos_w0 - 2 * np.sqrt(A) * alpha)
        a0 = (A + 1) - (A - 1) * cos_w0 + 2 * np.sqrt(A) * alpha
        a1 = 2 * ((A - 1) - (A + 1) * cos_w0)
        a2 = (A + 1) - (A - 1) * cos_w0 - 2 * np.sqrt(A) * alpha

    b = np.array([b0, b1, b2]) / a0
    a = np.array([a0, a1, a2]) / a0

    result = signal.lfilter(b, a, audio).astype(np.float32)

    return result


def apply_presence_boost(audio: np.ndarray, sr: int = 44100) -> np.ndarray:
    """
    Apply vocal presence boost (optimized for speech/singing).

    Boosts 800-1500 Hz range for clarity and intelligibility.

    Args:
        audio: Input audio
        sr: Sample rate

    Returns:
        Presence-boosted audio
    """
    # Dual-band boost for natural presence
    result = apply_midrange_boost(audio, sr, center_freq=900, gain_db=2.5, q_factor=1.2)
    result = apply_midrange_boost(result, sr, center_freq=1400, gain_db=2.0, q_factor=1.5)

    return result
