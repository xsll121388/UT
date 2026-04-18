"""Audio smoothing utilities to prevent clicks and pops."""
from __future__ import annotations
import numpy as np


def find_zero_crossings(audio: np.ndarray, hop: int = 256) -> list[int]:
    """
    Find zero-crossing points in audio signal.
    
    Args:
        audio: Audio signal
        hop: Search hop size
        
    Returns:
        List of sample indices where zero crossings occur
    """
    crossings = []
    for i in range(0, len(audio) - hop, hop):
        segment = audio[i:i + hop]
        if segment[0] * segment[-1] < 0:  # Sign change
            # Find exact crossing point
            for j in range(len(segment) - 1):
                if segment[j] * segment[j + 1] < 0:
                    crossings.append(i + j)
                    break
    return crossings


def apply_smooth_fade(audio: np.ndarray, fade_in_samples: int = 256, fade_out_samples: int = 256) -> np.ndarray:
    """
    Apply smooth fade-in and fade-out to audio segment.
    
    Args:
        audio: Audio segment
        fade_in_samples: Fade-in duration in samples
        fade_out_samples: Fade-out duration in samples
        
    Returns:
        Faded audio segment
    """
    if len(audio) < fade_in_samples + fade_out_samples:
        # Segment too short, apply minimal fade
        fade_in_samples = min(fade_in_samples, len(audio) // 4)
        fade_out_samples = min(fade_out_samples, len(audio) // 4)
    
    result = audio.copy()
    
    # Apply fade-in using cosine curve
    if fade_in_samples > 0:
        fade_in = 0.5 * (1.0 - np.cos(np.linspace(0, np.pi, fade_in_samples, dtype=np.float32)))
        result[:fade_in_samples] *= fade_in
    
    # Apply fade-out using cosine curve
    if fade_out_samples > 0:
        fade_out = 0.5 * (1.0 + np.cos(np.linspace(0, np.pi, fade_out_samples, dtype=np.float32)))
        result[-fade_out_samples:] *= fade_out
    
    return result


def align_to_zero_crossing(audio: np.ndarray, target_pos: int, search_range: int = 256) -> int:
    """
    Adjust target position to nearest zero crossing.
    
    Args:
        audio: Full audio signal
        target_pos: Desired position
        search_range: How far to search for zero crossing
        
    Returns:
        Adjusted position (at zero crossing)
    """
    start = max(0, target_pos - search_range)
    end = min(len(audio), target_pos + search_range)
    
    if start >= end:
        return target_pos
    
    segment = audio[start:end]
    
    # Find zero crossings
    crossings = []
    for i in range(len(segment) - 1):
        if segment[i] * segment[i + 1] < 0:
            crossings.append(start + i)
    
    if not crossings:
        return target_pos
    
    # Find closest crossing to target
    closest = min(crossings, key=lambda x: abs(x - target_pos))
    return closest


def remove_dc_offset(audio: np.ndarray) -> np.ndarray:
    """
    Remove DC offset from audio signal.
    
    Args:
        audio: Audio signal
        
    Returns:
        DC-corrected audio signal
    """
    return audio - np.mean(audio)


def apply_crossfade(
    audio1: np.ndarray,
    audio2: np.ndarray,
    crossfade_samples: int = 882
) -> np.ndarray:
    """
    Crossfade between two audio segments.
    
    Args:
        audio1: First audio segment
        audio2: Second audio segment
        crossfade_samples: Crossfade duration in samples
        
    Returns:
        Crossfaded audio
    """
    if len(audio1) != len(audio2):
        raise ValueError("Audio segments must have same length")
    
    if len(audio1) < crossfade_samples:
        crossfade_samples = len(audio1) // 2
    
    result = np.zeros(len(audio1), dtype=np.float32)
    
    # Create crossfade curves
    fade_out = 0.5 * (1.0 + np.cos(np.linspace(0, np.pi, crossfade_samples, dtype=np.float32)))
    fade_in = 0.5 * (1.0 - np.cos(np.linspace(0, np.pi, crossfade_samples, dtype=np.float32)))
    
    # First segment (fade out)
    result[:crossfade_samples] = audio1[:crossfade_samples] * fade_out
    result[crossfade_samples:] = audio1[crossfade_samples:]
    
    # Second segment (fade in)
    result[:crossfade_samples] += audio2[:crossfade_samples] * fade_in
    result[:crossfade_samples] = np.clip(result[:crossfade_samples], -1.0, 1.0)
    
    return result
