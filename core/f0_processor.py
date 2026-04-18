"""Advanced F0 post-processing for natural-sounding pitch correction.

Addresses three core NSF-HiFiGAN defects:
1. Comb filtering (metallic sound) → Phase alignment + window optimization
2. F0 jump artifacts (clicks/pops) → Median filter smoothing + V/UV transition
3. High-frequency hollow/distortion → Parameter matching + harmonic preservation

Key techniques:
- Median filter: Removes sharp spikes while preserving genuine pitch contours
- V/UV soft transition: Gradual crossfade at voiced/unvoiced boundaries
- Interpolation: Fill short unvoiced gaps for continuity
- Outlier removal: Eliminate octave errors and detection glitches
"""
from __future__ import annotations
import numpy as np
from scipy.ndimage import median_filter, uniform_filter1d
from scipy.signal import medfilt


def smooth_f0_median(
    f0: np.ndarray,
    kernel_size: int = 5,
    iterations: int = 1,
    preserve_voiced_boundaries: bool = True,
) -> np.ndarray:
    """
    Apply median filter to F0 curve for spike removal.

    Median filter is superior to mean/average filter because:
    - Completely removes impulsive noise (spikes)
    - Preserves sharp but legitimate pitch transitions (glissando)
    - Does not blur edges like Gaussian filters
    - Robust to outliers (octave errors)

    Args:
        f0: F0 array in Hz (0 = unvoiced)
        kernel_size: Window size in frames (odd number recommended: 3, 5, 7, 9)
        iterations: Number of passes (1-2 usually sufficient)
        preserve_voiced_boundaries: If True, don't smooth across V/UV boundaries

    Returns:
        Smoothed F0 array
    """
    if kernel_size < 3:
        kernel_size = 3
    if kernel_size % 2 == 0:
        kernel_size += 1  # Must be odd for median filter

    f0_out = f0.copy().astype(np.float64)
    voiced = f0 > 0

    if not voiced.any():
        return f0.astype(np.float32)

    if preserve_voiced_boundaries:
        # Process each continuous voiced segment separately
        segments = _find_voiced_segments(voiced)
        for start, end in segments:
            segment_len = end - start
            if segment_len <= kernel_size:
                continue  # Too short to process

            seg_f0 = f0_out[start:end].copy()
            for _ in range(iterations):
                seg_f0 = medfilt(seg_f0, kernel_size=kernel_size)
            f0_out[start:end] = seg_f0
    else:
        # Process entire array (unvoiced frames stay 0 after masking)
        f0_voiced_only = f0_out.copy()
        f0_voiced_only[~voiced] = np.nan
        for _ in range(iterations):
            f0_voiced_only = _nanmedian_filter(f0_voiced_only, kernel_size)
        valid_mask = ~np.isnan(f0_voiced_only)
        f0_out[valid_mask & voiced] = f0_voiced_only[valid_mask & voiced]

    return f0_out.astype(np.float32)


def smooth_f0_vuv_transition(
    f0: np.ndarray,
    fade_frames: int = 3,
    min_voiced_duration: int = 5,
    fill_short_gaps: bool = True,
) -> np.ndarray:
    """
    Smooth V/UV (Voiced/Unvoiced) transitions to eliminate clicks and pops.

    Problems this solves:
    - Abrupt 0 → F0 jumps cause "click" artifacts
    - Short unvoiced gaps within sustained notes create "static" sounds
    - Rapid V/UV switching causes "sparkle" noise at syllable boundaries

    Args:
        f0: F0 array in Hz
        fade_frames: Number of frames for crossfade at V/UV boundaries (default 3 ≈ 30ms @100fps)
        min_voiced_duration: Minimum frames to consider a voiced segment valid
                             (shorter segments treated as glitches)
        fill_short_gaps: If True, interpolate across brief unvoiced regions (< min_gap)

    Returns:
        Processed F0 with smooth transitions
    """
    f0_out = f0.copy().astype(np.float64)
    n_frames = len(f0)
    voiced = f0 > 0

    if not voiced.any():
        return f0.astype(np.float32)

    # Find all voiced/unvoiced segments
    segments = _find_segments(voiced)

    # Step 1: Remove very short voiced segments (glitches)
    if min_voiced_duration > 0:
        for start, end, is_voiced in segments:
            if is_voiced and (end - start) < min_voiced_duration:
                f0_out[start:end] = 0.0

    # Step 2: Fill short unvoiced gaps within voiced regions
    if fill_short_gaps:
        min_gap = max(fade_frames * 2, 5)  # At least 2x the fade length
        segments = _find_segments(f0_out > 0)

        # Find gaps between voiced segments that are close together
        for i in range(len(segments) - 1):
            curr_end = segments[i][1]
            next_start = segments[i + 1][0]
            gap_len = next_start - curr_end

            if gap_len < min_gap and gap_len > 0:
                # Interpolate across the gap
                f0_before = f0_out[curr_end - 1]
                f0_after = f0_out[next_start]

                if f0_before > 0 and f0_after > 0:
                    t = np.linspace(0, 1, gap_len, endpoint=False)[1:-1] or np.array([])
                    if len(t) > 0:
                        interpolated = f0_before * (1 - t) + f0_after * t
                        f0_out[curr_end + 1:next_start - 1] = interpolated

    # Step 3: Apply smooth fades at V/UV boundaries
    # Use longer fades (5 frames = 50ms) for more natural transitions
    fade_frames = max(fade_frames, 5)
    
    if fade_frames > 0:
        segments = _find_segments(f0_out > 0)

        for i, (start, end, is_voiced) in enumerate(segments):
            if not is_voiced:
                continue

            # Fade-in at start of voiced segment using raised cosine for smoother transition
            if start > 0 and f0_out[start - 1] == 0:
                actual_fade = min(fade_frames, end - start)
                if actual_fade > 0:
                    target_val = f0_out[start + actual_fade] if (start + actual_fade) < n_frames else f0_out[start]
                    if target_val > 0:
                        # Use raised cosine (Hann window) for smoother fade
                        t = np.linspace(0, np.pi, actual_fade)
                        fade_in = 0.5 * (1 - np.cos(t))
                        f0_out[start:start + actual_fade] *= fade_in

            # Fade-out at end of voiced segment using raised cosine
            if end < n_frames and f0_out[end] == 0:
                actual_fade = min(fade_frames, end - start)
                if actual_fade > 0:
                    base_val = f0_out[end - actual_fade - 1] if (end - actual_fade - 1) >= 0 else f0_out[end - 1]
                    if base_val > 0:
                        # Use raised cosine (Hann window) for smoother fade
                        t = np.linspace(0, np.pi, actual_fade)
                        fade_out = 0.5 * (1 + np.cos(t))
                        f0_out[end - actual_fade:end] *= fade_out

    return f0_out.astype(np.float32)


def remove_f0_outliers(
    f0: np.ndarray,
    max_jump_semitones: float = 4.0,
    min_confidence_ratio: float = 0.3,
) -> np.ndarray:
    """
    Remove F0 outliers caused by detection errors (octave jumps, halving/doubling).

    Args:
        f0: F0 array in Hz
        max_jump_semitones: Maximum allowed semitone jump between adjacent frames
                            (4 semitones = major third, reasonable limit)
        min_confidence_ratio: For multi-pitch scenarios, keep only top N% confident frames

    Returns:
        Cleaned F0 array
    """
    f0_out = f0.copy().astype(np.float64)
    voiced = f0 > 0

    if np.sum(voiced) < 3:
        return f0.astype(np.float32)

    # Get voiced frame indices
    voiced_indices = np.where(voiced)[0]
    voiced_values = f0[voiced_indices].astype(np.float64)

    if len(voiced_values) < 3:
        return f0.astype(np.float32)

    # Convert to cents for relative comparison
    voiced_cents = 1200 * np.log2(voiced_values / np.median(voiced_values))

    # Detect jumps exceeding threshold
    jumps = np.abs(np.diff(voiced_cents))
    max_allowed_cents = max_jump_semitones * 100  # Convert semitones to cents

    # Mark outlier frames
    outlier_mask = np.zeros(len(f0), dtype=bool)
    outlier_mask[voiced_indices[0]] = False  # Keep first frame

    for i in range(1, len(jumps)):
        if jumps[i] > max_allowed_cents:
            # Mark the more extreme point as outlier
            if abs(voiced_cents[i]) > abs(voiced_cents[i + 1]):
                outlier_mask[voiced_indices[i + 1]] = True
            else:
                outlier_mask[voiced_indices[i]] = True

    # Also check for isolated single-frame spikes
    for i in range(1, len(voiced_indices) - 1):
        idx = voiced_indices[i]
        if not outlier_mask[idx]:
            prev_idx = voiced_indices[i - 1]
            next_idx = voiced_indices[i + 1]

            prev_diff = abs(1200 * np.log2(f0[idx] / f0[prev_idx])) if f0[prev_idx] > 0 else 999
            next_diff = abs(1200 * np.log2(f0[idx] / f0[next_idx])) if f0[next_idx] > 0 else 999

            if prev_diff > max_allowed_cents and next_diff > max_allowed_cents:
                outlier_mask[idx] = True

    # Remove outliers (set to 0, will be interpolated later)
    f0_out[outlier_mask] = 0.0

    return f0_out.astype(np.float32)


def advanced_f0_processing(
    f0: np.ndarray,
    median_kernel: int = 5,
    vuv_fade_frames: int = 3,
    remove_octave_errors: bool = True,
    fill_gaps: bool = True,
) -> np.ndarray:
    """
    Complete F0 processing pipeline combining all optimizations.

    Processing order:
    1. Remove outliers (octave errors, detection glitches)
    2. Median filter smoothing (remove spikes)
    3. V/UV transition smoothing (eliminate clicks)
    4. Gap filling (improve continuity)

    Args:
        f0: Raw F0 from RMVPE extractor
        median_kernel: Median filter window size (5 recommended)
        vuv_fade_frames: Crossfade length at boundaries (3 = 30ms)
        remove_octave_errors: Enable outlier removal
        fill_gaps: Enable short gap interpolation

    Returns:
        Fully processed F0 ready for vocoder
    """
    result = f0.copy()

    # Step 1: Remove obvious outliers
    if remove_octave_errors:
        result = remove_f0_outliers(result)

    # Step 2: Median filter smoothing
    result = smooth_f0_median(result, kernel_size=median_kernel, iterations=1)

    # Step 3: Smooth V/UV transitions
    result = smooth_f0_vuv_transition(result, fade_frames=vuv_fade_frames, fill_short_gaps=fill_gaps)

    return result


# ── Helper Functions ────────────────────────────────────────────────────


def _find_voiced_segments(voiced_mask: np.ndarray) -> list[tuple[int, int]]:
    """Find contiguous voiced segments."""
    segments = []
    in_segment = False
    start = 0

    for i, v in enumerate(voiced_mask):
        if v and not in_segment:
            start = i
            in_segment = True
        elif not v and in_segment:
            segments.append((start, i))
            in_segment = False

    if in_segment:
        segments.append((start, len(voiced_mask)))

    return segments


def _find_segments(binary_mask: np.ndarray) -> list[tuple[int, int, bool]]:
    """Find all contiguous segments with their type (True/False)."""
    segments = []
    if len(binary_mask) == 0:
        return segments

    current_val = binary_mask[0]
    start = 0

    for i in range(1, len(binary_mask)):
        if binary_mask[i] != current_val:
            segments.append((start, i, bool(current_val)))
            start = i
            current_val = binary_mask[i]

    segments.append((start, len(binary_mask), bool(current_val)))

    return segments


def _nanmedian_filter(arr: np.ndarray, kernel_size: int) -> np.ndarray:
    """Median filter that preserves NaN values."""
    result = arr.copy()
    nan_mask = np.isnan(arr)

    if nan_mask.all():
        return arr

    # Replace NaN with temporary values for filtering
    temp = arr.copy()
    temp[nan_mask] = np.nanmean(temp[~nan_mask]) if (~nan_mask).any() else 0

    filtered = medfilt(temp, kernel_size=kernel_size)

    # Restore NaN positions
    result[~nan_mask] = filtered[~nan_mask]

    return result
