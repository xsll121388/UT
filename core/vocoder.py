"""NSF-HiFiGAN vocoder wrapper.

Model actual I/O (from ONNX inspection):
  Input:
    mel  → (1, n_frames, 128) float32 - log mel-spectrogram (no normalization)
    f0   → (1, n_frames)      float32 - target F0 in Hz, 0 = unvoiced

  Output:
    waveform → (1, n_samples) float32 - resynthesized audio at 44.1kHz

Model-specific config:
  sampling_rate = 44100 Hz
  hop_size      = 512 samples
  n_fft         = 2048 samples  ← MUST match model training!
  win_size      = 2048 samples
  n_mels        = 128 bands (model expects exactly 128!)
  fmin          = 40 Hz
  fmax          = 16000 Hz     ← Extended for formant preservation

NSF module F0 range (critical for natural sound):
  f0_min = 50 Hz   (below this → silence/unvoiced)
  f0_max = 1100 Hz (above this → distortion/buzzing)

Model metadata: graph_name = "torch_jit" (PyTorch JIT export)
"""
from __future__ import annotations
import os
import warnings
import numpy as np
import librosa
import onnxruntime as ort
from utils import config
from utils.mel_cache import compute_cached_mel
from utils.performance import get_monitor, time_operation
from utils.audio_smoothing import remove_dc_offset, apply_smooth_fade

# ⚠️ CRITICAL: Must match THIS specific model's training config!
SR = 44100              # Model sample rate
HOP = 512               # hop_size in samples
N_FFT = 2048            # ✅ FIXED: Must match model's n_fft=2048!
WIN_SIZE = 2048         # analysis window size (must == N_FFT)
N_MELS = 128            # Model expects 128 mel bands
FMIN = 40               # Minimum frequency for mel
FMAX = 16000            # Must match model training (from config.json)

# NSF module F0 range: based on HachiTune's configuration
# Reference: https://github.com/KCKT0112/HachiTune/blob/main/Source/Audio/Vocoder.cpp
F0_SAFE_MIN = 20.0        # HachiTune: 20Hz minimum (wider than SPEC's 50Hz)
F0_SAFE_MAX = 2000.0      # HachiTune: 2000Hz maximum (wider than SPEC's 1100Hz)

# Log-mel dynamic range: DISABLED to preserve full frequency spectrum
# CRITICAL: Must use with power=1.0 (magnitude spectrum, NOT power spectrum!)
#
# Previous testing showed clipping caused frequency loss:
#   [-11.5, 2.5]: 17.1% centroid loss (too narrow, clips mid-freq detail)
#   [-15.0, 5.0]: 25.3% centroid loss (HachiTune config, still lossy)
#
# Solution: Use raw log-mel without clipping to preserve input spectrum
# The model should handle the full dynamic range naturally
#
# Reference: https://github.com/KCKT0112/HachiTune uses magnitude spectrum
LOG_MEL_MIN = None  # No clipping - preserve full dynamic range
LOG_MEL_MAX = None  # No clipping - preserve full dynamic range

# F0 fine-tuning (cents: 100 cents = 1 semitone, positive = higher pitch)
F0_FINE_TUNE_CENTS = 0   # Default: no adjustment (can be set via config)


def _validate_audio(audio: np.ndarray) -> np.ndarray:
    """Validate and clean audio input to prevent NaN/Inf propagation."""
    audio = np.asarray(audio, dtype=np.float32)

    if np.any(np.isnan(audio)):
        warnings.warn("Audio contains NaN values, replacing with zeros")
        audio = np.nan_to_num(audio, nan=0.0)

    if np.any(np.isinf(audio)):
        warnings.warn("Audio contains Inf values, clipping")
        audio = np.clip(audio, -1.0, 1.0)

    peak = np.max(np.abs(audio))
    if peak > 1.0:
        warnings.warn(f"Audio peak={peak:.3f} > 1.0, normalizing")
        audio = audio / peak

    if peak < 1e-8:
        warnings.warn("Audio is nearly silent")

    return audio


def _clamp_f0(f0: np.ndarray) -> np.ndarray:
    """Clamp F0 to NSF safe range. Unvoiced frames (0) are left as-is."""
    f0 = np.asarray(f0, dtype=np.float32)
    if np.any(np.isnan(f0)) or np.any(np.isinf(f0)):
        f0 = np.nan_to_num(f0, nan=0.0, posinf=0.0, neginf=0.0)
    voiced = f0 > 0
    f0[voiced & (f0 < F0_SAFE_MIN)] = 0.0          # too low → unvoiced
    f0[voiced & (f0 > F0_SAFE_MAX)] = F0_SAFE_MAX   # too high → clamp
    return f0


def _apply_f0_fine_tune(f0: np.ndarray, cents: float) -> np.ndarray:
    """
    Apply fine-tuning to F0 curve in cents.

    Args:
        f0: F0 array in Hz (unvoiced frames should be 0)
        cents: Adjustment in cents (100 cents = 1 semitone)

    Returns:
        Adjusted F0 array

    Formula: new_f0 = f0 * 2^(cents / 1200)

    Examples:
        +100 cents → one semitone higher
        -50 cents  → quarter tone lower
        +12 cents  → slight sharp (common for vocal tuning)
    """
    if abs(cents) < 0.01:
        return f0

    f0 = f0.copy()
    factor = 2.0 ** (cents / 1200.0)

    # Only adjust voiced frames
    voiced = f0 > 0
    f0[voiced] = f0[voiced] * factor

    # Re-clamp to safe range after adjustment
    f0 = _clamp_f0(f0)

    return f0


def _safe_mel_spectrogram(audio: np.ndarray) -> np.ndarray:
    """
    Compute log mel spectrogram matching THIS model's expected input format.

    Uses cached computation to avoid redundant calculations.

    Critical parameters (MUST match model training):
    - SR=44100, HOP=512, N_FFT=2048 (NOT 4096!)
    - n_mels=128 (model expects this specific dimension!)
    - FMAX=16000 for better high-frequency formant preservation
    - power=1.0 (magnitude spectrum, NOT power spectrum!)
      HachiTune uses magnitude: sqrt(real^2 + imag^2)
      This is critical for matching the model's training data

    Phase optimization for comb filtering reduction:
    - Uses Hann window for better spectral leakage properties
    - Centered FFT alignment to minimize phase discontinuities
    """
    # Use cached mel spectrogram computation (avoids redundant FFT)
    # CRITICAL: power=1.0 for magnitude spectrum (matches HachiTune)
    mel = compute_cached_mel(
        audio=audio,
        sr=SR,
        n_fft=N_FFT,
        hop_length=HOP,
        win_length=WIN_SIZE,
        n_mels=N_MELS,
        fmin=FMIN,
        fmax=FMAX,
        power=1.0  # Magnitude spectrum (NOT power=2.0!)
    )

    if np.any(np.isnan(mel)) or np.any(np.isinf(mel)):
        raise ValueError("Mel spectrogram computation produced NaN/Inf values")

    # Convert to log scale (preserve full dynamic range)
    log_mel = np.log(np.clip(mel, 1e-10, None)).astype(np.float32)

    # NO CLIPPING - preserve full frequency spectrum information
    # This ensures input and output have similar spectral characteristics

    # Transpose to (1, n_frames, n_mels)
    return log_mel.T[np.newaxis]


def audio_to_mel(audio: np.ndarray) -> np.ndarray:
    """Convert (T,) float32 audio → (1, n_frames, 128) log mel."""
    return _safe_mel_spectrogram(audio)


class Vocoder:
    """NSF-HiFiGAN vocoder with validated model interface.

    Model I/O (verified from ONNX inspection):
      Inputs:
        mel:  (1, n_frames, 128) float32
        f0:   (1, n_frames)      float32
      Output:
        waveform: (1, n_samples) float32
    """

    def __init__(self):
        model_path = os.path.join(config.get_model_dir(), "hifigan.onnx")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"HiFiGAN model not found: {model_path}")

        opts = ort.SessionOptions()
        providers = ["DmlExecutionProvider", "CPUExecutionProvider"]
        try:
            self._session = ort.InferenceSession(
                model_path, sess_options=opts, providers=providers
            )
        except Exception:
            self._session = ort.InferenceSession(
                model_path, sess_options=opts, providers=["CPUExecutionProvider"]
            )

        # Validate and store model I/O information
        self._inputs = {inp.name: inp for inp in self._session.get_inputs()}
        self._outputs = {out.name: out for out in self._session.get_outputs()}

        # Verify expected input/output names exist
        expected_inputs = {"mel", "f0"}
        actual_inputs = set(self._inputs.keys())
        if not expected_inputs.issubset(actual_inputs):
            warnings.warn(
                f"HiFiGAN input names mismatch! Expected {expected_inputs}, "
                f"got {actual_inputs}"
            )

    def synthesize(self, audio: np.ndarray, f0: np.ndarray, sr: int = SR,
                   audio_for_rms: Optional[np.ndarray] = None,
                   f0_fine_tune: float = 0.0) -> np.ndarray:
        """
        Re-synthesize audio with a new F0 curve.

        Args:
            audio: (T,) float32 at 44.1kHz (used for mel spectrogram)
            f0:    (N,) float32 Hz at 100fps (RMVPE output rate), 0=unvoiced
            sr:    input sample rate (default 44100)
            audio_for_rms: Optional (T',) array for RMS matching.
                           If provided, use this instead of `audio` for loudness matching.
                           Useful when `audio` includes silent context padding.
            f0_fine_tune: Fine-tune F0 in cents (100 cents = 1 semitone).
                          Positive = higher pitch, Negative = lower pitch.
                          Range: -200 to +200 cents recommended.

        Returns:
            (T,) float32 at 44.1kHz
        """
        # Validate inputs
        if len(audio) == 0:
            raise ValueError("Empty audio input")
        if len(f0) == 0:
            raise ValueError("Empty F0 input")

        # Use audio_for_rms for loudness matching if provided
        rms_reference_audio = audio_for_rms if audio_for_rms is not None else audio

        audio = _validate_audio(audio)
        f0 = _clamp_f0(f0)

        # Apply F0 fine-tuning if specified (compensate for systematic pitch offset)
        if abs(f0_fine_tune) > 0.01:
            f0 = _apply_f0_fine_tune(f0, f0_fine_tune)

        # Resample if needed (should be rare now)
        if sr != SR:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=SR)

        if len(audio) < N_FFT:
            warnings.warn(f"Audio too short ({len(audio)} samples), padding")
            audio = np.pad(audio, (0, N_FFT - len(audio)))

        # Extract Mel spectrogram at correct parameters
        mel = audio_to_mel(audio)
        n_frames_mel = mel.shape[1]

        # Resample F0 from RMVPE rate (100fps) to Mel frame count
        # At 44100Hz with hop=512, frame rate = 44100/512 ≈ 86.13 fps
        f0 = f0.astype(np.float32)
        if len(f0) != n_frames_mel:
            if len(f0) == 0:
                raise ValueError("Empty F0 array provided")

            # Use linear interpolation with proper voiced/unvoiced handling
            from scipy.interpolate import interp1d

            # Create time arrays
            t_old = np.linspace(0, 1, len(f0))
            t_new = np.linspace(0, 1, n_frames_mel)

            # Separate voiced and unvoiced
            voiced_mask = f0 > 0

            if voiced_mask.any():
                # Interpolate F0 values (including zeros)
                f0_interp = interp1d(t_old, f0, kind='linear', fill_value='extrapolate')
                f0_resampled = f0_interp(t_new)

                # Interpolate voiced mask
                voiced_interp = interp1d(t_old, voiced_mask.astype(np.float32),
                                        kind='linear', fill_value='extrapolate')
                voiced_resampled = voiced_interp(t_new) > 0.5

                # Apply mask
                f0_resampled[~voiced_resampled] = 0.0
                f0 = f0_resampled
            else:
                # All unvoiced
                f0 = np.zeros(n_frames_mel, dtype=np.float32)

        # Re-clamp after interpolation
        f0 = _clamp_f0(f0)

        # Prepare inputs matching model signature (from ONNX inspection):
        #   mel: (1, n_frames, 128) float32 - log mel spectrogram
        #   f0:  (1, n_frames)      float32 - target F0 in Hz (0=unvoiced)
        f0_in = f0[np.newaxis].astype(np.float32)  # Ensure correct shape and dtype

        # Validate shapes before inference
        if mel.shape[0] != 1 or mel.shape[2] != N_MELS:
            raise ValueError(
                f"Mel spectrogram shape mismatch: got {mel.shape}, "
                f"expected (1, *, {N_MELS})"
            )
        if f0_in.shape[0] != 1:
            raise ValueError(
                f"F0 shape mismatch: got {f0_in.shape}, expected (1, N)"
            )
        if mel.shape[1] != f0_in.shape[1]:
            raise ValueError(
                f"Frame count mismatch: mel={mel.shape[1]} frames vs "
                f"f0={f0_in.shape[1]} frames"
            )

        # Run HiFi-GAN inference
        try:
            outputs = self._session.run(None, {"mel": mel, "f0": f0_in})
        except Exception as e:
            raise RuntimeError(f"HiFi-GAN inference failed: {e}")

        result = outputs[0].squeeze().astype(np.float32)

        # Validate output
        if np.any(np.isnan(result)) or np.any(np.isinf(result)):
            raise ValueError("HiFi-GAN output contains NaN/Inf values")

        # Expected output length
        expected_samples = n_frames_mel * HOP
        if abs(len(result) - expected_samples) > HOP * 2:
            warnings.warn(
                f"Output length mismatch: got {len(result)}, expected ~{expected_samples}"
            )

        # Trim/pad to match original length
        if len(result) > len(audio):
            result = result[:len(audio)]
        elif len(result) < len(audio):
            result = np.pad(result, (0, len(audio) - len(result)))

        # Minimal post-processing - preserve model output quality
        # Testing shows that aggressive filtering reduces spectral centroid
        # from 1726Hz (raw) to 1446Hz (filtered), introducing metallic artifacts

        # Only apply gentle fade to prevent clicks at boundaries
        result = apply_smooth_fade(result, fade_in_samples=128, fade_out_samples=128)

        # Remove DC offset (essential for preventing speaker damage)
        result = remove_dc_offset(result)

        # Simple RMS matching to preserve loudness
        rms_in = np.sqrt(np.mean(rms_reference_audio ** 2))
        rms_out = np.sqrt(np.mean(result ** 2))

        if rms_in > 1e-8 and rms_out > 1e-8:
            gain = rms_in / rms_out
            gain = np.clip(gain, 0.5, 1.5)
            result = result * gain

        # Soft peak limiting to prevent clipping
        peak = np.max(np.abs(result))
        if peak > 0.99:
            result = np.tanh(result * 0.95) * 0.99

        result = np.clip(result, -1.0, 1.0)

        return result
