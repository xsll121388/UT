"""Diagnostic script for vocoder output quality check.

Run this to inspect the full data pipeline and identify issues.
Usage: python diagnose_vocoder.py <audio_file.wav>
"""
import sys
import os
import numpy as np
import librosa
import warnings

sys.path.insert(0, os.path.dirname(__file__))
from core.vocoder import (
    SR, HOP, N_FFT, WIN_SIZE, N_MELS, FMIN, FMAX,
    F0_SAFE_MIN, F0_SAFE_MAX,
    _validate_audio, _clamp_f0, audio_to_mel, Vocoder
)
from core.pitch_tracker import PitchTracker


def diagnose_audio_file(audio_path: str):
    """Full pipeline diagnosis."""
    print("=" * 70)
    print("OPENTUNE VOCODER DIAGNOSTIC REPORT")
    print("=" * 70)

    # Step 1: Load audio
    print(f"\n[1] Loading audio: {audio_path}")
    audio, sr = librosa.load(audio_path, sr=SR, dtype=np.float32)
    print(f"    Original duration: {len(audio)/SR:.2f}s ({len(audio)} samples @ {sr}Hz)")
    print(f"    Peak amplitude: {np.max(np.abs(audio)):.4f}")
    print(f"    RMS energy: {np.sqrt(np.mean(audio**2)):.6f}")

    # Validate
    audio_clean = _validate_audio(audio)
    if np.array_equal(audio, audio_clean):
        print("    Validation: PASS (no changes)")
    else:
        print(f"    Validation: MODIFIED (peak was > 1.0)")

    # Step 2: Extract Mel spectrogram
    print(f"\n[2] Mel Spectrogram Extraction")
    print(f"    Parameters: SR={SR}, HOP={HOP}, N_FFT={N_FFT}, WIN={WIN_SIZE}")
    print(f"    N_MELS={N_MELS}, FMIN={FMIN}, FMAX={FMAX}")
    
    # Check N_FFT vs WIN_SIZE mismatch
    if N_FFT != WIN_SIZE:
        print(f"    ⚠️  WARNING: N_FFT({N_FFT}) != WIN_SIZE({WIN_SIZE})")
        print(f"       This causes zero-padding in FFT. Frequency bins may shift.")
    
    mel_raw = librosa.feature.melspectrogram(
        y=audio_clean, sr=SR, n_fft=N_FFT, hop_length=HOP,
        win_length=WIN_SIZE, n_mels=N_MELS, fmin=FMIN, fmax=FMAX, power=2.0
    )
    log_mel_raw = np.log(np.clip(mel_raw, 1e-5, None))
    log_mel_clipped = np.clip(log_mel_raw, -6.0, 1.8)
    
    n_frames = mel_raw.shape[1]
    mel_frame_rate = SR / HOP
    print(f"    Mel frames: {n_frames} ({mel_frame_rate:.2f} fps)")
    print(f"    Expected output samples: {n_frames * HOP}")
    print(f"    Raw log_mel range: [{log_mel_raw.min():.2f}, {log_mel_raw.max():.2f}]")
    print(f"    Clipped log_mel range: [{log_mel_clipped.min():.2f}, {log_mel_clipped.max():.2f}]")
    
    clipped_high = (log_mel_raw > 1.8).sum()
    clipped_low = (log_mel_raw < -6).sum()
    total_vals = log_mel_raw.size
    print(f"    Clip stats: {clipped_high} values above 1.8 ({100*clipped_high/total_vals:.2f}%), "
          f"{clipped_low} values below -6 ({100*clipped_low/total_vals:.2f}%)")

    # Step 3: Extract F0
    print(f"\n[3] F0 (Pitch) Extraction")
    tracker = PitchTracker()
    audio_16k = librosa.resample(audio_clean, orig_sr=SR, target_sr=16000)
    f0_raw = tracker.extract_from_44k(audio_clean)
    
    f0_voiced = f0_raw[f0_raw > 0]
    print(f"    F0 frames: {len(f0_raw)} (100fps from RMVPE)")
    print(f"    Voiced frames: {len(f0_voiced)} ({100*len(f0_voiced)/len(f0_raw):.1f}%)")
    if len(f0_voiced) > 0:
        print(f"    F0 range: [{f0_voiced.min():.1f}, {f0_voiced.max():.1f}] Hz")
        print(f"    F0 mean: {f0_voiced.mean():.1f} Hz, std: {f0_voiced.std():.1f} Hz")
    
    f0_clamped = _clamp_f0(f0_raw.copy())
    n_clamped_low = ((f0_raw > 0) & (f0_raw < F0_SAFE_MIN) & (f0_clamped == 0)).sum()
    n_clamped_high = (f0_clamped == F0_SAFE_MAX).sum()
    print(f"    After clamp: {n_clamped_low} → unvoiced (<{F0_SAFE_MIN}Hz), "
          f"{n_clamped_high} → clamped (>{F0_SAFE_MAX}Hz)")

    # Step 4: F0-Mel alignment check
    print(f"\n[4] F0 ↔ Mel Frame Alignment")
    print(f"    F0 input length: {len(f0_clamped)} (at ~100fps)")
    print(f"    Mel frame count: {n_frames} (at {mel_frame_rate:.2f}fps)")
    print(f"    Length ratio (F0/Mel): {len(f0_clamped)/n_frames:.3f}")
    
    if len(f0_clamped) != n_frames:
        print(f"    ⚠️  Length mismatch! Interpolation needed.")
        x_old = np.linspace(0, 1, len(f0_clamped))
        x_new = np.linspace(0, 1, n_frames)
        f0_resampled = np.interp(x_new, x_old, f0_clamped).astype(np.float32)
        
        f0_res_voiced = f0_resampled[f0_resampled > 0]
        if len(f0_res_voiced) > 0:
            # Check F0 discontinuity after interpolation
            f0_diff = np.abs(np.diff(f0_res_voiced))
            big_jumps = (f0_diff > 100).sum()  # jumps > 100Hz
            print(f"    After interpolation: {big_jumps} large F0 jumps (>100Hz) detected")
            print(f"    Max F0 jump: {f0_diff.max():.1f} Hz")
            print(f"    ⚠️  No smoothing applied — may cause robotic artifacts!")
    else:
        print(f"    ✅ Lengths match exactly")

    # Step 5: Model input summary
    print(f"\n[5] Model Input Summary")
    final_mel = log_mel_clipped.T[np.newaxis].astype(np.float32)
    final_f0 = f0_resampled if len(f0_clamped) != n_frames else f0_clamped
    final_f0 = _clamp_f0(final_f0.astype(np.float32))
    final_f0 = final_f0[np.newaxis]
    
    print(f"    mel shape: {final_mel.shape}  (expected: (1, {n_frames}, 128))")
    print(f"    f0 shape:  {final_f0.shape}  (expected: (1, {n_frames}))")
    print(f"    mel dtype: {final_mel.dtype}, range: [{final_mel.min():.3f}, {final_mel.max():.3f}]")
    print(f"    f0 dtype:  {final_f0.dtype}, voiced range: [{final_f0[final_f0>0].min():.1f}, {final_f0[final_f0>0].max():.1f}]")

    # Step 6: Check for potential issues
    print(f"\n[6] Issue Detection")
    issues = []
    
    if N_FFT != WIN_SIZE:
        issues.append(f"N_FFT({N_FFT}) ≠ WIN_SIZE({WIN_SIZE}) — zero-padding affects freq resolution")
    
    if clipped_high / total_vals > 0.01:
        issues.append(f"{100*clipped_high/total_vals:.1f}% of mel values clipped at upper bound (1.8) — loses strong harmonics")
    
    if big_jumps if 'big_jumps' in dir() else 0 > n_frames * 0.05:
        issues.append(f"Many F0 jumps ({big_jumps}) with no smoothing — robotic sound likely")
    
    if len(f0_voiced) > 0 and f0_voiced.std() / f0_voiced.mean() > 0.5:
        issues.append(f"High F0 variability (CV={100*f0_voiced.std()/f0_voiced.mean():.1f}%) — unstable pitch")
    
    if not issues:
        print("    ✅ No major issues detected")
    else:
        for i, issue in enumerate(issues, 1):
            print(f"    ⚠️  Issue #{i}: {issue}")

    # Step 7: Chunk processing simulation
    print(f"\n[7] Chunk Processing Simulation")
    chunk_sec = 5.0
    chunk_len = int(SR * chunk_sec)
    overlap = HOP * 2
    n_chunks = max(1, (len(audio_clean) + chunk_len - 1) // chunk_len)
    print(f"    Chunk size: {chunk_sec}s ({chunk_len} samples)")
    print(f"    Overlap context: {overlap} samples ({1000*overlap/SR:.1f}ms)")
    print(f"    Estimated chunks: {n_chunks}")
    
    for i in range(min(3, n_chunks)):
        cs = i * chunk_len
        ce = min((i + 1) * chunk_len, len(audio_clean))
        ctx_s = max(0, cs - overlap)
        ctx_e = min(len(audio_clean), ce + overlap)
        ctx_audio = audio_clean[ctx_s:ctx_e]
        
        fps = 100
        f0_s = int(ctx_s / SR * fps)
        f0_e = int(ctx_e / SR * fps) + 1
        f0_c = f0_raw[f0_s:min(f0_e, len(f0_raw))]
        
        mel_c = audio_to_mel(ctx_audio)
        expected_out = mel_c.shape[1] * HOP
        
        print(f"    Chunk {i}: audio={len(ctx_audio)}samples, f0={len(f0_c)}frames, "
              f"mel_frames={mel_c.shape[1]}, expected_out≈{expected_out}")

    print("\n" + "=" * 70)
    print("DIAGNOSIS COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python diagnose_vocoder.py <audio_file.wav>")
        sys.exit(1)
    diagnose_audio_file(sys.argv[1])
