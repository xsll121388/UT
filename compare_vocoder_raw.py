"""对比vocoder.py的synthesize()和原始推理的差异"""
import numpy as np
import soundfile as sf
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent))

from core.pitch_tracker import PitchTracker
from core.vocoder import Vocoder
import onnxruntime as ort
import librosa
from scipy.interpolate import interp1d

SR = 44100
HOP = 512
N_FFT = 2048
N_MELS = 128
FMIN = 40
FMAX = 16000
MEL_MIN = -11.5
MEL_MAX = 2.5
F0_MIN = 20.0
F0_MAX = 2000.0

def compute_mel(audio):
    mel = librosa.feature.melspectrogram(
        y=audio, sr=SR, n_fft=N_FFT, hop_length=HOP,
        win_length=N_FFT, n_mels=N_MELS, fmin=FMIN, fmax=FMAX, power=1.0
    )
    log_mel = np.log(np.clip(mel, 1e-10, None)).astype(np.float32)
    return np.clip(log_mel, MEL_MIN, MEL_MAX)

def analyze_spectrum(audio, label):
    spec = np.abs(np.fft.rfft(audio))
    freqs = np.fft.rfftfreq(len(audio), 1.0 / SR)
    centroid = np.sum(freqs * spec) / (np.sum(spec) + 1e-8)
    rms = np.sqrt(np.mean(audio**2))
    peak = np.max(np.abs(audio))
    print(f"{label}:")
    print(f"  Centroid: {centroid:.0f} Hz")
    print(f"  RMS: {rms:.6f}")
    print(f"  Peak: {peak:.6f}")
    return centroid

def test_comparison(audio_file: str):
    print("Comparing vocoder.synthesize() vs raw inference")
    print("=" * 70)

    # Load audio
    audio, sr = sf.read(audio_file, frames=int(16000 * 5))
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)
    if sr != SR:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=SR)

    # Extract F0
    tracker = PitchTracker()
    f0 = tracker.extract_from_44k(audio)

    # Method 1: Using vocoder.synthesize()
    print("\n1. Using vocoder.synthesize()...")
    vocoder = Vocoder()
    result_vocoder = vocoder.synthesize(audio, f0, sr=SR, f0_fine_tune=0.0)
    c1 = analyze_spectrum(result_vocoder, "  Result")

    # Method 2: Raw inference
    print("\n2. Using raw ONNX inference...")
    mel = compute_mel(audio)
    n_frames = mel.shape[1]

    # Resample F0
    if len(f0) != n_frames:
        t_old = np.linspace(0, 1, len(f0))
        t_new = np.linspace(0, 1, n_frames)
        voiced = f0 > 0
        if voiced.any():
            f0_interp = interp1d(t_old, f0, kind='linear', fill_value='extrapolate')
            f0_resampled = f0_interp(t_new)
            voiced_interp = interp1d(t_old, voiced.astype(np.float32), kind='linear', fill_value='extrapolate')
            voiced_resampled = voiced_interp(t_new) > 0.5
            f0_resampled[~voiced_resampled] = 0.0
            f0 = f0_resampled

    voiced = f0 > 0
    f0[voiced & (f0 < F0_MIN)] = 0.0
    f0[voiced & (f0 > F0_MAX)] = F0_MAX

    session = ort.InferenceSession("models/hifigan.onnx", providers=["CPUExecutionProvider"])
    mel_in = mel.T[np.newaxis].astype(np.float32)
    f0_in = f0[np.newaxis].astype(np.float32)

    output = session.run(None, {"mel": mel_in, "f0": f0_in})
    result_raw = output[0].squeeze()

    if len(result_raw) > len(audio):
        result_raw = result_raw[:len(audio)]
    elif len(result_raw) < len(audio):
        result_raw = np.pad(result_raw, (0, len(audio) - len(result_raw)))

    c2 = analyze_spectrum(result_raw, "  Result")

    print("\n" + "=" * 70)
    print(f"Centroid difference: {c1:.0f}Hz (vocoder) vs {c2:.0f}Hz (raw)")
    print(f"Loss: {abs(c1 - c2):.0f}Hz ({abs(c1-c2)/c2*100:.1f}%)")

    # Save both
    output_dir = Path("test_output")
    output_dir.mkdir(exist_ok=True)
    sf.write(output_dir / "compare_vocoder.wav", result_vocoder, SR)
    sf.write(output_dir / "compare_raw.wav", result_raw, SR)
    print("\nSaved:")
    print("  test_output/compare_vocoder.wav")
    print("  test_output/compare_raw.wav")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python compare_vocoder_raw.py <audio_file>")
        sys.exit(1)
    test_comparison(sys.argv[1])
