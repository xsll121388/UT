"""
Parameter consistency validator for OpenTune Pro.
Ensures all modules use matching parameters based on actual model I/O.
"""

# ═════════════════════════════════════════════════════════════════
# SINGLE SOURCE OF TRUTH - Model Parameters (from ONNX inspection)
# ═════════════════════════════════════════════════════════════════

class RMVPEConfig:
    """RMVPE model configuration (verified from rmvpe.onnx)."""

    SAMPLE_RATE = 16000       # Input sample rate (fixed!)
    HOP_LENGTH = 160         # Hop size in samples @16kHz (fixed!)
    OUTPUT_FPS = 100          # Output frame rate (fixed!)

    F0_MIN = 32.70            # C1 - model's lower F0 bound
    F0_MAX = 1975.5           # B6 - model's upper F0 bound

    INPUT_NAMES = ["waveform", "threshold"]
    OUTPUT_NAMES = ["f0", "uv"]

    INPUT_SHAPES = {
        "waveform":  (1, None),      # (1, T) dynamic length
        "threshold": (),             # scalar
    }

    OUTPUT_SHAPES = {
        "f0": (1, None),              # (1, N) 2D array!
        "uv": (1, None),              # (1, N) 2D array!
    }


class HiFiGANConfig:
    """NSF-HiFiGAN vocoder configuration (verified from hifigan.onnx)."""

    SAMPLE_RATE = 44100        # Output sample rate (fixed!)
    HOP_SIZE = 256            # Hop size in samples (fixed!)
    N_FFT = 2048              # FFT size (MUST match training!)
    WIN_SIZE = 2048           # Window size (should == N_FFT)
    N_MELS = 128              # Mel bands (model expects exactly 128!)

    FMIN = 0                # Mel filterbank minimum frequency
    FMAX = 8000              # Mel filterbank maximum frequency (optimized)

    LOG_MEL_MIN = -12.0       # Log-mel lower clip
    LOG_MEL_MAX = -1.0        # Log-mel upper clip (empirically tuned)

    WINDOW_TYPE = "hann"      # Window function type (for phase coherence)
    MEL_SCALE = "htk"         # HTK mel scale (librosa default)

    NSF_F0_MIN = 50.0         # NSF module safe F0 minimum
    NSF_F0_MAX = 1100.0       # NSF module safe F0 maximum

    INPUT_NAMES = ["mel", "f0"]
    OUTPUT_NAMES = ["waveform"]

    INPUT_SHAPES = {
        "mel": (1, None, N_MELS),   # (1, n_frames, 128) 3D!
        "f0":  (1, None),           # (1, n_frames) 2D!
    }

    OUTPUT_SHAPES = {
        "waveform": (1, None),       # (1, n_samples) 2D!
    }


def validate_rmvpe_params(module_params: dict) -> list[str]:
    """Validate that RMVPE-related parameters match the model config.

    Args:
        module_params: Dictionary of parameter names to values from a module

    Returns:
        List of error messages (empty if all valid)
    """
    errors = []

    # Check critical fixed parameters
    checks = [
        ("sample_rate", module_params.get("sample_rate"), RMVPEConfig.SAMPLE_RATE),
        ("hop_length", module_params.get("hop_length"), RMVPEConfig.HOP_LENGTH),
        ("output_fps", module_params.get("output_fps"), RMVPEConfig.OUTPUT_FPS),
    ]

    for name, value, expected in checks:
        if value is not None and value != expected:
            errors.append(
                f"RMVPE {name} mismatch: got {value}, expected {expected}"
            )

    return errors


def validate_hifigan_params(module_params: dict) -> list[str]:
    """Validate that HiFiGAN-related parameters match the model config.

    Args:
        module_params: Dictionary of parameter names to values from a module

    Returns:
        List of error messages (empty if all valid)
    """
    errors = []

    # Check critical parameters
    checks = [
        ("SR / sample_rate", module_params.get("SR") or module_params.get("sample_rate"),
         HiFiGANConfig.SAMPLE_RATE),
        ("HOP / hop_size", module_params.get("HOP") or module_params.get("hop_size"),
         HiFiGANConfig.HOP_SIZE),
        ("N_FFT / n_fft", module_params.get("N_FFT") or module_params.get("n_fft"),
         HiFiGANConfig.N_FFT),
        ("WIN_SIZE / win_size", module_params.get("WIN_SIZE") or module_params.get("win_size"),
         HiFiGANConfig.WIN_SIZE),
        ("N_MELS / n_mels", module_params.get("N_MELS") or module_params.get("n_mels"),
         HiFiGANConfig.N_MELS),
        ("FMIN / fmin", module_params.get("FMIN") or module_params.get("fmin"),
         HiFiGANConfig.FMIN),
        ("FMAX / fmax", module_params.get("FMAX") or module_params.get("fmax"),
         HiFiGANConfig.FMAX),
    ]

    for name, value, expected in checks:
        if value is not None and value != expected:
            errors.append(
                f"HiFiGAN {name} mismatch: got {value}, expected {expected}"
            )

    return errors


def get_parameter_summary() -> dict:
    """
    Get a complete summary of all validated parameters.
    Useful for debugging and documentation.
    """
    return {
        "rmvpe": {
            "sample_rate": RMVPEConfig.SAMPLE_RATE,
            "hop_length": RMVPEConfig.HOP_LENGTH,
            "output_fps": RMVPEConfig.OUTPUT_FPS,
            "f0_range": (RMVPEConfig.F0_MIN, RMVPEConfig.F0_MAX),
            "input_names": RMVPEConfig.INPUT_NAMES,
            "output_names": RMVPEConfig.OUTPUT_NAMES,
        },
        "hifigan": {
            "sample_rate": HiFiGANConfig.SAMPLE_RATE,
            "hop_size": HiFiGANConfig.HOP_SIZE,
            "n_fft": HiFiGANConfig.N_FFT,
            "win_size": HiFiGANConfig.WIN_SIZE,
            "n_mels": HiFiGANConfig.N_MELS,
            "fmin": HiFiGANConfig.FMIN,
            "fmax": HiFiGANConfig.FMAX,
            "log_mel_range": (HiFiGANConfig.LOG_MEL_MIN, HiFiGANConfig.LOG_MEL_MAX),
            "window_type": HiFiGANConfig.WINDOW_TYPE,
            "nsf_f0_range": (HiFiGANConfig.NSF_F0_MIN, HiFiGANConfig.NSF_F0_MAX),
            "input_names": HiFiGANConfig.INPUT_NAMES,
            "output_names": HiFiGANConfig.OUTPUT_NAMES,
        },
    }


if __name__ == "__main__":
    print("=" * 70)
    print("🔍 OpenTune Pro - Parameter Consistency Validator")
    print("=" * 70)

    import sys
    sys.path.insert(0, r"c:\Users\zhu\Downloads\OpenTune.V1.1.Win.x64_\OpenTune V1.1 Win x64\OpenTunePro")

    # Import modules to check their parameters
    try:
        from core.vocoder import SR as vocoder_sr, HOP as vocoder_hop, \
            N_FFT as vocoder_nfft, WIN_SIZE as vocoder_winsize, \
            N_MELS as vocoder_nmels, FMIN as vocoder_fmin, FMAX as vocoder_fmax, \
            LOG_MEL_MIN, LOG_MEL_MAX, F0_SAFE_MIN, F0_SAFE_MAX

        from core.pitch_tracker import SAMPLE_RATE as rmvpe_sr, \
            HOP_LENGTH as rmvpe_hop, OUTPUT_FPS as rmvpe_fps, \
            FMIN as rmvpe_fmin, FMAX as rmvpe_fmax

        print("\n✅ Modules imported successfully\n")
    except ImportError as e:
        print(f"\n❌ Import failed: {e}")
        sys.exit(1)

    # Validate RMVPE
    print("📦 Checking RMVPE parameters...")
    rmvpe_errors = validate_rmvpe_params({
        "sample_rate": rmvpe_sr,
        "hop_length": rmvpe_hop,
        "output_fps": rmvpe_fps,
    })

    if rmvpe_errors:
        print("  ❌ Errors found:")
        for err in rmvpe_errors:
            print(f"     • {err}")
    else:
        print("  ✅ All RMVPE parameters OK")

    # Validate HiFiGAN
    print("\n📦 Checking HiFiGAN parameters...")
    hifigan_errors = validate_hifigan_params({
        "SR": vocoder_sr,
        "HOP": vocoder_hop,
        "N_FFT": vocoder_nfft,
        "WIN_SIZE": vocoder_winsize,
        "N_MELS": vocoder_nmels,
        "FMIN": vocoder_fmin,
        "FMAX": vocoder_fmax,
    })

    if hifigan_errors:
        print("  ❌ Errors found:")
        for err in hifigan_errors:
            print(f"     • {err}")
    else:
        print("  ✅ All HiFiGAN parameters OK")

    # Print summary
    print("\n" + "=" * 70)
    print("📋 Complete Parameter Summary")
    print("=" * 70)
    summary = get_parameter_summary()

    import json
    print(json.dumps(summary, indent=2, default=str))

    # Final verdict
    all_errors = rmvpe_errors + hifigan_errors
    print("\n" + "=" * 70)
    if all_errors:
        print(f"❌ VALIDATION FAILED: {len(all_errors)} error(s) found")
        sys.exit(1)
    else:
        print("✅ ALL PARAMETERS VALIDATED SUCCESSFULLY!")
        print("=" * 70)
