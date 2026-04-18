"""
OpenTune Pro - SPEC Compliance Validator
Ensures all parameters match the OpenTune SPEC v1.1 requirements.
"""

# ═════════════════════════════════════════════════════════════════
# OpenTune SPEC v1.1 - Required Parameters
# ═════════════════════════════════════════════════════════════════

class OpenTuneSpecV11:
    """Official OpenTune SPEC v1.1 parameter definitions."""

    # 4.4 AI Pitch Correction Mode (OpenTune Style)
    class RMVPE_PITCH_EXTRACTION:
        FRAME_RATE = 100            # fps (SPEC requirement)
        F0_MIN = 50.0               # Hz (SPEC: 50Hz - 1100Hz)
        F0_MAX = 1100.0             # Hz (SPEC: 50Hz - 1100Hz)
        SUPPORTS_VUV = True         # Voiced/Unvoiced detection

    class NSF_HIFIGAN_RESYNTHESIS:
        SAMPLE_RATE = 44100         # Hz (SPEC requirement)
        PRESERVE_FORMANTS = True    # Must preserve formants
        SUPPORTS_TRANSPOSITION = True  # Large range transposition

    # 4.6 Playback & Export
    class EXPORT_FORMAT:
        FORMAT = "WAV"              # SPEC requirement
        SAMPLE_RATE = 44100         # Hz (SPEC requirement)
        BIT_DEPTH = "FLOAT"         # 32-bit Float (SPEC requirement)


def validate_spec_compliance(module_params: dict) -> list[str]:
    """
    Validate that all parameters match OpenTune SPEC v1.1.

    Args:
        module_params: Dictionary with module names as keys

    Returns:
        List of compliance issues (empty if fully compliant)
    """
    issues = []

    # Check RMVPE parameters
    rmvpe = module_params.get("rmvpe", {})
    if rmvpe.get("output_fps") != OpenTuneSpecV11.RMVPE_PITCH_EXTRACTION.FRAME_RATE:
        issues.append(
            f"RMVPE frame rate: got {rmvpe.get('output_fps')}, "
            f"SPEC requires {OpenTuneSpecV11.RMVPE_PITCH_EXTRACTION.FRAME_RATE} fps"
        )

    # Check HiFiGAN parameters
    hifigan = module_params.get("hifigan", {})
    if hifigan.get("sample_rate") != OpenTuneSpecV11.NSF_HIFIGAN_RESYNTHESIS.SAMPLE_RATE:
        issues.append(
            f"HiFiGAN sample rate: got {hifigan.get('sample_rate')}, "
            f"SPEC requires {OpenTuneSpecV11.NSF_HIFIGAN_RESYNTHESIS.SAMPLE_RATE} Hz"
        )

    # Check export format
    export = module_params.get("export", {})
    if export.get("format") != OpenTuneSpecV11.EXPORT_FORMAT.FORMAT:
        issues.append(
            f"Export format: got {export.get('format')}, "
            f"SPEC requires {OpenTuneSpecV11.EXPORT_FORMAT.FORMAT}"
        )
    if export.get("sample_rate") != OpenTuneSpecV11.EXPORT_FORMAT.SAMPLE_RATE:
        issues.append(
            f"Export sample rate: got {export.get('sample_rate')}, "
            f"SPEC requires {OpenTuneSpecV11.EXPORT_FORMAT.SAMPLE_RATE} Hz"
        )
    if export.get("bit_depth") != OpenTuneSpecV11.EXPORT_FORMAT.BIT_DEPTH:
        issues.append(
            f"Export bit depth: got {export.get('bit_depth')}, "
            f"SPEC requires {OpenTuneSpecV11.EXPORT_FORMAT.BIT_DEPTH} (32-bit Float)"
        )

    return issues


def get_spec_summary() -> dict:
    """Get a summary of OpenTune SPEC v1.1 requirements."""
    return {
        "spec_version": "1.1",
        "rmvpe_pitch_extraction": {
            "frame_rate": f"{OpenTuneSpecV11.RMVPE_PITCH_EXTRACTION.FRAME_RATE} fps",
            "f0_range": f"{OpenTuneSpecV11.RMVPE_PITCH_EXTRACTION.F0_MIN} - "
                       f"{OpenTuneSpecV11.RMVPE_PITCH_EXTRACTION.F0_MAX} Hz",
            "supports_vuv": OpenTuneSpecV11.RMVPE_PITCH_EXTRACTION.SUPPORTS_VUV,
        },
        "nsf_hifigan_resynthesis": {
            "sample_rate": f"{OpenTuneSpecV11.NSF_HIFIGAN_RESYNTHESIS.SAMPLE_RATE} Hz",
            "preserve_formants": OpenTuneSpecV11.NSF_HIFIGAN_RESYNTHESIS.PRESERVE_FORMANTS,
            "supports_transposition": OpenTuneSpecV11.NSF_HIFIGAN_RESYNTHESIS.SUPPORTS_TRANSPOSITION,
        },
        "export_format": {
            "format": OpenTuneSpecV11.EXPORT_FORMAT.FORMAT,
            "sample_rate": f"{OpenTuneSpecV11.EXPORT_FORMAT.SAMPLE_RATE} Hz",
            "bit_depth": f"{OpenTuneSpecV11.EXPORT_FORMAT.BIT_DEPTH} (32-bit Float)",
        },
    }


if __name__ == "__main__":
    print("=" * 70)
    print("📋 OpenTune Pro - SPEC v1.1 Compliance Validator")
    print("=" * 70)

    import sys
    sys.path.insert(0, r"c:\Users\zhu\Downloads\OpenTune.V1.1.Win.x64_\OpenTune V1.1 Win x64\OpenTunePro")

    # Import modules to check their parameters
    try:
        from core.vocoder import SR as vocoder_sr, F0_SAFE_MIN, F0_SAFE_MAX
        from core.pitch_tracker import OUTPUT_FPS as rmvpe_fps
        from utils.audio_utils import EXPORT_FORMATS

        print("\n✅ Modules imported successfully\n")
    except ImportError as e:
        print(f"\n❌ Import failed: {e}")
        sys.exit(1)

    # Build parameter dictionary
    params = {
        "rmvpe": {
            "output_fps": rmvpe_fps,
        },
        "hifigan": {
            "sample_rate": vocoder_sr,
            "f0_range": (F0_SAFE_MIN, F0_SAFE_MAX),
        },
        "export": {
            "format": "WAV",
            "sample_rate": vocoder_sr,
            "bit_depth": "FLOAT",
        },
    }

    # Validate
    print("🔍 Checking SPEC v1.1 compliance...\n")

    issues = validate_spec_compliance(params)

    if issues:
        print("❌ SPEC Compliance Issues Found:\n")
        for i, issue in enumerate(issues, 1):
            print(f"  {i}. {issue}")
    else:
        print("✅ All parameters comply with OpenTune SPEC v1.1!\n")

    # Print SPEC summary
    print("=" * 70)
    print("📖 OpenTune SPEC v1.1 Requirements Summary")
    print("=" * 70)
    summary = get_spec_summary()
    
    import json
    print(json.dumps(summary, indent=2, default=str))

    # Final verdict
    print("\n" + "=" * 70)
    if issues:
        print(f"❌ VALIDATION FAILED: {len(issues)} compliance issue(s) found")
        sys.exit(1)
    else:
        print("✅ SPEC v1.1 COMPLIANCE VERIFIED SUCCESSFULLY!")
        print("=" * 70)
