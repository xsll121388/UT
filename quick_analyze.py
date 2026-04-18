"""
Quick audio analyzer for large files - analyzes only first 10 seconds.
"""

import numpy as np
import soundfile as sf
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from analyze_rendered_audio import analyze_audio_file

def quick_analyze(filepath: str, duration: float = 10.0):
    """Analyze only the first N seconds of audio."""
    
    print(f"🔍 快速分析：{filepath}")
    print(f"   分析时长：前 {duration} 秒")
    print("=" * 70)
    
    # Load audio
    y, sr = sf.read(filepath, start=0, stop=int(sr * duration) if 'sr' in locals() else int(44100 * duration))
    
    # Save as temporary file for full analysis
    temp_file = Path(filepath).parent / f"_quick_test_{Path(filepath).stem}.wav"
    sf.write(temp_file, y, sr)
    
    # Run full analysis on the short segment
    results = analyze_audio_file(str(temp_file))
    
    # Clean up
    try:
        temp_file.unlink()
    except:
        pass
    
    return results


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("用法：python quick_analyze.py <音频文件路径> [分析时长 (秒)]")
        sys.exit(1)
    
    filepath = sys.argv[1]
    duration = float(sys.argv[2]) if len(sys.argv) > 2 else 10.0
    
    quick_analyze(filepath, duration)
