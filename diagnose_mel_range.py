"""诊断实际音频的Mel频谱范围"""
import numpy as np
import soundfile as sf
import librosa
from pathlib import Path
import sys

def analyze_mel_range(audio_file: str):
    """分析音频文件的mel频谱范围"""

    print(f"分析文件: {audio_file}")
    print("=" * 70)

    # 加载音频
    audio, sr = sf.read(audio_file)
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)

    # 重采样到44100
    if sr != 44100:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=44100)
        sr = 44100

    print(f"音频长度: {len(audio)/sr:.2f}秒")
    print(f"RMS: {np.sqrt(np.mean(audio**2)):.6f}")
    print(f"峰值: {np.max(np.abs(audio)):.6f}")

    # 计算mel频谱（使用模型配置）
    mel = librosa.feature.melspectrogram(
        y=audio,
        sr=44100,
        n_fft=2048,
        hop_length=512,
        win_length=2048,
        n_mels=128,
        fmin=40,
        fmax=16000,
        power=1.0  # Magnitude spectrum (matches vocoder training)
    )

    # 转换为log-mel
    log_mel = np.log(np.clip(mel, 1e-10, None))

    print(f"\nMel频谱形状: {mel.shape}")
    print(f"\n原始Mel统计:")
    print(f"  最小值: {np.min(mel):.6f}")
    print(f"  最大值: {np.max(mel):.6f}")
    print(f"  均值: {np.mean(mel):.6f}")
    print(f"  中位数: {np.median(mel):.6f}")

    print(f"\nLog-Mel统计:")
    print(f"  最小值: {np.min(log_mel):.2f}")
    print(f"  最大值: {np.max(log_mel):.2f}")
    print(f"  均值: {np.mean(log_mel):.2f}")
    print(f"  中位数: {np.median(log_mel):.2f}")

    # 百分位数分析
    percentiles = [1, 5, 10, 25, 50, 75, 90, 95, 99]
    print(f"\nLog-Mel百分位数:")
    for p in percentiles:
        val = np.percentile(log_mel, p)
        print(f"  {p:2d}%: {val:6.2f}")

    # 推荐范围
    p1 = np.percentile(log_mel, 1)
    p99 = np.percentile(log_mel, 99)

    print(f"\n" + "=" * 70)
    print(f"推荐的LOG_MEL范围:")
    print(f"  LOG_MEL_MIN: {np.floor(p1):.0f}  (当前1%分位: {p1:.2f})")
    print(f"  LOG_MEL_MAX: {np.ceil(p99):.0f}  (当前99%分位: {p99:.2f})")
    print("=" * 70)

    # 检查当前设置
    current_min = -12.0
    current_max = 2.0

    clipped_low = np.sum(log_mel < current_min)
    clipped_high = np.sum(log_mel > current_max)
    total = log_mel.size

    print(f"\n使用当前设置 [{current_min}, {current_max}]:")
    print(f"  被截断的低值: {clipped_low} / {total} ({100*clipped_low/total:.2f}%)")
    print(f"  被截断的高值: {clipped_high} / {total} ({100*clipped_high/total:.2f}%)")

    if clipped_high > total * 0.01:
        print(f"\n⚠️  警告: {100*clipped_high/total:.1f}% 的高能量帧被截断!")
        print(f"   这会导致: 音量降低、高频空洞、动态范围压缩")
        print(f"   建议: 增大 LOG_MEL_MAX 到至少 {np.ceil(p99):.0f}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("用法: python diagnose_mel_range.py <音频文件>")
        print("\n示例:")
        print('  python diagnose_mel_range.py "test.wav"')
        sys.exit(1)

    analyze_mel_range(sys.argv[1])
