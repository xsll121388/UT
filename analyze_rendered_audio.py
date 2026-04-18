"""
Rendered Audio Analyzer - 渲染音频分析工具
Analyzes rendered audio to identify quality issues and artifacts.
"""

import numpy as np
import librosa
import librosa.display
from scipy import signal
from scipy.fft import fft, fftfreq
import warnings
from pathlib import Path


def analyze_audio_file(filepath: str) -> dict:
    """
    Comprehensive audio analysis.
    
    Returns:
        Dictionary containing all analysis results
    """
    print(f"🔍 分析文件：{filepath}")
    print("=" * 70)
    
    # Load audio
    y, sr = librosa.load(filepath, sr=None)
    
    results = {
        'filepath': filepath,
        'sample_rate': sr,
        'duration': len(y) / sr,
        'samples': len(y),
    }
    
    # 1. Basic Statistics
    print("\n📊 基本统计信息:")
    print(f"  采样率：{sr} Hz")
    print(f"  时长：{results['duration']:.2f} 秒")
    print(f"  采样点数：{results['samples']:,}")
    
    peak = np.max(np.abs(y))
    rms = np.sqrt(np.mean(y**2))
    
    print(f"  峰值幅度：{peak:.6f}")
    print(f"  RMS 响度：{rms:.6f}")
    
    results['peak'] = peak
    results['rms'] = rms
    
    # Check clipping
    clipped = np.sum(np.abs(y) >= 0.99)
    clip_percent = clipped / len(y) * 100
    print(f"  削波样本：{clipped} ({clip_percent:.4f}%)")
    results['clipped_samples'] = clipped
    results['clip_percent'] = clip_percent
    
    # 2. Frequency Analysis
    print("\n🎼 频谱分析:")
    
    # Compute spectrum
    n = len(y)
    yf = fft(y)
    xf = fftfreq(n, 1/sr)
    
    # Get positive frequencies
    pos_mask = xf > 0
    xf = xf[pos_mask]
    yf = 2.0/n * np.abs(yf[pos_mask])
    
    # Find dominant frequencies
    top_indices = np.argsort(yf)[-10:][::-1]
    print("  主要频率成分:")
    for i in top_indices[:5]:
        print(f"    {xf[i]:.1f} Hz (幅度：{yf[i]:.6f})")
    
    results['dominant_frequencies'] = [(float(xf[i]), float(yf[i])) for i in top_indices[:5]]
    
    # Spectral centroid
    spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
    mean_centroid = np.mean(spectral_centroid)
    print(f"  频谱质心：{mean_centroid:.1f} Hz")
    results['spectral_centroid'] = float(mean_centroid)
    
    # Spectral rolloff
    spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr, roll_percent=0.85)
    mean_rolloff = np.mean(spectral_rolloff)
    print(f"  频谱滚降点：{mean_rolloff:.1f} Hz")
    results['spectral_rolloff'] = float(mean_rolloff)
    
    # 3. Check for artifacts
    print("\n⚠️  伪影检测:")
    
    # Check for DC offset
    dc_offset = np.mean(y)
    print(f"  DC 偏移：{dc_offset:.8f}", end="")
    if abs(dc_offset) > 0.01:
        print(" ⚠️  警告：DC 偏移过大")
        results['dc_offset_warning'] = True
    else:
        print(" ✓")
        results['dc_offset_warning'] = False
    
    # Check for high-frequency noise
    hf_energy = np.sum(yf[xf > 16000])
    total_energy = np.sum(yf)
    hf_ratio = hf_energy / total_energy * 100
    print(f"  高频能量 (>16kHz): {hf_ratio:.2f}%")
    
    if hf_ratio > 15:
        print("    ⚠️  警告：高频能量过高 - 可能有金属音/数字失真")
        results['hf_noise_warning'] = True
    else:
        print("    ✓")
        results['hf_noise_warning'] = False
    
    # Check for low-frequency rumble
    lf_energy = np.sum(yf[xf < 50])
    lf_ratio = lf_energy / total_energy * 100
    print(f"  低频能量 (<50Hz): {lf_ratio:.2f}%")
    
    if lf_ratio > 5:
        print("    ⚠️  警告：低频能量过高 - 可能有低频轰鸣")
        results['lf_noise_warning'] = True
    else:
        print("    ✓")
        results['lf_noise_warning'] = False
    
    # 4. Check for comb filtering (metallic sound)
    print("\n🔍 梳状滤波检测:")
    
    # Compute autocorrelation
    autocorr = np.correlate(y, y, mode='full')
    autocorr = autocorr[len(autocorr)//2:]
    autocorr = autocorr / np.max(autocorr)
    
    # Look for periodic peaks (indicative of comb filtering)
    # Comb filtering typically creates delays between 1-20ms
    min_delay = int(sr * 0.001)   # 1ms
    max_delay = int(sr * 0.02)    # 20ms
    
    comb_detected = False
    for delay in range(min_delay, max_delay):
        if delay < len(autocorr):
            if autocorr[delay] > 0.3:  # Significant correlation at delay
                print(f"  ⚠️  检测到梳状滤波特征 (延迟：{delay/sr*1000:.2f}ms, 相关性：{autocorr[delay]:.3f})")
                comb_detected = True
                results['comb_filtering_delay'] = float(delay/sr*1000)
                results['comb_filtering_correlation'] = float(autocorr[delay])
                break
    
    if not comb_detected:
        print("  ✓ 未检测到明显梳状滤波")
        results['comb_filtering_detected'] = False
    else:
        results['comb_filtering_detected'] = True
    
    # 5. Check for F0 artifacts (pitch jumps)
    print("\n🎵 音高连续性检测:")
    
    try:
        f0, voiced_flag, _ = librosa.pyin(
            y, 
            fmin=50, 
            fmax=1100, 
            sr=sr,
            frame_length=2048,
            win_length=2048,
            hop_length=512
        )
        
        f0 = f0[voiced_flag]
        
        if len(f0) > 10:
            # Check for sudden jumps
            f0_diff = np.diff(f0)
            jump_threshold = 100  # Hz - sudden jumps indicate artifacts
            
            jumps = np.sum(np.abs(f0_diff) > jump_threshold)
            jump_rate = jumps / len(f0) * 100
            
            print(f"  检测到 {jumps} 次音高突变 (>100Hz)")
            print(f"  音高突变率：{jump_rate:.2f}%")
            
            if jump_rate > 5:
                print("    ⚠️  警告：音高突变过多 - F0 处理可能有问题")
                results['f0_jump_warning'] = True
            else:
                print("    ✓")
                results['f0_jump_warning'] = False
            
            # Check F0 range
            print(f"  F0 范围：[{f0.min():.1f}, {f0.max():.1f}] Hz")
            print(f"  F0 中位数：{np.median(f0):.1f} Hz")
            
            results['f0_range'] = (float(f0.min()), float(f0.max()))
            results['f0_median'] = float(np.median(f0))
        else:
            print("  ⚠️  无法检测： voiced frames 太少")
            results['f0_jump_warning'] = None
            
    except Exception as e:
        print(f"  ⚠️  音高检测失败：{e}")
        results['f0_jump_warning'] = None
    
    # 6. Check for formant preservation
    print("\n🎤 共振峰检测:")
    
    # Use LPC to estimate formants
    try:
        # Analyze a voiced segment
        voiced_mask = f0 > 0 if 'f0' in dir() else np.abs(y) > 0.1 * peak
        if np.sum(voiced_mask) > sr:  # At least 1 second
            # Take a segment
            voiced_indices = np.where(voiced_mask)[0]
            start = voiced_indices[len(voiced_indices)//2]
            segment = y[start:start+int(sr*0.1)]  # 100ms segment
            
            # LPC analysis
            order = int(2 + sr/1000)  # Typical formant order
            a = signal.lpc(segment, order)
            
            # Get formants from LPC coefficients
            # (simplified - just check if LPC works)
            print(f"  ✓ LPC 分析成功 (阶数：{order})")
            results['formant_analysis_possible'] = True
        else:
            print("  ⚠️  无法分析：voiced segment 太短")
            results['formant_analysis_possible'] = False
            
    except Exception as e:
        print(f"  ⚠️  共振峰分析失败：{e}")
        results['formant_analysis_possible'] = False
    
    # 7. Check for high-frequency distortion (common with wrong LOG_MEL_MAX)
    print("\n🔬 高频失真检测:")
    
    # Compute spectrogram
    D = librosa.amplitude_to_db(np.abs(librosa.stft(y, n_fft=2048, hop_length=512)), ref=np.max)
    
    # Check energy distribution
    freq_bins = librosa.fft_frequencies(sr=sr, n_fft=2048)
    
    # Energy in different bands
    lf_band = np.sum(D[freq_bins < 500])
    mf_band = np.sum(D[(freq_bins >= 500) & (freq_bins < 4000)])
    hf_band = np.sum(D[freq_bins >= 4000])
    
    total_band = lf_band + mf_band + hf_band
    
    lf_percent = lf_band / total_band * 100
    mf_percent = mf_band / total_band * 100
    hf_percent = hf_band / total_band * 100
    
    print(f"  低频能量 (<500Hz): {lf_percent:.1f}%")
    print(f"  中频能量 (500-4kHz): {mf_percent:.1f}%")
    print(f"  高频能量 (>4kHz): {hf_percent:.1f}%")
    
    # Typical voice should have balanced distribution
    if hf_percent < 10:
        print("  ⚠️  警告：高频能量过低 - 声音可能发闷 (LOG_MEL_MAX 可能过小)")
        results['hf_deficiency_warning'] = True
    elif hf_percent > 25:
        print("  ⚠️  警告：高频能量过高 - 声音可能刺耳 (LOG_MEL_MAX 可能过大)")
        results['hf_excess_warning'] = True
    else:
        print("  ✓ 高频能量分布正常")
        results['hf_deficiency_warning'] = False
        results['hf_excess_warning'] = False
    
    results['energy_distribution'] = {
        'low_freq': float(lf_percent),
        'mid_freq': float(mf_percent),
        'high_freq': float(hf_percent)
    }
    
    # 8. Summary and Recommendations
    print("\n" + "=" * 70)
    print("📋 诊断总结:")
    
    issues = []
    
    if results.get('clip_percent', 0) > 1:
        issues.append("❌ 削波失真")
    
    if results.get('dc_offset_warning'):
        issues.append("❌ DC 偏移")
    
    if results.get('comb_filtering_detected'):
        issues.append(f"❌ 梳状滤波 (延迟 {results.get('comb_filtering_delay', 0):.2f}ms)")
    
    if results.get('hf_noise_warning'):
        issues.append("❌ 高频噪声/金属音")
    
    if results.get('f0_jump_warning'):
        issues.append("❌ 音高突变")
    
    if results.get('hf_deficiency_warning'):
        issues.append("❌ 高频不足 (声音发闷)")
    
    if results.get('hf_excess_warning'):
        issues.append("❌ 高频过量 (声音刺耳)")
    
    if issues:
        print("\n⚠️  检测到的问题:")
        for issue in issues:
            print(f"  {issue}")
        
        print("\n💡 建议:")
        if results.get('hf_deficiency_warning'):
            print("  → 尝试增大 LOG_MEL_MAX (当前 2.5，可尝试 2.8-3.0)")
        if results.get('comb_filtering_detected'):
            print("  → 检查上下文重叠设置和窗口函数")
        if results.get('f0_jump_warning'):
            print("  → 增强 F0 平滑处理 (增大 median_kernel)")
        if results.get('hf_noise_warning'):
            print("  → 尝试减小 LOG_MEL_MAX 或检查 vocoder 实现")
    else:
        print("\n✅ 未检测到明显问题")
    
    print("\n" + "=" * 70)
    
    return results


def compare_audio_files(original: str, rendered: str):
    """Compare original and rendered audio to identify differences."""
    
    print("\n🔍 对比分析：原音频 vs 渲染音频")
    print("=" * 70)
    
    orig_results = analyze_audio_file(original)
    print("\n" + "=" * 70)
    print("\n")
    rend_results = analyze_audio_file(rendered)
    
    print("\n" + "=" * 70)
    print("📊 对比结果:")
    print("=" * 70)
    
    # Compare key metrics
    print(f"\n采样率：{orig_results['sample_rate']} → {rend_results['sample_rate']} Hz")
    print(f"时长：{orig_results['duration']:.2f} → {rend_results['duration']:.2f} 秒")
    print(f"RMS: {orig_results['rms']:.6f} → {rend_results['rms']:.6f}")
    print(f"峰值：{orig_results['peak']:.6f} → {rend_results['peak']:.6f}")
    
    if 'spectral_centroid' in orig_results and 'spectral_centroid' in rend_results:
        print(f"\n频谱质心：{orig_results['spectral_centroid']:.1f} → {rend_results['spectral_centroid']:.1f} Hz")
        
        centroid_diff = rend_results['spectral_centroid'] - orig_results['spectral_centroid']
        if centroid_diff > 500:
            print("  ⚠️  渲染后频谱质心显著上移 - 声音可能变亮/变薄")
        elif centroid_diff < -500:
            print("  ⚠️  渲染后频谱质心显著下移 - 声音可能变暗/变闷")
        else:
            print("  ✓ 频谱质心变化在正常范围内")
    
    if 'energy_distribution' in orig_results and 'energy_distribution' in rend_results:
        print(f"\n能量分布对比:")
        orig_e = orig_results['energy_distribution']
        rend_e = rend_results['energy_distribution']
        
        print(f"  低频：{orig_e['low_freq']:.1f}% → {rend_e['low_freq']:.1f}% ({rend_e['low_freq']-orig_e['low_freq']:+.1f}%)")
        print(f"  中频：{orig_e['mid_freq']:.1f}% → {rend_e['mid_freq']:.1f}% ({rend_e['mid_freq']-orig_e['mid_freq']:+.1f}%)")
        print(f"  高频：{orig_e['high_freq']:.1f}% → {rend_e['high_freq']:.1f}% ({rend_e['high_freq']-orig_e['high_freq']:+.1f}%)")
        
        hf_diff = rend_e['high_freq'] - orig_e['high_freq']
        if hf_diff < -5:
            print("  ⚠️  渲染后高频能量显著损失 - 共振峰可能受损")
        elif hf_diff > 5:
            print("  ⚠️  渲染后高频能量异常增加 - 可能有谐波失真")
        else:
            print("  ✓ 能量分布变化在正常范围内")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) == 1:
        print("用法：python analyze_rendered_audio.py <音频文件路径>")
        print("或：python analyze_rendered_audio.py <原文件> <渲染文件> (对比模式)")
        print("\n示例:")
        print("  python analyze_rendered_audio.py output.wav")
        print("  python analyze_rendered_audio.py input.wav output.wav")
        sys.exit(1)
    
    if len(sys.argv) == 2:
        # Single file analysis
        analyze_audio_file(sys.argv[1])
    elif len(sys.argv) == 3:
        # Comparison mode
        compare_audio_files(sys.argv[1], sys.argv[2])
