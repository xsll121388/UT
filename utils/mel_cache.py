"""Mel spectrogram cache to avoid redundant computations."""
from __future__ import annotations
import numpy as np
from typing import Optional, Dict
import hashlib
from collections import OrderedDict
from utils.performance import get_monitor


class MelSpectrogramCache:
    """
    Cache for mel spectrograms to avoid redundant computations.
    
    When the same audio segment is processed multiple times (e.g., during
    F0 editing), the mel spectrogram can be reused from cache instead of
    being recomputed.
    
    Usage:
        cache = MelSpectrogramCache(max_size=100)
        
        # First call - computes mel
        mel1 = cache.get_or_compute(audio, sr=44100, n_fft=2048, ...)
        
        # Second call with same audio - returns cached mel
        mel2 = cache.get_or_compute(audio, sr=44100, n_fft=2048, ...)
        
        # Check cache efficiency
        print(f"Hit rate: {cache.hit_rate:.1%}")
    """
    
    def __init__(self, max_size: int = 100):
        """
        Initialize the mel spectrogram cache.
        
        Args:
            max_size: Maximum number of mel spectrograms to cache
        """
        self._cache: OrderedDict[str, np.ndarray] = OrderedDict()
        self._max_size = max_size
        self._hit_count = 0
        self._miss_count = 0
        self._monitor = get_monitor()
    
    def get_or_compute(
        self,
        audio: np.ndarray,
        sr: int = 44100,
        n_fft: int = 2048,
        hop_length: int = 512,
        win_length: Optional[int] = None,
        n_mels: int = 128,
        fmin: int = 40,
        fmax: int = 16000,
        power: float = 2.0
    ) -> np.ndarray:
        """
        Get mel spectrogram from cache or compute it.
        
        Args:
            audio: Audio signal (samples,)
            sr: Sample rate
            n_fft: FFT window size
            hop_length: Hop length
            win_length: Window length (default: n_fft)
            n_mels: Number of mel bands
            fmin: Minimum frequency
            fmax: Maximum frequency
            power: Power exponent for spectrogram
            
        Returns:
            Mel spectrogram (n_mels, n_frames)
        """
        # Create cache key from audio hash and parameters
        key = self._create_cache_key(
            audio, sr, n_fft, hop_length, win_length, n_mels, fmin, fmax, power
        )
        
        # Check cache
        if key in self._cache:
            self._hit_count += 1
            # Move to end (most recently used)
            self._cache.move_to_end(key)
            return self._cache[key].copy()
        
        # Cache miss - compute mel spectrogram
        self._miss_count += 1
        
        with self._monitor.timer("mel_spectrogram_compute"):
            import librosa
            mel = librosa.feature.melspectrogram(
                y=audio,
                sr=sr,
                n_fft=n_fft,
                hop_length=hop_length,
                win_length=win_length or n_fft,
                n_mels=n_mels,
                fmin=fmin,
                fmax=fmax,
                power=power
            )
        
        # Store in cache
        self._cache[key] = mel
        
        # Enforce size limit
        while len(self._cache) > self._max_size:
            self._cache.popitem(last=False)
        
        return mel.copy()
    
    def _create_cache_key(
        self,
        audio: np.ndarray,
        sr: int,
        n_fft: int,
        hop_length: int,
        win_length: Optional[int],
        n_mels: int,
        fmin: int,
        fmax: int,
        power: float
    ) -> str:
        """Create a unique cache key for audio and parameters."""
        # Hash audio data (first 10 seconds for efficiency)
        audio_sample = audio[:sr * 10].tobytes()
        audio_hash = hashlib.md5(audio_sample).hexdigest()
        
        # Include parameters in key
        param_hash = hashlib.md5(
            f"{sr}:{n_fft}:{hop_length}:{win_length}:{n_mels}:{fmin}:{fmax}:{power}".encode()
        ).hexdigest()
        
        return f"{audio_hash}:{param_hash}"
    
    def clear(self):
        """Clear the cache."""
        self._cache.clear()
        self._hit_count = 0
        self._miss_count = 0
    
    @property
    def hit_count(self) -> int:
        """Number of cache hits."""
        return self._hit_count
    
    @property
    def miss_count(self) -> int:
        """Number of cache misses."""
        return self._miss_count
    
    @property
    def hit_rate(self) -> float:
        """Cache hit rate (0.0 to 1.0)."""
        total = self._hit_count + self._miss_count
        return self._hit_count / total if total > 0 else 0.0
    
    @property
    def size(self) -> int:
        """Current cache size."""
        return len(self._cache)
    
    @property
    def stats(self) -> dict:
        """Get cache statistics."""
        return {
            'size': self.size,
            'max_size': self._max_size,
            'hits': self._hit_count,
            'misses': self._miss_count,
            'hit_rate': self.hit_rate
        }
    
    def report(self) -> str:
        """Generate a cache statistics report."""
        stats = self.stats
        return (
            f"Mel Spectrogram Cache Statistics:\n"
            f"  Size: {stats['size']}/{stats['max_size']}\n"
            f"  Hits: {stats['hits']}\n"
            f"  Misses: {stats['misses']}\n"
            f"  Hit Rate: {stats['hit_rate']:.1%}\n"
        )


# Global mel spectrogram cache instance
_mel_cache = MelSpectrogramCache(max_size=100)


def get_mel_cache() -> MelSpectrogramCache:
    """Get the global mel spectrogram cache instance."""
    return _mel_cache


def compute_cached_mel(
    audio: np.ndarray,
    sr: int = 44100,
    n_fft: int = 2048,
    hop_length: int = 512,
    win_length: Optional[int] = None,
    n_mels: int = 128,
    fmin: int = 40,
    fmax: int = 16000,
    power: float = 2.0
) -> np.ndarray:
    """
    Compute mel spectrogram using the global cache.
    
    This is a convenience function that uses the global mel cache instance.
    
    Args:
        audio: Audio signal (samples,)
        sr: Sample rate
        n_fft: FFT window size
        hop_length: Hop length
        win_length: Window length (default: n_fft)
        n_mels: Number of mel bands
        fmin: Minimum frequency
        fmax: Maximum frequency
        power: Power exponent for spectrogram
        
    Returns:
        Mel spectrogram (n_mels, n_frames)
    """
    return get_mel_cache().get_or_compute(
        audio, sr, n_fft, hop_length, win_length, n_mels, fmin, fmax, power
    )
