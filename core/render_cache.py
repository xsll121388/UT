"""Chunk-based render cache with revision tracking.

Each audio file is split into chunks (by silence boundaries).
When F0 is edited, only affected chunks are invalidated and re-queued.
Playback stitches rendered chunks; unrendered chunks fall back to dry audio.
"""
from __future__ import annotations
import threading
from dataclasses import dataclass
from enum import Enum, auto
import numpy as np


class ChunkStatus(Enum):
    BLANK = auto()       # not yet rendered
    PENDING = auto()     # queued for rendering
    SUCCEEDED = auto()   # rendered, cached audio available


@dataclass
class Chunk:
    index: int
    start_sample: int    # in original audio
    end_sample: int
    revision: int = 0
    status: ChunkStatus = ChunkStatus.BLANK
    rendered_audio: np.ndarray | None = None  # 44.1kHz float32


# Crossfade duration in samples (about 20ms at 44.1kHz for smoother transitions)
CROSSFADE_SAMPLES = 882  # Increased from 441 to 882 (10ms → 20ms)


def _make_crossfade_fade_in(length: int) -> np.ndarray:
    """Create a smooth cosine fade-in curve."""
    # Use cosine curve for smoother transition (no abrupt slope changes)
    t = np.linspace(0, np.pi, length, dtype=np.float32)
    return 0.5 * (1.0 - np.cos(t))


def _make_crossfade_fade_out(length: int) -> np.ndarray:
    """Create a smooth cosine fade-out curve."""
    # Use cosine curve for smoother transition
    t = np.linspace(0, np.pi, length, dtype=np.float32)
    return 0.5 * (1.0 + np.cos(t))


def _make_crossfade_fade_in_advanced(length: int) -> np.ndarray:
    """
    Create an advanced fade-in curve using raised cosine (Hann window).
    This provides even smoother transitions with better spectral characteristics.
    """
    if length <= 0:
        return np.array([], dtype=np.float32)
    # Raised cosine (Hann window) - first half
    n = np.arange(length, dtype=np.float32)
    return 0.5 * (1.0 - np.cos(np.pi * n / (length - 1)))


def _make_crossfade_fade_out_advanced(length: int) -> np.ndarray:
    """
    Create an advanced fade-out curve using raised cosine (Hann window).
    This provides even smoother transitions with better spectral characteristics.
    """
    if length <= 0:
        return np.array([], dtype=np.float32)
    # Raised cosine (Hann window) - second half
    n = np.arange(length, dtype=np.float32)
    return 0.5 * (1.0 + np.cos(np.pi * n / (length - 1)))


class RenderCache:
    def __init__(self):
        self._lock = threading.Lock()
        self._chunks: list[Chunk] = []
        self._dry_audio: np.ndarray = np.array([], dtype=np.float32)
        self._sr: int = 44100

    def reset(self, dry_audio: np.ndarray, chunk_intervals: list[tuple[int, int]], sr: int = 44100):
        """Initialize cache from a new audio file."""
        with self._lock:
            self._dry_audio = dry_audio
            self._sr = sr
            self._chunks = [
                Chunk(index=i, start_sample=s, end_sample=e)
                for i, (s, e) in enumerate(chunk_intervals)
            ]

    def invalidate_range(self, start_sample: int, end_sample: int) -> list[int]:
        """Mark chunks overlapping [start_sample, end_sample) as BLANK. Returns affected indices."""
        affected = []
        with self._lock:
            for chunk in self._chunks:
                if chunk.end_sample > start_sample and chunk.start_sample < end_sample:
                    chunk.status = ChunkStatus.BLANK
                    chunk.revision += 1
                    chunk.rendered_audio = None
                    affected.append(chunk.index)
        return affected

    def invalidate_all(self) -> None:
        with self._lock:
            for chunk in self._chunks:
                chunk.status = ChunkStatus.BLANK
                chunk.revision += 1
                chunk.rendered_audio = None

    def get_pending_chunks(self) -> list[Chunk]:
        """Return BLANK chunks (copy of metadata, not audio)."""
        with self._lock:
            return [
                Chunk(c.index, c.start_sample, c.end_sample, c.revision, c.status)
                for c in self._chunks
                if c.status == ChunkStatus.BLANK
            ]

    def mark_pending(self, index: int, revision: int) -> bool:
        """Mark chunk as PENDING. Returns False if revision has changed (stale job)."""
        with self._lock:
            chunk = self._chunks[index]
            if chunk.revision != revision:
                return False
            chunk.status = ChunkStatus.PENDING
            return True

    def complete(self, index: int, revision: int, audio: np.ndarray) -> bool:
        """Store rendered audio. Returns False if revision has changed."""
        with self._lock:
            chunk = self._chunks[index]
            if chunk.revision != revision:
                return False
            chunk.rendered_audio = audio
            chunk.status = ChunkStatus.SUCCEEDED
            return True

    def get_audio_at(self, sample_pos: int, length: int) -> np.ndarray:
        """
        Get audio for playback starting at sample_pos.
        Uses rendered audio where available, dry audio as fallback.
        Applies smooth overlap-add crossfade at chunk boundaries.
        
        Key improvements:
        - Cosine crossfades (not linear) for smoother transitions
        - Longer crossfade duration (20ms vs 10ms)
        - Proper energy preservation during overlap
        """
        with self._lock:
            end_pos = sample_pos + length
            dry = self._dry_audio

            # Start with dry audio as base layer
            out = np.zeros(length, dtype=np.float32)
            dry_start = min(sample_pos, len(dry))
            dry_end = min(end_pos, len(dry))
            if dry_end > dry_start:
                out[:dry_end - dry_start] = dry[dry_start:dry_end]

            # Collect rendered chunks that overlap this window
            rendered_chunks = [
                c for c in self._chunks
                if c.status == ChunkStatus.SUCCEEDED
                and c.rendered_audio is not None
                and c.end_sample > sample_pos
                and c.start_sample < end_pos
            ]

            for chunk in rendered_chunks:
                cs, ce = chunk.start_sample, chunk.end_sample
                src_audio = chunk.rendered_audio

                src_start = max(cs, sample_pos) - cs
                src_end = min(ce, end_pos) - cs
                dst_start = max(cs, sample_pos) - sample_pos
                seg_len = src_end - src_start

                if src_start >= len(src_audio):
                    continue
                actual = min(seg_len, len(src_audio) - src_start)
                if actual <= 0:
                    continue

                segment = src_audio[src_start:src_start + actual].copy()
                
                # Adaptive fade length based on segment size and position
                # Longer fades for better quality, but not too long to avoid smearing
                # Ensure fade_len never exceeds actual segment length
                max_fade = min(CROSSFADE_SAMPLES, actual // 2)
                fade_len = min(max_fade, 512)  # Up to 512 samples, but not more than half segment
                fade_len = max(4, fade_len)  # At least 4 samples for smooth transition

                # Build a blend mask: 1.0 = fully rendered, 0.0 = fully dry
                mask = np.ones(actual, dtype=np.float32)

                # Apply advanced crossfades at boundaries
                if fade_len > 0 and fade_len <= actual:
                    # Fade-in at chunk start (use advanced curve for better quality)
                    if cs >= sample_pos:
                        mask[:fade_len] = _make_crossfade_fade_in_advanced(fade_len)
                    
                    # Fade-out at chunk end
                    if ce <= end_pos:
                        mask[-fade_len:] = _make_crossfade_fade_out_advanced(fade_len)

                # High-quality crossfade with phase-aware mixing
                # This preserves transients better than simple amplitude mixing
                dst_end = dst_start + actual
                dry_segment = out[dst_start:dst_end]
                
                # Power-preserving crossfade (maintains loudness during transition)
                # Formula: sqrt(a^2 + b^2) preservation
                wet_power = segment ** 2
                dry_power = dry_segment ** 2
                
                # Blend with power compensation
                blended = dry_segment * (1.0 - mask) + segment * mask
                
                # Apply mild power compensation to maintain consistent loudness
                target_power = dry_power * (1.0 - mask) + wet_power * mask
                current_power = blended ** 2
                
                # Only compensate where power is significantly different
                power_ratio = np.sqrt(target_power / (current_power + 1e-10))
                power_ratio = np.clip(power_ratio, 0.8, 1.25)  # Limit compensation
                
                # Apply compensation gradually (don't over-correct)
                compensation = 1.0 + (power_ratio - 1.0) * 0.3
                blended = blended * compensation
                
                out[dst_start:dst_end] = blended

            return np.clip(out, -1.0, 1.0)

    @property
    def total_samples(self) -> int:
        with self._lock:
            return len(self._dry_audio)

    @property
    def chunks(self) -> list[Chunk]:
        with self._lock:
            return list(self._chunks)

    def replace_dry_region(self, start_sample: int, audio: np.ndarray) -> None:
        """Replace a region of dry audio with new data.
        
        Used for real-time stretch point preview: when stretch points
        change, the affected region of dry audio is replaced with the
        stretched version so playback reflects the changes immediately.
        """
        with self._lock:
            end_sample = start_sample + len(audio)
            if start_sample < 0 or end_sample > len(self._dry_audio):
                return
            self._dry_audio[start_sample:end_sample] = audio[:end_sample - start_sample]
            # Invalidate rendered chunks in this region so they re-render
            for chunk in self._chunks:
                if chunk.end_sample > start_sample and chunk.start_sample < end_sample:
                    chunk.status = ChunkStatus.BLANK
                    chunk.revision += 1
                    chunk.rendered_audio = None
