"""Background render worker: serial QThread that processes vocoder jobs."""
from __future__ import annotations
import threading
import warnings
import numpy as np
from PyQt6.QtCore import QThread, pyqtSignal

from core.render_cache import RenderCache, ChunkStatus
from core.vocoder import Vocoder, HOP, N_FFT
from core.pitch_corrector import snap_f0_to_scale, smooth_f0
from core.f0_processor import advanced_f0_processing


class RenderWorker(QThread):
    chunk_done = pyqtSignal(int)   # emits chunk index when rendered
    error = pyqtSignal(str)

    def __init__(self, cache: RenderCache, parent=None):
        super().__init__(parent)
        self._cache = cache
        self._vocoder: Vocoder | None = None
        self._wake = threading.Event()
        self._stop = False

        # Current render parameters (set by main thread)
        self._dry_audio: np.ndarray = np.array([], dtype=np.float32)
        self._f0_target: np.ndarray = np.array([], dtype=np.float32)
        self._sr: int = 44100
        self._f0_fine_tune: float = 0.0  # Default: no pitch adjustment
        self._enable_advanced_f0: bool = True  # Enable professional-grade F0 processing
        self._lock = threading.Lock()

    def set_params(self, dry_audio: np.ndarray, f0_target: np.ndarray, sr: int = 44100,
                   f0_fine_tune: float = -15.0,
                   enable_advanced_f0: bool = True):
        with self._lock:
            self._dry_audio = dry_audio
            self._f0_target = f0_target
            self._sr = sr
            self._f0_fine_tune = f0_fine_tune
            self._enable_advanced_f0 = enable_advanced_f0

    def wake(self):
        """Signal worker to check for pending chunks."""
        self._wake.set()

    def stop(self):
        self._stop = True
        self._wake.set()

    def run(self):
        try:
            self._vocoder = Vocoder()
        except Exception as e:
            self.error.emit(str(e))
            return

        while not self._stop:
            self._wake.wait(timeout=1.0)
            self._wake.clear()
            if self._stop:
                break
            self._process_pending()

    def _process_pending(self):
        pending = self._cache.get_pending_chunks()
        for chunk_meta in pending:
            if self._stop:
                break
            idx = chunk_meta.index
            rev = chunk_meta.revision

            if not self._cache.mark_pending(idx, rev):
                continue  # stale

            with self._lock:
                dry = self._dry_audio
                f0 = self._f0_target
                sr = self._sr

            if len(dry) == 0 or len(f0) == 0:
                continue

            cs, ce = chunk_meta.start_sample, chunk_meta.end_sample
            chunk_dur_samples = ce - cs

            # Add overlap context for smoother boundaries
            # Use 4 hop sizes at 44.1kHz ≈ 46ms of context for better quality
            overlap = HOP * 4  # 2048 samples context
            context_start = max(0, cs - overlap)
            context_end = min(len(dry), ce + overlap)

            chunk_audio_with_context = dry[context_start:context_end]

            # Calculate F0 frames for the extended context window
            # RMVPE outputs at 100fps (16kHz input with hop=160)
            # Use round() instead of int() to avoid systematic alignment drift
            fps = 100
            f0_context_start = max(0, round(context_start / sr * fps))
            f0_context_end = min(len(f0), round(context_end / sr * fps) + 1)
            f0_chunk = f0[f0_context_start:f0_context_end]

            if len(f0_chunk) == 0:
                warnings.warn(f"Chunk {idx}: Empty F0 after slicing")
                continue

            # Apply advanced F0 processing to eliminate artifacts.
            # Skip outlier removal and median smoothing — the f0_target is
            # already the user's intended curve (hand-drawn or MIDI-snapped).
            # Only keep V/UV transition fades to prevent click artifacts.
            if self._enable_advanced_f0 and len(f0_chunk) > 10:
                f0_chunk = advanced_f0_processing(
                    f0_chunk,
                    median_kernel=1,            # No smoothing — preserve user curve
                    vuv_fade_frames=5,          # Smooth V/UV transitions (50ms) - longer for better quality
                    remove_octave_errors=False, # User curve is intentional
                    fill_gaps=True              # Improve continuity
                )

            if len(chunk_audio_with_context) < N_FFT:
                warnings.warn(f"Chunk {idx}: Audio too short ({len(chunk_audio_with_context)} samples), need {N_FFT}")
                continue

            try:
                # Extract the original chunk (without context) for RMS matching
                audio_for_rms = dry[cs:ce]

                rendered_with_context = self._vocoder.synthesize(
                    chunk_audio_with_context, f0_chunk, sr,
                    audio_for_rms=audio_for_rms,
                    f0_fine_tune=self._f0_fine_tune  # Apply pitch fine-tuning
                )

                # Extract only the original chunk region (remove context)
                offset_start = cs - context_start
                offset_end = offset_start + chunk_dur_samples

                if offset_end <= len(rendered_with_context):
                    rendered = rendered_with_context[offset_start:offset_end]
                else:
                    # Fallback if context extraction fails
                    rendered = rendered_with_context[:chunk_dur_samples]
                    warnings.warn(f"Chunk {idx}: Context extraction issue, using fallback")

                # Ensure exact length match
                if len(rendered) > chunk_dur_samples:
                    rendered = rendered[:chunk_dur_samples]
                elif len(rendered) < chunk_dur_samples:
                    rendered = np.pad(rendered, (0, chunk_dur_samples - len(rendered)))

                if self._cache.complete(idx, rev, rendered):
                    self.chunk_done.emit(idx)

            except ValueError as ve:
                warnings.warn(f"Chunk {idx}: Validation error - {ve}")
                self.error.emit(f"Chunk {idx}: {ve}")
            except RuntimeError as re:
                warnings.warn(f"Chunk {idx}: Inference failed - {re}")
                self.error.emit(f"Chunk {idx}: {re}")
            except Exception as e:
                self.error.emit(f"Chunk {idx}: {e}")
