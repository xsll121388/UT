"""Batch processor: QThread that processes a queue of audio files."""
from __future__ import annotations
import os
import numpy as np
from PyQt6.QtCore import QThread, pyqtSignal

from core.pitch_tracker import PitchTracker
from core.vocoder import Vocoder
from core.pitch_corrector import snap_f0_to_scale
from utils.audio_utils import load_audio, save_audio, split_into_chunks


class BatchJob:
    def __init__(self, input_path: str, output_path: str, params: dict):
        self.input_path = input_path
        self.output_path = output_path
        self.params = params


class BatchProcessor(QThread):
    job_started = pyqtSignal(int, str)       # (index, filename)
    job_progress = pyqtSignal(int, int, int) # (index, done_chunks, total_chunks)
    job_done = pyqtSignal(int)               # index
    job_error = pyqtSignal(int, str)         # (index, error message)
    all_done = pyqtSignal()

    def __init__(self, jobs: list[BatchJob], parent=None):
        super().__init__(parent)
        self._jobs = jobs

    def run(self):
        try:
            tracker = PitchTracker()
            vocoder = Vocoder()
        except Exception as e:
            self.job_error.emit(-1, f"模型加载失败: {e}")
            return

        for i, job in enumerate(self._jobs):
            self.job_started.emit(i, os.path.basename(job.input_path))
            try:
                self._process(i, job, tracker, vocoder)
                self.job_done.emit(i)
            except Exception as e:
                self.job_error.emit(i, str(e))

        self.all_done.emit()

    def _process(self, idx: int, job: BatchJob, tracker: PitchTracker, vocoder: Vocoder):
        audio, sr = load_audio(job.input_path, target_sr=44100)
        f0 = tracker.extract_from_44k(audio)

        params = job.params
        f0_target = snap_f0_to_scale(
            f0,
            root=params.get("key", "C"),
            scale=params.get("scale", "major"),
            retune_speed=params.get("retune_speed", 0.5),
        )

        intervals = split_into_chunks(audio, sr)
        out_chunks = []
        total = len(intervals)

        for ci, (cs, ce) in enumerate(intervals):
            chunk_audio = audio[cs:ce]
            fps = 100
            f0_start = int(cs / sr * fps)
            f0_end = int(ce / sr * fps) + 1
            f0_chunk = f0_target[f0_start:min(f0_end, len(f0_target))]

            if len(f0_chunk) == 0:
                out_chunks.append(chunk_audio)
            else:
                rendered = vocoder.synthesize(chunk_audio, f0_chunk, sr)
                if len(rendered) > len(chunk_audio):
                    rendered = rendered[:len(chunk_audio)]
                elif len(rendered) < len(chunk_audio):
                    rendered = np.pad(rendered, (0, len(chunk_audio) - len(rendered)))
                out_chunks.append(rendered)

            self.job_progress.emit(idx, ci + 1, total)

        # Stitch output
        output = np.zeros(len(audio), dtype=np.float32)
        for (cs, ce), chunk in zip(intervals, out_chunks):
            output[cs:ce] = chunk

        os.makedirs(os.path.dirname(job.output_path) or ".", exist_ok=True)
        save_audio(job.output_path, output, sr)
