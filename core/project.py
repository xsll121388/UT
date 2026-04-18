"""Project state: dataclass + JSON save/load.

Architecture:
- TuningData: Independent tuning data for a specific audio context,
  keyed by audio identity (file path + MIDI hash).
- ClipState: Audio clip metadata with a reference (tuning_id) to its TuningData.
- ProjectState: Project container with a tuning_map for all TuningData entries.

The tuning_map decouples tuning data from audio clips, so that:
- Re-importing audio (e.g., lyrics change) preserves tuning data
- Control point operations don't reset F0 curves
- Project save/load includes all tuning history
"""
from __future__ import annotations
import base64
import hashlib
import json
import os
from dataclasses import dataclass, field
import numpy as np
from utils.audio_utils import MidiNote, midi_note_to_dict, midi_note_from_dict


@dataclass
class TuningData:
    f0_original: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.float32))
    f0_target: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.float32))
    f0_manually_edited: bool = False
    key: str = "C"
    scale: str = "major"
    retune_speed: float = 0.5

    def to_dict(self) -> dict:
        return {
            "f0_original": _arr_to_b64(self.f0_original),
            "f0_target": _arr_to_b64(self.f0_target),
            "f0_manually_edited": self.f0_manually_edited,
            "key": self.key,
            "scale": self.scale,
            "retune_speed": self.retune_speed,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "TuningData":
        return cls(
            f0_original=_b64_to_arr(d.get("f0_original", "")),
            f0_target=_b64_to_arr(d.get("f0_target", "")),
            f0_manually_edited=d.get("f0_manually_edited", False),
            key=d.get("key", "C"),
            scale=d.get("scale", "major"),
            retune_speed=d.get("retune_speed", 0.5),
        )


@dataclass
class ClipState:
    audio_path: str
    tuning_id: str = ""


def compute_tuning_id(audio_path: str = "", midi_hash: str = "") -> str:
    raw = f"{audio_path}|{midi_hash}"
    return hashlib.md5(raw.encode()).hexdigest()[:12]


def compute_midi_hash(midi_notes: list) -> str:
    parts = []
    for n in midi_notes:
        lyric = getattr(n, 'lyric', '')
        parts.append(f"{n.pitch}:{n.start_sec:.4f}:{n.end_sec:.4f}:{lyric}")
    raw = "|".join(parts)
    return hashlib.md5(raw.encode()).hexdigest()[:12]


@dataclass
class ProjectState:
    name: str = "Untitled"
    sample_rate: int = 44100
    clips: list[ClipState] = field(default_factory=list)
    tuning_map: dict[str, TuningData] = field(default_factory=dict)
    midi_notes: list[MidiNote] = field(default_factory=list)
    sample_folder: str = ""  # Path to sample folder if used

    def get_tuning(self, tuning_id: str) -> TuningData | None:
        return self.tuning_map.get(tuning_id)

    def set_tuning(self, tuning_id: str, tuning: TuningData) -> None:
        self.tuning_map[tuning_id] = tuning

    def ensure_tuning(self, tuning_id: str) -> TuningData:
        if tuning_id not in self.tuning_map:
            self.tuning_map[tuning_id] = TuningData()
        return self.tuning_map[tuning_id]

    def get_clip_tuning(self, clip_index: int = 0) -> TuningData:
        if clip_index >= len(self.clips):
            return TuningData()
        clip = self.clips[clip_index]
        if not clip.tuning_id:
            clip.tuning_id = compute_tuning_id(clip.audio_path)
        return self.ensure_tuning(clip.tuning_id)

    def to_dict(self) -> dict:
        d = {
            "name": self.name,
            "sample_rate": self.sample_rate,
            "clips": [],
            "tuning_map": {},
            "midi_notes": [midi_note_to_dict(n) for n in self.midi_notes],
            "sample_folder": self.sample_folder,
        }
        for clip in self.clips:
            d["clips"].append({
                "audio_path": clip.audio_path,
                "tuning_id": clip.tuning_id,
            })
        for tid, tuning in self.tuning_map.items():
            d["tuning_map"][tid] = tuning.to_dict()
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "ProjectState":
        proj = cls(
            name=d.get("name", "Untitled"),
            sample_rate=d.get("sample_rate", 44100),
            midi_notes=[midi_note_from_dict(n) for n in d.get("midi_notes", [])],
            sample_folder=d.get("sample_folder", ""),
        )
        for cd in d.get("clips", []):
            proj.clips.append(ClipState(
                audio_path=cd["audio_path"],
                tuning_id=cd.get("tuning_id", ""),
            ))
        for tid, td in d.get("tuning_map", {}).items():
            proj.tuning_map[tid] = TuningData.from_dict(td)
        return proj

    def save(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, path: str) -> "ProjectState":
        with open(path, "r", encoding="utf-8") as f:
            return cls.from_dict(json.load(f))

    def migrate_legacy(self, old_clips: list[dict]) -> None:
        for i, cd in enumerate(old_clips):
            if "f0_original" in cd or "f0_target" in cd:
                tuning_id = cd.get("tuning_id", f"legacy_{i}")
                tuning = TuningData(
                    f0_original=_b64_to_arr(cd.get("f0_original", "")),
                    f0_target=_b64_to_arr(cd.get("f0_target", "")),
                    f0_manually_edited=cd.get("f0_manually_edited", False),
                    key=cd.get("key", "C"),
                    scale=cd.get("scale", "major"),
                    retune_speed=cd.get("retune_speed", 0.5),
                )
                self.tuning_map[tuning_id] = tuning
                if i < len(self.clips):
                    self.clips[i].tuning_id = tuning_id


def _arr_to_b64(arr: np.ndarray) -> str:
    return base64.b64encode(arr.astype(np.float32).tobytes()).decode()


def _b64_to_arr(s: str) -> np.ndarray:
    if not s:
        return np.array([], dtype=np.float32)
    return np.frombuffer(base64.b64decode(s), dtype=np.float32).copy()
