"""Person detection + multi-object tracking, via a single ultralytics YOLO
model run in `.track()` mode (ByteTrack by default, BoT-SORT+ReID selectable
in config as a production upgrade — both ship inside ultralytics, so no extra
tracking dependency is needed).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np
import torch
from ultralytics import YOLO

logger = logging.getLogger(__name__)

# tracker.config_file is what actually reaches ultralytics (it's the filename
# passed to model.track(tracker=...)) — tracker.type is otherwise just a
# label. This map is only used (a) to fill in config_file when it's omitted,
# and (b) to sanity-check that an explicitly-set config_file agrees with the
# stated type, so a config that says type: botsort but still points at a
# bytetrack config fails loudly instead of silently doing nothing.
_DEFAULT_CONFIG_FILE = {
    "bytetrack": "bytetrack.yaml",
    "botsort": "botsort.yaml",
}


@dataclass
class TrackedPerson:
    track_id: int
    bbox: List[float]        # [x1, y1, x2, y2], pixel coords, original frame
    confidence: float        # this frame's detection confidence
    track_age: int           # frames since this track_id was first seen


def resolve_device(requested: str) -> str:
    if requested == "auto":
        return "cuda:0" if torch.cuda.is_available() else "cpu"
    return requested


def resolve_tracker_config(tcfg: dict) -> str:
    tracker_type = tcfg.get("type", "bytetrack")
    config_file = tcfg.get("config_file")
    default_file = _DEFAULT_CONFIG_FILE.get(tracker_type)

    if not config_file:
        if default_file is None:
            raise ValueError(f"tracker.config_file must be set explicitly for unknown tracker.type={tracker_type!r}")
        return default_file

    if default_file is not None and tracker_type not in config_file.lower():
        logger.warning(
            "tracker.type=%r does not match tracker.config_file=%r — config_file is what "
            "actually selects the tracker backend; type is otherwise just documentation. "
            "Update both together or this setting won't do what it says.",
            tracker_type, config_file,
        )
    return config_file


class PersonTracker:
    def __init__(self, cfg: dict):
        pcfg = cfg["person_detector"]
        tcfg = cfg["tracker"]
        self.device = resolve_device(pcfg["device"])
        self.model = YOLO(pcfg["model"])
        self.conf = pcfg["conf_threshold"]
        self.imgsz = pcfg["imgsz"]
        self.classes = pcfg["classes"]
        self.tracker_cfg = resolve_tracker_config(tcfg)
        self._first_seen: Dict[int, int] = {}

    def update(self, frame: np.ndarray, frame_id: int) -> List[TrackedPerson]:
        results = self.model.track(
            frame,
            persist=True,
            tracker=self.tracker_cfg,
            conf=self.conf,
            imgsz=self.imgsz,
            classes=self.classes,
            device=self.device,
            verbose=False,
        )[0]

        out: List[TrackedPerson] = []
        boxes = results.boxes
        if boxes is None or boxes.id is None:
            return out

        ids = boxes.id.int().tolist()
        xyxy = boxes.xyxy.tolist()
        confs = boxes.conf.tolist()
        for track_id, bbox, conf in zip(ids, xyxy, confs):
            if track_id not in self._first_seen:
                self._first_seen[track_id] = frame_id
            track_age = frame_id - self._first_seen[track_id]
            out.append(TrackedPerson(track_id=track_id, bbox=bbox, confidence=conf, track_age=track_age))
        return out
