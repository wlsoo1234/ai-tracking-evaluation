"""Video decoding: sequential frame_id, real timestamps from the source fps."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, Tuple

import cv2
import numpy as np


@dataclass
class VideoInfo:
    fps: float
    width: int
    height: int
    frame_count: int


class VideoReader:
    def __init__(self, path: str):
        self.path = path
        self.cap = cv2.VideoCapture(path)
        if not self.cap.isOpened():
            raise IOError(f"Could not open video: {path}")
        fps = self.cap.get(cv2.CAP_PROP_FPS)
        if not fps or fps <= 0:
            # Some containers don't report fps reliably; 25 is only a last-resort
            # fallback, never used for sample.mp4 itself (its fps reads correctly).
            fps = 25.0
        self.info = VideoInfo(
            fps=fps,
            width=int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            height=int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            frame_count=int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        )

    def frames(self) -> Iterator[Tuple[int, float, np.ndarray]]:
        frame_id = 0
        while True:
            ok, frame = self.cap.read()
            if not ok:
                break
            timestamp_s = frame_id / self.info.fps
            yield frame_id, timestamp_s, frame
            frame_id += 1

    def close(self):
        self.cap.release()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()
