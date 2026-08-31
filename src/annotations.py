"""Render accepted tag-wearer tracks as full-frame annotated images."""
from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Sequence

import cv2

from src.video_io import VideoReader


@dataclass(frozen=True)
class AnnotationJob:
    frame_id: int
    track_id: int
    bbox_xyxy: tuple[float, float, float, float]


def annotation_filename(frame_id: int, track_id: int) -> str:
    """Return the stable filename used for one accepted frame."""
    if frame_id < 0 or track_id < 0:
        raise ValueError("frame_id and track_id must be non-negative")
    return f"frame_{frame_id:06d}_track_{track_id:04d}.png"


def clip_bbox(
    bbox_xyxy: Sequence[float],
    width: int,
    height: int,
) -> tuple[int, int, int, int]:
    """Round an XYXY box outward and clip it to valid image coordinates."""
    if len(bbox_xyxy) != 4:
        raise ValueError("bbox_xyxy must contain exactly four coordinates")
    if width <= 0 or height <= 0:
        raise ValueError("image dimensions must be positive")
    x1, y1, x2, y2 = (float(value) for value in bbox_xyxy)
    left = max(0, min(width - 1, math.floor(x1)))
    top = max(0, min(height - 1, math.floor(y1)))
    right = max(0, min(width - 1, math.ceil(x2)))
    bottom = max(0, min(height - 1, math.ceil(y2)))
    if right <= left or bottom <= top:
        raise ValueError(f"bbox has no visible area after clipping: {bbox_xyxy}")
    return left, top, right, bottom


def clean_generated_images(output_dir: str) -> None:
    """Remove only files produced by this renderer from its output directory."""
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    for path in directory.glob("frame_*_track_*.png"):
        if path.is_file():
            path.unlink()


def render_annotated_frames(
    video_path: str,
    jobs: Sequence[AnnotationJob],
    output_dir: str,
) -> Dict[int, str]:
    """Decode selected frames, draw red boxes, and return paths by frame ID."""
    clean_generated_images(output_dir)
    if not jobs:
        return {}

    jobs_by_frame: Dict[int, AnnotationJob] = {}
    for job in jobs:
        if job.frame_id in jobs_by_frame:
            raise ValueError(f"Multiple annotation jobs for frame {job.frame_id}")
        jobs_by_frame[job.frame_id] = job

    output_paths: Dict[int, str] = {}
    last_frame = max(jobs_by_frame)
    reader = VideoReader(video_path)
    try:
        for frame_id, _, frame in reader.frames():
            if frame_id > last_frame:
                break
            job = jobs_by_frame.get(frame_id)
            if job is None:
                continue
            height, width = frame.shape[:2]
            left, top, right, bottom = clip_bbox(job.bbox_xyxy, width, height)
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), thickness=3, lineType=cv2.LINE_8)
            path = Path(output_dir) / annotation_filename(job.frame_id, job.track_id)
            if not cv2.imwrite(str(path), frame):
                raise IOError(f"Could not write annotated image: {path}")
            output_paths[frame_id] = path.as_posix()
    finally:
        reader.close()

    missing = sorted(set(jobs_by_frame) - set(output_paths))
    if missing:
        raise IOError(f"Could not decode requested video frames: {missing}")
    return output_paths
