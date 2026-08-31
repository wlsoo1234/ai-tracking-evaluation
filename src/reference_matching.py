"""Pretrained person-ReID matching against a gallery of reference photos.

The supplied reference photos are examples of one identity.  Each original
image and configured rotation is embedded independently; candidate people are
matched to the closest gallery embedding.  Track-level aggregation lives here
as pure helpers so it can be tested without loading model weights.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Protocol, Sequence

import cv2
import numpy as np

logger = logging.getLogger(__name__)


def normalize_embedding(value: Optional[np.ndarray]) -> Optional[np.ndarray]:
    """Return a flat L2-normalized float32 embedding, or ``None`` if invalid."""
    if value is None:
        return None
    embedding = np.asarray(value, dtype=np.float32).reshape(-1)
    norm = float(np.linalg.norm(embedding))
    if norm < 1e-12:
        return None
    return embedding / norm


def cosine_similarity(a: Optional[np.ndarray], b: Optional[np.ndarray]) -> float:
    """Cosine similarity for normalized or unnormalized embedding vectors."""
    a_norm = normalize_embedding(a)
    b_norm = normalize_embedding(b)
    if a_norm is None or b_norm is None:
        return 0.0
    return float(np.clip(np.dot(a_norm, b_norm), -1.0, 1.0))


class RegionEncoder(Protocol):
    """Small injectable interface used by the production and fake encoders."""

    def encode_regions(self, image: np.ndarray, bboxes_xyxy: Sequence[Sequence[float]]) -> List[Optional[np.ndarray]]:
        ...


class UltralyticsReIDEncoder:
    """Adapter around Ultralytics' shared tracker ReID encoder."""

    def __init__(self, model: str, device: str):
        # Import lazily so unit tests using a fake encoder need neither model
        # weights nor ONNX Runtime.
        from ultralytics.trackers.utils.reid import ReID

        self._encoder = ReID(model=model, device=device)

    def encode_regions(self, image: np.ndarray, bboxes_xyxy: Sequence[Sequence[float]]) -> List[Optional[np.ndarray]]:
        if not bboxes_xyxy:
            return []
        dets = []
        for x1, y1, x2, y2 in bboxes_xyxy:
            dets.append([(x1 + x2) / 2.0, (y1 + y2) / 2.0, max(0.0, x2 - x1), max(0.0, y2 - y1)])
        raw = self._encoder(image, np.asarray(dets, dtype=np.float32))
        return [normalize_embedding(feature) for feature in raw]


@dataclass(frozen=True)
class CandidateTrack:
    track_id: int
    confidence: float
    observation_count: int
    start_frame: int
    end_frame: int
    top_similarities: tuple[float, ...]


@dataclass(frozen=True)
class ReferenceMatch:
    """Reference-specific evidence for one sampled person crop."""

    similarity: float
    matched_reference_image: str
    matched_reference_index: int
    reference_similarities: tuple[float, ...]


class ReferenceMatcher:
    """Embed a reference gallery and score detected people against it."""

    def __init__(
        self,
        reference_paths: List[str],
        cfg: dict,
        device: str,
        encoder: Optional[RegionEncoder] = None,
    ):
        if not reference_paths:
            raise ValueError("At least one reference image is required")

        rm = cfg.get("reference_matching", {})
        self.reference_paths = list(reference_paths)
        self.rotations = list(rm.get("reference_rotations", [0, 90, 180, 270]))
        if 0 not in self.rotations:
            raise ValueError("reference_matching.reference_rotations must include 0")
        invalid_rotations = [angle for angle in self.rotations if angle not in {0, 90, 180, 270}]
        if invalid_rotations:
            raise ValueError(f"Unsupported reference rotations: {invalid_rotations}")

        self.encoder = encoder or UltralyticsReIDEncoder(rm.get("model", "yolo26n-reid.onnx"), device)
        self.original_embeddings: List[np.ndarray] = []
        self.gallery_embeddings: List[np.ndarray] = []
        self.reference_gallery_embeddings: List[List[np.ndarray]] = []

        for path in self.reference_paths:
            image = cv2.imread(path)
            if image is None:
                raise ValueError(f"Reference image could not be read: {path}")
            reference_embeddings = []
            for angle in self.rotations:
                rotated = _rotate_right_angle(image, angle)
                embedding = self._encode_full_image(rotated, path)
                reference_embeddings.append(embedding)
                self.gallery_embeddings.append(embedding)
                if angle == 0:
                    self.original_embeddings.append(embedding)
            self.reference_gallery_embeddings.append(reference_embeddings)

        consistency_floor = float(rm.get("reference_consistency_warning", 0.40))
        for i in range(len(self.original_embeddings)):
            for j in range(i + 1, len(self.original_embeddings)):
                similarity = cosine_similarity(self.original_embeddings[i], self.original_embeddings[j])
                if similarity < consistency_floor:
                    logger.warning(
                        "Reference images %s and %s have low ReID consistency (%.3f < %.3f); "
                        "keeping both as independent gallery exemplars",
                        self.reference_paths[i], self.reference_paths[j], similarity, consistency_floor,
                    )

    def _encode_full_image(self, image: np.ndarray, path: str) -> np.ndarray:
        h, w = image.shape[:2]
        encoded = self.encoder.encode_regions(image, [[0.0, 0.0, float(w), float(h)]])
        embedding = encoded[0] if encoded else None
        if embedding is None:
            raise ValueError(f"ReID encoder produced no embedding for reference image: {path}")
        return embedding

    def matches(self, frame: np.ndarray, bboxes_xyxy: Sequence[Sequence[float]]) -> List[ReferenceMatch]:
        """Return gallery and source-reference evidence for every supplied box."""
        embeddings = self.encoder.encode_regions(frame, bboxes_xyxy)
        matches = []
        for embedding in embeddings:
            if embedding is None:
                reference_scores = [0.0] * len(self.reference_paths)
            else:
                reference_scores = [
                    max(cosine_similarity(embedding, reference) for reference in reference_gallery)
                    for reference_gallery in self.reference_gallery_embeddings
                ]
            # max() keeps the first item on exact ties, making attribution to
            # the lowest reference index deterministic.
            reference_index = max(range(len(reference_scores)), key=reference_scores.__getitem__)
            matches.append(ReferenceMatch(
                similarity=float(reference_scores[reference_index]),
                matched_reference_image=self.reference_paths[reference_index],
                matched_reference_index=reference_index,
                reference_similarities=tuple(float(score) for score in reference_scores),
            ))
        return matches

    def similarities(self, frame: np.ndarray, bboxes_xyxy: Sequence[Sequence[float]]) -> List[float]:
        """Compatibility wrapper returning only maximum-gallery similarities."""
        return [match.similarity for match in self.matches(frame, bboxes_xyxy)]


def propagate_reference_attribution(
    frame_ids: Sequence[int],
    sampled_matches: Dict[int, ReferenceMatch],
) -> Dict[int, ReferenceMatch]:
    """Attribute every stored track frame using nearby sampled evidence.

    Attribution never crosses a discontinuity in a track when that contiguous
    segment has its own sample.  A segment with no samples falls back to the
    strongest observation for the complete track.  Equidistant samples prefer
    the earlier frame for deterministic output.
    """
    ordered_frames = sorted(set(frame_ids))
    if not ordered_frames or not sampled_matches:
        return {}

    fallback_frame, fallback_match = max(
        sampled_matches.items(),
        key=lambda item: (item[1].similarity, -item[0], -item[1].matched_reference_index),
    )
    del fallback_frame

    segments: List[List[int]] = []
    for frame_id in ordered_frames:
        if not segments or frame_id != segments[-1][-1] + 1:
            segments.append([frame_id])
        else:
            segments[-1].append(frame_id)

    attributed: Dict[int, ReferenceMatch] = {}
    for segment in segments:
        segment_samples = [frame_id for frame_id in segment if frame_id in sampled_matches]
        if not segment_samples:
            for frame_id in segment:
                attributed[frame_id] = fallback_match
            continue
        for frame_id in segment:
            nearest = min(segment_samples, key=lambda sample_frame: (abs(sample_frame - frame_id), sample_frame))
            attributed[frame_id] = sampled_matches[nearest]
    return attributed


def aggregate_track_candidates(
    observations: Dict[int, List[float]],
    frame_ranges: Dict[int, tuple[int, int]],
    min_observations: int,
    top_k: int,
) -> List[CandidateTrack]:
    """Aggregate sampled similarities into deterministic track candidates."""
    candidates = []
    for track_id, scores in observations.items():
        if len(scores) < min_observations:
            continue
        strongest = sorted((float(s) for s in scores), reverse=True)[:top_k]
        confidence = float(np.mean(strongest))
        start_frame, end_frame = frame_ranges[track_id]
        candidates.append(CandidateTrack(track_id, confidence, len(scores), start_frame, end_frame, tuple(strongest)))
    return sorted(candidates, key=lambda candidate: (-candidate.confidence, candidate.track_id))


def accepted_track_scores(
    candidates: Sequence[CandidateTrack],
    threshold: float,
    peak_threshold: Optional[float] = None,
) -> Dict[int, float]:
    """Return accepted track IDs mapped to their strongest qualifying score.

    The aggregate threshold is the normal path.  A separately configured,
    stricter peak threshold preserves short tracks that contain one
    near-reference view but are heavily distorted in their other views.
    Minimum-observation filtering has already happened during aggregation, so
    a single-frame detection cannot enter through this rescue path.
    """
    accepted = {}
    for candidate in candidates:
        peak = candidate.top_similarities[0]
        if candidate.confidence >= threshold:
            accepted[candidate.track_id] = candidate.confidence
        elif peak_threshold is not None and peak >= peak_threshold:
            accepted[candidate.track_id] = peak
    return accepted


def resolve_frame_track_ids(
    frame_track_ids: Dict[int, Sequence[int]],
    accepted_scores: Dict[int, float],
    sampled_scores: Optional[Dict[tuple[int, int], float]] = None,
) -> Dict[int, int]:
    """Choose at most one accepted identity track per frame.

    Aggregate track confidence wins first.  A sampled per-frame score breaks
    exact confidence ties, followed by the lower track ID for determinism.
    """
    sampled_scores = sampled_scores or {}
    resolved: Dict[int, int] = {}
    for frame_id, track_ids in frame_track_ids.items():
        eligible = [track_id for track_id in track_ids if track_id in accepted_scores]
        if not eligible:
            continue
        resolved[frame_id] = max(
            eligible,
            key=lambda track_id: (
                accepted_scores[track_id],
                sampled_scores.get((frame_id, track_id), -1.0),
                -track_id,
            ),
        )
    return resolved


def track_ids_to_sample(
    frame_id: int,
    track_ids: Sequence[int],
    observation_counts: Dict[int, int],
    sample_interval_frames: int,
    bootstrap_observations: int,
) -> List[int]:
    """Select tracks for dense bootstrap sampling, then periodic sampling.

    Short border-entry tracks may exist for fewer frames than a periodic
    sampler needs to collect enough evidence.  Their first observations are
    therefore embedded densely until the normal minimum-support count is met.
    Established tracks return to the configured global cadence.
    """
    periodic_frame = frame_id % sample_interval_frames == 0
    return [
        track_id
        for track_id in track_ids
        if observation_counts.get(track_id, 0) < bootstrap_observations or periodic_frame
    ]


def _rotate_right_angle(image: np.ndarray, angle: int) -> np.ndarray:
    if angle == 0:
        return image
    if angle == 90:
        return cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    if angle == 180:
        return cv2.rotate(image, cv2.ROTATE_180)
    if angle == 270:
        return cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
    raise ValueError(f"Unsupported rotation: {angle}")
