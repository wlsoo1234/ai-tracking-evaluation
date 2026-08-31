#!/usr/bin/env python3
"""Find a known person in video from a gallery of reference photos.

Repeated ``--reference`` arguments are exemplars of the same identity.  The
pipeline uses pretrained person detection, multi-object tracking, and ReID;
it performs no task-specific training.
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import platform
import random
import resource
import sys
import time
from datetime import datetime, timezone
from typing import Dict, List

import numpy as np
import torch
import ultralytics
import yaml

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src import coordinates
from src.annotations import AnnotationJob, render_annotated_frames
from src.reference_matching import (
    ReferenceMatcher,
    ReferenceMatch,
    accepted_track_scores,
    aggregate_track_candidates,
    propagate_reference_attribution,
    resolve_frame_track_ids,
    track_ids_to_sample,
)
from src.tracker import PersonTracker, resolve_device
from src.video_io import VideoReader


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def build_segments(hits: List[dict]) -> List[dict]:
    """Group sorted per-frame hits into contiguous runs."""
    segments = []
    start = None
    previous = None
    for hit in hits:
        if start is None:
            start = hit
        elif hit["frame_id"] != previous["frame_id"] + 1:
            segments.append(_segment(start, previous))
            start = hit
        previous = hit
    if start is not None:
        segments.append(_segment(start, previous))
    return segments


def _segment(start: dict, end: dict) -> dict:
    return {
        "start_frame": start["frame_id"],
        "end_frame": end["frame_id"],
        "start_time_s": start["timestamp_s"],
        "end_time_s": end["timestamp_s"],
        "duration_s": round(end["timestamp_s"] - start["timestamp_s"], 3),
    }


def _set_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def run_search(video_path: str, references: List[str], label: str, cfg: dict, annotated_dir: str) -> dict:
    """Run detection, tracking, sampled ReID, and track-level selection."""
    _set_seeds(int(cfg["runtime"]["seed"]))
    device = resolve_device(cfg["person_detector"]["device"])
    matcher = ReferenceMatcher(references, cfg, device=device)
    reader = VideoReader(video_path)
    tracker = PersonTracker(cfg)
    fps, width, height = reader.info.fps, reader.info.width, reader.info.height

    rm_cfg = cfg["reference_matching"]
    sample_interval = max(1, round(float(rm_cfg["sample_interval_s"]) * fps))
    min_observations = int(rm_cfg["min_observations"])
    top_k = int(rm_cfg["top_k"])
    threshold = float(rm_cfg["track_similarity_threshold"])
    peak_threshold = float(rm_cfg["peak_similarity_threshold"])

    # Stored track rows are lightweight and let accepted tracks be reported on
    # every detection frame even though expensive embeddings are sampled.
    track_rows: Dict[int, Dict[int, dict]] = {}
    frame_track_ids: Dict[int, List[int]] = {}
    observations: Dict[int, List[float]] = {}
    sampled_scores: Dict[tuple[int, int], float] = {}
    sampled_matches: Dict[tuple[int, int], ReferenceMatch] = {}
    started = time.perf_counter()

    for frame_id, timestamp_s, frame in reader.frames():
        tracks = tracker.update(frame, frame_id)
        frame_track_ids[frame_id] = [track.track_id for track in tracks]
        for track in tracks:
            x, y = coordinates.bottom_center(track.bbox)
            xn, yn = coordinates.normalize(x, y, width, height)
            track_rows.setdefault(track.track_id, {})[frame_id] = {
                "frame_id": frame_id,
                "timestamp_s": round(timestamp_s, 3),
                "track_id": track.track_id,
                "x": round(x, 2),
                "y": round(y, 2),
                "x_normalized": round(xn, 4),
                "y_normalized": round(yn, 4),
                "bbox_xyxy": tuple(float(value) for value in track.bbox),
            }

        sample_ids = set(track_ids_to_sample(
            frame_id,
            [track.track_id for track in tracks],
            {track_id: len(scores) for track_id, scores in observations.items()},
            sample_interval,
            min_observations,
        ))
        sampled_tracks = [track for track in tracks if track.track_id in sample_ids]
        if sampled_tracks:
            matches = matcher.matches(frame, [track.bbox for track in sampled_tracks])
            for track, match in zip(sampled_tracks, matches):
                observations.setdefault(track.track_id, []).append(match.similarity)
                sampled_scores[(frame_id, track.track_id)] = match.similarity
                sampled_matches[(frame_id, track.track_id)] = match

    reader.close()
    frame_ranges = {
        track_id: (min(rows), max(rows))
        for track_id, rows in track_rows.items()
    }
    candidates = aggregate_track_candidates(observations, frame_ranges, min_observations, top_k)
    accepted_scores = accepted_track_scores(candidates, threshold, peak_threshold)
    resolved = resolve_frame_track_ids(frame_track_ids, accepted_scores, sampled_scores)

    track_attribution: Dict[int, Dict[int, ReferenceMatch]] = {}
    for track_id, rows in track_rows.items():
        track_samples = {
            frame_id: match
            for (frame_id, sampled_track_id), match in sampled_matches.items()
            if sampled_track_id == track_id
        }
        track_attribution[track_id] = propagate_reference_attribution(rows.keys(), track_samples)

    frames = []
    annotation_jobs = []
    for frame_id, track_id in sorted(resolved.items()):
        row = dict(track_rows[track_id][frame_id])
        bbox_xyxy = row.pop("bbox_xyxy")
        row["confidence"] = round(accepted_scores[track_id], 4)
        attribution = track_attribution[track_id][frame_id]
        row["matched_reference_image"] = attribution.matched_reference_image
        row["matched_reference_index"] = attribution.matched_reference_index
        row["reference_similarity"] = round(attribution.similarity, 4)
        frames.append(row)
        annotation_jobs.append(AnnotationJob(frame_id, track_id, bbox_xyxy))

    annotation_paths = render_annotated_frames(video_path, annotation_jobs, annotated_dir)
    for row in frames:
        row["annotation_image"] = annotation_paths[row["frame_id"]]

    processing_seconds = time.perf_counter() - started

    segments = build_segments(frames)
    status = "matched" if frames else "not_found"
    accepted_ids = sorted({row["track_id"] for row in frames})
    accepted_confidence = {str(track_id): round(accepted_scores[track_id], 4) for track_id in accepted_ids}
    accepted_frame_keys = {
        (row["frame_id"], row["track_id"], row["matched_reference_index"])
        for row in frames
    }
    candidate_diagnostics = []
    for candidate in candidates[:5]:
        track_samples = sorted(
            (frame_id, match)
            for (frame_id, track_id), match in sampled_matches.items()
            if track_id == candidate.track_id
        )
        reference_diagnostics = []
        for reference_index, reference_image in enumerate(references):
            reference_scores = [match.reference_similarities[reference_index] for _, match in track_samples]
            attributed_samples = [
                frame_id for frame_id, match in track_samples
                if match.matched_reference_index == reference_index
            ]
            detected = any(
                track_id == candidate.track_id and matched_index == reference_index
                for _, track_id, matched_index in accepted_frame_keys
            )
            reference_diagnostics.append({
                "reference_image": reference_image,
                "reference_index": reference_index,
                "status": "detected" if detected else "not_detected",
                "best_similarity": round(max(reference_scores), 4) if reference_scores else None,
                "observation_count": len(reference_scores),
                "attributed_observation_count": len(attributed_samples),
                "attributed_start_frame": min(attributed_samples) if attributed_samples else None,
                "attributed_end_frame": max(attributed_samples) if attributed_samples else None,
            })
        candidate_diagnostics.append({
            "track_id": candidate.track_id,
            "track_confidence": round(candidate.confidence, 4),
            "observation_count": candidate.observation_count,
            "start_frame": candidate.start_frame,
            "end_frame": candidate.end_frame,
            "start_time_s": round(candidate.start_frame / fps, 3),
            "end_time_s": round(candidate.end_frame / fps, 3),
            "peak_similarity": round(candidate.top_similarities[0], 4),
            "top_similarities": [round(value, 4) for value in candidate.top_similarities],
            "reference_diagnostics": reference_diagnostics,
        })

    reference_evidence = []
    for reference_index, reference_image in enumerate(references):
        reference_frames = [
            row for row in frames
            if row["matched_reference_index"] == reference_index
        ]
        all_reference_scores = [
            match.reference_similarities[reference_index]
            for match in sampled_matches.values()
        ]
        reference_evidence.append({
            "reference_image": reference_image,
            "reference_index": reference_index,
            "status": "detected" if reference_frames else "not_detected",
            "supporting_track_ids": sorted({row["track_id"] for row in reference_frames}),
            "best_similarity": round(max(all_reference_scores), 4) if all_reference_scores else None,
            "frame_count": len(reference_frames),
            "segments": build_segments(reference_frames),
        })

    peak_memory_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
    return {
        "video": os.path.basename(video_path),
        "coordinate_system": "pixel_bottom_center",
        "method": "pretrained YOLO person detection + BoT-SORT + gallery cosine similarity using pretrained ReID embeddings; no task-specific training",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "reference_matching": {
            "model": rm_cfg["model"],
            "sample_interval_s": rm_cfg["sample_interval_s"],
            "track_similarity_threshold": threshold,
            "peak_similarity_threshold": peak_threshold,
            "min_observations": min_observations,
            "top_k": top_k,
        },
        "runtime": {
            "device": device,
            "processing_seconds": round(processing_seconds, 3),
            "processing_fps": round(len(frame_track_ids) / max(processing_seconds, 1e-9), 3),
            "peak_memory_mb": round(peak_memory_mb, 2),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "ultralytics": ultralytics.__version__,
        },
        "candidate_tracks": candidate_diagnostics,
        "tag_wearers": [{
            "label": label,
            "reference_images": list(references),
            "reference_evidence": reference_evidence,
            "annotated_images_dir": annotated_dir,
            "annotation_image_count": len(annotation_paths),
            "status": status,
            "accepted_track_ids": accepted_ids,
            "track_confidence": accepted_confidence,
            "frame_count": len(frames),
            "total_duration_s": round(sum(segment["duration_s"] for segment in segments), 3),
            "segments": segments,
            "frames": frames,
        }],
    }


def write_outputs(output_path: str, output: dict) -> str:
    out_dir = os.path.dirname(output_path) or "."
    os.makedirs(out_dir, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    csv_path = os.path.splitext(output_path)[0] + ".csv"
    target = output["tag_wearers"][0]
    references = ";".join(target["reference_images"])
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "label", "reference_images", "frame_id", "timestamp_s", "track_id",
            "x", "y", "x_normalized", "y_normalized", "confidence",
            "matched_reference_image", "matched_reference_index", "reference_similarity",
            "annotation_image",
        ])
        for hit in target["frames"]:
            writer.writerow([
                target["label"], references, hit["frame_id"], hit["timestamp_s"], hit["track_id"],
                hit["x"], hit["y"], hit["x_normalized"], hit["y_normalized"], hit["confidence"],
                hit["matched_reference_image"], hit["matched_reference_index"], hit["reference_similarity"],
                hit["annotation_image"],
            ])
    return csv_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--video", required=True, help="Path to input video")
    parser.add_argument(
        "--reference", action="append", required=True, dest="references",
        help="Reference photo of the same target identity; repeat to build the gallery.",
    )
    parser.add_argument("--label", default="tag_wearer", help="Identity label written to JSON and CSV")
    parser.add_argument("--config", default="config/config.yaml", help="Path to YAML config")
    parser.add_argument("--output", default="outputs/tag_wearer.json", help="Output JSON path")
    parser.add_argument(
        "--annotated-dir", default=None,
        help="Annotated PNG directory (default: <output JSON stem>_frames)",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    logging.basicConfig(
        level=getattr(logging, cfg["runtime"]["log_level"]),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    logger = logging.getLogger("match_tag_wearer")
    annotated_dir = args.annotated_dir or os.path.splitext(args.output)[0] + "_frames"
    output = run_search(args.video, args.references, args.label, cfg, annotated_dir)
    csv_path = write_outputs(args.output, output)
    target = output["tag_wearers"][0]
    logger.info(
        "%s: status=%s, tracks=%s, frames=%d, segments=%d, processing=%.2fs",
        target["label"], target["status"], target["accepted_track_ids"],
        target["frame_count"], len(target["segments"]), output["runtime"]["processing_seconds"],
    )
    logger.info("Wrote %s and %s", args.output, csv_path)


if __name__ == "__main__":
    main()
