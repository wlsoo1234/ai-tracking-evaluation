from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from src.reference_matching import (
    ReferenceMatcher,
    ReferenceMatch,
    accepted_track_scores,
    aggregate_track_candidates,
    cosine_similarity,
    normalize_embedding,
    propagate_reference_attribution,
    resolve_frame_track_ids,
    track_ids_to_sample,
)


class MeanColorEncoder:
    """Deterministic fake whose embedding is the region's mean BGR color."""

    def encode_regions(self, image, bboxes_xyxy):
        outputs = []
        for x1, y1, x2, y2 in bboxes_xyxy:
            crop = image[int(y1):int(y2), int(x1):int(x2)]
            outputs.append(normalize_embedding(crop.reshape(-1, 3).mean(axis=0)) if crop.size else None)
        return outputs


def config(**overrides):
    values = {
        "model": "unused.onnx",
        "reference_rotations": [0, 90, 180, 270],
        "reference_consistency_warning": 0.40,
    }
    values.update(overrides)
    return {"reference_matching": values}


class GalleryMatcherTests(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.root = Path(self.tempdir.name)

    def tearDown(self):
        self.tempdir.cleanup()

    def image(self, name, bgr):
        path = self.root / name
        data = np.full((12, 8, 3), bgr, dtype=np.uint8)
        self.assertTrue(cv2.imwrite(str(path), data))
        return str(path)

    def test_normalization_and_cosine(self):
        np.testing.assert_allclose(normalize_embedding(np.array([3.0, 4.0])), [0.6, 0.8])
        self.assertAlmostEqual(cosine_similarity(np.array([2.0, 0.0]), np.array([5.0, 0.0])), 1.0)
        self.assertEqual(cosine_similarity(None, np.array([1.0])), 0.0)

    def test_gallery_contains_every_rotation_and_uses_best_reference(self):
        red = self.image("red.png", [0, 0, 255])
        green = self.image("green.png", [0, 255, 0])
        with self.assertLogs("src.reference_matching", level="WARNING"):
            matcher = ReferenceMatcher([red, green], config(), "cpu", encoder=MeanColorEncoder())
        self.assertEqual(len(matcher.original_embeddings), 2)
        self.assertEqual(len(matcher.gallery_embeddings), 8)
        self.assertEqual([len(group) for group in matcher.reference_gallery_embeddings], [4, 4])

        frame = np.full((20, 20, 3), [0, 255, 0], dtype=np.uint8)
        match = matcher.matches(frame, [[0, 0, 20, 20]])[0]
        self.assertAlmostEqual(match.similarity, 1.0, places=6)
        self.assertEqual(match.matched_reference_image, green)
        self.assertEqual(match.matched_reference_index, 1)
        self.assertAlmostEqual(match.reference_similarities[0], 0.0, places=6)
        self.assertAlmostEqual(match.reference_similarities[1], 1.0, places=6)
        self.assertEqual(matcher.similarities(frame, [[0, 0, 20, 20]]), [match.similarity])

    def test_reference_tie_uses_lowest_index(self):
        first = self.image("first.png", [255, 0, 0])
        second = self.image("second.png", [255, 0, 0])
        matcher = ReferenceMatcher([first, second], config(), "cpu", encoder=MeanColorEncoder())
        frame = np.full((20, 20, 3), [255, 0, 0], dtype=np.uint8)
        match = matcher.matches(frame, [[0, 0, 20, 20]])[0]
        self.assertEqual(match.matched_reference_index, 0)
        self.assertEqual(match.matched_reference_image, first)

    def test_low_reference_consistency_warns_but_keeps_gallery(self):
        red = self.image("red.png", [0, 0, 255])
        green = self.image("green.png", [0, 255, 0])
        with self.assertLogs("src.reference_matching", level="WARNING") as logs:
            matcher = ReferenceMatcher([red, green], config(), "cpu", encoder=MeanColorEncoder())
        self.assertIn("low ReID consistency", " ".join(logs.output))
        self.assertEqual(len(matcher.gallery_embeddings), 8)

    def test_invalid_reference_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "could not be read"):
            ReferenceMatcher([str(self.root / "missing.png")], config(), "cpu", encoder=MeanColorEncoder())

    def test_rotations_must_include_original(self):
        red = self.image("red.png", [0, 0, 255])
        with self.assertRaisesRegex(ValueError, "must include 0"):
            ReferenceMatcher([red], config(reference_rotations=[90]), "cpu", encoder=MeanColorEncoder())


class TrackAggregationTests(unittest.TestCase):
    def test_nearest_reference_attribution_and_segment_fallback(self):
        first = ReferenceMatch(0.7, "one.png", 0, (0.7, 0.2))
        second = ReferenceMatch(0.9, "two.png", 1, (0.3, 0.9))
        attributed = propagate_reference_attribution(
            [1, 2, 3, 4, 5, 10, 11],
            {1: first, 5: second},
        )
        self.assertEqual(attributed[2], first)
        self.assertEqual(attributed[3], first)  # equal distance prefers earlier sample
        self.assertEqual(attributed[4], second)
        self.assertEqual(attributed[10], second)  # unsampled segment uses strongest track evidence
        self.assertEqual(attributed[11], second)

    def test_short_tracks_are_sampled_densely_until_supported(self):
        selected = track_ids_to_sample(
            frame_id=503,
            track_ids=[1, 2, 3],
            observation_counts={1: 0, 2: 4, 3: 5},
            sample_interval_frames=5,
            bootstrap_observations=5,
        )
        self.assertEqual(selected, [1, 2])

        periodic = track_ids_to_sample(
            frame_id=505,
            track_ids=[1, 2, 3],
            observation_counts={1: 5, 2: 6, 3: 20},
            sample_interval_frames=5,
            bootstrap_observations=5,
        )
        self.assertEqual(periodic, [1, 2, 3])

    def test_short_tracks_rejected_and_top_k_mean_used(self):
        observations = {
            1: [0.99, 0.98, 0.97, 0.10],
            2: [0.9, 0.8, 0.7, 0.6, 0.1, 0.0],
        }
        candidates = aggregate_track_candidates(observations, {1: (0, 3), 2: (10, 15)}, 5, 3)
        self.assertEqual([candidate.track_id for candidate in candidates], [2])
        self.assertAlmostEqual(candidates[0].confidence, 0.8)
        self.assertEqual(candidates[0].observation_count, 6)

    def test_threshold_selection_can_return_not_found(self):
        candidates = aggregate_track_candidates({7: [0.5] * 5}, {7: (0, 20)}, 5, 10)
        self.assertEqual(accepted_track_scores(candidates, 0.6, 0.7), {})

    def test_strict_peak_can_rescue_supported_short_track(self):
        candidates = aggregate_track_candidates(
            {7: [0.73, 0.50, 0.45, 0.42, 0.40]}, {7: (0, 20)}, 5, 10
        )
        self.assertEqual(accepted_track_scores(candidates, 0.6, 0.7), {7: 0.73})

    def test_four_frame_border_track_can_qualify_on_aggregate(self):
        candidates = aggregate_track_candidates(
            {113: [0.621, 0.739, 0.585, 0.576]}, {113: (505, 508)}, 4, 10
        )
        accepted = accepted_track_scores(candidates, 0.6, 0.7)
        self.assertEqual(list(accepted), [113])
        self.assertAlmostEqual(accepted[113], 0.63025)

    def test_multiple_disjoint_tracks_and_unsampled_frames_are_retained(self):
        accepted = {3: 0.8, 8: 0.7}
        frames = {0: [3], 1: [3], 2: [3], 20: [8], 21: [8]}
        resolved = resolve_frame_track_ids(frames, accepted, {(0, 3): 0.9, (20, 8): 0.8})
        self.assertEqual(resolved, {0: 3, 1: 3, 2: 3, 20: 8, 21: 8})

    def test_overlap_uses_track_confidence_then_sampled_score(self):
        frames = {5: [10, 11], 6: [10, 11]}
        resolved = resolve_frame_track_ids(frames, {10: 0.8, 11: 0.7})
        self.assertEqual(resolved, {5: 10, 6: 10})

        tied = resolve_frame_track_ids({5: [10, 11]}, {10: 0.8, 11: 0.8}, {(5, 10): 0.7, (5, 11): 0.9})
        self.assertEqual(tied, {5: 11})


if __name__ == "__main__":
    unittest.main()
