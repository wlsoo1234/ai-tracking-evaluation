from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from match_tag_wearer import build_segments, write_outputs


class CliHelperTests(unittest.TestCase):
    def test_segment_grouping(self):
        hits = [
            {"frame_id": 1, "timestamp_s": 0.04},
            {"frame_id": 2, "timestamp_s": 0.08},
            {"frame_id": 5, "timestamp_s": 0.20},
        ]
        segments = build_segments(hits)
        self.assertEqual([(s["start_frame"], s["end_frame"]) for s in segments], [(1, 2), (5, 5)])

    def test_json_csv_counts_and_gallery_column_match(self):
        with tempfile.TemporaryDirectory() as tempdir:
            path = Path(tempdir) / "result.json"
            output = {
                "tag_wearers": [{
                    "label": "tag_wearer",
                    "reference_images": ["one.png", "two.png"],
                    "frames": [{
                        "frame_id": 4, "timestamp_s": 0.16, "track_id": 9,
                        "x": 10.0, "y": 20.0, "x_normalized": 0.1,
                        "y_normalized": 0.2, "confidence": 0.75,
                        "matched_reference_image": "two.png",
                        "matched_reference_index": 1,
                        "reference_similarity": 0.81,
                        "annotation_image": "frames/frame_000004_track_0009.png",
                    }],
                }],
            }
            csv_path = write_outputs(str(path), output)
            loaded = json.loads(path.read_text())
            with open(csv_path, newline="") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(loaded["tag_wearers"][0]["frames"]), len(rows))
            self.assertEqual(rows[0]["reference_images"], "one.png;two.png")
            self.assertEqual(rows[0]["label"], "tag_wearer")
            self.assertEqual(rows[0]["matched_reference_image"], "two.png")
            self.assertEqual(rows[0]["matched_reference_index"], "1")
            self.assertEqual(rows[0]["reference_similarity"], "0.81")
            self.assertEqual(rows[0]["annotation_image"], "frames/frame_000004_track_0009.png")

    def test_canonical_output_attributes_snapshot_two_source_frame(self):
        root = Path(__file__).resolve().parents[1]
        json_path = root / "outputs" / "tag_wearer.json"
        csv_path = root / "outputs" / "tag_wearer.csv"
        output = json.loads(json_path.read_text())
        target = output["tag_wearers"][0]
        frame = next(hit for hit in target["frames"] if hit["frame_id"] == 506 and hit["track_id"] == 113)
        self.assertEqual(frame["matched_reference_image"], "reference/tag_wearer_snapshot_2.png")
        self.assertEqual(frame["matched_reference_index"], 1)

        evidence = target["reference_evidence"][1]
        self.assertEqual(evidence["reference_image"], "reference/tag_wearer_snapshot_2.png")
        self.assertEqual(evidence["status"], "detected")
        self.assertGreater(evidence["frame_count"], 0)
        self.assertTrue(any(segment["start_frame"] <= 506 <= segment["end_frame"] for segment in evidence["segments"]))

        with open(csv_path, newline="") as handle:
            rows = list(csv.DictReader(handle))
        self.assertEqual(len(rows), target["frame_count"])
        csv_frame = next(row for row in rows if row["frame_id"] == "506" and row["track_id"] == "113")
        self.assertEqual(csv_frame["matched_reference_image"], "reference/tag_wearer_snapshot_2.png")
        self.assertEqual(csv_frame["annotation_image"], frame["annotation_image"])

        self.assertEqual(target["annotation_image_count"], target["frame_count"])
        annotation_paths = []
        for hit in target["frames"]:
            path = Path(hit["annotation_image"])
            annotation_paths.append(path if path.is_absolute() else root / path)
        self.assertEqual(len(annotation_paths), len(set(annotation_paths)))
        self.assertTrue(all(path.is_file() for path in annotation_paths))
        generated_paths = set((root / target["annotated_images_dir"]).glob("frame_*_track_*.png"))
        self.assertEqual(generated_paths, set(annotation_paths))

        frame_path = Path(frame["annotation_image"])
        if not frame_path.is_absolute():
            frame_path = root / frame_path
        annotated_frame = cv2.imread(str(frame_path))
        self.assertEqual(annotated_frame.shape, (720, 960, 3))
        red_pixels = np.all(annotated_frame == np.array([0, 0, 255], dtype=np.uint8), axis=2)
        self.assertTrue(np.any(red_pixels))


if __name__ == "__main__":
    unittest.main()
