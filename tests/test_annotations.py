from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from src.annotations import (
    AnnotationJob,
    annotation_filename,
    clip_bbox,
    render_annotated_frames,
)


class AnnotationTests(unittest.TestCase):
    def test_filename_and_box_clipping(self):
        self.assertEqual(annotation_filename(506, 113), "frame_000506_track_0113.png")
        self.assertEqual(clip_bbox([-2.4, 2.2, 15.2, 30.0], 32, 24), (0, 2, 16, 23))

    def test_renders_full_frame_and_cleans_only_generated_files(self):
        with tempfile.TemporaryDirectory() as tempdir:
            root = Path(tempdir)
            video_path = root / "source.avi"
            output_dir = root / "frames"
            output_dir.mkdir()
            stale = output_dir / "frame_999999_track_9999.png"
            unrelated = output_dir / "keep.png"
            self.assertTrue(cv2.imwrite(str(stale), np.zeros((2, 2, 3), dtype=np.uint8)))
            self.assertTrue(cv2.imwrite(str(unrelated), np.zeros((2, 2, 3), dtype=np.uint8)))

            writer = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*"MJPG"), 5.0, (32, 24))
            self.assertTrue(writer.isOpened())
            for value in (10, 20, 30):
                writer.write(np.full((24, 32, 3), value, dtype=np.uint8))
            writer.release()

            paths = render_annotated_frames(
                str(video_path),
                [AnnotationJob(1, 7, (-2.4, 2.2, 15.2, 30.0))],
                str(output_dir),
            )
            self.assertFalse(stale.exists())
            self.assertTrue(unrelated.exists())
            self.assertEqual(Path(paths[1]).name, "frame_000001_track_0007.png")
            rendered = cv2.imread(paths[1])
            self.assertEqual(rendered.shape, (24, 32, 3))
            np.testing.assert_array_equal(rendered[2, 0], [0, 0, 255])
            np.testing.assert_array_equal(rendered[23, 10], [0, 0, 255])

    def test_no_matches_creates_empty_generated_set(self):
        with tempfile.TemporaryDirectory() as tempdir:
            output_dir = Path(tempdir) / "frames"
            output_dir.mkdir()
            stale = output_dir / "frame_000001_track_0001.png"
            stale.write_bytes(b"stale")
            paths = render_annotated_frames("unused.mp4", [], str(output_dir))
            self.assertEqual(paths, {})
            self.assertFalse(stale.exists())
            self.assertEqual(list(output_dir.glob("frame_*_track_*.png")), [])


if __name__ == "__main__":
    unittest.main()
