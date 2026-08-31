# Pretrained Reference-Photo Person Search

Finds the person shown in one or more reference photos inside an overhead
video, then reports every matched frame and the person's image-plane
coordinates. This repository targets the two supplied FootfallCam snapshots;
both images are treated as examples of **one identity**.

```text
video → YOLO26n person detector → BoT-SORT tracker
      → dense bootstrap + sampled YOLO26 ReID embeddings
      → track-level gallery matching
      → accepted frames + retained boxes
      ├→ JSON / CSV coordinates, attribution, and candidate diagnostics
      └→ full-resolution PNGs with red wearer boxes
```

All models are pretrained. There is no task-specific dataset, training, or
fine-tuning.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --index-url https://download.pytorch.org/whl/cpu torch torchvision
pip install -r requirements.txt
```

The first run downloads `yolo26n.pt`, `yolo26n-cls.pt` (BoT-SORT's fallback
appearance encoder), and `yolo26n-reid.onnx`. Later runs reuse the local files.

## Run

```bash
python match_tag_wearer.py --video sample.mp4 \
  --reference reference/tag_wearer_snapshot_1.png \
  --reference reference/tag_wearer_snapshot_2.png \
  --config config/config.yaml \
  --output outputs/tag_wearer.json
```

Repeat `--reference` to add another view of the same person. Use `--label` to
change the output label from its default, `tag_wearer`. Accepted frames are
also saved as full-resolution PNGs with a red wearer box under
`outputs/tag_wearer_frames/`. By default the directory is the output JSON path
without its extension plus `_frames`; use `--annotated-dir` to override it.

## Outputs

- `outputs/tag_wearer.json`: one gallery target, accepted track IDs, track
  confidence, per-reference evidence, segments, per-frame coordinates and
  winning-reference attribution, runtime metadata, and the five strongest
  candidate tracks. `annotated_images_dir` and `annotation_image_count`
  summarize the generated images.
- `outputs/tag_wearer.csv`: the same matched frames in flat form. The
  semicolon-separated `reference_images` column records the gallery, while
  `matched_reference_image`, `matched_reference_index`, and
  `reference_similarity` identify the reference supporting each frame.
- `outputs/tag_wearer_frames/`: one annotated PNG per accepted frame. JSON and
  CSV rows link to their image through `annotation_image`. Files use stable
  names such as `frame_000506_track_0113.png`.

The image exporter decodes the video a second time after track acceptance; it
does not rerun detection and does not hold full video frames in memory. On a
rerun, only files matching `frame_*_track_*.png` in the selected annotation
directory are replaced. Other files in that directory are preserved.

Coordinates are the bottom-center of the detected person box in pixels plus
normalized `[0,1]` values. They are not physical coordinates because the
video contains no calibration or depth data.

## Tests

```bash
python -m unittest discover -v
```

The current implementation and measured sample result are documented in
[solution_overview.md](solution_overview.md) ([PDF](solution_overview.pdf)),
[system_architecture.md](system_architecture.md), and [result.md](result.md).
The full flow is shown in [diagrams/architecture.svg](diagrams/architecture.svg).
The test suite currently contains 20 tests, including synthetic red-box
rendering, boundary clipping, stale-file cleanup, and the frame-506 regression.

## Limitations

The supplied views have only `0.277` cosine consistency in the pretrained
ReID model, reflecting the severe pose, lighting, and fisheye changes. Gallery
matching therefore keeps every view independently and includes a conservative
peak-confirmation path for short tracks. Four observations are required so a
brief bottom-border passage such as snapshot 2 can qualify without accepting
a one-frame coincidence. This solves retrieval of this known
identity; it does not learn the general concept of a staff badge.
