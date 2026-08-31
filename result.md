# Results — Pretrained Gallery ReID

## Sample result

The final CPU run used `yolo26n.pt`, BoT-SORT with appearance enabled, and
`yolo26n-reid.onnx` against both supplied snapshots as one identity gallery.

| Reference evidence | Accepted track | Frames | Selection score |
|---|---:|---:|---:|
| Snapshot-2-like passage | 91 | 429–432 | 0.6338 aggregate / 0.7225 peak |
| Snapshot 2 exact passage | 113 | 505–508 | 0.6302 aggregate / 0.7387 peak |
| Snapshot 1 exact passage | 194 | 827–865, except 851 | 0.7319 peak |

The combined output is `matched`: **46 frames in four contiguous segments**.
Track IDs are runtime artifacts and are shown only to make this run auditable;
no ID or frame range is hardcoded in the implementation or tests.

Snapshot 2 is an exact crop from video frame 506. Independent verification
found its source rectangle at `[455, 537, 585, 720]`. YOLO detects the person
there and the detected crop scores 0.739 against snapshot 2. The person is at
the bottom border and BoT-SORT only emits track 113 for four frames, which is
why the earlier five-observation floor incorrectly removed it.

The final sampler embeds each new track densely until four observations have
been collected, then returns to the configured 0.20-second cadence. Similarity
thresholds remain unchanged at 0.60 aggregate and 0.70 strict peak.

## Candidate separation

| Candidate | Aggregate | Peak | Outcome |
|---|---:|---:|---|
| 91 | **0.6338** | 0.7225 | accepted, snapshot-2-like |
| 113 | **0.6302** | 0.7387 | accepted, exact snapshot 2 |
| 107 | 0.5376 | 0.5657 | rejected |
| 201 | 0.5271 | 0.5382 | rejected |
| 194 | 0.4939 | **0.7319** | accepted, exact snapshot 1 |

Lowering the aggregate threshold was unnecessary and would have admitted more
look-alikes. Snapshot 1 instead uses the stricter peak path because the true
short track contains one reference-like view surrounded by fisheye-distorted
views. Snapshot 2 qualifies through its four-frame aggregate.

## Performance and verification

The final annotation-enabled 1,341-frame CPU regression took 196.86 seconds
(6.81 processing fps) and produced 46 full-resolution PNGs using 50 MB. The
accepted identities, scores, and reference attribution stayed unchanged;
timestamps and runtime measurements are intentionally volatile metadata.

- 20 tests pass, including annotation rendering, dense bootstrap sampling, and the measured
  four-frame snapshot-2 score sequence.
- JSON and CSV both contain 46 matched rows, with at most one person per frame.
- Every matched frame records its winning source reference; frame 506 is
  explicitly attributed to `reference/tag_wearer_snapshot_2.png`.
- Every matched row links to a unique 960×720 annotated PNG containing a red
  wearer box; frame 506 was also inspected visually.
- The source tree compiles with `compileall` and installed dependencies pass
  `pip check`.
- The previous long-duration light-shirt/color-histogram false positive remains
  rejected.

## Limitations

This is query-by-example retrieval for a known individual. It does not detect
an arbitrary badge wearer without reference photos. The two references have
only 0.277 embedding similarity, so their gallery entries must remain separate.
The thresholds should be revalidated if camera geometry, compression, uniforms,
or the ReID model changes.
