# Changelog

## Unreleased

- `protrackformer-track`: binary masks whose foreground is stored as 255 or 65535 (ProTrack3D, ImageJ) are now
  split into connected components like 0/1 masks. Before, any mask with values above 1 was read as a label image,
  so such a mask became a single object per frame and tracking returned almost nothing.

## 1.0.0

- ProTrackFormer: link-to-link attention, independent link probabilities, degree heads and exact decoding
  over candidate objects
- Bazooka (28 crops) and Gish (one field) benchmarks in the CTC-DAG format, with cross-validation folds (CC BY 4.0)
- `protrackformer-train`, `protrackformer-track`, `protrackformer-evaluate`
- checkpoints for Bazooka (exact instances, segmenter output) and Gish (segmenter output)
