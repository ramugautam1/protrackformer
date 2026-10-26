# Commands

All commands run from the repository root after `pip install -e ".[dev]"`.

## Setup and tests

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"            # add ".[nifti]" to read NIfTI movies
pytest -q                          # unit tests and a small end-to-end run, about 15 s on a CPU
pytest -q -m "not slow"            # unit tests only
```

## Tracking (inference)

Gish movie with a segmentation from any segmenter:

```bash
protrackformer-track --image movie.tif --masks segmentation.tif \
    --checkpoint checkpoints/gish_segmentation.pt \
    --voxel-size 0.5 0.132 0.132 --frame-interval 9.54 --out results/movie
```

Bazooka movie, multi-channel file, frames 0 to 39 only:

```bash
protrackformer-track --image movie.tif --channel 0 --frames 0:39 --masks segmentation.tif \
    --checkpoint checkpoints/bazooka_segmentation.pt \
    --voxel-size 0.5 0.124 0.124 --frame-interval 7.5 --out results/movie
```

Exact instance masks (every label is one punctum):

```bash
protrackformer-track --image movie.tif --masks instances.tif \
    --checkpoint checkpoints/bazooka_instances.pt --out results/movie
```

Options:

- `--decode exact` solves one integer program for the whole movie; `--decode blocks` always uses overlapping
  blocks (default `auto`: blocks for fields wider than 160 px or longer than 30 frames)
- `--tile 0` predicts the whole field at once instead of in 96 px tiles
- `--workers N` sets the number of parallel block solves
- `--device cpu` forces the CPU

## Training

One fold of the Bazooka cross-validation, exact instances:

```bash
protrackformer-train configs/bazooka_instances.yaml --fold 0 --seed 0 --out runs/bazooka_instances/s0/fold0
```

Each run writes `model.pt` (the checkpoint used by `protrackformer-track`), `result.json` (training curve, chosen
decoder) and, for folds, `test/<sequence>/` results with `test/metrics.txt`.

Override any setting from the command line:

```bash
protrackformer-train configs/bazooka_instances.yaml --fold 0 --out runs/test --set optim.steps=500 seed=3
```

### Full cross-validation (paper)

Bazooka, exact instances, 4 folds x 3 seeds:

```bash
for s in 0 1 2; do for k in 0 1 2 3; do
    protrackformer-train configs/bazooka_instances.yaml --fold $k --seed $s --out runs/bazooka_instances/s$s/fold$k
done; done
for s in 0 1 2; do
    protrackformer-evaluate --data data/bazooka --pred runs/bazooka_instances/s$s/fold*/test \
        --json runs/bazooka_instances/s$s/pooled.json
done
```

Bazooka, annotated foreground (touching puncta fused; candidate selection):

```bash
for k in 0 1 2 3; do
    protrackformer-train configs/bazooka_foreground.yaml --fold $k --out runs/bazooka_foreground/fold$k
done
protrackformer-evaluate --data data/bazooka --pred runs/bazooka_foreground/fold*/test
```

Gish, exact instances (folds are quadrants of the field):

```bash
for k in 0 1 2 3; do
    protrackformer-train configs/gish_instances.yaml --fold $k --out runs/gish_instances/fold$k
done
protrackformer-evaluate --data data/gish --pred runs/gish_instances/fold*/test
```

Gish, fine-tuned from a Bazooka model of the same fold:

```bash
for k in 0 1 2 3; do
    protrackformer-train configs/gish_instances.yaml --fold $k --out runs/gish_from_bazooka/fold$k \
        --set init=runs/bazooka_instances/s0/fold$k/model.pt
done
```

### Segmenter output

Place one segmentation per sequence under `segmentations/` (labels or binary masks; binary masks are split into
6-connected components of at least 6 voxels):

```
segmentations/bazooka/Crop01.tif ... Crop28.tif     T x Z x Y x X, or Crop01/mask000.tif ...
segmentations/gish/01.tif                           the whole field; folds cut it into quadrants
```

```bash
for k in 0 1 2 3; do
    protrackformer-train configs/bazooka_segmentation.yaml --fold $k --out runs/bazooka_segmentation/fold$k
done
```

### Released checkpoint

```bash
protrackformer-train configs/release_bazooka_instances.yaml --seed 0 --out runs/release
cp runs/release/model.pt checkpoints/bazooka_instances.pt
```

## Evaluation

Score any folder of per-sequence results (`<folder>/<sequence>/tracks.txt` and `tracked_masks.tif`):

```bash
protrackformer-evaluate --data data/bazooka --pred runs/bazooka_instances/s0/fold*/test --json scores.json
```

Intervals are 95% cluster-bootstrap intervals over crops (Bazooka) or 64 px tiles (Gish).

## Python

```python
from protrackformer.inference import Tracker
from protrackformer.io import read_stack, to_instances, write_result

image = read_stack("movie.tif")                    # T x Z x Y x X
masks = to_instances(read_stack("segmentation.tif"))
tracker = Tracker("checkpoints/gish_segmentation.pt")
lineage, labels, graph = tracker.track(image, masks)
write_result("results/movie", lineage, labels, voxel_um=(0.5, 0.132, 0.132), frame_s=9.54)
for event in lineage.events():
    print(event.kind, event.t, event.anchor, event.partners)
```
