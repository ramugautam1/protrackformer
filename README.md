# ProTrackFormer

Tracking protein puncta that split and merge in 3D live microscopy.

Puncta such as Par-3/Bazooka clusters in the *Drosophila* embryo fuse and divide all the time, so their
lineages are directed acyclic graphs (DAGs), not the division trees that cell trackers produce.
ProTrackFormer tracks them through both kinds of events:

- every candidate link gets an independent probability (no softmax over parents, so merges are not suppressed)
- every object gets a predicted number of parents and children
- links that compete for the same object attend to each other
- an exact integer program decodes a consistent DAG and, given a foreground, also chooses which candidate
  segments are real objects, so touching puncta can be separated by their temporal context
- training on movies played backward turns every split into a merge

The repository also contains two merge-aware benchmarks (Bazooka and Gish, in an extended Cell Tracking
Challenge format) and the DAG metrics used to score them.

## Results

Bazooka benchmark, exact instances, crop-level 4-fold cross-validation (strict event F1 unless noted):

| method | merges | link | branch link | split | merge | split (tol.) | merge (tol.) | CHOTA-DAG |
|---|---|---|---|---|---|---|---|---|
| mask overlap | yes | 0.89 | 0.81 | 0.47 | 0.59 | 0.54 | 0.66 | 0.74 |
| LAP (laptrack) | yes | 0.85 | 0.74 | 0.45 | 0.49 | 0.51 | 0.58 | 0.69 |
| Ultrack | no | 0.87 | 0.65 | 0.46 | 0.00 | 0.56 | 0.00 | 0.71 |
| Trackastra (retrained) | no | 0.91 | 0.74 | 0.45 | 0.48 | 0.52 | 0.56 | 0.78 |
| **ProTrackFormer** | yes | **0.96** | **0.88** | **0.71** | **0.73** | **0.75** | **0.78** | **0.89** |

On the Gish benchmark, ProTrackFormer reaches split and merge F1 of 0.54 and 0.61 (0.55 and 0.63 when fine-tuned
from Bazooka), against at most 0.44 and 0.54 for the baselines.

## Install

```bash
git clone https://github.com/ramugautam1/protrackformer.git
cd protrackformer
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest -q
```

Python 3.10 or newer. A GPU speeds up training (about two minutes per model) but everything runs on a CPU.

## Track a movie

```bash
protrackformer-track \
    --image movie.tif \
    --masks segmentation.tif \
    --checkpoint checkpoints/gish_segmentation.pt \
    --voxel-size 0.5 0.132 0.132 --frame-interval 9.54 \
    --out results/movie
```

- `--image`: a T x Z x Y x X TIFF (add `--channel` for multi-channel movies), a NIfTI file, or a folder of `t###.tif`
- `--masks`: a segmentation of the same movie, as instance labels or a binary mask
- outputs in `--out`:
  - `tracks.txt`: one row per track, `id start end parent_1 ... parent_K`
  - `tracked_masks.tif`: every object labelled with its track id (T x Z x Y x X; opens in Fiji and napari)
  - `events.csv`: every split and merge with its frame, time, tracks and position in µm
  - `links.csv`: every link of the lineage
  - `summary.json`: counts and settings

Large movies are predicted in overlapping tiles and decoded in overlapping blocks automatically.

## Checkpoints

| file | input | trained on |
|---|---|---|
| `checkpoints/bazooka_instances.pt` | exact instance masks | 25 Bazooka crops |
| `checkpoints/bazooka_segmentation.pt` | segmenter output | 18 Bazooka crops (FC-DenseNet segmentation) |
| `checkpoints/gish_segmentation.pt` | segmenter output | Gish field, fine-tuned from Bazooka |

See [checkpoints/README.md](checkpoints/README.md).

## Benchmarks

- `data/bazooka`: 28 hand-annotated crops of Bazooka puncta, 40 frames each; 108 splits and 139 merges
- `data/gish`: one field of Gish puncta, 20 frames; 110 splits and 106 merges

Both use the CTC-DAG format: the Cell Tracking Challenge layout with a track table that keeps every parent of a
merge. See [data/README.md](data/README.md).

## Train and evaluate

Every command, from a single training run to the full cross-validation, is in [COMMANDS.md](COMMANDS.md).

```bash
protrackformer-train configs/bazooka_instances.yaml --fold 0 --out runs/bazooka_instances/fold0
protrackformer-evaluate --data data/bazooka --pred runs/bazooka_instances/fold*/test
```

## Repository layout

```
configs/                 training configurations (YAML)
data/                    Bazooka and Gish benchmarks (CTC-DAG)
checkpoints/             trained models
src/protrackformer/
    candidates.py        candidate segments of a foreground (components and watershed parts)
    graph.py             candidate graph: objects, links and their measurements
    model.py             the network
    windows.py           sliding windows and augmentation, including time reversal
    inference.py         window-averaged prediction and the Tracker class
    decode.py            integer program and block decoding
    lineage.py           lineage DAGs and track tables
    io.py                reading movies and masks, writing results
    data.py              benchmark sequences and folds
    metrics.py           DAG tracking metrics
    train.py             protrackformer-train
    track.py             protrackformer-track
    evaluate.py          protrackformer-evaluate
tests/
```

## Citation

If you use ProTrackFormer or its benchmarks, please cite the paper (see [CITATION.cff](CITATION.cff)).

## License

- code: MIT, see [LICENSE](LICENSE)
- benchmarks (`data/`): CC BY 4.0, see [data/LICENSE](data/LICENSE); please cite the paper when you use them
