# Benchmarks

Two tracking benchmarks of protein puncta that split and merge, in the CTC-DAG format.

Licensed under CC BY 4.0 ([LICENSE](LICENSE)): free to use and adapt with credit to the paper.

| | Bazooka | Gish |
|---|---|---|
| protein | Par-3/Bazooka-GFP, *Drosophila* embryo | Gilgamesh-GFP, *Drosophila* embryo |
| sequences | 28 crops of 32 x 32 px | 1 field of 256 x 256 px |
| frames, slices | 40, 13 | 20, 13 |
| voxel (z, y, x) | 0.5, 0.124, 0.124 µm | 0.5, 0.132, 0.132 µm |
| frame interval | 7.5 s | 9.54 s |
| objects, links | 3,031, 2,755 | 2,635, 2,035 |
| splits (3-way, 4-way) | 108 (7, 1) | 110 (13, 0) |
| merges (3 parents) | 139 (9) | 106 (9) |
| ground truth | drawn by hand | ProTrack3D output, corrected event by event and reviewed |

## Format

CTC-DAG keeps the layout of the Cell Tracking Challenge (CTC) and extends its track table to several parents.

```
bazooka/
    meta.json                     voxel size, frame interval, totals
    folds.json                    4-fold cross-validation: train / val / test crops per fold
    Crop01/t000.tif ... t039.tif  images, one Z x Y x X uint16 TIFF per frame
    Crop01_GT/TRA/
        man_track000.tif ...      masks, labelled by track id
        man_track_multi.txt       track table
        lineage_edges.csv         every link
    Crop02/ ...
gish/
    meta.json                     also defines the views Q0-Q3 (quadrants) used as folds
    folds.json
    01/t000.tif ... t019.tif
    01_GT/TRA/...
```

### Track table

`man_track_multi.txt` has no header and one row per track:

```
id start end parent_1 parent_2 ... parent_K
```

- frames are numbered from 0, as in the CTC
- a track covers one object per frame over contiguous frames; a new track starts at every split, merge, birth
  and gap, so no track passes through an event
- `K` is the largest number of parents in the sequence; unused columns are 0
- a birth has no parent, both children of a split list the same parent, and the product of a merge lists all
  of its parents
- a gap (a punctum missing for one frame) is a track that ends and a new track whose parent is the old one

Examples from `Crop07`:

```
3 9 9 1 2 0        track 3 exists in frame 9 only and is the merge of tracks 1 and 2
22 19 21 15 18 19  track 22 exists in frames 19-21 and is the merge of three tracks
```

The first four columns are the CTC's `man_track.txt` for every track with at most one parent.

### Links

`lineage_edges.csv` lists every link with the track ids of its two objects, so the DAG can be read without
rebuilding it from the table:

```
t_src,id_src,t_dst,id_dst
0,1,1,1
```

## Cross-validation

- Bazooka: crop-level 4-fold cross-validation; every fold tests 7 crops, selects on 3 and trains on 18, and every
  crop is tested once. Intervals resample crops.
- Gish: the field is cut into four 128 x 128 quadrants (views `Q0` to `Q3` in `meta.json`). An object belongs to the
  quadrant that holds its centroid and keeps its whole mask; links between quadrants are cut. Fold `k` tests on
  quadrant `k`, selects on quadrant `k + 1` and trains on the other two. Intervals resample 64 x 64 tiles.

## Loading

```python
from protrackformer.data import load_sequence

seq = load_sequence("data/bazooka", "Crop07")      # or load_sequence("data/gish", "Q0")
seq.image, seq.labels                               # T x Z x Y x X
seq.lineage.events()                                # splits and merges
```
