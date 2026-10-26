"""Benchmark sequences and cross-validation folds.

- a benchmark folder holds CTC-DAG sequences, `meta.json` (voxel size, frame interval, views) and
  `folds.json` (train / val / test names per fold)
- a view is a box of a sequence, e.g. a quadrant of the Gish field; an object belongs to the view that
  holds its centroid and keeps its whole mask, and links leaving the view are cut
- statistics resample units: whole sequences (Bazooka crops) or square tiles of a view (Gish)
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import numpy as np

from .io import read_sequence, read_stack, to_instances
from .lineage import Lineage
from .metrics import centroids


@dataclass
class Sequence:
    name: str
    image: np.ndarray  # T x Z x Y x X
    labels: np.ndarray  # ground truth, labelled by track id
    lineage: Lineage
    unit_tile: int | None = None  # resample tiles of this size instead of the whole sequence
    origin: tuple = (0, 0)  # (y, x) of the view's core inside the stored image
    meta: dict = field(default_factory=dict)

    def unit_of(self, node, c) -> str:
        if not self.unit_tile:
            return self.name
        a, b = int((c[1] - self.origin[0]) // self.unit_tile), int((c[2] - self.origin[1]) // self.unit_tile)
        return f"{self.name}_{a}{b}"


def load_meta(root: Path) -> dict:
    return json.loads((Path(root) / "meta.json").read_text())


def load_folds(root: Path) -> dict:
    return json.loads((Path(root) / "folds.json").read_text())


@lru_cache(maxsize=8)
def _read(seq_dir: str):
    return read_sequence(Path(seq_dir))


def load_sequence(root: Path, name: str) -> Sequence:
    root = Path(root)
    meta = load_meta(root)
    view = meta.get("views", {}).get(name)
    if view is None:
        image, labels, lineage = _read(str(root / name))
        return Sequence(name, image, labels, lineage, meta=meta)
    image, labels, lineage = _read(str(root / view["sequence"]))
    img, lab, keep, origin = cut_view(image, labels, view["box"], view.get("margin", 0))
    nodes = {n for n in lineage.nodes if n in keep}
    sub = Lineage(nodes, {(u, v) for u, v in lineage.edges if u in nodes and v in nodes})
    return Sequence(name, img, lab, sub, unit_tile=view.get("unit_tile"), origin=origin, meta=meta)


def cut_view(image: np.ndarray, labels: np.ndarray, box, margin: int = 0):
    """Objects whose centroid lies in `box = (y0, x0, h, w)`, with `margin` pixels of image around the box."""
    y0, x0, h, w = box
    H, W = labels.shape[-2:]
    ys, xs = slice(max(0, y0 - margin), min(H, y0 + h + margin)), slice(max(0, x0 - margin), min(W, x0 + w + margin))
    keep = {n for n, c in centroids(labels).items() if y0 <= c[1] < y0 + h and x0 <= c[2] < x0 + w}
    lab = np.zeros(labels.shape[:2] + (ys.stop - ys.start, xs.stop - xs.start), labels.dtype)
    for t in range(len(labels)):
        ids = [i for (tt, i) in keep if tt == t]
        sub = labels[t][:, ys, xs]
        lab[t] = np.where(np.isin(sub, ids), sub, 0)
    return image[:, :, ys, xs], lab, keep, (y0 - ys.start, x0 - xs.start)


def cut_masks(masks: np.ndarray, root: Path, name: str) -> np.ndarray:
    """Apply a view of the benchmark to a segmentation of its full sequence (same ownership rule)."""
    view = load_meta(root).get("views", {}).get(name)
    if view is None:
        return masks
    return cut_view(masks, masks, view["box"], view.get("margin", 0))[1]


def instance_masks(labels: np.ndarray) -> np.ndarray:
    """Ground-truth instances relabelled 1..n in every frame, so labels carry no track identity."""
    out = np.zeros(labels.shape, np.int32)
    for t in range(len(labels)):
        ids = np.unique(labels[t])
        out[t] = np.searchsorted(ids, labels[t])
    return out


def segmentation_masks(path: Path, root: Path, name: str) -> np.ndarray:
    """A segmentation of a benchmark sequence: `<path>/<sequence>.tif` or `<path>/<sequence>/` per-frame files."""
    path = Path(path)
    meta = load_meta(root)
    base = meta.get("views", {}).get(name, {}).get("sequence", name)
    src = path / f"{base}.tif" if (path / f"{base}.tif").exists() else path / base
    return cut_masks(to_instances(read_stack(src)), root, name)

