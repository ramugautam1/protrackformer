from pathlib import Path

import numpy as np
import pytest

from protrackformer.lineage import Lineage

DATA = Path(__file__).resolve().parents[1] / "data"
CHECKPOINTS = Path(__file__).resolve().parents[1] / "checkpoints"

# track id -> (first frame, last frame, start (y, x), velocity (dy, dx))
TRACKS = {1: (0, 2, (10, 6), (0, 2)), 2: (0, 2, (10, 26), (0, -2)), 3: (3, 4, (10, 16), (0, 0)),
          4: (5, 7, (8, 13), (-1, -2)), 5: (5, 7, (12, 19), (1, 2)), 6: (0, 7, (24, 8), (0, 1))}
PARENTS = {3: [1, 2], 4: [3], 5: [3]}


def _sphere(shape, z, y, x, r):
    zz, yy, xx = np.ogrid[:shape[0], :shape[1], :shape[2]]
    return ((zz - z) * 2.0) ** 2 + (yy - y) ** 2 + (xx - x) ** 2 <= r * r


@pytest.fixture
def toy():
    """Eight frames: two puncta merge, the product splits in two, and a sixth punctum drifts alone."""
    T, shape = 8, (5, 32, 32)
    rng = np.random.default_rng(0)
    image = rng.normal(100, 3, (T, *shape)).astype(np.float32)
    labels = np.zeros((T, *shape), np.uint16)
    for tid, (t0, t1, (y, x), (dy, dx)) in TRACKS.items():
        for t in range(t0, t1 + 1):
            cy, cx = y + dy * (t - t0), x + dx * (t - t0)
            m = _sphere(shape, 2, cy, cx, 3.2 if tid == 3 else 2.5)
            labels[t][m & (labels[t] == 0)] = tid
            image[t] += 400 * np.exp(-(((np.indices(shape) - np.array([2, cy, cx])[:, None, None, None])
                                        * np.array([2.0, 1, 1])[:, None, None, None]) ** 2).sum(0) / 8)
    nodes = {(t, tid) for tid, (t0, t1, _, _) in TRACKS.items() for t in range(t0, t1 + 1)}
    edges = {((t, tid), (t + 1, tid)) for tid, (t0, t1, _, _) in TRACKS.items() for t in range(t0, t1)}
    edges |= {((TRACKS[p][1], p), (TRACKS[c][0], c)) for c, ps in PARENTS.items() for p in ps}
    return image.astype(np.uint16), labels, Lineage(nodes, edges)
