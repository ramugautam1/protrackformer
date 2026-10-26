"""Candidate segments of a foreground mask.

- level 0: the given instances, or the connected components of the foreground
- finer levels: marker-controlled watershed of each component, with markers from h-maxima
  of the smoothed intensity at decreasing h (fractions of the frame's robust peak)
- identical segments across levels are kept once; overlapping candidates exclude each other
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi
from skimage.morphology import h_maxima
from skimage.segmentation import watershed

S26 = ndi.generate_binary_structure(3, 3)


def levels(img: np.ndarray, fg: np.ndarray, h_levels=(0.5, 0.3, 0.2, 0.1), sigma=(0.3, 1.0, 1.0),
           inst: np.ndarray | None = None) -> list[np.ndarray]:
    """Label images `[level 0, watershed at each h]` for one 3D frame.

    With `inst`, level 0 is those instances and finer levels never cross their borders.
    """
    if inst is not None:
        cc = np.unique(inst, return_inverse=True)[1].reshape(inst.shape).astype(np.int32)
        n = int(cc.max())
    else:
        cc, n = ndi.label(fg, S26)
    out = [cc]
    if n == 0:
        return out + [cc.copy() for _ in h_levels]
    sm = ndi.gaussian_filter(img.astype(np.float32), sigma)
    bg = float(np.median(sm[~fg])) if (~fg).any() else 0.0
    peak = float(np.percentile(sm[fg], 99)) - bg
    for h in h_levels:
        markers, _ = ndi.label(h_maxima(np.where(fg, sm - bg, 0).astype(np.float32), h * peak) & fg, S26)
        ws = watershed(-sm, markers, mask=fg)
        unmarked = fg & (ws == 0)
        if unmarked.any():
            ws[unmarked] = cc[unmarked] + ws.max()
        if inst is not None:
            pair = ws.astype(np.int64) * (n + 1) + cc
            ws = np.where(fg, np.unique(pair, return_inverse=True)[1].reshape(ws.shape), 0)
        out.append(ws.astype(np.int32))
    return out


def frame_layers(img: np.ndarray, fg: np.ndarray, regions, h_levels,
                 inst: np.ndarray | None = None) -> list[np.ndarray]:
    """Candidate levels of one frame, computed independently inside each region `(y0, x0, h, w)`."""
    out = [np.zeros(fg.shape, np.int32) for _ in range(1 + len(h_levels))]
    for y0, x0, h, w in regions:
        sl = (slice(None), slice(y0, y0 + h), slice(x0, x0 + w))
        for i, lab in enumerate(levels(img[sl], fg[sl], h_levels, inst=None if inst is None else inst[sl])):
            offset = out[i].max()
            out[i][sl][lab > 0] = lab[lab > 0] + offset
    return out


def unique_segments(layers: list[np.ndarray]):
    """Canonical `(layer, label)` of every distinct segment, how many layers contain it, and an alias map."""
    keys, canon, stability, alias = {}, [], [], {}
    for li, lab in enumerate(layers):
        flat = lab.ravel()
        order = np.argsort(flat, kind="stable")
        vals = flat[order]
        top = int(vals[-1]) if len(vals) else 0
        starts = np.searchsorted(vals, np.arange(1, top + 2))
        for label in range(1, top + 1):
            idx = order[starts[label - 1]:starts[label]]
            if len(idx) == 0:
                continue
            key = hash(np.sort(idx).tobytes())
            if key in keys:
                stability[keys[key]] += 1
                alias[(li, label)] = canon[keys[key]]
            else:
                keys[key] = len(canon)
                canon.append((li, label))
                stability.append(1)
                alias[(li, label)] = (li, label)
    return canon, stability, alias


def merge_boxes(boxes) -> list[tuple[int, int, int, int]]:
    """Bounding boxes of groups of overlapping or adjacent boxes."""
    boxes = [tuple(int(v) for v in b) for b in boxes]
    parent = list(range(len(boxes)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i, (y0, x0, h, w) in enumerate(boxes):
        for j in range(i + 1, len(boxes)):
            y1, x1, h1, w1 = boxes[j]
            if y0 <= y1 + h1 and y1 <= y0 + h and x0 <= x1 + w1 and x1 <= x0 + w:
                parent[find(i)] = find(j)
    groups = {}
    for i, b in enumerate(boxes):
        groups.setdefault(find(i), []).append(b)
    out = []
    for g in groups.values():
        ys, xs = min(b[0] for b in g), min(b[1] for b in g)
        out.append((ys, xs, max(b[0] + b[2] for b in g) - ys, max(b[1] + b[3] for b in g) - xs))
    return out
