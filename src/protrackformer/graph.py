"""Candidate graph over a sequence.

- nodes: candidate objects, either the given instances or the candidate segments of a foreground
- links: t -> t + dt for dt <= 2; masks that overlap after an in-plane dilation (dt = 1), plus the
  nearest candidates within a radius (anisotropic distance)
- object measurements (11): volume, intensities, extents, shape, contact with the first or last slice,
  distance to the border; candidate segments add 4 (stability, whole component, nesting counts)
- link measurements (23): displacement, gap, overlaps at several dilations, IoU, ratios, mass balance, ranks
- training labels come from the ground truth by one-to-one IoU matching
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
from scipy import ndimage as ndi
from scipy.optimize import linear_sum_assignment
from scipy.spatial import cKDTree

from .candidates import frame_layers, merge_boxes, unique_segments
from .lineage import Lineage

MAX_DEG = 3  # degree classes 0, 1, 2, 3+
LABEL_BASE = 100000  # node label = layer * LABEL_BASE + label within that layer

OBJECT_FEATURES = ["log_vol", "log_int_sum", "log_int_mean", "log_int_max", "z_extent", "xy_extent",
                   "log_ev1", "log_ev2", "log_ev3", "touch_z", "border_dist"]
CANDIDATE_FEATURES = ["stability", "is_component", "n_inside", "n_around"]
LINK_FEATURES = (["dz", "dy", "dx", "dist", "dt2"] + [f"ovl_min_r{r}" for r in range(5)] +
                 ["iou", "ovl_src_r2", "ovl_dst_r2", "log_vol_ratio", "log_int_ratio",
                  "mass_in_dst", "mass_out_src", "rank_out", "rank_in"])
LF = {n: i for i, n in enumerate(LINK_FEATURES)}


@dataclass
class GraphConfig:
    max_dt: int = 2
    k_nn: int = 5
    k_nn_gap: int = 3
    radius: float = 15.0
    overlap_dilate: int = 4
    overlap_levels: tuple = (0, 1, 2, 3, 4)
    z_scale: float = 0.5 / 0.124  # z step in units of the xy pixel
    h_levels: tuple = (0.5, 0.3, 0.2, 0.1)
    instance_iou: float = 0.3  # matching to ground truth, given instances
    candidate_iou: float = 0.5  # existence targets of candidate segments

    @classmethod
    def from_dict(cls, d: dict | None) -> GraphConfig:
        d = dict(d or {})
        for k in ("overlap_levels", "h_levels"):
            if k in d:
                d[k] = tuple(d[k])
        return cls(**d)


@dataclass
class Graph:
    node_t: np.ndarray
    node_label: np.ndarray
    node_pos: np.ndarray  # (N, 3): z (scaled), y, x in xy pixels
    node_feat: np.ndarray
    edge_src: np.ndarray
    edge_dst: np.ndarray
    edge_feat: np.ndarray
    n_frames: int
    candidates: bool = False
    conflicts: np.ndarray = None  # (C, 2) candidates that overlap in a frame
    node_cc: np.ndarray = None  # foreground component of each candidate
    edge_y: np.ndarray = None
    node_indeg: np.ndarray = None
    node_outdeg: np.ndarray = None
    node_exists: np.ndarray = None
    layers: list = field(default=None, repr=False)  # per frame, the candidate label images
    stats: dict = field(default_factory=dict)

    def mask(self, i: int) -> np.ndarray:
        t, lab = int(self.node_t[i]), int(self.node_label[i])
        return self.layers[t][lab // LABEL_BASE] == lab % LABEL_BASE


ARRAYS = ("node_t", "node_label", "node_pos", "node_feat", "edge_src", "edge_dst", "edge_feat", "conflicts", "node_cc",
          "edge_y", "node_indeg", "node_outdeg", "node_exists")


def save_graph(g: Graph, path: Path) -> None:
    arrays = {k: getattr(g, k) for k in ARRAYS if getattr(g, k) is not None}
    layers = np.array([np.stack(lt) for lt in g.layers])
    layers = layers.astype(np.uint16 if layers.max() < 65535 else np.uint32)
    np.savez_compressed(path, **arrays, layers=layers, n_frames=g.n_frames, candidates=g.candidates,
                        stats=np.array(g.stats, dtype=object))


def load_graph(path: Path) -> Graph:
    z = np.load(path, allow_pickle=True)
    g = Graph(**{k: z[k] for k in ARRAYS if k in z}, n_frames=int(z["n_frames"]), candidates=bool(z["candidates"]),
              stats=z["stats"].item())
    g.layers = [[layer.astype(np.int64) for layer in lt] for lt in z["layers"]]
    return g


def dilate_xy(labels: np.ndarray, r: int) -> np.ndarray:
    """Grow labels in-plane by `r` pixels without overwriting other labels."""
    if r <= 0:
        return labels
    yy, xx = np.mgrid[-r:r + 1, -r:r + 1]
    fp = (yy ** 2 + xx ** 2 <= r * r)[None]
    return np.where(labels > 0, labels, ndi.grey_dilation(labels, footprint=fp))


def pair_overlaps(a: np.ndarray, b: np.ndarray) -> dict[tuple[int, int], int]:
    a, b = a.ravel().astype(np.int64), b.ravel().astype(np.int64)
    fg = (a > 0) & (b > 0)
    if not fg.any():
        return {}
    base = int(b.max()) + 1
    keys, counts = np.unique(a[fg] * base + b[fg], return_counts=True)
    return {(int(k // base), int(k % base)): int(c) for k, c in zip(keys, counts)}


def object_features(img: np.ndarray, coords, bg: float, border: float, z_scale: float) -> tuple[np.ndarray, list]:
    """Centroid and the 11 object measurements of one object given its voxel coordinates."""
    zz, yy, xx = coords
    vals = img[zz, yy, xx].astype(np.float64) / bg
    c = np.array([zz.mean() * z_scale, yy.mean(), xx.mean()])
    pts = np.stack([zz * z_scale, yy, xx], 1) - c
    ev = np.sort(np.linalg.eigvalsh(pts.T @ pts / max(1, len(pts)) + 1e-3 * np.eye(3)))[::-1]
    nz = img.shape[0]
    feats = [np.log(len(zz)), np.log(vals.sum() + 1e-6), np.log(vals.mean() + 1e-6), np.log(vals.max() + 1e-6),
             zz.max() - zz.min() + 1, max(yy.max() - yy.min() + 1, xx.max() - xx.min() + 1),
             *np.log(ev + 1e-3), float(zz.min() == 0 or zz.max() + 1 == nz), min(border, 16.0) / 16.0]
    return c, feats


def label_voxels(labels: np.ndarray) -> dict[int, np.ndarray]:
    """label -> sorted flat indices of its voxels."""
    flat = labels.ravel()
    order = np.argsort(flat, kind="stable")
    vals = flat[order]
    top = int(vals[-1]) if len(vals) else 0
    starts = np.searchsorted(vals, np.arange(1, top + 2))
    return {lab: order[starts[lab - 1]:starts[lab]] for lab in range(1, top + 1) if starts[lab] > starts[lab - 1]}


def _border_fn(boxes, shape):
    if not boxes:
        h, w = shape[-2:]
        return lambda y, x: min(y, x, h - 1 - y, w - 1 - x)

    def border(y, x):
        ds = [min(y - y0, x - x0, y0 + h - 1 - y, x0 + w - 1 - x) for y0, x0, h, w in boxes
              if y0 <= y < y0 + h and x0 <= x < x0 + w]
        return min(ds) if ds else 0.0
    return border


def build_graph(image: np.ndarray, masks: np.ndarray, cfg: GraphConfig | None = None, candidates: bool = False,
                instances: bool = True, boxes=None) -> Graph:
    """Candidate graph of a T x Z x Y x X sequence.

    - `candidates=False`: every labelled object in `masks` is an object
    - `candidates=True`: candidate segments of the foreground; level 0 is the labels of `masks`
      (`instances=True`) or the 26-connected components of `masks > 0`
    - `boxes`: independent fields `(y0, x0, h, w)` such as crops; candidates never cross them
    """
    cfg = cfg or GraphConfig()
    T = len(masks)
    border = _border_fn(boxes, masks.shape)
    regions = merge_boxes(boxes) if boxes else [(0, 0, *masks.shape[-2:])]
    node_t, node_label, pos, feat, extra, node_cc, conflicts, layers = [], [], [], [], [], [], [], []
    alias_t, cc_base = [], 0
    for t in range(T):
        img = image[t].astype(np.float32)
        fg = masks[t] > 0
        bg = float(np.median(img[~fg])) if (~fg).any() else 1.0
        bg = bg or 1.0
        if candidates:
            lay = frame_layers(img, fg, regions, cfg.h_levels, inst=masks[t] if instances else None)
            canon, stability, alias = unique_segments(lay)
        else:
            lay = [masks[t].astype(np.int64)]
            labs = [int(v) for v in np.unique(lay[0]) if v]
            canon, stability, alias = [(0, v) for v in labs], [1] * len(labs), {(0, v): (0, v) for v in labs}
        layers.append(lay)
        alias_t.append(alias)
        start = len(node_t)
        voxels = [label_voxels(layer) for layer in lay]
        idx = {(li, lab): voxels[li][lab] for li, lab in canon}
        cc0 = lay[0].ravel()
        for (li, lab), stab in zip(canon, stability):
            vox = idx[(li, lab)]
            coords = np.unravel_index(vox, fg.shape)
            c, f = object_features(img, coords, bg, 0.0, cfg.z_scale)
            f[-1] = min(border(c[1], c[2]), 16.0) / 16.0
            node_t.append(t)
            node_label.append(li * LABEL_BASE + lab)
            pos.append(c)
            feat.append(f)
            ccs = cc0[vox]
            node_cc.append(cc_base + int(ccs[ccs > 0][0]))
            extra.append([stab / len(lay), 0.0, 0.0, 0.0])
        if candidates:
            n = len(node_t) - start
            keys = list(canon)
            for i in range(n):
                vox = idx[keys[i]]
                comp = voxels[0][int(cc0[vox[0]])]
                extra[start + i][1] = float(len(comp) == len(vox) and np.array_equal(comp, vox))
            for i in range(n):
                for j in range(i + 1, n):
                    if node_cc[start + i] != node_cc[start + j]:
                        continue
                    a, b = idx[keys[i]], idx[keys[j]]
                    ov = len(np.intersect1d(a, b, assume_unique=True))
                    if ov:
                        conflicts.append((start + i, start + j))
                        if ov == len(a):
                            extra[start + i][3] += 1
                            extra[start + j][2] += 1
                        elif ov == len(b):
                            extra[start + j][3] += 1
                            extra[start + i][2] += 1
        cc_base += int(lay[0].max()) + 1
    node_t = np.array(node_t, np.int64)
    node_label = np.array(node_label, np.int64)
    node_pos = np.array(pos, np.float32).reshape(-1, 3)
    node_feat = np.array(feat, np.float32).reshape(-1, len(OBJECT_FEATURES))
    if candidates:
        node_feat = np.concatenate([node_feat, np.array(extra, np.float32).reshape(-1, 4)], 1)
    g = Graph(node_t, node_label, node_pos, node_feat, *_links(node_t, node_label, node_pos, node_feat, layers,
                                                                alias_t, cfg, 2 if candidates else 1),
              n_frames=T, candidates=candidates, node_cc=np.array(node_cc, np.int64),
              conflicts=np.array(conflicts, np.int64).reshape(-1, 2), layers=layers)
    return g


def _links(node_t, node_label, node_pos, node_feat, layers, alias_t, cfg: GraphConfig, k_factor: int):
    T = len(layers)
    vol, ints = np.exp(node_feat[:, 0]), np.exp(node_feat[:, 1])
    idx = {(int(t), int(lab)): i for i, (t, lab) in enumerate(zip(node_t, node_label))}
    by_t = [np.nonzero(node_t == t)[0] for t in range(T)]
    src, dst, ef = [], [], []
    for t in range(T):
        for dt in range(1, cfg.max_dt + 1):
            if t + dt >= T or len(by_t[t]) == 0 or len(by_t[t + dt]) == 0:
                continue
            la, lb = layers[t], layers[t + dt]
            ovl = {}
            for r in cfg.overlap_levels:
                da, db = [dilate_xy(x, r) for x in la], [dilate_xy(x, r) for x in lb]
                for i in range(len(la)):
                    for j in range(len(lb)):
                        for (a, b), c in pair_overlaps(da[i], db[j]).items():
                            ca, cb = alias_t[t][(i, a)], alias_t[t + dt][(j, b)]
                            u = idx[(t, ca[0] * LABEL_BASE + ca[1])]
                            v = idx[(t + dt, cb[0] * LABEL_BASE + cb[1])]
                            ovl[(r, u, v)] = max(ovl.get((r, u, v), 0), c)
            cand = {(u, v) for (r, u, v) in ovl if r == cfg.overlap_dilate} if dt == 1 else set()
            k = min((cfg.k_nn if dt == 1 else cfg.k_nn_gap) * k_factor, len(by_t[t + dt]))
            dd, ii = cKDTree(node_pos[by_t[t + dt]]).query(node_pos[by_t[t]], k=k)
            dd, ii = dd.reshape(len(by_t[t]), k), ii.reshape(len(by_t[t]), k)
            for a_i in range(len(by_t[t])):
                for d_, j in zip(dd[a_i], ii[a_i]):
                    if d_ <= cfg.radius:
                        cand.add((int(by_t[t][a_i]), int(by_t[t + dt][j])))
            for u, v in sorted(cand):
                d = node_pos[v] - node_pos[u]
                fr = [ovl.get((r, u, v), 0) / min(vol[u], vol[v]) for r in cfg.overlap_levels]
                n0, n2 = ovl.get((0, u, v), 0), ovl.get((2, u, v), 0)
                src.append(u)
                dst.append(v)
                ef.append([*d, np.linalg.norm(d), float(dt == 2), *fr, n0 / (vol[u] + vol[v] - n0), n2 / vol[u],
                           n2 / vol[v], np.log(vol[v] / vol[u]), np.log(ints[v] / ints[u]), 0, 0, 0, 0])
    src, dst = np.array(src, np.int64), np.array(dst, np.int64)
    ef = np.array(ef, np.float32).reshape(-1, len(LINK_FEATURES))
    ov = ef[:, LF["ovl_min_r2"]] > 0
    for v in np.unique(dst):
        es = np.nonzero(dst == v)[0]
        sel = es[ov[es]]
        ef[es, LF["mass_in_dst"]] = np.log((vol[src[sel]].sum() + 1e-3) / vol[v]) if len(sel) else -5
        ef[es[np.argsort(ef[es, LF["dist"]])], LF["rank_in"]] = np.arange(len(es)) / 4.0
    for u in np.unique(src):
        es = np.nonzero(src == u)[0]
        sel = es[ov[es]]
        ef[es, LF["mass_out_src"]] = np.log((vol[dst[sel]].sum() + 1e-3) / vol[u]) if len(sel) else -5
        ef[es[np.argsort(ef[es, LF["dist"]])], LF["rank_out"]] = np.arange(len(es)) / 4.0
    return src, dst, ef


def label_graph(g: Graph, lineage: Lineage, gt: np.ndarray, cfg: GraphConfig | None = None) -> None:
    """Training targets from ground truth (`gt`: T x Z x Y x X, labels = node labels of `lineage`)."""
    cfg = cfg or GraphConfig()
    thr = cfg.candidate_iou if g.candidates else cfg.instance_iou
    to_gt = {}
    for t in range(g.n_frames):
        nodes = np.nonzero(g.node_t == t)[0]
        gm = gt[t]
        glabs = [int(v) for v in np.unique(gm) if v]
        if not len(nodes) or not glabs:
            continue
        gsize = np.bincount(gm.ravel())
        gi = {lab: i for i, lab in enumerate(glabs)}
        iou = np.zeros((len(glabs), len(nodes)))
        for j, n in enumerate(nodes):
            m = g.mask(n)
            labs, cnt = np.unique(gm[m], return_counts=True)
            for lab, c in zip(labs, cnt):
                if lab:
                    iou[gi[int(lab)], j] = c / (gsize[lab] + m.sum() - c)
        rows, cols = linear_sum_assignment(-iou)
        for a, b in zip(rows, cols):
            if iou[a, b] >= thr:
                to_gt[int(nodes[b])] = (t, glabs[a])
    s, p = lineage.succ(), lineage.pred()
    n = len(g.node_t)
    g.node_exists = np.zeros(n, np.float32)
    g.node_exists[list(to_gt)] = 1
    g.edge_y = np.array([float(u in to_gt and v in to_gt and (to_gt[u], to_gt[v]) in lineage.edges)
                         for u, v in zip(g.edge_src, g.edge_dst)], np.float32)
    g.node_indeg = np.array([min(len(p.get(to_gt[i], [])), MAX_DEG) if i in to_gt else 0 for i in range(n)], np.int64)
    g.node_outdeg = np.array([min(len(s.get(to_gt[i], [])), MAX_DEG) if i in to_gt else 0 for i in range(n)], np.int64)
    back = {v: k for k, v in to_gt.items()}
    reachable = [(back[a], back[b]) for a, b in lineage.edges if a in back and b in back]
    cand = set(zip(g.edge_src.tolist(), g.edge_dst.tolist()))
    g.stats = {"objects": n, "gt_objects": len(lineage.nodes), "gt_objects_matched": len(to_gt),
               "links": len(g.edge_src), "gt_links": len(lineage.edges),
               "link_recall": sum(e in cand for e in reachable) / max(1, len(reachable))}


def subgraph(g: Graph, nsel: np.ndarray, t0: int = 0, n_frames: int | None = None) -> tuple[Graph, np.ndarray]:
    """Nodes `nsel` with the links among them (times shifted by `t0`). Returns the subgraph and the link indices."""
    remap = -np.ones(len(g.node_t), np.int64)
    remap[nsel] = np.arange(len(nsel))
    esel = np.nonzero((remap[g.edge_src] >= 0) & (remap[g.edge_dst] >= 0))[0]
    cf = g.conflicts
    if cf is not None and len(cf):
        cf = remap[cf[(remap[cf[:, 0]] >= 0) & (remap[cf[:, 1]] >= 0)]]
    sub = Graph(g.node_t[nsel] - t0, g.node_label[nsel], g.node_pos[nsel], g.node_feat[nsel],
                remap[g.edge_src[esel]], remap[g.edge_dst[esel]], g.edge_feat[esel],
                n_frames=n_frames if n_frames is not None else g.n_frames, candidates=g.candidates, conflicts=cf,
                node_cc=None if g.node_cc is None else g.node_cc[nsel])
    for k in ("edge_y",):
        if getattr(g, k) is not None:
            setattr(sub, k, getattr(g, k)[esel])
    for k in ("node_indeg", "node_outdeg", "node_exists"):
        if getattr(g, k) is not None:
            setattr(sub, k, getattr(g, k)[nsel])
    return sub, esel


def config_dict(cfg: GraphConfig) -> dict:
    return {k: list(v) if isinstance(v, tuple) else v for k, v in asdict(cfg).items()}
