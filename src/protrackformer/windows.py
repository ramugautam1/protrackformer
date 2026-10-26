"""Sliding windows over a candidate graph.

- a window holds the objects of `W` consecutive frames and the links among them
- several sequences can share one window (training batches); their objects never attend to each other
- augmentation: in-plane flips and transposes, a global log-intensity jitter, and time reversal,
  which swaps sources and targets, in- and out-degrees, and so turns every split into a merge
"""

from __future__ import annotations

import numpy as np
import torch

from .graph import LF, Graph

SWAP = [("ovl_src_r2", "ovl_dst_r2"), ("mass_in_dst", "mass_out_src"), ("rank_out", "rank_in")]
NEGATE = ["dz", "dy", "dx", "log_vol_ratio", "log_int_ratio"]
SEPARATION = 10_000.0  # x offset between sequences that share a window


def window_arrays(g: Graph, t0: int, W: int) -> dict:
    nsel = np.nonzero((g.node_t >= t0) & (g.node_t < t0 + W))[0]
    remap = -np.ones(len(g.node_t), np.int64)
    remap[nsel] = np.arange(len(nsel))
    esel = np.nonzero((remap[g.edge_src] >= 0) & (remap[g.edge_dst] >= 0))[0]
    w = {"node_feat": g.node_feat[nsel].copy(), "node_pos": g.node_pos[nsel].copy(), "trel": g.node_t[nsel] - t0,
         "src": remap[g.edge_src[esel]], "dst": remap[g.edge_dst[esel]], "edge_feat": g.edge_feat[esel].copy(),
         "nsel": nsel, "esel": esel}
    if g.edge_y is not None:
        w.update(y=g.edge_y[esel].copy(), indeg=g.node_indeg[nsel].copy(), outdeg=g.node_outdeg[nsel].copy(),
                 exists=g.node_exists[nsel].copy())
    return w


def concat(windows: list[dict]) -> dict:
    out, offset = {}, 0
    for i, w in enumerate(windows):
        w = dict(w)
        w["node_pos"] = w["node_pos"] + np.array([0, 0, i * SEPARATION], np.float32)
        w["src"], w["dst"] = w["src"] + offset, w["dst"] + offset
        offset += len(w["trel"])
        for k, v in w.items():
            out.setdefault(k, []).append(v)
    return {k: np.concatenate(v) for k, v in out.items() if k not in ("nsel", "esel")}


def augment(w: dict, W: int, rng: np.random.Generator, time_reversal: bool = True) -> dict:
    pos, ef, nf = w["node_pos"], w["edge_feat"], w["node_feat"]
    if rng.random() < 0.5:
        pos[:, 1] = -pos[:, 1]
        ef[:, LF["dy"]] *= -1
    if rng.random() < 0.5:
        pos[:, 2] = -pos[:, 2]
        ef[:, LF["dx"]] *= -1
    if rng.random() < 0.5:
        pos[:, [1, 2]] = pos[:, [2, 1]]
        ef[:, [LF["dy"], LF["dx"]]] = ef[:, [LF["dx"], LF["dy"]]]
    if time_reversal and rng.random() < 0.5:
        w["trel"] = W - 1 - w["trel"]
        w["src"], w["dst"] = w["dst"], w["src"]
        for name in NEGATE:
            ef[:, LF[name]] *= -1
        for a, b in SWAP:
            ef[:, [LF[a], LF[b]]] = ef[:, [LF[b], LF[a]]]
        if "indeg" in w:
            w["indeg"], w["outdeg"] = w["outdeg"], w["indeg"]
        w["reversed"] = True
    nf[:, 1:4] += rng.normal(0, 0.05)
    return w


def to_tensors(w: dict, W: int, max_dt: int, device) -> dict:
    def t(a, dtype=torch.float32):
        return torch.as_tensor(a, dtype=dtype, device=device)

    out = {"node_feat": t(w["node_feat"]), "node_pos": t(w["node_pos"]), "node_trel": t(w["trel"], torch.long),
           "src": t(w["src"], torch.long), "dst": t(w["dst"], torch.long), "edge_feat": t(w["edge_feat"])}
    if "y" in w:
        in_mask, out_mask = w["trel"] >= max_dt, w["trel"] <= W - 1 - max_dt
        out.update(y=t(w["y"]), indeg=t(w["indeg"], torch.long), outdeg=t(w["outdeg"], torch.long),
                   exists=t(w["exists"]), in_mask=t(in_mask, torch.bool), out_mask=t(out_mask, torch.bool))
    return out


def inputs(s: dict) -> tuple:
    return s["node_feat"], s["node_pos"], s["node_trel"], s["src"], s["dst"], s["edge_feat"]
