"""Applying a trained model to a sequence.

- `predict`: probabilities averaged over all sliding windows that contain an object or link
- degrees count only from windows in which they are observable (not at the window's open ends)
- `predict_tiled`: the same over overlapping spatial tiles, for fields too large for one pass
- `Tracker`: checkpoint + graph + prediction + decoding, from arrays to a lineage
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from .decode import decode, to_lineage
from .graph import (CANDIDATE_FEATURES, LINK_FEATURES, MAX_DEG, OBJECT_FEATURES, Graph, GraphConfig, build_graph,
                    subgraph)
from .lineage import Lineage
from .model import ProTrackFormer
from .windows import inputs, to_tensors, window_arrays


@dataclass
class Prediction:
    link: np.ndarray  # (E,)
    p_in: np.ndarray  # (N, MAX_DEG + 1)
    p_out: np.ndarray
    exists: np.ndarray | None  # (N,), candidate segments only


@torch.no_grad()
def predict(model: ProTrackFormer, g: Graph, max_dt: int = 2, device="cpu") -> Prediction:
    model.eval()
    W = model.cfg["window"]
    E, N, K = len(g.edge_src), len(g.node_t), MAX_DEG + 1
    pe, ne = np.zeros(E), np.zeros(E)
    pex, nex = np.zeros(N), np.zeros(N)
    pin, pout, nin, nout = np.zeros((N, K)), np.zeros((N, K)), np.zeros(N), np.zeros(N)
    for t0 in range(max(1, g.n_frames - W + 1)):
        w = window_arrays(g, t0, W)
        if not len(w["nsel"]):
            continue
        link, lin, lout, lex = model(*inputs(to_tensors(w, W, max_dt, device)))
        nsel, esel = w["nsel"], w["esel"]
        if len(esel):
            pe[esel] += torch.sigmoid(link).cpu().numpy()
            ne[esel] += 1
        if lex is not None:
            pex[nsel] += torch.sigmoid(lex).cpu().numpy()
            nex[nsel] += 1
        trel = w["trel"]
        mi = (trel >= max_dt) | (t0 == 0)
        mo = (trel <= W - 1 - max_dt) | (t0 + W >= g.n_frames)
        pin[nsel[mi]] += lin.softmax(-1).cpu().numpy()[mi]
        nin[nsel[mi]] += 1
        pout[nsel[mo]] += lout.softmax(-1).cpu().numpy()[mo]
        nout[nsel[mo]] += 1
    uniform = 1.0 / K
    return Prediction(pe / np.maximum(ne, 1),
                      np.where(nin[:, None] > 0, pin / np.maximum(nin, 1)[:, None], uniform),
                      np.where(nout[:, None] > 0, pout / np.maximum(nout, 1)[:, None], uniform),
                      pex / np.maximum(nex, 1) if model.exist_head is not None else None)


def predict_tiled(model: ProTrackFormer, g: Graph, max_dt: int = 2, device="cpu", tile: int = 96,
                  margin: int = 24) -> Prediction:
    """Each tile sees its core plus a margin; only objects (and links leaving objects) in the core are kept."""
    N, E, K = len(g.node_t), len(g.edge_src), MAX_DEG + 1
    out = Prediction(np.zeros(E), np.full((N, K), 1.0 / K), np.full((N, K), 1.0 / K),
                     np.zeros(N) if model.exist_head is not None else None)
    if not N:
        return out
    y, x = g.node_pos[:, 1], g.node_pos[:, 2]
    for y0 in range(0, int(y.max()) + 1, tile):
        for x0 in range(0, int(x.max()) + 1, tile):
            core = (y >= y0) & (y < y0 + tile) & (x >= x0) & (x < x0 + tile)
            if not core.any():
                continue
            box = (y >= y0 - margin) & (y < y0 + tile + margin) & (x >= x0 - margin) & (x < x0 + tile + margin)
            nsel = np.nonzero(box)[0]
            sub, esel = subgraph(g, nsel)
            p = predict(model, sub, max_dt, device)
            keep_e = core[g.edge_src[esel]]
            out.link[esel[keep_e]] = p.link[keep_e]
            keep_n = core[nsel]
            out.p_in[nsel[keep_n]], out.p_out[nsel[keep_n]] = p.p_in[keep_n], p.p_out[keep_n]
            if out.exists is not None:
                out.exists[nsel[keep_n]] = p.exists[keep_n]
    return out


def load_checkpoint(path: Path, device="cpu") -> tuple[ProTrackFormer, dict]:
    ck = torch.load(path, map_location=device, weights_only=False)
    model = ProTrackFormer(**ck["model_config"]).to(device)
    model.load_state_dict(ck["state_dict"])
    return model.eval(), ck


def new_model(candidates: bool, **kw) -> ProTrackFormer:
    node_dim = len(OBJECT_FEATURES) + (len(CANDIDATE_FEATURES) if candidates else 0)
    return ProTrackFormer(node_dim, len(LINK_FEATURES), existence=candidates, **kw)


class Tracker:
    """A trained ProTrackFormer with its graph settings and decoder."""

    def __init__(self, checkpoint: Path, device: str | None = None):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model, ck = load_checkpoint(checkpoint, self.device)
        self.candidates = ck["candidates"]
        self.graph_cfg = GraphConfig.from_dict(ck["graph_config"])
        self.decoder = ck["decoder"]
        self.meta = {k: v for k, v in ck.items() if k not in ("state_dict",)}

    def graph(self, image: np.ndarray, masks: np.ndarray, instances: bool = True, boxes=None) -> Graph:
        return build_graph(image, masks, self.graph_cfg, candidates=self.candidates, instances=instances, boxes=boxes)

    def predict(self, g: Graph, tile: int | None = 96) -> Prediction:
        span = float(np.ptp(g.node_pos[:, 1:], axis=0).max()) if len(g.node_t) else 0.0
        if tile and span > tile:
            return predict_tiled(self.model, g, self.graph_cfg.max_dt, self.device, tile=tile)
        return predict(self.model, g, self.graph_cfg.max_dt, self.device)

    def track(self, image: np.ndarray, masks: np.ndarray, instances: bool | None = None, cover: bool | None = None,
              boxes=None, tile: int | None = 96, blocks: bool | None = None, workers: int | None = None
              ) -> tuple[Lineage, np.ndarray, Graph]:
        """Track a T x Z x Y x X movie given masks; returns the lineage, its label array and the graph.

        - `instances`: use the labels of `masks` as level-0 candidates (default: as in training)
        - `cover`: every foreground component must keep a candidate (default: as in training)
        """
        instances = self.meta.get("masks") != "foreground" if instances is None else instances
        cover = self.meta.get("cover", False) if cover is None else cover
        g = self.graph(image, masks, instances=instances, boxes=boxes)
        p = self.predict(g, tile)
        sel, keep = decode(g, p, self.decoder, cover=cover, blocks=blocks, workers=workers)
        lineage, labels = to_lineage(g, sel, keep)
        return lineage, labels, g
