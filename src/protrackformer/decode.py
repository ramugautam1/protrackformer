"""From probabilities to a lineage.

- `ilp`: maximise sum_e x_e logit(p_e) + mu sum_v s_v logit(q_v) + lam sum_v,k y_vk log P_deg(v = k)
  subject to: one degree class per selected object and direction; incident links = chosen degree;
  links only between selected objects; overlapping candidates exclude each other; optionally every
  foreground component keeps at least one candidate (`cover`, for an annotated foreground)
- degrees are constrained only where observable: no in-degree in the first frame, no out-degree in the last
- given instances, every candidate is an object and the program reduces to links and degrees
- `threshold`: keep links with p_e >= tau
- `blocks`: solve overlapping space-time blocks in parallel and stitch them, for whole movies
"""

from __future__ import annotations

import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import coo_matrix

from .graph import MAX_DEG, Graph, subgraph
from .lineage import Lineage


def _logit(p, eps):
    p = np.clip(p, eps, 1 - eps)
    return np.log(p / (1 - p))


def solve_ilp(g: Graph, link, p_in, p_out, exists=None, lam: float = 1.0, mu: float = 1.0, cover: bool = False,
              eps: float = 1e-4, time_limit: float = 600.0, mip_rel_gap: float | None = None):
    """Returns (selected objects, kept links) as boolean arrays."""
    E, N, K, T = len(link), len(g.node_t), MAX_DEG + 1, g.n_frames
    select = exists is not None
    in_nodes, out_nodes = np.nonzero(g.node_t > 0)[0], np.nonzero(g.node_t < T - 1)[0]
    S0 = E
    Y0 = S0 + (N if select else 0)
    Z0 = Y0 + K * len(in_nodes)
    nv = Z0 + K * len(out_nodes)
    c = np.zeros(nv)
    c[:E] = -_logit(link, eps)
    if select:
        c[S0:Y0] = -mu * _logit(exists, eps)
    c[Y0:Z0] = -lam * np.log(np.clip(p_in[in_nodes], eps, 1)).ravel()
    c[Z0:] = -lam * np.log(np.clip(p_out[out_nodes], eps, 1)).ravel()
    rows, cols, vals, lo, hi = [], [], [], [], []

    def add(cs, vs, low, high):
        r = len(lo)
        rows.extend([r] * len(cs))
        cols.extend(cs)
        vals.extend(vs)
        lo.append(low)
        hi.append(high)

    if select:
        for e, (u, v) in enumerate(zip(g.edge_src, g.edge_dst)):
            add([e, S0 + u], [1, -1], -np.inf, 0)
            add([e, S0 + v], [1, -1], -np.inf, 0)
        for a, b in g.conflicts:
            add([S0 + a, S0 + b], [1, 1], -np.inf, 1)
        if cover:
            for cc in np.unique(g.node_cc):
                members = np.nonzero(g.node_cc == cc)[0]
                add([S0 + m for m in members], [1] * len(members), 1, np.inf)
    inc_in, inc_out = [[] for _ in range(N)], [[] for _ in range(N)]
    for e, (u, v) in enumerate(zip(g.edge_src, g.edge_dst)):
        inc_out[u].append(e)
        inc_in[v].append(e)
    for nodes, base, inc in ((in_nodes, Y0, inc_in), (out_nodes, Z0, inc_out)):
        for i, n in enumerate(nodes):
            ys = list(range(base + K * i, base + K * (i + 1)))
            if select:
                add(ys + [S0 + n], [1] * K + [-1], 0, 0)
            else:
                add(ys, [1] * K, 1, 1)
            add(inc[n] + ys, [1] * len(inc[n]) + [-k for k in range(K)], 0, 0)
    options = {"time_limit": time_limit}
    if mip_rel_gap:
        options["mip_rel_gap"] = mip_rel_gap
    if not len(lo):
        return np.ones(N, bool), link >= 0.5
    A = coo_matrix((vals, (rows, cols)), shape=(len(lo), nv)).tocsr()
    res = milp(c, constraints=LinearConstraint(A, lo, hi), integrality=np.ones(nv), bounds=Bounds(0, 1),
               options=options)
    if res.x is None:
        raise RuntimeError(f"integer program failed: {res.message}")
    sel = res.x[S0:Y0] > 0.5 if select else np.ones(N, bool)
    return sel, res.x[:E] > 0.5


def _solve_job(job):
    sub, link, p_in, p_out, exists, lam, mu, cover, gap, time_limit = job
    return solve_ilp(sub, link, p_in, p_out, exists, lam, mu, cover, time_limit=time_limit, mip_rel_gap=gap)


def solve_blocks(g: Graph, link, p_in, p_out, exists=None, lam=1.0, mu=1.0, cover=False, core=128, margin=16,
                 tlen=20, toverlap=4, gap=None, time_limit=900.0, workers=None):
    """Integer program over overlapping blocks (core 128 px x 20 frames, overlap 16 px and 4 frames).

    - an object is kept if the block whose core holds it selects it
    - a link is kept if the block whose core holds its source keeps it and both objects are kept
    """
    N, E = len(g.node_t), len(g.edge_src)
    sel, keep = np.zeros(N, bool), np.zeros(E, bool)
    t, y, x = g.node_t, g.node_pos[:, 1], g.node_pos[:, 2]
    jobs, blocks = [], []
    for c0 in range(0, g.n_frames, tlen):
        c1 = min(g.n_frames, c0 + tlen)
        b0, b1 = max(0, c0 - toverlap), min(g.n_frames, c1 + toverlap)
        for y0 in range(0, int(y.max()) + 1, core):
            for x0 in range(0, int(x.max()) + 1, core):
                in_core = (t >= c0) & (t < c1) & (y >= y0) & (y < y0 + core) & (x >= x0) & (x < x0 + core)
                if not in_core.any():
                    continue
                box = ((t >= b0) & (t < b1) & (y >= y0 - margin) & (y < y0 + core + margin) &
                       (x >= x0 - margin) & (x < x0 + core + margin))
                nsel = np.nonzero(box)[0]
                sub, esel = subgraph(g, nsel, t0=b0, n_frames=b1 - b0)
                blocks.append((nsel, esel, in_core))
                jobs.append((sub, link[esel], p_in[nsel], p_out[nsel], None if exists is None else exists[nsel],
                             lam, mu, cover, gap, time_limit))
    workers = workers or max(1, min(len(jobs), (os.cpu_count() or 2) // 2))
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for (nsel, esel, in_core), (s_sel, s_keep) in zip(blocks, ex.map(_solve_job, jobs)):
            sel[nsel[s_sel & in_core[nsel]]] = True
            keep[esel[s_keep & in_core[g.edge_src[esel]]]] = True
    return sel, keep & sel[g.edge_src] & sel[g.edge_dst]


def decode(g: Graph, p, decoder: dict, cover: bool = False, blocks: bool | None = None, workers=None):
    """Apply the decoder chosen on validation (`{"kind": "threshold", "tau": ...}` or `{"kind": "ilp", ...}`)."""
    if decoder["kind"] == "threshold":
        return np.ones(len(g.node_t), bool), p.link >= decoder["tau"]
    lam, mu = decoder.get("lam", 1.0), decoder.get("mu", 1.0)
    if blocks is None:
        span = float(np.ptp(g.node_pos[:, 1:], axis=0).max()) if len(g.node_t) else 0.0
        blocks = span > 160 or g.n_frames > 30
    if blocks:
        return solve_blocks(g, p.link, p.p_in, p.p_out, p.exists, lam, mu, cover, workers=workers)
    return solve_ilp(g, p.link, p.p_in, p.p_out, p.exists, lam, mu, cover)


def to_lineage(g: Graph, sel: np.ndarray, keep: np.ndarray) -> tuple[Lineage, np.ndarray]:
    """Selected objects and kept links -> lineage over `(t, node label)` and a T x Z x Y x X label array."""
    shape = g.layers[0][0].shape
    labels = np.zeros((g.n_frames, *shape), np.uint32)
    for n in np.nonzero(sel)[0]:
        labels[g.node_t[n]][g.mask(n)] = g.node_label[n]
    node = lambda i: (int(g.node_t[i]), int(g.node_label[i]))  # noqa: E731
    nodes = {node(n) for n in np.nonzero(sel)[0]}
    keep = keep & sel[g.edge_src] & sel[g.edge_dst]
    edges = {(node(u), node(v)) for u, v in zip(g.edge_src[keep], g.edge_dst[keep])}
    return Lineage(nodes, edges), labels
