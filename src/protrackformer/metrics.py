"""Split- and merge-aware tracking metrics on lineage DAGs.

- objects are matched one-to-one per frame (Hungarian on IoU, IoU >= 0.3)
- DET: object detection F1
- LINK: link F1, for all links, one-to-one continuations and branch links (those touching an event)
- EVENT: splits and merges as separate detection tasks
  - strict: same anchor object and the exact set of partners
  - anchor: same anchor object, any partners
  - tolerant: an event of the same kind within one frame and 6 px (0.74 um)
- TRA-DAG: the CTC's AOGM measure applied to the DAG
- CHOTA-DAG: CHOTA with a node's trajectory = all its ancestors and descendants
- confidence intervals: cluster bootstrap over independent units (crops, or tiles of a field)
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np
from scipy import ndimage as ndi
from scipy.optimize import linear_sum_assignment

from .lineage import Event, Lineage

Node = tuple[int, int]
FAMILIES = [("det",), ("link", "all"), ("link", "cont"), ("link", "branch"),
            ("split", "strict"), ("split", "anchor"), ("split", "tolerant"),
            ("merge", "strict"), ("merge", "anchor"), ("merge", "tolerant")]


@dataclass
class EvalConfig:
    min_iou: float = 0.3
    tol_frames: int = 1
    tol_radius_px: float = 6.0
    z_scale: float = 0.5 / 0.124
    transient_window: int = 2
    aogm_weights: dict = field(default_factory=lambda: dict(ns=5, fn=10, fp=1, ed=1, ea=1.5, ec=1))
    bootstrap: int = 2000
    seed: int = 0


@dataclass
class Matching:
    gt_to_pred: dict = field(default_factory=dict)
    pred_to_gt: dict = field(default_factory=dict)
    iou: dict = field(default_factory=dict)
    majority: dict = field(default_factory=dict)  # gt -> pred covering more than half of it (for AOGM)


def match(gt: np.ndarray, pred: np.ndarray, min_iou: float = 0.3) -> Matching:
    if len(gt) != len(pred):
        raise ValueError(f"frame count differs: ground truth {len(gt)}, prediction {len(pred)}")
    m = Matching()
    for t in range(len(gt)):
        g, p = gt[t].ravel().astype(np.int64), pred[t].ravel().astype(np.int64)
        fg = (g > 0) & (p > 0)
        if not fg.any():
            continue
        base = int(p.max()) + 1
        keys, inter = np.unique(g[fg] * base + p[fg], return_counts=True)
        gi, pi = keys // base, keys % base
        gsize, psize = np.bincount(g), np.bincount(p)
        iou = inter / (gsize[gi] + psize[pi] - inter)
        for a, b, n in zip(gi, pi, inter):
            if n > 0.5 * gsize[a]:
                m.majority[(t, int(a))] = (t, int(b))
        ok = iou >= min_iou
        if not ok.any():
            continue
        gl, pl = np.unique(gi[ok]), np.unique(pi[ok])
        cost = np.zeros((len(gl), len(pl)))
        cost[np.searchsorted(gl, gi[ok]), np.searchsorted(pl, pi[ok])] = -iou[ok]
        for a, b in zip(*linear_sum_assignment(cost)):
            if cost[a, b] < 0:
                gn, pn = (t, int(gl[a])), (t, int(pl[b]))
                m.gt_to_pred[gn], m.pred_to_gt[pn], m.iou[gn] = pn, gn, float(-cost[a, b])
    return m


def centroids(labels: np.ndarray) -> dict[Node, np.ndarray]:
    out = {}
    for t in range(len(labels)):
        labs = [int(v) for v in np.unique(labels[t]) if v]
        for lab, c in zip(labs, ndi.center_of_mass(labels[t] > 0, labels[t], labs)):
            out[(t, lab)] = np.asarray(c, float)
    return out


def touching_pairs(frame: np.ndarray) -> set[tuple[int, int]]:
    """Pairs of labels with 26-adjacent voxels."""
    pairs = set()
    for off in np.argwhere(np.ones((3, 3, 3), bool)) - 1:
        if not off.any() or tuple(off) < (0, 0, 0):
            continue
        sa = tuple(slice(max(0, -o), frame.shape[i] - max(0, o)) for i, o in enumerate(off))
        sb = tuple(slice(max(0, o), frame.shape[i] - max(0, -o)) for i, o in enumerate(off))
        a, b = frame[sa], frame[sb]
        k = (a > 0) & (b > 0) & (a != b)
        if k.any():
            u = np.unique(np.stack([np.minimum(a[k], b[k]), np.maximum(a[k], b[k])], 1), axis=0)
            pairs |= {(int(x), int(y)) for x, y in u}
    return pairs


def _strata(lin: Lineage, events: list[Event], window: int, touching: dict) -> dict:
    s, p = lin.succ(), lin.pred()

    def reach(adj, n, depth):
        out, frontier = {n}, {n}
        for _ in range(depth):
            frontier = {q for f in frontier for q in adj.get(f, [])}
            out |= frontier
        return out

    out = {}
    for e in events:
        parts = list(e.partners)
        sets = [reach(p if e.kind == "merge" else s, q, window) for q in parts]
        transient = any(sets[i] & sets[j] for i in range(len(sets)) for j in range(i + 1, len(sets)))
        frame = parts[0][0]
        labs = sorted(q[1] for q in parts if q[0] == frame)
        pairs = touching.get(frame, set())
        touch = any((labs[i], labs[j]) in pairs for i in range(len(labs)) for j in range(i + 1, len(labs)))
        out[e] = {"fan": "2" if len(parts) == 2 else "3+", "transient": transient, "touching": touch}
    return out


def evaluate(gt: Lineage, gt_labels: np.ndarray, pred: Lineage, pred_labels: np.ndarray,
             cfg: EvalConfig | None = None, unit_of=None) -> dict:
    """Counts per independent unit for every metric family, plus TRA-DAG and CHOTA-DAG terms.

    `unit_of(node, centroid) -> str` assigns objects to units (default: one unit).
    Pool one or several results with `summarize`.
    """
    cfg = cfg or EvalConfig()
    m = match(gt_labels, pred_labels, cfg.min_iou)
    gc, pc = centroids(gt_labels), centroids(pred_labels)
    unit_of = unit_of or (lambda n, c: "all")
    gu = {n: unit_of(n, c) for n, c in gc.items()}
    pu = {n: unit_of(n, c) for n, c in pc.items()}
    counts = {f: defaultdict(lambda: [0, 0, 0, 0]) for f in FAMILIES}  # tp (gt side), fp, fn, tp (pred side)

    def add(family, unit, i, n=1):
        counts[family][unit][i] += n

    for n in gt.nodes:
        add(("det",), gu.get(n, "?"), 0 if n in m.gt_to_pred and m.gt_to_pred[n] in pred.nodes else 2)
    for n in pred.nodes:
        add(("det",), pu.get(n, "?"), 3 if n in m.pred_to_gt and m.pred_to_gt[n] in gt.nodes else 1)

    gk, pk = gt.edge_kind(), pred.edge_kind()
    mapped = {e: (m.pred_to_gt.get(e[0]), m.pred_to_gt.get(e[1])) for e in pred.edges}
    hit = {v for v in mapped.values() if v in gt.edges}
    for kind in ("all", "cont", "branch"):
        for e in gt.edges:
            if kind == "all" or gk[e] == kind:
                add(("link", kind), gu.get(e[0], "?"), 0 if e in hit else 2)
        for e in pred.edges:
            if kind == "all" or pk[e] == kind:
                add(("link", kind), pu.get(e[0], "?"), 3 if mapped[e] in gt.edges else 1)

    gev, pev = gt.events(), pred.events()
    touching = {t: touching_pairs(gt_labels[t]) for t in range(len(gt_labels))}
    strata = _strata(gt, gev, cfg.transient_window, touching)
    recall = {}
    for kind in ("split", "merge"):
        G, P = [e for e in gev if e.kind == kind], [e for e in pev if e.kind == kind]
        by_anchor = {e.anchor: e for e in P}
        strict, anchor, used_strict, used_anchor = set(), set(), set(), set()
        for e in G:
            pe = by_anchor.get(m.gt_to_pred.get(e.anchor))
            if pe is None:
                continue
            anchor.add(e)
            used_anchor.add(pe)
            if {m.pred_to_gt.get(q) for q in pe.partners} == set(e.partners):
                strict.add(e)
                used_strict.add(pe)
        tol_g, tol_p = set(), set()
        if G and P:
            big = 1e9
            cost = np.full((len(G), len(P)), big)
            for i, e in enumerate(G):
                for j, f in enumerate(P):
                    a, b = gc.get(e.anchor), pc.get(f.anchor)
                    if a is None or b is None or abs(e.t - f.t) > cfg.tol_frames:
                        continue
                    dist = float(np.linalg.norm((a - b) * np.array([cfg.z_scale, 1, 1])))
                    if dist <= cfg.tol_radius_px:
                        cost[i, j] = dist + abs(e.t - f.t)
            for i, j in zip(*linear_sum_assignment(cost)):
                if cost[i, j] < big:
                    tol_g.add(G[i])
                    tol_p.add(P[j])
        for level, tp_g, tp_p in (("strict", strict, used_strict), ("anchor", anchor, used_anchor),
                                  ("tolerant", tol_g, tol_p)):
            for e in G:
                add((kind, level), gu.get(e.anchor, "?"), 0 if e in tp_g else 2)
            for e in P:
                add((kind, level), pu.get(e.anchor, "?"), 3 if e in tp_p else 1)
        recall[kind] = {f"{k}={v}": [sum(1 for e in G if str(strata[e][k]) == v and e in strict),
                                     sum(1 for e in G if str(strata[e][k]) == v and e in tol_g),
                                     sum(1 for e in G if str(strata[e][k]) == v)]
                        for k in ("fan", "transient", "touching") for v in sorted({str(strata[e][k]) for e in G})}

    return {"counts": {"/".join(f): dict(c) for f, c in counts.items()}, "strata": recall,
            "tra": _aogm(gt, pred, m, cfg.aogm_weights), "chota": _chota(gt, pred, m),
            "gt_summary": gt.summary(), "pred_summary": pred.summary()}


def _lineage_sets(lin: Lineage) -> dict[Node, set]:
    s, p = lin.succ(), lin.pred()
    order = sorted(lin.nodes)
    anc, des = {}, {}
    for n in order:
        anc[n] = set().union(*[anc[q] | {q} for q in p.get(n, [])]) if p.get(n) else set()
    for n in reversed(order):
        des[n] = set().union(*[des[q] | {q} for q in s.get(n, [])]) if s.get(n) else set()
    return {n: anc[n] | des[n] | {n} for n in lin.nodes}


def _chota(gt: Lineage, pred: Lineage, m: Matching) -> dict:
    pairs = {g: q for g, q in m.gt_to_pred.items() if g in gt.nodes and q in pred.nodes}
    out = {"n_tp": len(pairs), "n_gt": len(gt.nodes), "n_pred": len(pred.nodes), "sum_a": 0.0}
    if pairs:
        lg, lp = _lineage_sets(gt), _lineage_sets(pred)
        for g, q in pairs.items():
            tpa = sum(1 for g2 in lg[g] if g2 in pairs and pairs[g2] in lp[q])
            out["sum_a"] += tpa / (len(lg[g]) + len(lp[q]) - tpa)
    return out


def _aogm(gt: Lineage, pred: Lineage, m: Matching, w: dict) -> dict:
    M = {g: q for g, q in m.majority.items() if g in gt.nodes and q in pred.nodes}
    inv = defaultdict(list)
    for g, q in M.items():
        inv[q].append(g)
    ns = sum(len(v) - 1 for v in inv.values())
    fn, fp = len(gt.nodes) - len(M), len(pred.nodes - set(inv))

    def semantics(lin):
        s, p = lin.succ(), lin.pred()
        return {(u, v): len(s[u]) == 1 and len(p[v]) == 1 and v[0] - u[0] == 1 for u, v in lin.edges}

    gsem, psem = semantics(gt), semantics(pred)
    ea = ec = 0
    covered = set()
    for g1, g2 in gt.edges:
        q = (M.get(g1), M.get(g2))
        if None not in q and q in pred.edges:
            covered.add(q)
            ec += int(gsem[(g1, g2)] != psem[q])
        else:
            ea += 1
    ed = len(pred.edges - covered)
    aogm = w["ns"] * ns + w["fn"] * fn + w["fp"] * fp + w["ed"] * ed + w["ea"] * ea + w["ec"] * ec
    return {"aogm": aogm, "aogm0": w["fn"] * len(gt.nodes) + w["ea"] * len(gt.edges),
            "ns": ns, "fn": fn, "fp": fp, "ed": ed, "ea": ea, "ec": ec}


def _f1(tp_g, fp, fn, tp_p):
    if tp_p + fp == 0 and tp_g + fn == 0:
        return float("nan")
    if tp_p + fp == 0 or tp_g + fn == 0:
        return 0.0
    p, r = tp_p / (tp_p + fp), tp_g / (tp_g + fn)
    return 2 * p * r / (p + r) if p + r else 0.0


def summarize(results: list[dict], cfg: EvalConfig | None = None) -> dict:
    """Pool results (e.g. the test sequences of all folds) and add cluster-bootstrap 95% intervals."""
    cfg = cfg or EvalConfig()
    out = {}
    for fam in FAMILIES:
        key = "/".join(fam)
        per = {}
        for r in results:
            for u, c in r["counts"][key].items():
                per[u] = [a + b for a, b in zip(per.get(u, [0, 0, 0, 0]), c)]
        units = sorted(per)
        A = np.array([per[u] for u in units], float).reshape(-1, 4)
        tp_g, fp, fn, tp_p = A.sum(0) if len(A) else (0, 0, 0, 0)
        d = {"f1": _f1(tp_g, fp, fn, tp_p), "precision": tp_p / (tp_p + fp) if tp_p + fp else float("nan"),
             "recall": tp_g / (tp_g + fn) if tp_g + fn else float("nan"), "tp": int(tp_g), "fp": int(fp),
             "fn": int(fn), "units": len(units)}
        rng = np.random.default_rng(cfg.seed)
        draws = cfg.bootstrap if len(units) > 1 else 0
        boot = [_f1(*A[rng.integers(0, len(units), len(units))].sum(0)) for _ in range(draws)]
        boot = [b for b in boot if b == b]
        d["ci95"] = [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))] if boot else [float("nan")] * 2
        out[key] = d
    tra = [r["tra"] for r in results]
    aogm0 = sum(x["aogm0"] for x in tra)
    out["tra_dag"] = 1 - min(sum(x["aogm"] for x in tra), aogm0) / aogm0 if aogm0 else float("nan")
    ch = [r["chota"] for r in results]
    ntp = sum(x["n_tp"] for x in ch)
    den = sum(x["n_gt"] + x["n_pred"] for x in ch) - ntp
    det_a = ntp / den if den else float("nan")
    ass_a = sum(x["sum_a"] for x in ch) / ntp if ntp else float("nan")
    out["chota_dag"] = float(np.sqrt(det_a * ass_a)) if ntp else 0.0
    out["sequences"] = len(results)
    return out


def selection_score(summary: dict) -> float:
    """Mean of link F1 and tolerant split and merge F1; the model-selection criterion."""
    vals = [summary["link/all"]["f1"], summary["split/tolerant"]["f1"], summary["merge/tolerant"]["f1"]]
    return float(np.mean([v if v == v else 0.0 for v in vals]))


def format_summary(s: dict) -> str:
    f = lambda x: "  -  " if x != x else f"{x:.3f}"  # noqa: E731
    lines = [f"{s['sequences']} sequence(s)"]
    for key, name in (("det", "objects"), ("link/all", "links"), ("link/branch", "branch links"),
                      ("split/strict", "splits (strict)"), ("split/tolerant", "splits (tolerant)"),
                      ("merge/strict", "merges (strict)"), ("merge/tolerant", "merges (tolerant)")):
        d = s[key]
        lines.append(f"{name:18s} F1 {f(d['f1'])}  [{f(d['ci95'][0])}, {f(d['ci95'][1])}]  "
                     f"TP {d['tp']} FP {d['fp']} FN {d['fn']}")
    lines.append(f"{'TRA-DAG':18s} {f(s['tra_dag'])}")
    lines.append(f"{'CHOTA-DAG':18s} {f(s['chota_dag'])}")
    return "\n".join(lines)
