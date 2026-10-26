"""Training ProTrackFormer.

- each step takes a window of `W` frames from every training sequence, as one batch
- loss: link BCE (branch links weighted 1 + branch_weight), degree cross-entropy where the degree is
  observable in the window, consistency between expected degree and the sum of incident link
  probabilities, and existence BCE for candidate segments
- every `eval_every` steps the validation sequences are decoded; the checkpoint with the best
  mean(link F1, tolerant split F1, tolerant merge F1) is kept
- the decoder settings are then chosen on validation, and test sequences, if given, are tracked and scored
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml

from . import __version__
from .data import instance_masks, load_folds, load_sequence, segmentation_masks
from .decode import decode, to_lineage
from .graph import MAX_DEG, Graph, GraphConfig, build_graph, config_dict, label_graph, load_graph, save_graph
from .inference import new_model, predict
from .io import write_result
from .metrics import EvalConfig, evaluate, format_summary, selection_score, summarize
from .windows import augment, concat, inputs, to_tensors, window_arrays

log = logging.getLogger("protrackformer")

DEFAULTS = {
    "masks": "instances",
    "graph": {},
    "model": {"d": 128, "heads": 4, "node_layers": 2, "edge_layers": 3, "window": 6},
    "optim": {"steps": 3000, "lr": 3.0e-4, "weight_decay": 1.0e-4, "warmup": 200, "eval_every": 250},
    "loss": {"branch_weight": 2.0, "deg_weight": 0.5, "con_weight": 0.2, "exist_weight": 1.0},
    "time_reversal": True,
    "seed": 0,
    "cache": "cache/graphs",
}


def load_config(path: Path, overrides: list[str]) -> dict:
    cfg = json.loads(json.dumps(DEFAULTS))
    for k, v in (yaml.safe_load(Path(path).read_text()) or {}).items():
        cfg[k] = {**cfg[k], **v} if isinstance(cfg.get(k), dict) and isinstance(v, dict) else v
    for item in overrides:
        key, value = item.split("=", 1)
        node = cfg
        *parents, last = key.split(".")
        for p in parents:
            node = node.setdefault(p, {})
        node[last] = yaml.safe_load(value)
    return cfg


def mode_of(cfg: dict) -> dict:
    """`instances`: the ground-truth instances; `foreground`: candidates of the annotated foreground;
    anything else: a folder of segmentations whose objects (and their watershed parts) are candidates."""
    m = cfg["masks"]
    if m == "instances":
        return {"candidates": False, "instances": True, "cover": False}
    if m == "foreground":
        return {"candidates": True, "instances": False, "cover": True}
    return {"candidates": True, "instances": True, "cover": False}


def sequence_graph(cfg: dict, name: str, gcfg: GraphConfig) -> tuple[Graph, object]:
    seq = load_sequence(cfg["data"], name)
    mode = mode_of(cfg)
    key = json.dumps([str(Path(cfg["data"]).resolve()), name, cfg["masks"], config_dict(gcfg)], sort_keys=True)
    path = Path(cfg["cache"]) / f"{name}_{hashlib.sha1(key.encode()).hexdigest()[:12]}.npz"
    if path.exists():
        return load_graph(path), seq
    if cfg["masks"] == "instances":
        masks = instance_masks(seq.labels)
    elif cfg["masks"] == "foreground":
        masks = (seq.labels > 0).astype(np.uint8)
    else:
        masks = segmentation_masks(cfg["masks"], cfg["data"], name)
    g = build_graph(seq.image, masks, gcfg, candidates=mode["candidates"], instances=mode["instances"])
    label_graph(g, seq.lineage, seq.labels, gcfg)
    path.parent.mkdir(parents=True, exist_ok=True)
    save_graph(g, path)
    log.info("graph %s: %s", name, g.stats)
    return g, seq


def loss_fn(out, s, deg_w, pos_w, c: dict):
    link, lin, lout, lex = out
    y = s["y"]
    loss = 0.0
    if len(y):
        branch = y * ((s["outdeg"][s["src"]] >= 2) | (s["indeg"][s["dst"]] >= 2)).float()
        weight = 1 + c["branch_weight"] * branch
        loss = (F.binary_cross_entropy_with_logits(link, y, reduction="none") * weight).mean()
    l_deg = 0.0
    for logits, lab, m in ((lin, s["indeg"], s["in_mask"]), (lout, s["outdeg"], s["out_mask"])):
        if m.any():
            l_deg = l_deg + F.cross_entropy(logits[m], lab[m], weight=deg_w)
    n = lin.shape[0]
    p = torch.sigmoid(link)
    k = torch.arange(MAX_DEG + 1, device=p.device, dtype=p.dtype)
    s_in = torch.zeros(n, device=p.device).index_add_(0, s["dst"], p)
    s_out = torch.zeros(n, device=p.device).index_add_(0, s["src"], p)
    l_con = 0.0
    if s["in_mask"].any():
        l_con = l_con + ((s_in - (lin.softmax(-1) * k).sum(-1))[s["in_mask"]] ** 2).mean()
    if s["out_mask"].any():
        l_con = l_con + ((s_out - (lout.softmax(-1) * k).sum(-1))[s["out_mask"]] ** 2).mean()
    loss = loss + c["deg_weight"] * l_deg + c["con_weight"] * l_con
    if lex is not None:
        loss = loss + c["exist_weight"] * F.binary_cross_entropy_with_logits(lex, s["exists"], pos_weight=pos_w)
    return loss


def run_sequence(model, g: Graph, seq, decoder: dict, cover: bool, device):
    p = predict(model, g, device=device)
    sel, keep = decode(g, p, decoder, cover=cover, blocks=False)
    return to_lineage(g, sel, keep)


def score(model, graphs, seqs, decoder, cover, device, bootstrap=0):
    results = []
    for g, seq in zip(graphs, seqs):
        lin, lab = run_sequence(model, g, seq, decoder, cover, device)
        results.append(evaluate(seq.lineage, seq.labels, lin, lab, EvalConfig(), unit_of=seq.unit_of))
    return summarize(results, EvalConfig(bootstrap=bootstrap)), results


def decoder_grid(candidates: bool) -> list[dict]:
    if candidates:
        return [{"kind": "ilp", "lam": lam, "mu": mu} for lam in (0.5, 1.0, 2.0) for mu in (0.5, 1.0, 2.0)]
    return ([{"kind": "threshold", "tau": tau} for tau in (0.3, 0.4, 0.5, 0.6)] +
            [{"kind": "ilp", "lam": lam} for lam in (0.5, 1.0, 2.0)])


def train(cfg: dict, out: Path) -> dict:
    torch.manual_seed(cfg["seed"])
    rng = np.random.default_rng(cfg["seed"])
    device = torch.device(cfg.get("device") or ("cuda" if torch.cuda.is_available() else "cpu"))
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
    if "fold" in cfg and cfg["fold"] is not None:
        fold = load_folds(cfg["data"])[str(cfg["fold"])]
        cfg = {**cfg, **{k: cfg.get(k) or fold[k] for k in ("train", "val", "test")}}
    gcfg = GraphConfig.from_dict(cfg["graph"])
    mode = mode_of(cfg)
    train_g = [sequence_graph(cfg, n, gcfg)[0] for n in cfg["train"]]
    val = [sequence_graph(cfg, n, gcfg) for n in cfg["val"]]
    val_g, val_s = [v[0] for v in val], [v[1] for v in val]

    mc, oc, lc = cfg["model"], cfg["optim"], cfg["loss"]
    W = mc["window"]
    model = new_model(mode["candidates"], **mc).to(device)
    nf, ef = np.concatenate([g.node_feat for g in train_g]), np.concatenate([g.edge_feat for g in train_g])
    model.set_normalisation(nf.mean(0), nf.std(0) + 1e-6, ef.mean(0), ef.std(0) + 1e-6)
    if cfg.get("init"):
        model.load_state_dict(torch.load(cfg["init"], map_location=device, weights_only=False)["state_dict"])
    degs = np.concatenate([np.concatenate([g.node_indeg, g.node_outdeg]) for g in train_g])
    freq = np.bincount(degs, minlength=MAX_DEG + 1) + 1.0
    deg_w = torch.tensor((freq.sum() / freq) ** 0.5, dtype=torch.float32, device=device)
    deg_w = deg_w / deg_w.mean()
    ex = np.concatenate([g.node_exists for g in train_g]).mean()
    pos_w = torch.tensor((1 - ex) / max(ex, 1e-3), device=device)
    opt = torch.optim.AdamW(model.parameters(), lr=oc["lr"], weight_decay=oc["weight_decay"])
    steps = oc["steps"]
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1, (s + 1) / oc["warmup"]) * 0.5 * (1 + math.cos(math.pi * min(s, steps) / steps)))
    t_max = max(0, min(g.n_frames for g in train_g) - W)
    default = {"kind": "ilp", "lam": 1.0, "mu": 1.0} if mode["candidates"] else {"kind": "threshold", "tau": 0.5}
    history, best, start = [], (-1.0, 0), time.time()
    for step in range(1, steps + 1):
        model.train()
        t0 = int(rng.integers(0, t_max + 1))
        w = concat([window_arrays(g, max(0, min(t0, g.n_frames - W)), W) for g in train_g])
        s = to_tensors(augment(w, W, rng, cfg["time_reversal"]), W, gcfg.max_dt, device)
        loss = loss_fn(model(*inputs(s)), s, deg_w, pos_w, lc)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        if step % oc["eval_every"] == 0 or step == steps:
            summary, _ = score(model, val_g, val_s, default, mode["cover"], device)
            sc = selection_score(summary)
            history.append({"step": step, "loss": round(float(loss), 4), "val_score": round(sc, 4),
                            "val_link_f1": summary["link/all"]["f1"], "val_split_tol": summary["split/tolerant"]["f1"],
                            "val_merge_tol": summary["merge/tolerant"]["f1"],
                            "minutes": round((time.time() - start) / 60, 1)})
            log.info(json.dumps(history[-1]))
            if sc > best[0]:
                best = (sc, step)
                torch.save(model.state_dict(), out / "best_state.pt")

    model.load_state_dict(torch.load(out / "best_state.pt", map_location=device))
    decoder, best_dec = None, -1.0
    for d in decoder_grid(mode["candidates"]):
        sc = selection_score(score(model, val_g, val_s, d, mode["cover"], device)[0])
        if sc > best_dec:
            decoder, best_dec = d, sc
    ckpt = {"state_dict": model.state_dict(), "model_config": model.cfg, "graph_config": config_dict(gcfg),
            "candidates": mode["candidates"], "masks": cfg["masks"], "cover": mode["cover"], "decoder": decoder,
            "train": cfg["train"], "val": cfg["val"], "data": str(cfg["data"]), "best_step": best[1],
            "version": __version__}
    torch.save(ckpt, out / "model.pt")
    (out / "best_state.pt").unlink()
    result = {"best_step": best[1], "val_score": best[0], "decoder": decoder, "decoder_val_score": best_dec,
              "history": history}
    if cfg.get("test"):
        result["test"] = test(model, cfg, gcfg, decoder, mode, out / "test", device)
    (out / "result.json").write_text(json.dumps(result, indent=1, default=float))
    return result


def test(model, cfg, gcfg, decoder, mode, out: Path, device) -> dict:
    results = []
    for name in cfg["test"]:
        g, seq = sequence_graph(cfg, name, gcfg)
        lin, lab = run_sequence(model, g, seq, decoder, mode["cover"], device)
        meta = seq.meta
        write_result(out / name, lin, lab, tuple(meta.get("voxel_um_zyx", (0.5, 0.124, 0.124))), meta.get("frame_s"))
        results.append(evaluate(seq.lineage, seq.labels, lin, lab, EvalConfig(), unit_of=seq.unit_of))
    summary = summarize(results)
    (out / "metrics.txt").write_text(format_summary(summary) + "\n")
    log.info("test\n%s", format_summary(summary))
    return summary


def main(argv=None):
    ap = argparse.ArgumentParser(description="Train ProTrackFormer.")
    ap.add_argument("config", type=Path, help="YAML file, e.g. configs/bazooka_instances.yaml")
    ap.add_argument("--fold", type=int, help="cross-validation fold of folds.json")
    ap.add_argument("--seed", type=int)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE",
                    help="override config entries, e.g. optim.steps=500")
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    cfg = load_config(a.config, a.set)
    if a.fold is not None:
        cfg["fold"] = a.fold
    if a.seed is not None:
        cfg["seed"] = a.seed
    train(cfg, a.out)


if __name__ == "__main__":
    main()
