"""Score tracking results against a benchmark.

- `--pred` folders hold one result per sequence (`<pred>/<sequence>/tracks.txt` + `tracked_masks.tif`),
  e.g. the `test` folders of all cross-validation folds
- results of all sequences are pooled; intervals resample crops (Bazooka) or 64 px tiles (Gish)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .data import load_sequence
from .io import read_result
from .metrics import EvalConfig, evaluate, format_summary, summarize


def evaluate_folders(data: Path, folders: list[Path], cfg: EvalConfig | None = None) -> tuple[dict, list[str]]:
    results, names = [], []
    for folder in folders:
        for res in sorted(p for p in Path(folder).iterdir() if (p / "tracks.txt").exists()):
            seq = load_sequence(data, res.name)
            lin, lab = read_result(res)
            results.append(evaluate(seq.lineage, seq.labels, lin, lab, cfg, unit_of=seq.unit_of))
            names.append(res.name)
    if not results:
        raise SystemExit(f"no results (folders with tracks.txt) under {folders}")
    return summarize(results, cfg), names


def main(argv=None):
    ap = argparse.ArgumentParser(description="Score ProTrackFormer results against a CTC-DAG benchmark.")
    ap.add_argument("--data", type=Path, required=True, help="benchmark folder, e.g. data/bazooka")
    ap.add_argument("--pred", type=Path, nargs="+", required=True, help="folders of per-sequence results")
    ap.add_argument("--bootstrap", type=int, default=2000)
    ap.add_argument("--json", type=Path, help="write the pooled scores here")
    a = ap.parse_args(argv)
    summary, names = evaluate_folders(a.data, a.pred, EvalConfig(bootstrap=a.bootstrap))
    print(f"{len(names)} sequences: {' '.join(names)}")
    print(format_summary(summary))
    if a.json:
        a.json.write_text(json.dumps(summary, indent=1, default=float))


if __name__ == "__main__":
    main()
