import json

import numpy as np
import pytest
import tifffile

from conftest import CHECKPOINTS, DATA
from protrackformer import evaluate, track, train
from protrackformer.data import load_sequence
from protrackformer.inference import Tracker
from protrackformer.metrics import EvalConfig, evaluate as score, summarize

needs_data = pytest.mark.skipif(not (DATA / "bazooka").exists(), reason="benchmark data not present")


@needs_data
@pytest.mark.skipif(not (CHECKPOINTS / "bazooka_instances.pt").exists(), reason="checkpoint not present")
def test_released_checkpoint_tracks_a_crop():
    seq = load_sequence(DATA / "bazooka", "Crop07")
    lin, labels, _ = Tracker(CHECKPOINTS / "bazooka_instances.pt", "cpu").track(seq.image, seq.labels)
    s = summarize([score(seq.lineage, seq.labels, lin, labels)], EvalConfig(bootstrap=0))
    assert s["link/all"]["f1"] > 0.9
    assert s["merge/tolerant"]["f1"] > 0.5


@needs_data
@pytest.mark.slow
def test_train_track_evaluate(tmp_path):
    cfg = tmp_path / "tiny.yaml"
    cfg.write_text(f"data: {DATA / 'bazooka'}\nmasks: instances\ntrain: [Crop07, Crop09]\nval: [Crop03]\n"
                   f"test: [Crop01]\ncache: {tmp_path / 'cache'}\ndevice: cpu\n"
                   "optim: {steps: 40, eval_every: 20}\nmodel: {d: 32, node_layers: 1, edge_layers: 1}\n")
    train.main([str(cfg), "--out", str(tmp_path / "run")])
    result = json.loads((tmp_path / "run" / "result.json").read_text())
    assert result["decoder"]["kind"] in ("threshold", "ilp")
    assert (tmp_path / "run" / "test" / "Crop01" / "tracks.txt").exists()

    seq = load_sequence(DATA / "bazooka", "Crop01")
    tifffile.imwrite(tmp_path / "movie.tif", seq.image, imagej=True, metadata={"axes": "TZYX"})
    tifffile.imwrite(tmp_path / "masks.tif", (seq.labels > 0).astype(np.uint8), imagej=True, metadata={"axes": "TZYX"})
    out = tmp_path / "tracked"
    track.main(["--image", str(tmp_path / "movie.tif"), "--masks", str(tmp_path / "masks.tif"),
                "--checkpoint", str(tmp_path / "run" / "model.pt"), "--out", str(out), "--device", "cpu"])
    assert tifffile.imread(out / "tracked_masks.tif").shape == seq.labels.shape
    assert json.loads((out / "summary.json").read_text())["objects"] > 0

    evaluate.main(["--data", str(DATA / "bazooka"), "--pred", str(tmp_path / "run" / "test"), "--bootstrap", "10",
                   "--json", str(tmp_path / "scores.json")])
    assert "link/all" in json.loads((tmp_path / "scores.json").read_text())
