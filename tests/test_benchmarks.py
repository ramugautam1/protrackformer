import json

import pytest

from conftest import DATA
from protrackformer.data import load_folds, load_meta, load_sequence

pytestmark = pytest.mark.skipif(not (DATA / "bazooka").exists(), reason="benchmark data not present")


def test_bazooka_totals():
    meta = load_meta(DATA / "bazooka")
    assert len(meta["sequences"]) == 28
    assert (meta["totals"]["splits"], meta["totals"]["merges"]) == (108, 139)
    seq = load_sequence(DATA / "bazooka", "Crop07")
    assert seq.image.shape == seq.labels.shape == (40, 13, 32, 32)


def test_bazooka_folds_test_every_crop_once():
    folds = load_folds(DATA / "bazooka")
    tested = sorted(c for f in folds.values() for c in f["test"])
    assert tested == load_meta(DATA / "bazooka")["sequences"]
    for f in folds.values():
        assert not (set(f["train"]) & set(f["val"])) and not (set(f["train"]) & set(f["test"]))


def test_gish_quadrant_views():
    totals = {"splits": 0, "merges": 0}
    for q in range(4):
        s = load_sequence(DATA / "gish", f"Q{q}").lineage.summary()
        totals["splits"] += s["splits"]
        totals["merges"] += s["merges"]
    assert totals == {"splits": 108, "merges": 102}  # events whose objects stay inside one quadrant
    assert json.loads((DATA / "gish" / "meta.json").read_text())["totals"]["splits"] == 110
