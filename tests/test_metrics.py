from protrackformer.lineage import Lineage
from protrackformer.metrics import EvalConfig, evaluate, summarize


def test_perfect_prediction(toy):
    _, labels, lin = toy
    s = summarize([evaluate(lin, labels, lin, labels)], EvalConfig(bootstrap=0))
    for key in ("det", "link/all", "link/branch", "split/strict", "merge/strict", "merge/tolerant"):
        assert s[key]["f1"] == 1.0, key
    assert s["tra_dag"] == 1.0 and abs(s["chota_dag"] - 1.0) < 1e-12


def test_a_lost_parent_breaks_the_strict_merge(toy):
    _, labels, lin = toy
    merge = next(e for e in lin.events() if e.kind == "merge")
    pred = Lineage(lin.nodes, lin.edges - {(merge.partners[0], merge.anchor)})
    s = summarize([evaluate(lin, labels, pred, labels)], EvalConfig(bootstrap=0))
    assert s["merge/strict"]["f1"] == 0.0
    assert s["link/all"]["f1"] < 1.0
    assert s["split/strict"]["f1"] == 1.0
