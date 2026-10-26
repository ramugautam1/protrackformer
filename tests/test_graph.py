import numpy as np

from protrackformer.graph import CANDIDATE_FEATURES, LF, LINK_FEATURES, OBJECT_FEATURES, build_graph, label_graph
from protrackformer.windows import augment, window_arrays


def test_instance_graph_contains_every_link(toy):
    image, labels, lin = toy
    g = build_graph(image, labels)
    label_graph(g, lin, labels)
    assert g.node_feat.shape == (len(lin.nodes), len(OBJECT_FEATURES))
    assert g.edge_feat.shape[1] == len(LINK_FEATURES)
    assert g.stats["link_recall"] == 1.0
    assert g.edge_y.sum() == len(lin.edges)
    assert g.node_indeg.max() == 2 and g.node_outdeg.max() == 2


def test_candidates_of_a_fused_foreground(toy):
    image, labels, lin = toy
    g = build_graph(image, (labels > 0).astype(np.uint8), candidates=True, instances=False)
    label_graph(g, lin, labels)
    assert g.node_feat.shape[1] == len(OBJECT_FEATURES) + len(CANDIDATE_FEATURES)
    assert g.node_exists.sum() >= len(lin.nodes) - 2
    for a, b in g.conflicts:
        assert g.node_t[a] == g.node_t[b] and (g.mask(a) & g.mask(b)).any()


class Draws:
    """Stand-in random generator: no flips or transposes, then time reversal, no intensity jitter."""

    def __init__(self):
        self.values = iter([0.9, 0.9, 0.9, 0.1])

    def random(self):
        return next(self.values)

    def normal(self, *args):
        return 0.0


def test_time_reversal_swaps_splits_and_merges(toy):
    image, labels, lin = toy
    g = build_graph(image, labels)
    label_graph(g, lin, labels)
    w = window_arrays(g, 0, 8)
    indeg, outdeg, dx = w["indeg"].copy(), w["outdeg"].copy(), w["edge_feat"][:, LF["dx"]].copy()
    r = augment(w, 8, Draws())
    assert r["reversed"]
    assert np.array_equal(r["indeg"], outdeg) and np.array_equal(r["outdeg"], indeg)
    assert np.allclose(r["edge_feat"][:, LF["dx"]], -dx)
