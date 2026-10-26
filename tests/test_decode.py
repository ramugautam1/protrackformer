import numpy as np

from protrackformer.decode import solve_ilp, to_lineage
from protrackformer.graph import MAX_DEG, build_graph
from protrackformer.inference import Prediction


def _ideal(g, lin, labels):
    """Probabilities that a perfect network would give: links right, degrees right."""
    from protrackformer.graph import label_graph
    label_graph(g, lin, labels)
    eps = 0.05
    link = np.where(g.edge_y > 0, 0.7, 0.2)
    p_in = np.full((len(g.node_t), MAX_DEG + 1), eps)
    p_out = p_in.copy()
    p_in[np.arange(len(g.node_t)), g.node_indeg] = 1 - 3 * eps
    p_out[np.arange(len(g.node_t)), g.node_outdeg] = 1 - 3 * eps
    return Prediction(link, p_in, p_out, None)


def test_ilp_keeps_both_parents_of_a_merge(toy):
    image, labels, lin = toy
    g = build_graph(image, labels)
    p = _ideal(g, lin, labels)
    sel, keep = solve_ilp(g, p.link, p.p_in, p.p_out)
    assert sel.all()
    out, _ = to_lineage(g, sel, keep)
    assert sorted((e.kind, len(e.partners)) for e in out.events()) == [("merge", 2), ("split", 2)]
    assert keep.sum() == len(lin.edges)


def test_ilp_follows_degrees_over_weak_links(toy):
    image, labels, lin = toy
    g = build_graph(image, labels)
    p = _ideal(g, lin, labels)
    p.link[g.edge_y > 0] = 0.45  # every true link alone would be dropped by a 0.5 threshold
    _, keep = solve_ilp(g, p.link, p.p_in, p.p_out, lam=2.0)
    assert np.array_equal(keep, g.edge_y > 0)
