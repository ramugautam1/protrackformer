from protrackformer.lineage import Lineage


def test_events(toy):
    _, _, lin = toy
    kinds = sorted((e.kind, e.t, len(e.partners)) for e in lin.events())
    assert kinds == [("merge", 3, 2), ("split", 5, 2)]


def test_tracks_round_trip(toy):
    _, _, lin = toy
    node_tid, table = lin.to_tracks()
    assert len(table) == 6
    back = Lineage.from_tracks(table)
    assert back.edges == {((u[0], node_tid[u]), (v[0], node_tid[v])) for u, v in lin.edges}


def test_three_parent_merge_and_gap():
    nodes = {(0, 1), (0, 2), (0, 3), (1, 4), (3, 4)}
    lin = Lineage(nodes, {((0, 1), (1, 4)), ((0, 2), (1, 4)), ((0, 3), (1, 4)), ((1, 4), (3, 4))})
    node_tid, table = lin.to_tracks()
    merged = table[table.id == node_tid[(1, 4)]].iloc[0]
    assert sorted(merged.iloc[3:].tolist()) == sorted(node_tid[(0, i)] for i in (1, 2, 3))
    assert node_tid[(3, 4)] != node_tid[(1, 4)]  # a gap starts a new track
    assert lin.summary()["gap_links"] == 1
