"""Lineage DAGs.

- a node is `(t, label)`: object `label` in frame `t`
- an edge points from a node to a node in a later frame
- in- and out-degree are unbounded, so a merge keeps all of its parents
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

Node = tuple[int, int]
Edge = tuple[Node, Node]


@dataclass(frozen=True)
class Event:
    kind: str  # "split" or "merge"
    t: int  # first frame after the event
    anchor: Node  # split: the parent; merge: the merged object
    partners: tuple[Node, ...]  # split: the children; merge: the parents


@dataclass
class Lineage:
    nodes: set[Node] = field(default_factory=set)
    edges: set[Edge] = field(default_factory=set)

    def __post_init__(self):
        missing = {n for e in self.edges for n in e} - self.nodes
        if missing:
            raise ValueError(f"{len(missing)} edge endpoints are not nodes, e.g. {sorted(missing)[:3]}")
        if any(v[0] <= u[0] for u, v in self.edges):
            raise ValueError("edges must point forward in time")

    def succ(self) -> dict[Node, list[Node]]:
        s = defaultdict(list)
        for u, v in self.edges:
            s[u].append(v)
        return s

    def pred(self) -> dict[Node, list[Node]]:
        p = defaultdict(list)
        for u, v in self.edges:
            p[v].append(u)
        return p

    def events(self) -> list[Event]:
        s, p = self.succ(), self.pred()
        out = [Event("split", min(c[0] for c in ch), n, tuple(sorted(ch))) for n, ch in s.items() if len(ch) >= 2]
        out += [Event("merge", n[0], n, tuple(sorted(pa))) for n, pa in p.items() if len(pa) >= 2]
        return sorted(out, key=lambda e: (e.t, e.kind, e.anchor))

    def edge_kind(self) -> dict[Edge, str]:
        """`cont` for one-to-one links, `branch` for links that touch a split or a merge."""
        s, p = self.succ(), self.pred()
        return {(u, v): "cont" if len(s[u]) == 1 and len(p[v]) == 1 else "branch" for u, v in self.edges}

    def restrict(self, t_lo: int, t_hi: int) -> Lineage:
        nodes = {n for n in self.nodes if t_lo <= n[0] <= t_hi}
        return Lineage(nodes, {(u, v) for u, v in self.edges if u in nodes and v in nodes})

    def relabel(self, mapping: dict[Node, int]) -> Lineage:
        """Replace every node label by `mapping[node]` (e.g. node -> track id)."""
        return Lineage({(n[0], mapping[n]) for n in self.nodes},
                       {((u[0], mapping[u]), (v[0], mapping[v])) for u, v in self.edges})

    def summary(self) -> dict:
        ev, s, p = self.events(), self.succ(), self.pred()
        ts = sorted({t for t, _ in self.nodes}) or [0]
        return {
            "objects": len(self.nodes),
            "links": len(self.edges),
            "splits": sum(e.kind == "split" for e in ev),
            "merges": sum(e.kind == "merge" for e in ev),
            "merges_3plus_parents": sum(e.kind == "merge" and len(e.partners) > 2 for e in ev),
            "births": sum(1 for n in self.nodes if n[0] > ts[0] and not p[n]),
            "deaths": sum(1 for n in self.nodes if n[0] < ts[-1] and not s[n]),
            "gap_links": sum(1 for u, v in self.edges if v[0] - u[0] > 1),
        }

    def to_tracks(self) -> tuple[dict[Node, int], pd.DataFrame]:
        """Cut the DAG into tracks, as the CTC does.

        - a track covers one object per frame over contiguous frames
        - a new track starts at every split, merge, birth and gap
        - returns node -> track id, and the table `id start end parent_1 ... parent_K`
        """
        s, p = self.succ(), self.pred()
        node_tid: dict[Node, int] = {}
        rows: dict[int, list] = {}
        for n in sorted(self.nodes):
            ps = p[n]
            if len(ps) == 1 and len(s[ps[0]]) == 1 and n[0] - ps[0][0] == 1:
                tid = node_tid[ps[0]]
                rows[tid][2] = n[0]
            else:
                tid = len(rows) + 1
                rows[tid] = [tid, n[0], n[0], sorted({node_tid[q] for q in ps})]
            node_tid[n] = tid
        k = max([len(r[3]) for r in rows.values()] + [1])
        table = pd.DataFrame([[i, t0, t1, *(par + [0] * (k - len(par)))] for i, t0, t1, par in rows.values()],
                             columns=["id", "start", "end"] + [f"parent_{j + 1}" for j in range(k)])
        return node_tid, table

    @classmethod
    def from_tracks(cls, table: pd.DataFrame, present: dict[int, set[int]] | None = None) -> Lineage:
        """Inverse of `to_tracks`. `present` (id -> frames) restricts nodes to frames that have a mask."""
        parents = [c for c in table.columns[3:]]
        end = dict(zip(table.iloc[:, 0], table.iloc[:, 2]))
        nodes, edges = set(), set()
        for row in table.itertuples(index=False):
            tid, t0, t1 = int(row[0]), int(row[1]), int(row[2])
            ts = sorted(present.get(tid, ())) if present is not None else list(range(t0, t1 + 1))
            nodes |= {(t, tid) for t in ts}
            edges |= {((a, tid), (b, tid)) for a, b in zip(ts[:-1], ts[1:])}
        for row in table.itertuples(index=False):
            tid, t0 = int(row[0]), int(row[1])
            for par in row[3:3 + len(parents)]:
                if par and par in end:
                    edges.add(((int(end[par]), int(par)), (t0, tid)))
        return cls(nodes, edges)


def read_track_table(path: Path) -> pd.DataFrame:
    """`man_track.txt` (one parent column) or `man_track_multi.txt` (K parent columns)."""
    df = pd.read_csv(path, sep=r"\s+", header=None)
    df.columns = ["id", "start", "end"] + [f"parent_{j + 1}" for j in range(df.shape[1] - 3)]
    return df


def write_track_table(table: pd.DataFrame, path: Path) -> None:
    table.to_csv(path, sep=" ", header=False, index=False)


def write_links(lineage: Lineage, path: Path) -> None:
    rows = sorted((u[0], u[1], v[0], v[1]) for u, v in lineage.edges)
    pd.DataFrame(rows, columns=["t_src", "id_src", "t_dst", "id_dst"]).to_csv(path, index=False)


def presence(labels: np.ndarray) -> dict[int, set[int]]:
    """label -> frames in which it has a mask, for a T x Z x Y x X label array."""
    out = defaultdict(set)
    for t in range(len(labels)):
        for lab in np.unique(labels[t]):
            if lab:
                out[int(lab)].add(t)
    return out
