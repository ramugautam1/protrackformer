"""ProTrackFormer network.

- object tokens: measurements + frame embedding, then attention restricted to nearby objects,
  with a learned bias from relative position, distance and time offset
- link tokens: `MLP([h_src, h_dst, link measurements])`, then link-to-link attention among links that
  share a source, share a target or form a chain, with a learned bias per relation
- heads: an independent probability per link (no softmax over parents), in- and out-degree
  distributions over {0, 1, 2, 3+}, and, for candidate segments, an existence probability
"""

from __future__ import annotations

import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F

warnings.filterwarnings("ignore", message="index_reduce")

N_RELATIONS = 5  # same source, same target, chain, reverse chain, self


def mlp(d_in: int, d_hidden: int, d_out: int, n: int = 2) -> nn.Sequential:
    layers, d = [], d_in
    for _ in range(n - 1):
        layers += [nn.Linear(d, d_hidden), nn.GELU()]
        d = d_hidden
    return nn.Sequential(*layers, nn.Linear(d, d_out))


class BiasedAttention(nn.Module):
    """Pre-norm transformer layer whose attention takes an additive (heads, N, N) bias."""

    def __init__(self, d: int, heads: int):
        super().__init__()
        self.h = heads
        self.n1, self.n2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.qkv = nn.Linear(d, 3 * d)
        self.o = nn.Linear(d, d)
        self.ff = mlp(d, 4 * d, d)

    def forward(self, x: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
        n, d = x.shape
        q, k, v = self.qkv(self.n1(x)).view(n, 3, self.h, d // self.h).permute(1, 2, 0, 3)
        a = F.scaled_dot_product_attention(q, k, v, attn_mask=bias)
        x = x + self.o(a.transpose(0, 1).reshape(n, d))
        return x + self.ff(self.n2(x))


class ProTrackFormer(nn.Module):
    def __init__(self, node_dim: int, edge_dim: int, d: int = 128, heads: int = 4, node_layers: int = 2,
                 edge_layers: int = 3, window: int = 6, att_radius: float = 24.0, n_deg: int = 4,
                 existence: bool = False):
        super().__init__()
        self.cfg = dict(node_dim=node_dim, edge_dim=edge_dim, d=d, heads=heads, node_layers=node_layers,
                        edge_layers=edge_layers, window=window, att_radius=att_radius, n_deg=n_deg, existence=existence)
        self.att_radius = att_radius
        self.register_buffer("nf_mu", torch.zeros(node_dim))
        self.register_buffer("nf_sd", torch.ones(node_dim))
        self.register_buffer("ef_mu", torch.zeros(edge_dim))
        self.register_buffer("ef_sd", torch.ones(edge_dim))
        self.node_in = mlp(node_dim, d, d)
        self.t_emb = nn.Embedding(window, d)
        self.rel_bias = mlp(5, 32, heads)
        self.node_layers = nn.ModuleList([BiasedAttention(d, heads) for _ in range(node_layers)])
        self.edge_in = mlp(edge_dim, d, d)
        self.edge_tok = mlp(3 * d, d, d)
        self.rel_emb = nn.Embedding(N_RELATIONS, heads)
        self.edge_layers = nn.ModuleList([BiasedAttention(d, heads) for _ in range(edge_layers)])
        self.node_upd = mlp(5 * d, d, d)
        self.edge_head = mlp(3 * d, d, 1)
        self.in_head = mlp(d, d, n_deg)
        self.out_head = mlp(d, d, n_deg)
        self.exist_head = mlp(d, d, 1) if existence else None

    def set_normalisation(self, nf_mu, nf_sd, ef_mu, ef_sd) -> None:
        for name, v in (("nf_mu", nf_mu), ("nf_sd", nf_sd), ("ef_mu", ef_mu), ("ef_sd", ef_sd)):
            getattr(self, name).copy_(torch.as_tensor(v, dtype=torch.float32))

    def forward(self, node_feat, node_pos, node_trel, src, dst, edge_feat):
        n, e_count = node_feat.shape[0], edge_feat.shape[0]
        h = self.node_in((node_feat - self.nf_mu) / self.nf_sd) + self.t_emb(node_trel)
        rel = node_pos[None] - node_pos[:, None]
        dist = rel.norm(dim=-1, keepdim=True)
        dtt = (node_trel[None] - node_trel[:, None]).float()[..., None]
        bias = self.rel_bias(torch.cat([rel / 10, dist / 10, dtt], -1)).permute(2, 0, 1)
        bias = bias.masked_fill((dist[..., 0] > self.att_radius)[None], float("-inf"))
        for layer in self.node_layers:
            h = layer(h, bias)

        e = self.edge_tok(torch.cat([h[src], h[dst], self.edge_in((edge_feat - self.ef_mu) / self.ef_sd)], -1))
        if e_count:
            relation = torch.full((e_count, e_count), -1, dtype=torch.long, device=e.device)
            relation[src[:, None] == src[None]] = 0
            relation[dst[:, None] == dst[None]] = 1
            relation[dst[:, None] == src[None]] = 2
            relation[src[:, None] == dst[None]] = 3
            relation.fill_diagonal_(4)
            ebias = self.rel_emb(relation.clamp(min=0)).permute(2, 0, 1)
            ebias = ebias.masked_fill((relation < 0)[None], float("-inf"))
            for layer in self.edge_layers:
                e = layer(e, ebias)

        def pool(idx, how):
            out = torch.zeros(n, e.shape[1], device=e.device, dtype=e.dtype)
            if how == "mean":
                cnt = torch.zeros(n, 1, device=e.device, dtype=e.dtype).index_add_(
                    0, idx, torch.ones(e_count, 1, device=e.device, dtype=e.dtype))
                return out.index_add_(0, idx, e) / cnt.clamp(min=1)
            return out.index_reduce_(0, idx, e, "amax", include_self=False)

        h = h + self.node_upd(torch.cat([h, pool(dst, "mean"), pool(dst, "amax"), pool(src, "mean"),
                                         pool(src, "amax")], -1))
        link = self.edge_head(torch.cat([e, h[src], h[dst]], -1)).squeeze(-1)
        exist = self.exist_head(h).squeeze(-1) if self.exist_head is not None else None
        return link, self.in_head(h), self.out_head(h), exist
