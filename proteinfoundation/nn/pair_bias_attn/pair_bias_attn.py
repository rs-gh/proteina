# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.


from typing import Optional, TYPE_CHECKING

import torch
from einops import rearrange
from torch import Tensor, einsum, nn

if TYPE_CHECKING:
    from proteinfoundation.analysis.crystallization_hooks import AttentionCapture


def exists(val) -> bool:
    """returns whether val is not none"""
    return val is not None


def default(x, y):
    """returns x if it exists, otherwise y"""
    return x if exists(x) else y


max_neg_value = lambda x: torch.finfo(x.dtype).min


class PairBiasAttention(nn.Module):
    """
    Scalar Feature masked attention with pair bias and gating.
    Code modified from
    https://github.com/MattMcPartlon/protein-docking/blob/main/protein_learning/network/modules/node_block.py
    """

    def __init__(
        self,
        node_dim: int,
        dim_head: int,
        heads: int,
        bias: bool,
        dim_out: int,
        qkln: bool,
        pair_dim: Optional[int] = None,
        **kawrgs  # noqa
    ):
        super().__init__()
        inner_dim = dim_head * heads
        self.node_dim, self.pair_dim = node_dim, pair_dim
        self.heads, self.scale = heads, dim_head**-0.5
        self.to_qkv = nn.Linear(node_dim, inner_dim * 3, bias=bias)
        self.to_g = nn.Linear(node_dim, inner_dim)
        self.to_out_node = nn.Linear(inner_dim, default(dim_out, node_dim))
        self.node_norm = nn.LayerNorm(node_dim)
        self.q_layer_norm = nn.LayerNorm(inner_dim) if qkln else nn.Identity()
        self.k_layer_norm = nn.LayerNorm(inner_dim) if qkln else nn.Identity()
        if exists(pair_dim):
            self.to_bias = nn.Linear(pair_dim, heads, bias=False)
            self.pair_norm = nn.LayerNorm(pair_dim)
        else:
            self.to_bias, self.pair_norm = None, None

    def forward(
        self,
        node_feats: Tensor,
        pair_feats: Optional[Tensor],
        mask: Optional[Tensor],
        capture: Optional["AttentionCapture"] = None,
        ablate_bias: str = '',
    ) -> Tensor:
        """Multi-head scalar Attention Layer

        :param node_feats: scalar features of shape (b,n,d_s)
        :param pair_feats: pair features of shape (b,n,n,d_e)
        :param mask: boolean tensor of node adjacencies
        :param capture: optional AttentionCapture to store intermediates for analysis
        :param ablate_bias: '' = no ablation, 'zero' = set B=0, 'random' = random noise
        :return:
        """
        assert exists(self.to_bias) or not exists(pair_feats)
        node_feats, h = self.node_norm(node_feats), self.heads
        pair_feats = self.pair_norm(pair_feats) if exists(pair_feats) else None
        q, k, v = self.to_qkv(node_feats).chunk(3, dim=-1)
        q = self.q_layer_norm(q)
        k = self.k_layer_norm(k)
        g = self.to_g(node_feats)
        b = (
            rearrange(self.to_bias(pair_feats), "b ... h -> b h ...")
            if exists(pair_feats)
            else 0
        )
        q, k, v, g = map(
            lambda t: rearrange(t, "b ... (h d) -> b h ... d", h=h), (q, k, v, g)
        )
        attn_feats = self._attn(q, k, v, b, mask, capture, ablate_bias=ablate_bias)
        attn_feats = rearrange(
            torch.sigmoid(g) * attn_feats, "b h n d -> b n (h d)", h=h
        )
        return self.to_out_node(attn_feats)

    def _attn(
        self,
        q,
        k,
        v,
        b,
        mask: Optional[Tensor],
        capture: Optional["AttentionCapture"] = None,
        ablate_bias: str = '',
    ) -> Tensor:
        """Perform attention update

        Args:
            q: Query tensor, shape [b, h, n, d]
            k: Key tensor, shape [b, h, n, d]
            v: Value tensor, shape [b, h, n, d]
            b: Pair bias, shape [b, h, n, n] or scalar 0
            mask: Optional pair mask, shape [b, n, n]
            capture: Optional AttentionCapture to store intermediates for analysis
            ablate_bias: '' = no ablation, 'zero' = set B=0, 'random' = replace
                         B with magnitude-matched Gaussian noise

        Returns:
            Attention output, shape [b, h, n, d]
        """
        # Compute QK^T (before scaling) for capture
        qk_raw = einsum("b h i d, b h j d -> b h i j", q, k)

        # Apply scaling
        sim = qk_raw * self.scale

        if exists(mask):
            mask = rearrange(mask, "b i j -> b () i j")
            sim = sim.masked_fill(~mask, max_neg_value(sim))

        # Ablate pair bias if requested (causal intervention)
        if ablate_bias == 'zero':
            bias_for_attn = 0
        elif ablate_bias == 'random' and isinstance(b, Tensor):
            bias_for_attn = torch.randn_like(b) * b.std()
        else:
            bias_for_attn = b

        # Compute attention weights
        attn = torch.softmax(sim + bias_for_attn, dim=-1)

        # Capture intermediates if requested (always capture original bias, not ablated)
        if capture is not None:
            capture.qk_raw = qk_raw
            capture.bias = b if isinstance(b, Tensor) else None
            capture.attn_weights = attn

        return einsum("b h i j, b h j d -> b h i d", attn, v)
