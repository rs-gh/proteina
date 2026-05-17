# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.


from typing import Optional

import torch
from einops import rearrange
from torch import Tensor, einsum, nn


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
        use_sdpa: bool = True,
        **kawrgs  # noqa
    ):
        super().__init__()
        inner_dim = dim_head * heads
        self.node_dim, self.pair_dim = node_dim, pair_dim
        self.heads, self.scale = heads, dim_head**-0.5
        self.use_sdpa = use_sdpa
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
    ) -> Tensor:
        """Multi-head scalar Attention Layer

        :param node_feats: scalar features of shape (b,n,d_s)
        :param pair_feats: pair features of shape (b,n,n,d_e)
        :param mask: boolean tensor of node adjacencies
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
        attn_feats = self._attn_sdpa(q, k, v, b, mask) if self.use_sdpa else self._attn(q, k, v, b, mask)
        attn_feats = rearrange(
            torch.sigmoid(g) * attn_feats, "b h n d -> b n (h d)", h=h
        )
        return self.to_out_node(attn_feats)

    def _attn(self, q, k, v, b, mask: Optional[Tensor]) -> Tensor:
        """Manual attention: Q @ K^T * scale + bias -> softmax -> @ V."""
        sim = einsum("b h i d, b h j d -> b h i j", q, k) * self.scale
        if exists(mask):
            mask = rearrange(mask, "b i j -> b () i j")
            sim = sim.masked_fill(~mask, max_neg_value(sim))
        attn = torch.softmax(sim + b, dim=-1)
        return einsum("b h i j, b h j d -> b h i d", attn, v)

    @torch.compiler.disable  # noqa: see comment below
    def _attn_sdpa(self, q, k, v, b, mask: Optional[Tensor]) -> Tensor:
        """SDPA attention: fused kernels via F.scaled_dot_product_attention.

        The ``attn_mask`` parameter is an additive float bias applied before
        softmax - same semantics as the manual ``sim + b`` path.

        ``@torch.compiler.disable`` makes this method opaque to torch.compile
        (added 2026-05-16). Without it, inductor's reduce-overhead cudagraph
        pipeline was tracing the ``attn_bias.contiguous()`` call away (or
        baking trace-time strides into the cudagraph) and at runtime — when
        a partial batch hit a graph specialised on a different batch — the
        EFFICIENT_ATTENTION kernel would raise
            RuntimeError: (*bias): last dimension must be contiguous
        Disabling compile on this call keeps the SDPA fused kernel speedup
        (cuDNN/CUDA-level fusion is independent of torch.compile) while
        guaranteeing the .contiguous() runs at runtime as written.
        """
        attn_bias = b if not isinstance(b, int) else None

        if exists(mask):
            mask_bias = rearrange(mask, "b i j -> b () i j")
            mask_bias = torch.zeros_like(mask_bias, dtype=q.dtype).masked_fill(
                ~mask_bias, max_neg_value(q)
            )
            attn_bias = attn_bias + mask_bias if exists(attn_bias) else mask_bias

        # Fused SDPA kernels (FLASH / EFFICIENT) require stride(-1) == 1 on
        # attn_mask. The upstream rearrange("b ... h -> b h ...") leaves the
        # last dim with stride = H; .contiguous() materializes a fresh buffer
        # matching the (B, H, N, N) logical layout so EFFICIENT_ATTENTION
        # dispatches instead of falling back to MATH.
        # Validated 2026-04-18 via hpc-scripts/proteina/bench/diagnose_sdpa.py:
        # MATH(strided) vs EFFICIENT(contiguous) on identical inputs agree to
        # worst 5.0e-3 relative (below bf16 eps ~7.8e-3) across out/gq/gk/gv/gb.
        # Timing at B=6,H=8,N=512,D=64,bf16: 2.11 ms (MATH) -> 0.51 ms (EFFICIENT),
        # ~4x speedup for this attention call.
        if attn_bias is not None:
            attn_bias = attn_bias.contiguous()

        return torch.nn.functional.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_bias, scale=self.scale,
        )
