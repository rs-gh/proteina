# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary

"""
Crystallization Point Analysis - Metric Functions

This module provides the three core metrics for analyzing crystallization:

1. Logit Dominance (R): R = ||B||_F / ||C||_F
   - Measures if geometric memory (B) dominates conditioning content (C)

2. Attention Entropy (H): H = -sum(p * log(p))
   - Measures sharpness of attention distribution
   - Drop in H indicates "crystallization"

3. Spatial Alignment (rho): Pearson correlation between attention and GT distances
   - Measures if attention is biologically accurate
"""

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor
import numpy as np


def compute_logit_dominance(
    qk_raw: Tensor,
    bias: Tensor,
    mask: Optional[Tensor] = None,
    eps: float = 1e-8,
) -> Tensor:
    """
    Compute Logit Dominance R = ||B||_F / ||C||_F per head.

    This metric measures whether the "Geometric Memory" (bias B) is louder
    than the "Input Instructions" (QK^T content C).

    Args:
        qk_raw: QK^T before scaling, shape [b, h, n, n]. This is C.
        bias: Pair bias B, shape [b, h, n, n].
        mask: Optional pair mask, shape [b, n, n] or [b, h, n, n].
        eps: Small constant for numerical stability.

    Returns:
        R values per batch and head, shape [b, h].
        Higher R means geometric bias dominates.
    """
    # Ensure same shape
    assert qk_raw.shape == bias.shape, f"Shape mismatch: {qk_raw.shape} vs {bias.shape}"

    # Apply mask if provided
    if mask is not None:
        if mask.dim() == 3:  # [b, n, n]
            mask = mask.unsqueeze(1)  # [b, 1, n, n]
        qk_raw = qk_raw * mask
        bias = bias * mask

    # Compute Frobenius norms over spatial dimensions (last two dims)
    # ||C||_F = sqrt(sum(c_ij^2))
    c_norm = torch.norm(qk_raw, p='fro', dim=(-2, -1))  # [b, h]
    b_norm = torch.norm(bias, p='fro', dim=(-2, -1))    # [b, h]

    # R = ||B||_F / ||C||_F
    R = b_norm / (c_norm + eps)

    return R


def compute_logit_dominance_centered(
    qk_raw: Tensor,
    bias: Tensor,
    mask: Optional[Tensor] = None,
    eps: float = 1e-8,
) -> Tensor:
    """
    Compute row-centered Logit Dominance R_c per head.

    Like compute_logit_dominance but subtracts the row mean before computing
    Frobenius norms.  This accounts for the softmax invariance to row-wise
    constant shifts: only the *within-row variance* of B and C actually
    influences the attention pattern.

    Args:
        qk_raw: QK^T before scaling, shape [b, h, n, n]. This is C.
        bias: Pair bias B, shape [b, h, n, n].
        mask: Optional pair mask, shape [b, n, n] or [b, h, n, n].
        eps: Small constant for numerical stability.

    Returns:
        R_c values per batch and head, shape [b, h].
    """
    assert qk_raw.shape == bias.shape, f"Shape mismatch: {qk_raw.shape} vs {bias.shape}"

    if mask is not None:
        if mask.dim() == 3:  # [b, n, n]
            mask = mask.unsqueeze(1)  # [b, 1, n, n]
        qk_raw = qk_raw * mask
        bias = bias * mask
        # Row mean over valid positions only
        row_counts = mask.sum(dim=-1, keepdim=True).clamp(min=1)  # [b, 1/h, n, 1]
    else:
        row_counts = qk_raw.shape[-1]

    # Subtract row means
    qk_centered = qk_raw - qk_raw.sum(dim=-1, keepdim=True) / row_counts
    bias_centered = bias - bias.sum(dim=-1, keepdim=True) / row_counts

    # Re-apply mask (centering can introduce non-zero at masked positions)
    if mask is not None:
        qk_centered = qk_centered * mask
        bias_centered = bias_centered * mask

    c_norm = torch.norm(qk_centered, p='fro', dim=(-2, -1))  # [b, h]
    b_norm = torch.norm(bias_centered, p='fro', dim=(-2, -1))  # [b, h]

    R_c = b_norm / (c_norm + eps)
    return R_c


def compute_attention_entropy(
    attn_weights: Tensor,
    mask: Optional[Tensor] = None,
    per_query: bool = False,
    eps: float = 1e-10,
) -> Tensor:
    """
    Compute Shannon entropy H = -sum(p * log(p)) of attention distribution.

    A drop in entropy over time indicates "crystallization" - the model
    stopping to weigh multiple possibilities and locking onto specific contacts.

    Args:
        attn_weights: Post-softmax attention, shape [b, h, n, n].
        mask: Optional sequence mask, shape [b, n] or pair mask [b, n, n].
        per_query: If True, return entropy per query position [b, h, n].
                   If False, return mean entropy per head [b, h].
        eps: Small constant for numerical stability.

    Returns:
        Entropy values:
        - If per_query=True: shape [b, h, n]
        - If per_query=False: shape [b, h]

        Lower entropy means sharper (more "crystallized") attention.
    """
    # Clamp for numerical stability in log
    p = attn_weights.clamp(min=eps)
    log_p = torch.log(p)

    # Entropy per query position: H_i = -sum_j(p_ij * log(p_ij))
    H_per_query = -torch.sum(p * log_p, dim=-1)  # [b, h, n]

    if per_query:
        return H_per_query

    # Average over query positions
    if mask is not None:
        if mask.dim() == 2:  # [b, n] sequence mask
            # Create mask for valid query positions
            seq_mask = mask.unsqueeze(1).float()  # [b, 1, n]
            H_per_query = H_per_query * seq_mask
            H = H_per_query.sum(dim=-1) / (seq_mask.sum(dim=-1) + eps)  # [b, h]
        elif mask.dim() == 3:  # [b, n, n] pair mask
            # Use diagonal as sequence mask (valid residues)
            seq_mask = torch.diagonal(mask, dim1=-2, dim2=-1).unsqueeze(1).float()  # [b, 1, n]
            H_per_query = H_per_query * seq_mask
            H = H_per_query.sum(dim=-1) / (seq_mask.sum(dim=-1) + eps)  # [b, h]
        else:
            H = H_per_query.mean(dim=-1)  # [b, h]
    else:
        H = H_per_query.mean(dim=-1)  # [b, h]

    return H


def compute_gt_distance_matrix(
    coords: Tensor,
    mask: Optional[Tensor] = None,
) -> Tensor:
    """
    Compute ground truth CA distance matrix from coordinates.

    Args:
        coords: CA coordinates, shape [b, n, 3] in nm or Angstroms.
        mask: Optional sequence mask, shape [b, n].

    Returns:
        Distance matrix, shape [b, n, n].
        Distances in same units as input coords.
    """
    # Pairwise distances: d_ij = ||x_i - x_j||
    # coords[:, :, None, :] has shape [b, n, 1, 3]
    # coords[:, None, :, :] has shape [b, 1, n, 3]
    diff = coords[:, :, None, :] - coords[:, None, :, :]  # [b, n, n, 3]
    dist = torch.norm(diff, dim=-1)  # [b, n, n]

    if mask is not None:
        # Zero out distances for invalid residue pairs
        pair_mask = mask[:, :, None] * mask[:, None, :]  # [b, n, n]
        dist = dist * pair_mask

    return dist


def compute_spatial_alignment(
    attn_weights: Tensor,
    gt_distance_matrix: Tensor,
    mask: Optional[Tensor] = None,
    per_head: bool = True,
    invert_correlation: bool = True,
) -> Tensor:
    """
    Compute 2D Pearson correlation between attention and GT distance matrix.

    This measures if "pointy" attention is biologically accurate - whether
    positions with high attention correspond to positions that are actually
    close in 3D space.

    Args:
        attn_weights: Post-softmax attention, shape [b, h, n, n].
        gt_distance_matrix: Ground truth distance matrix, shape [b, n, n].
        mask: Optional pair mask, shape [b, n, n].
        per_head: If True, return correlation per head [b, h].
                  If False, return mean correlation per batch [b].
        invert_correlation: If True, negate correlation so positive = good alignment.
                           (Higher attention should correlate with LOWER distance)

    Returns:
        Pearson correlation (rho):
        - If per_head=True: shape [b, h]
        - If per_head=False: shape [b]

        Positive rho (after inversion) means attention aligns with true contacts.
    """
    b, h, n, _ = attn_weights.shape
    device = attn_weights.device

    # Create mask for valid pairs
    if mask is not None:
        valid_mask = mask.bool()  # [b, n, n]
    else:
        valid_mask = torch.ones(b, n, n, dtype=torch.bool, device=device)

    # Compute correlation per batch and head
    correlations = torch.zeros(b, h, device=device)

    for batch_idx in range(b):
        # Get valid pair indices for this batch
        valid = valid_mask[batch_idx]  # [n, n]

        # Flatten ground truth distances for valid pairs
        gt_flat = gt_distance_matrix[batch_idx][valid].float()  # [num_valid]

        if gt_flat.numel() < 3:
            # Not enough points for correlation
            continue

        for head_idx in range(h):
            # Flatten attention for valid pairs
            attn_flat = attn_weights[batch_idx, head_idx][valid].float()  # [num_valid]

            # Compute Pearson correlation
            # rho = cov(X, Y) / (std(X) * std(Y))
            attn_mean = attn_flat.mean()
            gt_mean = gt_flat.mean()

            attn_centered = attn_flat - attn_mean
            gt_centered = gt_flat - gt_mean

            covariance = (attn_centered * gt_centered).mean()
            attn_std = attn_centered.std()
            gt_std = gt_centered.std()

            if attn_std > 1e-8 and gt_std > 1e-8:
                rho = covariance / (attn_std * gt_std)
            else:
                rho = torch.tensor(0.0, device=device)

            # Negate so positive = good alignment
            # (high attention should correlate with LOW distance)
            if invert_correlation:
                rho = -rho

            correlations[batch_idx, head_idx] = rho

    if not per_head:
        correlations = correlations.mean(dim=-1)  # [b]

    return correlations


def _seqsep_range_mask(n: int, lo: int, hi: int, device) -> Tensor:
    """Create [n, n] boolean mask for pairs where lo <= |i-j| < hi."""
    i_idx = torch.arange(n, device=device).unsqueeze(1)
    j_idx = torch.arange(n, device=device).unsqueeze(0)
    sep = (i_idx - j_idx).abs()
    return (sep >= lo) & (sep < hi)


def compute_contact_map(
    dist_matrix: Tensor,
    threshold: float = 0.8,
    min_seqsep: int = 6,
) -> Tensor:
    """
    Compute binary contact map from a CA/Cbeta distance matrix.

    A contact is defined as dist < threshold with |i-j| >= min_seqsep,
    following the convention of Rao et al. (2020) "Transformer protein language
    models are unsupervised structure learners."

    Args:
        dist_matrix: Pairwise distances, shape [b, n, n].
                     Units should match threshold (default nm, 0.8nm = 8 Angstroms).
        threshold: Distance threshold for contacts (default 0.8 nm = 8 Angstroms).
        min_seqsep: Minimum sequence separation to avoid counting local backbone contacts.

    Returns:
        Binary contact map, shape [b, n, n].
    """
    b, n, _ = dist_matrix.shape
    device = dist_matrix.device

    contacts = (dist_matrix < threshold) & (dist_matrix > 0)

    seqsep_mask = _seqsep_range_mask(n, min_seqsep, n, device)  # |i-j| >= min_seqsep
    contacts = contacts & seqsep_mask.unsqueeze(0)

    return contacts


def compute_contact_precision(
    attn_matrix: Tensor,
    contact_map: Tensor,
    k: Optional[int] = None,
    min_seqsep: int = 6,
    apply_apc: bool = True,
    eps: float = 1e-8,
) -> Tensor:
    """
    Compute Precision@k contact prediction from an attention matrix.

    Inspired by Rao et al. (2020) "Transformer protein language models are
    unsupervised structure learners," which showed that PLM attention heads
    predict contacts with Precision@L/5 competitive with co-evolutionary methods.

    We use this to compare:
      - Full attention (C+B): how much do combined scores predict contacts?
      - C-only attention (softmax(QK^T * scale)): does content score learn geometry?
      - B-only attention (softmax(B)): does the geometric bias alone predict contacts?

    Args:
        attn_matrix: Attention weights (post-softmax), shape [b, h, n, n] or [b, n, n].
                     Can be full attention, C-only, or B-only (caller computes these).
        contact_map: Binary contact map (1=contact), shape [b, n, n].
                     Use compute_contact_map() to generate from distance matrix.
        k: Number of top pairs to evaluate. If None, uses n//5 (Precision@L/5).
        min_seqsep: Minimum sequence separation for evaluated pairs (avoid local backbone).
        apply_apc: If True, apply Average Product Correction (APC) before ranking.
                   APC removes background marginal distributions, improving precision.
        eps: Small constant for numerical stability.

    Returns:
        Precision@k per batch and head, shape [b, h].
        Higher values mean the attention better predicts 3D contacts.
    """
    if attn_matrix.dim() == 3:
        attn_matrix = attn_matrix.unsqueeze(1)  # [b, n, n] → [b, 1, n, n]

    b, h, n, _ = attn_matrix.shape
    device = attn_matrix.device

    if k is None:
        k = max(1, n // 5)

    # Upper triangle with min_seqsep: only evaluate pairs (i, j) where j >= i + min_seqsep
    j_idx = torch.arange(n, device=device).unsqueeze(0)
    i_idx = torch.arange(n, device=device).unsqueeze(1)
    eval_mask = (j_idx - i_idx) >= min_seqsep  # [n, n]

    precision = torch.zeros(b, h, device=device)

    for bi in range(b):
        contacts_flat = contact_map[bi][eval_mask].float()  # [num_valid]
        num_valid = eval_mask.sum().item()

        if num_valid < k:
            continue

        for hi in range(h):
            A = attn_matrix[bi, hi].float()  # [n, n]

            # Symmetrize: high attention in either direction counts as a contact prediction
            A_sym = (A + A.T) / 2

            # APC correction: A_apc[i,j] = A[i,j] - row_mean[i] * col_mean[j] / total_mean
            # Removes background marginal signal, improving contact-specific precision
            if apply_apc:
                row_mean = A_sym.mean(dim=-1, keepdim=True)   # [n, 1]
                col_mean = A_sym.mean(dim=-2, keepdim=True)   # [1, n]
                total_mean = A_sym.mean()
                A_sym = A_sym - (row_mean * col_mean) / (total_mean + eps)

            # Extract scores for valid (upper-triangle, min-seqsep) pairs
            scores = A_sym[eval_mask]  # [num_valid]

            # Top-k pairs by score
            actual_k = min(k, num_valid)
            _, top_idx = scores.topk(actual_k)

            # Precision: fraction of top-k that are contacts
            precision[bi, hi] = contacts_flat[top_idx].mean()

    return precision


def compute_seqsep_metrics(
    qk_raw: Tensor,
    bias: Tensor,
    attn_weights: Tensor,
    gt_distance_matrix: Optional[Tensor] = None,
    mask: Optional[Tensor] = None,
    bins: Optional[List[Tuple[int, int]]] = None,
    eps: float = 1e-8,
) -> Dict[str, Dict[str, Optional[Tensor]]]:
    """
    Compute R, H, rho decomposed by sequence separation |i-j|.

    This directly tests the proposal's central hypothesis: does the geometric
    bias B matter more for long-range interactions than short-range ones?
    Expected result: R is higher in the long-range bin (geometry is more important
    when sequence context alone cannot determine proximity).

    Args:
        qk_raw: QK^T before scaling, shape [b, h, n, n].
        bias: Pair bias B, shape [b, h, n, n].
        attn_weights: Post-softmax attention, shape [b, h, n, n].
        gt_distance_matrix: GT distance matrix, shape [b, n, n]. Optional.
        mask: Optional pair mask [b, n, n] or sequence mask [b, n].
        bins: List of (lo, hi) sequence separation bins (hi is exclusive).
              Default: [(1, 7), (7, 24), (24, 10000)] for local/medium/long-range.
        eps: Small constant for stability.

    Returns:
        Dict mapping bin_label → {'R': Tensor[b,h], 'R_centered': Tensor[b,h],
            'H': Tensor[b,h], 'rho': Tensor[b,h] or None}
    """
    if bins is None:
        bins = [(1, 7), (7, 24), (24, 10000)]
        bin_labels = ['local (1-6)', 'medium (7-23)', 'long (>=24)']
    else:
        bin_labels = [f'sep{lo}-{hi-1}' for lo, hi in bins]

    b, h, n, _ = qk_raw.shape
    device = qk_raw.device

    results: Dict[str, Dict[str, Optional[Tensor]]] = {}

    for (lo, hi), label in zip(bins, bin_labels):
        sep_mask_2d = _seqsep_range_mask(n, lo, min(hi, n), device)  # [n, n]

        # Build [b, h, n, n] combined mask for R and H computations
        sep_mask_bhnn = sep_mask_2d.float().unsqueeze(0).unsqueeze(0)  # [1, 1, n, n]

        if mask is not None:
            if mask.dim() == 3:  # [b, n, n] pair mask
                pair_mask_bhnn = mask.unsqueeze(1) * sep_mask_bhnn  # [b, 1, n, n]
            else:  # [b, n] sequence mask
                seq_pair = mask[:, :, None] * mask[:, None, :]  # [b, n, n]
                pair_mask_bhnn = seq_pair.unsqueeze(1).float() * sep_mask_bhnn  # [b, 1, n, n]
        else:
            pair_mask_bhnn = sep_mask_bhnn.expand(b, 1, n, n)

        # R: Frobenius norm ratio, restricted to this seqsep bin
        qk_m = qk_raw * pair_mask_bhnn
        bias_m = bias * pair_mask_bhnn
        c_norm = torch.norm(qk_m, p='fro', dim=(-2, -1))   # [b, h]
        b_norm = torch.norm(bias_m, p='fro', dim=(-2, -1))  # [b, h]
        R_bin = b_norm / (c_norm + eps)

        # R_centered: row-centered variant (softmax-aware)
        row_counts = pair_mask_bhnn.sum(dim=-1, keepdim=True).clamp(min=1)
        qk_c = qk_m - qk_m.sum(dim=-1, keepdim=True) / row_counts
        bias_c = bias_m - bias_m.sum(dim=-1, keepdim=True) / row_counts
        qk_c = qk_c * pair_mask_bhnn
        bias_c = bias_c * pair_mask_bhnn
        c_norm_c = torch.norm(qk_c, p='fro', dim=(-2, -1))
        b_norm_c = torch.norm(bias_c, p='fro', dim=(-2, -1))
        R_bin_centered = b_norm_c / (c_norm_c + eps)

        # H: entropy of attention re-normalized within this seqsep bin
        # This is H(attention | key in bin), measuring how "decided" the model
        # is about which contacts within this range to attend to.
        attn_m = attn_weights * pair_mask_bhnn  # [b, h, n, n]
        row_sum = attn_m.sum(dim=-1, keepdim=True)  # [b, h, n, 1]
        attn_renorm = attn_m / (row_sum + eps)
        # 0 * log(0) = 0 by convention; handle via masking
        log_p = torch.where(
            attn_renorm > eps,
            torch.log(attn_renorm.clamp(min=eps)),
            torch.zeros_like(attn_renorm),
        )
        H_per_q = -(attn_renorm * log_p).sum(dim=-1)  # [b, h, n]
        # Only average over query rows that have at least one valid key in this bin
        valid_queries = (row_sum.squeeze(-1) > eps).float()  # [b, h, n]
        H_bin = (H_per_q * valid_queries).sum(dim=-1) / (valid_queries.sum(dim=-1) + eps)

        # rho: spatial alignment restricted to pairs in this seqsep bin
        rho_bin = None
        if gt_distance_matrix is not None:
            sep_b = sep_mask_2d.unsqueeze(0).expand(b, n, n)  # [b, n, n]
            if mask is not None:
                if mask.dim() == 3:
                    spatial_mask = (sep_b & mask.bool()).float()
                else:
                    seq_pair = (mask[:, :, None] * mask[:, None, :]).bool()
                    spatial_mask = (sep_b & seq_pair).float()
            else:
                spatial_mask = sep_b.float()
            rho_bin = compute_spatial_alignment(attn_weights, gt_distance_matrix, spatial_mask)

        results[label] = {'R': R_bin, 'R_centered': R_bin_centered, 'H': H_bin, 'rho': rho_bin}

    return results


def compute_all_metrics(
    qk_raw: Tensor,
    bias: Tensor,
    attn_weights: Tensor,
    gt_distance_matrix: Optional[Tensor] = None,
    mask: Optional[Tensor] = None,
) -> Tuple[Tensor, Tensor, Optional[Tensor]]:
    """
    Compute all three crystallization metrics.

    Args:
        qk_raw: QK^T before scaling, shape [b, h, n, n].
        bias: Pair bias B, shape [b, h, n, n].
        attn_weights: Post-softmax attention, shape [b, h, n, n].
        gt_distance_matrix: Ground truth distance matrix, shape [b, n, n]. Optional.
        mask: Optional pair mask, shape [b, n, n].

    Returns:
        Tuple of (R, H, rho) where:
        - R: Logit dominance, shape [b, h]
        - H: Attention entropy, shape [b, h]
        - rho: Spatial alignment, shape [b, h] or None if gt_distance_matrix not provided
    """
    R = compute_logit_dominance(qk_raw, bias, mask)
    H = compute_attention_entropy(attn_weights, mask)

    if gt_distance_matrix is not None:
        rho = compute_spatial_alignment(attn_weights, gt_distance_matrix, mask)
    else:
        rho = None

    return R, H, rho
