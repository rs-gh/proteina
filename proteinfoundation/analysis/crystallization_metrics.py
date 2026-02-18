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

from typing import Optional, Tuple

import torch
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
