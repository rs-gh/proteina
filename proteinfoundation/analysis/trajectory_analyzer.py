# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary

"""
Crystallization Point Analysis - Trajectory Analyzer

This module provides orchestration for computing crystallization metrics
across the entire flow-matching trajectory.

Key classes:
- TrajectoryMetrics: Stores computed metrics as numpy arrays
- TrajectoryAnalyzer: Computes all metrics from captured attention data
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from pathlib import Path

import numpy as np
import torch
from torch import Tensor

from .crystallization_hooks import CrystallizationTracker
import torch.nn.functional as F

from .crystallization_metrics import (
    compute_logit_dominance,
    compute_logit_dominance_centered,
    compute_attention_entropy,
    compute_spatial_alignment,
    compute_gt_distance_matrix,
    compute_contact_map,
    compute_contact_precision,
    compute_seqsep_metrics,
)


@dataclass
class TrajectoryMetrics:
    """
    Stores computed metrics across the trajectory.

    All arrays have shape [T, L, H] where:
    - T = number of timesteps
    - L = number of layers
    - H = number of heads

    Attributes:
        timesteps: Array of timestep values (t in [0, 1]), shape [T]
        timestep_indices: Array of timestep indices, shape [T]
        logit_dominance: R = ||B||_F / ||C||_F, shape [T, L, H]
        logit_dominance_centered: Row-centered R_c, shape [T, L, H], or None
        entropy: Shannon entropy of attention, shape [T, L, H]
        spatial_alignment: Pearson correlation with GT distances, shape [T, L, H] or None
        spatial_alignment_label: Human-readable description of what rho is measured against,
                                 e.g. "vs. external GT" or "vs. final predicted structure"
        num_layers: Number of transformer layers
        num_heads: Number of attention heads
        protein_length: Length of the protein
    """
    timesteps: np.ndarray
    timestep_indices: np.ndarray
    logit_dominance: np.ndarray
    logit_dominance_centered: Optional[np.ndarray]
    entropy: np.ndarray
    spatial_alignment: Optional[np.ndarray]
    spatial_alignment_label: str
    num_layers: int
    num_heads: int
    protein_length: int

    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        d = {
            'timesteps': self.timesteps,
            'timestep_indices': self.timestep_indices,
            'logit_dominance': self.logit_dominance,
            'entropy': self.entropy,
            'num_layers': self.num_layers,
            'num_heads': self.num_heads,
            'protein_length': self.protein_length,
        }
        if self.logit_dominance_centered is not None:
            d['logit_dominance_centered'] = self.logit_dominance_centered
        if self.spatial_alignment is not None:
            d['spatial_alignment'] = self.spatial_alignment
        d['spatial_alignment_label'] = self.spatial_alignment_label
        return d

    def save(self, path: str):
        """Save metrics to npz file."""
        np.savez(path, **self.to_dict())

    @classmethod
    def load(cls, path: str) -> 'TrajectoryMetrics':
        """Load metrics from npz file."""
        data = np.load(path)
        return cls(
            timesteps=data['timesteps'],
            timestep_indices=data['timestep_indices'],
            logit_dominance=data['logit_dominance'],
            logit_dominance_centered=data.get('logit_dominance_centered'),
            entropy=data['entropy'],
            spatial_alignment=data.get('spatial_alignment'),
            spatial_alignment_label=str(data.get('spatial_alignment_label', 'vs. ground truth')),
            num_layers=int(data['num_layers']),
            num_heads=int(data['num_heads']),
            protein_length=int(data['protein_length']),
        )

    def get_crystallization_point(
        self,
        metric: str = 'entropy',
        layer: Optional[int] = None,
        head: Optional[int] = None,
        threshold_percentile: float = 10.0,
    ) -> Tuple[float, int]:
        """
        Find the crystallization point - where a metric crosses a threshold.

        Args:
            metric: Which metric to use ('entropy', 'logit_dominance', 'spatial_alignment')
            layer: Specific layer to analyze, or None for mean across layers
            head: Specific head to analyze, or None for mean across heads
            threshold_percentile: Percentile of metric range to use as threshold

        Returns:
            Tuple of (timestep_value, timestep_index) where crystallization occurs
        """
        if metric == 'entropy':
            data = self.entropy
        elif metric == 'logit_dominance':
            data = self.logit_dominance
        elif metric == 'spatial_alignment':
            if self.spatial_alignment is None:
                raise ValueError("spatial_alignment not available")
            data = self.spatial_alignment
        else:
            raise ValueError(f"Unknown metric: {metric}")

        # Select layer/head or average
        if layer is not None:
            data = data[:, layer, :]
        else:
            data = data.mean(axis=1)

        if head is not None:
            data = data[:, head]
        else:
            data = data.mean(axis=-1)

        # Find threshold
        data_range = data.max() - data.min()
        if metric == 'entropy':
            # For entropy, crystallization is when it drops below threshold
            threshold = data.max() - (threshold_percentile / 100.0) * data_range
            crossing_idx = np.argmax(data < threshold)
        else:
            # For R and rho, crystallization is when they rise above threshold
            threshold = data.min() + (threshold_percentile / 100.0) * data_range
            crossing_idx = np.argmax(data > threshold)

        return self.timesteps[crossing_idx], self.timestep_indices[crossing_idx]

    def summary(self) -> str:
        """Generate a summary string of the metrics."""
        lines = [
            f"TrajectoryMetrics Summary",
            f"=" * 40,
            f"Protein length: {self.protein_length}",
            f"Timesteps: {len(self.timesteps)} (t={self.timesteps[0]:.3f} to {self.timesteps[-1]:.3f})",
            f"Layers: {self.num_layers}",
            f"Heads: {self.num_heads}",
            f"",
            f"Logit Dominance (R = ||B||/||C||):",
            f"  Early (t~0): {self.logit_dominance[0].mean():.3f} +/- {self.logit_dominance[0].std():.3f}",
            f"  Late (t~1):  {self.logit_dominance[-1].mean():.3f} +/- {self.logit_dominance[-1].std():.3f}",
        ]

        if self.logit_dominance_centered is not None:
            lines.extend([
                f"",
                f"Logit Dominance Centered (R_c, row-centered):",
                f"  Early (t~0): {self.logit_dominance_centered[0].mean():.3f} +/- {self.logit_dominance_centered[0].std():.3f}",
                f"  Late (t~1):  {self.logit_dominance_centered[-1].mean():.3f} +/- {self.logit_dominance_centered[-1].std():.3f}",
            ])

        lines.extend([
            f"",
            f"Attention Entropy (H / log n, normalized):",
            f"  Early (t~0): {self.entropy[0].mean() / np.log(self.protein_length):.3f} +/- {self.entropy[0].std() / np.log(self.protein_length):.3f}",
            f"  Late (t~1):  {self.entropy[-1].mean() / np.log(self.protein_length):.3f} +/- {self.entropy[-1].std() / np.log(self.protein_length):.3f}",
        ])

        if self.spatial_alignment is not None:
            lines.extend([
                f"",
                f"Spatial Alignment (rho, {self.spatial_alignment_label}):",
                f"  Early (t~0): {self.spatial_alignment[0].mean():.3f} +/- {self.spatial_alignment[0].std():.3f}",
                f"  Late (t~1):  {self.spatial_alignment[-1].mean():.3f} +/- {self.spatial_alignment[-1].std():.3f}",
            ])

        return "\n".join(lines)


@dataclass
class SeqsepMetrics:
    """
    Crystallization metrics decomposed by sequence separation |i-j|.

    Attributes:
        timesteps: Timestep values, shape [T].
        bin_labels: Human-readable labels for each bin, e.g. ['local (1-6)', ...].
        logit_dominance: R per bin, shape [num_bins, T, L, H].
        logit_dominance_centered: Row-centered R_c per bin, shape [num_bins, T, L, H], or None.
        entropy: H per bin, shape [num_bins, T, L, H].
        spatial_alignment: rho per bin, shape [num_bins, T, L, H], or None.
        num_layers: Number of transformer layers.
        num_heads: Number of attention heads.
        protein_length: Length of the analyzed protein.
    """
    timesteps: np.ndarray
    bin_labels: List[str]
    logit_dominance: np.ndarray
    logit_dominance_centered: Optional[np.ndarray]
    entropy: np.ndarray
    spatial_alignment: Optional[np.ndarray]
    num_layers: int
    num_heads: int
    protein_length: int

    def save(self, path: str):
        """Save to npz file."""
        d: Dict = {
            'timesteps': self.timesteps,
            'bin_labels': np.array(self.bin_labels),
            'logit_dominance': self.logit_dominance,
            'entropy': self.entropy,
            'num_layers': self.num_layers,
            'num_heads': self.num_heads,
            'protein_length': self.protein_length,
        }
        if self.logit_dominance_centered is not None:
            d['logit_dominance_centered'] = self.logit_dominance_centered
        if self.spatial_alignment is not None:
            d['spatial_alignment'] = self.spatial_alignment
        np.savez(path, **d)

    @classmethod
    def load(cls, path: str) -> 'SeqsepMetrics':
        """Load from npz file."""
        data = np.load(path, allow_pickle=True)
        return cls(
            timesteps=data['timesteps'],
            bin_labels=list(data['bin_labels']),
            logit_dominance=data['logit_dominance'],
            logit_dominance_centered=data.get('logit_dominance_centered'),
            entropy=data['entropy'],
            spatial_alignment=data['spatial_alignment'] if 'spatial_alignment' in data else None,
            num_layers=int(data['num_layers']),
            num_heads=int(data['num_heads']),
            protein_length=int(data['protein_length']),
        )


@dataclass
class ContactPrecisionMetrics:
    """
    Precision@k contact prediction from attention, C-only, and B-only matrices.

    Inspired by Rao et al. (2020): PLM attention heads predict structural contacts.
    Here we decompose this into contributions from the content score C = QK^T and
    the geometric bias B, measuring which component drives contact prediction at
    each point in the denoising trajectory.

    Attributes:
        timesteps: Timestep values, shape [T].
        k: Number of top pairs evaluated (Precision@k).
        precision_full: From full attention (C+B), shape [T, L, H].
        precision_b_only: From softmax(B) only, shape [T, L, H].
        precision_c_only: From softmax(QK^T * scale) only, shape [T, L, H].
        num_layers, num_heads, protein_length: Model/protein dimensions.
    """
    timesteps: np.ndarray
    k: int
    precision_full: np.ndarray
    precision_b_only: np.ndarray
    precision_c_only: np.ndarray
    num_layers: int
    num_heads: int
    protein_length: int

    def save(self, path: str):
        """Save to npz file."""
        np.savez(path,
            timesteps=self.timesteps,
            k=self.k,
            precision_full=self.precision_full,
            precision_b_only=self.precision_b_only,
            precision_c_only=self.precision_c_only,
            num_layers=self.num_layers,
            num_heads=self.num_heads,
            protein_length=self.protein_length,
        )

    @classmethod
    def load(cls, path: str) -> 'ContactPrecisionMetrics':
        """Load from npz file."""
        data = np.load(path)
        return cls(
            timesteps=data['timesteps'],
            k=int(data['k']),
            precision_full=data['precision_full'],
            precision_b_only=data['precision_b_only'],
            precision_c_only=data['precision_c_only'],
            num_layers=int(data['num_layers']),
            num_heads=int(data['num_heads']),
            protein_length=int(data['protein_length']),
        )


@dataclass
class RegisterMetrics:
    """
    Register token attention analysis across the denoising trajectory.

    Based on Darcet et al. (2023) "Vision Transformers Need Registers":
    register tokens act as attention sinks for global computation. Here we
    measure how much attention residue tokens send to registers, and how
    this evolves across layers and denoising timesteps.

    The Mix-Compress-Refine hypothesis predicts register attention should
    peak in the 'Compress' phase (middle layers) and decrease in the
    'Refine' phase as attention sharpens to specific residue contacts.

    Attributes:
        timesteps: Timestep values, shape [T].
        register_attn_fraction: Mean fraction of attention to registers per head,
                                shape [T, L, H]. Higher = more attention to registers.
        num_layers, num_heads, num_registers, protein_length: Dimensions.
    """
    timesteps: np.ndarray
    register_attn_fraction: np.ndarray
    num_layers: int
    num_heads: int
    num_registers: int
    protein_length: int

    def save(self, path: str):
        """Save to npz file."""
        np.savez(path,
            timesteps=self.timesteps,
            register_attn_fraction=self.register_attn_fraction,
            num_layers=self.num_layers,
            num_heads=self.num_heads,
            num_registers=self.num_registers,
            protein_length=self.protein_length,
        )

    @classmethod
    def load(cls, path: str) -> 'RegisterMetrics':
        """Load from npz file."""
        data = np.load(path)
        return cls(
            timesteps=data['timesteps'],
            register_attn_fraction=data['register_attn_fraction'],
            num_layers=int(data['num_layers']),
            num_heads=int(data['num_heads']),
            num_registers=int(data['num_registers']),
            protein_length=int(data['protein_length']),
        )


class TrajectoryAnalyzer:
    """
    Analyzes crystallization metrics across the flow-matching trajectory.

    Usage:
        tracker = CrystallizationTracker()
        # ... run generation with tracker ...

        analyzer = TrajectoryAnalyzer(tracker, num_layers=15, num_heads=8)
        metrics = analyzer.compute_metrics(gt_coords, mask)

        print(metrics.summary())
        metrics.save("crystallization_metrics.npz")
    """

    def __init__(
        self,
        tracker: CrystallizationTracker,
        num_layers: int,
        num_heads: int,
        num_registers: int = 0,
    ):
        """
        Initialize the analyzer.

        Args:
            tracker: CrystallizationTracker with captured attention data
            num_layers: Number of transformer layers in the model
            num_heads: Number of attention heads per layer
            num_registers: Number of register tokens prepended to the sequence.
                          These are stripped from captured attention before analysis.
        """
        self.tracker = tracker
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.num_registers = num_registers

    def compute_metrics(
        self,
        gt_coords: Optional[Tensor] = None,
        mask: Optional[Tensor] = None,
        batch_idx: int = 0,
        spatial_alignment_label: str = "vs. ground truth",
    ) -> TrajectoryMetrics:
        """
        Compute all three metrics across the captured trajectory.

        Args:
            gt_coords: Ground truth CA coordinates, shape [b, n, 3].
                       Required for spatial alignment metric.
            mask: Sequence mask, shape [b, n]. Optional.
            batch_idx: Which batch element to analyze (default 0).
            spatial_alignment_label: Description of what rho is measured against,
                                     shown in plot titles and summary output.
                                     E.g. "vs. external GT" or "vs. final predicted structure".

        Returns:
            TrajectoryMetrics object with all computed metrics.
        """
        timestep_indices = self.tracker.get_timestep_indices()
        num_timesteps = len(timestep_indices)

        if num_timesteps == 0:
            raise ValueError("No captures found in tracker")

        # Get protein length from first capture
        first_capture = self.tracker.get_capture(timestep_indices[0], 0)
        if first_capture is None or first_capture.attn_weights is None:
            raise ValueError("No attention data in first capture")

        protein_length = first_capture.attn_weights.shape[-1] - self.num_registers

        # Compute GT distance matrix if coordinates provided
        # Move mask and coords to CPU since captures are stored on CPU
        if gt_coords is not None:
            gt_coords = gt_coords.cpu()
        if mask is not None:
            mask = mask.cpu()

        gt_dist = None
        if gt_coords is not None:
            gt_dist = compute_gt_distance_matrix(gt_coords, mask)

        # Create pair mask if sequence mask provided
        pair_mask = None
        if mask is not None:
            pair_mask = mask[:, :, None] * mask[:, None, :]  # [b, n, n]

        # Initialize output arrays
        timesteps = np.zeros(num_timesteps)
        R_all = np.zeros((num_timesteps, self.num_layers, self.num_heads))
        Rc_all = np.zeros((num_timesteps, self.num_layers, self.num_heads))
        H_all = np.zeros((num_timesteps, self.num_layers, self.num_heads))
        rho_all = np.zeros((num_timesteps, self.num_layers, self.num_heads)) if gt_dist is not None else None

        # Compute metrics for each timestep and layer
        for t_idx, timestep_idx in enumerate(timestep_indices):
            # Get timestep value from first layer capture
            first_layer_capture = self.tracker.get_capture(timestep_idx, 0)
            if first_layer_capture is not None:
                timesteps[t_idx] = first_layer_capture.timestep or 0.0

            for layer_idx in range(self.num_layers):
                capture = self.tracker.get_capture(timestep_idx, layer_idx)
                if capture is None:
                    continue

                # Get tensors for this batch element
                qk_raw = capture.qk_raw
                bias = capture.bias
                attn = capture.attn_weights

                if qk_raw is None or bias is None or attn is None:
                    continue

                # Strip register tokens if present.
                # Registers are prepended to the sequence, so captured attention
                # has shape [b, h, n+r, n+r] where r = num_registers.
                r = self.num_registers
                if r > 0:
                    qk_raw = qk_raw[:, :, r:, r:]
                    bias = bias[:, :, r:, r:]
                    attn = attn[:, :, r:, r:]
                    # Re-normalize attention after stripping registers
                    attn = attn / (attn.sum(dim=-1, keepdim=True) + 1e-10)

                # Select batch element and ensure on same device
                qk_raw_b = qk_raw[batch_idx:batch_idx+1]
                bias_b = bias[batch_idx:batch_idx+1]
                attn_b = attn[batch_idx:batch_idx+1]

                pair_mask_b = pair_mask[batch_idx:batch_idx+1] if pair_mask is not None else None
                gt_dist_b = gt_dist[batch_idx:batch_idx+1] if gt_dist is not None else None

                # Metric 1: Logit Dominance
                R = compute_logit_dominance(qk_raw_b, bias_b, pair_mask_b)
                R_all[t_idx, layer_idx] = R.squeeze(0).cpu().numpy()

                # Metric 1b: Row-centered Logit Dominance
                Rc = compute_logit_dominance_centered(qk_raw_b, bias_b, pair_mask_b)
                Rc_all[t_idx, layer_idx] = Rc.squeeze(0).cpu().numpy()

                # Metric 2: Entropy
                H = compute_attention_entropy(attn_b, pair_mask_b)
                H_all[t_idx, layer_idx] = H.squeeze(0).cpu().numpy()

                # Metric 3: Spatial Alignment
                if gt_dist_b is not None:
                    rho = compute_spatial_alignment(attn_b, gt_dist_b, pair_mask_b)
                    rho_all[t_idx, layer_idx] = rho.squeeze(0).cpu().numpy()

        return TrajectoryMetrics(
            timesteps=timesteps,
            timestep_indices=np.array(timestep_indices),
            logit_dominance=R_all,
            logit_dominance_centered=Rc_all,
            entropy=H_all,
            spatial_alignment=rho_all,
            spatial_alignment_label=spatial_alignment_label,
            num_layers=self.num_layers,
            num_heads=self.num_heads,
            protein_length=protein_length,
        )

    def compute_seqsep_trajectory(
        self,
        gt_coords: Optional[Tensor] = None,
        mask: Optional[Tensor] = None,
        batch_idx: int = 0,
        bins: Optional[List[Tuple[int, int]]] = None,
    ) -> SeqsepMetrics:
        """
        Compute R, H, rho decomposed by sequence separation across the trajectory.

        This tests whether geometric bias B matters more for long-range vs.
        local interactions — the central hypothesis of the project.

        Args:
            gt_coords: CA coordinates [b, n, 3] for spatial alignment. Optional.
            mask: Sequence mask [b, n]. Optional.
            batch_idx: Batch element to analyze.
            bins: Sequence separation bins as (lo, hi) pairs (hi exclusive).
                  Default: [(1,7), (7,24), (24,10000)] = local/medium/long-range.

        Returns:
            SeqsepMetrics with R, H, rho per bin across layers and timesteps.
        """
        # Keep track of whether bins were user-specified.
        # When None, compute_seqsep_metrics will also default to the same bins+labels.
        bins_for_call = bins
        if bins is None:
            bins = [(1, 7), (7, 24), (24, 10000)]
            bin_labels = ['local (1-6)', 'medium (7-23)', 'long (>=24)']
        else:
            bin_labels = [f'sep{lo}-{hi-1}' for lo, hi in bins]

        timestep_indices = self.tracker.get_timestep_indices()
        num_timesteps = len(timestep_indices)
        num_bins = len(bins)

        if gt_coords is not None:
            gt_coords = gt_coords.cpu()
        if mask is not None:
            mask = mask.cpu()

        gt_dist = None
        if gt_coords is not None:
            gt_dist = compute_gt_distance_matrix(gt_coords, mask)

        pair_mask = None
        if mask is not None:
            pair_mask = mask[:, :, None] * mask[:, None, :]

        timesteps = np.zeros(num_timesteps)
        R_all = np.zeros((num_bins, num_timesteps, self.num_layers, self.num_heads))
        Rc_all = np.zeros((num_bins, num_timesteps, self.num_layers, self.num_heads))
        H_all = np.zeros((num_bins, num_timesteps, self.num_layers, self.num_heads))
        rho_all = np.zeros((num_bins, num_timesteps, self.num_layers, self.num_heads)) if gt_dist is not None else None

        for t_idx, timestep_idx in enumerate(timestep_indices):
            first_layer_capture = self.tracker.get_capture(timestep_idx, 0)
            if first_layer_capture is not None:
                timesteps[t_idx] = first_layer_capture.timestep or 0.0

            for layer_idx in range(self.num_layers):
                capture = self.tracker.get_capture(timestep_idx, layer_idx)
                if capture is None:
                    continue

                qk_raw = capture.qk_raw
                bias = capture.bias
                attn = capture.attn_weights

                if qk_raw is None or bias is None or attn is None:
                    continue

                r = self.num_registers
                if r > 0:
                    qk_raw = qk_raw[:, :, r:, r:]
                    bias = bias[:, :, r:, r:]
                    attn = attn[:, :, r:, r:]
                    attn = attn / (attn.sum(dim=-1, keepdim=True) + 1e-10)

                qk_b = qk_raw[batch_idx:batch_idx+1]
                bias_b = bias[batch_idx:batch_idx+1]
                attn_b = attn[batch_idx:batch_idx+1]
                pair_mask_b = pair_mask[batch_idx:batch_idx+1] if pair_mask is not None else None
                gt_dist_b = gt_dist[batch_idx:batch_idx+1] if gt_dist is not None else None

                bin_results = compute_seqsep_metrics(
                    qk_b, bias_b, attn_b,
                    gt_distance_matrix=gt_dist_b,
                    mask=pair_mask_b,
                    bins=bins_for_call,
                )

                for bin_idx, label in enumerate(bin_labels):
                    br = bin_results[label]
                    R_all[bin_idx, t_idx, layer_idx] = br['R'].squeeze(0).cpu().numpy()
                    Rc_all[bin_idx, t_idx, layer_idx] = br['R_centered'].squeeze(0).cpu().numpy()
                    H_all[bin_idx, t_idx, layer_idx] = br['H'].squeeze(0).cpu().numpy()
                    if rho_all is not None and br['rho'] is not None:
                        rho_all[bin_idx, t_idx, layer_idx] = br['rho'].squeeze(0).cpu().numpy()

        first_capture = self.tracker.get_capture(timestep_indices[0], 0)
        protein_length = first_capture.attn_weights.shape[-1] - self.num_registers

        return SeqsepMetrics(
            timesteps=timesteps,
            bin_labels=bin_labels,
            logit_dominance=R_all,
            logit_dominance_centered=Rc_all,
            entropy=H_all,
            spatial_alignment=rho_all,
            num_layers=self.num_layers,
            num_heads=self.num_heads,
            protein_length=protein_length,
        )

    def compute_contact_precision_trajectory(
        self,
        contact_map: Tensor,
        head_dim: int = 64,
        k: Optional[int] = None,
        batch_idx: int = 0,
        min_seqsep: int = 6,
        apply_apc: bool = True,
    ) -> ContactPrecisionMetrics:
        """
        Compute Precision@k contact prediction across the trajectory for three conditions.

        Computes precision for:
        1. Full attention (post-softmax, C+B)
        2. B-only: softmax(bias) — geometric bias alone as contact predictor
        3. C-only: softmax(QK^T * scale) — content score alone as contact predictor

        This directly measures which component (geometric bias vs. content score)
        drives contact prediction at each layer and timestep.

        Args:
            contact_map: Binary contact map [b, n, n] (1=contact, 0=no contact).
                         Generate via compute_contact_map(gt_dist_matrix).
            head_dim: Attention head dimension for scaling QK^T (default 64).
                      Scale = head_dim^(-0.5). Applied when computing C-only attention.
            k: Number of top pairs for Precision@k. Default: n//5 (Precision@L/5).
            batch_idx: Batch element to analyze.
            min_seqsep: Minimum sequence separation for evaluated pairs.
            apply_apc: Whether to apply Average Product Correction before ranking.

        Returns:
            ContactPrecisionMetrics with precision arrays of shape [T, L, H].
        """
        scale = head_dim ** -0.5

        timestep_indices = self.tracker.get_timestep_indices()
        num_timesteps = len(timestep_indices)

        contact_map = contact_map.cpu()
        contact_map_b = contact_map[batch_idx:batch_idx+1]

        first_capture = self.tracker.get_capture(timestep_indices[0], 0)
        protein_length = first_capture.attn_weights.shape[-1] - self.num_registers

        if k is None:
            k = max(1, protein_length // 5)

        timesteps = np.zeros(num_timesteps)
        prec_full = np.zeros((num_timesteps, self.num_layers, self.num_heads))
        prec_b = np.zeros((num_timesteps, self.num_layers, self.num_heads))
        prec_c = np.zeros((num_timesteps, self.num_layers, self.num_heads))

        for t_idx, timestep_idx in enumerate(timestep_indices):
            first_layer_capture = self.tracker.get_capture(timestep_idx, 0)
            if first_layer_capture is not None:
                timesteps[t_idx] = first_layer_capture.timestep or 0.0

            for layer_idx in range(self.num_layers):
                capture = self.tracker.get_capture(timestep_idx, layer_idx)
                if capture is None:
                    continue

                qk_raw = capture.qk_raw
                bias = capture.bias
                attn = capture.attn_weights

                if qk_raw is None or bias is None or attn is None:
                    continue

                r = self.num_registers
                if r > 0:
                    qk_raw = qk_raw[:, :, r:, r:]
                    bias = bias[:, :, r:, r:]
                    attn = attn[:, :, r:, r:]
                    attn = attn / (attn.sum(dim=-1, keepdim=True) + 1e-10)

                qk_b = qk_raw[batch_idx:batch_idx+1].float()
                bias_b = bias[batch_idx:batch_idx+1].float()
                attn_b = attn[batch_idx:batch_idx+1].float()

                # C-only: softmax(QK^T * scale)
                attn_c_only = F.softmax(qk_b * scale, dim=-1)
                # B-only: softmax(B)
                attn_b_only = F.softmax(bias_b, dim=-1)

                pf = compute_contact_precision(attn_b, contact_map_b, k, min_seqsep, apply_apc)
                pb = compute_contact_precision(attn_b_only, contact_map_b, k, min_seqsep, apply_apc)
                pc = compute_contact_precision(attn_c_only, contact_map_b, k, min_seqsep, apply_apc)

                prec_full[t_idx, layer_idx] = pf.squeeze(0).cpu().numpy()
                prec_b[t_idx, layer_idx] = pb.squeeze(0).cpu().numpy()
                prec_c[t_idx, layer_idx] = pc.squeeze(0).cpu().numpy()

        return ContactPrecisionMetrics(
            timesteps=timesteps,
            k=k,
            precision_full=prec_full,
            precision_b_only=prec_b,
            precision_c_only=prec_c,
            num_layers=self.num_layers,
            num_heads=self.num_heads,
            protein_length=protein_length,
        )

    def compute_register_metrics(
        self,
        batch_idx: int = 0,
    ) -> RegisterMetrics:
        """
        Compute register token attention fraction across the trajectory.

        Based on Darcet et al. (2023) "Vision Transformers Need Registers":
        register tokens act as attention sinks for global computation.
        We measure how much residue attention is directed to registers.

        A value of 0.1 means residues send 10% of their attention to registers.
        If registers are attention sinks, this should be high in middle layers
        ('Compress' phase) and lower in early/late layers ('Mix'/'Refine').

        Args:
            batch_idx: Batch element to analyze.

        Returns:
            RegisterMetrics with register_attn_fraction of shape [T, L, H].

        Raises:
            ValueError: If num_registers == 0 (no registers to analyze).
        """
        if self.num_registers == 0:
            raise ValueError("num_registers=0, no register tokens to analyze. "
                             "Pass num_registers to TrajectoryAnalyzer constructor.")

        r = self.num_registers
        timestep_indices = self.tracker.get_timestep_indices()
        num_timesteps = len(timestep_indices)

        first_capture = self.tracker.get_capture(timestep_indices[0], 0)
        protein_length = first_capture.attn_weights.shape[-1] - r

        timesteps = np.zeros(num_timesteps)
        reg_frac = np.zeros((num_timesteps, self.num_layers, self.num_heads))

        for t_idx, timestep_idx in enumerate(timestep_indices):
            first_layer_capture = self.tracker.get_capture(timestep_idx, 0)
            if first_layer_capture is not None:
                timesteps[t_idx] = first_layer_capture.timestep or 0.0

            for layer_idx in range(self.num_layers):
                capture = self.tracker.get_capture(timestep_idx, layer_idx)
                if capture is None or capture.attn_weights is None:
                    continue

                # Full attention INCLUDING registers: [b, h, n+r, n+r]
                attn_full = capture.attn_weights[batch_idx]  # [h, n+r, n+r]

                # Attention from residue rows to register columns:
                # rows r: are residue query positions
                # cols :r are register key positions
                # attn_full[h, r:, :r] has shape [h, n, r]
                attn_to_reg = attn_full[:, r:, :r]  # [h, n, r]

                # Sum over register keys → fraction of attention each residue sends to registers
                frac_per_residue = attn_to_reg.sum(dim=-1)  # [h, n]

                # Mean over residue query positions
                frac_mean = frac_per_residue.mean(dim=-1)  # [h]

                reg_frac[t_idx, layer_idx] = frac_mean.cpu().numpy()

        return RegisterMetrics(
            timesteps=timesteps,
            register_attn_fraction=reg_frac,
            num_layers=self.num_layers,
            num_heads=self.num_heads,
            num_registers=r,
            protein_length=protein_length,
        )

    def compute_metrics_streaming(
        self,
        gt_coords: Optional[Tensor] = None,
        mask: Optional[Tensor] = None,
        batch_idx: int = 0,
        clear_after: bool = True,
    ) -> TrajectoryMetrics:
        """
        Compute metrics in streaming fashion, clearing captures as we go.

        This is more memory efficient for long trajectories.

        Args:
            gt_coords: Ground truth CA coordinates, shape [b, n, 3].
            mask: Sequence mask, shape [b, n]. Optional.
            batch_idx: Which batch element to analyze.
            clear_after: If True, clear tracker captures after computing.

        Returns:
            TrajectoryMetrics object with all computed metrics.
        """
        metrics = self.compute_metrics(gt_coords, mask, batch_idx)

        if clear_after:
            self.tracker.clear()

        return metrics
