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
from .crystallization_metrics import (
    compute_logit_dominance,
    compute_attention_entropy,
    compute_spatial_alignment,
    compute_gt_distance_matrix,
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
        entropy: Shannon entropy of attention, shape [T, L, H]
        spatial_alignment: Pearson correlation with GT distances, shape [T, L, H] or None
        num_layers: Number of transformer layers
        num_heads: Number of attention heads
        protein_length: Length of the protein
    """
    timesteps: np.ndarray
    timestep_indices: np.ndarray
    logit_dominance: np.ndarray
    entropy: np.ndarray
    spatial_alignment: Optional[np.ndarray]
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
        if self.spatial_alignment is not None:
            d['spatial_alignment'] = self.spatial_alignment
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
            entropy=data['entropy'],
            spatial_alignment=data.get('spatial_alignment'),
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
            f"",
            f"Attention Entropy (H):",
            f"  Early (t~0): {self.entropy[0].mean():.3f} +/- {self.entropy[0].std():.3f}",
            f"  Late (t~1):  {self.entropy[-1].mean():.3f} +/- {self.entropy[-1].std():.3f}",
        ]

        if self.spatial_alignment is not None:
            lines.extend([
                f"",
                f"Spatial Alignment (rho):",
                f"  Early (t~0): {self.spatial_alignment[0].mean():.3f} +/- {self.spatial_alignment[0].std():.3f}",
                f"  Late (t~1):  {self.spatial_alignment[-1].mean():.3f} +/- {self.spatial_alignment[-1].std():.3f}",
            ])

        return "\n".join(lines)


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
    ):
        """
        Initialize the analyzer.

        Args:
            tracker: CrystallizationTracker with captured attention data
            num_layers: Number of transformer layers in the model
            num_heads: Number of attention heads per layer
        """
        self.tracker = tracker
        self.num_layers = num_layers
        self.num_heads = num_heads

    def compute_metrics(
        self,
        gt_coords: Optional[Tensor] = None,
        mask: Optional[Tensor] = None,
        batch_idx: int = 0,
    ) -> TrajectoryMetrics:
        """
        Compute all three metrics across the captured trajectory.

        Args:
            gt_coords: Ground truth CA coordinates, shape [b, n, 3].
                       Required for spatial alignment metric.
            mask: Sequence mask, shape [b, n]. Optional.
            batch_idx: Which batch element to analyze (default 0).

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

        protein_length = first_capture.attn_weights.shape[-1]

        # Compute GT distance matrix if coordinates provided
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

                # Select batch element and ensure on same device
                qk_raw_b = qk_raw[batch_idx:batch_idx+1]
                bias_b = bias[batch_idx:batch_idx+1]
                attn_b = attn[batch_idx:batch_idx+1]

                pair_mask_b = pair_mask[batch_idx:batch_idx+1] if pair_mask is not None else None
                gt_dist_b = gt_dist[batch_idx:batch_idx+1] if gt_dist is not None else None

                # Metric 1: Logit Dominance
                R = compute_logit_dominance(qk_raw_b, bias_b, pair_mask_b)
                R_all[t_idx, layer_idx] = R.squeeze(0).cpu().numpy()

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
            entropy=H_all,
            spatial_alignment=rho_all,
            num_layers=self.num_layers,
            num_heads=self.num_heads,
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
