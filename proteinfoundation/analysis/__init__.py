# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary

"""
Crystallization Point Analysis Module

This module provides tools for analyzing the "crystallization point" in protein
structure generation - the timestep/layer where the model transitions from
global architecture search to local geometric locking.

Key components:
- CrystallizationTracker: Captures attention data during inference
- compute_logit_dominance, compute_attention_entropy, compute_spatial_alignment: Metric functions
- TrajectoryAnalyzer: Orchestrates metric computation across trajectories
- Visualization utilities for plotting results
"""

from .crystallization_hooks import (
    AttentionCapture,
    BiasAblationConfig,
    CrystallizationTracker,
)
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
from .trajectory_analyzer import (
    TrajectoryMetrics,
    SeqsepMetrics,
    ContactPrecisionMetrics,
    RegisterMetrics,
    TrajectoryAnalyzer,
)
from .visualization import (
    plot_crystallization_trajectory,
    plot_layer_heatmap,
    plot_attention_heatmap,
    plot_crystallization_summary,
    plot_per_head_trajectory,
    plot_seqsep_decomposition,
    plot_contact_precision_trajectory,
    plot_register_heatmap,
    plot_attention_decomposition_grid,
)

__all__ = [
    # Hooks
    "AttentionCapture",
    "BiasAblationConfig",
    "CrystallizationTracker",
    # Metrics
    "compute_logit_dominance",
    "compute_logit_dominance_centered",
    "compute_attention_entropy",
    "compute_spatial_alignment",
    "compute_gt_distance_matrix",
    "compute_contact_map",
    "compute_contact_precision",
    "compute_seqsep_metrics",
    # Analyzer
    "TrajectoryMetrics",
    "SeqsepMetrics",
    "ContactPrecisionMetrics",
    "RegisterMetrics",
    "TrajectoryAnalyzer",
    # Visualization
    "plot_crystallization_trajectory",
    "plot_layer_heatmap",
    "plot_attention_heatmap",
    "plot_crystallization_summary",
    "plot_per_head_trajectory",
    "plot_seqsep_decomposition",
    "plot_contact_precision_trajectory",
    "plot_register_heatmap",
    "plot_attention_decomposition_grid",
]
