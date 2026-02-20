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
    CrystallizationTracker,
)
from .crystallization_metrics import (
    compute_logit_dominance,
    compute_attention_entropy,
    compute_spatial_alignment,
    compute_gt_distance_matrix,
)
from .trajectory_analyzer import (
    TrajectoryMetrics,
    TrajectoryAnalyzer,
)
from .visualization import (
    plot_crystallization_trajectory,
    plot_layer_heatmap,
    plot_attention_heatmap,
    plot_crystallization_summary,
)

__all__ = [
    # Hooks
    "AttentionCapture",
    "CrystallizationTracker",
    # Metrics
    "compute_logit_dominance",
    "compute_attention_entropy",
    "compute_spatial_alignment",
    "compute_gt_distance_matrix",
    # Analyzer
    "TrajectoryMetrics",
    "TrajectoryAnalyzer",
    # Visualization
    "plot_crystallization_trajectory",
    "plot_layer_heatmap",
    "plot_attention_heatmap",
    "plot_crystallization_summary",
]
