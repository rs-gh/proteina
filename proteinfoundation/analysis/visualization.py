# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary

"""
Crystallization Point Analysis - Visualization

This module provides plotting utilities for visualizing crystallization metrics
across the flow-matching trajectory.

Key functions:
- plot_crystallization_trajectory: 3-panel plot of R, H, rho vs timestep
- plot_layer_heatmap: Heatmap of metrics across layers and timesteps
- plot_attention_heatmap: Side-by-side attention vs GT distance matrix
"""

from pathlib import Path
from typing import List, Optional, Tuple, Union

import numpy as np

from .trajectory_analyzer import TrajectoryMetrics


def plot_crystallization_trajectory(
    metrics: TrajectoryMetrics,
    layers_to_plot: Optional[List[int]] = None,
    heads_to_plot: Optional[List[int]] = None,
    save_path: Optional[Union[str, Path]] = None,
    figsize: Tuple[int, int] = (15, 5),
    title: Optional[str] = None,
) -> "matplotlib.figure.Figure":
    """
    Plot crystallization metrics across the trajectory.

    Creates a 3-panel figure showing:
    1. Logit Dominance (R) vs timestep
    2. Entropy (H) vs timestep
    3. Spatial Alignment (rho) vs timestep (if available)

    Args:
        metrics: TrajectoryMetrics object with computed metrics
        layers_to_plot: List of layer indices to plot, or None for [0, mid, last]
        heads_to_plot: List of head indices to average over, or None for all
        save_path: Path to save figure, or None to not save
        figsize: Figure size (width, height)
        title: Optional title for the figure

    Returns:
        matplotlib Figure object
    """
    import matplotlib.pyplot as plt

    timesteps = metrics.timesteps
    T, L, H = metrics.logit_dominance.shape

    if layers_to_plot is None:
        layers_to_plot = [0, L // 2, L - 1]  # First, middle, last
    if heads_to_plot is None:
        heads_to_plot = list(range(H))

    # Determine number of panels
    has_spatial = metrics.spatial_alignment is not None
    n_panels = 3 if has_spatial else 2

    fig, axes = plt.subplots(1, n_panels, figsize=figsize)
    if n_panels == 2:
        axes = list(axes) + [None]

    # Colors for different layers
    colors = plt.cm.viridis(np.linspace(0, 1, len(layers_to_plot)))

    # Panel 1: Logit Dominance (R)
    ax = axes[0]
    for l_idx, l in enumerate(layers_to_plot):
        mean_R = metrics.logit_dominance[:, l, heads_to_plot].mean(axis=-1)
        std_R = metrics.logit_dominance[:, l, heads_to_plot].std(axis=-1)

        ax.plot(timesteps, mean_R, color=colors[l_idx], label=f'Layer {l}', linewidth=2)
        ax.fill_between(timesteps, mean_R - std_R, mean_R + std_R,
                       color=colors[l_idx], alpha=0.2)

    ax.set_xlabel('Timestep (t)')
    ax.set_ylabel('R = ||B||/||C||')
    ax.set_title('Logit Dominance (R)\nHigher = Geometric bias dominates')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)

    # Panel 2: Entropy (H)
    ax = axes[1]
    for l_idx, l in enumerate(layers_to_plot):
        mean_H = metrics.entropy[:, l, heads_to_plot].mean(axis=-1)
        std_H = metrics.entropy[:, l, heads_to_plot].std(axis=-1)

        ax.plot(timesteps, mean_H, color=colors[l_idx], label=f'Layer {l}', linewidth=2)
        ax.fill_between(timesteps, mean_H - std_H, mean_H + std_H,
                       color=colors[l_idx], alpha=0.2)

    ax.set_xlabel('Timestep (t)')
    ax.set_ylabel('H = -sum(p*log(p))')
    ax.set_title('Attention Entropy (H)\nLower = More crystallized')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)

    # Panel 3: Spatial Alignment (rho)
    if has_spatial:
        ax = axes[2]
        for l_idx, l in enumerate(layers_to_plot):
            mean_rho = metrics.spatial_alignment[:, l, heads_to_plot].mean(axis=-1)
            std_rho = metrics.spatial_alignment[:, l, heads_to_plot].std(axis=-1)

            ax.plot(timesteps, mean_rho, color=colors[l_idx], label=f'Layer {l}', linewidth=2)
            ax.fill_between(timesteps, mean_rho - std_rho, mean_rho + std_rho,
                           color=colors[l_idx], alpha=0.2)

        ax.set_xlabel('Timestep (t)')
        ax.set_ylabel('rho (Pearson correlation)')
        ax.set_title('Spatial Alignment (rho)\nHigher = Biologically accurate')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)

    if title:
        fig.suptitle(title, fontsize=14, y=1.02)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig


def plot_layer_heatmap(
    metrics: TrajectoryMetrics,
    metric_name: str = 'entropy',
    head_idx: Optional[int] = None,
    save_path: Optional[Union[str, Path]] = None,
    figsize: Tuple[int, int] = (12, 6),
    cmap: str = 'viridis',
) -> "matplotlib.figure.Figure":
    """
    Plot a heatmap of a metric across layers (y-axis) and timesteps (x-axis).

    Args:
        metrics: TrajectoryMetrics object
        metric_name: Which metric to plot ('entropy', 'logit_dominance', 'spatial_alignment')
        head_idx: Specific head to plot, or None to average over heads
        save_path: Path to save figure
        figsize: Figure size
        cmap: Colormap name

    Returns:
        matplotlib Figure object
    """
    import matplotlib.pyplot as plt

    # Select metric
    if metric_name == 'entropy':
        data = metrics.entropy
        label = 'Entropy (H)'
    elif metric_name == 'logit_dominance':
        data = metrics.logit_dominance
        label = 'Logit Dominance (R)'
    elif metric_name == 'spatial_alignment':
        if metrics.spatial_alignment is None:
            raise ValueError("Spatial alignment not available")
        data = metrics.spatial_alignment
        label = 'Spatial Alignment (rho)'
    else:
        raise ValueError(f"Unknown metric: {metric_name}")

    # Average over heads if not specified
    if head_idx is not None:
        data = data[:, :, head_idx]
    else:
        data = data.mean(axis=-1)

    # Transpose so layers are on y-axis, timesteps on x-axis
    data = data.T  # [L, T]

    fig, ax = plt.subplots(figsize=figsize)

    im = ax.imshow(data, aspect='auto', cmap=cmap, origin='lower')

    # Set axis labels
    n_timesteps = len(metrics.timesteps)
    n_layers = metrics.num_layers

    # Set x ticks (timesteps)
    x_tick_indices = np.linspace(0, n_timesteps - 1, min(10, n_timesteps)).astype(int)
    ax.set_xticks(x_tick_indices)
    ax.set_xticklabels([f'{metrics.timesteps[i]:.2f}' for i in x_tick_indices])

    # Set y ticks (layers)
    ax.set_yticks(range(n_layers))
    ax.set_yticklabels([f'L{i}' for i in range(n_layers)])

    ax.set_xlabel('Timestep (t)')
    ax.set_ylabel('Layer')
    ax.set_title(f'{label} across Layers and Timesteps')

    plt.colorbar(im, ax=ax, label=label)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig


def plot_attention_heatmap(
    attn_weights: np.ndarray,
    gt_distance: Optional[np.ndarray] = None,
    title: str = '',
    save_path: Optional[Union[str, Path]] = None,
    figsize: Tuple[int, int] = (12, 5),
) -> "matplotlib.figure.Figure":
    """
    Plot attention heatmap alongside ground truth distance matrix.

    Args:
        attn_weights: Attention weights, shape [n, n] or [h, n, n]
        gt_distance: Ground truth distance matrix, shape [n, n]
        title: Title for the plot
        save_path: Path to save figure
        figsize: Figure size

    Returns:
        matplotlib Figure object
    """
    import matplotlib.pyplot as plt

    # Average over heads if needed
    if attn_weights.ndim == 3:
        attn_weights = attn_weights.mean(axis=0)

    n_plots = 2 if gt_distance is not None else 1
    fig, axes = plt.subplots(1, n_plots, figsize=figsize)

    if n_plots == 1:
        axes = [axes]

    # Plot attention
    im1 = axes[0].imshow(attn_weights, cmap='viridis', aspect='equal')
    axes[0].set_title(f'Attention Weights\n{title}')
    axes[0].set_xlabel('Key position')
    axes[0].set_ylabel('Query position')
    plt.colorbar(im1, ax=axes[0], fraction=0.046, pad=0.04)

    # Plot ground truth distance
    if gt_distance is not None:
        im2 = axes[1].imshow(gt_distance, cmap='RdBu_r', aspect='equal')
        axes[1].set_title('Ground Truth Distance (nm)')
        axes[1].set_xlabel('Residue j')
        axes[1].set_ylabel('Residue i')
        plt.colorbar(im2, ax=axes[1], fraction=0.046, pad=0.04)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig


def plot_crystallization_summary(
    metrics: TrajectoryMetrics,
    save_path: Optional[Union[str, Path]] = None,
    figsize: Tuple[int, int] = (16, 12),
) -> "matplotlib.figure.Figure":
    """
    Create a comprehensive summary plot with multiple visualizations.

    Includes:
    - Trajectory plots for all metrics
    - Heatmaps across layers and timesteps
    - Summary statistics

    Args:
        metrics: TrajectoryMetrics object
        save_path: Path to save figure
        figsize: Figure size

    Returns:
        matplotlib Figure object
    """
    import matplotlib.pyplot as plt

    has_spatial = metrics.spatial_alignment is not None
    n_cols = 3 if has_spatial else 2

    fig = plt.figure(figsize=figsize)

    # Row 1: Trajectory plots
    ax1 = fig.add_subplot(2, n_cols, 1)
    ax2 = fig.add_subplot(2, n_cols, 2)
    if has_spatial:
        ax3 = fig.add_subplot(2, n_cols, 3)

    # Row 2: Heatmaps
    ax4 = fig.add_subplot(2, n_cols, n_cols + 1)
    ax5 = fig.add_subplot(2, n_cols, n_cols + 2)
    if has_spatial:
        ax6 = fig.add_subplot(2, n_cols, n_cols + 3)

    timesteps = metrics.timesteps
    colors = plt.cm.viridis(np.linspace(0, 1, 3))
    layers = [0, metrics.num_layers // 2, metrics.num_layers - 1]

    # Trajectory plots
    for l_idx, l in enumerate(layers):
        ax1.plot(timesteps, metrics.logit_dominance[:, l, :].mean(axis=-1),
                color=colors[l_idx], label=f'L{l}', linewidth=2)
        ax2.plot(timesteps, metrics.entropy[:, l, :].mean(axis=-1),
                color=colors[l_idx], label=f'L{l}', linewidth=2)
        if has_spatial:
            ax3.plot(timesteps, metrics.spatial_alignment[:, l, :].mean(axis=-1),
                    color=colors[l_idx], label=f'L{l}', linewidth=2)

    ax1.set_title('Logit Dominance (R)')
    ax1.set_xlabel('Timestep')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.set_title('Entropy (H)')
    ax2.set_xlabel('Timestep')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    if has_spatial:
        ax3.set_title('Spatial Alignment (rho)')
        ax3.set_xlabel('Timestep')
        ax3.legend()
        ax3.grid(True, alpha=0.3)

    # Heatmaps
    im1 = ax4.imshow(metrics.logit_dominance.mean(axis=-1).T, aspect='auto',
                     cmap='viridis', origin='lower')
    ax4.set_title('R across Layers')
    ax4.set_xlabel('Timestep')
    ax4.set_ylabel('Layer')
    plt.colorbar(im1, ax=ax4)

    im2 = ax5.imshow(metrics.entropy.mean(axis=-1).T, aspect='auto',
                     cmap='viridis', origin='lower')
    ax5.set_title('H across Layers')
    ax5.set_xlabel('Timestep')
    ax5.set_ylabel('Layer')
    plt.colorbar(im2, ax=ax5)

    if has_spatial:
        im3 = ax6.imshow(metrics.spatial_alignment.mean(axis=-1).T, aspect='auto',
                         cmap='RdBu_r', origin='lower', vmin=-1, vmax=1)
        ax6.set_title('rho across Layers')
        ax6.set_xlabel('Timestep')
        ax6.set_ylabel('Layer')
        plt.colorbar(im3, ax=ax6)

    plt.suptitle(f'Crystallization Analysis (n={metrics.protein_length})', fontsize=14, y=1.02)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig
