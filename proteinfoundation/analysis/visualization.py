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
from typing import Any, List, Optional, Tuple, Union

import numpy as np

from .trajectory_analyzer import (
    TrajectoryMetrics,
    SeqsepMetrics,
    ContactPrecisionMetrics,
    RegisterMetrics,
)


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

    # Panel 2: Entropy (H), normalized by log(n) so range is [0, 1]
    ax = axes[1]
    log_n = np.log(metrics.protein_length)
    for l_idx, l in enumerate(layers_to_plot):
        mean_H = metrics.entropy[:, l, heads_to_plot].mean(axis=-1) / log_n
        std_H = metrics.entropy[:, l, heads_to_plot].std(axis=-1) / log_n

        ax.plot(timesteps, mean_H, color=colors[l_idx], label=f'Layer {l}', linewidth=2)
        ax.fill_between(timesteps, mean_H - std_H, mean_H + std_H,
                       color=colors[l_idx], alpha=0.2)

    ax.set_xlabel('Timestep (t)')
    ax.set_ylabel('H / log(n)  [0 = sharp, 1 = uniform]')
    ax.set_title('Attention Entropy (H / log n)\nLower = More crystallized')
    ax.set_ylim(bottom=0)
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
        ax.set_title(f'Spatial Alignment (rho, {metrics.spatial_alignment_label})\nHigher = Biologically accurate')
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

    # Normalize entropy by log(n) so the colorscale is length-independent
    if metric_name == 'entropy':
        data = data / np.log(metrics.protein_length)
        label = 'Entropy H / log(n)  [0 = sharp, 1 = uniform]'

    # Transpose so layers are on y-axis, timesteps on x-axis
    data = data.T  # [L, T]

    fig, ax = plt.subplots(figsize=figsize)

    vmin, vmax = (0, 1) if metric_name == 'entropy' else (None, None)
    im = ax.imshow(data, aspect='auto', cmap=cmap, origin='lower', vmin=vmin, vmax=vmax)

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
    log_n = np.log(metrics.protein_length)
    for l_idx, l in enumerate(layers):
        ax1.plot(timesteps, metrics.logit_dominance[:, l, :].mean(axis=-1),
                color=colors[l_idx], label=f'L{l}', linewidth=2)
        ax2.plot(timesteps, metrics.entropy[:, l, :].mean(axis=-1) / log_n,
                color=colors[l_idx], label=f'L{l}', linewidth=2)
        if has_spatial:
            ax3.plot(timesteps, metrics.spatial_alignment[:, l, :].mean(axis=-1),
                    color=colors[l_idx], label=f'L{l}', linewidth=2)

    ax1.set_title('Logit Dominance (R)')
    ax1.set_xlabel('Timestep')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.set_title('Entropy (H / log n)')
    ax2.set_xlabel('Timestep')
    ax2.set_ylabel('H / log(n)  [0 = sharp, 1 = uniform]')
    ax2.set_ylim(bottom=0)
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    if has_spatial:
        ax3.set_title(f'Spatial Alignment (rho, {metrics.spatial_alignment_label})')
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

    im2 = ax5.imshow(metrics.entropy.mean(axis=-1).T / log_n, aspect='auto',
                     cmap='viridis', origin='lower', vmin=0, vmax=1)
    ax5.set_title('H / log(n) across Layers')
    ax5.set_xlabel('Timestep')
    ax5.set_ylabel('Layer')
    plt.colorbar(im2, ax=ax5, label='H / log(n)')

    if has_spatial:
        im3 = ax6.imshow(metrics.spatial_alignment.mean(axis=-1).T, aspect='auto',
                         cmap='RdBu_r', origin='lower', vmin=-1, vmax=1)
        ax6.set_title(f'rho across Layers ({metrics.spatial_alignment_label})')
        ax6.set_xlabel('Timestep')
        ax6.set_ylabel('Layer')
        plt.colorbar(im3, ax=ax6)

    plt.suptitle(f'Crystallization Analysis (n={metrics.protein_length})', fontsize=14, y=1.02)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig


def plot_per_head_trajectory(
    metrics: TrajectoryMetrics,
    layer: int = 0,
    metric_name: str = 'entropy',
    save_path: Optional[Union[str, Path]] = None,
    figsize: Tuple[int, int] = (10, 5),
) -> "matplotlib.figure.Figure":
    """
    Plot each attention head as a separate line for a given layer.

    Unlike plot_crystallization_trajectory (which averages over heads), this
    shows individual head trajectories, revealing head specialization:
    - Geometric specialist heads: high rho, converge early
    - Content specialist heads: low R, moderate rho, converge late
    - Computation/sink heads: persistently high entropy, low rho

    Args:
        metrics: TrajectoryMetrics with shape [T, L, H].
        layer: Which layer to visualize.
        metric_name: Metric to plot ('entropy', 'logit_dominance', 'spatial_alignment').
        save_path: Path to save figure, or None.
        figsize: Figure size.

    Returns:
        matplotlib Figure object.
    """
    import matplotlib.pyplot as plt

    if metric_name == 'entropy':
        data = metrics.entropy[:, layer, :]  # [T, H]
        log_n = np.log(metrics.protein_length)
        data = data / log_n
        ylabel = 'H / log(n)'
        title = f'Per-Head Entropy (Layer {layer})'
    elif metric_name == 'logit_dominance':
        data = metrics.logit_dominance[:, layer, :]  # [T, H]
        ylabel = 'R = ||B||/||C||'
        title = f'Per-Head Logit Dominance (Layer {layer})'
    elif metric_name == 'spatial_alignment':
        if metrics.spatial_alignment is None:
            raise ValueError("spatial_alignment not available")
        data = metrics.spatial_alignment[:, layer, :]  # [T, H]
        ylabel = 'rho (Pearson correlation)'
        title = f'Per-Head Spatial Alignment (Layer {layer})'
    else:
        raise ValueError(f"Unknown metric: {metric_name}")

    T, H = data.shape
    timesteps = metrics.timesteps
    colors = plt.cm.tab10(np.linspace(0, 1, H))

    fig, ax = plt.subplots(figsize=figsize)
    for h in range(H):
        ax.plot(timesteps, data[:, h], color=colors[h], label=f'Head {h}', linewidth=1.5)

    ax.set_xlabel('Timestep (t)')
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(loc='best', ncol=2, fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig


def plot_seqsep_decomposition(
    seqsep_metrics: SeqsepMetrics,
    layers_to_plot: Optional[List[int]] = None,
    save_path: Optional[Union[str, Path]] = None,
    figsize: Tuple[int, int] = (15, 12),
) -> "matplotlib.figure.Figure":
    """
    Plot R, H, rho decomposed by sequence separation bin.

    Creates a (num_bins x 3) grid showing R, H, rho for each seqsep bin,
    with lines for selected layers. This directly tests whether the geometric
    bias B matters more for long-range than local interactions.

    Args:
        seqsep_metrics: SeqsepMetrics from compute_seqsep_trajectory().
        layers_to_plot: Layers to show. Default: [0, mid, last].
        save_path: Path to save figure.
        figsize: Figure size.

    Returns:
        matplotlib Figure object.
    """
    import matplotlib.pyplot as plt

    num_bins = len(seqsep_metrics.bin_labels)
    timesteps = seqsep_metrics.timesteps
    log_n = np.log(seqsep_metrics.protein_length)
    has_rho = seqsep_metrics.spatial_alignment is not None
    n_cols = 3 if has_rho else 2

    L = seqsep_metrics.num_layers
    if layers_to_plot is None:
        layers_to_plot = [0, L // 2, L - 1]

    layer_colors = plt.cm.viridis(np.linspace(0, 1, len(layers_to_plot)))

    fig, axes = plt.subplots(num_bins, n_cols, figsize=figsize,
                             squeeze=False, sharex=True)

    for bin_idx, label in enumerate(seqsep_metrics.bin_labels):
        for l_idx, l in enumerate(layers_to_plot):
            color = layer_colors[l_idx]
            lbl = f'L{l}'

            # R
            mean_R = seqsep_metrics.logit_dominance[bin_idx, :, l, :].mean(axis=-1)
            std_R = seqsep_metrics.logit_dominance[bin_idx, :, l, :].std(axis=-1)
            axes[bin_idx, 0].plot(timesteps, mean_R, color=color, label=lbl, linewidth=2)
            axes[bin_idx, 0].fill_between(timesteps, mean_R - std_R, mean_R + std_R,
                                          color=color, alpha=0.15)

            # H (normalized)
            mean_H = seqsep_metrics.entropy[bin_idx, :, l, :].mean(axis=-1) / log_n
            std_H = seqsep_metrics.entropy[bin_idx, :, l, :].std(axis=-1) / log_n
            axes[bin_idx, 1].plot(timesteps, mean_H, color=color, label=lbl, linewidth=2)
            axes[bin_idx, 1].fill_between(timesteps, mean_H - std_H, mean_H + std_H,
                                          color=color, alpha=0.15)

            # rho
            if has_rho:
                mean_rho = seqsep_metrics.spatial_alignment[bin_idx, :, l, :].mean(axis=-1)
                std_rho = seqsep_metrics.spatial_alignment[bin_idx, :, l, :].std(axis=-1)
                axes[bin_idx, 2].plot(timesteps, mean_rho, color=color, label=lbl, linewidth=2)
                axes[bin_idx, 2].fill_between(timesteps, mean_rho - std_rho, mean_rho + std_rho,
                                              color=color, alpha=0.15)

        # Row labels
        axes[bin_idx, 0].set_ylabel(f'{label}\nR = ||B||/||C||')
        axes[bin_idx, 1].set_ylabel('H / log(n)')
        if has_rho:
            axes[bin_idx, 2].set_ylabel('rho')
        for col in range(n_cols):
            axes[bin_idx, col].legend(fontsize=8, loc='best')
            axes[bin_idx, col].grid(True, alpha=0.3)

    # Column titles
    axes[0, 0].set_title('Logit Dominance (R)\nGeometry vs. Content')
    axes[0, 1].set_title('Attention Entropy (H / log n)\nSharpness')
    if has_rho:
        axes[0, 2].set_title('Spatial Alignment (rho)\nGeometric Accuracy')

    # X-axis labels on bottom row
    for col in range(n_cols):
        axes[-1, col].set_xlabel('Timestep (t)')

    plt.suptitle(
        f'Metrics by Sequence Separation (n={seqsep_metrics.protein_length})',
        fontsize=13, y=1.01,
    )
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig


def plot_contact_precision_trajectory(
    precision_metrics: ContactPrecisionMetrics,
    layers_to_plot: Optional[List[int]] = None,
    save_path: Optional[Union[str, Path]] = None,
    figsize: Tuple[int, int] = (15, 5),
) -> "matplotlib.figure.Figure":
    """
    Plot Precision@k contact prediction for full, B-only, and C-only attention.

    Shows three panels (full, B-only, C-only), each with lines for selected
    layers. This reveals:
    - Whether B or C individually predict structural contacts
    - When during denoising the content score C develops geometric specificity
    - Which layers have the most contact-predictive attention

    Args:
        precision_metrics: ContactPrecisionMetrics from compute_contact_precision_trajectory().
        layers_to_plot: Layers to show. Default: [0, mid, last].
        save_path: Path to save figure.
        figsize: Figure size.

    Returns:
        matplotlib Figure object.
    """
    import matplotlib.pyplot as plt

    timesteps = precision_metrics.timesteps
    L = precision_metrics.num_layers
    if layers_to_plot is None:
        layers_to_plot = [0, L // 2, L - 1]

    colors = plt.cm.viridis(np.linspace(0, 1, len(layers_to_plot)))

    fig, axes = plt.subplots(1, 3, figsize=figsize, sharey=True)

    panels = [
        (precision_metrics.precision_full, 'Full Attention (C+B)'),
        (precision_metrics.precision_b_only, 'B-only: softmax(B)'),
        (precision_metrics.precision_c_only, 'C-only: softmax(QKᵀ·s)'),
    ]

    for ax, (data, panel_title) in zip(axes, panels):
        for l_idx, l in enumerate(layers_to_plot):
            mean_p = data[:, l, :].mean(axis=-1)
            std_p = data[:, l, :].std(axis=-1)
            ax.plot(timesteps, mean_p, color=colors[l_idx], label=f'L{l}', linewidth=2)
            ax.fill_between(timesteps, mean_p - std_p, mean_p + std_p,
                            color=colors[l_idx], alpha=0.15)
        ax.set_xlabel('Timestep (t)')
        ax.set_title(panel_title)
        ax.legend(fontsize=8, loc='best')
        ax.grid(True, alpha=0.3)
        ax.set_ylim(bottom=0)

    axes[0].set_ylabel(f'Precision@{precision_metrics.k}')
    plt.suptitle(
        f'Contact Precision (n={precision_metrics.protein_length}, k={precision_metrics.k})',
        fontsize=13,
    )
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig


def plot_register_heatmap(
    register_metrics: RegisterMetrics,
    save_path: Optional[Union[str, Path]] = None,
    figsize: Tuple[int, int] = (10, 6),
) -> "matplotlib.figure.Figure":
    """
    Plot register token attention fraction as a layer x timestep heatmap.

    Shows where register tokens act as attention sinks across the trajectory,
    testing the Mix-Compress-Refine hypothesis: register attention should peak
    in middle layers (Compress phase).

    Args:
        register_metrics: RegisterMetrics from compute_register_metrics().
        save_path: Path to save figure.
        figsize: Figure size.

    Returns:
        matplotlib Figure object.
    """
    import matplotlib.pyplot as plt

    # Mean over heads: [T, L]
    data = register_metrics.register_attn_fraction.mean(axis=2)
    uniform = register_metrics.num_registers / (
        register_metrics.protein_length + register_metrics.num_registers
    )

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(data.T, aspect='auto', cmap='YlOrRd', origin='lower',
                   vmin=0, vmax=max(data.max(), 0.4))

    n_timesteps = data.shape[0]
    n_layers = data.shape[1]

    x_ticks = np.linspace(0, n_timesteps - 1, min(8, n_timesteps)).astype(int)
    ax.set_xticks(x_ticks)
    ax.set_xticklabels([f'{register_metrics.timesteps[i]:.2f}' for i in x_ticks])
    ax.set_yticks(range(n_layers))
    ax.set_yticklabels([f'L{i}' for i in range(n_layers)])

    ax.set_xlabel('Timestep (t)')
    ax.set_ylabel('Layer')
    ax.set_title(
        f'Register Token Attention Fraction '
        f'(n_reg={register_metrics.num_registers}, '
        f'uniform={uniform:.3f})'
    )

    plt.colorbar(im, ax=ax, label='Attention fraction to registers')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig


def plot_attention_decomposition_grid(
    tracker: Any,
    timestep_indices_to_show: Optional[List[int]] = None,
    layers_to_show: Optional[List[int]] = None,
    batch_idx: int = 0,
    num_registers: int = 0,
    contact_map: Optional[np.ndarray] = None,
    show_mode: str = 'attn',
    save_path: Optional[Union[str, Path]] = None,
    figsize: Optional[Tuple[int, int]] = None,
) -> "matplotlib.figure.Figure":
    """
    Plot a grid of attention heatmaps across layers (rows) and timesteps (columns).

    Inspired by AlphaFold2 Supplementary Figure 12, which shows per-layer
    attention heatmaps across recycling iterations to visualize crystallization.
    Here timestep plays the role of recycling iteration: columns progress from
    early denoising (diffuse attention, t~0) to late denoising (sharp contacts, t~1).

    Args:
        tracker: CrystallizationTracker with captured attention data.
        timestep_indices_to_show: Which timestep_indices to use as columns.
                                  Default: evenly-spaced selection of up to 6.
        layers_to_show: Which layers to use as rows. Default: all layers.
        batch_idx: Batch element to visualize.
        num_registers: Number of register tokens to strip (same as in analyzer).
        contact_map: Optional [n, n] binary contact map to overlay as contours.
        show_mode: 'attn' (post-softmax attention), 'bias' (pair bias B),
                   or 'content' (QK^T before softmax).
        save_path: Path to save figure.
        figsize: Figure size. Default: auto-scaled to grid dimensions.

    Returns:
        matplotlib Figure object.
    """
    import matplotlib.pyplot as plt

    all_timestep_indices = tracker.get_timestep_indices()
    all_layer_indices = tracker.get_layer_indices()

    if timestep_indices_to_show is None:
        n_show = min(6, len(all_timestep_indices))
        step = max(1, len(all_timestep_indices) // n_show)
        timestep_indices_to_show = all_timestep_indices[::step][:n_show]
    if layers_to_show is None:
        layers_to_show = all_layer_indices

    n_rows = len(layers_to_show)
    n_cols = len(timestep_indices_to_show)

    if figsize is None:
        figsize = (n_cols * 2.5 + 1, n_rows * 2.5 + 0.5)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize, squeeze=False)

    for row_idx, layer_idx in enumerate(layers_to_show):
        for col_idx, timestep_idx in enumerate(timestep_indices_to_show):
            ax = axes[row_idx, col_idx]
            capture = tracker.get_capture(timestep_idx, layer_idx)

            if capture is None:
                ax.axis('off')
                continue

            if show_mode == 'attn':
                raw = capture.attn_weights
            elif show_mode == 'bias':
                raw = capture.bias
            elif show_mode == 'content':
                raw = capture.qk_raw
            else:
                raise ValueError(f"Unknown show_mode: {show_mode}")

            if raw is None:
                ax.axis('off')
                continue

            # Strip registers and select batch element
            r = num_registers
            data = raw[batch_idx]  # [h, n+r, n+r]
            if r > 0:
                data = data[:, r:, r:]  # [h, n, n]

            # Average over heads
            img = data.mean(dim=0).cpu().numpy()  # [n, n]

            ax.imshow(img, cmap='viridis', aspect='equal', interpolation='nearest')

            # Overlay contact map as contour
            if contact_map is not None:
                ax.contour(contact_map, levels=[0.5], colors='red', linewidths=0.5, alpha=0.5)

            ax.set_xticks([])
            ax.set_yticks([])

            # Labels on edges only
            if col_idx == 0:
                ax.set_ylabel(f'L{layer_idx}', fontsize=9)
            if row_idx == 0:
                t_val = capture.timestep or 0.0
                ax.set_title(f't={t_val:.2f}', fontsize=9)

    mode_labels = {'attn': 'Attention (C+B)', 'bias': 'Pair Bias (B)', 'content': 'Content QKᵀ (C)'}
    plt.suptitle(
        f'{mode_labels.get(show_mode, show_mode)} — Layers × Timesteps',
        fontsize=12,
    )
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig
