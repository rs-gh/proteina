#!/usr/bin/env python
"""
Experiment: Multi-Seed Robustness — 200M model (tri), n=100, 3 seeds
=====================================================================

Date: 2026-03-15
Model: proteina_v1.1_DFS_200M_tri (200M params, 15 layers, 12 heads, triangle updates)
Protein length: 100 residues
Seeds: [5, 42, 123]

Purpose: Replicate the crystallization analysis on the 200M tri model
to compare with 200M notri and 60M notri. Triangle updates modify the
pair bias B via structure-aware message passing — this may change
crystallization dynamics.
"""

import subprocess
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent
while not (REPO_ROOT / ".git").exists():
    REPO_ROOT = REPO_ROOT.parent
EXPERIMENT_NAME = "2026-03-15-multiseed-200m-tri-n100"
ARTIFACTS_BASE = Path(__file__).resolve().parent / "artifacts"
DEFAULT_CKPT = "checkpoints/proteina_v1.1_dfs_200m_tri_v1.0"
CONFIG_NAME = "inference_ucond_200m_tri"

SEEDS = [5, 42, 123]
PROTEIN_LENGTH = 100


def run(ckpt_path: str = DEFAULT_CKPT):
    """Run crystallization analysis for each seed."""
    for seed in SEEDS:
        seed_dir = ARTIFACTS_BASE / f"seed_{seed}"
        seed_dir.mkdir(parents=True, exist_ok=True)

        cmd = [
            sys.executable,
            str(REPO_ROOT / "script_utils" / "crystallization_analysis.py"),
            "--config_name", CONFIG_NAME,
            "--ckpt_path", ckpt_path,
            "--protein_length", str(PROTEIN_LENGTH),
            "--output_dir", str(seed_dir),
            "--capture_every_n", "5",
            "--dt", "0.01",
            "--seed", str(seed),
            "--compute_seqsep",
            "--compute_contact_precision",
        ]

        print(f"\n{'='*60}")
        print(f"Running seed {seed}...")
        print(f"{'='*60}")
        subprocess.run(cmd, check=True)

    print(f"\nAll seeds complete. Artifacts in: {ARTIFACTS_BASE}")


def plot_aggregate(
    timesteps, R, H, rho=None,
    prec_full=None, prec_b=None, prec_c=None,
    seqsep_R=None, protein_length=100,
):
    """Generate aggregate plots showing mean +/- std across seeds."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    n_seeds, T, L, n_heads = R.shape
    log_n = np.log(protein_length)
    layers = [0, L // 2, L - 1]
    colors = plt.cm.viridis(np.linspace(0, 1, len(layers)))

    # --- 1. Aggregate trajectory (R, H, rho) ---
    has_rho = rho is not None
    n_panels = 3 if has_rho else 2
    fig, axes = plt.subplots(1, n_panels, figsize=(5 * n_panels, 5))
    if n_panels == 2:
        axes = list(axes) + [None]

    for l_idx, l in enumerate(layers):
        r_per_seed = R[:, :, l, :].mean(axis=-1)
        r_mean = r_per_seed.mean(axis=0)
        r_std = r_per_seed.std(axis=0)
        axes[0].plot(timesteps, r_mean, color=colors[l_idx], label=f'L{l}', linewidth=2)
        axes[0].fill_between(timesteps, r_mean - r_std, r_mean + r_std,
                             color=colors[l_idx], alpha=0.2)

        h_per_seed = H[:, :, l, :].mean(axis=-1) / log_n
        h_mean = h_per_seed.mean(axis=0)
        h_std = h_per_seed.std(axis=0)
        axes[1].plot(timesteps, h_mean, color=colors[l_idx], label=f'L{l}', linewidth=2)
        axes[1].fill_between(timesteps, h_mean - h_std, h_mean + h_std,
                             color=colors[l_idx], alpha=0.2)

        if has_rho:
            rho_per_seed = rho[:, :, l, :].mean(axis=-1)
            rho_mean = rho_per_seed.mean(axis=0)
            rho_std = rho_per_seed.std(axis=0)
            axes[2].plot(timesteps, rho_mean, color=colors[l_idx], label=f'L{l}', linewidth=2)
            axes[2].fill_between(timesteps, rho_mean - rho_std, rho_mean + rho_std,
                                 color=colors[l_idx], alpha=0.2)

    axes[0].set_title('Logit Dominance (R)\nHigher = Geometric bias dominates')
    axes[0].set_xlabel('Timestep (t)')
    axes[0].set_ylabel('R = ||B||/||C||')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].set_title('Attention Entropy (H / log n)\nLower = More crystallized')
    axes[1].set_xlabel('Timestep (t)')
    axes[1].set_ylabel('H / log(n)')
    axes[1].set_ylim(bottom=0)
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    if has_rho:
        axes[2].set_title('Spatial Alignment (rho)\nHigher = Biologically accurate')
        axes[2].set_xlabel('Timestep (t)')
        axes[2].set_ylabel('rho (Pearson)')
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)

    fig.suptitle(f'200M Tri — Aggregate Crystallization (n={protein_length}, {n_seeds} seeds, shaded = ±1 std)',
                 fontsize=13, y=1.02)
    plt.tight_layout()
    path = ARTIFACTS_BASE / "aggregate_trajectory.png"
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved {path}")

    # --- 2. Aggregate contact precision (full, B-only, C-only) ---
    if prec_full is not None:
        fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)
        panels = [
            (prec_full, 'Full (C+B)', '#2ca02c'),
            (prec_b, 'B-only: softmax(B)', '#1f77b4'),
            (prec_c, 'C-only: softmax(QKᵀ)', '#ff7f0e'),
        ]
        for ax, (data, title, color) in zip(axes, panels):
            for l_idx, l in enumerate(layers):
                d_per_seed = data[:, :, l, :].mean(axis=-1)
                d_mean = d_per_seed.mean(axis=0)
                d_std = d_per_seed.std(axis=0)
                ax.plot(timesteps, d_mean, color=colors[l_idx], label=f'L{l}', linewidth=2)
                ax.fill_between(timesteps, d_mean - d_std, d_mean + d_std,
                                color=colors[l_idx], alpha=0.2)
            ax.set_title(title)
            ax.set_xlabel('Timestep (t)')
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)
            ax.set_ylim(bottom=0)
        axes[0].set_ylabel('Precision@L/5')
        fig.suptitle(f'200M Tri — Contact Precision (n={protein_length}, {n_seeds} seeds)', fontsize=13)
        plt.tight_layout()
        path = ARTIFACTS_BASE / "aggregate_contact_precision.png"
        plt.savefig(path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"Saved {path}")

    # --- 3. Aggregate seqsep R at Layer 0 ---
    if seqsep_R is not None:
        bin_labels = ['local (1-6)', 'medium (7-23)', 'long (>=24)']
        bin_colors = ['#2ca02c', '#ff7f0e', '#d62728']
        fig, ax = plt.subplots(figsize=(8, 5))
        for bin_idx, (label, color) in enumerate(zip(bin_labels, bin_colors)):
            r_per_seed = seqsep_R[:, bin_idx, :, 0, :].mean(axis=-1)
            r_mean = r_per_seed.mean(axis=0)
            r_std = r_per_seed.std(axis=0)
            ax.plot(timesteps, r_mean, color=color, label=label, linewidth=2)
            ax.fill_between(timesteps, r_mean - r_std, r_mean + r_std,
                            color=color, alpha=0.2)
        ax.set_title(f'200M Tri — Logit Dominance by Sequence Separation (Layer 0, {n_seeds} seeds)')
        ax.set_xlabel('Timestep (t)')
        ax.set_ylabel('R = ||B||/||C||')
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        path = ARTIFACTS_BASE / "aggregate_seqsep_R.png"
        plt.savefig(path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"Saved {path}")


def aggregate():
    """Load metrics from all seeds and compute mean/std."""
    sys.path.insert(0, str(REPO_ROOT))
    from proteinfoundation.analysis import (
        TrajectoryMetrics, SeqsepMetrics, ContactPrecisionMetrics,
    )

    all_R, all_H, all_rho = [], [], []
    all_seqsep_R = []
    all_prec_full, all_prec_b, all_prec_c = [], [], []

    for seed in SEEDS:
        seed_dir = ARTIFACTS_BASE / f"seed_{seed}"

        core_path = seed_dir / "crystallization_metrics.npz"
        if core_path.exists():
            m = TrajectoryMetrics.load(str(core_path))
            all_R.append(m.logit_dominance)
            all_H.append(m.entropy)
            if m.spatial_alignment is not None:
                all_rho.append(m.spatial_alignment)

        seqsep_path = seed_dir / "seqsep_metrics.npz"
        if seqsep_path.exists():
            s = SeqsepMetrics.load(str(seqsep_path))
            all_seqsep_R.append(s.logit_dominance)

        prec_path = seed_dir / "contact_precision.npz"
        if prec_path.exists():
            p = ContactPrecisionMetrics.load(str(prec_path))
            all_prec_full.append(p.precision_full)
            all_prec_b.append(p.precision_b_only)
            all_prec_c.append(p.precision_c_only)

    n_seeds = len(all_R)
    if n_seeds == 0:
        print("No metrics found. Run the experiment first.")
        return

    print(f"\n{'='*60}")
    print(f"200M Tri — Aggregated results across {n_seeds} seeds")
    print(f"{'='*60}")

    R = np.stack(all_R)
    H = np.stack(all_H)

    R_l0_t0 = R[:, 0, 0, :].mean(axis=-1)
    print(f"\nLogit Dominance (R) — Layer 0 at t=0:")
    print(f"  Mean: {R_l0_t0.mean():.3f} +/- {R_l0_t0.std():.3f}")

    H_all_t0 = H[:, 0, :, :].mean(axis=(-1, -2))
    H_all_t1 = H[:, -1, :, :].mean(axis=(-1, -2))
    print(f"\nEntropy (H, normalized):")
    print(f"  t=0: {H_all_t0.mean():.3f} +/- {H_all_t0.std():.3f}")
    print(f"  t=1: {H_all_t1.mean():.3f} +/- {H_all_t1.std():.3f}")

    if all_rho:
        rho = np.stack(all_rho)
        rho_t1 = rho[:, -1, :, :].mean(axis=(-1, -2))
        print(f"\nSpatial Alignment (rho) at t=1:")
        print(f"  Mean: {rho_t1.mean():.3f} +/- {rho_t1.std():.3f}")

    if all_seqsep_R:
        seqsep_R = np.stack(all_seqsep_R)
        print(f"\nSeqsep R at t=0, Layer 0 (mean over heads):")
        for bin_idx, label in enumerate(['local (1-6)', 'medium (7-23)', 'long (>=24)']):
            vals = seqsep_R[:, bin_idx, 0, 0, :].mean(axis=-1)
            print(f"  {label:20s}: {vals.mean():.3f} +/- {vals.std():.3f}")

    if all_prec_full:
        prec_full = np.stack(all_prec_full)
        prec_b = np.stack(all_prec_b)
        prec_c = np.stack(all_prec_c)
        print(f"\nContact Precision@L/5 at t=1 (mean over layers/heads):")
        pf = prec_full[:, -1].mean(axis=(-1, -2))
        pb = prec_b[:, -1].mean(axis=(-1, -2))
        pc = prec_c[:, -1].mean(axis=(-1, -2))
        print(f"  Full:   {pf.mean():.3f} +/- {pf.std():.3f}")
        print(f"  B-only: {pb.mean():.3f} +/- {pb.std():.3f}")
        print(f"  C-only: {pc.mean():.3f} +/- {pc.std():.3f}")

    # Save aggregated
    agg_path = ARTIFACTS_BASE / "aggregated.npz"
    save_dict = {
        'seeds': np.array(SEEDS[:n_seeds]),
        'R_mean': R.mean(axis=0), 'R_std': R.std(axis=0),
        'H_mean': H.mean(axis=0), 'H_std': H.std(axis=0),
    }
    if all_rho:
        save_dict['rho_mean'] = rho.mean(axis=0)
        save_dict['rho_std'] = rho.std(axis=0)
    np.savez(str(agg_path), **save_dict)
    print(f"\nAggregated metrics saved to {agg_path}")

    # Generate aggregate plots
    rho_stacked = np.stack(all_rho) if all_rho else None
    prec_f_stacked = np.stack(all_prec_full) if all_prec_full else None
    prec_b_stacked = np.stack(all_prec_b) if all_prec_full else None
    prec_c_stacked = np.stack(all_prec_c) if all_prec_full else None
    seqsep_stacked = np.stack(all_seqsep_R) if all_seqsep_R else None
    plot_aggregate(
        m.timesteps, R, H,
        rho=rho_stacked,
        prec_full=prec_f_stacked,
        prec_b=prec_b_stacked,
        prec_c=prec_c_stacked,
        seqsep_R=seqsep_stacked,
        protein_length=PROTEIN_LENGTH,
    )


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt_path", default=DEFAULT_CKPT)
    parser.add_argument("--aggregate-only", action="store_true",
                        help="Just aggregate existing results")
    args = parser.parse_args()

    if args.aggregate_only:
        aggregate()
    else:
        run(args.ckpt_path)
        aggregate()
