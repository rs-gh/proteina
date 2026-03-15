#!/usr/bin/env python
"""
Experiment: Row-Centered Logit Dominance Comparison
====================================================

Date: 2026-03-01
Model: proteina_v1.3_DFS_60M_notri (60M params)
Protein length: 100 residues
Seeds: [5, 42, 123, 256, 999]

Purpose: Compare raw R = ||B||_F / ||C||_F with row-centered R_c that
accounts for softmax invariance to row-wise constant shifts.

The centered variant subtracts the row mean before computing Frobenius norms,
so only the within-row variance (which actually shapes the attention pattern)
is measured.
"""

import subprocess
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent
while not (REPO_ROOT / ".git").exists():
    REPO_ROOT = REPO_ROOT.parent
EXPERIMENT_NAME = "2026-03-01-centered-metric-comparison"
ARTIFACTS_BASE = Path(__file__).resolve().parent / "artifacts"
DEFAULT_CKPT = "checkpoints/proteina_v1.3_dfs_60m_notri_v1.0"

SEEDS = [5, 42, 123, 256, 999]
PROTEIN_LENGTH = 100


def run(ckpt_path: str = DEFAULT_CKPT):
    """Run crystallization analysis for each seed."""
    for seed in SEEDS:
        seed_dir = ARTIFACTS_BASE / f"seed_{seed}"
        seed_dir.mkdir(parents=True, exist_ok=True)

        cmd = [
            sys.executable,
            str(REPO_ROOT / "script_utils" / "crystallization_analysis.py"),
            "--config_name", "inference_ucond_60m_notri",
            "--ckpt_path", ckpt_path,
            "--protein_length", str(PROTEIN_LENGTH),
            "--output_dir", str(seed_dir),
            "--capture_every_n", "5",
            "--dt", "0.01",
            "--seed", str(seed),
            "--compute_seqsep",
        ]

        print(f"\n{'='*60}")
        print(f"Running seed {seed}...")
        print(f"{'='*60}")
        subprocess.run(cmd, check=True)

    print(f"\nAll seeds complete. Artifacts in: {ARTIFACTS_BASE}")


def plot_comparison(
    timesteps, R, Rc, seqsep_R=None, seqsep_Rc=None, protein_length=100,
):
    """Generate comparison plots: raw R vs row-centered R_c."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    n_seeds, T, L, n_heads = R.shape
    layers = [0, L // 2, L - 1]
    layer_labels = ['Layer 0 (early)', f'Layer {L // 2} (middle)', f'Layer {L - 1} (final)']

    # --- 1. Overlay plot: R vs R_c per layer ---
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=False)

    for ax, l, lbl in zip(axes, layers, layer_labels):
        # Raw R
        r_per_seed = R[:, :, l, :].mean(axis=-1)  # [seeds, T]
        r_mean = r_per_seed.mean(axis=0)
        r_std = r_per_seed.std(axis=0)
        ax.plot(timesteps, r_mean, color='#1f77b4', label='R (raw)', linewidth=2)
        ax.fill_between(timesteps, r_mean - r_std, r_mean + r_std,
                        color='#1f77b4', alpha=0.15)

        # Centered R_c
        rc_per_seed = Rc[:, :, l, :].mean(axis=-1)
        rc_mean = rc_per_seed.mean(axis=0)
        rc_std = rc_per_seed.std(axis=0)
        ax.plot(timesteps, rc_mean, color='#d62728', label='R_c (centered)', linewidth=2, linestyle='--')
        ax.fill_between(timesteps, rc_mean - rc_std, rc_mean + rc_std,
                        color='#d62728', alpha=0.15)

        ax.set_title(lbl)
        ax.set_xlabel('Timestep (t)')
        ax.set_ylabel('R')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    fig.suptitle(
        f'Raw vs Row-Centered Logit Dominance (n={protein_length}, {n_seeds} seeds, shaded = +/-1 std)',
        fontsize=13, y=1.02,
    )
    plt.tight_layout()
    path = ARTIFACTS_BASE / "comparison_R_vs_Rc.png"
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved {path}")

    # --- 2. Ratio plot: R / R_c per layer ---
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True)

    for ax, l, lbl in zip(axes, layers, layer_labels):
        r_per_seed = R[:, :, l, :].mean(axis=-1)
        rc_per_seed = Rc[:, :, l, :].mean(axis=-1)
        ratio_per_seed = r_per_seed / (rc_per_seed + 1e-8)
        ratio_mean = ratio_per_seed.mean(axis=0)
        ratio_std = ratio_per_seed.std(axis=0)

        ax.plot(timesteps, ratio_mean, color='#2ca02c', linewidth=2)
        ax.fill_between(timesteps, ratio_mean - ratio_std, ratio_mean + ratio_std,
                        color='#2ca02c', alpha=0.15)
        ax.axhline(y=1.0, color='gray', linestyle=':', alpha=0.5)
        ax.set_title(lbl)
        ax.set_xlabel('Timestep (t)')
        ax.set_ylabel('R / R_c')
        ax.grid(True, alpha=0.3)

    fig.suptitle(
        f'Ratio R/R_c (>1 means row-constant component inflates raw R)',
        fontsize=13, y=1.02,
    )
    plt.tight_layout()
    path = ARTIFACTS_BASE / "ratio_R_over_Rc.png"
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved {path}")

    # --- 3. Seqsep comparison at Layer 0 ---
    if seqsep_R is not None and seqsep_Rc is not None:
        bin_labels = ['local (1-6)', 'medium (7-23)', 'long (>=24)']
        bin_colors = ['#2ca02c', '#ff7f0e', '#d62728']

        fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)

        for ax, data, title in [
            (axes[0], seqsep_R, 'R (raw)'),
            (axes[1], seqsep_Rc, 'R_c (centered)'),
        ]:
            for bin_idx, (label, color) in enumerate(zip(bin_labels, bin_colors)):
                r_per_seed = data[:, bin_idx, :, 0, :].mean(axis=-1)  # [seeds, T]
                r_mean = r_per_seed.mean(axis=0)
                r_std = r_per_seed.std(axis=0)
                ax.plot(timesteps, r_mean, color=color, label=label, linewidth=2)
                ax.fill_between(timesteps, r_mean - r_std, r_mean + r_std,
                                color=color, alpha=0.15)

            ax.set_title(f'{title} — Layer 0')
            ax.set_xlabel('Timestep (t)')
            ax.set_ylabel('R')
            ax.legend(fontsize=9)
            ax.grid(True, alpha=0.3)

        fig.suptitle(
            f'Seqsep Decomposition: Raw vs Centered (n={protein_length}, {n_seeds} seeds)',
            fontsize=13, y=1.02,
        )
        plt.tight_layout()
        path = ARTIFACTS_BASE / "seqsep_R_vs_Rc.png"
        plt.savefig(path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"Saved {path}")


def aggregate():
    """Load metrics from all seeds and compare R vs R_c."""
    sys.path.insert(0, str(REPO_ROOT))
    from proteinfoundation.analysis import TrajectoryMetrics, SeqsepMetrics

    all_R, all_Rc = [], []
    all_seqsep_R, all_seqsep_Rc = [], []

    for seed in SEEDS:
        seed_dir = ARTIFACTS_BASE / f"seed_{seed}"

        core_path = seed_dir / "crystallization_metrics.npz"
        if core_path.exists():
            m = TrajectoryMetrics.load(str(core_path))
            all_R.append(m.logit_dominance)
            if m.logit_dominance_centered is not None:
                all_Rc.append(m.logit_dominance_centered)

        seqsep_path = seed_dir / "seqsep_metrics.npz"
        if seqsep_path.exists():
            s = SeqsepMetrics.load(str(seqsep_path))
            all_seqsep_R.append(s.logit_dominance)
            if s.logit_dominance_centered is not None:
                all_seqsep_Rc.append(s.logit_dominance_centered)

    n_seeds = len(all_R)
    if n_seeds == 0:
        print("No metrics found. Run the experiment first.")
        return

    R = np.stack(all_R)    # [seeds, T, L, H]
    Rc = np.stack(all_Rc) if all_Rc else None

    print(f"\n{'='*60}")
    print(f"Centered Metric Comparison ({n_seeds} seeds)")
    print(f"{'='*60}")

    if Rc is None:
        print("\nNo centered metrics found — re-run with updated code.")
        return

    # --- Table: R vs R_c at key timepoints ---
    L = R.shape[2]
    layers = [0, L // 2, L - 1]
    layer_names = ['L0', f'L{L // 2}', f'L{L - 1}']

    print(f"\n{'Metric':<12} {'Layer':<6} {'t=0 (mean +/- std)':<25} {'t=1 (mean +/- std)':<25}")
    print("-" * 70)

    for l, lname in zip(layers, layer_names):
        # Raw R
        r_t0 = R[:, 0, l, :].mean(axis=-1)  # [seeds]
        r_t1 = R[:, -1, l, :].mean(axis=-1)
        print(f"{'R (raw)':<12} {lname:<6} {r_t0.mean():.3f} +/- {r_t0.std():.3f}           {r_t1.mean():.3f} +/- {r_t1.std():.3f}")

        # Centered R_c
        rc_t0 = Rc[:, 0, l, :].mean(axis=-1)
        rc_t1 = Rc[:, -1, l, :].mean(axis=-1)
        print(f"{'R_c (cent.)':<12} {lname:<6} {rc_t0.mean():.3f} +/- {rc_t0.std():.3f}           {rc_t1.mean():.3f} +/- {rc_t1.std():.3f}")

        # Ratio
        ratio_t0 = r_t0 / (rc_t0 + 1e-8)
        ratio_t1 = r_t1 / (rc_t1 + 1e-8)
        print(f"{'R/R_c':<12} {lname:<6} {ratio_t0.mean():.3f} +/- {ratio_t0.std():.3f}           {ratio_t1.mean():.3f} +/- {ratio_t1.std():.3f}")
        print()

    # --- Seqsep comparison ---
    if all_seqsep_Rc:
        seqsep_R = np.stack(all_seqsep_R)    # [seeds, bins, T, L, H]
        seqsep_Rc = np.stack(all_seqsep_Rc)

        print(f"\nSeqsep R vs R_c at t=0, Layer 0 (mean over heads):")
        for bin_idx, label in enumerate(['local (1-6)', 'medium (7-23)', 'long (>=24)']):
            vals_r = seqsep_R[:, bin_idx, 0, 0, :].mean(axis=-1)
            vals_rc = seqsep_Rc[:, bin_idx, 0, 0, :].mean(axis=-1)
            print(f"  {label:20s}  R={vals_r.mean():.3f}+/-{vals_r.std():.3f}"
                  f"  R_c={vals_rc.mean():.3f}+/-{vals_rc.std():.3f}"
                  f"  ratio={vals_r.mean() / (vals_rc.mean() + 1e-8):.3f}")
    else:
        seqsep_R = None
        seqsep_Rc = None

    # --- Generate plots ---
    m = TrajectoryMetrics.load(str(ARTIFACTS_BASE / f"seed_{SEEDS[0]}" / "crystallization_metrics.npz"))
    plot_comparison(
        m.timesteps, R, Rc,
        seqsep_R=seqsep_R,
        seqsep_Rc=seqsep_Rc,
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
