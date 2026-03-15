#!/usr/bin/env python
"""
Experiment 4: Multi-Length Comparison — 60M model, n=50/100/200, 3 seeds each
==============================================================================

Date: 2026-03-01
Model: proteina_v1.3_DFS_60M_notri (60M params)
Protein lengths: 50, 100, 200 residues
Seeds: [5, 42, 123]

Purpose: Test whether crystallization patterns scale with protein size.
Hypothesis: Longer proteins crystallize later (more complex fold search).
"""

import subprocess
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent
while not (REPO_ROOT / ".git").exists():
    REPO_ROOT = REPO_ROOT.parent
EXPERIMENT_NAME = "2026-03-01-multilength-60m"
ARTIFACTS_BASE = Path(__file__).resolve().parent / "artifacts"
DEFAULT_CKPT = "checkpoints/proteina_v1.3_dfs_60m_notri_v1.0"

LENGTHS = [50, 100, 200]
SEEDS = [5, 42, 123]


def run(ckpt_path: str = DEFAULT_CKPT):
    """Run crystallization analysis for each length and seed."""
    for length in LENGTHS:
        for seed in SEEDS:
            run_dir = ARTIFACTS_BASE / f"n{length}_seed{seed}"
            run_dir.mkdir(parents=True, exist_ok=True)

            cmd = [
                sys.executable,
                str(REPO_ROOT / "script_utils" / "crystallization_analysis.py"),
                "--config_name", "inference_ucond_60m_notri",
                "--ckpt_path", ckpt_path,
                "--protein_length", str(length),
                "--output_dir", str(run_dir),
                "--capture_every_n", "5",
                "--dt", "0.01",
                "--seed", str(seed),
                "--compute_seqsep",
                "--compute_contact_precision",
            ]

            print(f"\n{'='*60}")
            print(f"Running n={length}, seed={seed}...")
            print(f"{'='*60}")
            subprocess.run(cmd, check=True)

    print(f"\nAll runs complete. Artifacts in: {ARTIFACTS_BASE}")


def aggregate():
    """Load metrics and compare crystallization across lengths."""
    sys.path.insert(0, str(REPO_ROOT))
    from proteinfoundation.analysis import TrajectoryMetrics, ContactPrecisionMetrics

    print(f"\n{'='*60}")
    print(f"Multi-Length Comparison")
    print(f"{'='*60}")

    for length in LENGTHS:
        all_R, all_H, all_rho = [], [], []
        all_prec_b, all_prec_c = [], []

        for seed in SEEDS:
            run_dir = ARTIFACTS_BASE / f"n{length}_seed{seed}"
            core_path = run_dir / "crystallization_metrics.npz"
            if core_path.exists():
                m = TrajectoryMetrics.load(str(core_path))
                all_R.append(m.logit_dominance)
                all_H.append(m.entropy)
                if m.spatial_alignment is not None:
                    all_rho.append(m.spatial_alignment)

            prec_path = run_dir / "contact_precision.npz"
            if prec_path.exists():
                p = ContactPrecisionMetrics.load(str(prec_path))
                all_prec_b.append(p.precision_b_only)
                all_prec_c.append(p.precision_c_only)

        if not all_R:
            print(f"\n  n={length}: no results found")
            continue

        R = np.stack(all_R)  # [seeds, T, L, H]
        H = np.stack(all_H)
        n_seeds = len(all_R)

        # Layer 0 R at t=0
        R_l0_t0 = R[:, 0, 0, :].mean(axis=-1)

        # Entropy at t=0 and t=1
        H_t0 = H[:, 0, :, :].mean(axis=(-1, -2))
        H_t1 = H[:, -1, :, :].mean(axis=(-1, -2))

        # Crystallization timing: when does mean entropy cross 50%?
        H_mean = H.mean(axis=(0, 2, 3))  # [T] mean over seeds, layers, heads
        H_range = H_mean.max() - H_mean.min()
        threshold = H_mean.max() - 0.5 * H_range
        crossed = np.where(H_mean < threshold)[0]
        t_cryst = m.timesteps[crossed[0]] if len(crossed) > 0 else float('nan')

        print(f"\n  n={length} ({n_seeds} seeds):")
        print(f"    R (Layer 0, t=0): {R_l0_t0.mean():.3f} +/- {R_l0_t0.std():.3f}")
        print(f"    H (t=0): {H_t0.mean():.3f} +/- {H_t0.std():.3f}")
        print(f"    H (t=1): {H_t1.mean():.3f} +/- {H_t1.std():.3f}")
        print(f"    Crystallization (50% entropy): t={t_cryst:.3f}")

        if all_rho:
            rho = np.stack(all_rho)
            rho_t1 = rho[:, -1, :, :].mean(axis=(-1, -2))
            print(f"    rho (t=1): {rho_t1.mean():.3f} +/- {rho_t1.std():.3f}")

        if all_prec_b:
            pb = np.stack(all_prec_b)[:, -1].mean(axis=(-1, -2))
            pc = np.stack(all_prec_c)[:, -1].mean(axis=(-1, -2))
            print(f"    Precision@L/5 at t=1: B={pb.mean():.3f}, C={pc.mean():.3f}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt_path", default=DEFAULT_CKPT)
    parser.add_argument("--aggregate-only", action="store_true")
    args = parser.parse_args()

    if args.aggregate_only:
        aggregate()
    else:
        run(args.ckpt_path)
        aggregate()
