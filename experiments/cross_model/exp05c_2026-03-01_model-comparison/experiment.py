#!/usr/bin/env python
"""
Experiment 5: Model Size Comparison — 60M vs 200M, n=100, 3 seeds
==================================================================

Date: 2026-03-01
Models:
  - proteina_v1.3_DFS_60M_notri (60M params, 10 layers, 8 heads)
  - proteina_v1.2_DFS_200M_notri (200M params, 15 layers, 12 heads)
Protein length: 100 residues
Seeds: [5, 42, 123]

Purpose: Compare crystallization dynamics across model scales.
Hypothesis: Larger model may show sharper crystallization or different
layer specialization patterns.

Prerequisites:
  Download the 200M checkpoint:
    ngc registry model download-version \
      "nvidia/clara/proteina_v1.2_dfs_200m_notri:1.0" \
      --dest checkpoints/
"""

import subprocess
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent
while not (REPO_ROOT / ".git").exists():
    REPO_ROOT = REPO_ROOT.parent
EXPERIMENT_NAME = "2026-03-01-model-comparison-200m"
ARTIFACTS_BASE = Path(__file__).resolve().parent / "artifacts"

MODELS = {
    "60m": {
        "config": "inference_ucond_60m_notri",
        "ckpt": "checkpoints/proteina_v1.3_dfs_60m_notri_v1.0",
    },
    "200m": {
        "config": "inference_ucond_200m_notri",
        "ckpt": "checkpoints/proteina_v1.2_dfs_200m_notri_v1.0",
    },
}

SEEDS = [5, 42, 123]
PROTEIN_LENGTH = 100


def run(models_to_run: list[str] | None = None):
    """Run crystallization analysis for each model and seed."""
    if models_to_run is None:
        models_to_run = list(MODELS.keys())

    for model_name in models_to_run:
        model_cfg = MODELS[model_name]
        ckpt_dir = REPO_ROOT / model_cfg["ckpt"]

        if not ckpt_dir.exists():
            print(f"\nSkipping {model_name}: checkpoint not found at {ckpt_dir}")
            print(f"  Download with: ngc registry model download-version ...")
            continue

        for seed in SEEDS:
            run_dir = ARTIFACTS_BASE / f"{model_name}_seed{seed}"
            run_dir.mkdir(parents=True, exist_ok=True)

            cmd = [
                sys.executable,
                str(REPO_ROOT / "script_utils" / "crystallization_analysis.py"),
                "--config_name", model_cfg["config"],
                "--ckpt_path", model_cfg["ckpt"],
                "--protein_length", str(PROTEIN_LENGTH),
                "--output_dir", str(run_dir),
                "--capture_every_n", "5",
                "--dt", "0.01",
                "--seed", str(seed),
                "--compute_seqsep",
                "--compute_contact_precision",
            ]

            print(f"\n{'='*60}")
            print(f"Running {model_name}, seed={seed}...")
            print(f"{'='*60}")
            subprocess.run(cmd, check=True)

    print(f"\nAll runs complete. Artifacts in: {ARTIFACTS_BASE}")


def aggregate():
    """Compare crystallization metrics across models."""
    sys.path.insert(0, str(REPO_ROOT))
    from proteinfoundation.analysis import TrajectoryMetrics

    print(f"\n{'='*60}")
    print(f"Model Size Comparison: 60M vs 200M")
    print(f"{'='*60}")

    for model_name in MODELS:
        all_R, all_H, all_rho = [], [], []

        for seed in SEEDS:
            run_dir = ARTIFACTS_BASE / f"{model_name}_seed{seed}"
            core_path = run_dir / "crystallization_metrics.npz"
            if core_path.exists():
                m = TrajectoryMetrics.load(str(core_path))
                all_R.append(m.logit_dominance)
                all_H.append(m.entropy)
                if m.spatial_alignment is not None:
                    all_rho.append(m.spatial_alignment)

        if not all_R:
            print(f"\n  {model_name}: no results found")
            continue

        R = np.stack(all_R)
        H = np.stack(all_H)
        n_seeds = len(all_R)

        R_l0_t0 = R[:, 0, 0, :].mean(axis=-1)
        H_t0 = H[:, 0, :, :].mean(axis=(-1, -2))
        H_t1 = H[:, -1, :, :].mean(axis=(-1, -2))

        print(f"\n  {model_name} ({n_seeds} seeds, {R.shape[2]} layers, {R.shape[3]} heads):")
        print(f"    R (Layer 0, t=0): {R_l0_t0.mean():.3f} +/- {R_l0_t0.std():.3f}")
        print(f"    H (t=0): {H_t0.mean():.3f} +/- {H_t0.std():.3f}")
        print(f"    H (t=1): {H_t1.mean():.3f} +/- {H_t1.std():.3f}")

        if all_rho:
            rho = np.stack(all_rho)
            rho_t1 = rho[:, -1, :, :].mean(axis=(-1, -2))
            print(f"    rho (t=1): {rho_t1.mean():.3f} +/- {rho_t1.std():.3f}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=list(MODELS.keys()), default=None,
                        help="Run only this model (default: all available)")
    parser.add_argument("--aggregate-only", action="store_true")
    args = parser.parse_args()

    if args.aggregate_only:
        aggregate()
    else:
        models = [args.model] if args.model else None
        run(models)
        aggregate()
