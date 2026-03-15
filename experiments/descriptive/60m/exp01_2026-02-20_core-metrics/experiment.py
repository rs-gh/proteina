#!/usr/bin/env python
"""
Experiment 1: Unconditional 60M model, n=100
=============================================

Date: 2026-02-20
Model: proteina_v1.3_DFS_60M_notri (60M params, no triangle updates)
Protein length: 100 residues
Sampling: SDE (sc), noise scale 0.45, log schedule, dt=0.01 (100 steps)
Capture: every 5th timestep -> 20 snapshots x 12 layers x 12 heads
Seed: 5 (config default)

Note: The original artifacts in artifacts/ were generated before seed control was
added to the analysis script, so they used an uncontrolled random state. Re-running
this script with --seed 5 will produce a different (but now reproducible) structure.

Run this script from the repo root to reproduce the experiment.
Artifacts are saved to the corresponding artifacts directory.
"""

import subprocess
import sys
from pathlib import Path

# Paths
REPO_ROOT = Path(__file__).resolve().parent
while not (REPO_ROOT / ".git").exists():
    REPO_ROOT = REPO_ROOT.parent
EXPERIMENT_NAME = "2026-02-20-1731-ucond-60m-n100"
ARTIFACTS_DIR = Path(__file__).resolve().parent / "artifacts"

# Default checkpoint path (override with --ckpt_path)
DEFAULT_CKPT = "checkpoints/proteina_v1.3_dfs_60m_notri_v1.0"


def run(ckpt_path: str = DEFAULT_CKPT):
    """Run the crystallization analysis experiment."""
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable,
        str(REPO_ROOT / "script_utils" / "crystallization_analysis.py"),
        "--config_name", "inference_ucond_60m_notri",
        "--ckpt_path", ckpt_path,
        "--protein_length", "100",
        "--output_dir", str(ARTIFACTS_DIR),
        "--capture_every_n", "5",
        "--dt", "0.01",
        "--seed", "5",
    ]

    print(f"Running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)
    print(f"\nArtifacts saved to: {ARTIFACTS_DIR}")


def load_metrics():
    """Load saved metrics from artifacts directory."""
    import numpy as np

    npz_path = ARTIFACTS_DIR / "crystallization_metrics.npz"
    if not npz_path.exists():
        raise FileNotFoundError(
            f"Metrics not found at {npz_path}. Run the experiment first."
        )

    metrics = np.load(str(npz_path))
    return {
        "timesteps": metrics["timesteps"],           # [20]
        "logit_dominance": metrics["logit_dominance"],  # [20, 12, 12] — R = ||B||_F / ||C||_F
        "entropy": metrics["entropy"],                  # [20, 12, 12] — Shannon entropy H
        "spatial_alignment": metrics["spatial_alignment"],  # [20, 12, 12] — Pearson corr rho
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--ckpt_path", default=DEFAULT_CKPT,
        help="Checkpoint directory path",
    )
    parser.add_argument(
        "--load-only", action="store_true",
        help="Just load and print metrics summary (don't re-run)",
    )
    args = parser.parse_args()

    if args.load_only:
        m = load_metrics()
        R, H, rho = m["logit_dominance"], m["entropy"], m["spatial_alignment"]
        print(f"Timesteps: {len(m['timesteps'])} (t={m['timesteps'][0]:.3f} to {m['timesteps'][-1]:.3f})")
        print(f"R (logit dominance):  early={R[0].mean():.3f}, late={R[-1].mean():.3f}")
        print(f"H (entropy):          early={H[0].mean():.3f}, late={H[-1].mean():.3f}")
        print(f"rho (spatial align):  early={rho[0].mean():.3f}, late={rho[-1].mean():.3f}")
    else:
        run(args.ckpt_path)
