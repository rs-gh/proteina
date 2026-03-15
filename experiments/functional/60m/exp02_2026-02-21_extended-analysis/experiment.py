#!/usr/bin/env python
"""
Experiment 2: Extended Analysis — Per-Head, Seqsep, Contact Precision, Registers
==================================================================================

Date: 2026-02-21
Model: proteina_v1.3_DFS_60M_notri (60M params, 10 layers, 8 heads, 10 register tokens)
Protein length: 100 residues (same as Exp 1 for direct comparison)
Sampling: SDE (sc), dt=0.01 (100 steps), seed=5
Capture: every 5th timestep -> 20 snapshots × 10 layers × 8 heads

New analyses (building on Exp 1 infrastructure):

  Exp 2: Per-head trajectory plots — are heads specialized?
         Data: metrics.logit_dominance/entropy/spatial_alignment [T, L, H]
         Plots: per_head/per_head_entropy_layer{i}.png

  Exp 3: Sequence-separation decomposition — does geometry matter more for long-range?
         Data: seqsep_metrics.npz [3 bins × T × L × H]
         Plot: seqsep_decomposition.png
         Key hypothesis: R higher in long-range (>=24) bin throughout trajectory

  Exp 4: Contact Precision@L/5 — does C or B independently predict contacts?
         Data: contact_precision.npz [T × L × H] × 3 (full/B/C)
         Plot: contact_precision.png
         Key hypotheses:
           - precision_b > precision_c at early timesteps (B has geometric info, C doesn't)
           - precision_c grows during denoising (C learns structure as trajectory progresses)
           - precision_full >= max(precision_b, precision_c) always

  Exp 5: Register token attention — do registers act as attention sinks?
         Data: register_metrics.npz [T × L × H]
         Key hypothesis: register attention fraction peaks in middle layers (Compress phase)
         from Mix-Compress-Refine framework (Ding et al. 2024, arXiv 2510.06477)

  Exp 7: AF2 Figure 12-style attention grid
         Plots: attention_grid_{attn,bias,content}.png
         Shows crystallization visually: diffuse (left, t~0) → sharp contacts (right, t~1)

Run this script from the repo root to reproduce.
"""

import subprocess
import sys
from pathlib import Path

# Paths
REPO_ROOT = Path(__file__).resolve().parent
while not (REPO_ROOT / ".git").exists():
    REPO_ROOT = REPO_ROOT.parent
EXPERIMENT_NAME = "2026-02-21-extended-analysis-60m-n100"
ARTIFACTS_DIR = Path(__file__).resolve().parent / "artifacts"

DEFAULT_CKPT = "checkpoints/proteina_v1.3_dfs_60m_notri_v1.0"


def run(ckpt_path: str = DEFAULT_CKPT):
    """Run all extended analyses for the 60M model, n=100."""
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
        # Extended analyses
        "--compute_seqsep",
        "--compute_contact_precision",
        "--compute_register_metrics",
        "--plot_grid",
    ]

    print(f"Running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)
    print(f"\nArtifacts saved to: {ARTIFACTS_DIR}")


def load_all():
    """Load all saved metric files and print a summary."""
    import numpy as np
    from proteinfoundation.analysis import TrajectoryMetrics, SeqsepMetrics, ContactPrecisionMetrics, RegisterMetrics

    results = {}

    # Core metrics (Exp 1 equivalent)
    core_path = ARTIFACTS_DIR / "crystallization_metrics.npz"
    if core_path.exists():
        m = TrajectoryMetrics.load(str(core_path))
        print(m.summary())
        results['core'] = m

    # Seqsep metrics
    seqsep_path = ARTIFACTS_DIR / "seqsep_metrics.npz"
    if seqsep_path.exists():
        s = SeqsepMetrics.load(str(seqsep_path))
        print(f"\nSeqsep Decomposition (R = ||B||/||C||):")
        for bin_idx, label in enumerate(s.bin_labels):
            R_early = s.logit_dominance[bin_idx, 0].mean()
            R_late = s.logit_dominance[bin_idx, -1].mean()
            print(f"  {label:20s}: R early={R_early:.3f}, R late={R_late:.3f}")
        results['seqsep'] = s

    # Contact precision
    prec_path = ARTIFACTS_DIR / "contact_precision.npz"
    if prec_path.exists():
        p = ContactPrecisionMetrics.load(str(prec_path))
        print(f"\nContact Precision@{p.k} at t=1 (mean over layers/heads):")
        print(f"  Full (C+B):  {p.precision_full[-1].mean():.3f}")
        print(f"  B-only:      {p.precision_b_only[-1].mean():.3f}")
        print(f"  C-only:      {p.precision_c_only[-1].mean():.3f}")
        print(f"\nContact Precision@{p.k} at t=0 (mean over layers/heads):")
        print(f"  Full (C+B):  {p.precision_full[0].mean():.3f}")
        print(f"  B-only:      {p.precision_b_only[0].mean():.3f}")
        print(f"  C-only:      {p.precision_c_only[0].mean():.3f}")
        results['precision'] = p

    # Register metrics
    reg_path = ARTIFACTS_DIR / "register_metrics.npz"
    if reg_path.exists():
        r = RegisterMetrics.load(str(reg_path))
        print(f"\nRegister Token Attention (n_registers={r.num_registers}):")
        print(f"  Mean fraction: {r.register_attn_fraction.mean():.3f}")
        print(f"  Expected (uniform): {r.num_registers / (r.protein_length + r.num_registers):.3f}")
        # Which layer has highest register attention (should be the "Compress" phase)?
        layer_mean = r.register_attn_fraction.mean(axis=(0, 2))  # mean over T and H → [L]
        peak_layer = layer_mean.argmax()
        print(f"  Peak register attention at layer {peak_layer} (Mix-Compress-Refine Compress phase)")
        results['register'] = r

    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt_path", default=DEFAULT_CKPT)
    parser.add_argument("--load-only", action="store_true",
                        help="Just load and print metrics summary")
    args = parser.parse_args()

    if args.load_only:
        load_all()
    else:
        run(args.ckpt_path)
