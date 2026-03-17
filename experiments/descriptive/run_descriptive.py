#!/usr/bin/env python
"""
Unified descriptive experiment runner.

Runs core metrics (R, R_c, H, rho), sequence separation analysis,
and contact precision for a given model with configurable seeds.

Usage:
    # Run all seeds
    python experiments/run_descriptive.py --model 60m

    # Aggregate only (no generation)
    python experiments/run_descriptive.py --model 60m --aggregate-only

    # Custom output dir
    python experiments/run_descriptive.py --model 200m_notri --output-dir experiments/descriptive/200m_notri/exp10_custom/artifacts
"""

import subprocess
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent
while not (REPO_ROOT / ".git").exists():
    REPO_ROOT = REPO_ROOT.parent

SEEDS = [5, 42, 123, 256, 999]
PROTEIN_LENGTH = 100

MODEL_CONFIGS = {
    "60m": {
        "config_name": "inference_ucond_60m_notri",
        "ckpt_path": "checkpoints/proteina_v1.3_dfs_60m_notri_v1.0",
        "label": "60M (v1.3, 12L, 12H, no tri)",
    },
    "200m_notri": {
        "config_name": "inference_ucond_200m_notri",
        "ckpt_path": "checkpoints/proteina_v1.2_dfs_200m_notri_v1.0",
        "label": "200M no-tri (v1.2, 15L, 12H)",
    },
    "200m_tri": {
        "config_name": "inference_ucond_200m_tri",
        "ckpt_path": "checkpoints/proteina_v1.1_dfs_200m_tri_v1.0",
        "label": "200M tri (v1.1, 15L, 12H)",
    },
    "400m_tri": {
        "config_name": "inference_ucond_400m_tri",
        "ckpt_path": "checkpoints/proteina_v1.4_d21m_400m_tri_v1.0",
        "label": "400M tri (v1.4, 18L, 16H)",
    },
}


def _compute_random_precision(pdb_path: Path, min_seqsep: int = 6,
                               contact_thresh: float = 0.8):
    """Compute random contact precision baseline from a PDB file.

    Random precision = num_contacts / num_eligible_pairs, where contacts
    are CA-CA distance < contact_thresh nm with |i-j| >= min_seqsep.
    This is the expected Precision@k for random attention rankings.
    """
    # Parse CA coordinates from PDB (coordinates in Angstroms, convert to nm)
    ca_coords = []
    with open(pdb_path) as f:
        for line in f:
            if line.startswith(("ATOM", "HETATM")) and line[12:16].strip() == "CA":
                x = float(line[30:38])
                y = float(line[38:46])
                z = float(line[46:54])
                ca_coords.append([x / 10.0, y / 10.0, z / 10.0])  # A -> nm
    if len(ca_coords) < min_seqsep + 1:
        return None
    coords = np.array(ca_coords)
    n = len(coords)

    # Pairwise distances
    diff = coords[:, None, :] - coords[None, :, :]
    dists = np.sqrt((diff ** 2).sum(axis=-1))

    # Eligible pairs: upper triangle with |i-j| >= min_seqsep
    i_idx, j_idx = np.meshgrid(np.arange(n), np.arange(n), indexing='ij')
    eligible = (j_idx - i_idx) >= min_seqsep
    num_eligible = eligible.sum()
    num_contacts = ((dists < contact_thresh) & eligible).sum()

    if num_eligible == 0:
        return None
    return num_contacts / num_eligible


def run(model: str, output_dir: Path, seeds: list = SEEDS):
    """Run crystallization analysis for each seed."""
    cfg = MODEL_CONFIGS[model]
    ckpt_path = str(REPO_ROOT / cfg["ckpt_path"])

    for seed in seeds:
        seed_dir = output_dir / f"seed_{seed}"
        seed_dir.mkdir(parents=True, exist_ok=True)

        # Skip if metrics already exist
        if (seed_dir / "crystallization_metrics.npz").exists():
            print(f"Seed {seed} already exists, skipping. Delete artifacts to re-run.")
            continue

        cmd = [
            sys.executable,
            str(REPO_ROOT / "script_utils" / "crystallization_analysis.py"),
            "--config_name", cfg["config_name"],
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
        print(f"[{cfg['label']}] Running seed {seed}...")
        print(f"{'='*60}")
        subprocess.run(cmd, check=True)

    print(f"\nAll seeds complete. Artifacts in: {output_dir}")


def aggregate(model: str, output_dir: Path, seeds: list = SEEDS):
    """Load metrics from all seeds and compute mean/std."""
    sys.path.insert(0, str(REPO_ROOT))
    from proteinfoundation.analysis import (
        TrajectoryMetrics, SeqsepMetrics, ContactPrecisionMetrics,
    )

    cfg = MODEL_CONFIGS[model]

    all_R, all_Rc, all_H, all_rho = [], [], [], []
    all_seqsep_R, all_seqsep_Rc = [], []
    all_prec_full, all_prec_b, all_prec_c = [], [], []

    for seed in seeds:
        seed_dir = output_dir / f"seed_{seed}"

        core_path = seed_dir / "crystallization_metrics.npz"
        if core_path.exists():
            m = TrajectoryMetrics.load(str(core_path))
            all_R.append(m.logit_dominance)
            all_H.append(m.entropy)
            if m.logit_dominance_centered is not None:
                all_Rc.append(m.logit_dominance_centered)
            if m.spatial_alignment is not None:
                all_rho.append(m.spatial_alignment)

        seqsep_path = seed_dir / "seqsep_metrics.npz"
        if seqsep_path.exists():
            s = SeqsepMetrics.load(str(seqsep_path))
            all_seqsep_R.append(s.logit_dominance)
            if s.logit_dominance_centered is not None:
                all_seqsep_Rc.append(s.logit_dominance_centered)

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
    print(f"{cfg['label']} — Aggregated results across {n_seeds} seeds")
    print(f"{'='*60}")

    R = np.stack(all_R)
    H = np.stack(all_H)

    R_l0_t0 = R[:, 0, 0, :].mean(axis=-1)
    R_l0_t1 = R[:, -1, 0, :].mean(axis=-1)
    print(f"\nLogit Dominance (R) — Layer 0:")
    print(f"  t=0: {R_l0_t0.mean():.3f} +/- {R_l0_t0.std():.3f}")
    print(f"  t=1: {R_l0_t1.mean():.3f} +/- {R_l0_t1.std():.3f}")

    if all_Rc:
        Rc = np.stack(all_Rc)
        Rc_l0_t0 = Rc[:, 0, 0, :].mean(axis=-1)
        Rc_l0_t1 = Rc[:, -1, 0, :].mean(axis=-1)
        print(f"\nRow-Centered Logit Dominance (R_c) — Layer 0:")
        print(f"  t=0: {Rc_l0_t0.mean():.3f} +/- {Rc_l0_t0.std():.3f}")
        print(f"  t=1: {Rc_l0_t1.mean():.3f} +/- {Rc_l0_t1.std():.3f}")

    H_all_t0 = H[:, 0, :, :].mean(axis=(-1, -2))
    H_all_t1 = H[:, -1, :, :].mean(axis=(-1, -2))
    print(f"\nEntropy (H):")
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

    # Compute random precision baseline from generated structures
    all_random_prec = []
    for seed in seeds[:n_seeds]:
        pdb_path = output_dir / f"seed_{seed}" / "generated_structure.pdb"
        if pdb_path.exists():
            rp = _compute_random_precision(pdb_path, min_seqsep=6, contact_thresh=0.8)
            if rp is not None:
                all_random_prec.append(rp)
    if all_random_prec:
        rp_arr = np.array(all_random_prec)
        print(f"\nRandom baseline (contact density, |i-j|>=6, <0.8nm):")
        print(f"  {rp_arr.mean():.4f} +/- {rp_arr.std():.4f}")

    # Save aggregated
    agg_path = output_dir / "aggregated.npz"
    save_dict = {
        'model': model,
        'seeds': np.array(seeds[:n_seeds]),
        'timesteps': m.timesteps,
        'R_mean': R.mean(axis=0), 'R_std': R.std(axis=0),
        'H_mean': H.mean(axis=0), 'H_std': H.std(axis=0),
    }
    if all_Rc:
        save_dict['Rc_mean'] = Rc.mean(axis=0)
        save_dict['Rc_std'] = Rc.std(axis=0)
    if all_rho:
        save_dict['rho_mean'] = rho.mean(axis=0)
        save_dict['rho_std'] = rho.std(axis=0)
    if all_seqsep_R:
        save_dict['seqsep_R_mean'] = seqsep_R.mean(axis=0)
        save_dict['seqsep_R_std'] = seqsep_R.std(axis=0)
    if all_seqsep_Rc:
        seqsep_Rc = np.stack(all_seqsep_Rc)
        save_dict['seqsep_Rc_mean'] = seqsep_Rc.mean(axis=0)
        save_dict['seqsep_Rc_std'] = seqsep_Rc.std(axis=0)
    if all_prec_full:
        save_dict['prec_full_mean'] = prec_full.mean(axis=0)
        save_dict['prec_full_std'] = prec_full.std(axis=0)
        save_dict['prec_b_mean'] = prec_b.mean(axis=0)
        save_dict['prec_b_std'] = prec_b.std(axis=0)
        save_dict['prec_c_mean'] = prec_c.mean(axis=0)
        save_dict['prec_c_std'] = prec_c.std(axis=0)
    if all_random_prec:
        rp_arr = np.array(all_random_prec)
        save_dict['random_precision_mean'] = rp_arr.mean()
        save_dict['random_precision_std'] = rp_arr.std()
    np.savez(str(agg_path), **save_dict)
    print(f"\nAggregated metrics saved to {agg_path}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True, choices=list(MODEL_CONFIGS.keys()),
                        help="Model to run")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Output directory (default: experiments/descriptive/<model>/run_<date>/artifacts)")
    parser.add_argument("--aggregate-only", action="store_true",
                        help="Just aggregate existing results, no generation")
    parser.add_argument("--seeds", type=int, nargs="+", default=SEEDS,
                        help=f"Seeds to use (default: {SEEDS})")
    args = parser.parse_args()

    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        from datetime import date
        today = date.today().strftime("%Y-%m-%d")
        output_dir = Path(f"experiments/descriptive/{args.model}/run_{today}/artifacts")

    output_dir.mkdir(parents=True, exist_ok=True)

    if args.aggregate_only:
        aggregate(args.model, output_dir, args.seeds)
    else:
        run(args.model, output_dir, args.seeds)
        aggregate(args.model, output_dir, args.seeds)
