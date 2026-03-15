#!/usr/bin/env python
"""
Experiment: Causal Ablation of Pair Bias B — 200M model (tri)
==============================================================

Date: 2026-03-15
Model: proteina_v1.1_DFS_200M_tri (200M params, 15 layers, 12 heads, triangle updates)
Protein length: 100 residues
Seeds: [42, 123, 256]

Purpose: Replicate the 60M bias ablation on the 200M model.
Tests whether B is causally necessary throughout generation.

Ablation conditions (deep_ablation adapts to model layer count):
1. baseline       — No ablation (control)
2. full_ablation  — B=0 everywhere
3. early_only_B   — B active for t <= 0.5, then B=0
4. late_only_B    — B=0 for t <= 0.5, then B active
5. layer0_ablation — B=0 at layer 0 only
6. deep_ablation   — B=0 at deep layers (last half)

For 200M (15 layers): deep_ablation = layers 8-14
(analogous to layers 6-11 in the 12-layer 60M model — last ~half)
"""

import subprocess
import sys
from pathlib import Path
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parent
while not (REPO_ROOT / ".git").exists():
    REPO_ROOT = REPO_ROOT.parent
EXPERIMENT_NAME = "2026-03-15-bias-ablation-200m-tri"
ARTIFACTS_BASE = Path(__file__).resolve().parent / "artifacts"
DEFAULT_CKPT = "checkpoints/proteina_v1.1_dfs_200m_tri_v1.0"
CONFIG_NAME = "inference_ucond_200m_tri"

SEEDS = [42, 123, 256]
PROTEIN_LENGTH = 100
NUM_SAMPLES = 1


@dataclass
class AblationCondition:
    name: str
    description: str
    ablate_layers: Optional[set] = None  # None = all layers
    ablate_t_min: float = 0.0
    ablate_t_max: float = 1.0
    enabled: bool = True


def build_conditions(num_layers: int) -> list[AblationCondition]:
    """Build ablation conditions with layer sets adapted to model size."""
    deep_start = num_layers // 2 + 1  # e.g., 8 for 15 layers
    deep_layers = set(range(deep_start, num_layers))
    return [
        AblationCondition(
            name="baseline",
            description="No ablation (control)",
            enabled=False,
        ),
        AblationCondition(
            name="full_ablation",
            description="B=0 at all layers, all timesteps",
        ),
        AblationCondition(
            name="early_only_B",
            description="B active for t<=0.5, then B=0 for t>0.5",
            ablate_t_min=0.5,
            ablate_t_max=1.0,
        ),
        AblationCondition(
            name="late_only_B",
            description="B=0 for t<=0.5, then B active for t>0.5",
            ablate_t_min=0.0,
            ablate_t_max=0.5,
        ),
        AblationCondition(
            name="layer0_ablation",
            description="B=0 at layer 0 only, all timesteps",
            ablate_layers={0},
        ),
        AblationCondition(
            name="deep_ablation",
            description=f"B=0 at layers {deep_start}-{num_layers-1}, all timesteps",
            ablate_layers=deep_layers,
        ),
    ]


def compute_structural_metrics(coords: torch.Tensor, mask: torch.Tensor) -> dict:
    """
    Compute lightweight structural quality metrics from CA coordinates.

    Args:
        coords: CA coordinates, shape [n, 3] (in nm)
        mask: sequence mask, shape [n]

    Returns:
        Dict of metric_name -> value
    """
    coords = coords[mask.bool()].float()
    n = coords.shape[0]

    # Radius of gyration
    centroid = coords.mean(dim=0)
    rg = torch.sqrt(((coords - centroid) ** 2).sum(dim=-1).mean()).item()

    # Pairwise CA distance matrix
    dists = torch.cdist(coords, coords)

    # Contact density (fraction of pairs within 0.8 nm with |i-j| >= 6)
    i_idx = torch.arange(n).unsqueeze(1)
    j_idx = torch.arange(n).unsqueeze(0)
    seqsep_mask = (i_idx - j_idx).abs() >= 6
    upper_mask = (i_idx < j_idx) & seqsep_mask
    long_range_dists = dists[upper_mask]
    contact_density = (long_range_dists < 0.8).float().mean().item()

    # Mean and std of pairwise distances
    upper_dists = dists[torch.triu(torch.ones(n, n), diagonal=1).bool()]
    mean_dist = upper_dists.mean().item()
    std_dist = upper_dists.std().item()

    # End-to-end distance
    end_to_end = torch.norm(coords[-1] - coords[0]).item()

    # Local bond distances (CA-CA, should be ~0.38 nm)
    bond_dists = torch.norm(coords[1:] - coords[:-1], dim=-1)
    mean_bond = bond_dists.mean().item()
    std_bond = bond_dists.std().item()

    # Clash count (CA-CA < 0.2 nm with |i-j| >= 3)
    seqsep3_mask = (i_idx - j_idx).abs() >= 3
    clash_mask = (dists < 0.2) & seqsep3_mask & (dists > 0)
    n_clashes = clash_mask.sum().item() // 2  # symmetric

    return {
        "rg": rg,
        "contact_density": contact_density,
        "mean_dist": mean_dist,
        "std_dist": std_dist,
        "end_to_end": end_to_end,
        "mean_bond": mean_bond,
        "std_bond": std_bond,
        "n_clashes": n_clashes,
        "n_residues": n,
    }


def run(ckpt_path: str = DEFAULT_CKPT):
    """Run bias ablation experiments."""
    sys.path.insert(0, str(REPO_ROOT))

    import os
    os.chdir(str(REPO_ROOT))

    from dotenv import load_dotenv
    load_dotenv()

    import lightning as L
    from hydra import compose, initialize_config_dir
    from proteinfoundation.analysis import CrystallizationTracker, BiasAblationConfig
    from proteinfoundation.proteinflow.proteina import Proteina
    from proteinfoundation.utils.ff_utils.pdb_utils import write_prot_to_pdb
    from proteinfoundation.utils.coors_utils import trans_nm_to_atom37

    config_path = str(REPO_ROOT / "configs" / "experiment_config")
    with initialize_config_dir(config_dir=config_path, version_base=None):
        cfg = compose(config_name=CONFIG_NAME)

    from omegaconf import OmegaConf
    OmegaConf.update(cfg, "ckpt_path", ckpt_path)

    ckpt_file = os.path.join(cfg.ckpt_path, cfg.ckpt_name)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Loading model from {ckpt_file}...")
    model = Proteina.load_from_checkpoint(ckpt_file)
    model.eval()
    model.to(device)

    num_layers = model.nn.nlayers
    num_heads = model.nn.transformer_layers[0].mhba.mha.heads
    print(f"Model loaded ({num_layers} layers, {num_heads} heads)")

    # Build conditions dynamically based on actual layer count
    conditions = build_conditions(num_layers)
    print(f"Deep ablation layers: {sorted(conditions[-1].ablate_layers)}")

    sampling_args = cfg.sampling_caflow

    all_results = {}

    for condition in conditions:
        print(f"\n{'='*60}")
        print(f"Condition: {condition.name} — {condition.description}")
        print(f"{'='*60}")

        cond_dir = ARTIFACTS_BASE / condition.name
        cond_dir.mkdir(parents=True, exist_ok=True)

        seed_results = []

        for seed in SEEDS:
            print(f"\n  Seed {seed}...")
            L.seed_everything(seed, workers=True)

            # Setup tracker with ablation config
            tracker = CrystallizationTracker()
            tracker.bias_ablation = BiasAblationConfig(
                enabled=condition.enabled,
                ablate_layers=condition.ablate_layers,
                ablate_t_min=condition.ablate_t_min,
                ablate_t_max=condition.ablate_t_max,
            )
            tracker.enable(capture_every_n=1000, move_to_cpu=True)

            mask = torch.ones(1, PROTEIN_LENGTH, dtype=torch.bool, device=device)

            with torch.no_grad():
                samples = model.generate_with_analysis(
                    nsamples=1,
                    n=PROTEIN_LENGTH,
                    dt=0.01,
                    self_cond=cfg.get("self_cond", True),
                    cath_code=None,
                    tracker=tracker,
                    mask=mask,
                    schedule_mode=cfg.schedule.schedule_mode,
                    schedule_p=cfg.schedule.schedule_p,
                    sampling_mode=sampling_args["sampling_mode"],
                    sc_scale_noise=sampling_args["sc_scale_noise"],
                    sc_scale_score=sampling_args["sc_scale_score"],
                    gt_mode=sampling_args["gt_mode"],
                    gt_p=sampling_args["gt_p"],
                    gt_clamp_val=sampling_args["gt_clamp_val"],
                )

            # Compute structural metrics
            metrics = compute_structural_metrics(
                samples[0].cpu(), mask[0].cpu()
            )
            metrics["seed"] = seed
            seed_results.append(metrics)

            # Save PDB
            atom37 = trans_nm_to_atom37(samples[:1].cpu())
            pdb_path = cond_dir / f"seed_{seed}.pdb"
            write_prot_to_pdb(atom37[0].numpy(), str(pdb_path), overwrite=True, no_indexing=True)

            print(f"    Rg={metrics['rg']:.3f} nm, contacts={metrics['contact_density']:.3f}, "
                  f"clashes={metrics['n_clashes']}, bond={metrics['mean_bond']:.3f}±{metrics['std_bond']:.3f}")

        all_results[condition.name] = seed_results

        # Save per-condition results
        np.savez(
            str(cond_dir / "metrics.npz"),
            **{k: np.array([r[k] for r in seed_results]) for k in seed_results[0]}
        )

    # Print summary table
    print_summary(all_results)
    save_summary(all_results)


def print_summary(all_results: dict):
    """Print a summary table of ablation results."""
    print(f"\n{'='*80}")
    print(f"{'Condition':<20} {'Rg (nm)':<14} {'Contact%':<14} {'Clashes':<12} {'Bond (nm)':<14}")
    print(f"{'='*80}")

    for name, results in all_results.items():
        rg = np.array([r["rg"] for r in results])
        cd = np.array([r["contact_density"] for r in results])
        cl = np.array([r["n_clashes"] for r in results])
        bd = np.array([r["mean_bond"] for r in results])

        print(f"{name:<20} {rg.mean():.3f}±{rg.std():.3f}    "
              f"{cd.mean():.3f}±{cd.std():.3f}    "
              f"{cl.mean():.0f}±{cl.std():.0f}       "
              f"{bd.mean():.3f}±{bd.std():.3f}")

    print(f"{'='*80}")


def save_summary(all_results: dict):
    """Save summary to a text file."""
    summary_path = ARTIFACTS_BASE / "summary.txt"
    with open(summary_path, "w") as f:
        f.write(f"Bias Ablation Experiment Summary\n")
        f.write(f"Model: 200M (tri), n={PROTEIN_LENGTH}, seeds={SEEDS}\n\n")

        f.write(f"{'Condition':<20} {'Rg (nm)':<14} {'Contact%':<14} {'Clashes':<12} {'Bond (nm)':<14}\n")
        f.write("-" * 80 + "\n")

        for name, results in all_results.items():
            rg = np.array([r["rg"] for r in results])
            cd = np.array([r["contact_density"] for r in results])
            cl = np.array([r["n_clashes"] for r in results])
            bd = np.array([r["mean_bond"] for r in results])

            f.write(f"{name:<20} {rg.mean():.3f}±{rg.std():.3f}    "
                    f"{cd.mean():.3f}±{cd.std():.3f}    "
                    f"{cl.mean():.0f}±{cl.std():.0f}       "
                    f"{bd.mean():.3f}±{bd.std():.3f}\n")

    print(f"\nSummary saved to {summary_path}")


def load_and_summarize():
    """Load existing results and print summary."""
    conditions = build_conditions(15)  # Default for 200M
    all_results = {}
    for condition in conditions:
        cond_dir = ARTIFACTS_BASE / condition.name
        metrics_path = cond_dir / "metrics.npz"
        if metrics_path.exists():
            data = np.load(str(metrics_path))
            results = []
            for i in range(len(data["seed"])):
                results.append({k: data[k][i] for k in data.files})
            all_results[condition.name] = results

    if not all_results:
        print("No results found. Run the experiment first.")
        return

    print_summary(all_results)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt_path", default=DEFAULT_CKPT)
    parser.add_argument("--load-only", action="store_true",
                        help="Just load and summarize existing results")
    args = parser.parse_args()

    if args.load_only:
        load_and_summarize()
    else:
        run(args.ckpt_path)
