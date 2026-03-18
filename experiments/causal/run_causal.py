#!/usr/bin/env python
"""
Unified causal ablation experiment runner.

Runs bias ablation experiments (6 conditions x 3 seeds) for a given model.

Usage:
    python experiments/run_causal.py --model 400m_tri
    python experiments/run_causal.py --model 60m --load-only
"""

import subprocess
import sys
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, Set

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parent
while not (REPO_ROOT / ".git").exists():
    REPO_ROOT = REPO_ROOT.parent

SEEDS = [42, 123, 256]
PROTEIN_LENGTH = 100

MODEL_CONFIGS = {
    "60m": {
        "config_name": "inference_ucond_60m_notri",
        "ckpt_path": "checkpoints/proteina_v1.3_dfs_60m_notri_v1.0",
        "label": "60M (v1.3, 12L)",
        "num_layers": 12,
        "deep_layers": set(range(6, 12)),       # layers 6-11
        "first_half_layers": set(range(0, 6)),  # layers 0-5
    },
    "200m_notri": {
        "config_name": "inference_ucond_200m_notri",
        "ckpt_path": "checkpoints/proteina_v1.2_dfs_200m_notri_v1.0",
        "label": "200M no-tri (v1.2, 15L)",
        "num_layers": 15,
        "deep_layers": set(range(8, 15)),       # layers 8-14
        "first_half_layers": set(range(0, 8)),  # layers 0-7
    },
    "200m_tri": {
        "config_name": "inference_ucond_200m_tri",
        "ckpt_path": "checkpoints/proteina_v1.1_dfs_200m_tri_v1.0",
        "label": "200M tri (v1.1, 15L)",
        "num_layers": 15,
        "deep_layers": set(range(8, 15)),       # layers 8-14
        "first_half_layers": set(range(0, 8)),  # layers 0-7
    },
    "400m_tri": {
        "config_name": "inference_ucond_400m_tri",
        "ckpt_path": "checkpoints/proteina_v1.4_d21m_400m_tri_v1.0",
        "label": "400M tri (v1.4, 18L)",
        "num_layers": 18,
        "deep_layers": set(range(10, 18)),       # layers 10-17
        "first_half_layers": set(range(0, 10)), # layers 0-9
    },
}


@dataclass
class AblationCondition:
    name: str
    description: str
    ablate_layers: Optional[Set[int]] = None  # None = all layers
    ablate_t_min: float = 0.0
    ablate_t_max: float = 1.0
    enabled: bool = True
    mode: str = 'zero'  # 'zero' or 'random'


def get_conditions(model: str):
    cfg = MODEL_CONFIGS[model]
    deep_layers = cfg["deep_layers"]
    first_half_layers = cfg["first_half_layers"]
    deep_desc = f"layers {min(deep_layers)}-{max(deep_layers)}"
    first_half_desc = f"layers {min(first_half_layers)}-{max(first_half_layers)}"
    return [
        AblationCondition(name="baseline", description="No ablation (control)", enabled=False),
        AblationCondition(name="full_ablation", description="B=0 everywhere"),
        AblationCondition(name="early_only_B", description="B active for t<=0.5, then B=0",
                          ablate_t_min=0.5, ablate_t_max=1.0),
        AblationCondition(name="late_only_B", description="B=0 for t<=0.5, then B active",
                          ablate_t_min=0.0, ablate_t_max=0.5),
        AblationCondition(name="first_half_ablation", description=f"B=0 at {first_half_desc}",
                          ablate_layers=first_half_layers),
        AblationCondition(name="deep_ablation", description=f"B=0 at {deep_desc}",
                          ablate_layers=deep_layers),
    ]


def compute_structural_metrics(coords: torch.Tensor, mask: torch.Tensor) -> dict:
    """Compute structural quality metrics from CA coordinates."""
    coords = coords[mask.bool()].float()
    n = coords.shape[0]

    centroid = coords.mean(dim=0)
    rg = torch.sqrt(((coords - centroid) ** 2).sum(dim=-1).mean()).item()

    dists = torch.cdist(coords, coords)
    i_idx = torch.arange(n).unsqueeze(1)
    j_idx = torch.arange(n).unsqueeze(0)
    seqsep_mask = (i_idx - j_idx).abs() >= 6
    upper_mask = (i_idx < j_idx) & seqsep_mask
    long_range_dists = dists[upper_mask]
    contact_density = (long_range_dists < 0.8).float().mean().item()

    bond_dists = torch.norm(coords[1:] - coords[:-1], dim=-1)
    mean_bond = bond_dists.mean().item()

    seqsep3_mask = (i_idx - j_idx).abs() >= 3
    clash_mask = (dists < 0.2) & seqsep3_mask & (dists > 0)
    n_clashes = clash_mask.sum().item() // 2

    return {
        "rg": rg,
        "contact_density": contact_density,
        "mean_bond": mean_bond,
        "n_clashes": n_clashes,
        "n_residues": n,
    }


def run(model: str, output_dir: Path, seeds: list = SEEDS, conditions=None,
        protein_length: int = PROTEIN_LENGTH):
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

    cfg_model = MODEL_CONFIGS[model]
    config_path = str(REPO_ROOT / "configs" / "experiment_config")
    with initialize_config_dir(config_dir=config_path, version_base=None):
        cfg = compose(config_name=cfg_model["config_name"])

    from omegaconf import OmegaConf
    ckpt_path = str(REPO_ROOT / cfg_model["ckpt_path"])
    OmegaConf.update(cfg, "ckpt_path", ckpt_path)

    ckpt_file = os.path.join(cfg.ckpt_path, cfg.ckpt_name)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Loading model from {ckpt_file}...")
    model_net = Proteina.load_from_checkpoint(ckpt_file)
    model_net.eval()
    model_net.to(device)

    sampling_args = cfg.sampling_caflow
    if conditions is None:
        conditions = get_conditions(model)
    all_results = {}

    for condition in conditions:
        print(f"\n{'='*60}")
        print(f"Condition: {condition.name} — {condition.description}")
        print(f"{'='*60}")

        cond_dir = output_dir / condition.name
        cond_dir.mkdir(parents=True, exist_ok=True)

        if (cond_dir / "metrics.npz").exists():
            print(f"  Already exists, skipping. Delete to re-run.")
            data = np.load(str(cond_dir / "metrics.npz"))
            results = [{k: data[k][i] for k in data.files} for i in range(len(data["rg"]))]
            all_results[condition.name] = results
            continue

        seed_results = []

        for seed in seeds:
            print(f"\n  Seed {seed}...")
            L.seed_everything(seed, workers=True)

            tracker = CrystallizationTracker()
            tracker.bias_ablation = BiasAblationConfig(
                enabled=condition.enabled,
                ablate_layers=condition.ablate_layers,
                ablate_t_min=condition.ablate_t_min,
                ablate_t_max=condition.ablate_t_max,
                mode=condition.mode,
            )
            tracker.enable(capture_every_n=1000, move_to_cpu=True)
            mask = torch.ones(1, protein_length, dtype=torch.bool, device=device)

            with torch.no_grad():
                samples = model_net.generate_with_analysis(
                    nsamples=1,
                    n=protein_length,
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

            metrics = compute_structural_metrics(samples[0].cpu(), mask[0].cpu())
            metrics["seed"] = seed
            seed_results.append(metrics)

            atom37 = trans_nm_to_atom37(samples[:1].cpu())
            pdb_path = cond_dir / f"seed_{seed}.pdb"
            write_prot_to_pdb(atom37[0].numpy(), str(pdb_path), overwrite=True, no_indexing=True)

            print(f"    Rg={metrics['rg']:.3f} nm, contacts={metrics['contact_density']:.3f}, "
                  f"clashes={metrics['n_clashes']}, bond={metrics['mean_bond']:.3f}")

        all_results[condition.name] = seed_results
        np.savez(
            str(cond_dir / "metrics.npz"),
            **{k: np.array([r[k] for r in seed_results]) for k in seed_results[0]}
        )

    print_summary(model, all_results)
    save_summary(model, all_results, output_dir)


def print_summary(model: str, all_results: dict):
    """Print a summary table."""
    label = MODEL_CONFIGS[model]["label"]
    print(f"\n{'='*80}")
    print(f"{label} — Bias Ablation Results")
    print(f"{'Condition':<20} {'Rg (nm)':<14} {'Contact%':<14} {'Clashes':<12} {'Bond (nm)':<14}")
    print(f"{'='*80}")

    for name, results in all_results.items():
        rg = np.array([r["rg"] for r in results])
        cd = np.array([r["contact_density"] for r in results])
        cl = np.array([r["n_clashes"] for r in results])
        bd = np.array([r["mean_bond"] for r in results])
        print(f"{name:<20} {rg.mean():.3f}+/-{rg.std():.3f}    "
              f"{cd.mean():.3f}+/-{cd.std():.3f}    "
              f"{cl.mean():.0f}+/-{cl.std():.0f}       "
              f"{bd.mean():.3f}+/-{bd.std():.3f}")

    print(f"{'='*80}")


def save_summary(model: str, all_results: dict, output_dir: Path):
    """Save summary to text file."""
    label = MODEL_CONFIGS[model]["label"]
    summary_path = output_dir / "summary.txt"
    with open(summary_path, "w") as f:
        f.write(f"Bias Ablation — {label}, n={PROTEIN_LENGTH}, seeds={SEEDS}\n\n")
        f.write(f"{'Condition':<20} {'Rg (nm)':<14} {'Contact%':<14} {'Clashes':<12} {'Bond (nm)':<14}\n")
        f.write("-" * 80 + "\n")
        for name, results in all_results.items():
            rg = np.array([r["rg"] for r in results])
            cd = np.array([r["contact_density"] for r in results])
            cl = np.array([r["n_clashes"] for r in results])
            bd = np.array([r["mean_bond"] for r in results])
            f.write(f"{name:<20} {rg.mean():.3f}+/-{rg.std():.3f}    "
                    f"{cd.mean():.3f}+/-{cd.std():.3f}    "
                    f"{cl.mean():.0f}+/-{cl.std():.0f}       "
                    f"{bd.mean():.3f}+/-{bd.std():.3f}\n")
    print(f"Summary saved to {summary_path}")


def load_and_summarize(model: str, output_dir: Path):
    """Load existing results and print summary."""
    conditions = get_conditions(model)
    all_results = {}
    for condition in conditions:
        cond_dir = output_dir / condition.name
        metrics_path = cond_dir / "metrics.npz"
        if metrics_path.exists():
            data = np.load(str(metrics_path))
            results = [{k: data[k][i] for k in data.files} for i in range(len(data["rg"]))]
            all_results[condition.name] = results

    if not all_results:
        print("No results found. Run the experiment first.")
        return

    print_summary(model, all_results)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True, choices=list(MODEL_CONFIGS.keys()))
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--load-only", action="store_true")
    parser.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    parser.add_argument("--protein-length", type=int, default=PROTEIN_LENGTH,
                        help=f"Protein length (default: {PROTEIN_LENGTH})")
    args = parser.parse_args()

    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        from datetime import date
        today = date.today().strftime("%Y-%m-%d")
        output_dir = Path(f"experiments/causal/{args.model}/run_{today}_n{args.protein_length}/artifacts")

    output_dir.mkdir(parents=True, exist_ok=True)

    if args.load_only:
        load_and_summarize(args.model, output_dir)
    else:
        run(args.model, output_dir, args.seeds, protein_length=args.protein_length)
