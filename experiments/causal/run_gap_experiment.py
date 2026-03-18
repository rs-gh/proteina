#!/usr/bin/env python
"""
"B Gap" experiment: Does removing B mid-generation cause irreversible damage?

Tests whether the temporal asymmetry is about:
  H1: "B must be present at the end" (gap should be fine)
  H2: "Removing B after showing it is harmful" (gap should damage)

Key comparison: gap_03_07 vs late_only_07
  Both have B during t>0.7, but gap also had B during t<0.3.
  If similar Rg → H1. If gap is worse → H2.

Usage:
    python experiments/run_gap_experiment.py --model 60m
    python experiments/run_gap_experiment.py --model 60m --load-only
"""

import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent
while not (REPO_ROOT / ".git").exists():
    REPO_ROOT = REPO_ROOT.parent

sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "experiments" / "causal"))

from run_causal import (
    MODEL_CONFIGS, AblationCondition, run, SEEDS,
)


def get_gap_conditions():
    """Generate gap experiment conditions."""
    return [
        # Control: no ablation
        AblationCondition(name="baseline", description="No ablation", enabled=False),

        # Gap conditions: B on → B off → B on
        # active_intervals is set via a custom field; we handle this below
        AblationCondition(name="gap_03_07",
                          description="B on [0,0.3]∪[0.7,1], off in between",
                          enabled=True),
        AblationCondition(name="gap_02_08",
                          description="B on [0,0.2]∪[0.8,1], off in between",
                          enabled=True),
        AblationCondition(name="gap_04_06",
                          description="B on [0,0.4]∪[0.6,1], off in between",
                          enabled=True),

        # Controls: late-only B (never saw B early)
        AblationCondition(name="late_only_03",
                          description="B on [0.3,1] only",
                          ablate_t_min=0.0, ablate_t_max=0.3),
        AblationCondition(name="late_only_07",
                          description="B on [0.7,1] only",
                          ablate_t_min=0.0, ablate_t_max=0.7),
        AblationCondition(name="late_only_08",
                          description="B on [0.8,1] only",
                          ablate_t_min=0.0, ablate_t_max=0.8),
    ]


# Map condition names to active_intervals for gap conditions
GAP_INTERVALS = {
    "gap_03_07": [(0.0, 0.3), (0.7, 1.0)],
    "gap_02_08": [(0.0, 0.2), (0.8, 1.0)],
    "gap_04_06": [(0.0, 0.4), (0.6, 1.0)],
}


def run_gap(model: str, output_dir: Path, seeds=SEEDS):
    """Run gap experiment with custom active_intervals support."""
    sys.path.insert(0, str(REPO_ROOT))
    import os
    os.chdir(str(REPO_ROOT))

    from dotenv import load_dotenv
    load_dotenv()

    import torch
    import lightning as L
    from hydra import compose, initialize_config_dir
    from proteinfoundation.analysis import CrystallizationTracker, BiasAblationConfig
    from proteinfoundation.proteinflow.proteina import Proteina
    from proteinfoundation.utils.ff_utils.pdb_utils import write_prot_to_pdb
    from proteinfoundation.utils.coors_utils import trans_nm_to_atom37
    from run_causal import compute_structural_metrics

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
    conditions = get_gap_conditions()
    protein_length = 100
    all_results = {}

    for condition in conditions:
        print(f"\n{'='*60}")
        print(f"Condition: {condition.name} — {condition.description}")
        print(f"{'='*60}")

        cond_dir = output_dir / condition.name
        cond_dir.mkdir(parents=True, exist_ok=True)

        if (cond_dir / "metrics.npz").exists():
            print(f"  Already exists, skipping.")
            data = np.load(str(cond_dir / "metrics.npz"))
            results = [{k: data[k][i] for k in data.files} for i in range(len(data["rg"]))]
            all_results[condition.name] = results
            continue

        seed_results = []
        for seed in seeds:
            print(f"\n  Seed {seed}...")
            L.seed_everything(seed, workers=True)

            tracker = CrystallizationTracker()

            # Set up ablation config
            if condition.name in GAP_INTERVALS:
                tracker.bias_ablation = BiasAblationConfig(
                    enabled=True,
                    active_intervals=GAP_INTERVALS[condition.name],
                    mode='zero',
                )
            else:
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
                    nsamples=1, n=protein_length, dt=0.01,
                    self_cond=cfg.get("self_cond", True),
                    cath_code=None, tracker=tracker, mask=mask,
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

            print(f"    Rg={metrics['rg']:.3f} nm, clashes={metrics['n_clashes']}, "
                  f"bond={metrics['mean_bond']:.3f}")

        all_results[condition.name] = seed_results
        np.savez(
            str(cond_dir / "metrics.npz"),
            **{k: np.array([r[k] for r in seed_results]) for k in seed_results[0]}
        )

    # Print summary
    print(f"\n{'='*80}")
    print(f"Gap Experiment — {MODEL_CONFIGS[model]['label']}")
    print(f"{'Condition':<25} {'Rg (nm)':<14} {'Clashes':<12}")
    print(f"{'='*80}")
    for name, results in all_results.items():
        rg = np.array([r["rg"] for r in results])
        cl = np.array([r["n_clashes"] for r in results])
        print(f"{name:<25} {rg.mean():.3f}+/-{rg.std():.3f}    {cl.mean():.0f}+/-{cl.std():.0f}")
    print(f"{'='*80}")

    print("\nKey comparison:")
    if "gap_03_07" in all_results and "late_only_07" in all_results:
        gap_rg = np.mean([r["rg"] for r in all_results["gap_03_07"]])
        late_rg = np.mean([r["rg"] for r in all_results["late_only_07"]])
        print(f"  gap_03_07 Rg = {gap_rg:.3f}")
        print(f"  late_only_07 Rg = {late_rg:.3f}")
        print(f"  Delta = {gap_rg - late_rg:.3f}")
        if abs(gap_rg - late_rg) < 0.1:
            print("  → H1 supported: gap does not damage, only end matters")
        else:
            print("  → H2 supported: removing B mid-generation causes additional damage")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True, choices=list(MODEL_CONFIGS.keys()))
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--load-only", action="store_true")
    parser.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    args = parser.parse_args()

    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = Path(f"experiments/causal/{args.model}/gap_experiment/artifacts")

    output_dir.mkdir(parents=True, exist_ok=True)

    if args.load_only:
        # Re-print summary from saved data
        conditions = get_gap_conditions()
        all_results = {}
        for cond in conditions:
            p = output_dir / cond.name / "metrics.npz"
            if p.exists():
                data = np.load(str(p))
                all_results[cond.name] = [{k: data[k][i] for k in data.files}
                                           for i in range(len(data["rg"]))]
        if all_results:
            print(f"{'Condition':<25} {'Rg (nm)':<14} {'Clashes':<12}")
            for name, results in all_results.items():
                rg = np.array([r["rg"] for r in results])
                cl = np.array([r["n_clashes"] for r in results])
                print(f"{name:<25} {rg.mean():.3f}+/-{rg.std():.3f}    {cl.mean():.0f}+/-{cl.std():.0f}")
    else:
        run_gap(args.model, output_dir, args.seeds)
