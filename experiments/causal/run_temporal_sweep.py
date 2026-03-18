#!/usr/bin/env python
"""
Fine-grained temporal ablation sweep.

Sweeps the temporal split point to identify exactly where the causal cliff occurs.
For each split t_s, runs two conditions:
  - "early_B": B active for t <= t_s, then zeroed (tests: is B before t_s sufficient?)
  - "late_B": B zeroed for t <= t_s, then active (tests: is B after t_s sufficient?)

Usage:
    python experiments/run_temporal_sweep.py --model 60m
    python experiments/run_temporal_sweep.py --model 60m --load-only
"""

import sys
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, Set

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parent
while not (REPO_ROOT / ".git").exists():
    REPO_ROOT = REPO_ROOT.parent

sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "experiments" / "causal"))

from run_causal import (
    MODEL_CONFIGS, AblationCondition, run, compute_structural_metrics,
    SEEDS,
)

SPLIT_POINTS = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]


def get_sweep_conditions(split_points=SPLIT_POINTS):
    """Generate ablation conditions for each split point."""
    conditions = [
        AblationCondition(name="baseline", description="No ablation", enabled=False),
    ]
    for t_s in split_points:
        # "early_B": B active for t <= t_s, then zeroed for t > t_s
        conditions.append(AblationCondition(
            name=f"early_B_t{t_s:.1f}",
            description=f"B active for t<={t_s}, zeroed after",
            ablate_t_min=t_s,
            ablate_t_max=1.0,
        ))
        # "late_B": B zeroed for t <= t_s, then active for t > t_s
        conditions.append(AblationCondition(
            name=f"late_B_t{t_s:.1f}",
            description=f"B zeroed for t<={t_s}, active after",
            ablate_t_min=0.0,
            ablate_t_max=t_s,
        ))
    return conditions


def load_and_plot(model: str, output_dir: Path):
    """Load results and print summary table + produce plot data."""
    conditions = get_sweep_conditions()

    # Load baseline
    baseline_path = output_dir / "baseline" / "metrics.npz"
    if baseline_path.exists():
        bl = np.load(str(baseline_path))
        bl_rg = bl["rg"].mean()
    else:
        bl_rg = None

    print(f"\n{'='*80}")
    print(f"Temporal Sweep — {MODEL_CONFIGS[model]['label']}")
    print(f"{'Condition':<25} {'Rg (nm)':<14} {'Clashes':<12}")
    print(f"{'='*80}")

    if bl_rg is not None:
        print(f"{'baseline':<25} {bl_rg:.3f}{'':>8} {'0':>6}")

    early_rg, late_rg = [], []
    for t_s in SPLIT_POINTS:
        for prefix, store in [("early_B", early_rg), ("late_B", late_rg)]:
            name = f"{prefix}_t{t_s:.1f}"
            path = output_dir / name / "metrics.npz"
            if path.exists():
                d = np.load(str(path))
                rg_mean = d["rg"].mean()
                cl_mean = d["n_clashes"].mean()
                store.append((t_s, rg_mean, cl_mean))
                print(f"{name:<25} {rg_mean:.3f}{'':>8} {cl_mean:.0f}")
            else:
                print(f"{name:<25} (missing)")

    print(f"{'='*80}")

    # Save summary for plotting
    if early_rg and late_rg:
        np.savez(
            str(output_dir / "sweep_summary.npz"),
            split_points=np.array([x[0] for x in early_rg]),
            early_rg=np.array([x[1] for x in early_rg]),
            early_clashes=np.array([x[2] for x in early_rg]),
            late_rg=np.array([x[1] for x in late_rg]),
            late_clashes=np.array([x[2] for x in late_rg]),
            baseline_rg=np.array([bl_rg] if bl_rg else []),
        )
        print(f"Sweep summary saved to {output_dir / 'sweep_summary.npz'}")


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
        from datetime import date
        today = date.today().strftime("%Y-%m-%d")
        output_dir = Path(f"experiments/causal/{args.model}/temporal_sweep/artifacts")

    output_dir.mkdir(parents=True, exist_ok=True)

    if args.load_only:
        load_and_plot(args.model, output_dir)
    else:
        conditions = get_sweep_conditions()
        run(args.model, output_dir, args.seeds, conditions=conditions)
        load_and_plot(args.model, output_dir)
