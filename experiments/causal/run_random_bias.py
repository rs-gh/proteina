#!/usr/bin/env python
"""
Random bias swap experiment: replace B with magnitude-matched Gaussian noise.

Tests whether the model needs the *information* in B or just *any* additive
signal of similar scale. If random B also causes collapse, the geometric
information is essential, not just the presence of a bias term.

Usage:
    python experiments/run_random_bias.py --model 60m
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from run_causal import (
    AblationCondition, run, SEEDS, PROTEIN_LENGTH,
)

RANDOM_CONDITIONS = [
    AblationCondition(name="baseline", description="No ablation", enabled=False),
    AblationCondition(name="full_zero", description="B=0 everywhere", mode='zero'),
    AblationCondition(name="full_random", description="B=random noise (matched magnitude)",
                      mode='random'),
]

if __name__ == "__main__":
    import argparse
    from run_causal import MODEL_CONFIGS
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True, choices=list(MODEL_CONFIGS.keys()))
    parser.add_argument("--load-only", action="store_true")
    args = parser.parse_args()

    output_dir = Path(f"experiments/causal/{args.model}/random_bias_swap/artifacts")
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.load_only:
        from run_causal import load_and_summarize
        load_and_summarize(args.model, output_dir)
    else:
        run(args.model, output_dir, SEEDS[:3], conditions=RANDOM_CONDITIONS)
