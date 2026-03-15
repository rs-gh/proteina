#!/usr/bin/env python
"""
Run single-layer ablation at each model's peak R_c layer.

Peak R_c layers (at t=1, mean over heads):
  60M:        L0  (R_c=34.9) — already done in standard ablation
  200M notri: L0  (R_c=4.3)  — already done; also test L2 (R_c=3.9)
  200M tri:   L9  (R_c=27.0) — NEW; also test L7 (R_c=10.5)
  400M tri:   L2  (R_c=29.4) — NEW

Usage:
    python experiments/run_peak_layer_ablation.py --model 200m_tri
    python experiments/run_peak_layer_ablation.py --model 400m_tri
    python experiments/run_peak_layer_ablation.py --model 200m_notri
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from run_causal import (
    AblationCondition, run, load_and_summarize, print_summary, save_summary,
    SEEDS,
)

PEAK_LAYER_CONDITIONS = {
    "200m_notri": [
        AblationCondition(name="baseline", description="No ablation", enabled=False),
        AblationCondition(name="layer2_ablation", description="B=0 at layer 2 only",
                          ablate_layers={2}),
    ],
    "200m_tri": [
        AblationCondition(name="baseline", description="No ablation", enabled=False),
        AblationCondition(name="layer7_ablation", description="B=0 at layer 7 only",
                          ablate_layers={7}),
        AblationCondition(name="layer9_ablation", description="B=0 at layer 9 only",
                          ablate_layers={9}),
    ],
    "400m_tri": [
        AblationCondition(name="baseline", description="No ablation", enabled=False),
        AblationCondition(name="layer2_ablation", description="B=0 at layer 2 only",
                          ablate_layers={2}),
    ],
}

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True, choices=list(PEAK_LAYER_CONDITIONS.keys()))
    parser.add_argument("--load-only", action="store_true")
    args = parser.parse_args()

    output_dir = Path(f"experiments/causal/{args.model}/peak_layer_ablation/artifacts")
    output_dir.mkdir(parents=True, exist_ok=True)
    conditions = PEAK_LAYER_CONDITIONS[args.model]

    if args.load_only:
        load_and_summarize(args.model, output_dir)
    else:
        run(args.model, output_dir, SEEDS, conditions=conditions)
