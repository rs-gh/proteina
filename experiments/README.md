# Experiments

Analysis of geometric pair bias (B) vs content score (C) in Proteina's attention mechanism during flow-based protein structure generation.

## Directory Structure

```
experiments/
  descriptive/          # Lens 1: Pre-softmax signal decomposition (R, R_c, H, rho, seqsep)
    run_descriptive.py  #   unified runner
    60m/  200m_notri/  200m_tri/  400m_tri/
  functional/           # Lens 2: Post-softmax contact prediction (Precision@L/5)
    60m/  ...
  causal/               # Lens 3: Bias ablation and causal variants
    run_causal.py       #   standard 6-condition ablation
    run_gap_experiment.py
    run_peak_layer_ablation.py
    run_random_bias.py
    run_temporal_sweep.py
    60m/  200m_notri/  200m_tri/  400m_tri/
  structure_lens/       # Intermediate layer structure decoding (RMSD, Rg, Jaccard)
    run_structure_lens.py
    60m/  ...
  cross_model/          # Cross-model comparison experiments
  evals/                # Evaluation scripts (post-hoc analysis of generated structures)
    run_gearnet_eval.py #   GearNet fold classification (mean max fold probability)
  plot_figures.py       # Orchestrates figures across all lenses
  README.md
```

Within each lens, experiment outputs are organized by model: `60m/`, `200m_notri/`, `200m_tri/`, `400m_tri/`.

## Experiment Log

| Exp | Date | Lens | Model | Description | Seeds | Length |
|-----|------|------|-------|-------------|-------|--------|
| 01 | 2026-02-20 | Descriptive | 60M | Initial core metrics (R, H, rho) | 1 | 100 |
| 02 | 2026-02-21 | Functional | 60M | Extended analysis (per-head, seqsep, contact, registers) | 1 | 100 |
| 03 | 2026-03-01 | Descriptive | 60M | Multi-seed robustness (R, H, rho, contact, seqsep) | 5 | 100 |
| 04 | 2026-03-01 | Descriptive | 60M | Multi-length scaling | 3/length | 50,100,200 |
| 05a | 2026-03-01 | Causal | 60M | Bias ablation (6 conditions) | 3 | 100 |
| 05c | 2026-03-01 | Cross-model | 60M+200M | Scale comparison | 3/model | 100 |
| 06 | 2026-03-01 | Descriptive | 60M | Row-centered metric R_c comparison | 5 | 100 |
| 07 | 2026-03-01 | Structure | 60M | Structure lens (per-layer decoding) | 3 | 100 |
| 08a | 2026-03-15 | Descriptive | 200M notri | Multi-seed (R, H, rho, contact, seqsep) | 3 | 100 |
| 08b | 2026-03-15 | Causal | 200M notri | Bias ablation (6 conditions) | 3 | 100 |
| 09a | 2026-03-15 | Descriptive | 200M tri | Multi-seed (R, H, rho, contact, seqsep) | 3 | 100 |
| 09b | 2026-03-15 | Causal | 200M tri | Bias ablation (6 conditions) | 3 | 100 |

## Models

| Model | Version | Layers | Heads | d_token | d_pair | Triangle | Params |
|-------|---------|--------|-------|---------|--------|----------|--------|
| 60M | v1.3 | 12 | 12 | 512 | 256 | No | 60M |
| 200M notri | v1.2 | 15 | 12 | 768 | 256 | No | 191M |
| 200M tri | v1.1 | 15 | 12 | 768 | 512 | Yes | 208M |
| 400M tri | v1.4 | 18 | 16 | 1024 | 512 | Yes | ~415M |

## Running Experiments

Each experiment directory contains an `experiment.py` that can be run with `--load-only` to print summaries without re-running generation:

```bash
python experiments/descriptive/60m/exp03_2026-03-01_multiseed/experiment.py --load-only
```

All experiments require `DATA_PATH` environment variable (use `load_dotenv()` or set manually).
