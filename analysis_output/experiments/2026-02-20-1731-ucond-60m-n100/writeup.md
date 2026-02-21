# Experiment 1: Unconditional 60M model, n=100

**Date:** 2026-02-20
**Model:** proteina_v1.3_DFS_60M_notri (60M params, 12 layers, 12 heads, no triangle updates)
**Protein length:** 100 residues
**Sampling:** SDE (sc), noise scale 0.45, log schedule, dt=0.01 (100 steps)
**Capture:** every 5th timestep → 20 snapshots × 12 layers × 12 heads
**Ground truth:** Retrospective (final generated structure used as GT for ρ)
**Seed:** not set — run before seed control was added (artifacts are not exactly reproducible)

## Reproduce

```bash
python analysis_output/experiments/2026-02-20-1731-ucond-60m-n100/experiment.py \
    --ckpt_path checkpoints/proteina_v1.3_dfs_60m_notri_v1.0
```

Or equivalently:

```bash
python script_utils/crystallization_analysis.py \
    --config_name inference_ucond_60m_notri \
    --ckpt_path checkpoints/proteina_v1.3_dfs_60m_notri_v1.0 \
    --protein_length 100 \
    --output_dir ./analysis_output/artifacts/2026-02-20-1731-ucond-60m-n100 \
    --capture_every_n 5 \
    --dt 0.01
```

## Artifacts

Located in `analysis_output/artifacts/2026-02-20-1731-ucond-60m-n100/`:

| File | Description |
|------|-------------|
| `trajectory.png` | 3-panel plot: R, H, ρ vs timestep for layers 0, 6, 11 |
| `summary.png` | 6-panel: trajectories (top) + layer×timestep heatmaps (bottom) |
| `crystallization_metrics.npz` | Raw metrics arrays [20 timesteps, 12 layers, 12 heads] |
| `generated_structure.pdb` | Generated 100-residue backbone |

## Findings

### 1. Logit Dominance (R = ||B||\_F / ||C||\_F)

- **Layer 0 starts with R ≈ 8** at t=0 and drops to R ≈ 1 by t=0.5. The first layer is *heavily* geometry-dominated at the start of generation — the geometric pair bias is ~8× louder than the content score (QK^T).
- **Layers 6 and 11 stay near R ≈ 1** throughout, meaning content and geometry are roughly balanced in deeper layers at all times.
- The R heatmap shows a clear gradient: layer 0 is "hot" (high R) especially early in the trajectory, with a smooth transition to cooler values in deeper layers. This suggests the first layer acts as a **"geometric interpreter"** that bootstraps structural information from pair distances, while later layers integrate this with content-based reasoning.
- R *decreases* over time for layer 0. As the structure becomes cleaner, QK^T grows in magnitude relative to B. **Hypothesis:** early on, sequence representations are uninformative (derived from noise), making QK^T small and B relatively dominant. As structure emerges, representations become meaningful and the content score catches up.

### 2. Attention Entropy (H)

- All layers start at **high entropy (H ≈ 3.5–4.5 nats)** — attention is diffuse.
- **Layer 0 drops first** and most sharply. Crystallization point (10th percentile threshold) detected at **t ≈ 0.208**.
- **Layer 11 drops last**, with a more gradual curve — the last layer keeps exploring longer before committing.
- The H heatmap shows **wave-like propagation**: crystallization starts in early layers and propagates toward later layers. This is consistent with a "bottom-up" structure formation process.
- Late in the trajectory (t > 0.8), there is a slight *increase* in entropy for layer 0. This may be an SDE sampler artifact or indicate a role shift once structure is established.

### 3. Spatial Alignment (ρ)

- All layers start near **ρ ≈ 0** (no correlation — expected, since the structure doesn't exist yet).
- ρ increases monotonically, reaching **~0.3–0.5** by t=1.
- **Layer 11 achieves the highest spatial alignment (~0.45)**. The last layer's attention most closely reflects the true 3D contact pattern — consistent with it directly producing the coordinate prediction.
- The ρ heatmap shows alignment emerges gradually across all layers, with later layers consistently higher.

## Summary

The first layer acts as a **"geometric interpreter"** — heavily relying on pair bias B at early timesteps when content representations are still uninformative. As generation proceeds:

1. Entropy drops layer-by-layer in a wave from early→late layers
2. Spatial alignment with the final structure increases
3. The content score QK^T grows to match the geometric bias

This suggests a **division of labour**: early layers bootstrap geometric information from pair distances, while later layers refine it with content-based attention that increasingly reflects the emerging structure.

## Reproducibility

The original artifacts in `analysis_output/artifacts/2026-02-20-1731-ucond-60m-n100/` were generated **before seed control was added** to `crystallization_analysis.py`. The random state at run time is unknown, so the exact structure and metrics cannot be recovered.

Re-running `experiment.py` will produce a new, different (but fully reproducible) sample using seed 5. Qualitative conclusions from this writeup are expected to hold across seeds — the layer-depth and temporal patterns are structural properties of the model, not of a specific sample.

## Next experiments

- Different protein lengths (50, 200) to check if patterns scale
- Different fold classes (alpha, beta, mixed) via conditional generation
- Per-head analysis (don't reduce heads) to check for head specialization
- Sequence-separation decomposition: split R and H by |i-j| to distinguish local vs long-range interactions
