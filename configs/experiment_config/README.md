# Experiment Configs

## Directory structure

```
training/256/    — 256-residue training configs (baseline, REPA layer variants)
training/512/    — 512-residue training configs (baseline, REPA layer variants)
inference/       — all inference/FID evaluation configs
smoke/           — smoke test configs
model/           — model architecture configs (referenced via Hydra defaults)
```

## Shared base configs (root level)

- `training_ca.yaml` — shared training base, inherited by all training configs
- `training_ca_motif.yaml` — motif conditioning variant

## Symlinks (root level)

The `.yaml` symlinks at root level provide **backward compatibility** so that
existing scripts and pending SLURM jobs can still load configs by flat name
(e.g. `--config_name training_repa`). They point to the real files in subdirs.

These can be removed once all pending jobs have completed and scripts have been
updated to use `--config_subdir`.

## Usage

```bash
# New style (with subdir):
python train_repa.py --config_subdir training/256 --config_name training_baseline_256

# Old style (flat, via symlinks — still works):
python train_repa.py --config_name training_baseline_256
```
