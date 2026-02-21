# Quick Start: Generating Protein Backbones

This guide walks through generating protein backbones with Proteina's smallest model (~60M parameters). By the end you'll have PDB files of generated protein structures.

## Prerequisites

1. **Environment setup** (from the main README):

```bash
mamba env create -f environment.yaml
conda activate proteina_env
pip install -e .
```

2. **Create a `.env` file** in the repo root:

```
DATA_PATH=/directory/where/you/store/files
```

## 1. Download a checkpoint

Download the 60M parameter model (smallest and fastest):

```bash
# Using NGC CLI
ngc registry resource download-version \
  nvidia/clara/proteina_v1.3_dfs_60m_notri:1.0 \
  --dest checkpoints/
```

Or download manually from the [NGC catalog page](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/clara/resources/proteina_v1.3_dfs_60m_notri/files) and place the `.ckpt` file under `checkpoints/proteina_v1.3_dfs_60m_notri_v1.0/`.

Your directory should look like:
```
checkpoints/
  proteina_v1.3_dfs_60m_notri_v1.0/
    proteina_v1.3_DFS_60M_notri.ckpt
```

## 2. Set the checkpoint path

Edit `configs/experiment_config/inference_base.yaml` and set `ckpt_path` to the directory containing the checkpoint:

```yaml
ckpt_path: "checkpoints/proteina_v1.3_dfs_60m_notri_v1.0"
```

## 3. Run inference

We provide a lightweight config (`inference_ucond_60m_notri`) that generates 2 protein backbones of length 100 with designability computation disabled — ideal for a quick test:

```bash
python proteinfoundation/inference.py --config_name inference_ucond_60m_notri
```

This takes a few seconds on a single GPU. Generated PDB files are saved under `inference/inference_ucond_60m_notri/`.

### What this config does

The config file at `configs/experiment_config/inference_ucond_60m_notri.yaml`:

```yaml
defaults:
  - inference_base
  - _self_

run_name_: ucond_60M_notri
ckpt_name: proteina_v1.3_DFS_60M_notri.ckpt

self_cond: True

# Generate just a few samples for testing
nres_lens: [100]
nsamples_per_len: 2
max_nsamples: 2

# Skip heavy designability computation
compute_designability: False
compute_fid: False

sampling_caflow:
  sampling_mode: sc   # "vf" for ODE sampling, "sc" for SDE sampling
  sc_scale_noise: 0.45  # noise scale, used if sampling_mode == "sc"
```

Key parameters:
- `nres_lens: [100]` — generate proteins with 100 residues
- `nsamples_per_len: 2` — generate 2 samples
- `sampling_mode: sc` — use the SDE sampler (generally produces better samples than ODE)
- `sc_scale_noise: 0.45` — noise scale for SDE sampling (lower = less stochastic)
- `compute_designability: False` — skip the expensive ProteinMPNN + ESMFold designability check

## 4. Inspect the output

After inference, the output structure looks like:

```
inference/
  inference_ucond_60m_notri/
    n_100_id_0/
      n_100_id_0.pdb     # Generated backbone (100 residues)
    n_100_id_1/
      n_100_id_1.pdb
  results_inference_ucond_60m_notri.csv   # Summary CSV
```

## 5. Visualize generated structures

You can open the PDB files in [PyMOL](https://pymol.org/) or [ChimeraX](https://www.cgl.ucsf.edu/chimerax/), or visualize them inline with `py3Dmol` (install with `pip install py3Dmol`):

```python
import py3Dmol
import glob
import os

# Find all generated PDB files
pdb_dir = "inference/inference_ucond_60m_notri"
pdb_files = sorted(glob.glob(os.path.join(pdb_dir, "**/*.pdb"), recursive=True))
print(f"Found {len(pdb_files)} generated structures")

# Visualize a structure
def view_pdb(pdb_path, width=600, height=400):
    with open(pdb_path, 'r') as f:
        pdb_data = f.read()

    view = py3Dmol.view(width=width, height=height)
    view.addModel(pdb_data, 'pdb')
    view.setStyle({'cartoon': {'color': 'spectrum'}})  # Rainbow N-to-C
    view.addStyle({'atom': 'CA'}, {'sphere': {'radius': 0.3, 'color': 'gray'}})
    view.zoomTo()
    return view

view = view_pdb(pdb_files[0])
view.show()
```

A full visualization notebook is also available at [`visualize_generation.ipynb`](../visualize_generation.ipynb).

## Customizing generation

To generate more/longer proteins, create your own config or modify the parameters:

| What | Parameter | Example |
|------|-----------|---------|
| Protein lengths | `nres_lens` | `[50, 100, 200]` |
| Samples per length | `nsamples_per_len` | `100` |
| Batch size (GPU memory) | `max_nsamples` | `5` |
| ODE vs SDE sampler | `sampling_caflow.sampling_mode` | `vf` (ODE) or `sc` (SDE) |
| Noise scale (SDE only) | `sampling_caflow.sc_scale_noise` | `0.45` |
| Compute designability | `compute_designability` | `True` (requires ProteinMPNN weights) |

For full details on all config parameters and larger-scale sampling, see the "Sampling our models" section in the main [README](../README.md).
