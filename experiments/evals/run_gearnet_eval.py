#!/usr/bin/env python
"""
Run GearNet evaluation on ablation PDB files.

Computes fold predictions (C/A/T) and feature vectors for all PDB files
from causal ablation experiments. Compares to baseline to assess whether
ablated structures still resemble recognisable protein folds.

Usage:
    python experiments/run_gearnet_eval.py --model 60m
    python experiments/run_gearnet_eval.py --model 60m --experiment causal
"""

import sys
from pathlib import Path
from glob import glob

import numpy as np
import torch
from torch_geometric.data import Data, Batch

REPO_ROOT = Path(__file__).resolve().parent
while not (REPO_ROOT / ".git").exists():
    REPO_ROOT = REPO_ROOT.parent

sys.path.insert(0, str(REPO_ROOT))

GEARNET_CKPT = REPO_ROOT / "data" / "proteina_additional_files" / "metric_factory" / "model_weights" / "gearnet_ca.pth"


def pdb_to_ca_coords(pdb_path: str) -> torch.Tensor:
    """Extract CA coordinates from a PDB file. Returns tensor of shape [N, 3] in Angstroms."""
    coords = []
    with open(pdb_path, 'r') as f:
        for line in f:
            if line.startswith("ATOM") and line[12:16].strip() == "CA":
                x = float(line[30:38])
                y = float(line[38:46])
                z = float(line[46:54])
                coords.append([x, y, z])
    return torch.tensor(coords, dtype=torch.float32)


def ca_coords_to_pyg(ca_coords: torch.Tensor) -> Data:
    """Convert CA coordinates [N, 3] to a PyG Data object for GearNet CA model."""
    n = ca_coords.shape[0]
    # GearNet CA model expects:
    # - coords: [N, 37, 3] (atom37 format, only CA at index 1 used)
    # - coord_mask: [N, 37] (True for CA)
    # - batch: [N] (batch index)
    coords = torch.full((n, 37, 3), 1e-5, dtype=torch.float32)
    coords[:, 1, :] = ca_coords  # CA is at index 1 in atom37

    coord_mask = torch.zeros(n, 37, dtype=torch.bool)
    coord_mask[:, 1] = True  # Only CA

    graph = Data()
    graph.coords = coords
    graph.coord_mask = coord_mask
    graph.node_id = torch.arange(n).unsqueeze(-1)
    return graph


def run_gearnet_on_pdbs(pdb_paths: list, device: str = "cpu") -> dict:
    """Run GearNet CA model on a list of PDB files.

    Returns dict with:
        features: [N_structures, 512] feature vectors
        pred_C: [N_structures, 5] class-level logits
        pred_A: [N_structures, 43] architecture-level logits
        pred_T: [N_structures, 1336] topology-level logits
        mean_max_fold_prob_C/A/T: [N_structures] max softmax probability per structure (mean max fold probability)
    """
    from proteinfoundation.metrics.gearnet_utils import NoTrainCAGearNet

    model = NoTrainCAGearNet(str(GEARNET_CKPT))
    model.to(device)
    model.eval()

    all_features = []
    all_preds = {"C": [], "A": [], "T": []}

    for pdb_path in pdb_paths:
        ca_coords = pdb_to_ca_coords(str(pdb_path))
        if ca_coords.shape[0] == 0:
            print(f"  Warning: no CA atoms in {pdb_path}, skipping")
            continue

        graph = ca_coords_to_pyg(ca_coords)
        graph.batch = torch.zeros(ca_coords.shape[0], dtype=torch.long)

        # Move to device
        graph = graph.to(device)

        with torch.no_grad():
            out = model(graph)

        all_features.append(out["protein_feature"].cpu())
        for level in ["C", "A", "T"]:
            all_preds[level].append(out[f"pred_{level}"].cpu())

    features = torch.cat(all_features, dim=0)
    results = {"features": features.numpy()}

    for level in ["C", "A", "T"]:
        logits = torch.cat(all_preds[level], dim=0)
        probs = torch.softmax(logits, dim=-1)
        confidence = probs.max(dim=-1).values
        results[f"pred_{level}"] = logits.numpy()
        results[f"confidence_{level}"] = confidence.numpy()
        # Top predicted class
        results[f"top_class_{level}"] = probs.argmax(dim=-1).numpy()

    return results


def find_ablation_pdbs(model: str, experiment: str = "causal") -> dict:
    """Find all PDB files from ablation experiments, grouped by condition."""
    base = REPO_ROOT / "experiments" / experiment / model
    conditions = {}

    for cond_dir in sorted(base.rglob("**/artifacts/*")):
        if not cond_dir.is_dir():
            continue
        pdbs = sorted(cond_dir.glob("*.pdb"))
        if pdbs:
            # Use relative path from artifacts/ as condition name
            rel = cond_dir.relative_to(base)
            conditions[str(rel)] = pdbs

    # Also check direct condition dirs (e.g., run_2026-03-15/artifacts/baseline/)
    for run_dir in sorted(base.iterdir()):
        if not run_dir.is_dir():
            continue
        artifacts = run_dir / "artifacts" if (run_dir / "artifacts").exists() else run_dir
        for cond_dir in sorted(artifacts.iterdir()):
            if not cond_dir.is_dir():
                continue
            pdbs = sorted(cond_dir.glob("*.pdb"))
            if pdbs:
                key = f"{run_dir.name}/{cond_dir.name}"
                if key not in conditions:
                    conditions[key] = pdbs

    return conditions


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True, choices=["60m", "200m_notri", "200m_tri", "400m_tri"])
    parser.add_argument("--experiment", default="causal", help="Experiment type (causal, gap, temporal_sweep)")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Collect all PDB files from the specified experiment
    base = REPO_ROOT / "experiments" / "causal" / args.model
    conditions = {}

    # Look for PDBs in all subdirectories
    for subdir in sorted(base.rglob("*")):
        if subdir.is_dir():
            pdbs = sorted(subdir.glob("*.pdb"))
            if pdbs:
                # Get a clean condition name
                rel = subdir.relative_to(base)
                parts = [p for p in rel.parts if p != "artifacts"]
                key = "/".join(parts)
                conditions[key] = pdbs

    if not conditions:
        print(f"No PDB files found under {base}")
        sys.exit(1)

    print(f"Found {len(conditions)} conditions with PDB files for {args.model}")
    print(f"Using device: {device}")
    print()

    # Run GearNet on each condition
    all_results = {}
    for cond_name, pdbs in sorted(conditions.items()):
        print(f"  {cond_name}: {len(pdbs)} PDBs")
        results = run_gearnet_on_pdbs(pdbs, device=device)
        all_results[cond_name] = results

    # Print summary table
    print(f"\n{'='*90}")
    print(f"GearNet Fold Prediction — {args.model}")
    print(f"{'Condition':<40} {'MaxFoldP_C':<12} {'MaxFoldP_A':<12} {'MaxFoldP_T':<12} {'Top C':<8}")
    print(f"{'='*90}")

    for cond_name, results in sorted(all_results.items()):
        conf_c = results["confidence_C"].mean()
        conf_a = results["confidence_A"].mean()
        conf_t = results["confidence_T"].mean()
        top_c = results["top_class_C"]
        top_c_str = ",".join(map(str, top_c))
        print(f"{cond_name:<40} {conf_c:.3f}     {conf_a:.3f}     {conf_t:.3f}     {top_c_str}")

    print(f"{'='*90}")
    print("\nMaxFoldP_C/A/T = mean max fold probability at CATH class/architecture/topology level")
    print("= mean over structures of max(softmax(GearNet logits))")
    print("Higher = more recognisable as a known protein fold (distinct from distributional fS of Proteina paper)")
