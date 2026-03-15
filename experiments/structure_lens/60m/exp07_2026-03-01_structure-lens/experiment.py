#!/usr/bin/env python
"""
Experiment 6: Structure Lens — Per-Layer Intermediate Structure Predictions
===========================================================================

Date: 2026-03-01
Model: proteina_v1.3_DFS_60M_notri (60M params)
Protein length: 100 residues
Seeds: [42, 123, 256]

Purpose: Apply the model's coordinate decoder to intermediate node
representations at each transformer layer, creating a "structure lens"
(protein analog of the logit lens from NLP interpretability). This reveals
when the model commits to a fold during the forward pass at each timestep.

Captures at each (timestep, layer):
- Intermediate CA coordinates from coors_3d_decoder(seqs_at_layer_l)
- RMSD to final-layer prediction
- Radius of gyration
- Contact map similarity to final prediction

Produces a (layer x timestep) heatmap showing structural convergence.
"""

import os
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parent
while not (REPO_ROOT / ".git").exists():
    REPO_ROOT = REPO_ROOT.parent
EXPERIMENT_NAME = "2026-03-01-structure-lens"
ARTIFACTS_BASE = Path(__file__).resolve().parent / "artifacts"
DEFAULT_CKPT = "checkpoints/proteina_v1.3_dfs_60m_notri_v1.0"

SEEDS = [42, 123, 256]
PROTEIN_LENGTH = 100
CAPTURE_EVERY_N = 5  # capture every 5th timestep


def compute_structure_lens_metrics(
    node_reprs: dict,
    coors_decoder: torch.nn.Module,
    num_registers: int,
    mask: torch.Tensor,
) -> dict:
    """
    Apply coordinate decoder to intermediate node representations and compute
    structural similarity metrics.

    Args:
        node_reprs: Dict[timestep_idx][layer_idx] -> node_repr tensor [b, r+n, d]
        coors_decoder: The model's coors_3d_decoder (LayerNorm + Linear)
        num_registers: Number of register tokens to strip
        mask: Sequence mask [b, n]

    Returns:
        Dict with arrays:
        - rmsd_to_final: [T, L] RMSD of each layer's prediction to the final layer
        - rg: [T, L] radius of gyration
        - contact_sim: [T, L] contact map similarity (Jaccard) to final layer
        - timesteps: [T] timestep values
    """
    timestep_indices = sorted(node_reprs.keys())
    layer_indices = sorted(node_reprs[timestep_indices[0]].keys())
    T = len(timestep_indices)
    L = len(layer_indices)

    rmsd_to_final = np.zeros((T, L))
    delta_rmsd = np.zeros((T, L))
    rg_arr = np.zeros((T, L))
    contact_sim = np.zeros((T, L))
    timesteps = np.zeros(T)

    r = num_registers

    for t_idx, ts_idx in enumerate(timestep_indices):
        # Get timestep value from any capture
        first_layer = layer_indices[0]
        cap = node_reprs[ts_idx][first_layer]
        timesteps[t_idx] = cap["timestep"]

        # Decode coordinates at each layer
        # Determine decoder device
        decoder_device = next(coors_decoder.parameters()).device

        coords_per_layer = []
        for l_idx, layer in enumerate(layer_indices):
            node_repr = node_reprs[ts_idx][layer]["node_repr"]  # [b, r+n, d]

            with torch.no_grad():
                # Strip registers, move to decoder device, decode coordinates
                seqs_no_reg = node_repr[:, r:, :].to(decoder_device)  # [b, n, d]
                mask_dev = mask.to(decoder_device)
                coords = coors_decoder(seqs_no_reg) * mask_dev[..., None]  # [b, n, 3]

            coords_per_layer.append(coords[0].cpu())  # [n, 3]

        # Final layer prediction is the reference
        final_coords = coords_per_layer[-1]  # [n, 3]
        final_mask = mask[0].cpu().bool()
        final_ca = final_coords[final_mask]

        # Compute contact map for final layer (threshold 0.8 nm, |i-j| >= 6)
        n = final_ca.shape[0]
        final_dists = torch.cdist(final_ca, final_ca)
        i_idx = torch.arange(n).unsqueeze(1)
        j_idx = torch.arange(n).unsqueeze(0)
        seqsep_mask = (i_idx - j_idx).abs() >= 6
        final_contacts = (final_dists < 0.8) & seqsep_mask

        for l_idx in range(L):
            ca = coords_per_layer[l_idx][final_mask]  # [n, 3]

            # RMSD to final layer (with alignment)
            rmsd = _compute_rmsd(ca, final_ca)
            rmsd_to_final[t_idx, l_idx] = rmsd

            # Delta RMSD (to previous layer)
            if l_idx > 0:
                prev_ca = coords_per_layer[l_idx - 1][final_mask]
                delta_rmsd[t_idx, l_idx] = _compute_rmsd(ca, prev_ca)

            # Radius of gyration
            centroid = ca.mean(dim=0)
            rg_arr[t_idx, l_idx] = torch.sqrt(((ca - centroid) ** 2).sum(-1).mean()).item()

            # Contact map similarity (Jaccard index)
            layer_dists = torch.cdist(ca, ca)
            layer_contacts = (layer_dists < 0.8) & seqsep_mask
            intersection = (final_contacts & layer_contacts).sum().float()
            union = (final_contacts | layer_contacts).sum().float()
            contact_sim[t_idx, l_idx] = (intersection / union.clamp(min=1)).item()

    return {
        "rmsd_to_final": rmsd_to_final,
        "delta_rmsd": delta_rmsd,
        "rg": rg_arr,
        "contact_sim": contact_sim,
        "timesteps": timesteps,
    }


def _compute_rmsd(coords1: torch.Tensor, coords2: torch.Tensor) -> float:
    """Compute RMSD with Kabsch alignment between two sets of coordinates."""
    # Center
    c1 = coords1 - coords1.mean(dim=0)
    c2 = coords2 - coords2.mean(dim=0)

    # Kabsch: find optimal rotation
    H = c1.T @ c2  # [3, 3]
    U, S, Vt = torch.linalg.svd(H)

    # Handle reflection
    d = torch.det(Vt.T @ U.T)
    sign_matrix = torch.diag(torch.tensor([1.0, 1.0, d.sign()]))
    R = Vt.T @ sign_matrix @ U.T

    c1_aligned = (R @ c1.T).T
    rmsd = torch.sqrt(((c1_aligned - c2) ** 2).sum(-1).mean()).item()
    return rmsd


def plot_structure_lens(results: dict, save_dir: Path, seed: int):
    """Generate structure lens visualizations."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    timesteps = results["timesteps"]
    T, L = results["rmsd_to_final"].shape

    # 1. RMSD to final layer heatmap (layer x timestep)
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    for ax, (data, title, cmap) in zip(axes, [
        (results["rmsd_to_final"], "RMSD to Final Layer (nm)\nLower = more converged", "viridis_r"),
        (results["rg"], "Radius of Gyration (nm)", "coolwarm"),
        (results["contact_sim"], "Contact Map Similarity (Jaccard)\nHigher = more similar to final", "viridis"),
    ]):
        im = ax.imshow(data.T, aspect="auto", origin="lower", cmap=cmap,
                        extent=[timesteps[0], timesteps[-1], -0.5, L - 0.5])
        ax.set_xlabel("Timestep (t)")
        ax.set_ylabel("Layer")
        ax.set_title(title)
        plt.colorbar(im, ax=ax, shrink=0.8)

    fig.suptitle(f"Structure Lens (n={PROTEIN_LENGTH}, seed={seed})", fontsize=14)
    plt.tight_layout()
    path = save_dir / f"structure_lens_heatmap_seed{seed}.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")

    # 2. RMSD convergence curves across layers for select timesteps
    fig, ax = plt.subplots(figsize=(8, 5))
    t_indices = [0, T // 4, T // 2, 3 * T // 4, T - 1]
    colors = plt.cm.plasma(np.linspace(0, 1, len(t_indices)))

    for color, t_idx in zip(colors, t_indices):
        t_val = timesteps[t_idx]
        ax.plot(range(L), results["rmsd_to_final"][t_idx],
                color=color, label=f"t={t_val:.2f}", linewidth=2)

    ax.set_xlabel("Layer")
    ax.set_ylabel("RMSD to Final Layer (nm)")
    ax.set_title("Per-Layer Structural Convergence")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = save_dir / f"rmsd_convergence_seed{seed}.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")

    # 3. Delta RMSD (layer-to-layer change)
    fig, ax = plt.subplots(figsize=(8, 5))
    im = ax.imshow(results["delta_rmsd"].T, aspect="auto", origin="lower", cmap="hot",
                    extent=[timesteps[0], timesteps[-1], -0.5, L - 0.5])
    ax.set_xlabel("Timestep (t)")
    ax.set_ylabel("Layer")
    ax.set_title("Delta RMSD (change from previous layer)\nBright = large structural update")
    plt.colorbar(im, ax=ax, shrink=0.8)
    plt.tight_layout()
    path = save_dir / f"delta_rmsd_seed{seed}.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")


def plot_aggregate(all_results: list, save_dir: Path):
    """Plot aggregate (mean ± std) structure lens results across seeds."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    timesteps = all_results[0]["timesteps"]
    T, L = all_results[0]["rmsd_to_final"].shape

    # Stack across seeds
    rmsd_stack = np.stack([r["rmsd_to_final"] for r in all_results])  # [seeds, T, L]
    rg_stack = np.stack([r["rg"] for r in all_results])
    contact_stack = np.stack([r["contact_sim"] for r in all_results])

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    for ax, (data, title, cmap) in zip(axes, [
        (rmsd_stack.mean(axis=0), "RMSD to Final Layer (nm)\nmean over seeds", "viridis_r"),
        (rg_stack.mean(axis=0), "Radius of Gyration (nm)\nmean over seeds", "coolwarm"),
        (contact_stack.mean(axis=0), "Contact Similarity (Jaccard)\nmean over seeds", "viridis"),
    ]):
        im = ax.imshow(data.T, aspect="auto", origin="lower", cmap=cmap,
                        extent=[timesteps[0], timesteps[-1], -0.5, L - 0.5])
        ax.set_xlabel("Timestep (t)")
        ax.set_ylabel("Layer")
        ax.set_title(title)
        plt.colorbar(im, ax=ax, shrink=0.8)

    n_seeds = len(all_results)
    fig.suptitle(f"Structure Lens (n={PROTEIN_LENGTH}, {n_seeds} seeds, mean)", fontsize=14)
    plt.tight_layout()
    path = save_dir / "structure_lens_aggregate.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {path}")

    # Save aggregated data
    np.savez(
        str(save_dir / "aggregate.npz"),
        timesteps=timesteps,
        rmsd_mean=rmsd_stack.mean(axis=0),
        rmsd_std=rmsd_stack.std(axis=0),
        rg_mean=rg_stack.mean(axis=0),
        rg_std=rg_stack.std(axis=0),
        contact_sim_mean=contact_stack.mean(axis=0),
        contact_sim_std=contact_stack.std(axis=0),
    )


def run(ckpt_path: str = DEFAULT_CKPT):
    """Run structure lens experiment."""
    sys.path.insert(0, str(REPO_ROOT))
    os.chdir(str(REPO_ROOT))

    from dotenv import load_dotenv
    load_dotenv()

    import lightning as L
    from hydra import compose, initialize_config_dir
    from proteinfoundation.analysis import CrystallizationTracker
    from proteinfoundation.proteinflow.proteina import Proteina

    config_path = str(REPO_ROOT / "configs" / "experiment_config")
    with initialize_config_dir(config_dir=config_path, version_base=None):
        cfg = compose(config_name="inference_ucond_60m_notri")

    from omegaconf import OmegaConf
    OmegaConf.update(cfg, "ckpt_path", ckpt_path)

    ckpt_file = os.path.join(cfg.ckpt_path, cfg.ckpt_name)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Loading model from {ckpt_file}...")
    model = Proteina.load_from_checkpoint(ckpt_file)
    model.eval()
    model.to(device)

    num_layers = model.nn.nlayers
    num_registers = model.nn.num_registers
    coors_decoder = model.nn.coors_3d_decoder

    sampling_args = cfg.sampling_caflow
    all_results = []

    for seed in SEEDS:
        print(f"\n{'='*60}")
        print(f"Seed {seed}")
        print(f"{'='*60}")
        L.seed_everything(seed, workers=True)

        seed_dir = ARTIFACTS_BASE / f"seed_{seed}"
        seed_dir.mkdir(parents=True, exist_ok=True)

        # Setup tracker to capture node representations
        tracker = CrystallizationTracker()
        tracker.capture_node_repr = True
        tracker.enable(capture_every_n=CAPTURE_EVERY_N, move_to_cpu=True)

        mask = torch.ones(1, PROTEIN_LENGTH, dtype=torch.bool, device=device)

        with torch.no_grad():
            samples = model.generate_with_analysis(
                nsamples=1,
                n=PROTEIN_LENGTH,
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

        print(f"  Captured {len(tracker)} snapshots, {tracker.memory_usage_mb():.1f} MB")

        # Extract node representations into a dict structure
        node_reprs = {}
        for ts_idx in tracker.get_timestep_indices():
            node_reprs[ts_idx] = {}
            for layer_idx in tracker.get_layer_indices():
                cap = tracker.get_capture(ts_idx, layer_idx)
                if cap is not None and cap.node_repr is not None:
                    node_reprs[ts_idx][layer_idx] = {
                        "node_repr": cap.node_repr,
                        "timestep": cap.timestep,
                    }

        if not node_reprs:
            print("  WARNING: No node representations captured. Skipping seed.")
            continue

        # Compute structure lens metrics
        print("  Computing structure lens metrics...")
        results = compute_structure_lens_metrics(
            node_reprs, coors_decoder, num_registers, mask.cpu(),
        )
        all_results.append(results)

        # Save per-seed results
        np.savez(str(seed_dir / "structure_lens.npz"), **results)

        # Generate per-seed plots
        plot_structure_lens(results, seed_dir, seed)

        # Print summary
        final_rmsd = results["rmsd_to_final"][-1]  # last timestep
        print(f"  Final timestep — RMSD to final layer per layer:")
        print(f"    L0: {final_rmsd[0]:.3f}, L{num_layers//2}: {final_rmsd[num_layers//2]:.3f}, "
              f"L{num_layers-1}: {final_rmsd[-1]:.4f} (should be ~0)")

        # Free memory
        del tracker, node_reprs
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # Aggregate across seeds
    if len(all_results) >= 2:
        print(f"\nGenerating aggregate plots ({len(all_results)} seeds)...")
        ARTIFACTS_BASE.mkdir(parents=True, exist_ok=True)
        plot_aggregate(all_results, ARTIFACTS_BASE)


def load_and_summarize():
    """Load existing results and print summary."""
    all_results = []
    for seed in SEEDS:
        path = ARTIFACTS_BASE / f"seed_{seed}" / "structure_lens.npz"
        if path.exists():
            data = dict(np.load(str(path)))
            all_results.append(data)
            print(f"Loaded seed {seed}")

    if not all_results:
        print("No results found. Run the experiment first.")
        return

    # Print summary
    T, L = all_results[0]["rmsd_to_final"].shape
    timesteps = all_results[0]["timesteps"]
    rmsd_stack = np.stack([r["rmsd_to_final"] for r in all_results])

    print(f"\nRMSD to final layer (mean over {len(all_results)} seeds):")
    print(f"{'t':<8}" + "".join(f"{'L'+str(l):<8}" for l in [0, L//4, L//2, 3*L//4, L-1]))
    for t_idx in [0, T // 4, T // 2, 3 * T // 4, T - 1]:
        row = f"{timesteps[t_idx]:<8.2f}"
        for l in [0, L // 4, L // 2, 3 * L // 4, L - 1]:
            val = rmsd_stack[:, t_idx, l].mean()
            row += f"{val:<8.3f}"
        print(row)

    if len(all_results) >= 2:
        plot_aggregate(all_results, ARTIFACTS_BASE)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt_path", default=DEFAULT_CKPT)
    parser.add_argument("--load-only", action="store_true",
                        help="Just load and summarize existing results")
    args = parser.parse_args()

    if args.load_only:
        load_and_summarize()
    else:
        run(args.ckpt_path)
