#!/usr/bin/env python
"""
Unified structure lens experiment runner.

Applies the model's coordinate decoder to intermediate layer representations
to measure how 3D structure emerges across layers at each timestep.

Usage:
    python experiments/run_structure_lens.py --model 200m_notri
    python experiments/run_structure_lens.py --model 400m_tri --load-only
"""

import os
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parent
while not (REPO_ROOT / ".git").exists():
    REPO_ROOT = REPO_ROOT.parent

SEEDS = [42, 123, 256]
PROTEIN_LENGTH = 100
CAPTURE_EVERY_N = 5

MODEL_CONFIGS = {
    "60m": {
        "config_name": "inference_ucond_60m_notri",
        "ckpt_path": "checkpoints/proteina_v1.3_dfs_60m_notri_v1.0",
        "label": "60M (v1.3, 12L)",
    },
    "200m_notri": {
        "config_name": "inference_ucond_200m_notri",
        "ckpt_path": "checkpoints/proteina_v1.2_dfs_200m_notri_v1.0",
        "label": "200M no-tri (v1.2, 15L)",
    },
    "200m_tri": {
        "config_name": "inference_ucond_200m_tri",
        "ckpt_path": "checkpoints/proteina_v1.1_dfs_200m_tri_v1.0",
        "label": "200M tri (v1.1, 15L)",
    },
    "400m_tri": {
        "config_name": "inference_ucond_400m_tri",
        "ckpt_path": "checkpoints/proteina_v1.4_d21m_400m_tri_v1.0",
        "label": "400M tri (v1.4, 18L)",
    },
}


def _compute_rmsd(coords1: torch.Tensor, coords2: torch.Tensor) -> float:
    """Compute RMSD with Kabsch alignment."""
    c1 = coords1 - coords1.mean(dim=0)
    c2 = coords2 - coords2.mean(dim=0)
    H = c1.T @ c2
    U, S, Vt = torch.linalg.svd(H)
    d = torch.det(Vt.T @ U.T)
    sign_matrix = torch.diag(torch.tensor([1.0, 1.0, d.sign()]))
    R = Vt.T @ sign_matrix @ U.T
    c1_aligned = (R @ c1.T).T
    return torch.sqrt(((c1_aligned - c2) ** 2).sum(-1).mean()).item()


def compute_structure_lens_metrics(node_reprs, coors_decoder, num_registers, mask):
    """Apply coordinate decoder to intermediate representations and compute metrics."""
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

    decoder_device = next(coors_decoder.parameters()).device

    for t_idx, ts_idx in enumerate(timestep_indices):
        first_layer = layer_indices[0]
        timesteps[t_idx] = node_reprs[ts_idx][first_layer]["timestep"]

        coords_per_layer = []
        for l_idx, layer in enumerate(layer_indices):
            node_repr = node_reprs[ts_idx][layer]["node_repr"]
            with torch.no_grad():
                seqs_no_reg = node_repr[:, r:, :].to(decoder_device)
                mask_dev = mask.to(decoder_device)
                coords = coors_decoder(seqs_no_reg) * mask_dev[..., None]
            coords_per_layer.append(coords[0].cpu())

        final_coords = coords_per_layer[-1]
        final_mask = mask[0].cpu().bool()
        final_ca = final_coords[final_mask]

        n = final_ca.shape[0]
        final_dists = torch.cdist(final_ca, final_ca)
        i_idx = torch.arange(n).unsqueeze(1)
        j_idx = torch.arange(n).unsqueeze(0)
        seqsep_mask = (i_idx - j_idx).abs() >= 6
        final_contacts = (final_dists < 0.8) & seqsep_mask

        for l_idx in range(L):
            ca = coords_per_layer[l_idx][final_mask]
            rmsd_to_final[t_idx, l_idx] = _compute_rmsd(ca, final_ca)
            if l_idx > 0:
                prev_ca = coords_per_layer[l_idx - 1][final_mask]
                delta_rmsd[t_idx, l_idx] = _compute_rmsd(ca, prev_ca)
            centroid = ca.mean(dim=0)
            rg_arr[t_idx, l_idx] = torch.sqrt(((ca - centroid) ** 2).sum(-1).mean()).item()
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


def plot_structure_lens(results, save_dir, seed, model_label=""):
    """Generate per-seed structure lens visualizations."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    timesteps = results["timesteps"]
    T, L = results["rmsd_to_final"].shape

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for ax, (data, title, cmap) in zip(axes, [
        (results["rmsd_to_final"], "RMSD to Final Layer (nm)", "viridis_r"),
        (results["rg"], "Radius of Gyration (nm)", "coolwarm"),
        (results["contact_sim"], "Contact Similarity (Jaccard)", "viridis"),
    ]):
        im = ax.imshow(data.T, aspect="auto", origin="lower", cmap=cmap,
                        extent=[timesteps[0], timesteps[-1], -0.5, L - 0.5])
        ax.set_xlabel("Timestep (t)")
        ax.set_ylabel("Layer")
        ax.set_title(title)
        plt.colorbar(im, ax=ax, shrink=0.8)

    fig.suptitle(f"Structure Lens — {model_label} (n={PROTEIN_LENGTH}, seed={seed})", fontsize=14)
    plt.tight_layout()
    path = save_dir / f"structure_lens_heatmap_seed{seed}.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")


def plot_aggregate(all_results, save_dir, model_label=""):
    """Plot aggregate structure lens results."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    timesteps = all_results[0]["timesteps"]
    T, L = all_results[0]["rmsd_to_final"].shape
    n_seeds = len(all_results)

    rmsd_stack = np.stack([r["rmsd_to_final"] for r in all_results])
    rg_stack = np.stack([r["rg"] for r in all_results])
    contact_stack = np.stack([r["contact_sim"] for r in all_results])

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for ax, (data, title, cmap) in zip(axes, [
        (rmsd_stack.mean(axis=0), "RMSD to Final Layer (nm)", "viridis_r"),
        (rg_stack.mean(axis=0), "Radius of Gyration (nm)", "coolwarm"),
        (contact_stack.mean(axis=0), "Contact Similarity (Jaccard)", "viridis"),
    ]):
        im = ax.imshow(data.T, aspect="auto", origin="lower", cmap=cmap,
                        extent=[timesteps[0], timesteps[-1], -0.5, L - 0.5])
        ax.set_xlabel("Timestep (t)")
        ax.set_ylabel("Layer")
        ax.set_title(title)
        plt.colorbar(im, ax=ax, shrink=0.8)

    fig.suptitle(f"Structure Lens — {model_label} (n={PROTEIN_LENGTH}, {n_seeds} seeds, mean)", fontsize=14)
    plt.tight_layout()
    path = save_dir / "structure_lens_aggregate.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {path}")

    np.savez(
        str(save_dir / "aggregate.npz"),
        timesteps=timesteps,
        rmsd_mean=rmsd_stack.mean(axis=0), rmsd_std=rmsd_stack.std(axis=0),
        rg_mean=rg_stack.mean(axis=0), rg_std=rg_stack.std(axis=0),
        contact_sim_mean=contact_stack.mean(axis=0), contact_sim_std=contact_stack.std(axis=0),
    )


def run(model: str, output_dir: Path, seeds: list = SEEDS):
    """Run structure lens experiment."""
    sys.path.insert(0, str(REPO_ROOT))
    os.chdir(str(REPO_ROOT))

    from dotenv import load_dotenv
    load_dotenv()

    import lightning as L
    from hydra import compose, initialize_config_dir
    from proteinfoundation.analysis import CrystallizationTracker
    from proteinfoundation.proteinflow.proteina import Proteina

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

    num_layers = model_net.nn.nlayers
    num_registers = model_net.nn.num_registers
    coors_decoder = model_net.nn.coors_3d_decoder
    sampling_args = cfg.sampling_caflow
    all_results = []

    for seed in seeds:
        print(f"\n{'='*60}")
        print(f"[{cfg_model['label']}] Seed {seed}")
        print(f"{'='*60}")

        seed_dir = output_dir / f"seed_{seed}"

        # Skip if already exists
        if (seed_dir / "structure_lens.npz").exists():
            print(f"  Already exists, loading.")
            data = dict(np.load(str(seed_dir / "structure_lens.npz")))
            all_results.append(data)
            continue

        seed_dir.mkdir(parents=True, exist_ok=True)
        L.seed_everything(seed, workers=True)

        tracker = CrystallizationTracker()
        tracker.capture_node_repr = True
        tracker.enable(capture_every_n=CAPTURE_EVERY_N, move_to_cpu=True)
        mask = torch.ones(1, PROTEIN_LENGTH, dtype=torch.bool, device=device)

        with torch.no_grad():
            samples = model_net.generate_with_analysis(
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

        print("  Computing structure lens metrics...")
        results = compute_structure_lens_metrics(node_reprs, coors_decoder, num_registers, mask.cpu())
        all_results.append(results)

        np.savez(str(seed_dir / "structure_lens.npz"), **results)
        plot_structure_lens(results, seed_dir, seed, cfg_model["label"])

        final_rmsd = results["rmsd_to_final"][-1]
        print(f"  Final timestep RMSD to final layer: "
              f"L0={final_rmsd[0]:.3f}, L{num_layers//2}={final_rmsd[num_layers//2]:.3f}, "
              f"L{num_layers-1}={final_rmsd[-1]:.4f}")

        del tracker, node_reprs
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if len(all_results) >= 2:
        print(f"\nGenerating aggregate plots ({len(all_results)} seeds)...")
        plot_aggregate(all_results, output_dir, cfg_model["label"])


def load_and_summarize(model: str, output_dir: Path, seeds: list = SEEDS):
    """Load existing results and print summary."""
    cfg_model = MODEL_CONFIGS[model]
    all_results = []
    for seed in seeds:
        path = output_dir / f"seed_{seed}" / "structure_lens.npz"
        if path.exists():
            data = dict(np.load(str(path)))
            all_results.append(data)
            print(f"Loaded seed {seed}")

    if not all_results:
        print("No results found.")
        return

    T, L = all_results[0]["rmsd_to_final"].shape
    timesteps = all_results[0]["timesteps"]
    rmsd_stack = np.stack([r["rmsd_to_final"] for r in all_results])

    print(f"\n{cfg_model['label']} — RMSD to final layer (mean over {len(all_results)} seeds):")
    print(f"{'t':<8}" + "".join(f"{'L'+str(l):<8}" for l in [0, L//4, L//2, 3*L//4, L-1]))
    for t_idx in [0, T//4, T//2, 3*T//4, T-1]:
        row = f"{timesteps[t_idx]:<8.2f}"
        for l in [0, L//4, L//2, 3*L//4, L-1]:
            row += f"{rmsd_stack[:, t_idx, l].mean():<8.3f}"
        print(row)

    if len(all_results) >= 2:
        plot_aggregate(all_results, output_dir, cfg_model["label"])


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
        output_dir = Path(f"experiments/structure_lens/{args.model}/run_{today}/artifacts")

    output_dir.mkdir(parents=True, exist_ok=True)

    if args.load_only:
        load_and_summarize(args.model, output_dir, args.seeds)
    else:
        run(args.model, output_dir, args.seeds)
