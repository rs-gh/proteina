#!/usr/bin/env python
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary

"""
Crystallization Point Analysis Script

This script runs the crystallization point analysis on a protein structure
generation model, capturing attention data across the flow-matching trajectory
and computing the three key metrics:
- R (Logit Dominance): ||B||_F / ||C||_F
- H (Entropy): Shannon entropy of attention
- rho (Spatial Alignment): Correlation with GT distance matrix

For the rho metric, there are two modes:
1. External ground truth: Provide --pdb_path to compare attention against a known structure
2. Retrospective ground truth: Without --pdb_path, uses the final generated structure.
   This reveals "how early does the model know the contacts it will eventually make?"

Usage:
    # With external ground truth (for validation/reconstruction tasks)
    python script_utils/crystallization_analysis.py \
        --config_name inference_base \
        --pdb_path path/to/ground_truth.pdb \
        --output_dir ./analysis_output

    # With retrospective ground truth (for de novo generation analysis)
    python script_utils/crystallization_analysis.py \
        --config_name inference_base \
        --protein_length 100 \
        --output_dir ./analysis_output
"""

import argparse
import os
import sys
from pathlib import Path

# Add project root to path
root = Path(__file__).parent.parent
sys.path.insert(0, str(root))

import torch
import numpy as np
from loguru import logger

import hydra
from hydra import compose, initialize_config_dir
import lightning as L

from proteinfoundation.analysis import (
    CrystallizationTracker,
    TrajectoryAnalyzer,
    TrajectoryMetrics,
    plot_crystallization_trajectory,
    plot_crystallization_summary,
    plot_per_head_trajectory,
    plot_seqsep_decomposition,
    plot_contact_precision_trajectory,
    plot_attention_decomposition_grid,
    compute_gt_distance_matrix,
    compute_contact_map,
)


def load_ground_truth_coords(pdb_path: str) -> torch.Tensor:
    """
    Load ground truth CA coordinates from a PDB file.

    Args:
        pdb_path: Path to PDB file

    Returns:
        CA coordinates, shape [n, 3] in nm
    """
    from graphein_utils.graphein_utils import protein_to_pyg
    from proteinfoundation.utils.coors_utils import ang_to_nm

    graph = protein_to_pyg(pdb_path)
    # CA atom is at index 1 in atom37 format
    ca_coords = graph.coords[:, 1, :]  # [n, 3] in Angstroms
    ca_coords_nm = ang_to_nm(ca_coords)  # Convert to nm
    return ca_coords_nm


def main():
    from dotenv import load_dotenv
    load_dotenv()

    parser = argparse.ArgumentParser(
        description="Run crystallization point analysis on protein generation model"
    )
    parser.add_argument(
        "--config_path",
        type=str,
        default=str(root / "configs" / "experiment_config"),
        help="Path to config directory",
    )
    parser.add_argument(
        "--config_name",
        type=str,
        required=True,
        help="Name of the inference config (e.g., inference_base)",
    )
    parser.add_argument(
        "--pdb_path",
        type=str,
        default=None,
        help="Path to ground truth PDB file (for spatial alignment metric)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./crystallization_analysis",
        help="Directory to save results",
    )
    parser.add_argument(
        "--capture_every_n",
        type=int,
        default=5,
        help="Capture attention every N timesteps (for memory efficiency)",
    )
    parser.add_argument(
        "--protein_length",
        type=int,
        default=100,
        help="Length of protein to generate (if no PDB provided)",
    )
    parser.add_argument(
        "--cath_code",
        type=str,
        default=None,
        help="CATH code for conditional generation (e.g., '3.40.50')",
    )
    parser.add_argument(
        "--dt",
        type=float,
        default=0.01,
        help="Timestep for sampling (default: 0.01 = 100 steps)",
    )
    parser.add_argument(
        "--reduce_heads",
        action="store_true",
        help="Average over attention heads to save memory",
    )
    parser.add_argument(
        "--skip_rho",
        action="store_true",
        help="Skip computing the spatial alignment (rho) metric entirely",
    )
    parser.add_argument(
        "--ckpt_path",
        type=str,
        default=None,
        help="Override checkpoint directory (e.g., checkpoints/proteina_v1.3_dfs_60m_notri_v1.0)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed (default: use config value, fallback 42)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run on",
    )
    # Extended analysis flags (Exp 2/3/4/5/7)
    parser.add_argument(
        "--compute_seqsep",
        action="store_true",
        help="Compute R/H/rho decomposed by sequence separation bin (local/medium/long-range)",
    )
    parser.add_argument(
        "--compute_contact_precision",
        action="store_true",
        help="Compute Precision@L/5 contact prediction for full, B-only, and C-only attention",
    )
    parser.add_argument(
        "--compute_register_metrics",
        action="store_true",
        help="Compute register token attention fraction across the trajectory",
    )
    parser.add_argument(
        "--plot_grid",
        action="store_true",
        help="Generate AF2 Figure 12-style (layers x timesteps) attention heatmap grid",
    )
    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Crystallization Point Analysis")
    logger.info(f"Config: {args.config_name}")
    logger.info(f"Output: {output_dir}")

    # Load config
    with initialize_config_dir(config_dir=args.config_path, version_base=None):
        cfg = compose(config_name=args.config_name)

    # Set random seed for reproducibility
    seed = getattr(cfg, "seed", 42)
    if args.seed is not None:
        seed = args.seed
    L.seed_everything(seed, workers=True)
    logger.info(f"[seed] {seed}")

    # Override checkpoint path if provided via CLI
    if args.ckpt_path is not None:
        from omegaconf import OmegaConf
        OmegaConf.update(cfg, "ckpt_path", args.ckpt_path)

    # Load model
    logger.info("Loading model...")
    from proteinfoundation.proteinflow.proteina import Proteina

    ckpt_file = os.path.join(cfg.ckpt_path, cfg.ckpt_name)
    assert os.path.exists(ckpt_file), f"Checkpoint not found: {ckpt_file}"
    model = Proteina.load_from_checkpoint(ckpt_file)
    model.eval()
    model.to(args.device)

    # Get model parameters from the loaded model
    num_layers = model.nn.nlayers
    num_heads = model.nn.transformer_layers[0].mhba.mha.heads
    logger.info(f"Model has {num_layers} layers, {num_heads} heads")

    # Load ground truth if provided
    gt_coords = None
    if args.pdb_path:
        logger.info(f"Loading ground truth from {args.pdb_path}")
        gt_coords = load_ground_truth_coords(args.pdb_path)
        n = gt_coords.shape[0]
        gt_coords = gt_coords.unsqueeze(0).to(args.device)  # [1, n, 3]
        logger.info(f"Ground truth protein length: {n}")
    else:
        n = args.protein_length
        logger.info(f"No ground truth provided, generating protein of length {n}")

    # Setup tracker
    tracker = CrystallizationTracker()
    tracker.enable(
        capture_every_n=args.capture_every_n,
        reduce_heads=args.reduce_heads,
        move_to_cpu=True,
    )

    # Setup CATH code if provided
    cath_code = None
    if args.cath_code:
        cath_code = [[args.cath_code]]
        logger.info(f"Using CATH code: {args.cath_code}")

    # Generate with analysis
    logger.info("Running generation with analysis...")
    mask = torch.ones(1, n, dtype=torch.bool, device=args.device)

    # Extract sampling parameters from config (matching inference.py structure)
    sampling_args = cfg.sampling_caflow

    with torch.no_grad():
        samples = model.generate_with_analysis(
            nsamples=1,
            n=n,
            dt=args.dt,
            self_cond=cfg.get("self_cond", True),
            cath_code=cath_code,
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

    logger.info(f"Generation complete. Captured {len(tracker)} attention snapshots")
    logger.info(f"Memory usage: {tracker.memory_usage_mb():.1f} MB")

    # Use retrospective ground truth if no PDB provided (unless --skip_rho is set)
    # This computes ρ by correlating attention at each timestep with the final structure
    if args.skip_rho:
        logger.info("Skipping spatial alignment (rho) metric as requested")
        gt_coords = None
    elif gt_coords is None and samples is not None:
        logger.info("Using retrospective ground truth (final generated structure)")
        logger.info("  -> rho measures 'how early does the model know its final contacts?'")
        gt_coords = samples.unsqueeze(0) if samples.dim() == 2 else samples[:1]  # [1, n, 3]
        gt_coords = gt_coords.to(args.device)

    # Compute metrics
    logger.info("Computing crystallization metrics...")
    num_registers = model.nn.num_registers
    logger.info(f"Stripping {num_registers} register tokens from attention maps")
    analyzer = TrajectoryAnalyzer(tracker, num_layers, num_heads, num_registers=num_registers)
    spatial_alignment_label = "vs. external GT" if args.pdb_path else "vs. final predicted structure"
    metrics = analyzer.compute_metrics(gt_coords, mask, spatial_alignment_label=spatial_alignment_label)

    # Print summary
    print("\n" + "=" * 60)
    print(metrics.summary())
    print("=" * 60 + "\n")

    # Save metrics
    metrics_path = output_dir / "crystallization_metrics.npz"
    metrics.save(str(metrics_path))
    logger.info(f"Saved metrics to {metrics_path}")

    # Find crystallization point
    try:
        t_crystal, idx_crystal = metrics.get_crystallization_point(metric='entropy')
        logger.info(f"Crystallization point (entropy): t={t_crystal:.3f} (step {idx_crystal})")
    except Exception as e:
        logger.warning(f"Could not determine crystallization point: {e}")

    # Generate plots
    logger.info("Generating visualizations...")

    # Trajectory plot
    fig1 = plot_crystallization_trajectory(
        metrics,
        save_path=output_dir / "trajectory.png",
        title=f"Crystallization Analysis (n={n})",
    )
    logger.info(f"Saved trajectory plot")

    # Summary plot
    fig2 = plot_crystallization_summary(
        metrics,
        save_path=output_dir / "summary.png",
    )
    logger.info(f"Saved summary plot")

    # Save generated structure as PDB
    if samples is not None:
        from proteinfoundation.utils.ff_utils.pdb_utils import write_prot_to_pdb
        from proteinfoundation.utils.coors_utils import trans_nm_to_atom37

        atom37 = trans_nm_to_atom37(samples[:1].cpu())  # [1, n, 37, 3]
        pdb_path = output_dir / "generated_structure.pdb"
        write_prot_to_pdb(atom37[0].numpy(), str(pdb_path), overwrite=True, no_indexing=True)
        logger.info(f"Generated structure saved to {pdb_path}")

    # --- Extended analyses (Exp 2/3/4/5/7) ---

    # Per-head visualization (Exp 2): existing metrics already have per-head data [T, L, H];
    # we just generate per-head trajectory plots for each layer.
    per_head_dir = output_dir / "per_head"
    if any([args.compute_seqsep, args.compute_contact_precision,
            args.compute_register_metrics, args.plot_grid]):
        per_head_dir.mkdir(exist_ok=True)
        for l in range(num_layers):
            for metric in ['entropy', 'logit_dominance']:
                fig = plot_per_head_trajectory(metrics, layer=l, metric_name=metric)
                import matplotlib
                matplotlib.pyplot.close(fig)
            if metrics.spatial_alignment is not None:
                fig = plot_per_head_trajectory(metrics, layer=l, metric_name='spatial_alignment')
                import matplotlib
                matplotlib.pyplot.close(fig)
        # Save representative per-head plots for first, middle, last layers
        for l, name in [(0, 'first'), (num_layers // 2, 'mid'), (num_layers - 1, 'last')]:
            fig = plot_per_head_trajectory(
                metrics, layer=l, metric_name='entropy',
                save_path=per_head_dir / f"per_head_entropy_layer{l}_{name}.png",
            )
            import matplotlib
            matplotlib.pyplot.close(fig)
        logger.info(f"Per-head trajectory plots saved to {per_head_dir}")

    # Sequence-separation decomposition (Exp 3)
    if args.compute_seqsep:
        logger.info("Computing sequence-separation decomposition...")
        seqsep_metrics = analyzer.compute_seqsep_trajectory(gt_coords, mask)
        seqsep_path = output_dir / "seqsep_metrics.npz"
        seqsep_metrics.save(str(seqsep_path))
        logger.info(f"Saved seqsep metrics to {seqsep_path}")

        fig_seqsep = plot_seqsep_decomposition(
            seqsep_metrics,
            save_path=output_dir / "seqsep_decomposition.png",
        )
        import matplotlib
        matplotlib.pyplot.close(fig_seqsep)
        logger.info("Saved seqsep decomposition plot")

    # Contact precision (Exp 4)
    if args.compute_contact_precision and gt_coords is not None:
        logger.info("Computing contact precision (Precision@L/5)...")
        # Build contact map from GT coordinates (0.8 nm = 8 Angstroms threshold)
        gt_dist_for_contacts = compute_gt_distance_matrix(gt_coords.cpu(), mask.cpu())
        contact_map = compute_contact_map(gt_dist_for_contacts, threshold=0.8)

        mha = model.nn.transformer_layers[0].mhba.mha
        head_dim = round(1.0 / (mha.scale ** 2))
        prec_metrics = analyzer.compute_contact_precision_trajectory(
            contact_map, head_dim=head_dim,
        )
        prec_path = output_dir / "contact_precision.npz"
        prec_metrics.save(str(prec_path))
        logger.info(f"Saved contact precision metrics to {prec_path}")

        fig_prec = plot_contact_precision_trajectory(
            prec_metrics,
            save_path=output_dir / "contact_precision.png",
        )
        import matplotlib
        matplotlib.pyplot.close(fig_prec)
        logger.info(
            f"Saved contact precision plot "
            f"(full={prec_metrics.precision_full[-1].mean():.3f}, "
            f"B={prec_metrics.precision_b_only[-1].mean():.3f}, "
            f"C={prec_metrics.precision_c_only[-1].mean():.3f} at t=1)"
        )
    elif args.compute_contact_precision and gt_coords is None:
        logger.warning("--compute_contact_precision requires GT coordinates. "
                       "Use --pdb_path or ensure retrospective GT is available.")

    # Register token analysis (Exp 5)
    if args.compute_register_metrics and num_registers > 0:
        logger.info("Computing register token attention metrics...")
        reg_metrics = analyzer.compute_register_metrics()
        reg_path = output_dir / "register_metrics.npz"
        reg_metrics.save(str(reg_path))
        logger.info(
            f"Saved register metrics to {reg_path} "
            f"(mean register attention fraction: {reg_metrics.register_attn_fraction.mean():.3f})"
        )
    elif args.compute_register_metrics and num_registers == 0:
        logger.warning("Model has no register tokens; skipping --compute_register_metrics")

    # AF2 Figure 12-style attention grid (Exp 7)
    if args.plot_grid:
        logger.info("Generating attention decomposition grid (AF2 Figure 12 style)...")
        gt_dist_cpu = compute_gt_distance_matrix(gt_coords.cpu(), mask.cpu()) if gt_coords is not None else None
        contact_map_np = compute_contact_map(gt_dist_cpu, threshold=0.8)[0].numpy() if gt_dist_cpu is not None else None

        for mode in ['attn', 'bias', 'content']:
            fig_grid = plot_attention_decomposition_grid(
                tracker,
                num_registers=num_registers,
                contact_map=contact_map_np,
                show_mode=mode,
                save_path=output_dir / f"attention_grid_{mode}.png",
            )
            import matplotlib
            matplotlib.pyplot.close(fig_grid)
        logger.info("Saved attention decomposition grids (attn, bias, content)")

    logger.info(f"\nAnalysis complete! Results saved to {output_dir}")


if __name__ == "__main__":
    main()
