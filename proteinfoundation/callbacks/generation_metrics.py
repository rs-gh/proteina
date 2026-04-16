"""Lightweight generation-quality validation callback for proteina training.

Generates a small batch of proteins at fixed length, computes structural
metrics (fold scores, diversity, optionally designability), and logs to WandB.
Runs every N training steps on rank-0 only.

Follows the tabasco callback pattern (MoleculeMetricsCallback).
"""

import os
import shutil
import tempfile
from typing import List, Optional

import lightning as L
import torch
from lightning import Callback
from loguru import logger


class ProteinGenerationMetricsCallback(Callback):
    """Generate proteins during training and compute quality metrics.

    Metrics (all optional):
        - GearNet fold scores (fS_C, fS_A, fS_T)
        - Structural diversity (TM-score clustering)
        - Designability (ProteinMPNN + ESMFold + scRMSD + TM-score)
    """

    def __init__(
        self,
        compute_every_n_steps: int = 50000,
        num_samples: int = 30,
        length: int = 100,
        compute_gearnet: bool = True,
        compute_diversity: bool = True,
        compute_designability: bool = False,
        gearnet_ckpt_path: Optional[str] = None,
        designability_seqs_per_struct: int = 8,
        pmpnn_sampling_temp: float = 0.1,
        # Sampling parameters (match inference defaults)
        sampling_dt: float = 0.0025,
        self_cond: bool = True,
        sc_scale_noise: float = 0.45,
        sc_scale_score: float = 1.0,
        schedule_mode: str = "log",
        schedule_p: float = 2.0,
        **kwargs,
    ):
        super().__init__()
        self.compute_every_n_steps = compute_every_n_steps
        self.num_samples = num_samples
        self.length = length
        self.compute_gearnet = compute_gearnet
        self.compute_diversity = compute_diversity
        self.compute_designability = compute_designability
        self.gearnet_ckpt_path = gearnet_ckpt_path
        self.designability_seqs_per_struct = designability_seqs_per_struct
        self.pmpnn_sampling_temp = pmpnn_sampling_temp
        self.sampling_dt = sampling_dt
        self.self_cond = self_cond
        self.sc_scale_noise = sc_scale_noise
        self.sc_scale_score = sc_scale_score
        self.schedule_mode = schedule_mode
        self.schedule_p = schedule_p
        self.next_compute_step = compute_every_n_steps

    def on_train_batch_end(
        self, trainer: L.Trainer, pl_module: L.LightningModule, *args, **kwargs
    ) -> None:
        if trainer.global_rank != 0:
            return
        if trainer.global_step < self.next_compute_step:
            return
        self.next_compute_step += self.compute_every_n_steps

        step = trainer.global_step
        logger.info(
            f"[GenMetrics] Running generation metrics at step {step} "
            f"({self.num_samples} proteins, length {self.length})"
        )

        tmpdir = tempfile.mkdtemp(prefix="gen_metrics_")
        try:
            # Phase 1: Generate proteins
            pdb_paths, atom37_list = self._generate_proteins(pl_module, tmpdir)
            if not pdb_paths:
                logger.warning("[GenMetrics] No proteins generated, skipping metrics")
                return

            # Phase 2: GearNet fold scores
            if self.compute_gearnet:
                self._compute_gearnet_metrics(pl_module, pdb_paths)

            # Phase 3: Diversity
            if self.compute_diversity:
                self._compute_diversity_metrics(pl_module, atom37_list)

            # Phase 4: Designability (expensive, off by default)
            if self.compute_designability:
                self._compute_designability_metrics(pl_module, pdb_paths, tmpdir)

        except Exception as e:
            logger.error(f"[GenMetrics] Error during generation metrics: {e}")
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

    def _generate_proteins(
        self, pl_module: L.LightningModule, tmpdir: str
    ) -> tuple:
        """Generate proteins and save PDBs.

        Returns:
            (pdb_paths, atom37_list): List of PDB file paths and list of
                numpy arrays [n_residues, 37, 3].
        """
        from proteinfoundation.utils.ff_utils.pdb_utils import write_prot_to_pdb

        logger.info(f"[GenMetrics] Generating {self.num_samples} proteins of length {self.length}")

        pl_module.eval()
        with torch.no_grad():
            samples = pl_module.generate(
                nsamples=self.num_samples,
                n=self.length,
                dt=self.sampling_dt,
                self_cond=self.self_cond,
                cath_code=None,
                dtype=torch.float32,
                schedule_mode=self.schedule_mode,
                schedule_p=self.schedule_p,
                sampling_mode="sc",
                sc_scale_noise=self.sc_scale_noise,
                sc_scale_score=self.sc_scale_score,
                gt_mode="1/t",
                gt_p=1.0,
                gt_clamp_val=None,
            )
            atom37_batch = pl_module.samples_to_atom37(samples).cpu()

        pdb_paths = []
        atom37_list = []
        samples_dir = os.path.join(tmpdir, "samples")
        os.makedirs(samples_dir, exist_ok=True)

        for i in range(atom37_batch.shape[0]):
            pdb_path = os.path.join(samples_dir, f"gen_{i:04d}.pdb")
            write_prot_to_pdb(
                atom37_batch[i].numpy(),
                pdb_path,
                overwrite=True,
                no_indexing=True,
            )
            pdb_paths.append(pdb_path)
            atom37_list.append(atom37_batch[i].numpy())

        logger.info(f"[GenMetrics] Generated {len(pdb_paths)} PDBs")
        return pdb_paths, atom37_list

    def _compute_gearnet_metrics(
        self, pl_module: L.LightningModule, pdb_paths: List[str]
    ) -> None:
        """Compute fold scores using GearNet."""
        from proteinfoundation.metrics.metric_factory import (
            GenerationMetricFactory,
            generation_metric_from_list,
        )

        if self.gearnet_ckpt_path is None:
            logger.warning("[GenMetrics] No gearnet_ckpt_path, skipping GearNet metrics")
            return

        logger.info("[GenMetrics] Computing GearNet fold scores")

        metric_factory = GenerationMetricFactory(
            metrics=["fS_C", "fS_A", "fS_T"],
            ckpt_path=self.gearnet_ckpt_path,
            ca_only=True,
        ).cuda()

        try:
            metrics = generation_metric_from_list(
                pdb_paths, metric_factory, num_workers=0
            )
            for key, value in metrics.items():
                val = value.cpu().item() if isinstance(value, torch.Tensor) else value
                pl_module.log(
                    f"val/gen_{key}", val,
                    on_step=True, on_epoch=False, prog_bar=False,
                    logger=True, sync_dist=False,
                )
                logger.info(f"[GenMetrics] {key}: {val:.4f}")
        finally:
            del metric_factory
            torch.cuda.empty_cache()

    def _compute_diversity_metrics(
        self, pl_module: L.LightningModule, atom37_list: list
    ) -> None:
        """Compute structural diversity via TM-score clustering."""
        from proteinfoundation.metrics.tm_score import compute_diversity

        logger.info("[GenMetrics] Computing structural diversity")
        n_clusters = compute_diversity(atom37_list, tm_threshold=0.5)

        pl_module.log(
            "val/gen_diversity", float(n_clusters),
            on_step=True, on_epoch=False, prog_bar=False,
            logger=True, sync_dist=False,
        )
        logger.info(f"[GenMetrics] Diversity: {n_clusters} clusters (from {len(atom37_list)} structures)")

    def _compute_designability_metrics(
        self, pl_module: L.LightningModule, pdb_paths: List[str], tmpdir: str
    ) -> None:
        """Compute designability via ProteinMPNN + ESMFold + scRMSD + TM-score."""
        from proteinfoundation.metrics.designability import batch_designability

        logger.info(f"[GenMetrics] Computing designability for {len(pdb_paths)} structures")
        desig_dir = os.path.join(tmpdir, "designability")

        results = batch_designability(
            pdb_paths,
            tmp_root=desig_dir,
            num_seq_per_target=self.designability_seqs_per_struct,
            pmpnn_sampling_temp=self.pmpnn_sampling_temp,
        )

        for key in ["scRMSD_mean", "scRMSD_median", "designability_rate", "tm_score_mean"]:
            val = results[key]
            pl_module.log(
                f"val/gen_{key}", val,
                on_step=True, on_epoch=False, prog_bar=False,
                logger=True, sync_dist=False,
            )
            logger.info(f"[GenMetrics] {key}: {val:.4f}")
