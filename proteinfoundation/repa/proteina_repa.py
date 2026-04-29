"""Proteina subclass with REPA (Representation Alignment) loss."""

import random

import torch
import torch.nn as nn
from typing import Dict

from proteinfoundation.proteinflow.proteina import Proteina
from proteinfoundation.repa.gearnet_encoder import GearNetPerResidueEncoder, MCGearNetEdgePerResidueEncoder, PWGearNetEdgePerResidueEncoder
from proteinfoundation.repa.protein_transformer_repa import (
    ProteinTransformerAF3WithHiddenStates,
)
from proteinfoundation.repa.repa_loss import Projector, ProteinaREPALoss
from proteinfoundation.utils.ff_utils.pdb_utils import mask_cath_code_by_level


def _build_encoder(repa_cfg) -> nn.Module:
    """Instantiate the frozen REPA target encoder from config.

    Supports ``repa.encoder.type ∈ {"gearnet", "esm"}``. Falls back to the
    legacy ``repa.gearnet_ckpt_path`` schema (pre-pluggable-encoder) so
    existing training YAMLs keep working without edits.
    """
    encoder_cfg = repa_cfg.get("encoder", None)
    if encoder_cfg is None:
        # Legacy path: only gearnet_ckpt_path at the top of repa_cfg.
        return GearNetPerResidueEncoder(ckpt_path=repa_cfg.gearnet_ckpt_path)

    enc_type = encoder_cfg.type
    random_init = encoder_cfg.get("random_init", False)
    random_seed = encoder_cfg.get("random_seed", 0)
    ckpt_path = encoder_cfg.get("gearnet_ckpt_path", None)
    if enc_type == "gearnet":
        return GearNetPerResidueEncoder(
            ckpt_path=ckpt_path,
            random_init=random_init,
            random_seed=random_seed,
        )
    if enc_type == "gearnet_mc_edge":
        return MCGearNetEdgePerResidueEncoder(
            ckpt_path=ckpt_path,
            random_init=random_init,
            random_seed=random_seed,
        )
    if enc_type == "pw_gearnet":
        return PWGearNetEdgePerResidueEncoder(
            ckpt_path=ckpt_path,
            random_init=random_init,
            random_seed=random_seed,
        )
    if enc_type == "esm":
        from proteinfoundation.repa.esm_encoder import ESMPerResidueEncoder

        return ESMPerResidueEncoder(
            model_id=encoder_cfg.get("model_id", "facebook/esm2_t33_650M_UR50D"),
            layer=encoder_cfg.get("layer", None),
        )
    raise ValueError(f"Unknown repa.encoder.type: {enc_type!r}")


class ProteinaREPA(Proteina):
    """Proteina with REPA alignment to a frozen GearNet encoder.

    Subclasses Proteina to:
    1. Replace the NN with a hidden-state-extracting variant
    2. Add a frozen GearNet encoder + trainable projectors
    3. Override training_step to compute and combine REPA loss
    """

    def __init__(self, cfg_exp, store_dir=None):
        super().__init__(cfg_exp, store_dir)

        repa_cfg = cfg_exp.repa
        repa_layers = list(repa_cfg.layers)

        # Replace self.nn with hidden-state-extracting version
        # Re-instantiate with same config but with repa_layers
        nn_kwargs = dict(cfg_exp.model.nn)
        self.nn = ProteinTransformerAF3WithHiddenStates(
            repa_layers=repa_layers, **nn_kwargs
        )

        # Create frozen encoder
        encoder = _build_encoder(repa_cfg)

        # Create trainable projectors (one per aligned layer)
        projector_hidden = repa_cfg.get("projector_hidden_dim", cfg_exp.model.nn.token_dim)
        projector_nlayers = repa_cfg.get("projector_num_layers", 2)
        token_dim = cfg_exp.model.nn.token_dim
        projectors = nn.ModuleList([
            Projector(
                hidden_dim=projector_hidden,
                encoder_dim=encoder.encoder_dim,
                num_layers=projector_nlayers,
                input_dim=token_dim,
            )
            for _ in repa_layers
        ])

        # Create REPA loss module
        self.repa_loss = ProteinaREPALoss(
            encoder=encoder,
            projectors=projectors,
            repa_layers=repa_layers,
            lambda_repa=repa_cfg.lambda_repa,
            combination_mode=repa_cfg.get("combination_mode", "additive"),
            similarity_type=repa_cfg.get("similarity_type", "cosine"),
            averaging=repa_cfg.get("averaging", "per_residue"),
        )

        # Update param count (exclude frozen encoder)
        self.nparams = sum(p.numel() for p in self.parameters() if p.requires_grad)

    def on_load_checkpoint(self, checkpoint: dict) -> None:
        """Remap encoder keys saved before encoder compile was introduced.

        Checkpoints saved without encoder compile have keys like
        ``repa_loss.encoder.gearnet.*``. After torch.compile wraps the encoder,
        the model expects ``repa_loss.encoder._orig_mod.gearnet.*``. Remap
        transparently so old checkpoints resume cleanly.
        """
        if not hasattr(self.repa_loss.encoder, "_orig_mod"):
            return
        sd = checkpoint["state_dict"]
        prefix = "repa_loss.encoder."
        compiled_prefix = "repa_loss.encoder._orig_mod."
        has_compiled = any(k.startswith(compiled_prefix) for k in sd)
        has_uncompiled = any(
            k.startswith(prefix) and not k.startswith(compiled_prefix) for k in sd
        )
        if not has_compiled and has_uncompiled:
            checkpoint["state_dict"] = {
                (compiled_prefix + k[len(prefix):] if k.startswith(prefix) else k): v
                for k, v in sd.items()
            }

    def predict_clean(self, batch: Dict, return_hidden_states: bool = False):
        """Override to support return_hidden_states passthrough."""
        nn_out = self.nn(batch, return_hidden_states=return_hidden_states)
        return self._nn_out_to_x_clean(nn_out, batch), nn_out

    def training_step(self, batch, batch_idx):
        """Training step with REPA loss.

        Replicates the parent training_step but:
        1. Passes return_hidden_states=True for the main prediction
        2. Computes REPA loss from hidden states + clean coordinates
        3. Combines REPA loss with FM + auxiliary losses
        """
        val_step = batch_idx == -1
        log_prefix = "validation_loss" if val_step else "train"

        # Extract clean sample (may apply augmentations)
        x_1, mask, batch_shape, n, dtype = self.extract_clean_sample(batch)
        x_1 = self.fm._mask_and_zero_com(x_1, mask)

        # Sample time and reference
        t = self.sample_t(batch_shape)
        x_0 = self.fm.sample_reference(
            n=n, shape=batch_shape, device=self.device, dtype=dtype, mask=mask
        )

        if self.motif_conditioning:
            batch.update(self.motif_factory(batch))
            x_1 = batch["x_1"]

        # Interpolation
        x_t = self.fm.interpolate(x_0, x_1, t)
        batch["t"] = t
        batch["mask"] = mask
        batch["x_t"] = x_t

        # Fold conditional training
        if self.cfg_exp.training.fold_cond:
            bs = x_1.shape[0]
            cath_code_list = batch.cath_code
            for i in range(bs):
                cath_code_list[i] = mask_cath_code_by_level(
                    cath_code_list[i], level="H"
                )
                if random.random() < self.cfg_exp.training.mask_T_prob:
                    cath_code_list[i] = mask_cath_code_by_level(
                        cath_code_list[i], level="T"
                    )
                    if random.random() < self.cfg_exp.training.mask_A_prob:
                        cath_code_list[i] = mask_cath_code_by_level(
                            cath_code_list[i], level="A"
                        )
                        if random.random() < self.cfg_exp.training.mask_C_prob:
                            cath_code_list[i] = mask_cath_code_by_level(
                                cath_code_list[i], level="C"
                            )
            batch.cath_code = cath_code_list
        else:
            if "cath_code" in batch:
                batch.pop("cath_code")

        # Self-conditioning (no REPA for the detached pass)
        if random.random() > 0.5 and self.cfg_exp.training.self_cond:
            x_pred_sc, _ = self.predict_clean(batch, return_hidden_states=False)
            batch["x_sc"] = self.detach_gradients(x_pred_sc)

        # Main prediction WITH hidden state extraction
        x_1_pred, nn_out = self.predict_clean(batch, return_hidden_states=True)

        # Flow matching loss
        fm_loss = self.compute_fm_loss(
            x_1, x_1_pred, x_t, t, mask, log_prefix=log_prefix
        )
        train_loss = torch.mean(fm_loss)

        # Auxiliary loss (distogram)
        if self.cfg_exp.loss.use_aux_loss:
            auxiliary_loss = self.compute_auxiliary_loss(
                x_1, x_1_pred, x_t, t, mask, nn_out=nn_out,
                log_prefix=log_prefix, batch=batch,
            )
            train_loss = train_loss + torch.mean(auxiliary_loss)

        # REPA loss
        hidden_states = nn_out.get("hidden_states", [])
        if hidden_states:
            residue_type = batch.residue_type if "residue_type" in batch else None
            repa_loss, repa_stats = self.repa_loss(
                hidden_states, x_1, mask, residue_type=residue_type
            )

            # Combine with FM loss
            lam = self.repa_loss.lambda_repa
            if self.repa_loss.combination_mode == "tradeoff":
                train_loss = (1.0 - lam) * train_loss + lam * repa_loss
            else:  # additive
                train_loss = train_loss + lam * repa_loss

            # Log REPA stats (prefixed for train vs validation)
            for key, value in repa_stats.items():
                self.log(
                    f"{log_prefix}/{key}", value,
                    on_step=True, on_epoch=True, prog_bar=False,
                    logger=True, batch_size=mask.shape[0],
                    sync_dist=True, add_dataloader_idx=False,
                )

        # Standard logging
        self.log(
            f"{log_prefix}/loss", train_loss,
            on_step=True, on_epoch=True, prog_bar=False,
            logger=True, batch_size=mask.shape[0],
            sync_dist=True, add_dataloader_idx=False,
        )

        if not val_step:
            self.log(
                "train_loss", train_loss,
                on_step=True, on_epoch=True, prog_bar=True,
                logger=True, batch_size=mask.shape[0],
                sync_dist=True, add_dataloader_idx=False,
            )

            b, n = mask.shape
            self.nsamples_processed = (
                self.nsamples_processed + b * self.trainer.world_size
            )
            self.log(
                "scaling/nsamples_processed", self.nsamples_processed * 1.0,
                on_step=True, on_epoch=False, prog_bar=False,
                logger=True, batch_size=1, sync_dist=True,
            )
            self.log(
                "scaling/nparams", self.nparams * 1.0,
                on_step=True, on_epoch=False, prog_bar=False,
                logger=True, batch_size=1, sync_dist=True,
            )

        return train_loss
