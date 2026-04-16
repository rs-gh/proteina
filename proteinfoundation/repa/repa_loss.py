"""REPA loss and Projector modules for Proteina."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, TYPE_CHECKING

if TYPE_CHECKING:
    from proteinfoundation.repa.gearnet_encoder import GearNetPerResidueEncoder


class Projector(nn.Module):
    """Trainable MLP that maps transformer hidden states to encoder space."""

    def __init__(self, hidden_dim: int, encoder_dim: int, num_layers: int = 2, input_dim: int | None = None):
        super().__init__()
        first_linear = nn.LazyLinear(hidden_dim) if input_dim is None else nn.Linear(input_dim, hidden_dim)
        layers = [first_linear, nn.SiLU()]
        for _ in range(num_layers - 2):
            layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.SiLU()])
        layers.append(nn.Linear(hidden_dim, encoder_dim))
        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        """
        Args:
            x: [b, n, input_dim] hidden states from transformer

        Returns:
            [b, n, encoder_dim] projected features
        """
        return self.mlp(x)


class ProteinaREPALoss(nn.Module):
    """Representation Alignment loss for Proteina.

    Aligns the flow model's intermediate hidden states with a frozen
    pretrained encoder's per-residue representations using cosine similarity.
    """

    def __init__(
        self,
        encoder: GearNetPerResidueEncoder,
        projectors: nn.ModuleList,
        repa_layers: List[int],
        lambda_repa: float = 0.5,
        combination_mode: str = "additive",
        similarity_type: str = "cosine",
        averaging: str = "per_sample",
    ):
        """
        Args:
            encoder: Frozen GearNet encoder producing per-residue features.
            projectors: One projector per aligned layer (trainable).
            repa_layers: Which transformer layers are being aligned.
            lambda_repa: REPA loss weight.
            combination_mode: "additive" (fm + λ*repa) or "tradeoff" ((1-λ)*fm + λ*repa).
            similarity_type: "cosine" or "mse".
            averaging: "per_sample" (paper default — each protein contributes equally)
                or "per_residue" (global mean over all unmasked residues).
        """
        super().__init__()
        if averaging not in ("per_sample", "per_residue"):
            raise ValueError(f"averaging must be 'per_sample' or 'per_residue', got '{averaging}'")
        self.encoder = encoder
        self.projectors = projectors
        self.repa_layers = repa_layers
        self.lambda_repa = lambda_repa
        self.combination_mode = combination_mode
        self.similarity_type = similarity_type
        self.averaging = averaging

        assert len(projectors) == len(repa_layers), (
            f"Need one projector per aligned layer, got {len(projectors)} projectors "
            f"for {len(repa_layers)} layers"
        )

    def _cosine_loss(self, projected, target_repr, real_mask):
        """Compute negative cosine similarity loss with the configured averaging."""
        if self.averaging == "per_sample":
            # Paper-style: mean per sample, then mean over batch.
            # Each protein contributes equally regardless of length.
            b = projected.shape[0]
            per_sample_sims = []
            for b_idx in range(b):
                m = real_mask[b_idx]
                if m.any():
                    cs = F.cosine_similarity(
                        projected[b_idx, m], target_repr[b_idx, m], dim=-1
                    )
                    per_sample_sims.append(cs.mean())
            mean_cos_sim = torch.stack(per_sample_sims).mean()
            return -mean_cos_sim, mean_cos_sim
        else:
            # per_residue: global mean over all unmasked tokens.
            cos_sim = F.cosine_similarity(
                projected[real_mask], target_repr[real_mask], dim=-1
            )
            return -cos_sim.mean(), cos_sim.mean()

    def _mse_loss(self, projected, target_repr, real_mask):
        """Compute MSE loss with the configured averaging."""
        if self.averaging == "per_sample":
            b = projected.shape[0]
            per_sample_losses = []
            for b_idx in range(b):
                m = real_mask[b_idx]
                if m.any():
                    per_sample_losses.append(
                        F.mse_loss(projected[b_idx, m], target_repr[b_idx, m])
                    )
            return torch.stack(per_sample_losses).mean()
        else:
            return F.mse_loss(projected[real_mask], target_repr[real_mask])

    def forward(self, hidden_states, x_1_nm, mask):
        """Compute REPA alignment loss.

        Args:
            hidden_states: List of [b, n, token_dim] from specified transformer layers.
            x_1_nm: [b, n, 3] clean CA coordinates in nm.
            mask: [b, n] boolean residue mask.

        Returns:
            repa_loss: Scalar tensor (negative mean cosine similarity or MSE).
            stats: Dict with per-layer losses for logging.
        """
        # Get target representations from frozen encoder
        target_repr = self.encoder(x_1_nm, mask)  # [b, n, encoder_dim]
        real_mask = mask.bool()  # [b, n]

        total_loss = torch.tensor(0.0, device=x_1_nm.device)
        stats = {}

        for i, (h, projector) in enumerate(zip(hidden_states, self.projectors)):
            projected = projector(h)  # [b, n, encoder_dim]

            if self.similarity_type == "cosine":
                layer_loss, mean_cos_sim = self._cosine_loss(
                    projected, target_repr, real_mask
                )
                stats[f"repa/cos_sim_layer_{self.repa_layers[i]}"] = mean_cos_sim.detach()
            else:
                layer_loss = self._mse_loss(projected, target_repr, real_mask)
                stats[f"repa/mse_layer_{self.repa_layers[i]}"] = layer_loss.detach()

            total_loss = total_loss + layer_loss

        # Average over aligned layers
        repa_loss = total_loss / len(hidden_states)
        stats["repa/loss"] = repa_loss.detach()

        return repa_loss, stats
