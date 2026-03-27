"""REPA loss and Projector modules for Proteina."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, TYPE_CHECKING

if TYPE_CHECKING:
    from proteinfoundation.repa.gearnet_encoder import GearNetPerResidueEncoder


class Projector(nn.Module):
    """Trainable MLP that maps transformer hidden states to encoder space.

    Uses LazyLinear for automatic input dimension inference (handles fused
    hidden states or varying token dimensions without manual configuration).
    """

    def __init__(self, hidden_dim: int, encoder_dim: int, num_layers: int = 2):
        super().__init__()
        layers = [nn.LazyLinear(hidden_dim), nn.SiLU()]
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
    ):
        """
        Args:
            encoder: Frozen GearNet encoder producing per-residue features.
            projectors: One projector per aligned layer (trainable).
            repa_layers: Which transformer layers are being aligned.
            lambda_repa: REPA loss weight.
            combination_mode: "additive" (fm + λ*repa) or "tradeoff" ((1-λ)*fm + λ*repa).
            similarity_type: "cosine" or "mse".
        """
        super().__init__()
        self.encoder = encoder
        self.projectors = projectors
        self.repa_layers = repa_layers
        self.lambda_repa = lambda_repa
        self.combination_mode = combination_mode
        self.similarity_type = similarity_type

        assert len(projectors) == len(repa_layers), (
            f"Need one projector per aligned layer, got {len(projectors)} projectors "
            f"for {len(repa_layers)} layers"
        )

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
                cos_sim = F.cosine_similarity(
                    projected[real_mask], target_repr[real_mask], dim=-1
                )
                layer_loss = -cos_sim.mean()
                stats[f"repa/cos_sim_layer_{self.repa_layers[i]}"] = cos_sim.mean().detach()
            else:
                layer_loss = F.mse_loss(
                    projected[real_mask], target_repr[real_mask]
                )
                stats[f"repa/mse_layer_{self.repa_layers[i]}"] = layer_loss.detach()

            total_loss = total_loss + layer_loss

        # Average over aligned layers
        repa_loss = total_loss / len(hidden_states)
        stats["repa/loss"] = repa_loss.detach()

        return repa_loss, stats
