"""ProteinMPNN CA-only structure encoder for REPA alignment.

Wraps the upstream ProteinMPNN model (`src/proteina/ProteinMPNN/`) so that we
only run its structure encoder (sequence-independent) and return per-residue
features [B, n, 128] in the format the rest of REPA expects.

Why this exists separately from GearNet/ESM:
- The upstream `ProteinMPNN.forward` is the full inverse-folding model and
  requires a sequence input + decoding-order randomness; we stop after the
  encoder loop where ``h_V`` is purely structural.
- The upstream module has no ``__init__.py``; we add the directory to
  ``sys.path`` like other proteina shims (see ``pyg_compat.py``).
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import torch
import torch.nn as nn

# Make the vendored ProteinMPNN package importable. It ships without an
# __init__.py upstream, so we extend sys.path rather than vendoring a copy.
_MPNN_DIR = Path(__file__).resolve().parents[2] / "ProteinMPNN"
if str(_MPNN_DIR) not in sys.path:
    sys.path.insert(0, str(_MPNN_DIR))

from protein_mpnn_utils import ProteinMPNN, gather_nodes  # noqa: E402

log = logging.getLogger(__name__)


_ENCODER_KEY_PREFIXES = (
    "features.",
    "W_e.",
    "W_v.",
    "encoder_layers.",
)


def _filter_encoder_state_dict(full_sd: dict) -> tuple[dict, list[str]]:
    """Keep only structure-encoder parameters from a ProteinMPNN checkpoint.

    Drops decoder + sequence-embedding weights (W_s, decoder_layers, W_out)
    that we never invoke, so we can call ``load_state_dict(strict=True)`` on
    the encoder subset and surface any silent shape drift.
    """
    encoder_sd = {
        k: v for k, v in full_sd.items() if k.startswith(_ENCODER_KEY_PREFIXES)
    }
    dropped = sorted(set(full_sd) - set(encoder_sd))
    return encoder_sd, dropped


class ProteinMPNNPerResidueEncoder(nn.Module):
    """Frozen ProteinMPNN CA-only encoder returning per-residue features.

    Runs only the structure-encoder portion of ProteinMPNN: featurize CA
    coords -> KNN graph -> stack of EncLayer message-passing -> ``h_V``.
    Sequence input is never consulted.
    """

    def __init__(
        self,
        ckpt_path: str | None = None,
        hidden_dim: int = 128,
        num_encoder_layers: int = 3,
        k_neighbors: int = 48,
        augment_eps: float = 0.0,
        random_init: bool = False,
        random_seed: int = 0,
    ):
        super().__init__()
        self.encoder_dim = hidden_dim
        self.k_neighbors = k_neighbors

        # Build full ProteinMPNN (CA variant) so the upstream state-dict keys
        # match exactly. We just won't call the decoder.
        if random_init:
            torch.manual_seed(random_seed)
        self.mpnn = ProteinMPNN(
            num_letters=21,
            node_features=hidden_dim,
            edge_features=hidden_dim,
            hidden_dim=hidden_dim,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_encoder_layers,
            vocab=21,
            k_neighbors=k_neighbors,
            augment_eps=augment_eps,
            dropout=0.0,
            ca_only=True,
        )

        if ckpt_path is not None and not random_init:
            ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            full_sd = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
            encoder_sd, dropped = _filter_encoder_state_dict(full_sd)
            missing, unexpected = self.mpnn.load_state_dict(encoder_sd, strict=False)
            # The mpnn submodule still contains decoder params; "missing" here
            # is exactly the decoder + W_s names we deliberately filtered out.
            decoder_only_missing = [
                k for k in missing if not k.startswith(_ENCODER_KEY_PREFIXES)
            ]
            structural_missing = [
                k for k in missing if k.startswith(_ENCODER_KEY_PREFIXES)
            ]
            if structural_missing or unexpected:
                raise RuntimeError(
                    f"Unexpected key mismatch loading ProteinMPNN encoder from "
                    f"{ckpt_path}. missing={structural_missing}, "
                    f"unexpected={unexpected}"
                )
            log.info(
                "Loaded ProteinMPNN encoder from %s (dropped %d decoder keys, "
                "left %d decoder params at random init - never used).",
                ckpt_path, len(dropped), len(decoder_only_missing),
            )

        for p in self.parameters():
            p.requires_grad = False

    def train(self, mode: bool = True) -> "ProteinMPNNPerResidueEncoder":
        return super().train(False)

    @torch.no_grad()
    def forward(self, ca_coords_nm, mask, residue_type=None):
        """Compute per-residue ProteinMPNN structure features.

        Args:
            ca_coords_nm: [b, n, 3] CA coordinates in nanometres.
            mask: [b, n] residue mask (bool or float; nonzero = valid).
            residue_type: unused (signature parity with sequence encoders).

        Returns:
            [b, n, 128] per-residue features; masked positions are zero.
        """
        b, n, _ = ca_coords_nm.shape
        device = ca_coords_nm.device

        # nm -> Angstroms (ProteinMPNN was trained in Angstrom space)
        ca = ca_coords_nm.float() * 10.0  # [b, n, 3]

        # CA_ProteinFeatures expects mask as float32 (multiplies it into
        # neighbour-mask tensors), and chain_labels / residue_idx as long.
        mask_f = mask.to(dtype=torch.float32, device=device)
        residue_idx = (
            torch.arange(n, device=device, dtype=torch.long)
            .unsqueeze(0)
            .expand(b, n)
            .contiguous()
        )
        chain_encoding_all = torch.zeros(b, n, dtype=torch.long, device=device)

        # ---- ProteinMPNN encoder, sequence-independent path ----
        # (mirrors ProteinMPNN.forward lines 1057-1069 in protein_mpnn_utils.py)
        E, E_idx = self.mpnn.features(ca, mask_f, residue_idx, chain_encoding_all)
        h_V = torch.zeros(
            (E.shape[0], E.shape[1], E.shape[-1]), device=E.device, dtype=E.dtype
        )
        h_E = self.mpnn.W_e(E)
        mask_attend = gather_nodes(mask_f.unsqueeze(-1), E_idx).squeeze(-1)
        mask_attend = mask_f.unsqueeze(-1) * mask_attend
        for layer in self.mpnn.encoder_layers:
            h_V, h_E = layer(h_V, h_E, E_idx, mask_f, mask_attend)

        # Zero out padded residues defensively (encoder layers already mask
        # message passing, but the residual h_V can carry the zero-init values
        # from invalid rows; leave those at zero for downstream loss masking).
        h_V = h_V * mask_f.unsqueeze(-1)
        return h_V
