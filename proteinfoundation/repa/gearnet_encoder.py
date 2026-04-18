"""GearNet wrapper that returns per-residue features for REPA alignment.

Wraps NoTrainCAGearNet to accept dense [b, n, 3] CA coordinates and return
per-residue features [b, n, encoder_dim] before the global pooling step.
"""

import torch
import torch.nn as nn

from proteinfoundation.metrics.gearnet_utils import NoTrainCAGearNet


class GearNetPerResidueEncoder(nn.Module):
    """Frozen GearNet CA-only encoder returning per-residue features.

    Accepts dense tensors from Proteina's training loop and converts them
    to the format GearNet expects internally, then returns per-residue h_v
    (before scatter_sum pooling) reshaped back to dense [b, n, encoder_dim].
    """

    def __init__(self, ckpt_path: str):
        super().__init__()
        self.gearnet = NoTrainCAGearNet(ckpt_path)
        self.encoder_dim = self.gearnet.output_dim  # 512

        # Freeze everything (NoTrainCAGearNet already does this, but be explicit)
        for param in self.parameters():
            param.requires_grad = False

    def train(self, mode: bool = True) -> "GearNetPerResidueEncoder":
        """Force evaluation mode always."""
        return super().train(False)

    def _dense_to_gearnet_inputs(self, ca_coords_ang, mask):
        """Convert dense [b, n, 3] CA coords (Angstroms) + [b, n] bool mask
        into flat atom-level tensors for GearNet's internal methods.

        Returns:
            coords: [total_atoms, 3] CA coordinates
            atom_type: [total_atoms] all ones (CA atom type index)
            atom_seq_pos: [total_atoms] residue index within each protein
            atom2batch: [total_atoms] batch assignment
        """
        b, n, _ = ca_coords_ang.shape
        device = ca_coords_ang.device

        # Build per-residue batch assignment [b, n]
        batch_ids = torch.arange(b, device=device)[:, None].expand(b, n)  # [b, n]

        # Flatten and select valid residues
        flat_coords = ca_coords_ang.reshape(b * n, 3)  # [b*n, 3]
        flat_batch = batch_ids.reshape(b * n)  # [b*n]
        flat_mask = mask.reshape(b * n)  # [b*n]

        # Residue indices within each protein: 0, 1, 2, ..., n-1 repeated per batch
        residue_idx = torch.arange(n, device=device)[None, :].expand(b, n)  # [b, n]
        flat_residue_idx = residue_idx.reshape(b * n)  # [b*n]

        # Select only valid (unmasked) residues
        valid = flat_mask.bool()
        coords = flat_coords[valid]  # [total_atoms, 3]
        atom2batch = flat_batch[valid]  # [total_atoms]
        atom_seq_pos = flat_residue_idx[valid]  # [total_atoms]
        # CA atom type is 1 in atom37 indexing
        atom_type = torch.ones(coords.shape[0], dtype=torch.long, device=device)

        return coords, atom_type, atom_seq_pos, atom2batch

    def _scatter_to_dense(self, h_v, atom2batch, b, n, mask):
        """Scatter flat per-atom features back to dense [b, n, dim] format.

        Args:
            h_v: [total_atoms, dim] per-atom features
            atom2batch: [total_atoms] batch assignment
            b: batch size
            n: max sequence length
            mask: [b, n] bool mask
        """
        dim = h_v.shape[-1]
        device = h_v.device

        output = torch.zeros(b, n, dim, device=device, dtype=h_v.dtype)

        # For each batch element, place features at the correct residue positions
        flat_mask = mask.reshape(b * n).bool()
        # Compute position within the dense [b*n] layout
        dense_idx = torch.arange(b * n, device=device)[flat_mask]  # [total_atoms]
        # Convert flat index to (batch, residue) and place
        batch_idx = dense_idx // n
        res_idx = dense_idx % n
        output[batch_idx, res_idx] = h_v

        return output

    @torch.no_grad()
    def forward(self, ca_coords_nm, mask, residue_type=None):
        """Compute per-residue GearNet features.

        Args:
            ca_coords_nm: [b, n, 3] CA coordinates in nanometers
            mask: [b, n] boolean residue mask
            residue_type: unused (signature-parity with sequence-based encoders)

        Returns:
            per_residue_features: [b, n, encoder_dim] (masked positions are zero)
        """
        b, n, _ = ca_coords_nm.shape

        # Convert nm to Angstroms (GearNet was trained in Angstrom space)
        ca_coords_ang = ca_coords_nm * 10.0

        # Build atom-level inputs
        coords, atom_type, atom_seq_pos, atom2batch = self._dense_to_gearnet_inputs(
            ca_coords_ang, mask
        )

        # Ensure float32 for GearNet (may be bf16 from mixed precision training)
        coords = coords.float()

        # Use GearNet's internal methods to build features and graph
        h_v = self.gearnet.node_feature(atom_type, atom_seq_pos)
        edge_list = self.gearnet.construct_graph(atom_seq_pos, coords, atom2batch)
        h_e = self.gearnet.edge_feature(edge_list, atom_seq_pos, coords, atom2batch)

        # Run GearNet layers (per-atom processing)
        for layer in self.gearnet.layers:
            h_v = layer(h_v, edge_list, h_e)

        # h_v is now [total_atoms, hidden_dim=512] — per-residue since CA-only
        # Scatter back to dense format
        return self._scatter_to_dense(h_v, atom2batch, b, n, mask)
