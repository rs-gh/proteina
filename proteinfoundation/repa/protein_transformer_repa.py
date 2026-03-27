"""ProteinTransformerAF3 subclass that extracts intermediate hidden states for REPA."""

from typing import Dict, List, Optional

import torch

from proteinfoundation.nn.protein_transformer import ProteinTransformerAF3


class ProteinTransformerAF3WithHiddenStates(ProteinTransformerAF3):
    """ProteinTransformerAF3 that optionally returns intermediate hidden states.

    When return_hidden_states=True, collects the sequence representation `seqs`
    at specified trunk layers. These are used by ProteinaREPALoss for alignment.
    """

    def __init__(self, repa_layers: Optional[List[int]] = None, **kwargs):
        """
        Args:
            repa_layers: List of layer indices (0-indexed) at which to capture
                hidden states. E.g., [4] for middle of a 10-layer model.
            **kwargs: All arguments passed to ProteinTransformerAF3.
        """
        super().__init__(**kwargs)
        self.repa_layers = repa_layers or []

    def forward(
        self,
        batch_nn: Dict[str, torch.Tensor],
        return_hidden_states: bool = False,
    ):
        """Forward pass with optional hidden state extraction.

        Replicates the parent forward() but captures intermediate `seqs` at
        specified layer indices. Hidden states have registers stripped.

        Args:
            batch_nn: Input batch dictionary (same as parent).
            return_hidden_states: If True, collect hidden states at repa_layers.

        Returns:
            nn_out dict with:
                - "coors_pred": [b, n, 3]
                - "pair_pred": [b, n, n, num_buckets] (if applicable)
                - "hidden_states": list of [b, n, token_dim] (if return_hidden_states)
        """
        mask = batch_nn["mask"]

        # Conditioning variables
        c = self.cond_factory(batch_nn)
        c = self.transition_c_2(self.transition_c_1(c, mask), mask)

        # Prepare input
        coors_3d = batch_nn["x_t"] * mask[..., None]
        coors_embed = self.linear_3d_embed(coors_3d) * mask[..., None]
        seq_f_repr = self.init_repr_factory(batch_nn)
        seqs = coors_embed + seq_f_repr
        seqs = seqs * mask[..., None]

        # Pair representation
        pair_rep = None
        if self.use_attn_pair_bias:
            pair_rep = self.pair_repr_builder(batch_nn)

        # Apply registers
        seqs, pair_rep, mask_ext, c = self._extend_w_registers(seqs, pair_rep, mask, c)

        # Run trunk with optional hidden state collection
        hidden_states = []
        for i in range(self.nlayers):
            seqs = self.transformer_layers[i](seqs, pair_rep, c, mask_ext)

            # Capture hidden states at specified layers (after the layer forward)
            if return_hidden_states and i in self.repa_layers:
                # Strip registers before storing
                if self.num_registers > 0:
                    h = seqs[:, self.num_registers:, :]
                else:
                    h = seqs
                hidden_states.append(h)

            if self.update_pair_repr:
                if i < self.nlayers - 1:
                    if self.pair_update_layers[i] is not None:
                        pair_rep = self.pair_update_layers[i](seqs, pair_rep, mask_ext)

        # Undo registers
        seqs, pair_rep, mask_ext = self._undo_registers(seqs, pair_rep, mask_ext)

        # Get final coordinates
        final_coors = self.coors_3d_decoder(seqs) * mask[..., None]
        nn_out = {}
        if self.update_pair_repr and self.num_buckets_predict_pair is not None:
            pair_pred = self.pair_head_prediction(pair_rep)
            final_coors = final_coors + torch.mean(pair_pred) * 0.0
            final_coors = final_coors * mask[..., None]
            nn_out["pair_pred"] = pair_pred
        nn_out["coors_pred"] = final_coors

        if return_hidden_states:
            nn_out["hidden_states"] = hidden_states

        return nn_out
