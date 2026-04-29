"""ESM-2 wrapper that returns per-residue features for REPA alignment.

Loads a frozen ESM-2 model via HuggingFace ``transformers`` and exposes the
same call contract as ``GearNetPerResidueEncoder``:

    forward(ca_coords_nm, mask, residue_type) -> [B, N, encoder_dim]

CA coordinates are ignored (ESM is a sequence-only language model); the
encoder reads ``residue_type`` - the OpenFold-indexed amino-acid tensor
already attached to every proteina batch - and maps it into the ESM
tokenizer's vocabulary.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from openfold.np.residue_constants import restypes, unk_restype_index

# OpenFold single-letter AA order: A R N D C Q E G H I L K M F P S T W Y V
# (resname_to_idx maps 3-letter ALA/ARG/... to these same indices).
# ESM-2's tokenizer uses a different permutation plus special tokens, so we
# build a lookup tensor at init time to remap 0..19 -> ESM token IDs.
_NUM_AA = len(restypes)  # 20


class ESMPerResidueEncoder(nn.Module):
    """Frozen ESM-2 encoder returning per-residue last-hidden-state features.

    Consumes the ``residue_type`` tensor proteina already attaches to every
    batch (values 0..20 in OpenFold order, padded with -1) and produces
    ``[B, N, hidden_size]`` embeddings aligned with the proteina mask.
    """

    def __init__(
        self,
        model_id: str = "facebook/esm2_t33_650M_UR50D",
        layer: int | None = None,
    ):
        """
        Args:
            model_id: HuggingFace ESM-2 checkpoint id.
            layer: Which hidden-states layer to return (None = last_hidden_state).
                Numbered from 0 (embeddings) to num_layers inclusive.
        """
        super().__init__()
        # Import lazily so proteina environments without `transformers` can still
        # import this module for e.g. mock-based unit tests.
        from transformers import AutoModel, AutoTokenizer

        self.model_id = model_id
        self.layer = layer

        tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.esm = AutoModel.from_pretrained(model_id)
        self.encoder_dim = self.esm.config.hidden_size

        # Build permanent index remap: proteina AA idx (0..19, 20=UNK) -> ESM token id.
        aa_to_esm = torch.full((_NUM_AA + 1,), tokenizer.unk_token_id, dtype=torch.long)
        for proteina_idx, letter in enumerate(restypes):
            esm_tok = tokenizer.convert_tokens_to_ids(letter)
            if esm_tok == tokenizer.unk_token_id:
                raise RuntimeError(
                    f"ESM tokenizer has no token for amino acid '{letter}' "
                    f"(model_id={model_id})"
                )
            aa_to_esm[proteina_idx] = esm_tok
        # UNK (index 20) maps to ESM's <unk>.
        aa_to_esm[unk_restype_index] = tokenizer.unk_token_id

        self.register_buffer("aa_to_esm_token", aa_to_esm, persistent=False)
        self.cls_token_id = tokenizer.cls_token_id
        self.eos_token_id = tokenizer.eos_token_id
        self.pad_token_id = tokenizer.pad_token_id

        # Freeze ESM - REPA target is a stop-gradient.
        for param in self.parameters():
            param.requires_grad = False

    def train(self, mode: bool = True) -> "ESMPerResidueEncoder":
        """Force evaluation mode always."""
        return super().train(False)

    @torch.no_grad()
    def forward(self, ca_coords_nm, mask, residue_type=None):
        """Compute per-residue ESM features.

        Args:
            ca_coords_nm: [b, n, 3] CA coordinates in nm. Unused (signature parity).
            mask: [b, n] boolean residue mask.
            residue_type: [b, n] long tensor of AA indices in OpenFold order
                (0..19 for standard AAs, 20 for UNK, negative for padding).

        Returns:
            [b, n, encoder_dim] per-residue ESM embeddings (masked positions zero).
        """
        if residue_type is None:
            raise ValueError(
                "ESMPerResidueEncoder requires `residue_type`; is the batch being "
                "constructed without the residue_type attribute?"
            )

        b, n = residue_type.shape
        device = residue_type.device
        mask_bool = mask.bool()

        # Clamp out-of-range (e.g. -1 padding) to a valid index before gather;
        # masked-out positions get overwritten with <pad> below.
        safe_idx = residue_type.clamp(min=0, max=_NUM_AA)
        aa_tokens = self.aa_to_esm_token.to(device)[safe_idx]  # [b, n]
        aa_tokens = torch.where(
            mask_bool, aa_tokens, torch.full_like(aa_tokens, self.pad_token_id)
        )

        # Prepend <cls> and append <eos> to every sequence.
        cls = torch.full((b, 1), self.cls_token_id, dtype=torch.long, device=device)
        eos = torch.full((b, 1), self.eos_token_id, dtype=torch.long, device=device)
        input_ids = torch.cat([cls, aa_tokens, eos], dim=1)  # [b, n+2]

        ones = torch.ones((b, 1), dtype=torch.bool, device=device)
        attention_mask = torch.cat([ones, mask_bool, ones], dim=1).long()

        want_hidden_states = self.layer is not None
        outputs = self.esm(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=want_hidden_states,
        )
        if want_hidden_states:
            feats = outputs.hidden_states[self.layer]
        else:
            feats = outputs.last_hidden_state  # [b, n+2, hidden_size]

        # Strip CLS (0) and EOS (n+1) -> [b, n, hidden_size].
        per_residue = feats[:, 1:-1, :]

        # Match the training dtype of the mask-providing batch (bf16/fp16 safe).
        per_residue = per_residue.to(ca_coords_nm.dtype)

        # Zero out padded positions (REPA also masks at the loss; this keeps
        # downstream projector inputs clean for logging / debugging).
        per_residue = per_residue * mask_bool.unsqueeze(-1).to(per_residue.dtype)
        return per_residue
