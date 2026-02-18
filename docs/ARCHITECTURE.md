# Proteina Model Architecture

This document provides a detailed walkthrough of how the Proteina protein structure generation model works, with exact file and line references.

## Overview

Proteina uses **flow matching** to generate protein structures. The core idea:
- Start with random noise at t=0
- Iteratively denoise over ~100-500 steps
- End with a clean protein structure at t=1

At each step, a neural network predicts what the final structure should look like, and the sampler takes a small step toward that prediction.

---

## High-Level Data Flow

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         FLOW MATCHING LOOP                               │
│   for t in [0 → 1]:                                                      │
│       x_t = noisy coordinates at timestep t                              │
│       x_1_pred = ProteinTransformerAF3(x_t, t, cath_code, ...)          │
│       x_{t+dt} = x_t + v * dt   (Euler integration)                      │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                      ProteinTransformerAF3                               │
│  ┌──────────────────────────────────────────────────────────────────┐   │
│  │ (1) INPUT PREPARATION                                             │   │
│  │     • x_t [b,n,3] → linear_3d_embed → coors_embed [b,n,512]      │   │
│  │     • init_repr_factory(features) → seq_f_repr [b,n,512]         │   │
│  │     • seqs = coors_embed + seq_f_repr                            │   │
│  │     • cond_factory(t, cath_code) → c [b,n,256] (conditioning)    │   │
│  │     • pair_repr_builder(distances) → pair_rep [b,n,n,256]        │   │
│  └──────────────────────────────────────────────────────────────────┘   │
│                                    │                                     │
│                                    ▼                                     │
│  ┌──────────────────────────────────────────────────────────────────┐   │
│  │ (2) TRANSFORMER TRUNK (15 layers)                                 │   │
│  │     for i in range(15):                                           │   │
│  │         seqs = MultiheadAttnAndTransition(seqs, pair_rep, c)     │   │
│  │         if i % 3 == 0:                                            │   │
│  │             pair_rep = PairReprUpdate(seqs, pair_rep)  # optional │   │
│  └──────────────────────────────────────────────────────────────────┘   │
│                                    │                                     │
│                                    ▼                                     │
│  ┌──────────────────────────────────────────────────────────────────┐   │
│  │ (3) COORDINATE DECODER                                            │   │
│  │     seqs [b,n,512] → LayerNorm → Linear → x_pred [b,n,3]         │   │
│  └──────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Detailed Data Flow Trace

### Entry Point: Inference Script

**File**: `proteinfoundation/inference.py`
```
Line ~328: model.predict_step(batch, batch_idx)
    → Calls the Lightning predict step
```

### Step 1: Model Prediction

**File**: `proteinfoundation/proteinflow/model_trainer_base.py`
```
Line 453: def predict_step(self, batch, batch_idx)
    │
    Line 471: cath_code = _extract_cath_code(batch) if fold_cond else None
    │
    Line 488: x = self.generate(...)
    │         ↓
    Line 279: def generate(self, nsamples, n, dt, self_cond, cath_code, ...)
              │
              │ # Calls the flow matching sampler
              Line 299: return self.fm.full_simulation(
                            predict_clean_n_v=self.predict_clean_n_v_w_guidance,
                            ...
                        )
```

### Step 2: Flow Matching Loop

**File**: `proteinfoundation/flow_matching/r3n_fm.py`
```
Line 401: def full_simulation(self, predict_clean_n_v, dt, nsamples, n, ...)
    │
    Line 459: nsteps = math.ceil(1.0 / dt)  # Typically 100-500 steps
    │
    Line 467: ts = self.get_schedule(mode, nsteps, p1)  # [0, 0.01, 0.02, ..., 1.0]
    │
    Line 486: x = self.sample_reference(n, shape=(nsamples,), ...)
    │         # Start with random noise x ~ N(0, I)
    │
    Line 493: for step in tqdm(range(nsteps)):  ◀── MAIN LOOP
    │    │
    │    Line 494: t = ts[step] * torch.ones(nsamples, device)
    │    │
    │    Line 500-504: nn_in = {"x_t": x, "t": t, "mask": mask}
    │    │             # Optional: add "cath_code", "x_sc" (self-conditioning)
    │    │
    │    Line 520: x_1_pred, v = predict_clean_n_v(nn_in)  ◀── CALLS MODEL
    │    │         # x_1_pred = predicted clean structure
    │    │         # v = velocity = (x_1_pred - x_t) / (1 - t)
    │    │
    │    Line 529: x, _ = self.simulation_step(x_t=x, v=v, t=t, dt=dt, ...)
    │              # Euler step: x_{t+dt} = x_t + v * dt
    │
    Line 540: return x  # Final clean structure
```

### Step 3: Model Forward Pass (with Guidance)

**File**: `proteinfoundation/proteinflow/model_trainer_base.py`
```
Line 113: def predict_clean_n_v_w_guidance(self, nn_in)
    │
    │ # Conditional prediction (with CATH code)
    Line 127: nn_out = self.nn(nn_in)  ◀── CALLS ProteinTransformerAF3
    │
    │ # CFG: x_pred = w * x_cond + (1-w) * x_uncond
    Line 147-154: x_pred = guidance_weight * x_pred + (1 - guidance_weight) * x_pred_uncond
    │
    │ # Compute velocity
    Line 161: v = self.fm.compute_v_from_x1_xt(x_pred, nn_in["x_t"], nn_in["t"])
    │
    Line 163: return x_pred, v
```

### Step 4: ProteinTransformerAF3 Forward

**File**: `proteinfoundation/nn/protein_transformer.py`
```
Line 643: def forward(self, batch_nn: Dict[str, torch.Tensor])
    │
    │ ═══════════════════════════════════════════════════════════════
    │ STAGE 1: CONDITIONING VARIABLES
    │ ═══════════════════════════════════════════════════════════════
    │
    Line 662: c = self.cond_factory(batch_nn)  # [b, n, dim_cond=256]
    │         │
    │         └── FeatureFactory builds conditioning from:
    │             • Time embedding (t → sinusoidal → MLP)
    │             • CATH fold embedding (if provided)
    │             • See feature_factory.py:133 FoldEmbeddingSeqFeat
    │
    Line 663: c = self.transition_c_2(self.transition_c_1(c, mask), mask)
    │         # Two MLP layers to process conditioning
    │
    │ ═══════════════════════════════════════════════════════════════
    │ STAGE 2: SEQUENCE REPRESENTATION
    │ ═══════════════════════════════════════════════════════════════
    │
    Line 666: coors_3d = batch_nn["x_t"] * mask[..., None]  # [b, n, 3]
    │
    Line 667-669: coors_embed = self.linear_3d_embed(coors_3d)  # [b, n, 512]
    │             # Linear projection: 3D coords → token space
    │
    Line 670: seq_f_repr = self.init_repr_factory(batch_nn)  # [b, n, 512]
    │         # Features: position index, chain breaks, etc.
    │
    Line 671: seqs = coors_embed + seq_f_repr  # [b, n, 512]
    │         # Combine coordinate embedding with feature embedding
    │
    │ ═══════════════════════════════════════════════════════════════
    │ STAGE 3: PAIR REPRESENTATION
    │ ═══════════════════════════════════════════════════════════════
    │
    Line 676-677: pair_rep = self.pair_repr_builder(batch_nn)  # [b, n, n, 256]
    │             │
    │             └── PairReprBuilder (Line 396-434) combines:
    │                 • rel_seq_sep: |i - j| binned, 127 dims
    │                 • x_sc_pair_dists: self-cond distances, 128 dims
    │                 • xt_pair_dists: current noisy distances, 64 dims
    │                 │
    │                 └── Concatenate → Linear → [b, n, n, 256]
    │
    │ ═══════════════════════════════════════════════════════════════
    │ STAGE 4: REGISTER TOKENS (optional)
    │ ═══════════════════════════════════════════════════════════════
    │
    Line 680: seqs, pair_rep, mask, c = self._extend_w_registers(...)
    │         # Prepend 10 learnable tokens to sequence
    │         # seqs: [b, n, 512] → [b, n+10, 512]
    │         # pair_rep: [b, n, n, 256] → [b, n+10, n+10, 256]
    │
    │ ═══════════════════════════════════════════════════════════════
    │ STAGE 5: TRANSFORMER TRUNK (15 layers) ◀── MAIN COMPUTATION
    │ ═══════════════════════════════════════════════════════════════
    │
    Line 683: for i in range(self.nlayers):  # nlayers = 15
    │    │
    │    Line 684-686: seqs = self.transformer_layers[i](seqs, pair_rep, c, mask)
    │    │             └── See Step 5 below
    │    │
    │    │ # Update pair representation every 3 layers (optional)
    │    Line 688-693: if i % 3 == 0 and self.update_pair_repr:
    │                      pair_rep = self.pair_update_layers[i](seqs, pair_rep, mask)
    │                      └── Triangular multiplicative updates (AlphaFold-style)
    │
    │ ═══════════════════════════════════════════════════════════════
    │ STAGE 6: COORDINATE DECODER
    │ ═══════════════════════════════════════════════════════════════
    │
    Line 696: seqs, pair_rep, mask = self._undo_registers(...)
    │         # Remove register tokens
    │
    Line 699: final_coors = self.coors_3d_decoder(seqs)  # [b, n, 3]
    │         │
    │         └── Sequential(LayerNorm, Linear(512 → 3))
    │
    Line 708: return {"coors_pred": final_coors}
```

### Step 5: Single Transformer Layer

**File**: `proteinfoundation/nn/protein_transformer.py`
```
Line 237: class MultiheadAttnAndTransition
    │
    Line 303: def forward(self, x, pair_rep, cond, mask)
        │
        │ # Parallel or sequential execution of attention + transition
        │
        Line 316-317 (parallel mode):
        │   x = self._apply_mha(x, pair_rep, cond, mask)
        │       + self._apply_transition(x, cond, mask)
        │
        └── _apply_mha (Line 291-295):
            │
            Line 292: x_attn = self.mhba(x, pair_rep, cond, mask)
                      └── MultiHeadBiasedAttentionADALN_MM (Line 170-206)
```

### Step 6: Attention with Pair Bias

**File**: `proteinfoundation/nn/protein_transformer.py`
```
Line 170: class MultiHeadBiasedAttentionADALN_MM
    │
    Line 191: def forward(self, x, pair_rep, cond, mask)
        │
        Line 202: pair_mask = mask[:, :, None] * mask[:, None, :]
        │
        Line 203: x = self.adaln(x, cond, mask)  # Adaptive LayerNorm
        │         └── Scales/shifts x based on conditioning (t, CATH)
        │
        Line 204: x = self.mha(node_feats=x, pair_feats=pair_rep, mask=pair_mask)
        │         └── PairBiasAttention (see Step 7)
        │
        Line 205: x = self.scale_output(x, cond, mask)
        │
        Line 206: return x * mask[..., None]
```

### Step 7: Core Attention Computation

**File**: `proteinfoundation/nn/pair_bias_attn/pair_bias_attn.py`
```
Line 32: class PairBiasAttention
    │
    Line 66: def forward(self, node_feats, pair_feats, mask)
        │
        Line 80: node_feats = self.node_norm(node_feats)  # LayerNorm
        │
        Line 81: pair_feats = self.pair_norm(pair_feats)  # LayerNorm on pairs
        │
        Line 82: q, k, v = self.to_qkv(node_feats).chunk(3, dim=-1)
        │         # Project to Q, K, V: [b, n, 512] → 3 × [b, n, 512]
        │
        Line 83-84: q = self.q_layer_norm(q)  # Optional Q/K normalization
        │           k = self.k_layer_norm(k)
        │
        Line 85: g = self.to_g(node_feats)  # Gating: [b, n, 512]
        │
        Line 86-90: b = self.to_bias(pair_feats)  # [b, n, n, 256] → [b, n, n, heads]
        │           b = rearrange(b, "b ... h -> b h ...")  # [b, heads, n, n]
        │           # THIS IS THE PAIR BIAS B
        │
        Line 91-93: q, k, v, g = map(rearrange(...), (q, k, v, g))
        │           # Reshape to [b, heads, n, dim_head]
        │
        Line 94: attn_feats = self._attn(q, k, v, b, mask)  ◀── CORE ATTENTION
        │
        Line 95-97: attn_feats = sigmoid(g) * attn_feats  # Gating
        │           attn_feats = rearrange(attn_feats, "b h n d -> b n (h d)")
        │
        Line 98: return self.to_out_node(attn_feats)  # Project back to 512
```

### Step 8: The Actual Attention Math

**File**: `proteinfoundation/nn/pair_bias_attn/pair_bias_attn.py`
```
Line 100: def _attn(self, q, k, v, b, mask) -> Tensor:
    │
    Line 102: sim = einsum("b h i d, b h j d -> b h i j", q, k) * self.scale
    │         │
    │         │  sim_before_scale = QK^T
    │         │  self.scale = dim_head ** -0.5 = 1/8 (for dim_head=64)
    │         │
    │         │  Shape: [batch, heads, n, n]
    │
    Line 103-105: if mask exists:
    │                 sim = sim.masked_fill(~mask, -inf)
    │
    Line 106: attn = torch.softmax(sim + b, dim=-1)
    │         │
    │         │  attn = softmax(QK^T × scale + B)
    │         │
    │         │  where B = pair bias (geometric memory)
    │
    Line 107: return einsum("b h i j, b h j d -> b h i d", attn, v)
              # Weighted sum of values
```

---

## Key Components Reference

| Component | File | Lines | Purpose |
|-----------|------|-------|---------|
| Flow Matching Loop | `flow_matching/r3n_fm.py` | 401-540 | Iterates t from 0→1, calls model at each step |
| Main Model | `nn/protein_transformer.py` | 437-709 | ProteinTransformerAF3 forward pass |
| Transformer Layer | `nn/protein_transformer.py` | 237-322 | Attention + MLP with AdaLN |
| Pair Bias Attention | `nn/pair_bias_attn/pair_bias_attn.py` | 32-107 | Core attention with geometric bias |
| Feature Factory | `nn/feature_factory.py` | - | Builds embeddings from features |
| Pair Representation | `nn/protein_transformer.py` | 396-434 | Builds pair features from distances |

---

## Model Dimensions (Default)

| Dimension | Value | Description |
|-----------|-------|-------------|
| token_dim | 512 | Sequence representation dimension |
| pair_repr_dim | 256 | Pair representation dimension |
| dim_cond | 256 | Conditioning dimension |
| nheads | 8 | Number of attention heads |
| dim_head | 64 | Dimension per head |
| nlayers | 15 | Number of transformer layers |
| num_registers | 10 | Learnable auxiliary tokens |

---

## Pair Bias Sources

The pair bias B that modifies attention encodes geometric priors from:

1. **rel_seq_sep** (127 dims): Relative sequence separation |i - j|, binned
2. **x_sc_pair_dists** (128 dims): Self-conditioned pairwise distances from previous prediction
3. **xt_pair_dists** (64 dims): Current noisy pairwise distances

These are:
- Concatenated: [b, n, n, ~320]
- Projected: Linear → [b, n, n, 256]
- Transformed per-head: Linear → [b, heads, n, n]
