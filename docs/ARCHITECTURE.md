# Proteina Model Architecture

This document provides a detailed walkthrough of how the Proteina protein structure generation model works, with exact file and line references.

## Overview

Proteina uses **flow matching** to generate protein structures. The core idea:
- Start with random noise at t=0
- Iteratively denoise over ~100-500 steps
- End with a clean protein structure at t=1

At each step, a neural network predicts a velocity field, which is converted to a "denoised estimate" (best guess of the final clean structure). The sampler then takes a small Euler step toward that prediction.

---

## High-Level Data Flow

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         FLOW MATCHING LOOP                               │
│   for t in [0 → 1]:                                                      │
│       x_t = noisy coordinates at timestep t                              │
│       v_pred = ProteinTransformerAF3(x_t, t, cath_code, ...)            │
│       x_1_pred = x_t + (1-t) * v_pred   (denoised estimate)             │
│       v = (x_1_pred - x_t) / (1-t)      (velocity field)                │
│       x_{t+dt} = x_t + v * dt            (Euler integration)            │
│       x_sc = x_1_pred                    (self-cond for next step)       │
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
│  │     • cond_factory(t, cath_code) → c [b,n,512] (conditioning)    │   │
│  │     • pair_repr_builder(distances) → pair_rep [b,n,n,256]        │   │
│  └──────────────────────────────────────────────────────────────────┘   │
│                                    │                                     │
│                                    ▼                                     │
│  ┌──────────────────────────────────────────────────────────────────┐   │
│  │ (2) TRANSFORMER TRUNK (nlayers varies: 10 for 60M, 15 for 200M) │   │
│  │     for i in range(nlayers):                                      │   │
│  │         seqs = MultiheadAttnAndTransition(seqs, pair_rep, c)     │   │
│  │         if update_pair_repr and pair_update_layers[i] is not None:│   │
│  │             pair_rep = PairReprUpdate(seqs, pair_rep)  # optional │   │
│  └──────────────────────────────────────────────────────────────────┘   │
│                                    │                                     │
│                                    ▼                                     │
│  ┌──────────────────────────────────────────────────────────────────┐   │
│  │ (3) COORDINATE DECODER                                            │   │
│  │     seqs [b,n,512] → LayerNorm → Linear → v_pred [b,n,3]         │   │
│  │     (optional) pair_rep → distogram prediction head               │   │
│  └──────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────┘
```

**Note**: The network outputs a **velocity** `v_pred`, not `x_1` directly. The conversion to the denoised estimate `x_1_pred = x_t + (1-t) * v_pred` happens outside the network in `_nn_out_to_x_clean` (see "The Denoised Estimate" section below).

---

## Detailed Data Flow Trace

### Entry Point: Inference Script

**File**: `proteinfoundation/inference.py`
```
Line ~326: trainer.predict(model, dataloader)
    → Calls Lightning predict step for each batch
```

### Step 1: Model Prediction

**File**: `proteinfoundation/proteinflow/model_trainer_base.py`
```
Line 462: def predict_step(self, batch, batch_idx)
    │
    Line 475-476: cath_code = _extract_cath_code(batch) if fold_cond else None
    │
    Line 491: x = self.generate(...)
    │         ↓
    Line 515: def generate(self, nsamples, n, dt, self_cond, cath_code, ...)
              │
              │ # Calls the flow matching sampler
              Line 548: return self.fm.full_simulation(
                            predict_clean_n_v=self.predict_clean_n_v_w_guidance,
                            ...
                        )
```

### Step 2: Flow Matching Loop

**File**: `proteinfoundation/flow_matching/r3n_fm.py`
```
Line 404: def full_simulation(self, predict_clean_n_v, dt, nsamples, n, ...)
    │
    Line 462-464: nsteps = math.ceil(1.0 / dt)  # Typically 100-500 steps
    │
    Line 470-477: ts = self.get_schedule(mode, nsteps, p1)  # [0, 0.01, 0.02, ..., 1.0]
    │
    Line 489-491: x = self.sample_reference(n, shape=(nsamples,), ...)
    │             # Start with random noise x ~ N(0, I)
    │
    Line 496: for step in tqdm(range(nsteps)):  ◀── MAIN LOOP
    │    │
    │    Line 497: t = ts[step] * torch.ones(nsamples, device)
    │    │
    │    Line 502-507: nn_in = {"x_t": x, "t": t, "mask": mask}
    │    │             # Optional: add "cath_code", "x_sc" (self-conditioning)
    │    │
    │    Line 520-521: if step > 0 and self_cond:
    │    │                 nn_in["x_sc"] = x_1_pred  # Feed back previous estimate
    │    │
    │    Line 523: x_1_pred, v = predict_clean_n_v(nn_in)  ◀── CALLS MODEL
    │    │         # x_1_pred = predicted clean structure (denoised estimate)
    │    │         # v = velocity = (x_1_pred - x_t) / (1 - t)
    │    │
    │    Line 532-542: x, _ = self.simulation_step(x_t=x, v=v, t=t, dt=dt, ...)
    │                  # Euler step: x_{t+dt} = x_t + v * dt (ODE mode)
    │                  # Or SDE mode with added stochastic noise
    │
    Line 543: return x  # Final clean structure
```

### Step 3: Model Forward Pass (with Guidance)

**File**: `proteinfoundation/proteinflow/model_trainer_base.py`
```
Line 116: def predict_clean_n_v_w_guidance(self, batch, guidance_weight, autoguidance_ratio)
    │
    │ # Forward pass through the transformer
    Line 139: nn_out = self.nn(batch)  ◀── CALLS ProteinTransformerAF3
    │
    │ # Convert velocity output to x_1 prediction
    Line 140: x_pred = self._nn_out_to_x_clean(nn_out, batch)
    │         └── x_1_pred = x_t + (1-t) * v_pred   (see model_trainer_base.py:83-84)
    │
    │ # Optional CFG + autoguidance blending:
    │ # x_pred = w * x_cond + (1-w) * (alpha * x_auto + (1-alpha) * x_uncond)
    Line 161-164: x_pred = guidance_weight * x_pred + (1 - guidance_weight) * (...)
    │
    │ # Compute velocity from the (possibly guided) denoised estimate
    Line 166: v = self.fm.xt_dot(x_pred, batch["x_t"], batch["t"], batch["mask"])
    │         └── v = (x_1_pred - x_t) / (1 - t)   (see r3n_fm.py:167-197)
    │
    Line 167: return x_pred, v
```

### Step 4: ProteinTransformerAF3 Forward

**File**: `proteinfoundation/nn/protein_transformer.py`
```
Line 672: def forward(self, batch_nn: Dict[str, torch.Tensor], tracker=None)
    │
    │ ═══════════════════════════════════════════════════════════════
    │ STAGE 1: CONDITIONING VARIABLES
    │ ═══════════════════════════════════════════════════════════════
    │
    Line 696: c = self.cond_factory(batch_nn)  # [b, n, dim_cond=512]
    │         │
    │         └── FeatureFactory builds conditioning from:
    │             • Time embedding (t → sinusoidal → MLP)
    │             • CATH fold embedding (if provided)
    │             • See feature_factory.py:133 FoldEmbeddingSeqFeat
    │
    Line 697: c = self.transition_c_2(self.transition_c_1(c, mask), mask)
    │         # Two MLP layers to process conditioning
    │
    │ ═══════════════════════════════════════════════════════════════
    │ STAGE 2: SEQUENCE REPRESENTATION
    │ ═══════════════════════════════════════════════════════════════
    │
    Line 700: coors_3d = batch_nn["x_t"] * mask[..., None]  # [b, n, 3]
    │
    Line 701-703: coors_embed = self.linear_3d_embed(coors_3d)  # [b, n, 512]
    │             # Linear projection: 3D coords → token space
    │
    Line 704: seq_f_repr = self.init_repr_factory(batch_nn)  # [b, n, 512]
    │         # Features: position index, chain breaks, self-cond coords, etc.
    │
    Line 705: seqs = coors_embed + seq_f_repr  # [b, n, 512]
    │         # Combine coordinate embedding with feature embedding
    │
    │ ═══════════════════════════════════════════════════════════════
    │ STAGE 3: PAIR REPRESENTATION
    │ ═══════════════════════════════════════════════════════════════
    │
    Line 710-711: pair_rep = self.pair_repr_builder(batch_nn)  # [b, n, n, 256]
    │             │
    │             └── PairReprBuilder (Line 425-463) combines:
    │                 • rel_seq_sep: |i - j| binned, 127 dims
    │                 • x_sc_pair_dists: self-cond distances, 128 dims
    │                 • xt_pair_dists: current noisy distances, 64 dims
    │                 │
    │                 └── Concatenate → Linear → [b, n, n, 256]
    │
    │ ═══════════════════════════════════════════════════════════════
    │ STAGE 4: REGISTER TOKENS
    │ ═══════════════════════════════════════════════════════════════
    │
    Line 714: seqs, pair_rep, mask, c = self._extend_w_registers(...)
    │         # Prepend 10 learnable tokens to sequence
    │         # seqs: [b, n, 512] → [b, n+10, 512]
    │         # pair_rep: [b, n, n, 256] → [b, n+10, n+10, 256]
    │
    │ ═══════════════════════════════════════════════════════════════
    │ STAGE 5: TRANSFORMER TRUNK ◀── MAIN COMPUTATION
    │ ═══════════════════════════════════════════════════════════════
    │
    Line 720: for i in range(self.nlayers):  # nlayers varies by model (10, 15, etc.)
    │    │
    │    Line 727-729: seqs = self.transformer_layers[i](seqs, pair_rep, c, mask)
    │    │             └── See Step 5 below
    │    │
    │    │ # Update pair representation (optional, model-dependent)
    │    Line 735-740: if self.update_pair_repr:
    │                      if self.pair_update_layers[i] is not None:
    │                          pair_rep = self.pair_update_layers[i](seqs, pair_rep, mask)
    │                          └── Triangular multiplicative updates (AlphaFold-style)
    │                              Uses gradient checkpointing for memory efficiency
    │
    │ ═══════════════════════════════════════════════════════════════
    │ STAGE 6: COORDINATE DECODER
    │ ═══════════════════════════════════════════════════════════════
    │
    Line 743: seqs, pair_rep, mask = self._undo_registers(...)
    │         # Remove register tokens
    │
    Line 746: final_coors = self.coors_3d_decoder(seqs)  # [b, n, 3]
    │         │
    │         └── Sequential(LayerNorm, Linear(512 → 3))
    │
    │ # Optional: auxiliary distogram prediction head
    Line 748-754: if self.update_pair_repr and self.num_buckets_predict_pair:
    │                 pair_pred = self.pair_head_prediction(pair_rep)
    │                 nn_out["pair_pred"] = pair_pred  # for auxiliary training loss
    │
    Line 755-756: nn_out["coors_pred"] = final_coors
                  return nn_out
```

### Step 5: Single Transformer Layer

**File**: `proteinfoundation/nn/protein_transformer.py`
```
Line 251: class MultiheadAttnAndTransition
    │
    Line 324: def forward(self, x, pair_rep, cond, mask, capture=None)
        │
        │ # Parallel or sequential execution of attention + transition
        │
        │ Parallel mode:
        │   x = self._apply_mha(x, pair_rep, cond, mask)
        │       + self._apply_transition(x, cond, mask)
        │
        │ Sequential mode:
        │   x = self._apply_mha(x, pair_rep, cond, mask)
        │   x = self._apply_transition(x, cond, mask)
        │
        └── _apply_mha (Line 305-316):
            │
            Line 313: x_attn = self.mhba(x, pair_rep, cond, mask, capture)
                      └── MultiHeadBiasedAttentionADALN_MM (Line 176)
```

Both `_apply_mha` and `_apply_transition` support optional residual connections.
In parallel mode, if both residuals are enabled, the transition residual is
disabled to avoid adding x twice (Line 287-288).

### Step 6: Attention with Pair Bias

**File**: `proteinfoundation/nn/protein_transformer.py`
```
Line 176: class MultiHeadBiasedAttentionADALN_MM
    │
    Line 197: def forward(self, x, pair_rep, cond, mask, capture=None)
        │
        Line 216: pair_mask = mask[:, :, None] * mask[:, None, :]
        │
        Line 217: x = self.adaln(x, cond, mask)  # Adaptive LayerNorm
        │         └── Scales/shifts x based on conditioning (t, CATH)
        │
        Line 218: x = self.mha(node_feats=x, pair_feats=pair_rep, mask=pair_mask)
        │         └── PairBiasAttention (see Step 7)
        │
        Line 219: x = self.scale_output(x, cond, mask)
        │
        return x * mask[..., None]
```

### Step 7: Core Attention Computation

**File**: `proteinfoundation/nn/pair_bias_attn/pair_bias_attn.py`
```
Line 35: class PairBiasAttention
    │
    Line 69: def forward(self, node_feats, pair_feats, mask, capture=None)
        │
        Line 85: node_feats = self.node_norm(node_feats)  # LayerNorm
        │
        Line 86: pair_feats = self.pair_norm(pair_feats)  # LayerNorm on pairs
        │
        Line 87: q, k, v = self.to_qkv(node_feats).chunk(3, dim=-1)
        │         # Project to Q, K, V: [b, n, 512] → 3 × [b, n, 512]
        │
        Line 88-89: q = self.q_layer_norm(q)  # Optional Q/K normalization
        │           k = self.k_layer_norm(k)
        │
        Line 90: g = self.to_g(node_feats)  # Gating: [b, n, 512]
        │
        Line 91-95: b = rearrange(self.to_bias(pair_feats), "b ... h -> b h ...")
        │           # [b, n, n, 256] → [b, heads, n, n]
        │           # THIS IS THE PAIR BIAS B
        │
        Line 96-98: q, k, v, g = map(rearrange(...), (q, k, v, g))
        │           # Reshape to [b, heads, n, dim_head]
        │
        Line 99: attn_feats = self._attn(q, k, v, b, mask, capture)  ◀── CORE ATTENTION
        │
        Line 100-102: attn_feats = sigmoid(g) * attn_feats  # Gating
        │             attn_feats = rearrange(attn_feats, "b h n d -> b n (h d)")
        │
        Line 103: return self.to_out_node(attn_feats)  # Project back to 512
```

### Step 8: The Actual Attention Math

**File**: `proteinfoundation/nn/pair_bias_attn/pair_bias_attn.py`
```
Line 105: def _attn(self, q, k, v, b, mask, capture) -> Tensor:
    │
    Line 128: qk_raw = einsum("b h i d, b h j d -> b h i j", q, k)
    │
    Line 131: sim = qk_raw * self.scale
    │         │
    │         │  self.scale = dim_head ** -0.5 = 1/8 (for dim_head=64)
    │         │
    │         │  Shape: [batch, heads, n, n]
    │
    Line 133-135: if mask exists:
    │                 sim = sim.masked_fill(~mask, -inf)
    │
    Line 138: attn = torch.softmax(sim + b, dim=-1)
    │         │
    │         │  attn = softmax(QK^T × scale + B)
    │         │
    │         │  where B = pair bias (geometric memory)
    │
    Line 148: return einsum("b h i j, b h j d -> b h i d", attn, v)
              # Weighted sum of values
```

---

## Training vs Inference

### Training Loop

**File**: `proteinfoundation/proteinflow/model_trainer_base.py` (Lines 227-306)

During training, the model sees **one random timestep per sample**:

```
Given: x_1 (clean protein backbone from dataset)

1. Sample noise:     x_0 ~ N(0, I)                                    [Line 249-251]
2. Sample time:      t ~ distribution(0, 1)                           [Line 248]
3. Interpolate:      x_t = (1-t) * x_0 + t * x_1                     [Line 257]
4. Self-conditioning (50% chance):                                     [Line 290-292]
   │  x_pred_sc, _ = self.predict_clean(batch)
   │  batch["x_sc"] = detach(x_pred_sc)
5. Forward pass:     nn_out = self.nn(batch)                           [Line 294]
   │  v_pred = nn_out["coors_pred"]
   │  x_1_pred = x_t + (1-t) * v_pred           (via _nn_out_to_x_clean)
6. Loss:             MSE(x_1, x_1_pred) * 1/(1-t)²                   [Line 297-299]
7. Auxiliary loss:   distogram prediction (optional)                   [Line 302-306]
```

### Inference Loop

**File**: `proteinfoundation/flow_matching/r3n_fm.py` (Lines 496-543)

During inference, the model runs **hundreds of sequential steps**:

```
Start: x = pure noise ~ N(0, I)

for step in 0..nsteps:
    t = schedule[step]
    if step > 0: x_sc = x_1_pred              # self-conditioning (every step)
    v_pred = NN(x, t, mask, x_sc)             # forward pass
    x_1_pred = x + (1-t) * v_pred             # denoised estimate
    v = (x_1_pred - x) / (1-t)               # velocity field
    x = x + v * dt                            # Euler step (ODE)
    OR: x = x + (v + g*score)*dt + noise      # SDE mode (adds stochasticity)

return x  # final protein structure
```

### Key Differences

| Aspect | Training | Inference |
|--------|----------|-----------|
| Input x_t | Interpolation from known x_1 | Iterative from pure noise |
| Time t | Random single sample | Sequential 0→1 over schedule |
| Forward passes | 1 (or 2 with self-cond) | ~400 (one per timestep) |
| Self-conditioning | 50% chance, gradients detached | Every step after first |
| Output used for | MSE loss computation | ODE/SDE integration step |
| Gradients | Yes (backprop) | No (torch.no_grad) |
| CATH code masking | Progressive random masking of hierarchy levels | Fixed level (C, A, or T) |

---

## The Denoised Estimate (x_1_pred)

At every sampling step, the model produces `x_1_pred` — its current best guess of what the **final clean protein structure** looks like, given the current noisy state `x_t` and time `t`.

### How it's generated

The network outputs a **velocity** `v_pred`, not `x_1` directly. The conversion happens in `_nn_out_to_x_clean` at `model_trainer_base.py:63-89`:

```python
# The network predicts velocity v
v_pred = nn_out["coors_pred"]

# Convert to denoised estimate: "if I follow this velocity for the remaining time, where do I end up?"
x_1_pred = x_t + (1 - t) * v_pred
```

### What it's used for

The denoised estimate serves three purposes:

1. **Velocity computation** — the velocity field `v = (x_1_pred - x_t) / (1-t)` drives the ODE/SDE integration (`r3n_fm.py:167-197`, the `xt_dot` method).

2. **Self-conditioning (x_sc)** — at the next sampling step, `x_1_pred` is fed back as input to the model (`r3n_fm.py:520-521`). The network uses it in two ways via `feature_factory.py`:
   - As direct 3D coordinate features embedded into the sequence representation (`XscSeqFeat`, feature_factory.py:391-404)
   - As binned pairwise distance features fed into the pair representation (`XscPairwiseDistancesPairFeat`, feature_factory.py:528-551)
   - When no self-conditioning is available (first step), both default to zeros.

3. **Training loss target** — `MSE(x_1_pred, x_1_true)` is the primary training objective.

---

## Loss Function

### Proteina's loss formulation

**File**: `proteinfoundation/proteinflow/proteina.py` (Lines 160-202)

```python
err = (x_1 - x_1_pred) * mask                    # prediction error
loss = sum(err**2) / nres                         # MSE per residue
loss = loss * 1 / ((1 - t)**2 + 1e-5)            # time weighting
```

### Equivalence to standard flow matching

The standard flow matching loss is `MSE(v_pred, v_target)` where `v_target = x_1 - x_0`.

Proteina's loss looks different (it compares `x_1_pred` to `x_1`, not velocities), but is **mathematically equivalent**. Here's why:

```
err = x_1 - x_1_pred
    = x_1 - (x_t + (1-t) * v_pred)                  # substituting x_1_pred conversion
    = x_1 - ((1-t)*x_0 + t*x_1 + (1-t)*v_pred)      # substituting x_t = (1-t)*x_0 + t*x_1
    = (1-t) * (x_1 - x_0) - (1-t) * v_pred
    = (1-t) * (v_target - v_pred)

So: err**2 = (1-t)**2 * (v_target - v_pred)**2
```

Then the time weighting `1/(1-t)**2` cancels exactly:

```
loss = err**2 * 1/(1-t)**2 = (v_target - v_pred)**2 = MSE(v_pred, v_target)
```

### Why use the x_1 parameterization?

Working in denoised-estimate space rather than velocity space is more convenient because:
- **Self-conditioning** feeds back `x_1_pred` (a predicted structure), not a velocity vector
- **Auxiliary losses** (distogram, motif) operate on predicted coordinates, not velocities
- **Flexibility** — the same loss code works for both `target_pred: v` and `target_pred: x_1` parameterizations

---

## Key Components Reference

| Component | File | Lines | Purpose |
|-----------|------|-------|---------|
| Flow Matching Loop | `flow_matching/r3n_fm.py` | 404-543 | Iterates t from 0→1, calls model at each step |
| Velocity Computation | `flow_matching/r3n_fm.py` | 167-197 | `xt_dot`: computes v = (x_1_pred - x_t)/(1-t) |
| SDE/ODE Step | `flow_matching/r3n_fm.py` | 255-337 | `step_euler`: Euler integration with optional noise |
| Guidance Logic | `proteinflow/model_trainer_base.py` | 116-167 | CFG + autoguidance blending |
| NN→x_1 Conversion | `proteinflow/model_trainer_base.py` | 63-89 | `_nn_out_to_x_clean`: velocity to denoised estimate |
| Training Step | `proteinflow/model_trainer_base.py` | 227-306 | Noising, forward pass, loss |
| Loss Computation | `proteinflow/proteina.py` | 160-202 | MSE(x_1, x_1_pred) with 1/(1-t)^2 weighting |
| Main Model | `nn/protein_transformer.py` | 467-756 | ProteinTransformerAF3 class and forward pass |
| Transformer Layer | `nn/protein_transformer.py` | 251-350 | Attention + MLP with AdaLN |
| Attention + AdaLN | `nn/protein_transformer.py` | 176-219 | MultiHeadBiasedAttentionADALN_MM |
| Pair Bias Attention | `nn/pair_bias_attn/pair_bias_attn.py` | 35-148 | Core attention with geometric bias |
| Pair Repr Builder | `nn/protein_transformer.py` | 425-463 | Builds pair features from distances |
| Pair Repr Update | `nn/protein_transformer.py` | 354-422 | Triangular multiplicative updates (optional) |
| Feature Factory | `nn/feature_factory.py` | - | Builds embeddings from features |
| Self-Cond Features | `nn/feature_factory.py` | 391-404, 528-551 | XscSeqFeat, XscPairwiseDistancesPairFeat |

---

## Model Dimensions (Default)

| Dimension | Value | Description |
|-----------|-------|-------------|
| token_dim | 512 | Sequence representation dimension |
| pair_repr_dim | 256 | Pair representation dimension |
| dim_cond | 512 | Conditioning dimension |
| nheads | 8 | Number of attention heads |
| dim_head | 64 | Dimension per head (token_dim / nheads) |
| nlayers | 10-15 | Number of transformer layers (model-dependent: 10 for 60M, 15 for 200M) |
| num_registers | 10 | Learnable auxiliary tokens prepended to sequence |

---

## Pair Bias Sources

The pair bias B that modifies attention encodes geometric priors from:

1. **rel_seq_sep** (127 dims): Relative sequence separation |i - j|, binned
2. **x_sc_pair_dists** (128 dims): Self-conditioned pairwise distances from previous prediction
3. **xt_pair_dists** (64 dims): Current noisy pairwise distances

These are:
- Concatenated: [b, n, n, ~320]
- Projected: Linear → [b, n, n, 256]
- Transformed per-head: Linear → [b, heads, n, n] (inside PairBiasAttention)
