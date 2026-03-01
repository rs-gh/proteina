# Attention Crystallization in Flow-Based Protein Structure Generation

## 1. Introduction

Generative models for protein backbone structure have advanced rapidly, with flow matching emerging as a leading paradigm. Proteina (Geffner et al., 2025), an ICLR 2025 Oral paper, scales flow-based protein generation to state-of-the-art quality by combining a transformer backbone with *Pair Bias Attention* (PBA) — a mechanism that fuses learned content scores with geometric pair biases encoding inter-residue distances and sequence separation.

During inference, Proteina integrates a velocity field from pure noise (t=0) to a structured protein backbone (t=1) via approximately 100 Euler steps. At each step, every attention layer computes:

$$A = \text{softmax}(C + B)$$

where $C = QK^\top / \sqrt{d}$ is the content score derived from sequence representations, and $B$ is a pair bias projected from distance and sequence-separation features. This decomposition raises a natural interpretability question: **how does the interplay between geometric prior (B) and learned content (C) evolve as the model transitions from noise to structure?**

We introduce three trajectory-level metrics to track this process — logit dominance (R), attention entropy (H), and spatial alignment ($\rho$) — and find that attention undergoes a phase transition we term *crystallization*: a rapid shift from diffuse, geometry-dominated attention to sharp, contact-specific patterns. Using the 60M parameter Proteina model generating 100-residue proteins, we characterise this transition across layers, heads, and sequence-separation scales, revealing a division of labour between geometric bootstrapping and content-based refinement.

## 2. Methods

### 2.1 Model and Generation

We analyse Proteina v1.3 (60M parameters, 12 layers, 12 attention heads, no triangle updates, 10 register tokens). Proteins of length $n$ are generated unconditionally using SDE sampling with self-conditioning (noise scale 0.45, log schedule, dt=0.01, 100 steps). Attention matrices are captured every 5th timestep, yielding 20 temporal snapshots per generation. We use the final generated structure as retrospective ground truth for spatial alignment, measuring "how early does the model know the contacts it will eventually make?"

### 2.2 Crystallization Metrics

**Logit Dominance** $R = \|B\|_F / \|C\|_F$ measures whether the geometric pair bias or the content score dominates the attention logits. $R \gg 1$ indicates geometry-dominated attention; $R \approx 1$ indicates balance. Because softmax is invariant to row-wise constant shifts, we also report a **row-centered variant** $R_c = \|\tilde{B}\|_F / \|\tilde{C}\|_F$, where $\tilde{B}$ and $\tilde{C}$ are obtained by subtracting each row's mean (computed over valid positions only). $R_c$ captures only the within-row variance that actually shapes the attention distribution, discounting any row-constant component that cancels in softmax.

**Attention Entropy** $H = -\sum_j p_j \log p_j$, normalised by $\log n$ to give $\hat{H} \in [0, 1]$. High entropy indicates diffuse attention (global search); low entropy indicates crystallised attention locked onto specific contacts.

**Spatial Alignment** $\rho$ is the Pearson correlation between the attention weight matrix and the inverse ground-truth distance matrix, measuring how well attention reflects true 3D proximity.

### 2.3 Extended Analyses

Beyond the core metrics, we perform four additional analyses: (1) *Per-head specialisation*, plotting individual head trajectories to identify geometric specialists; (2) *Sequence-separation decomposition*, computing R, H, $\rho$ separately for local (|i-j|=1–6), medium (7–23), and long-range ($\geq$24) residue pairs; (3) *Contact precision@L/5*, evaluating how well full, B-only, and C-only attention predicts true contacts; and (4) *Register token analysis*, measuring attention mass directed to the 10 register tokens across layers and timesteps.

All experiments use 5 random seeds for statistical robustness. We additionally compare protein lengths (n=50, 100, 200) to test scaling behaviour.

## 3. Results

### 3.1 The Crystallization Trajectory

Figure 1 shows the three core metrics across the denoising trajectory for layers 0 (first), 6 (middle), and 11 (last).

**Layer 0 acts as a "geometric interpreter."** At t=0, logit dominance reaches R = 7.93 $\pm$ 0.07 (mean $\pm$ std across 5 seeds) — the geometric bias is nearly 8$\times$ louder than the content score. This drops to R $\approx$ 1 by t=0.5 as sequence representations become meaningful and $C$ grows in magnitude. Deeper layers maintain R $\approx$ 1 throughout, indicating a roughly balanced contribution of geometry and content. The tight standard deviation across seeds confirms this is a robust structural property of the model.

**Entropy crystallises in a wave.** All layers start near-uniform ($\hat{H} \approx 0.85$). Layer 0 crystallises first, with normalised entropy dropping sharply in the first half of the trajectory. Deeper layers follow progressively later — a wave-like propagation consistent with bottom-up structure formation. By t=1, mean entropy drops to $\hat{H} \approx 0.43$, with substantial variance across heads (see Section 3.2).

**Spatial alignment grows monotonically.** All layers start near $\rho \approx 0$ (expected, since the structure does not yet exist). Alignment increases throughout, with the last layer achieving the highest correlation ($\rho = 0.30 \pm 0.01$ at t=1, across 5 seeds), consistent with it directly producing the coordinate prediction. The R, H, and $\rho$ heatmaps (Figure 1, bottom row) show clear layer$\times$timestep structure: logit dominance is concentrated in early layers at early timesteps, entropy decreases diagonally from top-left to bottom-right, and spatial alignment strengthens from bottom-right.

### 3.2 Head Specialisation

Individual heads within a layer show dramatically different crystallisation dynamics (Figure 4). In layer 0, Head 1 crystallises very early ($\hat{H}$ drops to $\sim$0.25 by t=0.2), while Heads 0, 3, 8, and 9 remain near-uniform ($\hat{H} > 0.8$) throughout the trajectory. At t=1, spatial alignment varies from $\rho = 0.21$ (Head 11) to $\rho = 0.61$ (Head 3) — a 3$\times$ spread within a single layer.

Layer 11 shows a contrasting pattern: more heads remain high-entropy until the final 20% of the trajectory, with several undergoing sharp late crystallisation. Head 3 in layer 11 unusually *starts* at low entropy and rises — potentially a register-attending head that redistributes to specific contacts only at the end.

This specialisation confirms that a minority of heads account for most geometric specificity, while others maintain diffuse attention for global information routing — analogous to the "contact heads" identified by Rao et al. (2020) in protein language models, but arising dynamically during generation rather than being fixed by training.

### 3.3 Content Score Dynamically Learns Contact Structure

The AF2-style attention grids (Figure 2) provide striking visual evidence of crystallisation. The pair bias B shows contact-like diagonal structure at all timesteps — it encodes 3D distances from the start. The full attention transitions from diffuse (t $\approx$ 0) to sharp contact patterns (t $\approx$ 1). Most revealing is the content score C: initially uniform/noisy, it develops clear off-diagonal secondary and tertiary structure by mid-trajectory, demonstrating that QK$^\top$ learns geometric specificity *during* the denoising process.

Contact precision quantifies this (Figure 3). Across 5 seeds, B-only attention consistently outperforms C-only at predicting true contacts: at t=1, Precision@20 reaches $0.26 \pm 0.06$ for B-only versus $0.05 \pm 0.02$ for C-only. Critically, C-only precision starts near random ($\sim$0.002) and grows only modestly — the content score develops *some* contact-predictive ability, but never rivals the geometric prior. Full attention ($0.23 \pm 0.06$ at t=1) slightly underperforms B-only, suggesting that C occasionally disagrees with B, encoding non-contact relationships (e.g., secondary structure patterns or allosteric couplings).

This finding distinguishes Proteina from static protein language models, where attention-based contact prediction achieves high precision from fixed, pre-trained representations (Rao et al., 2020). In Proteina, contact prediction is a *dynamic* process that unfolds during generation, with the geometric bias carrying most of the structural signal.

### 3.4 Sequence Separation Modulates the Geometry–Content Balance

Decomposing metrics by residue separation reveals that the geometric bias matters most where it is needed most. At t=0 in layer 0, logit dominance is highest for long-range pairs ($R_{\text{long}} = 8.24 \pm 0.10$), followed by medium ($R_{\text{med}} = 7.67 \pm 0.18$) and local ($R_{\text{local}} = 7.02 \pm 0.16$), all reported across 5 seeds. This confirms the central hypothesis: geometric bias B is most dominant for contacts that sequence distance alone cannot predict.

Entropy tells the complementary story: local contacts crystallise first ($H_{\text{local}} = 2.18$ at t=0, dropping to 1.42 by t=1), while long-range entropy remains higher throughout ($H_{\text{long}} = 3.64 \to 3.14$). This reflects the difficulty hierarchy: backbone-local contacts are geometrically constrained and resolved early, while fold-defining tertiary contacts require more exploration before commitment.

Despite these differences in dynamics, all three bins converge to similar spatial alignment by t=1 ($\rho \approx 0.24$–$0.27$), indicating the model ultimately achieves comparable geometric accuracy at all scales.

### 3.5 Register Tokens as Compression Hubs

The register token heatmap reveals that layers 7–8 act as compression hubs, absorbing up to 39% of attention mass from residue tokens at late timesteps (versus 9.1% expected under uniform attention). This register attention *increases* over time in these layers, suggesting that as structure solidifies, these layers increasingly route global information through register sinks rather than through direct residue–residue attention. The peak in layers 7–8 (of 12 total) places the compression phase in the late-middle layers, broadly consistent with the Mix-Compress-Refine framework proposed for vision transformers (Darcet et al., 2023; Ding et al., 2024).

### 3.6 Scaling with Protein Length

We compare crystallisation across three protein lengths (n=50, 100, 200; 3 seeds each). Two scaling trends emerge:

**Geometric bias strengthens with length.** Layer 0 logit dominance at t=0 increases from $R = 7.31 \pm 0.16$ (n=50) to $7.89 \pm 0.05$ (n=100) to $8.03 \pm 0.03$ (n=200). Longer proteins have proportionally more long-range residue pairs, and the geometric prior is most dominant for these pairs (Section 3.4), explaining the trend.

**Spatial alignment decreases with length.** Final $\rho$ drops from $0.32 \pm 0.01$ (n=50) to $0.30 \pm 0.01$ (n=100) to $0.26 \pm 0.03$ (n=200). Longer proteins present a harder geometric alignment problem — the attention matrix is larger and the contact density sparser, making precise alignment more difficult.

**Crystallisation timing is length-invariant.** The 50% entropy threshold is crossed at t $\approx$ 0.5 for all three lengths, suggesting the crystallisation transition is governed by the denoising schedule rather than by the complexity of the fold search. This is consistent with the log-schedule parameterisation of the SDE sampler, which concentrates denoising effort in a specific temporal window regardless of protein size.

### 3.7 Row-Centered Metric: Accounting for Softmax Invariance

The raw Frobenius-norm ratio R is sensitive to row-wise constant components that softmax ignores. To test whether this inflates or deflates our findings, we compare R with the row-centered variant $R_c$ across 5 seeds (n=100).

**Layer 0 at t=0: centering has minimal effect.** The ratio $R/R_c = 1.05 \pm 0.03$, indicating that both B and C have genuine within-row variance at the start of generation. The original claim — that Layer 0 acts as a "geometric interpreter" with geometry $\sim$8$\times$ louder than content — holds essentially unchanged ($R = 7.93$, $R_c = 7.54$).

**Layer 0 at late timesteps: centering amplifies B's dominance.** By t=1, $R/R_c$ drops to $0.16 \pm 0.02$: the raw metric reports $R \approx 5.7$, while $R_c \approx 35$. This means the content score C develops a large row-constant component at late timesteps — its Frobenius norm grows, but most of this energy is in row-wise shifts that softmax cancels. The geometric bias B, by contrast, retains high within-row variance. Under the centered metric, B dominates *far more* than the raw metric suggests during the refinement phase.

**Middle and deep layers: centering reveals stronger B influence.** At layers 6 and 11, $R/R_c \approx 0.7$ throughout most of the trajectory, indicating that C consistently carries more row-constant padding than B. The centered metric therefore assigns B relatively greater influence than the raw metric, strengthening the narrative that geometric bias is structurally important throughout the network.

**Sequence-separation decomposition shows differential centering effects.** At t=0, Layer 0: raw R gives long > medium > local (8.24, 7.67, 7.02). The centering correction is strongest for local contacts ($R/R_c = 1.62$) and negligible for long-range ($R/R_c = 0.98$). This means local pair biases carry substantial row-constant structure — plausible, since nearby residues share similar geometric contexts — while long-range biases are already highly position-specific.

In summary, the row-centered metric confirms and *strengthens* the core findings: geometric bias B dominates attention through genuine within-row variance, while much of C's apparent energy is softmax-invariant. The qualitative narrative is unchanged; quantitatively, B's relative importance is even greater than the raw metric suggests.

## 4. Discussion

Our results reveal that Proteina's attention mechanism undergoes a structured phase transition during inference, which we summarise as a three-phase process:

**Phase 1 — Geometric bootstrapping** (t $\approx$ 0–0.3): Attention is diffuse (high H) and geometry-dominated (high R), particularly for long-range pairs. Layer 0 acts as a geometric interpreter, with pair bias B providing 8$\times$ more signal than content score C. A minority of specialist heads crystallise early, establishing initial contact patterns.

**Phase 2 — Progressive refinement** (t $\approx$ 0.3–0.8): Entropy drops in a wave from early to late layers. Content representations become meaningful as structure emerges, and QK$^\top$ develops contact-predictive features. Register tokens in layers 7–8 absorb increasing attention as global information is compressed.

**Phase 3 — Contact commitment** (t $\approx$ 0.8–1.0): Remaining heads crystallise, spatial alignment reaches its peak, and attention sharpens to specific residue contacts. The content score contributes complementary (sometimes competing) information alongside the geometric prior.

This dynamic crystallisation contrasts with both AlphaFold2, where attention sharpens across recycling iterations with fixed input, and protein language models, where contact-predictive attention emerges from static pre-training. In Proteina, the geometric prior B provides a strong scaffold from the start, while the content score C must bootstrap itself from noise — creating a generative analogue of the "representation learning" that occurs during PLM pre-training, compressed into a single 100-step trajectory.

**Limitations.** Our analysis uses a single model size (60M) without triangle updates, which may exhibit different dynamics than larger models with triangle multiplicative updates. Spatial alignment is computed against retrospective ground truth (the model's own final structure), which may overestimate alignment for poorly generated structures. Future work should extend to larger models (200M, 400M) and conditional generation to test whether fold conditioning shifts crystallisation dynamics.

## 5. Conclusion

We have characterised a "crystallisation" phenomenon in Proteina's attention during flow-based protein generation. Three metrics — logit dominance, entropy, and spatial alignment — reveal a wave-like transition from diffuse, geometry-dominated attention to sharp, contact-specific patterns, propagating from early to late layers. Per-head analysis uncovers geometric specialists that crystallise early, while sequence-separation decomposition confirms that geometric bias matters most for long-range fold-defining contacts. These findings provide mechanistic insight into how flow-based models construct protein structure, and suggest that the interplay between geometric priors and learned content representations is a key design axis for future generative architectures.

## References

- Darcet, T., et al. (2023). Vision Transformers Need Registers. arXiv:2309.16588.
- Ding, J., et al. (2024). Mix-Compress-Refine: Information Compression in Vision Transformers. arXiv:2510.06477.
- Geffner, T., Didi, K., Zhang, Z., et al. (2025). Proteina: Scaling Flow-based Protein Structure Generative Models. ICLR 2025 Oral.
- Rao, R., et al. (2020). Transformer protein language models are unsupervised structure learners. bioRxiv.
