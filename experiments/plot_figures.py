#!/usr/bin/env python
"""
Unified figure generation for the report.

Generates consistent cross-model figures from aggregated experiment data.
All figures show all available models side-by-side.

Usage:
    python experiments/plot_figures.py --output-dir report/figures/

    # Use specific data directories
    python experiments/plot_figures.py \
        --data-60m experiments/descriptive/60m/run_2026-03-15_5seed/artifacts \
        --data-200m-notri experiments/descriptive/200m_notri/run_2026-03-15_5seed/artifacts \
        --data-200m-tri experiments/descriptive/200m_tri/run_2026-03-15_5seed/artifacts \
        --data-400m-tri experiments/descriptive/400m_tri/run_2026-03-15_5seed/artifacts
"""

import argparse
from pathlib import Path
from typing import Dict, Optional

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

# Consistent style
plt.rcParams.update({
    'font.size': 10,
    'axes.titlesize': 11,
    'axes.labelsize': 10,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 8,
    'figure.dpi': 150,
})

MODEL_ORDER = ['60m', '200m_notri', '200m_tri', '400m_tri']
MODEL_LABELS = {
    '60m': '60M (no tri)',
    '200m_notri': '200M (no tri)',
    '200m_tri': '200M (tri)',
    '400m_tri': '400M (tri)',
}
MODEL_COLORS = {
    '60m': '#1f77b4',
    '200m_notri': '#ff7f0e',
    '200m_tri': '#2ca02c',
    '400m_tri': '#d62728',
}


def load_model_data(path: Path) -> Optional[dict]:
    """Load aggregated.npz from a model's artifact directory."""
    agg_path = path / "aggregated.npz"
    if not agg_path.exists():
        print(f"Warning: {agg_path} not found, skipping")
        return None
    data = dict(np.load(str(agg_path), allow_pickle=True))
    return data


def get_representative_layers(num_layers: int):
    """Return [first, middle, last] layer indices."""
    return [0, num_layers // 2, num_layers - 1]


# Peak R_c layers per model (determined empirically from t=1 data)
PEAK_RC_LAYERS = {
    '60m': 0,         # R_c=34.9 at t=1
    '200m_notri': 0,  # R_c=4.3 at t=1 (L2 close at 3.9)
    '200m_tri': 9,    # R_c=27.0 at t=1 (L7=10.5 also high)
    '400m_tri': 2,    # R_c=29.4 at t=1
}


def get_key_layers(model: str, num_layers: int):
    """Return [peak_Rc_layer, first_or_alternate, last] for line plots."""
    peak = PEAK_RC_LAYERS.get(model, 0)
    last = num_layers - 1
    # Pick a contrasting layer: if peak is L0, show middle; if peak is deep, show L0
    if peak == 0:
        alt = num_layers // 2
    else:
        alt = 0
    return sorted(set([peak, alt, last]))


def fig1_Rc_lines(models: Dict[str, dict], output_dir: Path):
    """
    Figure 1a: R_c line plots for model-appropriate layers.
    Shows peak R_c layer + contrasting layers per model.
    """
    available = [m for m in MODEL_ORDER if m in models]
    n_models = len(available)

    fig, axes = plt.subplots(1, n_models, figsize=(4 * n_models, 3.5), squeeze=False)

    for col, model in enumerate(available):
        d = models[model]
        timesteps = d['timesteps']
        num_layers = d['R_mean'].shape[1]
        layers = get_key_layers(model, num_layers)
        colors = plt.cm.viridis(np.linspace(0, 1, len(layers)))

        use_Rc = 'Rc_mean' in d
        R_mean = d['Rc_mean'] if use_Rc else d['R_mean']
        R_std = d['Rc_std'] if use_Rc else d['R_std']

        for l_idx, l in enumerate(layers):
            r_m = R_mean[:, l, :].mean(axis=-1)
            r_s = R_std[:, l, :].mean(axis=-1)
            peak = PEAK_RC_LAYERS.get(model, 0)
            lw = 2.0 if l == peak else 1.2
            axes[0, col].plot(timesteps, r_m, color=colors[l_idx],
                              label=f'L{l}' + (' *' if l == peak else ''),
                              linewidth=lw)
            axes[0, col].fill_between(timesteps, r_m - r_s, r_m + r_s,
                                       color=colors[l_idx], alpha=0.15)

        axes[0, col].axhline(y=1.0, color='grey', linestyle='--', linewidth=0.8, alpha=0.5)
        axes[0, col].set_title(MODEL_LABELS[model])
        axes[0, col].set_xlabel('Timestep ($t$)')
        axes[0, col].legend(loc='best', fontsize=7)
        axes[0, col].grid(True, alpha=0.2)

    axes[0, 0].set_ylabel('$R_c$')
    n_seeds = len(models[available[0]].get('seeds', [0]))
    fig.suptitle(f'Row-centred logit dominance $R_c$ ($n$=100, {n_seeds} seeds, * = peak layer)',
                 fontsize=11, y=1.03)
    plt.tight_layout()
    path = output_dir / "fig1a-Rc-lines.png"
    plt.savefig(path, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved {path}")


def fig1_Rc_heatmaps(models: Dict[str, dict], output_dir: Path):
    """
    Figure 1b: R_c heatmaps (all layers x timesteps) for each model.
    Shows which layer is the 'geometric interpreter' at each point in generation.
    """
    available = [m for m in MODEL_ORDER if m in models]
    n_models = len(available)

    fig, axes = plt.subplots(1, n_models, figsize=(4 * n_models, 4), squeeze=False)

    for col, model in enumerate(available):
        d = models[model]
        timesteps = d['timesteps']

        use_Rc = 'Rc_mean' in d
        R_mean = d['Rc_mean'] if use_Rc else d['R_mean']

        # Mean over heads: [T, L]
        rc_map = R_mean.mean(axis=-1)

        im = axes[0, col].imshow(
            rc_map.T, aspect='auto', origin='lower',
            extent=[timesteps[0], timesteps[-1], -0.5, rc_map.shape[1] - 0.5],
            cmap='magma', vmin=0, vmax=min(rc_map.max(), 40))
        axes[0, col].set_title(MODEL_LABELS[model])
        axes[0, col].set_xlabel('Timestep ($t$)')
        plt.colorbar(im, ax=axes[0, col], shrink=0.8)

    axes[0, 0].set_ylabel('Layer')
    n_seeds = len(models[available[0]].get('seeds', [0]))
    fig.suptitle(f'$R_c$ heatmap — all layers ($n$=100, {n_seeds} seeds, mean over heads)',
                 fontsize=11, y=1.03)
    plt.tight_layout()
    path = output_dir / "fig1b-Rc-heatmaps.png"
    plt.savefig(path, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved {path}")


def fig_supplementary_H_rho(models: Dict[str, dict], output_dir: Path):
    """
    Supplementary figure: Entropy and spatial alignment (moved from main Fig 1).
    """
    available = [m for m in MODEL_ORDER if m in models]
    n_models = len(available)

    fig, axes = plt.subplots(n_models, 2, figsize=(10, 3 * n_models), squeeze=False)

    for row, model in enumerate(available):
        d = models[model]
        timesteps = d['timesteps']
        num_layers = d['R_mean'].shape[1]
        layers = get_key_layers(model, num_layers)
        colors = plt.cm.viridis(np.linspace(0, 1, len(layers)))

        protein_length = int(d.get('protein_length', 100))
        log_n = np.log(protein_length) if protein_length > 1 else 1.0

        for l_idx, l in enumerate(layers):
            h_m = d['H_mean'][:, l, :].mean(axis=-1) / log_n
            h_s = d['H_std'][:, l, :].mean(axis=-1) / log_n
            axes[row, 0].plot(timesteps, h_m, color=colors[l_idx],
                              label=f'L{l}', linewidth=1.5)
            axes[row, 0].fill_between(timesteps, h_m - h_s, h_m + h_s,
                                       color=colors[l_idx], alpha=0.15)

            if 'rho_mean' in d:
                rho_m = d['rho_mean'][:, l, :].mean(axis=-1)
                rho_s = d['rho_std'][:, l, :].mean(axis=-1)
                axes[row, 1].plot(timesteps, rho_m, color=colors[l_idx],
                                  label=f'L{l}', linewidth=1.5)
                axes[row, 1].fill_between(timesteps, rho_m - rho_s, rho_m + rho_s,
                                           color=colors[l_idx], alpha=0.15)

        axes[row, 0].axhline(y=1.0, color='grey', linestyle='--', linewidth=0.8, alpha=0.5)
        axes[row, 1].axhline(y=0.0, color='grey', linestyle='--', linewidth=0.8, alpha=0.5)
        axes[row, 0].set_ylabel(f'{MODEL_LABELS[model]}\n$\\hat{{H}}$')
        axes[row, 1].set_ylabel('$\\rho$')
        for c in range(2):
            axes[row, c].legend(loc='best', fontsize=7)
            axes[row, c].grid(True, alpha=0.2)
        if row == 0:
            axes[row, 0].set_title('Normalised Entropy ($\\hat{H}$)')
            axes[row, 1].set_title('Spatial Alignment ($\\rho$)')

    for c in range(2):
        axes[-1, c].set_xlabel('Timestep ($t$)')

    n_seeds = len(models[available[0]].get('seeds', [0]))
    fig.suptitle(f'Supplementary: Entropy and spatial alignment ($n$=100, {n_seeds} seeds)',
                 fontsize=11, y=1.01)
    plt.tight_layout()
    path = output_dir / "fig-supp-H-rho.png"
    plt.savefig(path, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved {path}")


def fig2_seqsep(models: Dict[str, dict], output_dir: Path):
    """
    Figure 2: Sequence separation decomposition of R_c in Layer 0.
    Layout: 1 row x N models.
    """
    available = [m for m in MODEL_ORDER if m in models
                 and ('seqsep_Rc_mean' in models[m] or 'seqsep_R_mean' in models[m])]
    if not available:
        print("No seqsep data available, skipping fig2")
        return

    n_models = len(available)
    fig, axes = plt.subplots(1, n_models, figsize=(4.5 * n_models, 4), squeeze=False)

    bin_labels = ['local (1-6)', 'medium (7-23)', 'long ($\\geq$24)']
    bin_colors = ['#2ca02c', '#ff7f0e', '#d62728']

    for col, model in enumerate(available):
        d = models[model]
        timesteps = d['timesteps']

        use_Rc = 'seqsep_Rc_mean' in d
        R_key = 'seqsep_Rc_mean' if use_Rc else 'seqsep_R_mean'
        S_key = 'seqsep_Rc_std' if use_Rc else 'seqsep_R_std'
        r_label = '$R_c$' if use_Rc else '$R$'

        for bin_idx, (label, color) in enumerate(zip(bin_labels, bin_colors)):
            # Shape: [bins, T, L, H] — take Layer 0, mean over heads
            r_m = d[R_key][bin_idx, :, 0, :].mean(axis=-1)
            r_s = d[S_key][bin_idx, :, 0, :].mean(axis=-1)
            axes[0, col].plot(timesteps, r_m, color=color, label=label, linewidth=1.5)
            axes[0, col].fill_between(timesteps, r_m - r_s, r_m + r_s,
                                       color=color, alpha=0.15)

        axes[0, col].set_title(f'{MODEL_LABELS[model]}')
        axes[0, col].set_xlabel('Timestep ($t$)')
        axes[0, col].legend(fontsize=7)
        axes[0, col].grid(True, alpha=0.2)

    axes[0, 0].set_ylabel(f'{r_label} (Layer 0)')
    fig.suptitle('Logit dominance by sequence separation (Layer 0)', fontsize=12, y=1.02)
    plt.tight_layout()
    path = output_dir / "fig2-seqsep-all-models.png"
    plt.savefig(path, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved {path}")


def fig3_contact_precision(models: Dict[str, dict], output_dir: Path):
    """
    Figure 3: Contact precision (full, B-only, C-only) for all models.
    Layout: N models x 3 columns.
    """
    available = [m for m in MODEL_ORDER if m in models and 'prec_full_mean' in models[m]]
    if not available:
        print("No contact precision data available, skipping fig3")
        return

    n_models = len(available)
    fig, axes = plt.subplots(n_models, 3, figsize=(14, 3.5 * n_models), squeeze=False)

    for row, model in enumerate(available):
        d = models[model]
        timesteps = d['timesteps']
        num_layers = d['prec_full_mean'].shape[1]
        layers = get_key_layers(model, num_layers)
        colors = plt.cm.viridis(np.linspace(0, 1, len(layers)))

        panels = [
            ('prec_full', 'Full ($C+B$)'),
            ('prec_b', '$B$-only'),
            ('prec_c', '$C$-only'),
        ]

        for col, (key_prefix, title) in enumerate(panels):
            mean = d[f'{key_prefix}_mean']
            std = d[f'{key_prefix}_std']
            for l_idx, l in enumerate(layers):
                m = mean[:, l, :].mean(axis=-1)
                s = std[:, l, :].mean(axis=-1)
                axes[row, col].plot(timesteps, m, color=colors[l_idx],
                                     label=f'L{l}', linewidth=1.5)
                axes[row, col].fill_between(timesteps, m - s, m + s,
                                             color=colors[l_idx], alpha=0.15)
            if row == 0:
                axes[row, col].set_title(title)
            axes[row, col].grid(True, alpha=0.2)
            axes[row, col].legend(fontsize=7)
            axes[row, col].set_ylim(bottom=0)

            # Random baseline
            rp = d.get('random_precision_mean', None)
            if rp is not None:
                rp_val = float(rp)
                axes[row, col].axhline(y=rp_val, color='grey', linestyle='--',
                                        linewidth=0.8, alpha=0.6)
                if col == 0:
                    axes[row, col].text(0.02, rp_val + 0.005, f'random={rp_val:.3f}',
                                         fontsize=6, color='grey', alpha=0.8)

        axes[row, 0].set_ylabel(f'{MODEL_LABELS[model]}\nPrecision@$L/5$')

    for col in range(3):
        axes[-1, col].set_xlabel('Timestep ($t$)')

    n_seeds = len(models[available[0]].get('seeds', [0]))
    fig.suptitle(f'Contact Precision@$L/5$ ($n$=100, {n_seeds} seeds)',
                 fontsize=12, y=1.01)
    plt.tight_layout()
    path = output_dir / "fig3-contact-precision-all-models.png"
    plt.savefig(path, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved {path}")


def fig6_R_vs_Rc(models: Dict[str, dict], output_dir: Path):
    """
    Figure 6: R vs R_c comparison for a representative model (60M).
    Shows why R_c is the better metric.
    """
    # Use 60M if available (most dramatic difference)
    model = '60m' if '60m' in models else next(iter(models))
    d = models[model]

    if 'Rc_mean' not in d:
        print(f"No R_c data for {model}, skipping fig6")
        return

    timesteps = d['timesteps']
    num_layers = d['R_mean'].shape[1]
    layers = get_representative_layers(num_layers)

    fig, axes = plt.subplots(1, len(layers), figsize=(5 * len(layers), 4))

    for col, l in enumerate(layers):
        r_m = d['R_mean'][:, l, :].mean(axis=-1)
        r_s = d['R_std'][:, l, :].mean(axis=-1)
        rc_m = d['Rc_mean'][:, l, :].mean(axis=-1)
        rc_s = d['Rc_std'][:, l, :].mean(axis=-1)

        axes[col].plot(timesteps, r_m, 'b-', label='$R$ (raw)', linewidth=1.5)
        axes[col].fill_between(timesteps, r_m - r_s, r_m + r_s, color='blue', alpha=0.1)
        axes[col].plot(timesteps, rc_m, 'r--', label='$R_c$ (centered)', linewidth=1.5)
        axes[col].fill_between(timesteps, rc_m - rc_s, rc_m + rc_s, color='red', alpha=0.1)

        axes[col].set_title(f'Layer {l}')
        axes[col].set_xlabel('Timestep ($t$)')
        axes[col].legend()
        axes[col].grid(True, alpha=0.2)

    axes[0].set_ylabel('Logit dominance')
    n_seeds = len(d.get('seeds', [0]))
    fig.suptitle(f'Raw vs Row-Centered Logit Dominance — {MODEL_LABELS[model]} ($n$=100, {n_seeds} seeds)',
                 fontsize=11, y=1.02)
    plt.tight_layout()
    path = output_dir / "fig6-raw-vs-centered.png"
    plt.savefig(path, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved {path}")


def fig4_temporal_sweep(output_dir: Path):
    """
    Figure 4: Fine-grained temporal ablation sweep.
    Shows Rg vs split point for early_B and late_B conditions.
    """
    sweep_data = {}
    for model in MODEL_ORDER:
        path = Path(f"experiments/causal/{model}/temporal_sweep/artifacts/sweep_summary.npz")
        if path.exists():
            sweep_data[model] = dict(np.load(str(path)))

    if not sweep_data:
        print("No temporal sweep data found, skipping fig4")
        return

    available = [m for m in MODEL_ORDER if m in sweep_data]
    n_models = len(available)

    fig, axes = plt.subplots(1, n_models, figsize=(4.5 * n_models, 3.5), squeeze=False)

    for col, model in enumerate(available):
        d = sweep_data[model]
        splits = d['split_points']
        early_rg = d['early_rg']
        late_rg = d['late_rg']
        bl_rg = d['baseline_rg']

        ax = axes[0, col]
        ax.plot(splits, late_rg, 'o-', color='#2ca02c', label='Late-only $B$',
                linewidth=2, markersize=5)
        ax.plot(splits, early_rg, 's-', color='#d62728', label='Early-only $B$',
                linewidth=2, markersize=5)
        if len(bl_rg) > 0:
            ax.axhline(y=bl_rg[0], color='black', linestyle='-', linewidth=1,
                       alpha=0.5, label=f'Baseline ({bl_rg[0]:.2f})')
        ax.axhline(y=0.12, color='grey', linestyle=':', linewidth=0.8, alpha=0.5)

        ax.set_title(MODEL_LABELS[model])
        ax.set_xlabel('Split point $t_s$')
        ax.set_ylim(-0.05, 1.7)
        ax.legend(fontsize=7, loc='center right')
        ax.grid(True, alpha=0.2)

        # Annotate
        ax.annotate('collapsed\n(thousands of clashes)', xy=(0.5, 0.12),
                    fontsize=6, color='#d62728', ha='center', va='bottom')

    axes[0, 0].set_ylabel('$R_g$ (nm)')
    fig.suptitle('Temporal ablation sweep: $R_g$ vs split point ($n$=100, 3 seeds)',
                 fontsize=11, y=1.03)
    plt.tight_layout()
    path = output_dir / "fig4-temporal-sweep.png"
    plt.savefig(path, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved {path}")


def fig5_structure_lens(output_dir: Path):
    """
    Figure 5: Structure lens heatmaps for all models.
    Layout: N models x 3 columns (RMSD, Rg, Jaccard).
    """
    # Find structure lens data
    sl_data = {}
    for model in MODEL_ORDER:
        # Check new runs first, then old experiments
        for pattern in [
            f"experiments/structure_lens/{model}/run_*/artifacts/aggregate.npz",
            f"experiments/structure_lens/{model}/*/artifacts/aggregate.npz",
        ]:
            from glob import glob
            matches = sorted(glob(str(Path(pattern))))
            if matches:
                sl_data[model] = dict(np.load(matches[-1], allow_pickle=True))
                break

    if not sl_data:
        print("No structure lens data found, skipping fig5")
        return

    available = [m for m in MODEL_ORDER if m in sl_data]
    n_models = len(available)

    fig, axes = plt.subplots(n_models, 3, figsize=(16, 4 * n_models), squeeze=False)

    for row, model in enumerate(available):
        d = sl_data[model]
        timesteps = d['timesteps']
        T, L = d['rmsd_mean'].shape

        for col, (data, title, cmap) in enumerate([
            (d['rmsd_mean'], 'RMSD to Final Layer (nm)', 'viridis_r'),
            (d['rg_mean'], 'Radius of Gyration (nm)', 'coolwarm'),
            (d['contact_sim_mean'], 'Contact Similarity (Jaccard)', 'viridis'),
        ]):
            im = axes[row, col].imshow(
                data.T, aspect='auto', origin='lower', cmap=cmap,
                extent=[timesteps[0], timesteps[-1], -0.5, L - 0.5])
            axes[row, col].set_xlabel('Timestep ($t$)')
            plt.colorbar(im, ax=axes[row, col], shrink=0.8)
            if row == 0:
                axes[row, col].set_title(title)

        axes[row, 0].set_ylabel(f'{MODEL_LABELS[model]}\nLayer')

    fig.suptitle('Structure Lens — Intermediate Layer Representations ($n$=100, 3 seeds, mean)',
                 fontsize=12, y=1.01)
    plt.tight_layout()
    path = output_dir / "fig5-structure-lens-all-models.png"
    plt.savefig(path, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved {path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output-dir", type=str, default="report/figures",
                        help="Where to save figures")
    parser.add_argument("--data-60m", type=str, default=None)
    parser.add_argument("--data-200m-notri", type=str, default=None)
    parser.add_argument("--data-200m-tri", type=str, default=None)
    parser.add_argument("--data-400m-tri", type=str, default=None)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Auto-discover data directories if not specified
    data_dirs = {
        '60m': args.data_60m,
        '200m_notri': args.data_200m_notri,
        '200m_tri': args.data_200m_tri,
        '400m_tri': args.data_400m_tri,
    }

    # If not specified, look for most recent run
    for model, path in data_dirs.items():
        if path is None:
            base = Path(f"experiments/descriptive/{model}")
            if base.exists():
                runs = sorted([d for d in base.iterdir() if d.is_dir() and d.name.startswith('run_')],
                              reverse=True)
                if runs:
                    data_dirs[model] = str(runs[0] / "artifacts")

    # Load all available models
    models = {}
    for model, path in data_dirs.items():
        if path is not None:
            data = load_model_data(Path(path))
            if data is not None:
                models[model] = data
                print(f"Loaded {model} from {path}")

    if not models:
        print("No model data found. Run experiments first.")
        exit(1)

    print(f"\nGenerating figures for {len(models)} models: {list(models.keys())}")

    fig1_Rc_lines(models, output_dir)
    fig1_Rc_heatmaps(models, output_dir)
    fig2_seqsep(models, output_dir)
    fig3_contact_precision(models, output_dir)
    fig4_temporal_sweep(output_dir)
    fig5_structure_lens(output_dir)
    fig6_R_vs_Rc(models, output_dir)
    fig_supplementary_H_rho(models, output_dir)

    print(f"\nAll figures saved to {output_dir}")
