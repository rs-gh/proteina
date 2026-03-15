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


def fig1_trajectory(models: Dict[str, dict], output_dir: Path):
    """
    Figure 1: Core trajectory metrics (R_c, H, rho) for all models.
    Layout: 4 rows (models) x 3 columns (R_c, H, rho).
    """
    available = [m for m in MODEL_ORDER if m in models]
    n_models = len(available)

    fig, axes = plt.subplots(n_models, 3, figsize=(14, 3.5 * n_models),
                              squeeze=False)

    for row, model in enumerate(available):
        d = models[model]
        timesteps = d['timesteps']
        n_seeds = len(d.get('seeds', [0]))

        # Determine number of layers from R_mean shape
        num_layers = d['R_mean'].shape[1]
        layers = get_representative_layers(num_layers)
        colors = plt.cm.viridis(np.linspace(0, 1, len(layers)))

        # Use R_c if available, otherwise fall back to R
        use_Rc = 'Rc_mean' in d
        R_mean = d['Rc_mean'] if use_Rc else d['R_mean']
        R_std = d['Rc_std'] if use_Rc else d['R_std']
        r_label = '$R_c$' if use_Rc else '$R$'

        for l_idx, l in enumerate(layers):
            # R_c / R — mean over heads
            r_m = R_mean[:, l, :].mean(axis=-1)
            r_s = R_std[:, l, :].mean(axis=-1)
            axes[row, 0].plot(timesteps, r_m, color=colors[l_idx],
                              label=f'L{l}', linewidth=1.5)
            axes[row, 0].fill_between(timesteps, r_m - r_s, r_m + r_s,
                                       color=colors[l_idx], alpha=0.15)

            # Entropy — normalized by log(n)
            protein_length = int(d.get('protein_length', 100))
            log_n = np.log(protein_length) if protein_length > 1 else 1.0
            h_m = d['H_mean'][:, l, :].mean(axis=-1) / log_n
            h_s = d['H_std'][:, l, :].mean(axis=-1) / log_n
            axes[row, 1].plot(timesteps, h_m, color=colors[l_idx],
                              label=f'L{l}', linewidth=1.5)
            axes[row, 1].fill_between(timesteps, h_m - h_s, h_m + h_s,
                                       color=colors[l_idx], alpha=0.15)

            # Spatial alignment
            if 'rho_mean' in d:
                rho_m = d['rho_mean'][:, l, :].mean(axis=-1)
                rho_s = d['rho_std'][:, l, :].mean(axis=-1)
                axes[row, 2].plot(timesteps, rho_m, color=colors[l_idx],
                                  label=f'L{l}', linewidth=1.5)
                axes[row, 2].fill_between(timesteps, rho_m - rho_s, rho_m + rho_s,
                                           color=colors[l_idx], alpha=0.15)

        # Labels
        axes[row, 0].set_ylabel(f'{MODEL_LABELS[model]}\n{r_label}')
        axes[row, 1].set_ylabel('$\\hat{{H}}$')
        if 'rho_mean' in d:
            axes[row, 2].set_ylabel('$\\rho$')

        # Reference lines
        axes[row, 0].axhline(y=1.0, color='grey', linestyle='--', linewidth=0.8, alpha=0.5)
        axes[row, 1].axhline(y=1.0, color='grey', linestyle='--', linewidth=0.8, alpha=0.5)
        axes[row, 2].axhline(y=0.0, color='grey', linestyle='--', linewidth=0.8, alpha=0.5)

        for col in range(3):
            axes[row, col].legend(loc='best')
            axes[row, col].grid(True, alpha=0.2)

        if row == 0:
            axes[row, 0].set_title(f'Logit Dominance ({r_label})')
            axes[row, 1].set_title('Normalised Entropy ($\\hat{{H}}$)')
            axes[row, 2].set_title('Spatial Alignment ($\\rho$)')

    for col in range(3):
        axes[-1, col].set_xlabel('Timestep ($t$)')

    fig.suptitle(f'Core metrics across denoising trajectory ($n$=100, {n_seeds} seeds, shaded = $\\pm$1 std)',
                 fontsize=12, y=1.01)
    plt.tight_layout()
    path = output_dir / "fig1-trajectory-all-models.png"
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
        layers = get_representative_layers(num_layers)
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

    fig1_trajectory(models, output_dir)
    fig2_seqsep(models, output_dir)
    fig3_contact_precision(models, output_dir)
    fig5_structure_lens(output_dir)
    fig6_R_vs_Rc(models, output_dir)

    print(f"\nAll figures saved to {output_dir}")
