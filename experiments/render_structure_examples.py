"""Render CA-trace examples for the appendix figure.

Produces a 2×3 grid (rows: baseline / full ablation; columns: n=50/100/200)
from the 60M model, seed 42. Output: report/figures/figA3-structure-examples.png
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

PDBS = {
    ("baseline", 50):      "experiments/causal/60m/run_2026-03-15_n50/artifacts/baseline/seed_42.pdb",
    ("full_ablation", 50): "experiments/causal/60m/run_2026-03-15_n50/artifacts/full_ablation/seed_42.pdb",
    ("baseline", 100):     "experiments/causal/60m/exp05a_2026-03-01_bias-ablation/artifacts/baseline/seed_42.pdb",
    ("full_ablation", 100):"experiments/causal/60m/exp05a_2026-03-01_bias-ablation/artifacts/full_ablation/seed_42.pdb",
    ("baseline", 200):     "experiments/causal/60m/run_2026-03-15_n200/artifacts/baseline/seed_42.pdb",
    ("full_ablation", 200):"experiments/causal/60m/run_2026-03-15_n200/artifacts/full_ablation/seed_42.pdb",
}

METRICS_NPZ = {
    ("baseline", 50):      "experiments/causal/60m/run_2026-03-15_n50/artifacts/baseline/metrics.npz",
    ("full_ablation", 50): "experiments/causal/60m/run_2026-03-15_n50/artifacts/full_ablation/metrics.npz",
    ("baseline", 100):     "experiments/causal/60m/exp05a_2026-03-01_bias-ablation/artifacts/baseline/metrics.npz",
    ("full_ablation", 100):"experiments/causal/60m/exp05a_2026-03-01_bias-ablation/artifacts/full_ablation/metrics.npz",
    ("baseline", 200):     "experiments/causal/60m/run_2026-03-15_n200/artifacts/baseline/metrics.npz",
    ("full_ablation", 200):"experiments/causal/60m/run_2026-03-15_n200/artifacts/full_ablation/metrics.npz",
}


def load_clash_stats(cond, n):
    """Return (mean, std) of n_clashes across 3 seeds from metrics.npz."""
    path = os.path.join(REPO_ROOT, METRICS_NPZ[(cond, n)])
    d = np.load(path)
    clashes = d["n_clashes"].astype(float)
    return clashes.mean(), clashes.std()


def read_ca_coords(pdb_path):
    coords = []
    with open(pdb_path) as f:
        for line in f:
            if line.startswith("ATOM") and line[12:16].strip() == "CA":
                x, y, z = float(line[30:38]), float(line[38:46]), float(line[46:54])
                coords.append([x, y, z])
    return np.array(coords)


def compute_rg(coords):
    centroid = coords.mean(axis=0)
    return np.sqrt(((coords - centroid) ** 2).sum(axis=1).mean()) / 10.0  # Å → nm


def plot_structure(ax, coords, lim, color):
    c = coords - coords.mean(axis=0)  # centre
    ax.plot(c[:, 0], c[:, 1], c[:, 2], color=color, linewidth=0.8, alpha=0.7)
    ax.scatter(c[:, 0], c[:, 1], c[:, 2], color=color, s=8, alpha=0.6, linewidths=0)
    for spine in ["top", "bottom", "left", "right"]:
        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_zlim(-lim, lim)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.grid(False)


def main():
    lengths = [50, 100, 200]
    conditions = ["baseline", "full_ablation"]
    row_labels = ["Baseline", r"Full ablation ($B=0$)"]
    colors = {"baseline": "#2171b5", "full_ablation": "#cb181d"}

    fig = plt.figure(figsize=(12, 8))
    fig.patch.set_facecolor("white")

    # Preload all coords and compute per-column axis limits from baseline
    coords_cache = {}
    col_lims = {}
    for n in lengths:
        pdb = os.path.join(REPO_ROOT, PDBS[("baseline", n)])
        c = read_ca_coords(pdb)
        coords_cache[("baseline", n)] = c
        centred = c - c.mean(axis=0)
        col_lims[n] = np.abs(centred).max() * 1.15

        pdb2 = os.path.join(REPO_ROOT, PDBS[("full_ablation", n)])
        coords_cache[("full_ablation", n)] = read_ca_coords(pdb2)

    for row, cond in enumerate(conditions):
        for col, n in enumerate(lengths):
            ax = fig.add_subplot(2, 3, row * 3 + col + 1, projection="3d")
            c = coords_cache[(cond, n)]
            rg = compute_rg(c)
            clash_mean, clash_std = load_clash_stats(cond, n)
            lim = col_lims[n]

            plot_structure(ax, c, lim, colors[cond])

            if clash_mean == 0:
                clash_line = "clashes = 0"
            else:
                clash_line = f"clashes = {clash_mean:.0f} \u00b1 {clash_std:.0f}"
            ax.set_title(
                f"$R_g = {rg:.2f}$ nm\n{clash_line}",
                fontsize=9, pad=2,
            )

    # Column headers (n= labels)
    for col, n in enumerate(lengths):
        fig.text(
            (col + 0.5) / 3, 0.97,
            f"$n = {n}$",
            ha="center", va="top", fontsize=11, fontweight="bold",
        )

    # Row labels
    for row, label in enumerate(row_labels):
        fig.text(
            0.01, 1 - (row + 0.5) / 2,
            label,
            ha="left", va="center", fontsize=10, fontweight="bold",
            rotation=90,
        )

    plt.tight_layout(rect=[0.04, 0.0, 1.0, 0.96])

    out = os.path.join(REPO_ROOT, "report", "figures", "figA3-structure-examples.png")
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
