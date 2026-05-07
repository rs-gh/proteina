"""TM-score computation utilities for protein structure comparison.

Uses biotite.structure.tm_score for the core computation and
biotite.structure.superimpose for optimal structural alignment.
"""

from typing import List

import numpy as np
import biotite.structure as struc


def _ca_coords_from_atom37(coords_atom37: np.ndarray) -> np.ndarray:
    """Extract CA coordinates from atom37 representation.

    Args:
        coords_atom37: [n_residues, 37, 3] atom37 coordinate array.

    Returns:
        [n_residues, 3] CA coordinates (atom37 index 1).
    """
    return coords_atom37[:, 1, :]


def _make_ca_atom_array(ca_coords: np.ndarray) -> struc.AtomArray:
    """Build a biotite AtomArray from CA coordinates.

    Args:
        ca_coords: [n_residues, 3] CA coordinate array.

    Returns:
        AtomArray with CA atoms labeled as ALA residues.
    """
    n = len(ca_coords)
    atoms = struc.AtomArray(n)
    atoms.coord = ca_coords.astype(np.float32)
    atoms.atom_name = np.array(["CA"] * n)
    atoms.res_name = np.array(["ALA"] * n)
    atoms.res_id = np.arange(1, n + 1)
    atoms.chain_id = np.array(["A"] * n)
    atoms.element = np.array(["C"] * n)
    return atoms


def compute_tm_score(
    coords_1_atom37: np.ndarray,
    coords_2_atom37: np.ndarray,
    reference_length: str = "shorter",
) -> float:
    """Compute TM-score between two protein structures.

    Extracts CA atoms, superimposes structures via Kabsch alignment,
    then computes TM-score.

    Args:
        coords_1_atom37: [n1, 37, 3] atom37 coords of first structure.
        coords_2_atom37: [n2, 37, 3] atom37 coords of second structure.
        reference_length: Normalization length for TM-score.
            "shorter", "longer", or an integer.

    Returns:
        TM-score (float in [0, 1]). 1.0 = identical structures.
    """
    ca_1 = _ca_coords_from_atom37(coords_1_atom37)
    ca_2 = _ca_coords_from_atom37(coords_2_atom37)

    n1, n2 = len(ca_1), len(ca_2)

    if n1 == n2:
        # Same length: direct superimpose and score
        arr_1 = _make_ca_atom_array(ca_1)
        arr_2 = _make_ca_atom_array(ca_2)
        arr_2_sup, _ = struc.superimpose(arr_1, arr_2)
        indices = np.arange(n1)
        return float(struc.tm_score(arr_1, arr_2_sup, indices, indices, reference_length))
    else:
        # Different lengths: truncate to common length for superimposition,
        # then score on the common subset
        n_common = min(n1, n2)
        arr_1_sub = _make_ca_atom_array(ca_1[:n_common])
        arr_2_sub = _make_ca_atom_array(ca_2[:n_common])
        arr_2_sup, _ = struc.superimpose(arr_1_sub, arr_2_sub)
        indices = np.arange(n_common)
        return float(struc.tm_score(arr_1_sub, arr_2_sup, indices, indices, reference_length))


def compute_pairwise_tm_matrix(
    coords_list: List[np.ndarray],
) -> np.ndarray:
    """Compute pairwise TM-score matrix for a set of structures.

    Args:
        coords_list: List of [n_i, 37, 3] atom37 coordinate arrays.

    Returns:
        [N, N] symmetric matrix of TM-scores. Diagonal is 1.0.
    """
    n = len(coords_list)
    matrix = np.eye(n)

    for i in range(n):
        for j in range(i + 1, n):
            score = compute_tm_score(coords_list[i], coords_list[j])
            matrix[i, j] = score
            matrix[j, i] = score

    return matrix


def compute_diversity(
    coords_list: List[np.ndarray],
    tm_threshold: float = 0.5,
) -> dict:
    """Compute structural diversity from intra-set pairwise TM-scores.

    Returns two complementary diversity statistics off the same N x N matrix:

      n_clusters         (higher = more diverse)  number of TM>=threshold clusters
      mean_pairwise_tm   (lower  = more diverse)  mean TM over off-diagonal pairs

    The cluster metric mirrors Yim et al. 2023b / Foldseek `easy-cluster
    --tmscore-threshold 0.5`; the pairwise-TM mean mirrors Bose et al. 2024.
    Proteina Tab 1 reports both side by side.

    NOTE on Foldseek divergence: the Proteina paper uses Foldseek's 3Di
    structural alignment + TM-align refinement. We use biotite's Kabsch
    superimpose with sequence-index correspondence on Cα-only coords, and a
    greedy single-pass clusterer that adds a new cluster when the incoming
    structure has TM<threshold to every existing cluster *center* (not every
    cluster *member*). For same-length intra-set comparisons the differences
    are small but absolute numbers are not directly comparable to paper
    Tab 1. Switching to Foldseek would require shelling out to its CLI.

    Args:
        coords_list: List of [n_i, 37, 3] atom37 coordinate arrays.
        tm_threshold: TM-score threshold for cluster membership.

    Returns:
        dict with:
            n_clusters (int): cluster count (>=1 when input non-empty; 0 when empty).
            mean_pairwise_tm (float): mean TM over the upper triangle (i<j).
                NaN if fewer than 2 structures.
    """
    n = len(coords_list)
    if n == 0:
        return {"n_clusters": 0, "mean_pairwise_tm": float("nan")}

    tm_matrix = compute_pairwise_tm_matrix(coords_list)

    cluster_centers = [0]
    for i in range(1, n):
        is_novel = True
        for center in cluster_centers:
            if tm_matrix[i, center] >= tm_threshold:
                is_novel = False
                break
        if is_novel:
            cluster_centers.append(i)

    if n >= 2:
        upper = tm_matrix[np.triu_indices(n, k=1)]
        mean_pairwise_tm = float(upper.mean())
    else:
        mean_pairwise_tm = float("nan")

    return {
        "n_clusters": len(cluster_centers),
        "mean_pairwise_tm": mean_pairwise_tm,
    }
