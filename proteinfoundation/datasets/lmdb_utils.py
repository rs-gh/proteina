"""Utilities for building LMDB datasets from raw protein structure files.

Processes raw CIF/PDB files directly into LMDB, skipping intermediate .pt
files entirely. Supports incremental builds: re-running appends new
structures without duplicating existing ones.
"""

import os
import pathlib
import pickle
from typing import List, Optional, Set, Tuple, Union

import lmdb
import torch
from loguru import logger
from tqdm import tqdm

from proteinfoundation.utils.constants import PDB_TO_OPENFOLD_INDEX_TENSOR


def _get_existing_ids(db) -> Set[str]:
    """Return the set of protein IDs already stored in an LMDB."""
    ids = set()
    with db.begin() as txn:
        cursor = txn.cursor()
        for key, value in cursor:
            try:
                graph = pickle.loads(value)
                if hasattr(graph, "id"):
                    ids.add(str(graph.id))
            except Exception:
                pass
    return ids


def _get_next_key(db) -> int:
    """Return the next sequential integer key for appending."""
    with db.begin() as txn:
        cursor = txn.cursor()
        if cursor.last():
            return int(cursor.key().decode()) + 1
    return 0


def process_raw_to_lmdb(
    raw_dir: str,
    output_path: str,
    pdb_codes: List[str],
    chains: Optional[List[str]] = None,
    file_format: str = "cif",
    apply_coord_reorder: bool = True,
    store_het: bool = False,
    store_bfactor: bool = True,
    map_size_gb: int = 50,
) -> int:
    """Process raw structure files directly into LMDB.

    Parses each CIF/PDB file, creates a PyG Data graph, applies coordinate
    reordering, and writes directly to LMDB. No intermediate .pt files.

    Supports incremental builds: re-running skips structures already in the
    LMDB (matched by protein ID).

    Args:
        raw_dir: Directory containing raw structure files.
        output_path: Path for the output .lmdb file.
        pdb_codes: List of PDB codes to process.
        chains: Optional list of chain IDs (one per pdb_code). If None,
            all chains are processed.
        file_format: Raw file format ("cif", "pdb", "mmtf", "ent").
        apply_coord_reorder: Apply PDB→OpenFold coordinate reordering.
        store_het: Whether to store heteroatoms.
        store_bfactor: Whether to store B-factors.
        map_size_gb: Maximum LMDB map size in GB.

    Returns:
        Number of NEW samples written in this call.
    """
    # Import here to avoid import at module level (heavy deps)
    from graphein_utils.graphein_utils import protein_to_pyg
    from openfold.np.residue_constants import resname_to_idx

    raw_dir = pathlib.Path(raw_dir)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    is_existing = os.path.exists(output_path)

    db = lmdb.open(
        output_path,
        map_size=map_size_gb * (1024 ** 3),
        create=True,
        subdir=False,
        readonly=False,
        lock=True,
        readahead=False,
        meminit=False,
    )

    # For incremental mode, find what's already there
    existing_ids = set()
    next_key = 0
    if is_existing:
        existing_ids = _get_existing_ids(db)
        next_key = _get_next_key(db)
        if existing_ids:
            logger.info(
                f"Existing LMDB has {len(existing_ids)} entries, "
                f"appending new ones (next key: {next_key})"
            )

    # Build list of items to process
    items: List[Tuple] = []
    for i, pdb in enumerate(pdb_codes):
        chain = chains[i] if chains is not None else None
        protein_id = f"{pdb}_{chain}" if chain else pdb

        if protein_id in existing_ids:
            continue

        items.append((pdb, chain, protein_id))

    logger.info(
        f"Processing {len(items)} new structures "
        f"({len(existing_ids)} already in LMDB, "
        f"{len(pdb_codes) - len(items) - len(existing_ids)} missing from raw)"
    )

    n_written = 0
    n_failed = 0
    fill_value_coords = 1e-5

    with db.begin(write=True) as txn:
        for pdb, chain, protein_id in tqdm(items, desc="Processing → LMDB"):
            try:
                # Find the raw file
                path = raw_dir / f"{pdb}.{file_format}"
                if not path.exists():
                    path = path.with_suffix(f".{file_format}.gz")
                if not path.exists():
                    n_failed += 1
                    continue

                # Parse structure to PyG graph
                chain_selection = chain if chain else "all"
                graph = protein_to_pyg(
                    path=str(path),
                    chain_selection=chain_selection,
                    keep_insertions=True,
                    store_het=store_het,
                    store_bfactor=store_bfactor,
                    fill_value_coords=fill_value_coords,
                )

                # Add metadata fields (same as _load_and_process_pdb)
                graph.id = protein_id
                coord_mask = graph.coords != fill_value_coords
                graph.coord_mask = coord_mask[..., 0]
                graph.residue_type = torch.tensor(
                    [resname_to_idx[r] for r in graph.residues]
                ).long()
                graph.database = "pdb"
                graph.bfactor_avg = torch.mean(graph.bfactor, dim=-1)
                graph.residue_pdb_idx = torch.tensor(
                    [int(s.split(":")[2]) for s in graph.residue_id],
                    dtype=torch.long,
                )
                graph.seq_pos = torch.arange(graph.coords.shape[0]).unsqueeze(-1)

                # Apply coordinate reordering (PDB → OpenFold convention)
                if apply_coord_reorder:
                    graph.coords = graph.coords[:, PDB_TO_OPENFOLD_INDEX_TENSOR, :]
                    graph.coord_mask = graph.coord_mask[:, PDB_TO_OPENFOLD_INDEX_TENSOR]

                # Write to LMDB
                txn.put(
                    key=str(next_key + n_written).encode(),
                    value=pickle.dumps(graph),
                )
                n_written += 1

            except Exception as e:
                logger.warning(f"Failed to process {protein_id}: {e}")
                n_failed += 1

    db.close()

    total = next_key + n_written
    logger.info(
        f"LMDB: {output_path} — "
        f"{n_written} new, {len(existing_ids)} existing, {n_failed} failed, "
        f"{total} total entries"
    )
    return n_written


def convert_pt_to_lmdb(
    processed_dir: str,
    file_names: List[str],
    output_path: str,
    apply_coord_reorder: bool = True,
    map_size_gb: int = 50,
) -> int:
    """Convert existing .pt files into LMDB (for already-processed data).

    Simpler alternative to process_raw_to_lmdb when .pt files already exist.
    Supports incremental conversion.

    Args:
        processed_dir: Directory containing {name}.pt files.
        file_names: List of file basenames (without .pt extension).
        output_path: Path for the output .lmdb file.
        apply_coord_reorder: Apply PDB→OpenFold coordinate reordering.
        map_size_gb: Maximum LMDB map size in GB.

    Returns:
        Number of NEW samples written.
    """
    processed_dir = pathlib.Path(processed_dir)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    is_existing = os.path.exists(output_path)

    db = lmdb.open(
        output_path,
        map_size=map_size_gb * (1024 ** 3),
        create=True,
        subdir=False,
        readonly=False,
        lock=True,
        readahead=False,
        meminit=False,
    )

    existing_ids = set()
    next_key = 0
    if is_existing:
        existing_ids = _get_existing_ids(db)
        next_key = _get_next_key(db)
        if existing_ids:
            logger.info(f"Existing LMDB has {len(existing_ids)} entries, appending")

    n_written = 0
    n_skipped = 0

    with db.begin(write=True) as txn:
        for name in tqdm(file_names, desc="Converting to LMDB"):
            if name in existing_ids:
                continue

            pt_path = processed_dir / f"{name}.pt"
            if not pt_path.exists():
                n_skipped += 1
                continue

            try:
                graph = torch.load(pt_path, weights_only=False)

                if apply_coord_reorder and hasattr(graph, "coords"):
                    graph.coords = graph.coords[:, PDB_TO_OPENFOLD_INDEX_TENSOR, :]
                    graph.coord_mask = graph.coord_mask[:, PDB_TO_OPENFOLD_INDEX_TENSOR]

                txn.put(
                    key=str(next_key + n_written).encode(),
                    value=pickle.dumps(graph),
                )
                n_written += 1
            except Exception as e:
                logger.warning(f"Failed to process {name}: {e}")
                n_skipped += 1

    db.close()

    total = next_key + n_written
    logger.info(
        f"LMDB: {output_path} — "
        f"{n_written} new, {len(existing_ids)} existing, {n_skipped} skipped, "
        f"{total} total"
    )
    return n_written
