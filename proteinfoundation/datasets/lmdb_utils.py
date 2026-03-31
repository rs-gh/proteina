"""Utilities for building LMDB datasets from raw protein structure files.

Processes raw CIF/PDB files directly into LMDB, skipping intermediate .pt
files entirely. Supports incremental builds: re-running appends new
structures without duplicating existing ones.

Parallelized: CIF parsing (the bottleneck) runs across multiple processes,
while LMDB writes are serialized in the main process.
"""

import multiprocessing
import os
import pathlib
import pickle
import signal
import sys
from typing import List, Optional, Set, Tuple, Union

import lmdb
import torch
from loguru import logger
from tqdm import tqdm

from proteinfoundation.utils.constants import PDB_TO_OPENFOLD_INDEX_TENSOR


def _parse_one_structure(args):
    """Parse a single CIF/PDB file into a pickled PyG Data graph.

    Runs in a worker process. Returns (protein_id, pickled_bytes) on success,
    or (protein_id, None) on failure.
    """
    pdb, chain, protein_id, raw_dir, file_format, store_het, store_bfactor, apply_coord_reorder = args

    try:
        from graphein_utils.graphein_utils import protein_to_pyg
        from openfold.np.residue_constants import resname_to_idx

        raw_dir = pathlib.Path(raw_dir)
        path = raw_dir / f"{pdb}.{file_format}"
        if not path.exists():
            path = path.with_suffix(f".{file_format}.gz")
        if not path.exists():
            return (protein_id, None)

        fill_value_coords = 1e-5
        chain_selection = chain if chain else "all"
        graph = protein_to_pyg(
            path=str(path),
            chain_selection=chain_selection,
            keep_insertions=True,
            store_het=store_het,
            store_bfactor=store_bfactor,
            fill_value_coords=fill_value_coords,
        )

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

        if apply_coord_reorder:
            graph.coords = graph.coords[:, PDB_TO_OPENFOLD_INDEX_TENSOR, :]
            graph.coord_mask = graph.coord_mask[:, PDB_TO_OPENFOLD_INDEX_TENSOR]

        return (protein_id, pickle.dumps(graph))

    except Exception as e:
        logger.warning(f"Failed to parse {pdb} chain={chain}: {e}")
        return (protein_id, None)


_IDS_META_KEY = b"__ids__"


def _get_existing_ids(db) -> Set[str]:
    """Return the set of protein IDs already stored in an LMDB.

    Fast path: reads a pickled set from the ``__ids__`` metadata key.
    Slow fallback: scans and unpickles every entry (for legacy LMDBs).
    """
    with db.begin() as txn:
        meta = txn.get(_IDS_META_KEY)
        if meta is not None:
            ids = pickle.loads(meta)
            logger.info(f"Loaded {len(ids)} existing IDs from metadata key")
            return ids

    # Slow fallback for LMDBs created before metadata key was added
    logger.info("No metadata key found, scanning all entries (one-time migration)...")
    ids = set()
    with db.begin() as txn:
        cursor = txn.cursor()
        for key, value in cursor:
            if key == _IDS_META_KEY:
                continue
            try:
                graph = pickle.loads(value)
                if hasattr(graph, "id"):
                    ids.add(str(graph.id))
            except Exception:
                pass

    # Write the metadata key so future resumes are fast
    with db.begin(write=True) as txn:
        txn.put(_IDS_META_KEY, pickle.dumps(ids))
    logger.info(f"Migrated {len(ids)} IDs to metadata key")
    return ids


def _get_next_key(db) -> int:
    """Return the next sequential integer key for appending."""
    with db.begin() as txn:
        cursor = txn.cursor()
        if cursor.last():
            key = cursor.key()
            # Skip non-integer metadata keys
            if key == _IDS_META_KEY:
                if cursor.prev():
                    return int(cursor.key().decode()) + 1
                return 0
            return int(key.decode()) + 1
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
    num_workers: int = 0,
) -> int:
    """Process raw structure files directly into LMDB.

    Parses each CIF/PDB file, creates a PyG Data graph, applies coordinate
    reordering, and writes directly to LMDB. No intermediate .pt files.

    When num_workers > 0, CIF parsing is parallelized across processes
    (the bottleneck). LMDB writes remain serialized in the main process.

    Supports incremental builds: re-running skips structures already in the
    LMDB (matched by protein ID).

    Args:
        raw_dir: Directory containing raw structure files.
        output_path: Path for the output .lmdb file.
        pdb_codes: List of PDB codes to process.
        chains: Optional list of chain IDs (one per pdb_code). If None,
            all chains are processed.
        file_format: Raw file format ("cif", "pdb", "mmtf", "ent").
        apply_coord_reorder: Apply PDB->OpenFold coordinate reordering.
        store_het: Whether to store heteroatoms.
        store_bfactor: Whether to store B-factors.
        map_size_gb: Maximum LMDB map size in GB.
        num_workers: Number of parallel workers for CIF parsing (0 = single-threaded).

    Returns:
        Number of NEW samples written in this call.
    """
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
    items = []
    for i, pdb in enumerate(pdb_codes):
        chain = chains[i] if chains is not None else None
        protein_id = f"{pdb}_{chain}" if chain else pdb
        if protein_id not in existing_ids:
            items.append((pdb, chain, protein_id))

    logger.info(
        f"Processing {len(items)} new structures "
        f"({len(existing_ids)} already in LMDB) "
        f"with {num_workers if num_workers > 0 else 1} workers"
    )

    n_written = 0
    n_failed = 0

    # Build worker args
    worker_args = [
        (pdb, chain, protein_id, str(raw_dir), file_format,
         store_het, store_bfactor, apply_coord_reorder)
        for pdb, chain, protein_id in items
    ]

    # Commit every BATCH_SIZE entries to avoid buffering the entire dataset
    # in memory (579k pickled graphs in one transaction caused OOM).
    BATCH_SIZE = 5000

    # Track all written IDs for the metadata key
    all_ids = set(existing_ids)

    def _write_batch(db, batch, batch_ids, start_key):
        all_ids.update(batch_ids)
        with db.begin(write=True) as txn:
            for i, pickled in enumerate(batch):
                txn.put(key=str(start_key + i).encode(), value=pickled)
            txn.put(_IDS_META_KEY, pickle.dumps(all_ids))

    batch = []
    batch_ids = []

    # SIGTERM handler: flush current batch before exit (SLURM sends SIGTERM
    # before SIGKILL, typically with 30s grace period)
    _sigterm_received = False

    def _sigterm_handler(signum, frame):
        nonlocal _sigterm_received, batch, batch_ids, n_written
        _sigterm_received = True
        if batch:
            logger.info(
                f"SIGTERM received — flushing {len(batch)} buffered entries..."
            )
            _write_batch(db, batch, batch_ids, next_key + n_written)
            n_written += len(batch)
            batch = []
            batch_ids = []
        logger.info(
            f"Graceful shutdown: {n_written} new entries saved, "
            f"{len(all_ids)} total in LMDB"
        )
        db.close()
        sys.exit(0)

    signal.signal(signal.SIGTERM, _sigterm_handler)

    if num_workers > 0:
        # Parallel: parse CIF in workers, write LMDB in main process
        with multiprocessing.Pool(num_workers) as pool:
            for protein_id, pickled in tqdm(
                pool.imap_unordered(_parse_one_structure, worker_args, chunksize=32),
                total=len(worker_args),
                desc="Processing → LMDB",
            ):
                if _sigterm_received:
                    break
                if pickled is not None:
                    batch.append(pickled)
                    batch_ids.append(protein_id)
                    if len(batch) >= BATCH_SIZE:
                        _write_batch(db, batch, batch_ids, next_key + n_written)
                        n_written += len(batch)
                        logger.info(
                            f"Committed batch: {n_written} new / "
                            f"{len(all_ids)} total"
                        )
                        batch = []
                        batch_ids = []
                else:
                    n_failed += 1
    else:
        # Single-threaded fallback
        for args in tqdm(worker_args, desc="Processing → LMDB"):
            if _sigterm_received:
                break
            protein_id, pickled = _parse_one_structure(args)
            if pickled is not None:
                batch.append(pickled)
                batch_ids.append(protein_id)
                if len(batch) >= BATCH_SIZE:
                    _write_batch(db, batch, batch_ids, next_key + n_written)
                    n_written += len(batch)
                    logger.info(
                        f"Committed batch: {n_written} new / "
                        f"{len(all_ids)} total"
                    )
                    batch = []
                    batch_ids = []
            else:
                n_failed += 1

    # Flush remaining
    if batch:
        _write_batch(db, batch, batch_ids, next_key + n_written)
        n_written += len(batch)

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
