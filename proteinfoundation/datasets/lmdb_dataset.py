"""LMDB-backed dataset for proteina protein structures.

Drop-in replacement for PDBDataset that reads from a single LMDB file
instead of individual .pt files. Returns the same PyG Data objects,
so the collation, transforms, and training pipeline are unchanged.

Compatible with num_workers > 0: the database connection is opened
lazily in each worker process (LMDB Environment objects cannot be
pickled across process boundaries).
"""

import os
import pickle
from typing import Callable, Optional

import lmdb
import numpy as np
from loguru import logger
from torch.utils.data import Dataset
from torch_geometric.data import Data


class ProteinLMDBDataset(Dataset):
    """Dataset that reads PyG protein graphs from an LMDB file.

    Each entry is a pickled ``torch_geometric.data.Data`` object stored
    with a sequential string key ("0", "1", ...).  Coordinate reordering
    (PDB -> OpenFold) should be applied during LMDB creation, not here.

    Supports optional max_num_residues filtering using a precomputed
    length index (built by hpc-scripts/proteina/data_prep/build_lmdb_length_index.py). If no index
    exists, falls back to scanning the LMDB on first connect.

    Args:
        lmdb_path: Path to the .lmdb file.
        transform: Optional transform applied to each sample.
        max_num_residues: If set, only include proteins with <= this many
            residues. Requires a length index or triggers a scan.
    """

    def __init__(
        self,
        lmdb_path: str,
        transform: Optional[Callable] = None,
        max_num_residues: Optional[int] = None,
    ):
        super().__init__()
        self.lmdb_path = lmdb_path
        self.transform = transform
        self.max_num_residues = max_num_residues
        self._db = None
        self._keys = None

        # Build filtered key list eagerly if max_num_residues is set,
        # otherwise just count entries. Needed by DataLoader before forking.
        self._filtered_lengths: Optional[np.ndarray] = None
        if max_num_residues is not None:
            self._filtered_keys = self._build_filtered_keys()
            self._len = len(self._filtered_keys)
        else:
            self._filtered_keys = None
            self._len = self._query_len()

    def _query_len(self) -> int:
        """Open DB, count entries, close. Safe for pre-fork main process."""
        db = lmdb.open(
            self.lmdb_path,
            map_size=50 * (1024 ** 3),
            create=False,
            subdir=False,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
        )
        length = db.stat()["entries"]
        db.close()
        return length

    def _build_filtered_keys(self):
        """Build a filtered key list based on max_num_residues.

        Tries to load a precomputed length index ({split}_lengths.npy and
        {split}_keys.pkl) from the same directory as the LMDB. If not found,
        falls back to scanning the LMDB directly.
        """
        lmdb_dir = os.path.dirname(self.lmdb_path)
        # e.g. "train" from "train.lmdb"
        split = os.path.basename(self.lmdb_path).replace(".lmdb", "")

        lengths_path = os.path.join(lmdb_dir, f"{split}_lengths.npy")
        keys_path = os.path.join(lmdb_dir, f"{split}_keys.pkl")

        if os.path.exists(lengths_path) and os.path.exists(keys_path):
            logger.info(
                f"Loading length index from {lengths_path} "
                f"(filtering to <= {self.max_num_residues} residues)"
            )
            lengths = np.load(lengths_path)
            with open(keys_path, "rb") as f:
                all_keys = pickle.load(f)

            mask = lengths <= self.max_num_residues
            filtered = [k for k, m in zip(all_keys, mask) if m]
            self._filtered_lengths = lengths[mask].astype(np.int64, copy=False)
            logger.info(
                f"Filtered {len(filtered)}/{len(all_keys)} entries "
                f"(<= {self.max_num_residues} residues)"
            )
            return filtered
        else:
            logger.warning(
                f"Length index not found at {lengths_path}. "
                f"Scanning LMDB to filter (run hpc-scripts/proteina/data_prep/build_lmdb_length_index.py to speed this up)."
            )
            return self._scan_and_filter()

    def _scan_and_filter(self):
        """Scan the LMDB to build a filtered key list. Slow fallback."""
        db = lmdb.open(
            self.lmdb_path,
            map_size=50 * (1024 ** 3),
            create=False,
            subdir=False,
            readonly=True,
            lock=False,
            readahead=True,
            meminit=False,
        )
        filtered = []
        filtered_lengths: list[int] = []
        with db.begin() as txn:
            cursor = txn.cursor()
            total = db.stat()["entries"]
            for i, (key, value) in enumerate(cursor):
                data = pickle.loads(value)
                n = int(data.coords.shape[0])
                if n <= self.max_num_residues:
                    filtered.append(key)
                    filtered_lengths.append(n)
                if (i + 1) % 50000 == 0:
                    logger.info(f"  Scanned {i + 1}/{total}, kept {len(filtered)} so far")

        db.close()
        self._filtered_lengths = np.asarray(filtered_lengths, dtype=np.int64)
        logger.info(
            f"Scan complete: {len(filtered)}/{total} entries "
            f"(<= {self.max_num_residues} residues)"
        )
        return filtered

    def get_lengths(self) -> np.ndarray:
        """Per-sample lengths aligned with dataset indexing (required by
        LengthBucketedBatchSampler). Raises if the dataset was constructed
        without `max_num_residues` - in that case the filtered key list is
        not built and no length index has been loaded.
        """
        if self._filtered_lengths is None:
            raise RuntimeError(
                "get_lengths() requires max_num_residues to be set so the length "
                "index is loaded at construction time."
            )
        return self._filtered_lengths

    def _connect_db(self):
        """Open LMDB in read-only mode. Called lazily in each worker."""
        self._db = lmdb.open(
            self.lmdb_path,
            map_size=50 * (1024 ** 3),
            create=False,
            subdir=False,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
        )
        if self._filtered_keys is not None:
            # Use pre-filtered keys
            self._keys = self._filtered_keys
        else:
            # No filtering - use all keys
            with self._db.begin() as txn:
                self._keys = list(txn.cursor().iternext(values=False))

    def __len__(self) -> int:
        return self._len

    def __getitem__(self, idx: int) -> Data:
        if self._db is None:
            self._connect_db()

        key = self._keys[idx]
        graph = pickle.loads(self._db.begin().get(key))

        if self.transform is not None:
            graph = self.transform(graph)

        return graph

    def close(self):
        """Close the database connection."""
        if self._db is not None:
            self._db.close()
            self._db = None
            self._keys = None
