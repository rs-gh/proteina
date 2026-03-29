"""LMDB-backed dataset for proteina protein structures.

Drop-in replacement for PDBDataset that reads from a single LMDB file
instead of individual .pt files. Returns the same PyG Data objects,
so the collation, transforms, and training pipeline are unchanged.

Compatible with num_workers > 0: the database connection is opened
lazily in each worker process (LMDB Environment objects cannot be
pickled across process boundaries).
"""

import pickle
from typing import Callable, Optional

import lmdb
from torch.utils.data import Dataset
from torch_geometric.data import Data


class ProteinLMDBDataset(Dataset):
    """Dataset that reads PyG protein graphs from an LMDB file.

    Each entry is a pickled ``torch_geometric.data.Data`` object stored
    with a sequential string key ("0", "1", ...).  Coordinate reordering
    (PDB -> OpenFold) should be applied during LMDB creation, not here.

    Args:
        lmdb_path: Path to the .lmdb file.
        transform: Optional transform applied to each sample.
    """

    def __init__(
        self,
        lmdb_path: str,
        transform: Optional[Callable] = None,
    ):
        super().__init__()
        self.lmdb_path = lmdb_path
        self.transform = transform
        self._db = None
        self._keys = None

        # Query length eagerly (needed by DataLoader before forking),
        # then close the connection so the object is picklable.
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
