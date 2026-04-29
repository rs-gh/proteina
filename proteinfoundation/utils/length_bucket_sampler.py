"""Length-bucketed batch sampler.

Groups dataset indices into buckets by protein length, then yields
fixed-size batches drawn from within a single bucket. Each batch is
length-homogeneous, so the collator can pad to `bucket_max` rather than a
global `max_size` - cutting attention FLOP-waste by removing the large gap
between short proteins and the tier's max.

Not imported by any existing code path. Becomes active only when a dataset
config sets ``sampling_mode: length-bucketed`` and provides
``bucket_boundaries`` / ``bucket_batch_sizes``.

Boundaries are interpreted as **inclusive upper edges**, using
``np.searchsorted(side='left')``:

    bucket_boundaries = [128, 256, 384, 512]
    -> bucket 0 = (_, 128], bucket 1 = (128, 256],
      bucket 2 = (256, 384], bucket 3 = (384, 512]

Samples with length above the last boundary are dropped (dataset-level
filtering via ``max_num_residues`` should already prevent this).
"""

from __future__ import annotations

from typing import Iterator, List, Sequence

import numpy as np
from torch.utils.data import Sampler


class LengthBucketedBatchSampler(Sampler[List[int]]):
    """BatchSampler yielding length-homogeneous batches.

    Arguments:
        lengths: per-sample protein length, one entry per dataset index.
        bucket_boundaries: inclusive upper edges, ascending. Must be the
            same length as ``bucket_batch_sizes``.
        bucket_batch_sizes: batch size used within each bucket.
        shuffle: shuffle sample order within buckets and batch order
            across buckets, per epoch. Default True.
        drop_last: drop the trailing partial batch in each bucket. Default
            True (so every yielded batch has exactly the bucket's BS,
            which is required for a fixed-shape compiled graph).
        seed: deterministic per-epoch seed. ``set_epoch(epoch)`` bumps it.
    """

    def __init__(
        self,
        lengths: Sequence[int],
        bucket_boundaries: Sequence[int],
        bucket_batch_sizes: Sequence[int],
        shuffle: bool = True,
        drop_last: bool = True,
        seed: int = 0,
    ):
        if len(bucket_boundaries) != len(bucket_batch_sizes):
            raise ValueError(
                f"len(bucket_boundaries)={len(bucket_boundaries)} must equal "
                f"len(bucket_batch_sizes)={len(bucket_batch_sizes)}"
            )
        if list(bucket_boundaries) != sorted(bucket_boundaries):
            raise ValueError("bucket_boundaries must be ascending")
        if any(bs <= 0 for bs in bucket_batch_sizes):
            raise ValueError("bucket_batch_sizes must be positive")

        self.lengths = np.asarray(lengths, dtype=np.int64)
        self.boundaries = np.asarray(bucket_boundaries, dtype=np.int64)
        self.bs_per_bucket = list(bucket_batch_sizes)
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.seed = int(seed)
        self.epoch = 0

        # Assign each sample to a bucket (right-exclusive below, inclusive above).
        # side='left' => length == boundary lands in the lower bucket.
        bucket_of = np.searchsorted(self.boundaries, self.lengths, side="left")

        # Clamp: samples above the last boundary land in no bucket; drop them.
        # (dataset filtering by max_num_residues should make this empty.)
        n_dropped = int((bucket_of >= len(self.boundaries)).sum())
        if n_dropped > 0:
            import warnings

            warnings.warn(
                f"LengthBucketedBatchSampler: dropping {n_dropped} samples with "
                f"length > {int(self.boundaries[-1])} (max boundary).",
                stacklevel=2,
            )

        self.buckets: List[np.ndarray] = [
            np.where(bucket_of == b)[0] for b in range(len(self.boundaries))
        ]

    def set_epoch(self, epoch: int) -> None:
        """Call this before iterating each epoch for deterministic per-epoch shuffle."""
        self.epoch = int(epoch)

    def __iter__(self) -> Iterator[List[int]]:
        rng = np.random.default_rng(self.seed + self.epoch)

        batches: List[List[int]] = []
        for b, idxs in enumerate(self.buckets):
            if len(idxs) == 0:
                continue
            bs = self.bs_per_bucket[b]
            order = rng.permutation(idxs) if self.shuffle else idxs
            n_full = len(order) // bs
            for i in range(n_full):
                batches.append(order[i * bs : (i + 1) * bs].tolist())
            if not self.drop_last and len(order) % bs:
                batches.append(order[n_full * bs :].tolist())

        if self.shuffle:
            batches = [batches[i] for i in rng.permutation(len(batches))]
        return iter(batches)

    def __len__(self) -> int:
        total = 0
        for idxs, bs in zip(self.buckets, self.bs_per_bucket):
            if self.drop_last:
                total += len(idxs) // bs
            else:
                total += (len(idxs) + bs - 1) // bs
        return total

    def describe(self) -> str:
        """Human-readable bucket summary for logging."""
        lines = ["LengthBucketedBatchSampler:"]
        prev = 0
        total_samples = sum(len(b) for b in self.buckets)
        total_batches = 0
        for i, (upper, bs, idxs) in enumerate(
            zip(self.boundaries, self.bs_per_bucket, self.buckets)
        ):
            n = len(idxs)
            n_batches = n // bs if self.drop_last else (n + bs - 1) // bs
            total_batches += n_batches
            pct = 100 * n / max(total_samples, 1)
            lines.append(
                f"  bucket {i}: ({prev}, {upper}]  BS={bs}  "
                f"n_samples={n} ({pct:.1f}%)  n_batches={n_batches}"
            )
            prev = int(upper)
        lines.append(f"  total batches per epoch: {total_batches}")
        return "\n".join(lines)
