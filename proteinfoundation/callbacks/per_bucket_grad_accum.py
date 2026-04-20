"""Per-bucket gradient accumulation callback for length-bucketed training.

When `LengthBucketedBatchSampler` yields variable `(B, N)` batches across
buckets, the effective optimizer-step batch size varies by bucket: e.g., the
n=128 bucket may run BS=80 while the n=512 bucket runs BS=6. This breaks LR
schedule parity with a fixed-BS baseline.

This callback equalizes effective BS across buckets by mutating
`trainer.accumulate_grad_batches` per batch, computed as:

    accum[b] = round(target_effective_bs / bucket_batch_sizes[b])

so that `accum[b] * bucket_batch_sizes[b] ≈ target_effective_bs` for every
bucket. The collator tags each batch with `batch.bucket_length` (set by
`dense_padded_from_data_list` when `bucket_boundaries` is provided); the
callback looks up the accum value via that tag.

Usage:
    trainer = Trainer(
        callbacks=[
            PerBucketGradAccumCallback(
                bucket_boundaries=[128, 256, 384, 512],
                bucket_batch_sizes=[80, 24, 10, 6],
                target_effective_bs=80,
            ),
            ...
        ],
        accumulate_grad_batches=1,  # callback overrides this per-step
        ...
    )
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import lightning as L


class PerBucketGradAccumCallback(L.Callback):
    def __init__(
        self,
        bucket_boundaries: Sequence[int],
        bucket_batch_sizes: Sequence[int],
        target_effective_bs: int,
    ):
        if len(bucket_boundaries) != len(bucket_batch_sizes):
            raise ValueError(
                f"bucket_boundaries ({len(bucket_boundaries)}) and "
                f"bucket_batch_sizes ({len(bucket_batch_sizes)}) must match length."
            )
        if target_effective_bs <= 0:
            raise ValueError(f"target_effective_bs must be > 0 (got {target_effective_bs}).")

        self.accum_by_bucket: Mapping[int, int] = {
            int(b): max(1, round(target_effective_bs / int(bs)))
            for b, bs in zip(bucket_boundaries, bucket_batch_sizes)
        }
        self.target_effective_bs = int(target_effective_bs)

    def on_train_batch_start(
        self,
        trainer: L.Trainer,
        pl_module: L.LightningModule,
        batch: Any,
        batch_idx: int,
    ) -> None:
        bucket_length = self._extract_bucket_length(batch)
        if bucket_length is None:
            return
        accum = self.accum_by_bucket.get(int(bucket_length))
        if accum is None:
            return
        trainer.accumulate_grad_batches = accum

    @staticmethod
    def _extract_bucket_length(batch: Any) -> int | None:
        """`dense_padded_from_data_list` sets this attribute when
        `bucket_boundaries` is provided. We support both direct-attr and
        dict-style lookup to stay robust across collator variants."""
        val = getattr(batch, "bucket_length", None)
        if val is not None:
            return int(val)
        if isinstance(batch, Mapping) and "bucket_length" in batch:
            return int(batch["bucket_length"])
        return None
