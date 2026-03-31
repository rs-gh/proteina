"""Compatibility shim for torch_scatter/torch_sparse/torch_cluster.

These PyG C extension packages have chronic ABI compatibility issues with
newer torch versions on HPC clusters (GLIBC too old for pre-built wheels,
source builds produce undefined symbols). This module provides pure-PyTorch
replacements and patches sys.modules so proteina's imports work without them.

Usage: import this module BEFORE importing any proteina code:
    import proteinfoundation.repa.pyg_compat  # patches sys.modules
    from proteinfoundation.nn.protein_transformer import ProteinTransformerAF3  # works
"""

import importlib
import sys
import types

import torch


def _check_torch_scatter():
    """Check if torch_scatter actually works (not just importable)."""
    try:
        from torch_scatter import scatter_mean  # noqa: F401
        # Try actually calling it to detect ABI issues
        x = torch.randn(4, 2)
        idx = torch.tensor([0, 0, 1, 1])
        scatter_mean(x, idx, dim=0)
        return True
    except (ImportError, OSError, RuntimeError):
        return False


def _scatter_mean_native(src, index, dim=0, out=None, dim_size=None, fill_value=0):
    """Drop-in replacement for torch_scatter.scatter_mean using native PyTorch."""
    if dim_size is None:
        dim_size = int(index.max()) + 1 if index.numel() > 0 else 0

    # Expand index to match src dimensions
    idx = index
    for _ in range(src.dim() - idx.dim()):
        idx = idx.unsqueeze(-1)
    idx = idx.expand_as(src)

    # Use scatter_reduce (available since torch 1.12)
    result = torch.zeros(
        *src.shape[:dim], dim_size, *src.shape[dim + 1:],
        dtype=src.dtype, device=src.device,
    )
    result.scatter_reduce_(dim, idx, src, reduce="mean", include_self=False)
    return result


def _scatter_sum_native(src, index, dim=0, out=None, dim_size=None, fill_value=0):
    """Drop-in replacement for torch_scatter.scatter using native PyTorch."""
    if dim_size is None:
        dim_size = int(index.max()) + 1 if index.numel() > 0 else 0

    idx = index
    for _ in range(src.dim() - idx.dim()):
        idx = idx.unsqueeze(-1)
    idx = idx.expand_as(src)

    result = torch.zeros(
        *src.shape[:dim], dim_size, *src.shape[dim + 1:],
        dtype=src.dtype, device=src.device,
    )
    result.scatter_add_(dim, idx, src)
    return result


def _scatter_native(src, index, dim=0, out=None, dim_size=None, fill_value=0, reduce="sum"):
    """Generic scatter with reduce argument."""
    if reduce == "sum" or reduce == "add":
        return _scatter_sum_native(src, index, dim, out, dim_size, fill_value)
    elif reduce == "mean":
        return _scatter_mean_native(src, index, dim, out, dim_size, fill_value)
    else:
        raise NotImplementedError(f"scatter reduce={reduce} not implemented in compat shim")


def _radius_graph_native(x, r, batch=None, loop=False, max_num_neighbors=32, flow="source_to_target", num_workers=1, batch_size=None):
    """Drop-in replacement for torch_cluster.radius_graph using native PyTorch.

    Returns (row, col) edge index for all pairs within radius r,
    respecting batch boundaries. num_workers and batch_size are accepted
    for API compatibility but ignored.
    """
    if batch is None:
        batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)

    rows, cols = [], []

    # Process each batch element separately to respect boundaries
    for b in batch.unique():
        mask = batch == b
        idx = mask.nonzero(as_tuple=True)[0]
        x_b = x[idx]
        n = x_b.size(0)

        # Skip empty or single-atom batch elements (no edges possible)
        if n <= 1:
            continue

        # Pairwise distances
        dists = torch.cdist(x_b.unsqueeze(0).float(), x_b.unsqueeze(0).float()).squeeze(0)

        # Mask: within radius, not self-loop (unless loop=True)
        valid = dists < r
        if not loop:
            valid.fill_diagonal_(False)

        # Enforce max_num_neighbors: for each node keep closest neighbors
        if max_num_neighbors is not None and n > max_num_neighbors:
            # Set invalid distances to inf so they sort last
            dists_masked = dists.clone()
            dists_masked[~valid] = float("inf")
            _, topk_idx = dists_masked.topk(max_num_neighbors, dim=1, largest=False)
            new_valid = torch.zeros_like(valid)
            new_valid.scatter_(1, topk_idx, True)
            valid = valid & new_valid

        src, dst = valid.nonzero(as_tuple=True)

        if flow == "source_to_target":
            rows.append(idx[src])
            cols.append(idx[dst])
        else:
            rows.append(idx[dst])
            cols.append(idx[src])

    if rows:
        row = torch.cat(rows)
        col = torch.cat(cols)
    else:
        row = torch.zeros(0, dtype=torch.long, device=x.device)
        col = torch.zeros(0, dtype=torch.long, device=x.device)

    return row, col


def _check_torch_cluster():
    """Check if torch_cluster actually works (not just importable)."""
    try:
        from torch_cluster import radius_graph  # noqa: F401
        x = torch.randn(4, 3)
        batch = torch.tensor([0, 0, 1, 1])
        radius_graph(x, 1.0, batch)
        return True
    except (ImportError, OSError, RuntimeError):
        return False


def patch_pyg_imports():
    """Patch sys.modules with native PyTorch replacements for torch_scatter/torch_sparse/torch_cluster.

    Only patches modules whose real C extensions don't work. Safe to call multiple times.
    """
    patched = []

    # Patch torch_scatter if needed
    if not _check_torch_scatter():
        patched.append("torch_scatter")

        torch_scatter = types.ModuleType("torch_scatter")
        torch_scatter.scatter_mean = _scatter_mean_native
        torch_scatter.scatter_sum = _scatter_sum_native
        torch_scatter.scatter = _scatter_native
        torch_scatter.scatter_add = _scatter_sum_native
        sys.modules["torch_scatter"] = torch_scatter

        torch_scatter_composite = types.ModuleType("torch_scatter.composite")
        torch_scatter_composite.scatter_mean = _scatter_mean_native
        sys.modules["torch_scatter.composite"] = torch_scatter_composite

    # Patch torch_sparse if needed (imported but not actually called in proteina)
    if "torch_sparse" not in sys.modules:
        try:
            import torch_sparse  # noqa: F401
            torch_sparse.SparseTensor
        except (ImportError, OSError, AttributeError):
            patched.append("torch_sparse")
            torch_sparse = types.ModuleType("torch_sparse")
            torch_sparse.SparseTensor = None
            sys.modules["torch_sparse"] = torch_sparse

    # Patch torch_cluster with native radius_graph if needed.
    # Must place in sys.modules BEFORE torch_geometric is imported, so that
    # torch_geometric.typing picks up our fake module instead of the broken one.
    if not _check_torch_cluster():
        patched.append("torch_cluster")

        class _FakeTorchCluster(types.ModuleType):
            """Fake torch_cluster that provides radius_graph and stubs everything else.

            torch_geometric imports many names from torch_cluster at module init
            (knn, knn_graph, graclus_cluster, etc.). We provide radius_graph as a
            real implementation and return callable stubs for everything else, so
            imports succeed but calling unimplemented functions raises clearly.
            """
            def __init__(self):
                super().__init__("torch_cluster")
                self.radius_graph = _radius_graph_native

            def __getattr__(self, name):
                # Let Python handle dunder attributes normally (inspect
                # needs __file__, __path__, etc. to be missing, not stubs).
                if name.startswith("__") and name.endswith("__"):
                    raise AttributeError(name)
                # Return a callable stub for any other attribute
                def _stub(*args, **kwargs):
                    raise NotImplementedError(
                        f"torch_cluster.{name} not implemented in pyg_compat shim"
                    )
                _stub.__name__ = name
                _stub.__doc__ = f"Stub for torch_cluster.{name}."
                return _stub

        torch_cluster = _FakeTorchCluster()
        sys.modules["torch_cluster"] = torch_cluster

        # If torch_geometric was already imported, fix its cached references.
        # Otherwise, torch_geometric.typing will pick up our fake module naturally.
        if "torch_geometric.typing" in sys.modules:
            _typing = sys.modules["torch_geometric.typing"]
            _typing.torch_cluster = torch_cluster
            _typing.WITH_TORCH_CLUSTER = True
        if "torch_geometric.nn.pool" in sys.modules:
            sys.modules["torch_geometric.nn.pool"].torch_cluster = torch_cluster

    if patched:
        print(f"[pyg_compat] Patched {', '.join(patched)} with native PyTorch ops")


# Auto-patch on import
patch_pyg_imports()
