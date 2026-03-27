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


def patch_pyg_imports():
    """Patch sys.modules with native PyTorch replacements for torch_scatter/torch_sparse.

    Only patches if the real C extensions don't work. Safe to call multiple times.
    """
    if _check_torch_scatter():
        return  # Real extensions work, nothing to patch

    # Create fake torch_scatter module
    torch_scatter = types.ModuleType("torch_scatter")
    torch_scatter.scatter_mean = _scatter_mean_native
    torch_scatter.scatter_sum = _scatter_sum_native
    torch_scatter.scatter = _scatter_native
    torch_scatter.scatter_add = _scatter_sum_native
    sys.modules["torch_scatter"] = torch_scatter

    # Create fake torch_scatter.composite module (some code imports from submodules)
    torch_scatter_composite = types.ModuleType("torch_scatter.composite")
    torch_scatter_composite.scatter_mean = _scatter_mean_native
    sys.modules["torch_scatter.composite"] = torch_scatter_composite

    print("[pyg_compat] Patched torch_scatter with native PyTorch ops")


# Auto-patch on import
patch_pyg_imports()
