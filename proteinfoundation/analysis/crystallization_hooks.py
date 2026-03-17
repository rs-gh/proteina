# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary

"""
Crystallization Point Analysis - Hook Infrastructure

This module provides data structures for capturing attention intermediates
during inference, enabling analysis of when the model "crystallizes" from
global architecture search to local geometric locking.

Key classes:
- AttentionCapture: Stores captured attention data for a single layer
- CrystallizationTracker: Manages captures across timesteps and layers
"""

from dataclasses import dataclass, field
from typing import Dict, Optional

import torch
from torch import Tensor


@dataclass
class AttentionCapture:
    """
    Stores captured attention data for a single forward pass through one layer.

    Attributes:
        qk_raw: QK^T before scaling, shape [b, h, n, n]. This is the "Conditioning Content" (C).
        bias: Pair bias B, shape [b, h, n, n]. This is the "Geometric Memory".
        attn_weights: Post-softmax attention weights, shape [b, h, n, n].
        layer_idx: Index of the transformer layer (0-14 for 15 layers).
        timestep: Flow matching timestep t in [0, 1].
        timestep_idx: Index of the timestep in the sampling trajectory.
    """
    qk_raw: Optional[Tensor] = None
    bias: Optional[Tensor] = None
    attn_weights: Optional[Tensor] = None
    layer_idx: Optional[int] = None
    timestep: Optional[float] = None
    timestep_idx: Optional[int] = None
    node_repr: Optional[Tensor] = None

    def detach_and_clone(self) -> "AttentionCapture":
        """
        Create a detached copy of all tensors to avoid memory leaks from retaining
        the computation graph.
        """
        return AttentionCapture(
            qk_raw=self.qk_raw.detach().clone() if self.qk_raw is not None else None,
            bias=self.bias.detach().clone() if self.bias is not None else None,
            attn_weights=self.attn_weights.detach().clone() if self.attn_weights is not None else None,
            layer_idx=self.layer_idx,
            timestep=self.timestep,
            timestep_idx=self.timestep_idx,
            node_repr=self.node_repr.detach().clone() if self.node_repr is not None else None,
        )

    def to_cpu(self) -> "AttentionCapture":
        """Move all tensors to CPU to free GPU memory."""
        return AttentionCapture(
            qk_raw=self.qk_raw.cpu() if self.qk_raw is not None else None,
            bias=self.bias.cpu() if self.bias is not None else None,
            attn_weights=self.attn_weights.cpu() if self.attn_weights is not None else None,
            layer_idx=self.layer_idx,
            timestep=self.timestep,
            timestep_idx=self.timestep_idx,
            node_repr=self.node_repr.cpu() if self.node_repr is not None else None,
        )

    def reduce_heads(self, method: str = "mean") -> "AttentionCapture":
        """
        Reduce across attention heads to save memory.

        Args:
            method: Reduction method, either "mean" or "max"

        Returns:
            New AttentionCapture with reduced head dimension [b, 1, n, n]
        """
        reduce_fn = torch.mean if method == "mean" else torch.amax
        keepdim = True

        return AttentionCapture(
            qk_raw=reduce_fn(self.qk_raw, dim=1, keepdim=keepdim) if self.qk_raw is not None else None,
            bias=reduce_fn(self.bias, dim=1, keepdim=keepdim) if self.bias is not None else None,
            attn_weights=reduce_fn(self.attn_weights, dim=1, keepdim=keepdim) if self.attn_weights is not None else None,
            layer_idx=self.layer_idx,
            timestep=self.timestep,
            timestep_idx=self.timestep_idx,
        )


@dataclass
class BiasAblationConfig:
    """
    Configuration for pair bias ablation experiments.

    Controls when and how pair bias B is modified during generation, enabling
    causal testing of which (layer, timestep) regions require geometric bias.

    Attributes:
        enabled: Whether ablation is active
        ablate_layers: Set of layer indices to ablate. None = all layers.
        ablate_t_min: Minimum timestep for ablation (active when t >= t_min)
        ablate_t_max: Maximum timestep for ablation (active when t <= t_max)
        mode: 'zero' sets B=0; 'random' replaces B with magnitude-matched
              Gaussian noise (same std, random content). The 'random' mode
              tests whether the model needs the *information* in B or just
              any additive signal of similar scale.
    """
    enabled: bool = False
    ablate_layers: Optional[set] = None
    ablate_t_min: float = 0.0
    ablate_t_max: float = 1.0
    mode: str = 'zero'  # 'zero' or 'random'
    active_intervals: Optional[list] = None  # list of (t_min, t_max) where B is ON

    def should_ablate(self, layer_idx: int, timestep: float) -> bool:
        """Check if pair bias should be ablated for this (layer, timestep)."""
        if not self.enabled:
            return False
        if self.ablate_layers is not None and layer_idx not in self.ablate_layers:
            return False
        if self.active_intervals is not None:
            # B is ON during active_intervals, OFF everywhere else
            for t_lo, t_hi in self.active_intervals:
                if t_lo <= timestep <= t_hi:
                    return False  # B is active here, don't ablate
            return True  # not in any active interval → ablate
        return self.ablate_t_min <= timestep <= self.ablate_t_max


@dataclass
class CrystallizationTracker:
    """
    Tracks attention captures across the flow-matching trajectory.

    Structure: captures[timestep_idx][layer_idx] = AttentionCapture

    Attributes:
        enabled: Whether capture is currently active
        captures: Nested dict mapping timestep_idx -> layer_idx -> AttentionCapture
        current_timestep_idx: Current timestep index being processed
        current_timestep: Current timestep value (t in [0, 1])
        capture_every_n: Only capture every N timesteps (for memory efficiency)
        reduce_heads: Whether to reduce across heads to save memory
        move_to_cpu: Whether to move captures to CPU immediately
        bias_ablation: Configuration for pair bias ablation experiments
        capture_node_repr: Whether to capture node representations for structure lens
    """
    enabled: bool = False
    captures: Dict[int, Dict[int, AttentionCapture]] = field(default_factory=dict)
    current_timestep_idx: int = 0
    current_timestep: float = 0.0
    capture_every_n: int = 1
    reduce_heads: bool = False
    move_to_cpu: bool = True
    bias_ablation: BiasAblationConfig = field(default_factory=BiasAblationConfig)
    capture_node_repr: bool = False

    def enable(self, capture_every_n: int = 1, reduce_heads: bool = False, move_to_cpu: bool = True):
        """
        Enable capture and reset state.

        Args:
            capture_every_n: Only capture every N timesteps
            reduce_heads: Whether to average across heads to save memory
            move_to_cpu: Whether to move captures to CPU immediately
        """
        self.enabled = True
        self.captures = {}
        self.current_timestep_idx = 0
        self.current_timestep = 0.0
        self.capture_every_n = capture_every_n
        self.reduce_heads = reduce_heads
        self.move_to_cpu = move_to_cpu

    def disable(self):
        """Disable capture."""
        self.enabled = False

    def set_timestep(self, timestep_idx: int, timestep: float):
        """
        Set the current timestep being processed.

        Args:
            timestep_idx: Index of timestep in trajectory
            timestep: Timestep value t in [0, 1]
        """
        self.current_timestep_idx = timestep_idx
        self.current_timestep = timestep

    def should_capture(self) -> bool:
        """Check if we should capture at the current timestep."""
        if not self.enabled:
            return False
        return self.current_timestep_idx % self.capture_every_n == 0

    def store(self, layer_idx: int, capture: AttentionCapture):
        """
        Store a capture for the current timestep and given layer.

        Args:
            layer_idx: Index of the transformer layer
            capture: AttentionCapture with attention data
        """
        if not self.should_capture():
            return

        timestep_idx = self.current_timestep_idx

        # Initialize timestep dict if needed
        if timestep_idx not in self.captures:
            self.captures[timestep_idx] = {}

        # Add metadata
        capture.layer_idx = layer_idx
        capture.timestep = self.current_timestep
        capture.timestep_idx = timestep_idx

        # Process capture
        capture = capture.detach_and_clone()

        if self.reduce_heads:
            capture = capture.reduce_heads(method="mean")

        if self.move_to_cpu:
            capture = capture.to_cpu()

        self.captures[timestep_idx][layer_idx] = capture

    def clear(self):
        """Clear all stored captures."""
        self.captures = {}

    def get_timestep_indices(self) -> list:
        """Get sorted list of captured timestep indices."""
        return sorted(self.captures.keys())

    def get_layer_indices(self) -> list:
        """Get sorted list of layer indices from first captured timestep."""
        if not self.captures:
            return []
        first_timestep = min(self.captures.keys())
        return sorted(self.captures[first_timestep].keys())

    def get_capture(self, timestep_idx: int, layer_idx: int) -> Optional[AttentionCapture]:
        """
        Get a specific capture.

        Args:
            timestep_idx: Timestep index
            layer_idx: Layer index

        Returns:
            AttentionCapture or None if not found
        """
        if timestep_idx not in self.captures:
            return None
        return self.captures[timestep_idx].get(layer_idx)

    def memory_usage_mb(self) -> float:
        """Estimate total memory usage of stored captures in MB."""
        total_bytes = 0
        for timestep_dict in self.captures.values():
            for capture in timestep_dict.values():
                for tensor in [capture.qk_raw, capture.bias, capture.attn_weights, capture.node_repr]:
                    if tensor is not None:
                        total_bytes += tensor.numel() * tensor.element_size()
        return total_bytes / (1024 * 1024)

    def __len__(self) -> int:
        """Return total number of captures."""
        return sum(len(d) for d in self.captures.values())

    def __repr__(self) -> str:
        n_timesteps = len(self.captures)
        n_layers = len(self.get_layer_indices())
        mem_mb = self.memory_usage_mb()
        return f"CrystallizationTracker(timesteps={n_timesteps}, layers={n_layers}, memory={mem_mb:.1f}MB)"
