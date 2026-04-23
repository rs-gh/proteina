# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

import math
import os
from typing import List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.data import Batch
from torch_geometric.nn import radius_graph
from torch_scatter import scatter_mean, scatter_sum


def rbf(d, rbf_dim, d_min=0.0, d_max=20.0):
    d_mu = torch.linspace(d_min, d_max, rbf_dim, device=d.device)
    d_mu = d_mu.view([1, -1])
    d_sigma = (d_max - d_min) / rbf_dim
    d_expand = torch.unsqueeze(d, -1)

    rbf = torch.exp(-(((d_expand - d_mu) / d_sigma) ** 2))
    return rbf


def clockwise_angle(p1, p2):
    assert p1.shape[-1] == 3
    assert p2.shape[-1] == 3
    x = (p1 * p2).sum(dim=-1)
    y = torch.cross(p1, p2, dim=-1)
    angle = torch.atan2(y.norm(dim=-1), x) * torch.sign(y[..., 2])
    return angle


def get_index_embedding(indices, embed_size, max_len=2056):
    """Creates sine / cosine positional embeddings from a prespecified indices.

    Args:
        indices: offsets of size [..., N_edges] of type integer
        max_len: maximum length.
        embed_size: dimension of the embeddings to create

    Returns:
        positional embedding of shape [N, embed_size]
    """
    K = torch.arange(embed_size // 2, device=indices.device)
    pos_embedding_sin = torch.sin(
        indices[..., None] * math.pi / (max_len ** (2 * K[None] / embed_size))
    ).to(indices.device)
    pos_embedding_cos = torch.cos(
        indices[..., None] * math.pi / (max_len ** (2 * K[None] / embed_size))
    ).to(indices.device)
    pos_embedding = torch.cat([pos_embedding_sin, pos_embedding_cos], axis=-1)
    return pos_embedding


class Linear(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        dropout: float = 0.0,
        bias: bool = False,
        leakyrelu_negative_slope: float = 0.1,
        momentum: float = 0.2,
    ):
        super(Linear, self).__init__()

        module = []
        module.append(nn.BatchNorm1d(in_channels, momentum=momentum))
        module.append(nn.LeakyReLU(leakyrelu_negative_slope))
        module.append(nn.Dropout(dropout))
        module.append(nn.Linear(in_channels, out_channels, bias=bias))
        self.module = nn.Sequential(*module)

    def forward(self, x):
        return self.module(x)


class MLP(nn.Module):
    def __init__(
        self,
        in_channels: int,
        mid_channels: int,
        out_channels: int,
        dropout: float = 0.0,
        bias: bool = True,
        leakyrelu_negative_slope: float = 0.2,
        momentum: float = 0.2,
    ):
        super(MLP, self).__init__()

        module = []
        module.append(nn.BatchNorm1d(in_channels, momentum=momentum))
        module.append(nn.LeakyReLU(leakyrelu_negative_slope))
        module.append(nn.Dropout(dropout))
        if mid_channels is None:
            module.append(nn.Linear(in_channels, out_channels, bias=bias))
        else:
            module.append(nn.Linear(in_channels, mid_channels, bias=bias))
        if mid_channels is None:
            module.append(nn.BatchNorm1d(out_channels, momentum=momentum))
        else:
            module.append(nn.BatchNorm1d(mid_channels, momentum=momentum))
        module.append(nn.LeakyReLU(leakyrelu_negative_slope))
        if mid_channels is None:
            module.append(nn.Dropout(dropout))
        else:
            module.append(nn.Linear(mid_channels, out_channels, bias=bias))

        self.module = nn.Sequential(*module)

    def forward(self, input):
        return self.module(input)


class GeometricRelationalGraphConv(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        edge_input_dim: Optional[int] = 0,
        num_relation: Optional[int] = 1,
        leakyrelu_negative_slope: Optional[float] = 0.1,
        dropout: Optional[float] = 0.2,
        bias: Optional[bool] = False,
    ):
        """
        Geometry-aware relational graph convolution operator from
        `Protein Representation Learning by Geometric Structure Pretraining`_.

        .. _Protein Representation Learning by Geometric Structure Pretraining:
            https://arxiv.org/abs/2203.06125

        Args:
            input_dim (int): Input dimension
            output_dim (int): Output dimension
            edge_input_dim (int): Input dimension of edge features
            num_relation (int): Number of relations.
            leakyrelu_negative_slope (Optional[float], optional): Controls the angle of the negative slope in LeakyReLU.
                Defaults to 0.1.
            dropout (Optional[float], optional): Probability in Dropout.
                Defaults to 0.2.
            bias (Optional[bool], optional): Whether to have bias in Linear layers.
                Defaults to False.
        """
        super(GeometricRelationalGraphConv, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.edge_input_dim = edge_input_dim
        self.num_relation = num_relation

        if input_dim != output_dim:
            self.identity = Linear(
                in_channels=input_dim,
                out_channels=output_dim,
                dropout=dropout,
                bias=bias,
                leakyrelu_negative_slope=leakyrelu_negative_slope,
            )
        else:
            self.identity = nn.Sequential()

        self.input = MLP(
            in_channels=input_dim,
            mid_channels=None,
            out_channels=input_dim,
            dropout=dropout,
            leakyrelu_negative_slope=leakyrelu_negative_slope,
        )

        if edge_input_dim > 0:
            self.edge_input = MLP(
                in_channels=edge_input_dim,
                mid_channels=None,
                out_channels=input_dim,
                dropout=dropout,
                leakyrelu_negative_slope=leakyrelu_negative_slope,
            )

        self.linear = Linear(
            in_channels=num_relation * input_dim,
            out_channels=output_dim,
            dropout=dropout,
            bias=bias,
            leakyrelu_negative_slope=leakyrelu_negative_slope,
        )

        self.output = Linear(
            in_channels=output_dim,
            out_channels=output_dim,
            dropout=dropout,
            bias=bias,
            leakyrelu_negative_slope=leakyrelu_negative_slope,
        )

    def forward(self, h_v, edge_index, h_e=None):
        identity = self.identity(h_v)
        h_v = self.input(h_v)

        node_in, node_out, relation_type = edge_index
        message = h_v[node_in]
        if self.edge_input_dim > 0:
            message = message + self.edge_input(h_e)

        assert relation_type.max() < self.num_relation
        node_out = node_out * self.num_relation + relation_type
        update = scatter_sum(
            message, node_out, dim=0, dim_size=h_v.shape[0] * self.num_relation
        )
        update = update.view(h_v.shape[0], self.num_relation * self.input_dim)

        output = self.linear(update)

        out = self.output(output) + identity
        return out


class GearNet(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_layers: int,
        edge_input_dim: int,
        num_relation: int,
        leakyrelu_negative_slope: Optional[float] = 0.1,
        dropout: Optional[float] = 0.2,
        radius: Optional[float] = 5.0,
        num_classes: Optional[List[Tuple[str, int]]] = None,
        max_len: Optional[int] = 3000,
        ca_only: Optional[bool] = False,
    ) -> None:
        """
        GearNet for protein structure reprentation learning and fold classification from
        `Protein Representation Learning by Geometric Structure Pretraining`_.

        .. _Protein Representation Learning by Geometric Structure Pretraining:
            https://arxiv.org/abs/2203.06125

        Args:
            input_dim (int): Input dimension
            hidden_dim (int): Hidden dimension
            num_layers (int): Number of layers
            edge_input_dim (int): Input dimension of edge features
            num_relation (int): Number of relations. One type for spatial edges, all the other types for sequential edges.
            leakyrelu_negative_slope (Optional[float], optional): Controls the angle of the negative slope in LeakyReLU.
                Defaults to 0.1.
            dropout (Optional[float], optional): Probability in Dropout.
                Defaults to 0.2.
            radius (Optional[float], optional): Spatial radius (Å) for constructing structure graph
                Defaults to 5.0.
            num_classes (Optional[List[Tuple[str, int]]], optional): List of tuples (level, num_class), indicating which fold level to predict and the number of classes at this level.
                Defaults to None.
            max_len (Optional[int], optional): Maximum length for calculating positional embeddings.
                Defaults to 3000.
            ca_only (Optional[bool], optional): Whether to use backbone structure model or CA-only structure model.
                Defaults to False.
        """
        super(GearNet, self).__init__()
        self.input_dim = input_dim
        self.output_dim = hidden_dim
        self.dims = [input_dim] + [hidden_dim] * num_layers
        self.edge_input_dim = edge_input_dim
        self.num_relation = num_relation
        self.radius = radius
        self.num_classes = num_classes
        self.max_len = max_len
        self.rbf_dim = edge_input_dim // 2
        self.ca_only = ca_only

        if self.ca_only:
            self.atom_embedding = nn.Embedding(
                num_embeddings=1, embedding_dim=input_dim // 2
            )  # CA embedding
        else:
            self.atom_embedding = nn.Embedding(
                num_embeddings=3, embedding_dim=input_dim // 2
            )  # N, CA, C embeddings
        self.layers = nn.ModuleList()
        for i in range(len(self.dims) - 1):
            self.layers.append(
                GeometricRelationalGraphConv(
                    self.dims[i],
                    self.dims[i + 1],
                    edge_input_dim=self.edge_input_dim,
                    num_relation=num_relation,
                    leakyrelu_negative_slope=leakyrelu_negative_slope,
                    dropout=dropout,
                )
            )

        self.mlp = MLP(
            in_channels=self.output_dim,
            mid_channels=None,  # 2 * self.output_dim,
            out_channels=self.output_dim,
            bias=True,
            dropout=dropout,
            leakyrelu_negative_slope=leakyrelu_negative_slope,
        )

        if num_classes:
            # num_classes should be given as a subset of [('H', num_class_H), ('T', num_class_T), ('A', num_class_A), ('C', num_class_C)]
            for k, num_class in num_classes:
                setattr(self, "pred_head_%s" % k, nn.Linear(self.output_dim, num_class))

    def construct_graph(self, atom_seq_pos, coords, atom2batch):
        # Sequential graph
        max_distance = (self.num_relation - 1) // 2
        node_in, node_out = radius_graph(
            atom_seq_pos.float(), max_distance + 0.1, batch=atom2batch
        )
        relation = atom_seq_pos[node_out] - atom_seq_pos[node_in] + max_distance
        relation = relation.clamp(0, self.num_relation - 2)
        seq_edge_list = torch.stack([node_in, node_out, relation], dim=0)

        # Spatial graph
        node_in, node_out = radius_graph(
            coords, self.radius, batch=atom2batch, max_num_neighbors=64
        )
        radius_edge_list = torch.stack(
            [node_in, node_out, torch.ones_like(node_in) * (self.num_relation - 1)],
            dim=0,
        )

        edge_list = torch.cat([seq_edge_list, radius_edge_list], dim=1)
        return edge_list

    def node_feature(self, atom_type, atom_seq_pos):
        # Atom type embedding
        if self.ca_only:
            # Reindex CA atom type for embedding
            atom_type_embedding = self.atom_embedding(atom_type - 1)
        else:
            atom_type_embedding = self.atom_embedding(atom_type)

        # Positional embedding
        position_embedding = get_index_embedding(
            atom_seq_pos, self.input_dim // 2, max_len=self.max_len
        )

        h_v = torch.cat(
            [
                atom_type_embedding,
                position_embedding,
            ],
            dim=-1,
        )

        return h_v

    def edge_feature(self, edge_list, atom_seq_pos, coords, atom2batch):
        node_in, node_out, _ = edge_list

        # RBF features
        pos_in, pos_out = coords[node_in], coords[node_out]
        rbf_feat = rbf((pos_out - pos_in).norm(dim=-1), self.rbf_dim)

        # Relative position embeddings
        rel_pos = atom_seq_pos[node_out] - atom_seq_pos[node_in]
        rel_pos_emebdding = get_index_embedding(
            rel_pos, self.rbf_dim - 2, max_len=self.max_len
        )  # Leave two dims for angle feat

        # Angle features to break reflection symmetry
        center_coord = scatter_mean(coords, atom2batch, dim=0)  # (batch_size, 3)
        diff_node_in = pos_in - center_coord[atom2batch[node_in]]  # (num_edge, 3)
        diff_node_out = pos_out - center_coord[atom2batch[node_out]]  # (num_edge, 3)
        angle = clockwise_angle(diff_node_in, diff_node_out)  # (num_edge, )
        angle_feat = torch.stack([angle.sin(), angle.cos()], dim=-1)

        h_e = torch.cat([rbf_feat, rel_pos_emebdding, angle_feat], dim=-1)

        return h_e

    def atom_info(self, batch, atom_mask):
        # Flatten residue info into atom info
        device = atom_mask.device
        coords = batch.coords[atom_mask]  # (num_atom, 3)
        atom2batch = batch.batch[:, None].expand_as(atom_mask)
        atom2batch = atom2batch[atom_mask]
        atom_type = torch.arange(atom_mask.shape[-1], device=device)[None, :].expand_as(
            atom_mask
        )
        atom_type = atom_type[atom_mask]

        num_residues = scatter_sum(
            torch.ones_like(batch.batch), batch.batch, dim=0
        )  # (batch_size, )
        num_cum_residues = num_residues.cumsum(dim=0)
        residue_id = torch.arange(batch.batch.shape[0], device=device)
        residue_id = (
            residue_id - (num_cum_residues - num_residues)[batch.batch]
        )  # Remove shift from batching
        residue_id = residue_id[:, None].expand_as(atom_mask)  # (num_residue, 37)
        atom_seq_pos = residue_id[atom_mask]  # (num_atom, )

        return atom_type, atom_type, coords, atom_seq_pos, atom2batch

    def forward(self, batch: Batch):
        atom_mask = batch.coord_mask.bool()  # (num_residue, 37)
        # Mask irrelevant atoms
        if self.ca_only:
            atom_mask[:, 2:] = 0
            atom_mask[:, 0] = 0
        else:
            atom_mask[:, 3:] = 0
            assert (
                atom_mask[:, 0].any() or atom_mask[:, 2].any()
            ), "Only find CA atoms if the structure, please set ca_only=True if you are using CA-only structures"

        atom_type, atom_type, coords, atom_seq_pos, atom2batch = self.atom_info(
            batch, atom_mask
        )
        h_v = self.node_feature(atom_type, atom_seq_pos)

        edge_list = self.construct_graph(atom_seq_pos, coords, atom2batch)
        h_e = self.edge_feature(edge_list, atom_seq_pos, coords, atom2batch)

        for i in range(len(self.layers)):
            h_v = self.layers[i](h_v, edge_list, h_e)

        protein_feature = scatter_sum(h_v, atom2batch, dim=0)
        protein_feature = self.mlp(protein_feature)

        output = {
            "protein_feature": protein_feature,
        }

        # Predict fold class based on different levels
        if self.num_classes:
            for k, num_class in self.num_classes:
                pred = getattr(self, "pred_head_%s" % k)(protein_feature)
                output["pred_%s" % k] = pred

        return output


class NoTrainBBGearNet(GearNet):
    """
    Pre-trained GearNet model on backbone structures
    """

    def __init__(self, ckpt_path: str):
        super().__init__(
            input_dim=512,
            hidden_dim=512,
            edge_input_dim=256,
            num_layers=8,
            num_relation=6,
            dropout=0.2,
            radius=10.0,
            ca_only=False,
            num_classes=[["T", 1336], ["A", 43], ["C", 5]],
        )

        if os.path.exists(ckpt_path):
            state_dict = torch.load(ckpt_path, map_location="cpu")
            self.load_state_dict(state_dict)
        else:
            raise ValueError(
                f"NoTrainBBGearNet checkpoint path {ckpt_path} does not exist."
            )
        # put into evaluation mode
        self.eval()
        for p in self.parameters():
            p.requires_grad = False  # Turn off gradient

    def train(self, mode: bool) -> "NoTrainBBGearNet":
        """Force network to always be in evaluation mode."""
        return super().train(False)


class NoTrainCAGearNet(GearNet):
    """
    Pre-trained GearNet model on CA-only structures
    """

    def __init__(self, ckpt_path: str):
        super().__init__(
            input_dim=512,
            hidden_dim=512,
            edge_input_dim=256,
            num_layers=8,
            num_relation=6,
            dropout=0.2,
            radius=10.0,
            ca_only=True,
            num_classes=[["T", 1336], ["A", 43], ["C", 5]],
        )

        if os.path.exists(ckpt_path):
            state_dict = torch.load(ckpt_path, map_location="cpu")
            self.load_state_dict(state_dict)
        else:
            raise ValueError(
                f"NoTrainAAGearNet checkpoint path {ckpt_path} does not exist."
            )
        # put into evaluation mode
        self.eval()
        for p in self.parameters():
            p.requires_grad = False

    def train(self, mode: bool) -> "NoTrainCAGearNet":
        """Force network to always be in evaluation mode."""
        return super().train(False)


# ── MC-GearNet-Edge: plain-PyTorch reimplementation ──────────────────────────
#
# Reproduces torchdrug GearNet with edge network (Zhang et al., ICLR 2023)
# without torchdrug. Module names exactly match mc_gearnet_edge.pth (Zenodo
# 7593637). The checkpoint has no IEConv; it is plain GearNet-Edge with:
#   layers.{i}.{self_loop, linear, batch_norm}
#   edge_layers.{i}.{self_loop, linear, batch_norm}
#   batch_norms.{i}   (top-level, applied after short-cut)


class _GearNetLayer(nn.Module):
    """One GearNet message-passing layer. Names match layers.{i}.* and edge_layers.{i}.*."""

    def __init__(self, input_dim: int, output_dim: int, num_relation: int):
        super().__init__()
        self.input_dim = input_dim
        self.num_relation = num_relation
        self.self_loop = nn.Linear(input_dim, output_dim)
        self.linear = nn.Linear(num_relation * input_dim, output_dim)
        self.batch_norm = nn.BatchNorm1d(output_dim)

    def forward(
        self,
        h: torch.Tensor,          # [N, input_dim]
        node_in: torch.Tensor,    # [E] source indices
        node_out: torch.Tensor,   # [E] dest indices
        rel_type: torch.Tensor,   # [E] relation indices
        num_nodes: int,
        edge_input: Optional[torch.Tensor] = None,  # [E, input_dim] optional edge contribution
    ) -> torch.Tensor:
        msg = h[node_in]
        if edge_input is not None:
            msg = msg + edge_input
        idx = node_out * self.num_relation + rel_type
        update = scatter_sum(msg, idx, dim=0, dim_size=num_nodes * self.num_relation)
        update = update.view(num_nodes, self.num_relation * self.input_dim)
        out = self.linear(update) + self.self_loop(h)
        out = self.batch_norm(out)
        return F.relu(out)


class GearNetEdge(nn.Module):
    """
    Plain-PyTorch reimplementation of torchdrug GearNet with edge network.

    Module names match mc_gearnet_edge.pth (Zenodo 7593637) so that
    ``load_state_dict(torch.load(path), strict=False)`` loads correctly.
    strict=False is required because the Zenodo checkpoint includes a
    contrastive projection head that we intentionally omit.

    Fixed hyperparameters (matching mc_gearnet_edge pretraining):
        input_dim=21, hidden_dims=[512]*6, num_relation=7
        edge_input_dim=59, num_angle_bin=8
        batch_norm=True, short_cut=True, concat_hidden=True
    Output: per-residue concat of all 6 hidden layers → 3072-dim.
    """

    _SEQ_MAX_DIST: int = 2
    _SPATIAL_RADIUS: float = 10.0
    _KNN_K: int = 10
    _SEQ_MIN_DIST: int = 5
    _SPATIAL_REL: int = 5
    _KNN_REL: int = 6
    _NUM_RELATION: int = 7
    _NUM_ANGLE_BIN: int = 8

    def __init__(self) -> None:
        super().__init__()
        node_input_dim = 21
        hidden_dims = [512] * 6
        num_relation = self._NUM_RELATION
        num_angle_bin = self._NUM_ANGLE_BIN
        edge_input_dim = 59  # 21 + 21 + 7 + 10 (see _build_edges)

        # edge hidden dims mirror node hidden dims but starting from edge_input_dim
        # layer 0: 59→21, layers 1-5: 21→512 / 512→512
        # Deduced from checkpoint: edge_layers.0 output=21, edge_layers.1+ output=512
        edge_hidden_dims = [21] + [512] * (len(hidden_dims) - 1)

        node_dims = [node_input_dim] + list(hidden_dims)         # [21, 512, 512, ...]
        edge_dims = [edge_input_dim] + list(edge_hidden_dims)    # [59, 21, 512, ...]
        self.output_dim = sum(hidden_dims)  # 3072

        self.layers = nn.ModuleList([
            _GearNetLayer(node_dims[i], node_dims[i + 1], num_relation)
            for i in range(len(hidden_dims))
        ])
        self.edge_layers = nn.ModuleList([
            _GearNetLayer(edge_dims[i], edge_dims[i + 1], num_angle_bin)
            for i in range(len(hidden_dims))
        ])
        # Top-level batch norms applied after short-cut addition (names: batch_norms.{i})
        self.batch_norms = nn.ModuleList([
            nn.BatchNorm1d(node_dims[i + 1])
            for i in range(len(hidden_dims))
        ])

    # ── Internal helpers ────────────────────────────────────────────────────

    @staticmethod
    def _local_idx(atom2batch: torch.Tensor) -> torch.Tensor:
        """Per-node index within its batch element. Fully vectorised."""
        order = atom2batch.argsort(stable=True)
        counts = torch.bincount(atom2batch)
        # arange within each group
        local = torch.zeros_like(atom2batch)
        local[order] = torch.arange(atom2batch.shape[0], device=atom2batch.device) - \
            torch.repeat_interleave(
                torch.cat([torch.zeros(1, device=atom2batch.device, dtype=torch.long),
                           counts.cumsum(0)[:-1]]),
                counts
            )[order.argsort(stable=True)]
        return local

    def _build_edges(
        self,
        coords: torch.Tensor,
        residue_types: torch.Tensor,
        atom2batch: torch.Tensor,
        local_idx: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Build edge_index [E, 3] and 59-dim edge features [E, 59]."""
        device = coords.device

        # ── Sequential edges ─────────────────────────────────────────────────
        # radius_graph on local sequence position — same trick as NoTrainCAGearNet.
        # Edges cross batch boundaries are impossible because local_idx resets to 0
        # per protein, but batch= argument enforces it explicitly.
        seq_no, seq_ni = radius_graph(
            local_idx.float(),
            self._SEQ_MAX_DIST + 0.1,
            batch=atom2batch,
            loop=True,
            max_num_neighbors=2 * self._SEQ_MAX_DIST + 1,
        )  # returns (row=target, col=source) in source_to_target flow
        seq_offset = local_idx[seq_no].long() - local_idx[seq_ni].long()
        seq_rel = (seq_offset + self._SEQ_MAX_DIST).clamp(0, 2 * self._SEQ_MAX_DIST)

        # ── Spatial + KNN edges (single batched cdist) ────────────────────────
        # Build the full N×N distance matrix then mask out cross-batch pairs.
        c = coords.float()
        dist_mat = torch.cdist(c, c)                          # [N, N]
        cross_batch = atom2batch.unsqueeze(0) != atom2batch.unsqueeze(1)  # [N, N]
        dist_mat = dist_mat.masked_fill(cross_batch, float("inf"))

        # Sequence distance in global node indices (local_idx handles offsets)
        seq_dist = (local_idx.unsqueeze(0) - local_idx.unsqueeze(1)).abs()  # [N, N]
        # Cross-batch pairs get seq_dist=inf so they're excluded by all filters
        seq_dist = seq_dist.masked_fill(cross_batch, int(1e9))

        # Spatial edges: within radius, seq_dist >= SEQ_MIN_DIST, no self-loops
        sp_mask = (dist_mat < self._SPATIAL_RADIUS) & (seq_dist >= self._SEQ_MIN_DIST)
        sp_ni, sp_no = sp_mask.nonzero(as_tuple=True)

        # KNN edges: k nearest by distance, seq_dist >= SEQ_MIN_DIST
        d_knn = dist_mat.clone()
        d_knn[seq_dist < self._SEQ_MIN_DIST] = float("inf")
        n_total = c.shape[0]
        # topk across all nodes; k capped at (smallest protein size - 1)
        min_protein_size = int(torch.bincount(atom2batch).min().item())
        k = min(self._KNN_K, min_protein_size - 1)
        knn_ni_t: List[torch.Tensor] = []
        knn_no_t: List[torch.Tensor] = []
        if k > 0:
            _, knn_j = d_knn.topk(k, dim=-1, largest=False)       # [N, k]
            ki = torch.arange(n_total, device=device).unsqueeze(1).expand_as(knn_j).reshape(-1)
            kj = knn_j.reshape(-1)
            valid = d_knn[ki, kj] < float("inf")
            knn_ni_t.append(ki[valid])
            knn_no_t.append(kj[valid])

        node_in  = torch.cat([seq_ni, sp_ni] + knn_ni_t)
        node_out = torch.cat([seq_no, sp_no] + knn_no_t)
        rel = torch.cat([
            seq_rel,
            torch.full((sp_ni.shape[0],),  self._SPATIAL_REL, device=device, dtype=torch.long),
        ] + ([torch.full((knn_ni_t[0].shape[0],), self._KNN_REL, device=device, dtype=torch.long)]
             if knn_ni_t else []))

        edge_index = torch.stack([node_in, node_out, rel], dim=1)  # [E, 3]

        ni, no = edge_index[:, 0], edge_index[:, 1]
        rt = edge_index[:, 2]
        res_i = F.one_hot(residue_types[ni].clamp(0, 20), 21).float()
        res_j = F.one_hot(residue_types[no].clamp(0, 20), 21).float()
        rel_oh = F.one_hot(rt, self._NUM_RELATION).float()
        d = (coords[no].float() - coords[ni].float()).norm(dim=-1)
        centers = torch.linspace(0, 50, 10, device=device)
        d_rbf = torch.exp(-0.5 * ((d.unsqueeze(-1) - centers) / 5.0) ** 2)
        edge_feat = torch.cat([res_i, res_j, rel_oh, d_rbf], dim=-1)  # [E, 59]

        return edge_index, edge_feat

    def _build_line_graph(
        self,
        coords: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> Tuple[torch.Tensor, int]:
        device = coords.device
        n_edges = edge_index.shape[0]
        ni, no = edge_index[:, 0], edge_index[:, 1]

        if n_edges == 0:
            return torch.zeros(0, 3, dtype=torch.long, device=device), n_edges

        dirs = F.normalize((coords[no] - coords[ni]).float(), dim=-1, eps=1e-6)

        # Fully vectorised line-graph: pairs (e1, e2) where no[e1] == ni[e2].
        # Sort both edge lists by the middle node, then use repeat_interleave
        # to form cartesian products within each group — no Python loops over nodes.
        e1_order = no.argsort(stable=True)   # e1 sorted by destination
        e2_order = ni.argsort(stable=True)   # e2 sorted by source
        e1_keys  = no[e1_order]
        e2_keys  = ni[e2_order]

        e1_uniq, e1_counts = torch.unique_consecutive(e1_keys, return_counts=True)
        e2_uniq, e2_counts = torch.unique_consecutive(e2_keys, return_counts=True)

        # Find nodes that appear in both sorted arrays (shared middle nodes)
        shared_mask_e1 = torch.isin(e1_uniq, e2_uniq)
        shared_mask_e2 = torch.isin(e2_uniq, e1_uniq)
        if not shared_mask_e1.any():
            return torch.zeros(0, 3, dtype=torch.long, device=device), n_edges

        # Counts for shared nodes only
        s1 = e1_counts[shared_mask_e1]  # [S] e1-edges per shared node
        s2 = e2_counts[shared_mask_e2]  # [S] e2-edges per shared node

        # Offsets into the sorted e1/e2 arrays for each group
        e1_starts = (e1_counts.cumsum(0) - e1_counts)[shared_mask_e1]  # [S]
        e2_starts = (e2_counts.cumsum(0) - e2_counts)[shared_mask_e2]  # [S]

        # Build flat indices into sorted arrays for e1 and e2.
        # For each shared node k: e1 positions are e1_starts[k]..e1_starts[k]+s1[k]
        # We expand into a product of size s1[k]*s2[k] using repeat_interleave.
        # e1: each position repeated s2[k] times; e2: s1[k] positions tiled.
        e1_rep = torch.repeat_interleave(s2)   # total pairs per e1 edge
        e2_rep = torch.repeat_interleave(s1)   # total pairs per e2 edge

        # Flat e1 position array: arange within each group, repeated s2[k] times
        e1_group_idx = torch.repeat_interleave(
            torch.arange(s1.sum(), device=device),
            torch.repeat_interleave(s2, s1)
        )
        e2_group_idx = torch.arange(s2.sum(), device=device).repeat_interleave(
            torch.repeat_interleave(s1, s2)
        )

        # Map group-local positions back to sorted-array positions
        e1_base = torch.repeat_interleave(e1_starts, s1)  # base offset per e1 edge
        e2_base = torch.repeat_interleave(e2_starts, s2)  # base offset per e2 edge

        # arange within each e1 group
        e1_local = torch.arange(s1.sum(), device=device) - torch.repeat_interleave(
            torch.cat([torch.zeros(1, device=device, dtype=torch.long),
                       s1.cumsum(0)[:-1]]),
            s1
        )
        e2_local = torch.arange(s2.sum(), device=device) - torch.repeat_interleave(
            torch.cat([torch.zeros(1, device=device, dtype=torch.long),
                       s2.cumsum(0)[:-1]]),
            s2
        )

        # Sorted-array positions for each edge in the product
        # e1: each e1 edge appears s2[group] times; e2: each e2 edge appears s1[group] times
        e1_sorted_pos = (e1_base + e1_local).repeat_interleave(
            torch.repeat_interleave(s2, s1)
        )
        e2_sorted_pos = torch.repeat_interleave(
            e2_base + e2_local,
            torch.repeat_interleave(s1, s2)
        )

        lg_src = e1_order[e1_sorted_pos]
        lg_dst = e2_order[e2_sorted_pos]
        cos_a = (dirs[lg_src] * dirs[lg_dst]).sum(dim=-1).clamp(-1 + 1e-6, 1 - 1e-6)
        angle_bin = (torch.acos(cos_a) / math.pi * self._NUM_ANGLE_BIN).long().clamp(0, self._NUM_ANGLE_BIN - 1)

        return torch.stack([lg_src, lg_dst, angle_bin], dim=1), n_edges

    # ── Forward ─────────────────────────────────────────────────────────────

    def forward(
        self,
        coords: torch.Tensor,         # [N, 3] CA coords in Angstroms (valid residues only)
        residue_types: torch.Tensor,  # [N] long, 0-19 = AA, 20 = UNK
        atom2batch: torch.Tensor,     # [N] long
    ) -> torch.Tensor:
        """Returns per-residue embeddings [N, 3072]."""
        n_nodes = coords.shape[0]
        local_idx = self._local_idx(atom2batch)

        # Node features: 21-dim residue one-hot (no initial linear projection)
        h_v = F.one_hot(residue_types.clamp(0, 20), 21).float()

        edge_index, edge_feat59 = self._build_edges(coords, residue_types, atom2batch, local_idx)
        ni, no, rel = edge_index[:, 0], edge_index[:, 1], edge_index[:, 2]

        lg_ei, n_lg_nodes = self._build_line_graph(coords, edge_index)
        lg_ni, lg_no, lg_rel = lg_ei[:, 0], lg_ei[:, 1], lg_ei[:, 2]

        edge_hidden = edge_feat59  # [E, 59] initial edge features

        hiddens: List[torch.Tensor] = []
        for layer, edge_layer, bn in zip(self.layers, self.edge_layers, self.batch_norms):
            # 1. Update edge features via line graph
            edge_hidden = edge_layer(edge_hidden, lg_ni, lg_no, lg_rel, n_lg_nodes)
            # 2. Node conv with updated edge context; layer() already applies BN+ReLU
            h_new = layer(h_v, ni, no, rel, n_nodes, edge_input=edge_hidden)
            # 3. Short-cut residual (only valid when dims match, i.e. layer 1+)
            if h_new.shape == h_v.shape:
                h_new = h_new + h_v
            # 4. Top-level batch norm
            h_new = bn(h_new)
            hiddens.append(h_new)
            h_v = h_new

        return torch.cat(hiddens, dim=-1)  # [N, 3072]


class NoTrainMCGearNetEdge(GearNetEdge):
    """Frozen GearNetEdge loaded from mc_gearnet_edge.pth (Zenodo 7593637).

    Download the checkpoint with:
        hpc-scripts/proteina/data_prep/fetch_mc_gearnet_edge.sh
    """

    def __init__(self, ckpt_path: str):
        super().__init__()
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(
                f"MC-GearNet-Edge checkpoint not found: {ckpt_path!r}. "
                "Run hpc-scripts/proteina/data_prep/fetch_mc_gearnet_edge.sh to download it."
            )
        state = torch.load(ckpt_path, map_location="cpu")
        missing, unexpected = self.load_state_dict(state, strict=False)
        if missing:
            raise RuntimeError(
                f"Missing keys loading {ckpt_path!r} — architecture mismatch? "
                f"First missing key: {missing[0]!r} ({len(missing)} total)"
            )
        self.eval()
        for p in self.parameters():
            p.requires_grad_(False)

    def train(self, mode: bool = True) -> "NoTrainMCGearNetEdge":
        return super().train(False)


if __name__ == "__main__":
    model = NoTrainBBGearNet(ckpt_path="./model_weights/gearnet.pth")
    ca_model = NoTrainCAGearNet(ckpt_path="./model_weights/gearnet_ca.pth")

    breakpoint()
