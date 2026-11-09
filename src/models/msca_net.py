"""MSCA-G: a Multi-Scale Convolutional Attention network with an adaptive
cross-sensor graph.

A compact, high-accuracy convolutional classifier for accelerometer
time-series. An adaptive sensor-graph front-end mixes information across the
sensor channels (the connectivity is learned, not fixed), an InceptionTime-style
**multi-scale** extractor with **squeeze-and-excitation** channel recalibration
builds features, and a learned **attention-pooling** head aggregates over time.
"""

from __future__ import annotations

from typing import Literal

import torch
from torch import Tensor, nn

InputLayout = Literal["btc", "bct"]


def make_group_norm(channels: int, max_groups: int = 8) -> nn.GroupNorm:
    """Create GroupNorm with a valid group count for the channel size."""
    groups = min(max_groups, channels)
    while channels % groups != 0:
        groups -= 1
    return nn.GroupNorm(num_groups=groups, num_channels=channels)


class SqueezeExcite1d(nn.Module):
    """Squeeze-and-excitation channel attention for 1D feature maps."""

    def __init__(self, channels: int, reduction: int = 8) -> None:
        super().__init__()
        hidden = max(1, channels // reduction)
        self.fc = nn.Sequential(
            nn.Linear(channels, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, channels),
            nn.Sigmoid(),
        )

    def forward(self, x: Tensor) -> Tensor:
        """Recalibrate channels of a ``[B, C, T]`` feature map."""
        weights = self.fc(x.mean(dim=-1)).unsqueeze(-1)
        return x * weights


class MultiScaleConvBlock(nn.Module):
    """Inception-style multi-scale Conv1D block with SE and a residual path."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernels: tuple[int, ...] = (9, 19, 39),
        bottleneck_channels: int = 32,
        dropout: float = 0.1,
        use_se: bool = True,
        use_residual: bool = True,
        pool_branch_channels: int | None = None,
    ) -> None:
        super().__init__()
        use_bottleneck = in_channels > 1
        branch_in = bottleneck_channels if use_bottleneck else in_channels
        self.bottleneck = (
            nn.Conv1d(in_channels, bottleneck_channels, kernel_size=1, bias=False)
            if use_bottleneck
            else nn.Identity()
        )
        if not kernels or any(k < 1 or k % 2 == 0 for k in kernels):
            raise ValueError('kernels must contain positive odd sizes')
        if pool_branch_channels is None:
            branch_channels = out_channels // (len(kernels) + 1)
            pool_channels = out_channels - branch_channels * len(kernels)
        else:
            pool_channels = pool_branch_channels
            if not 0 < pool_channels < out_channels or (out_channels-pool_channels) % len(kernels):
                raise ValueError('Invalid convolution/pooling branch widths')
            branch_channels = (out_channels-pool_channels)//len(kernels)
        self.use_residual = use_residual
        self.branches = nn.ModuleList(
            nn.Conv1d(branch_in, branch_channels, kernel_size=k, padding=k // 2, bias=False)
            for k in kernels
        )
        self.pool_branch = nn.Sequential(
            nn.MaxPool1d(kernel_size=3, stride=1, padding=1),
            nn.Conv1d(in_channels, pool_channels, kernel_size=1, bias=False),
        )
        self.norm = make_group_norm(out_channels)
        self.activation = nn.GELU()
        self.se = SqueezeExcite1d(out_channels) if use_se else nn.Identity()
        self.dropout = nn.Dropout(dropout)
        self.shortcut = (
            nn.Sequential(nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False), make_group_norm(out_channels))
            if in_channels != out_channels
            else nn.Identity()
        )

    def forward(self, x: Tensor) -> Tensor:
        """Return residual multi-scale features shaped ``[B, out_channels, T]``."""
        bottleneck = self.bottleneck(x)
        branches = [branch(bottleneck) for branch in self.branches]
        branches.append(self.pool_branch(x))
        features = self.activation(self.norm(torch.cat(branches, dim=1)))
        features = self.dropout(self.se(features))
        return self.activation(features + self.shortcut(x) if self.use_residual else features)


class AttentionPool1d(nn.Module):
    """Learned attention pooling over the temporal dimension."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.attention = nn.Conv1d(channels, 1, kernel_size=1)

    def forward(self, x: Tensor) -> Tensor:
        """Pool ``[B, C, T]`` into ``[B, C]`` with per-time-step attention."""
        weights = torch.softmax(self.attention(x), dim=-1)
        return (x * weights).sum(dim=-1)


class AdaptiveSensorGraph(nn.Module):
    """Adaptive cross-sensor mixing with a learned adjacency (graph front-end).

    Treats the raw sensor channels as graph nodes and propagates information
    across them through ``A = softmax(ReLU(E Eᵀ))``, where ``E`` are learnable
    node embeddings. Motivated by SHM: damage changes the correlation structure
    between sensors, which a fixed wiring cannot capture.
    """

    def __init__(self, num_sensors: int, embed_dim: int = 10) -> None:
        super().__init__()
        self.node_embeddings = nn.Parameter(torch.empty(num_sensors, embed_dim))
        nn.init.xavier_uniform_(self.node_embeddings)
        self.proj = nn.Conv1d(num_sensors, num_sensors, kernel_size=1)

    def forward(self, x: Tensor) -> Tensor:
        """Residually mix the ``[B, N, T]`` sensor channels over the graph."""
        adjacency = torch.softmax(torch.relu(self.node_embeddings @ self.node_embeddings.t()), dim=1)
        mixed = torch.einsum("nm,bmt->bnt", adjacency, x)
        return x + self.proj(mixed)


class MSCANet(nn.Module):
    """Multi-Scale Convolutional Attention Network.

    Args:
        num_channels: Number of input sensor channels.
        num_classes: Number of output classes.
        hidden_channels: Width of every multi-scale block.
        num_blocks: Number of multi-scale blocks.
        dropout: Dropout inside blocks and before the classifier.
        input_layout: ``"btc"`` for ``[B, T, C]`` or ``"bct"`` for ``[B, C, T]``.
        use_se: Enable squeeze-and-excitation channel attention.
        use_attention_pool: Use attention pooling (else global average pooling).
        downsample: Halve the time axis between blocks with max pooling.
        use_graph_front: Enable the adaptive cross-sensor graph front-end.
        graph_embed_dim: Node-embedding dimension for the sensor graph.
    """

    def __init__(
        self,
        num_channels: int,
        num_classes: int,
        hidden_channels: int = 96,
        num_blocks: int = 3,
        dropout: float = 0.1,
        input_layout: InputLayout = "btc",
        use_se: bool = True,
        use_attention_pool: bool = True,
        downsample: bool = True,
        use_graph_front: bool = False,
        graph_embed_dim: int = 10,
        kernels: tuple[int, ...] = (9, 19, 39),
        use_residual: bool = True,
        pool_branch_channels: int | None = None,
    ) -> None:
        super().__init__()
        if input_layout not in {"btc", "bct"}:
            raise ValueError("input_layout must be 'btc' or 'bct'")

        self.num_channels = num_channels
        self.num_classes = num_classes
        self.input_layout = input_layout

        self.graph_front = AdaptiveSensorGraph(num_channels, graph_embed_dim) if use_graph_front else None
        self.stem = nn.Sequential(
            nn.Conv1d(num_channels, hidden_channels, kernel_size=7, padding=3, bias=False),
            make_group_norm(hidden_channels),
            nn.GELU(),
        )
        blocks: list[nn.Module] = []
        for index in range(num_blocks):
            blocks.append(
                MultiScaleConvBlock(hidden_channels, hidden_channels, kernels=kernels, dropout=dropout, use_se=use_se,
                                    use_residual=use_residual, pool_branch_channels=pool_branch_channels)
            )
            if downsample and index < num_blocks - 1:
                blocks.append(nn.MaxPool1d(kernel_size=2))
        self.blocks = nn.Sequential(*blocks)
        self.pooling = AttentionPool1d(hidden_channels) if use_attention_pool else None
        self.classifier = nn.Sequential(
            nn.LayerNorm(hidden_channels),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, num_classes),
        )

    def forward(self, x: Tensor) -> Tensor:
        """Compute class logits (without softmax) shaped ``[B, num_classes]``."""
        if x.ndim != 3:
            raise ValueError(f"Expected 3D input, got shape {tuple(x.shape)}")
        if self.input_layout == "btc":
            if x.shape[-1] != self.num_channels:
                raise ValueError(f"Expected last dim C={self.num_channels}, got {x.shape[-1]}")
            x = x.transpose(1, 2)
        elif x.shape[1] != self.num_channels:
            raise ValueError(f"Expected channel dim C={self.num_channels}, got {x.shape[1]}")

        if self.graph_front is not None:
            x = self.graph_front(x)
        features = self.blocks(self.stem(x))
        pooled = self.pooling(features) if self.pooling is not None else features.mean(dim=-1)
        return self.classifier(pooled)
