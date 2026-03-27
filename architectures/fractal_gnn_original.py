"""
Fractal GNN Block — Hierarchical chunk-wise pooling with graph convolution
and gated residual feedback to the token level.

Modules
-------
SimpleGraphConv        – 1-hop message passing on a linear chain of chunk nodes.
DeeperGraphConv        – Stacked SimpleGraphConv with residual connections.
FractalGNNBlock        – Single-scale chunk pool → GNN → gated broadcast.
MultiScaleFractalLayer – Parallel FractalGNNBlocks at multiple chunk sizes,
                         fused back to token resolution.
"""

from __future__ import annotations

import math
from typing import Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Graph convolution on a linear chain
# ---------------------------------------------------------------------------

class SimpleGraphConv(nn.Module):
    """One-hop message passing on a chain graph (kernel-3 1-D convolution).

    Each node aggregates itself and its immediate left / right neighbours
    with equal weight, then applies a linear projection and ReLU.
    Boundary nodes are zero-padded.
    """

    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.lin = nn.Linear(d_model, d_model)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        h : (B, N, D)

        Returns
        -------
        (B, N, D)
        """
        # Shift-right (left neighbour) and shift-left (right neighbour)
        h_left = F.pad(h[:, :-1], (0, 0, 1, 0))   # zero at position 0
        h_right = F.pad(h[:, 1:], (0, 0, 0, 1))    # zero at position N-1
        h_agg = (h + h_left + h_right) / 3.0
        return F.relu(self.lin(h_agg))


class DeeperGraphConv(nn.Module):
    """Stack of :class:`SimpleGraphConv` layers with residual connections.

    After ``depth`` layers the effective receptive field spans
    ``2 * depth + 1`` chunk nodes.
    """

    def __init__(self, d_model: int, depth: int = 3) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [SimpleGraphConv(d_model) for _ in range(depth)]
        )

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            h = h + layer(h)          # residual around each hop
        return h


# ---------------------------------------------------------------------------
# Single-scale Fractal GNN Block
# ---------------------------------------------------------------------------

class FractalGNNBlock(nn.Module):
    """Chunk-pool → GNN → gated residual broadcast.

    1. Pad the sequence to a multiple of *chunk_size* and reshape into chunks.
    2. Mean-pool each chunk (with optional mask to ignore padding tokens).
    3. Apply *gnn_depth* rounds of :class:`SimpleGraphConv` over the chunk
       graph (a linear chain).
    4. Broadcast chunk features back to token positions.
    5. Produce a gated residual update:  ``y = x + sigmoid(gate) * MLP([x; h])``.

    Parameters
    ----------
    d_model : int
        Hidden dimension.
    chunk_size : int
        Number of tokens per chunk.
    gnn_depth : int
        Number of stacked graph-conv layers over the chunk graph.
    dropout : float
        Dropout probability applied inside the MLP and after the gate.
    layer_norm : bool
        If *True*, apply LayerNorm to chunk embeddings before the GNN
        and to the output after the residual update.
    """

    def __init__(
        self,
        d_model: int,
        chunk_size: int = 128,
        gnn_depth: int = 1,
        dropout: float = 0.0,
        layer_norm: bool = False,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.chunk_size = chunk_size

        # --- chunk-level transforms ---
        self.up_proj = nn.Linear(d_model, d_model)
        self.gnns = nn.ModuleList(
            [SimpleGraphConv(d_model) for _ in range(gnn_depth)]
        )

        # --- token-level gated residual ---
        self.down_mlp = nn.Sequential(
            nn.Linear(2 * d_model, 4 * d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4 * d_model, d_model),
            nn.Dropout(dropout),
        )
        self.gate = nn.Linear(2 * d_model, d_model)
        self.drop = nn.Dropout(dropout)

        # --- optional norms ---
        self.chunk_norm: Optional[nn.LayerNorm] = None
        self.out_norm: Optional[nn.LayerNorm] = None
        if layer_norm:
            self.chunk_norm = nn.LayerNorm(d_model)
            self.out_norm = nn.LayerNorm(d_model)

    # --------------------------------------------------------------------- #

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        x : (B, L, D)
            Token embeddings.
        mask : (B, L), optional
            1.0 for real tokens, 0.0 for padding.  If *None*, all tokens
            are assumed valid.

        Returns
        -------
        y : (B, L, D)
            Updated token embeddings (same shape as input).
        """
        B, L, D = x.shape
        c = self.chunk_size
        N = (L + c - 1) // c          # number of chunks (ceil division)
        pad_len = N * c - L

        # ---- pad sequence to a multiple of chunk_size ----
        mask_pad: Optional[torch.Tensor] = None      # FIX: always initialised
        if pad_len > 0:
            x_pad = torch.cat(
                [x, x.new_zeros(B, pad_len, D)], dim=1
            )
            if mask is not None:
                mask_pad = torch.cat(
                    [mask, mask.new_zeros(B, pad_len)], dim=1
                )
        else:
            x_pad = x
            mask_pad = mask

        # ---- reshape into chunks and pool ----
        x_chunks = x_pad.reshape(B, N, c, D)

        if mask_pad is not None:
            m_chunks = mask_pad.reshape(B, N, c, 1)
            denom = m_chunks.sum(dim=2).clamp_min(1.0)
            h_chunk = (x_chunks * m_chunks).sum(dim=2) / denom
        else:
            h_chunk = x_chunks.mean(dim=2)             # (B, N, D)

        # ---- chunk-level processing ----
        h_chunk = self.up_proj(h_chunk)
        if self.chunk_norm is not None:
            h_chunk = self.chunk_norm(h_chunk)

        for gnn in self.gnns:
            h_chunk = h_chunk + gnn(h_chunk)            # residual per hop

        # ---- broadcast back to token positions ----
        h_broadcast = (
            h_chunk
            .unsqueeze(2)
            .expand(B, N, c, D)
            .reshape(B, N * c, D)[:, :L]
        )

        # ---- gated residual update ----
        concat = torch.cat([x, h_broadcast], dim=-1)    # (B, L, 2D)
        gate = torch.sigmoid(self.gate(concat))          # (B, L, D)
        delta = self.down_mlp(concat)                    # (B, L, D)
        y = x + self.drop(gate * delta)

        if self.out_norm is not None:
            y = self.out_norm(y)

        return y


# ---------------------------------------------------------------------------
# Multi-scale layer
# ---------------------------------------------------------------------------

class MultiScaleFractalLayer(nn.Module):
    """Run :class:`FractalGNNBlock` at several chunk sizes in parallel and
    fuse the results with a linear projection.

    Each scale independently pools, graph-convolves, and broadcasts, so
    short-range and long-range chunk structure are captured simultaneously.

    Parameters
    ----------
    d_model : int
        Hidden dimension.
    chunk_sizes : sequence of int
        Chunk sizes to use (e.g. ``(16, 64, 256)``).
    gnn_depth : int
        Passed to each :class:`FractalGNNBlock`.
    dropout : float
        Passed to each :class:`FractalGNNBlock`.
    layer_norm : bool
        Passed to each :class:`FractalGNNBlock`.
    """

    def __init__(
        self,
        d_model: int,
        chunk_sizes: Sequence[int] = (16, 64, 256),
        gnn_depth: int = 1,
        dropout: float = 0.0,
        layer_norm: bool = False,
    ) -> None:
        super().__init__()
        self.blocks = nn.ModuleList(
            [
                FractalGNNBlock(
                    d_model,
                    chunk_size=c,
                    gnn_depth=gnn_depth,
                    dropout=dropout,
                    layer_norm=layer_norm,
                )
                for c in chunk_sizes
            ]
        )
        # fuse: original tokens + one output per scale
        self.fuse = nn.Linear(d_model * (1 + len(chunk_sizes)), d_model)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        x : (B, L, D)
        mask : (B, L), optional

        Returns
        -------
        (B, L, D)
        """
        feats = [x]
        for blk in self.blocks:
            feats.append(blk(x, mask))        # each scale sees the *same* x
        return self.fuse(torch.cat(feats, dim=-1))
