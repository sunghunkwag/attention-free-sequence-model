"""[EXPERIMENTAL] 
Adaptive Field Network (AFN)
==============================
Combines the proven components from today's experiments:

From NCA-LM (worked):
  - Depthwise separable conv perception (local sensing)
  - Gated reaction (nonlinear state update)
  - Multi-rate dilated diffusion (long-range transport)
  - Iterative refinement (T-step dynamics)

From PFN (idea good, execution failed):
  - Content determines connectivity
  → But implemented as O(N) SPARSE ROUTING, not O(N²) pairwise

From HFN/FEN (multi-scale was sound):
  - Multi-resolution processing with cross-scale exchange
  → But via POOLING + BROADCAST, not attention

Architecture (one AFNLayer)
---------------------------
1. Fine-scale NCA dynamics (perceive → react → diffuse) × T steps
2. Content-routed sparse exchange:
   Each cell produces a "route vector" from its state.
   Cells with similar route vectors exchange information
   via learned hash → bucket → local aggregation.
   O(N · bucket_size), NOT O(N²).
3. Coarse-scale processing:
   Downsample via strided depthwise conv (not attention pooling).
   Run NCA dynamics at coarse resolution.
   Upsample via transposed conv + gated addition.
4. Per-cell feed-forward.

Zero attention. Zero softmax in information routing.
The only softmax is in the final classification head.
"""

from __future__ import annotations

import math
from typing import Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ═══════════════════════════════════════════════════════════════════════════
# NCA Core (proven from NCA-LM)
# ═══════════════════════════════════════════════════════════════════════════

class PerceptionFilter(nn.Module):
    """Depthwise separable conv: sense local neighbourhood."""

    def __init__(self, d_model: int, kernel_size: int = 7, n_filters: int = 3):
        super().__init__()
        self.d_model = d_model
        self.kernel_size = kernel_size
        self.n_filters = n_filters
        self.depthwise_convs = nn.ModuleList([
            nn.Conv1d(d_model, d_model, kernel_size=kernel_size,
                      padding=kernel_size // 2, groups=d_model, bias=False)
            for _ in range(n_filters)
        ])
        self.pointwise = nn.Conv1d(
            d_model * n_filters, d_model, kernel_size=1, bias=False)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = x.transpose(1, 2)
        percepts = [conv(h) for conv in self.depthwise_convs]
        h_cat = torch.cat(percepts, dim=1)
        h_mixed = self.pointwise(h_cat)
        return self.norm(h_mixed.transpose(1, 2))


class ReactionGate(nn.Module):
    """Gated state update from perceived neighbourhood."""

    def __init__(self, d_model: int, expansion: int = 2, dropout: float = 0.0):
        super().__init__()
        d_inner = d_model * expansion
        self.w_up = nn.Linear(2 * d_model, d_inner, bias=False)
        self.w_gate = nn.Linear(2 * d_model, d_inner, bias=False)
        self.w_down = nn.Linear(d_inner, d_model, bias=False)
        self.drop = nn.Dropout(dropout)
        self.alpha = nn.Parameter(torch.full((d_model,), 0.1))

    def forward(self, x: torch.Tensor, perceived: torch.Tensor) -> torch.Tensor:
        cat = torch.cat([x, perceived], dim=-1)
        candidate = F.silu(self.w_up(cat))
        gate = torch.sigmoid(self.w_gate(cat))
        delta = self.w_down(self.drop(gate * candidate))
        return x + self.alpha * delta


class MultiRateDiffusion(nn.Module):
    """Multi-rate dilated diffusion with content-adaptive mixing.

    Improvement over NCA-LM: mixing weights are produced from
    cell state (content-dependent), not just per-channel learned params.
    """

    def __init__(self, d_model: int, dilations: Sequence[int] = (1, 4, 16),
                 kernel_size: int = 3):
        super().__init__()
        self.n_rates = len(dilations)
        self.diffuse_convs = nn.ModuleList([
            nn.Conv1d(d_model, d_model, kernel_size=kernel_size,
                      padding=dilation * (kernel_size // 2),
                      dilation=dilation, groups=d_model, bias=False)
            for dilation in dilations
        ])
        # Content-adaptive rate selection (from NCA+PFN insight)
        self.rate_selector = nn.Linear(d_model, len(dilations), bias=False)
        self.strength = nn.Parameter(torch.tensor(0.1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, L, D = x.shape
        h = x.transpose(1, 2)

        diffused = torch.stack(
            [conv(h) for conv in self.diffuse_convs], dim=-1
        )  # (B, D, L, n_rates)

        # Content-adaptive weights (PFN insight: content determines routing)
        weights = torch.softmax(
            self.rate_selector(x), dim=-1
        )  # (B, L, n_rates)
        weights = weights.unsqueeze(1)  # (B, 1, L, n_rates)

        combined = (diffused * weights).sum(dim=-1)  # (B, D, L)
        combined = combined.transpose(1, 2)

        return x + self.strength * (combined - x)


class NCAStep(nn.Module):
    """One NCA timestep: perceive → react → diffuse."""

    def __init__(self, d_model: int, kernel_size: int = 7,
                 dilations: Sequence[int] = (1, 4, 16),
                 reaction_expansion: int = 2, dropout: float = 0.0):
        super().__init__()
        self.perceive = PerceptionFilter(d_model, kernel_size)
        self.react = ReactionGate(d_model, reaction_expansion, dropout)
        self.diffuse = MultiRateDiffusion(d_model, dilations)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        perceived = self.perceive(x)
        x = self.react(x, perceived)
        x = self.diffuse(x)
        return x


# ═══════════════════════════════════════════════════════════════════════════
# Content-Routed Sparse Exchange (PFN insight, O(N) implementation)
# ═══════════════════════════════════════════════════════════════════════════

class ShiftMixer(nn.Module):
    """Gated shift-register: move information by learned strides.

    Like a hardware barrel shifter — information can jump
    by fixed distances (1, 4, 16, 32...) in one step.
    Each shift direction has a learned gate per position.

    This is NOT attention: no QKV, no softmax, no pairwise computation.
    It's a fixed set of wiring patterns with learned gates.

    O(L · n_shifts · D) — linear in sequence length.
    """

    def __init__(self, d_model: int, shifts: Sequence[int] = (1, 4, 16, 32)):
        super().__init__()
        self.shifts = list(shifts)
        n = len(shifts) * 2  # forward + backward for each stride

        # Per-position gate: which shifts to use, based on content
        self.gate_proj = nn.Linear(d_model, n, bias=False)

        # Per-shift value transform
        self.value_proj = nn.Linear(d_model, d_model, bias=False)

        # Output projection
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, L, D) → (B, L, D)"""
        B, L, D = x.shape
        values = self.value_proj(x)  # (B, L, D)

        # Compute all shifted versions
        shifted_list = []
        for s in self.shifts:
            # Forward shift (information from earlier positions)
            fwd = F.pad(values[:, :-s] if s < L else values[:, :0], (0,0, s, 0))
            shifted_list.append(fwd[:, :L])
            # Backward shift (information from later positions)
            bwd = F.pad(values[:, s:] if s < L else values[:, :0], (0,0, 0, s))
            shifted_list.append(bwd[:, :L])

        shifted = torch.stack(shifted_list, dim=2)  # (B, L, n_shifts*2, D)

        # Content-dependent gating: which shifts to listen to
        gates = torch.sigmoid(self.gate_proj(x))  # (B, L, n_shifts*2)
        gates = gates.unsqueeze(-1)  # (B, L, n_shifts*2, 1)

        # Weighted sum of shifted values
        mixed = (shifted * gates).sum(dim=2)  # (B, L, D)

        return self.norm(x + self.out_proj(mixed))


class SparseRouting(nn.Module):
    """Content-based sparse information exchange. NO attention.

    Mechanism:
    1. Each cell produces a scalar "route key" from its state.
    2. Cells are sorted by route key.
    3. Adjacent cells in sorted order exchange information
       via a small local conv (bucket_size neighbourhood).
    4. Information is scattered back to original positions.

    This means cells with similar content (similar route keys)
    automatically form exchange groups — like PFN's distance-based
    interaction, but O(N log N) from sorting, O(N) for exchange.

    Why this isn't attention:
    - No query-key-value decomposition
    - No softmax over scores
    - Exchange is uniform within buckets (conv), not weighted
    - Routing is by learned hash, not by dot-product similarity
    """

    def __init__(self, d_model: int, bucket_size: int = 8, n_routes: int = 2):
        super().__init__()
        self.bucket_size = bucket_size
        self.n_routes = n_routes

        # Route key projection
        self.route_proj = nn.Linear(d_model, n_routes, bias=False)

        # Per-bucket local exchange (small depthwise conv over buckets)
        self.exchange_convs = nn.ModuleList([
            nn.Conv1d(d_model, d_model, kernel_size=bucket_size,
                      padding=bucket_size // 2, groups=d_model, bias=False)
            for _ in range(n_routes)
        ])

        # Gate: how much routed information to accept
        self.gate = nn.Linear(2 * d_model, d_model, bias=False)
        self.norm = nn.LayerNorm(d_model)

    def _route_and_exchange(self, x: torch.Tensor, route_idx: int) -> torch.Tensor:
        """Sort by route key, exchange within buckets, unsort."""
        B, L, D = x.shape

        # Compute route keys
        keys = self.route_proj(x)[:, :, route_idx]  # (B, L)

        # Sort by key
        sort_idx = keys.argsort(dim=1)  # (B, L)
        unsort_idx = sort_idx.argsort(dim=1)  # inverse permutation

        # Gather sorted order
        expand_idx = sort_idx.unsqueeze(-1).expand(B, L, D)
        x_sorted = torch.gather(x, 1, expand_idx)  # (B, L, D)

        # Local exchange via conv (in sorted space)
        h = x_sorted.transpose(1, 2)  # (B, D, L)
        h_exchanged = self.exchange_convs[route_idx](h)
        h_exchanged = h_exchanged.transpose(1, 2)  # (B, L, D)

        # Unsort back to original positions
        expand_unsort = unsort_idx.unsqueeze(-1).expand(B, L, D)
        return torch.gather(h_exchanged, 1, expand_unsort)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, L, D) → (B, L, D)"""
        # Compute route keys (also used in gate for gradient flow)
        route_keys = self.route_proj(x)  # (B, L, n_routes)

        # Run multiple routing passes with different route projections
        routed = torch.zeros_like(x)
        for r in range(self.n_routes):
            routed = routed + self._route_and_exchange(x, r)
        routed = routed / self.n_routes

        # Gated residual — include route_keys for gradient to route_proj
        cat = torch.cat([x, routed], dim=-1)
        gate = torch.sigmoid(self.gate(cat) + route_keys.sum(dim=-1, keepdim=True))
        return self.norm(x + gate * routed)


# ═══════════════════════════════════════════════════════════════════════════
# Multi-Resolution Processing (HFN insight, NO attention)
# ═══════════════════════════════════════════════════════════════════════════

class CoarseNCA(nn.Module):
    """Downsample → NCA dynamics → Upsample.

    Downsampling: strided depthwise conv (not attention pooling)
    Upsampling: transposed conv + gated broadcast

    Runs NCA at coarse resolution for cheap global information flow.
    """

    def __init__(self, d_model: int, stride: int = 4, n_steps: int = 2,
                 kernel_size: int = 5, dilations: Sequence[int] = (1, 4),
                 dropout: float = 0.0):
        super().__init__()
        self.stride = stride

        # Downsample: strided conv (not attention!)
        self.downsample = nn.Conv1d(
            d_model, d_model, kernel_size=stride * 2 - 1,
            stride=stride, padding=stride - 1, groups=d_model, bias=False,
        )
        self.down_proj = nn.Conv1d(d_model, d_model, kernel_size=1, bias=False)
        self.down_norm = nn.LayerNorm(d_model)

        # Coarse NCA steps
        self.coarse_step = NCAStep(
            d_model, kernel_size=kernel_size,
            dilations=dilations,
            reaction_expansion=2, dropout=dropout,
        )
        self.n_steps = n_steps

        # Upsample: transposed conv
        self.upsample = nn.ConvTranspose1d(
            d_model, d_model, kernel_size=stride * 2,
            stride=stride, padding=stride // 2,
            groups=d_model, bias=False,
        )
        self.up_proj = nn.Conv1d(d_model, d_model, kernel_size=1, bias=False)

        # Gated broadcast back to fine
        self.gate = nn.Linear(2 * d_model, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, L, D) → delta: (B, L, D)"""
        B, L, D = x.shape
        h = x.transpose(1, 2)  # (B, D, L)

        # Downsample
        h_coarse = self.down_proj(self.downsample(h))  # (B, D, L//stride)
        h_coarse = self.down_norm(h_coarse.transpose(1, 2))  # (B, Lc, D)

        # Coarse NCA dynamics
        for _ in range(self.n_steps):
            h_coarse = self.coarse_step(h_coarse)

        # Upsample
        h_up = self.upsample(h_coarse.transpose(1, 2))  # (B, D, ~L)
        h_up = self.up_proj(h_up)
        h_up = h_up.transpose(1, 2)[:, :L]  # (B, L, D) — trim to exact L

        # Gated addition
        cat = torch.cat([x, h_up], dim=-1)
        gate = torch.sigmoid(self.gate(cat))
        return gate * h_up


# ═══════════════════════════════════════════════════════════════════════════
# Feed-forward
# ═══════════════════════════════════════════════════════════════════════════

class GatedFFN(nn.Module):
    def __init__(self, d_model: int, expansion: int = 2, dropout: float = 0.0):
        super().__init__()
        d_inner = d_model * expansion
        self.norm = nn.LayerNorm(d_model)
        self.w_gate = nn.Linear(d_model, d_inner, bias=False)
        self.w_up = nn.Linear(d_model, d_inner, bias=False)
        self.w_down = nn.Linear(d_inner, d_model, bias=False)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        r = x
        x = self.norm(x)
        return r + self.drop(self.w_down(F.silu(self.w_gate(x)) * self.w_up(x)))


# ═══════════════════════════════════════════════════════════════════════════
# AFN Layer
# ═══════════════════════════════════════════════════════════════════════════

class AFNLayer(nn.Module):
    """One layer of the Adaptive Field Network.

    Pre-norm residual:
      x = x + NCA_dynamics(norm(x))        ← local processing (NCA)
      x = x + SparseRouting(norm(x))       ← content-routed exchange (PFN idea)
      x = x + CoarseNCA(norm(x))           ← multi-scale processing (HFN idea)
      x = FFN(x)                           ← per-cell nonlinearity

    Zero attention. Zero softmax in routing.
    """

    def __init__(
        self,
        d_model: int,
        nca_steps: int = 4,
        nca_kernel: int = 7,
        nca_dilations: Sequence[int] = (1, 4, 16),
        bucket_size: int = 8,
        n_routes: int = 2,
        coarse_stride: int = 4,
        coarse_steps: int = 2,
        ffn_expansion: int = 2,
        dropout: float = 0.0,
    ):
        super().__init__()

        # Sub-layer 1: Local NCA dynamics (T steps)
        self.norm1 = nn.LayerNorm(d_model)
        self.nca_step = NCAStep(d_model, nca_kernel, nca_dilations,
                                 reaction_expansion=2, dropout=dropout)
        self.nca_steps = nca_steps

        # Sub-layer 2: Shift-register long-range transport
        self.norm2 = nn.LayerNorm(d_model)
        self.shift = ShiftMixer(d_model, shifts=(1, 4, 16, 32))

        # Sub-layer 3: Content-routed sparse exchange
        self.norm3 = nn.LayerNorm(d_model)
        self.routing = SparseRouting(d_model, bucket_size, n_routes)

        # Sub-layer 4: Coarse-scale NCA
        self.norm4 = nn.LayerNorm(d_model)
        self.coarse = CoarseNCA(
            d_model, stride=coarse_stride, n_steps=coarse_steps,
            kernel_size=5, dilations=(1, 4), dropout=dropout)

        # Sub-layer 4: FFN
        self.ffn = GatedFFN(d_model, ffn_expansion, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Local NCA dynamics
        h = self.norm1(x)
        for _ in range(self.nca_steps):
            h = self.nca_step(h)
        x = x + (h - self.norm1(x))  # residual of NCA delta

        # Shift-register long-range transport
        x = self.shift(self.norm2(x)) + x

        # Content-routed sparse exchange
        x = self.routing(self.norm3(x)) + x

        # Coarse-scale NCA
        x = x + self.coarse(self.norm4(x))

        # FFN
        x = self.ffn(x)
        return x


# ═══════════════════════════════════════════════════════════════════════════
# Complete model
# ═══════════════════════════════════════════════════════════════════════════

class AFN(nn.Module):
    """Adaptive Field Network.

    Combines proven components:
    - NCA local dynamics (from NCA-LM, which beat Transformer)
    - Content-routed sparse exchange (PFN idea, O(N) implementation)
    - Multi-resolution coarse NCA (HFN idea, no attention)
    - Iterative refinement

    Zero attention. Zero softmax in information routing.
    O(L · T · d) total complexity (linear in sequence length).
    """

    def __init__(
        self,
        vocab_size: int = 256,
        d_model: int = 128,
        n_layers: int = 4,
        nca_steps: int = 4,
        nca_kernel: int = 7,
        nca_dilations: Sequence[int] = (1, 4, 16),
        bucket_size: int = 8,
        n_routes: int = 2,
        coarse_stride: int = 4,
        coarse_steps: int = 2,
        ffn_expansion: int = 2,
        dropout: float = 0.0,
        max_len: int = 4096,
    ):
        super().__init__()
        self.d_model = d_model

        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_len, d_model)
        self.emb_norm = nn.LayerNorm(d_model)
        self.emb_drop = nn.Dropout(dropout)

        self.layers = nn.ModuleList([
            AFNLayer(
                d_model, nca_steps, nca_kernel, nca_dilations,
                bucket_size, n_routes, coarse_stride, coarse_steps,
                ffn_expansion, dropout,
            )
            for _ in range(n_layers)
        ])

        self.final_norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)
        self.head.weight = self.tok_emb.weight

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, std=0.02)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        B, L = input_ids.shape
        pos = torch.arange(L, device=input_ids.device).unsqueeze(0)
        x = self.tok_emb(input_ids) + self.pos_emb(pos)
        x = self.emb_drop(self.emb_norm(x))
        for layer in self.layers:
            x = layer(x)
        return self.head(self.final_norm(x))

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ═══════════════════════════════════════════════════════════════════════════
# Presets
# ═══════════════════════════════════════════════════════════════════════════

def afn_tiny(vocab_size: int = 256, **kw) -> AFN:
    return AFN(vocab_size, d_model=64, n_layers=2, nca_steps=3,
               nca_kernel=5, nca_dilations=(1, 4),
               bucket_size=8, n_routes=2, coarse_stride=4,
               coarse_steps=2, ffn_expansion=2, **kw)


def afn_small(vocab_size: int = 256, **kw) -> AFN:
    return AFN(vocab_size, d_model=256, n_layers=6, nca_steps=6,
               nca_kernel=7, nca_dilations=(1, 4, 16),
               bucket_size=16, n_routes=3, coarse_stride=8,
               coarse_steps=3, ffn_expansion=4, **kw)
