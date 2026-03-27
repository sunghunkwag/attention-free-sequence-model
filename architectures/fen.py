"""
Fractal Equilibrium Network (FEN)
==================================
Iterative cross-scale sequence model inspired by multi-grid V-cycle solvers.

Design thesis
-------------
Multi-grid V-cycles are the fastest known algorithms for elliptic PDEs.
They iterate: local refinement at fine scale + global propagation at coarse
scale, alternating until convergence.  No existing sequence model applies
this iterative cross-scale equilibrium pattern.

Transformer  = O(L²) global attention, single resolution, one pass.
Mamba        = O(L)  selective recurrence, single resolution, one pass.
HFN          = O(L)  multi-scale, but one-shot independent processing.
FEN          = O(L·T) multi-scale, iterative V-cycle cross-scale equilibrium.

Architecture (one FENLayer)
---------------------------
1. Maintain representations at K resolution levels simultaneously:
     r₀ = token-level (L positions)
     r₁ = chunk-level (L/c₁ positions)
     r₂ = chunk-level (L/c₂ positions)
     ...
2. Execute V-cycle iterations (fine→coarse→fine):
     a. Smooth r₀ (local attention within chunks)
     b. Restrict r₀ → r₁ (attention pooling)
     c. Smooth r₁ (full attention over chunks — cheap, N₁ is small)
     d. Restrict r₁ → r₂
     e. Smooth r₂ (full attention)
     f. Prolongate r₂ → r₁ (gated broadcast + add)
     g. Smooth r₁
     h. Prolongate r₁ → r₀ (gated broadcast + add)
     i. Smooth r₀
3. Adaptive halting: per-token halting probability decides iteration count.
4. Feed-forward (SwiGLU).

Structural novelty
------------------
- Cross-scale ITERATION (not one-shot parallel fusion)
- V-cycle pattern borrowed from numerical PDE solvers
- Adaptive computation time per token
- Content-adaptive chunk attention (not fixed chain GNN)

Complexity: O(L · T_avg · (c + sum_k(N_k²))) where T_avg = mean iterations,
N_k = L/c_k = number of chunks at scale k (small constant).

Modules
-------
RotaryEmbedding         – RoPE for local attention.
LocalSmoother           – Within-chunk attention (fine-scale smoothing).
ChunkAttention          – Full attention over chunk nodes (coarse smoothing).
Restrictor              – Fine→coarse: attention-weighted pooling.
Prolongator             – Coarse→fine: gated broadcast.
VCycle                  – One complete V-cycle: smooth-restrict-...-prolongate-smooth.
AdaptiveVCycle          – V-cycle with ACT halting.
SwiGLU                  – Gated feed-forward.
FENLayer                – One full layer: adaptive V-cycle + FFN.
FEN                     – Complete model: embed → N×FENLayer → head.
"""

from __future__ import annotations

import math
from typing import Optional, Sequence, Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F


# ═══════════════════════════════════════════════════════════════════════════
# Positional encoding (shared with HFN)
# ═══════════════════════════════════════════════════════════════════════════

class RotaryEmbedding(nn.Module):
    """Rotary Position Embedding (Su et al., 2021)."""

    def __init__(self, dim: int, max_len: int = 8192, base: float = 10000.0):
        super().__init__()
        self.dim = dim
        self.base = base
        self._build_cache(max_len)

    def _build_cache(self, seq_len: int) -> None:
        inv_freq = 1.0 / (
            self.base ** (torch.arange(0, self.dim, 2).float() / self.dim)
        )
        t = torch.arange(seq_len).float()
        freqs = torch.outer(t, inv_freq)
        self.register_buffer("cos_cached", freqs.cos(), persistent=False)
        self.register_buffer("sin_cached", freqs.sin(), persistent=False)

    def forward(self, seq_len: int) -> Tuple[torch.Tensor, torch.Tensor]:
        if seq_len > self.cos_cached.size(0):
            self._build_cache(seq_len)
        return self.cos_cached[:seq_len], self.sin_cached[:seq_len]


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([-x2, x1], dim=-1)


# ═══════════════════════════════════════════════════════════════════════════
# Fine-scale smoother: local chunk attention
# ═══════════════════════════════════════════════════════════════════════════

class LocalSmoother(nn.Module):
    """Within-chunk self-attention (fine-scale relaxation).

    Analogous to Jacobi/Gauss-Seidel smoothing in multigrid:
    reduces high-frequency error components locally.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int = 4,
        chunk_size: int = 32,
        dropout: float = 0.0,
    ):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.chunk_size = chunk_size
        self.scale = self.d_head ** -0.5

        self.norm = nn.LayerNorm(d_model)
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        self.rope = RotaryEmbedding(self.d_head)
        self.attn_drop = nn.Dropout(dropout)

    def forward(
        self, x: torch.Tensor, mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """x: (B, L, D) → (B, L, D).  Pre-norm residual."""
        B, L, D = x.shape
        c = self.chunk_size
        residual = x
        x = self.norm(x)

        N = (L + c - 1) // c
        pad_len = N * c - L

        if pad_len > 0:
            x = F.pad(x, (0, 0, 0, pad_len))
            if mask is not None:
                mask = F.pad(mask, (0, pad_len), value=0.0)
        Lp = N * c

        qkv = self.qkv(x).reshape(B, Lp, 3, self.n_heads, self.d_head)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)

        # RoPE per chunk
        cos, sin = self.rope(c)
        q_c = q.reshape(B, self.n_heads, N, c, self.d_head)
        k_c = k.reshape(B, self.n_heads, N, c, self.d_head)
        v_c = v.reshape(B, self.n_heads, N, c, self.d_head)

        cos_e = cos.unsqueeze(0).unsqueeze(0).unsqueeze(0).repeat(1, 1, 1, 1, 2)
        sin_e = sin.unsqueeze(0).unsqueeze(0).unsqueeze(0).repeat(1, 1, 1, 1, 2)
        q_c = q_c * cos_e + _rotate_half(q_c) * sin_e
        k_c = k_c * cos_e + _rotate_half(k_c) * sin_e

        attn = torch.einsum("bhnid,bhnjd->bhnij", q_c, k_c) * self.scale
        if mask is not None:
            m_c = mask.reshape(B, 1, N, c)
            attn = attn.masked_fill(m_c.unsqueeze(3) == 0, float("-inf"))
        attn = F.softmax(attn, dim=-1)
        attn = torch.nan_to_num(attn, nan=0.0)
        attn = self.attn_drop(attn)

        out = torch.einsum("bhnij,bhnjd->bhnid", attn, v_c)
        out = out.reshape(B, self.n_heads, Lp, self.d_head)
        out = out.permute(0, 2, 1, 3).reshape(B, Lp, D)
        out = self.out_proj(out)[:, :L]

        return residual + out


# ═══════════════════════════════════════════════════════════════════════════
# Coarse-scale smoother: full attention over chunks
# ═══════════════════════════════════════════════════════════════════════════

class ChunkAttention(nn.Module):
    """Full self-attention over chunk-level nodes (coarse-scale smoothing).

    Since N_chunks = L/chunk_size is small (e.g. 16-256),
    full O(N²) attention is cheap and gives global receptive field.
    """

    def __init__(self, d_model: int, n_heads: int = 4, dropout: float = 0.0):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.scale = self.d_head ** -0.5

        self.norm = nn.LayerNorm(d_model)
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        self.drop = nn.Dropout(dropout)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """h: (B, N, D) → (B, N, D).  Pre-norm residual."""
        B, N, D = h.shape
        residual = h
        h = self.norm(h)

        qkv = self.qkv(h).reshape(B, N, 3, self.n_heads, self.d_head)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)

        attn = torch.einsum("bhid,bhjd->bhij", q, k) * self.scale
        attn = F.softmax(attn, dim=-1)
        attn = self.drop(attn)

        out = torch.einsum("bhij,bhjd->bhid", attn, v)
        out = out.transpose(1, 2).reshape(B, N, D)
        return residual + self.out_proj(out)


# ═══════════════════════════════════════════════════════════════════════════
# Restrictor: fine → coarse (attention pooling)
# ═══════════════════════════════════════════════════════════════════════════

class Restrictor(nn.Module):
    """Restriction operator: fine-level → coarse-level via attention pooling.

    Analogous to the restriction operator R in multigrid (weighted averaging
    of fine grid values to coarse grid points).
    """

    def __init__(self, d_model: int, chunk_size: int):
        super().__init__()
        self.chunk_size = chunk_size
        self.pool_query = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.pool_kv = nn.Linear(d_model, 2 * d_model, bias=False)
        self.scale = d_model ** -0.5
        self.norm = nn.LayerNorm(d_model)

    def forward(
        self, x: torch.Tensor, mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, int, int]:
        """
        x    : (B, L, D) fine-level representation
        mask : (B, L) or None

        Returns
        -------
        h_coarse : (B, N, D)   coarse-level representation
        N        : int          number of chunks
        pad_len  : int          padding applied
        """
        B, L, D = x.shape
        c = self.chunk_size
        N = (L + c - 1) // c
        pad_len = N * c - L

        if pad_len > 0:
            x_pad = F.pad(x, (0, 0, 0, pad_len))
            mask_pad = (
                F.pad(mask, (0, pad_len), value=0.0)
                if mask is not None else None
            )
        else:
            x_pad = x
            mask_pad = mask

        x_chunks = x_pad.reshape(B, N, c, D)
        query = self.pool_query.expand(B, N, D)
        kv = self.pool_kv(x_chunks)
        k, v = kv.chunk(2, dim=-1)

        attn = torch.einsum("bnd,bncd->bnc", query, k) * self.scale
        if mask_pad is not None:
            m_c = mask_pad.reshape(B, N, c)
            attn = attn.masked_fill(m_c == 0, float("-inf"))
        attn = F.softmax(attn, dim=-1)
        attn = torch.nan_to_num(attn, nan=0.0)

        h = torch.einsum("bnc,bncd->bnd", attn, v)
        return self.norm(h), N, pad_len


# ═══════════════════════════════════════════════════════════════════════════
# Prolongator: coarse → fine (gated broadcast)
# ═══════════════════════════════════════════════════════════════════════════

class Prolongator(nn.Module):
    """Prolongation operator: coarse-level → fine-level via gated broadcast.

    Analogous to the interpolation operator P in multigrid
    (mapping coarse corrections back to the fine grid).
    """

    def __init__(self, d_model: int, chunk_size: int):
        super().__init__()
        self.chunk_size = chunk_size
        self.gate_proj = nn.Linear(2 * d_model, d_model)
        self.value_proj = nn.Linear(d_model, d_model)
        self.norm = nn.LayerNorm(d_model)

    def forward(
        self,
        x_fine: torch.Tensor,       # (B, L, D) current fine repr
        h_coarse: torch.Tensor,      # (B, N, D) coarse repr
        L_orig: int,                 # original length before padding
    ) -> torch.Tensor:
        """Returns updated fine representation (B, L_orig, D)."""
        B, N, D = h_coarse.shape
        c = self.chunk_size

        # Broadcast coarse → fine resolution
        h_broad = (
            h_coarse.unsqueeze(2)
            .expand(B, N, c, D)
            .reshape(B, N * c, D)[:, :L_orig]
        )

        # Gated addition (coarse correction to fine grid)
        cat = torch.cat([x_fine, h_broad], dim=-1)
        gate = torch.sigmoid(self.gate_proj(cat))
        correction = self.value_proj(h_coarse.unsqueeze(2)
                                     .expand(B, N, c, D)
                                     .reshape(B, N * c, D)[:, :L_orig])
        return self.norm(x_fine + gate * correction)


# ═══════════════════════════════════════════════════════════════════════════
# V-Cycle: one complete cycle across all scales
# ═══════════════════════════════════════════════════════════════════════════

class VCycle(nn.Module):
    """One V-cycle: fine → coarse → fine with smoothing at each level.

    For K scales with chunk_sizes [c₁, c₂, ...]:
      1. Smooth at fine (r₀)
      2. For each scale k = 1..K:
           Restrict(r_{k-1} → r_k)
           Smooth(r_k)
      3. For each scale k = K-1..0:
           Prolongate(r_{k+1} → r_k)
           Smooth(r_k)
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int = 4,
        fine_chunk_size: int = 32,
        coarse_chunk_sizes: Sequence[int] = (64, 256),
        dropout: float = 0.0,
    ):
        super().__init__()
        self.fine_chunk_size = fine_chunk_size
        self.coarse_chunk_sizes = list(coarse_chunk_sizes)
        n_coarse = len(coarse_chunk_sizes)

        # Fine-level smoother (pre and post)
        self.pre_smooth = LocalSmoother(
            d_model, n_heads, fine_chunk_size, dropout)
        self.post_smooth = LocalSmoother(
            d_model, n_heads, fine_chunk_size, dropout)

        # Coarse-level components (one per coarse scale)
        self.restrictors = nn.ModuleList()
        self.coarse_smoothers_down = nn.ModuleList()  # descending path
        self.coarse_smoothers_up = nn.ModuleList()    # ascending path
        self.prolongators = nn.ModuleList()

        # Build restriction chain: fine → c₁ → c₂ → ...
        # Each restrictor pools from the previous level's resolution
        prev_chunk = fine_chunk_size
        for cs in coarse_chunk_sizes:
            # Relative chunk size: how many positions at prev level → 1 at this
            rel_chunk = max(1, cs // prev_chunk)
            self.restrictors.append(Restrictor(d_model, rel_chunk))
            self.coarse_smoothers_down.append(
                ChunkAttention(d_model, n_heads, dropout))
            self.coarse_smoothers_up.append(
                ChunkAttention(d_model, n_heads, dropout))
            self.prolongators.append(Prolongator(d_model, rel_chunk))
            prev_chunk = cs

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """x: (B, L, D) → (B, L, D)"""
        B, L, D = x.shape

        # ---- Pre-smooth at fine level ----
        x = self.pre_smooth(x, mask)

        # ---- Descending path: restrict to coarser levels ----
        coarse_reprs: List[torch.Tensor] = []
        coarse_lens: List[int] = []
        current = x

        for i, (restrictor, smoother) in enumerate(
            zip(self.restrictors, self.coarse_smoothers_down)
        ):
            h_coarse, N, _ = restrictor(current, mask=None)
            h_coarse = smoother(h_coarse)
            coarse_reprs.append(h_coarse)
            coarse_lens.append(current.size(1))
            current = h_coarse

        # ---- Ascending path: post-smooth then prolongate at each level ----
        for i in range(len(self.coarse_chunk_sizes) - 1, -1, -1):
            # Post-smooth at this coarse level
            coarse_reprs[i] = self.coarse_smoothers_up[i](coarse_reprs[i])

            if i > 0:
                # Prolongate to next finer coarse level
                coarse_reprs[i - 1] = self.prolongators[i](
                    coarse_reprs[i - 1], coarse_reprs[i],
                    coarse_reprs[i - 1].size(1),
                )
            else:
                # Prolongate to fine level
                x = self.prolongators[0](x, coarse_reprs[0], L)

        # ---- Post-smooth at fine level ----
        x = self.post_smooth(x, mask)

        return x


# ═══════════════════════════════════════════════════════════════════════════
# Adaptive V-Cycle with ACT (Adaptive Computation Time)
# ═══════════════════════════════════════════════════════════════════════════

class AdaptiveVCycle(nn.Module):
    """V-cycle with per-token adaptive halting (Graves 2016).

    Each token accumulates halting probability across iterations.
    When cumulative p > 1 - ε, the token stops updating.
    The output is a weighted average of states across iterations.

    Parameters
    ----------
    d_model, n_heads, fine_chunk_size, coarse_chunk_sizes, dropout :
        Passed to VCycle.
    max_iterations : int
        Maximum V-cycle iterations.
    share_weights : bool
        If True, all iterations share the same VCycle parameters
        (like DEQ / Universal Transformer). If False, separate params.
    halting_threshold : float
        Cumulative halting threshold (typically 1 - ε).
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int = 4,
        fine_chunk_size: int = 32,
        coarse_chunk_sizes: Sequence[int] = (64, 256),
        dropout: float = 0.0,
        max_iterations: int = 4,
        share_weights: bool = True,
        halting_threshold: float = 0.99,
    ):
        super().__init__()
        self.max_iterations = max_iterations
        self.halting_threshold = halting_threshold
        self.share_weights = share_weights

        if share_weights:
            self.vcycle = VCycle(
                d_model, n_heads, fine_chunk_size,
                coarse_chunk_sizes, dropout)
        else:
            self.vcycles = nn.ModuleList([
                VCycle(d_model, n_heads, fine_chunk_size,
                       coarse_chunk_sizes, dropout)
                for _ in range(max_iterations)
            ])

        # Halting probability predictor
        self.halt_proj = nn.Linear(d_model, 1)

    def _get_vcycle(self, iteration: int) -> VCycle:
        if self.share_weights:
            return self.vcycle
        return self.vcycles[iteration]

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """x: (B, L, D) → (B, L, D)"""
        B, L, D = x.shape
        device = x.device

        # ACT accumulators
        halting_prob = torch.zeros(B, L, 1, device=device)
        remainders = torch.zeros(B, L, 1, device=device)
        n_updates = torch.zeros(B, L, 1, device=device)
        output_accum = torch.zeros(B, L, D, device=device)

        # Which tokens are still active
        still_running = torch.ones(B, L, 1, device=device, dtype=torch.bool)

        state = x

        for t in range(self.max_iterations):
            # Run one V-cycle
            state = self._get_vcycle(t)(state, mask)

            # Compute halting probability for this iteration
            p = torch.sigmoid(self.halt_proj(state))           # (B, L, 1)

            # Determine which tokens halt at this step
            new_halted = (
                still_running
                & ((halting_prob + p) > self.halting_threshold)
            )

            # Last iteration: force halt for remaining tokens
            if t == self.max_iterations - 1:
                new_halted = still_running

            # Compute update weights
            still_running_float = still_running.float()

            # For newly halted tokens: remainder weight
            halt_weight = torch.where(
                new_halted,
                1.0 - halting_prob,                            # remainder
                p * still_running_float,                       # normal weight
            )

            # Accumulate
            output_accum = output_accum + halt_weight * state
            halting_prob = halting_prob + p * still_running_float
            n_updates = n_updates + still_running_float

            # Update running mask
            still_running = still_running & ~new_halted

            # Early exit if all tokens halted
            if not still_running.any():
                break

        return output_accum


# ═══════════════════════════════════════════════════════════════════════════
# Feed-forward (SwiGLU)
# ═══════════════════════════════════════════════════════════════════════════

class SwiGLU(nn.Module):
    """SwiGLU feed-forward (Shazeer 2020)."""

    def __init__(self, d_model: int, expansion: int = 4, dropout: float = 0.0):
        super().__init__()
        d_inner = d_model * expansion
        self.norm = nn.LayerNorm(d_model)
        self.w_gate = nn.Linear(d_model, d_inner, bias=False)
        self.w_up = nn.Linear(d_model, d_inner, bias=False)
        self.w_down = nn.Linear(d_inner, d_model, bias=False)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.norm(x)
        return residual + self.drop(
            self.w_down(F.silu(self.w_gate(x)) * self.w_up(x))
        )


# ═══════════════════════════════════════════════════════════════════════════
# FEN Layer
# ═══════════════════════════════════════════════════════════════════════════

class FENLayer(nn.Module):
    """One layer of the Fractal Equilibrium Network.

    Pre-norm residual:
        x = AdaptiveVCycle(x)   ← iterative multi-scale processing
        x = SwiGLU(x)           ← per-token nonlinearity
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int = 4,
        fine_chunk_size: int = 32,
        coarse_chunk_sizes: Sequence[int] = (64, 256),
        dropout: float = 0.0,
        max_iterations: int = 4,
        share_weights: bool = True,
        ffn_expansion: int = 4,
    ):
        super().__init__()
        self.vcycle = AdaptiveVCycle(
            d_model, n_heads, fine_chunk_size,
            coarse_chunk_sizes, dropout,
            max_iterations, share_weights,
        )
        self.ffn = SwiGLU(d_model, ffn_expansion, dropout)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        x = self.vcycle(x, mask)
        x = self.ffn(x)
        return x


# ═══════════════════════════════════════════════════════════════════════════
# Complete model
# ═══════════════════════════════════════════════════════════════════════════

class FEN(nn.Module):
    """Fractal Equilibrium Network — complete sequence model.

    Parameters
    ----------
    vocab_size : int
    d_model : int
    n_layers : int
        Number of FENLayers.
    n_heads : int
    fine_chunk_size : int
        Window for fine-level local attention.
    coarse_chunk_sizes : sequence of int
        Chunk sizes for coarse levels in V-cycle.
    max_iterations : int
        Maximum V-cycle iterations per layer.
    share_weights : bool
        Share V-cycle weights across iterations (DEQ-like).
    ffn_expansion : int
    dropout : float
    max_len : int
    """

    def __init__(
        self,
        vocab_size: int = 32000,
        d_model: int = 512,
        n_layers: int = 8,
        n_heads: int = 8,
        fine_chunk_size: int = 32,
        coarse_chunk_sizes: Sequence[int] = (64, 256),
        max_iterations: int = 4,
        share_weights: bool = True,
        ffn_expansion: int = 4,
        dropout: float = 0.1,
        max_len: int = 8192,
    ):
        super().__init__()
        self.d_model = d_model

        # Embeddings
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_len, d_model)
        self.emb_drop = nn.Dropout(dropout)
        self.emb_norm = nn.LayerNorm(d_model)

        # Layers
        self.layers = nn.ModuleList([
            FENLayer(
                d_model=d_model,
                n_heads=n_heads,
                fine_chunk_size=fine_chunk_size,
                coarse_chunk_sizes=coarse_chunk_sizes,
                dropout=dropout,
                max_iterations=max_iterations,
                share_weights=share_weights,
                ffn_expansion=ffn_expansion,
            )
            for _ in range(n_layers)
        ])

        # Output
        self.final_norm = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.lm_head.weight = self.tok_emb.weight  # weight tying

        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(
        self,
        input_ids: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        input_ids : (B, L)
        mask      : (B, L) float 1=valid 0=pad

        Returns: logits (B, L, vocab_size)
        """
        B, L = input_ids.shape
        pos = torch.arange(L, device=input_ids.device).unsqueeze(0)

        x = self.tok_emb(input_ids) + self.pos_emb(pos)
        x = self.emb_drop(self.emb_norm(x))

        for layer in self.layers:
            x = layer(x, mask)

        return self.lm_head(self.final_norm(x))

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ═══════════════════════════════════════════════════════════════════════════
# Config presets
# ═══════════════════════════════════════════════════════════════════════════

def fen_small(vocab_size: int = 32000, **kw) -> FEN:
    """~15-25M params — ablation scale."""
    return FEN(
        vocab_size, d_model=256, n_layers=6, n_heads=4,
        fine_chunk_size=16, coarse_chunk_sizes=(32, 128),
        max_iterations=3, share_weights=True, **kw,
    )


def fen_base(vocab_size: int = 32000, **kw) -> FEN:
    """~100M params — BERT-base equivalent."""
    return FEN(
        vocab_size, d_model=768, n_layers=8, n_heads=12,
        fine_chunk_size=32, coarse_chunk_sizes=(64, 256),
        max_iterations=4, share_weights=True, **kw,
    )


def fen_large(vocab_size: int = 32000, **kw) -> FEN:
    """~300M+ params — GPT-2 medium equivalent."""
    return FEN(
        vocab_size, d_model=1024, n_layers=12, n_heads=16,
        fine_chunk_size=32, coarse_chunk_sizes=(64, 256, 1024),
        max_iterations=4, share_weights=True, **kw,
    )
