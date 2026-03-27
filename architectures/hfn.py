"""
Hierarchical Fractal Network (HFN)
===================================
O(L)-complexity sequence model built on hierarchical multi-scale
chunk processing with local attention, graph convolution across
scales, and global register tokens for long-range information flow.

Design thesis
-------------
Transformer = O(L²) global attention at one resolution.
Mamba       = O(L)  selective recurrence at one resolution.
HFN         = O(L)  multi-resolution: local attention within chunks,
              GNN across chunks at multiple scales, global register
              tokens for unbounded-range communication.

Architecture (one HFNLayer):
  1. LocalAttention    — within-chunk self-attention  (token↔token, O(Lc))
  2. MultiScaleFractal — chunk pool → GNN → broadcast at K scales (O(L))
  3. GlobalCondenser   — register tokens cross-attend to all positions (O(kL))
  4. FeedForward       — standard per-token MLP

Stack N × HFNLayer = full model.

Complexity: O(L · (c + K·g + k + d))  where c=chunk, K=scales,
g=gnn_depth, k=global_tokens, d=ffn_dim.  All constants → O(L).

Modules
-------
RotaryEmbedding         – RoPE for local attention.
LocalChunkAttention      – Windowed self-attention within chunks.
ChainGraphConv           – 1-hop message passing on chunk chain (improved).
HierarchicalScaleBlock   – Single-scale: pool → GNN → broadcast → gate.
MultiScaleFusion         – Parallel multi-scale blocks + learned fusion.
GlobalCondenser          – Small set of register tokens cross-attending to seq.
SwiGLU                   – Gated feed-forward (PaLM/LLaMA style).
HFNLayer                 – One full layer of the architecture.
HFN                      – Complete model: embed → N×HFNLayer → head.
"""

from __future__ import annotations

import math
from typing import Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ═══════════════════════════════════════════════════════════════════════════
# Positional encoding
# ═══════════════════════════════════════════════════════════════════════════

class RotaryEmbedding(nn.Module):
    """Rotary Position Embedding (Su et al., 2021).

    Generates sin/cos rotation matrices up to *max_len* positions.
    Cached and expanded on-the-fly if a longer sequence is seen.
    """

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
        freqs = torch.outer(t, inv_freq)              # (L, dim/2)
        self.register_buffer("cos_cached", freqs.cos(), persistent=False)
        self.register_buffer("sin_cached", freqs.sin(), persistent=False)

    def forward(self, seq_len: int) -> Tuple[torch.Tensor, torch.Tensor]:
        if seq_len > self.cos_cached.size(0):
            self._build_cache(seq_len)
        return (
            self.cos_cached[:seq_len],                 # (L, dim/2)
            self.sin_cached[:seq_len],
        )


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([-x2, x1], dim=-1)


def apply_rotary(q: torch.Tensor, k: torch.Tensor,
                 cos: torch.Tensor, sin: torch.Tensor):
    """Apply RoPE to query and key tensors.

    q, k : (B, H, L, Dh)
    cos, sin : (L, Dh/2)  → broadcast to (1, 1, L, Dh)
    """
    cos = cos.unsqueeze(0).unsqueeze(0)                # (1,1,L,Dh/2)
    sin = sin.unsqueeze(0).unsqueeze(0)
    cos = cos.repeat(1, 1, 1, 2)                       # (1,1,L,Dh)
    sin = sin.repeat(1, 1, 1, 2)
    q_rot = q * cos + _rotate_half(q) * sin
    k_rot = k * cos + _rotate_half(k) * sin
    return q_rot, k_rot


# ═══════════════════════════════════════════════════════════════════════════
# Local attention (within chunks)
# ═══════════════════════════════════════════════════════════════════════════

class LocalChunkAttention(nn.Module):
    """Self-attention restricted to within-chunk windows.

    Complexity: O(L · c) where c = chunk_size, since each token
    only attends to at most *c* neighbours.

    Overlapping windows (shift by c//2 on even/odd layers) ensure
    information crosses chunk boundaries over two layers.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int = 4,
        chunk_size: int = 64,
        dropout: float = 0.0,
        shift: bool = False,
    ):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.chunk_size = chunk_size
        self.shift = shift
        self.scale = self.d_head ** -0.5

        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        self.rope = RotaryEmbedding(self.d_head)
        self.attn_drop = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        x    : (B, L, D)
        mask : (B, L) or None
        """
        B, L, D = x.shape
        c = self.chunk_size
        shift_amt = c // 2 if self.shift else 0

        # ---- optional shift for cross-boundary information ----
        if shift_amt > 0 and L > shift_amt:
            x = torch.roll(x, shifts=-shift_amt, dims=1)
            if mask is not None:
                mask = torch.roll(mask, shifts=-shift_amt, dims=1)

        # ---- pad to multiple of chunk_size ----
        N = (L + c - 1) // c
        pad_len = N * c - L
        if pad_len > 0:
            x = F.pad(x, (0, 0, 0, pad_len))
            if mask is not None:
                mask = F.pad(mask, (0, pad_len), value=0.0)

        Lp = N * c  # padded length

        # ---- QKV ----
        qkv = self.qkv(x).reshape(B, Lp, 3, self.n_heads, self.d_head)
        qkv = qkv.permute(2, 0, 3, 1, 4)               # (3, B, H, Lp, Dh)
        q, k, v = qkv.unbind(0)

        # ---- RoPE (local positions within each chunk) ----
        cos, sin = self.rope(c)
        # Reshape to chunks, apply RoPE per chunk, reshape back
        q_c = q.reshape(B, self.n_heads, N, c, self.d_head)
        k_c = k.reshape(B, self.n_heads, N, c, self.d_head)
        # Apply same positional encoding within each chunk
        cos_c = cos.unsqueeze(0).unsqueeze(0).unsqueeze(0)   # (1,1,1,c,Dh/2)
        sin_c = sin.unsqueeze(0).unsqueeze(0).unsqueeze(0)
        cos_c = cos_c.repeat(1, 1, 1, 1, 2)                 # → Dh
        sin_c = sin_c.repeat(1, 1, 1, 1, 2)
        q_c = q_c * cos_c + _rotate_half(q_c) * sin_c
        k_c = k_c * cos_c + _rotate_half(k_c) * sin_c

        v_c = v.reshape(B, self.n_heads, N, c, self.d_head)

        # ---- chunked attention ----
        attn = torch.einsum("bhnid,bhnjd->bhnij", q_c, k_c) * self.scale

        # Mask: padding positions within chunks
        if mask is not None:
            m_c = mask.reshape(B, 1, N, c)                  # (B,1,N,c)
            # key mask: (B,1,N,1,c) — which keys are valid
            attn = attn.masked_fill(m_c.unsqueeze(3) == 0, float("-inf"))

        attn = F.softmax(attn, dim=-1)
        attn = torch.nan_to_num(attn, nan=0.0)              # all-masked chunks
        attn = self.attn_drop(attn)

        out_c = torch.einsum("bhnij,bhnjd->bhnid", attn, v_c)
        out = out_c.reshape(B, self.n_heads, Lp, self.d_head)
        out = out.permute(0, 2, 1, 3).reshape(B, Lp, D)
        out = self.out_proj(out)

        # ---- remove padding ----
        out = out[:, :L]

        # ---- reverse shift ----
        if shift_amt > 0 and L > shift_amt:
            out = torch.roll(out, shifts=shift_amt, dims=1)

        return out


# ═══════════════════════════════════════════════════════════════════════════
# Graph convolution (improved chain)
# ═══════════════════════════════════════════════════════════════════════════

class ChainGraphConv(nn.Module):
    """Message passing on a linear chain with learnable edge weights.

    Improvement over SimpleGraphConv: separate projections for
    self, left, right with learnable combination weights (softmax).
    """

    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.proj_self = nn.Linear(d_model, d_model, bias=False)
        self.proj_left = nn.Linear(d_model, d_model, bias=False)
        self.proj_right = nn.Linear(d_model, d_model, bias=False)
        self.mix = nn.Parameter(torch.zeros(3))          # learnable weights
        self.norm = nn.LayerNorm(d_model)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """h : (B, N, D) → (B, N, D)"""
        w = F.softmax(self.mix, dim=0)
        h_self = self.proj_self(h)
        h_left = F.pad(self.proj_left(h[:, :-1]), (0, 0, 1, 0))
        h_right = F.pad(self.proj_right(h[:, 1:]), (0, 0, 0, 1))
        h_agg = w[0] * h_self + w[1] * h_left + w[2] * h_right
        return self.norm(F.silu(h_agg))


# ═══════════════════════════════════════════════════════════════════════════
# Single-scale hierarchical block
# ═══════════════════════════════════════════════════════════════════════════

class HierarchicalScaleBlock(nn.Module):
    """Pool → GNN → Gated Broadcast at a single chunk scale.

    Structurally identical intent to FractalGNNBlock but with:
    - Learnable pool query (attention pooling, not mean)
    - Improved ChainGraphConv
    - Pre-norm pattern
    """

    def __init__(
        self,
        d_model: int,
        chunk_size: int,
        gnn_depth: int = 2,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.d_model = d_model
        self.chunk_size = chunk_size

        # Attention pooling: learnable query per chunk
        self.pool_query = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.pool_kv = nn.Linear(d_model, 2 * d_model, bias=False)
        self.pool_scale = d_model ** -0.5

        # GNN layers
        self.pre_norm = nn.LayerNorm(d_model)
        self.gnns = nn.ModuleList(
            [ChainGraphConv(d_model) for _ in range(gnn_depth)]
        )

        # Gated broadcast
        self.gate_proj = nn.Linear(2 * d_model, d_model)
        self.value_proj = nn.Linear(2 * d_model, d_model)
        self.drop = nn.Dropout(dropout)

    def _pool_chunks(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, int, int]:
        """Attention-weighted pooling within each chunk."""
        B, L, D = x.shape
        c = self.chunk_size
        N = (L + c - 1) // c
        pad_len = N * c - L

        # Pad
        if pad_len > 0:
            x_pad = F.pad(x, (0, 0, 0, pad_len))
            mask_pad = (
                F.pad(mask, (0, pad_len), value=0.0)
                if mask is not None else None
            )
        else:
            x_pad = x
            mask_pad = mask

        x_chunks = x_pad.reshape(B, N, c, D)            # (B, N, c, D)

        # Attention pooling
        query = self.pool_query.expand(B, N, D)          # (B, N, D)
        kv = self.pool_kv(x_chunks)                      # (B, N, c, 2D)
        k, v = kv.chunk(2, dim=-1)                       # each (B, N, c, D)

        attn = torch.einsum("bnd,bncd->bnc", query, k) * self.pool_scale
        if mask_pad is not None:
            m_c = mask_pad.reshape(B, N, c)
            attn = attn.masked_fill(m_c == 0, float("-inf"))
        attn = F.softmax(attn, dim=-1)
        attn = torch.nan_to_num(attn, nan=0.0)

        h_chunk = torch.einsum("bnc,bncd->bnd", attn, v)  # (B, N, D)
        return h_chunk, N, pad_len

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """x : (B, L, D) → (B, L, D)"""
        B, L, D = x.shape
        c = self.chunk_size

        # Pool
        h_chunk, N, pad_len = self._pool_chunks(x, mask)

        # GNN
        h_chunk = self.pre_norm(h_chunk)
        for gnn in self.gnns:
            h_chunk = h_chunk + gnn(h_chunk)

        # Broadcast back to token resolution
        h_broad = (
            h_chunk.unsqueeze(2)
            .expand(B, N, c, D)
            .reshape(B, N * c, D)[:, :L]
        )

        # Gated residual
        cat = torch.cat([x, h_broad], dim=-1)
        gate = torch.sigmoid(self.gate_proj(cat))
        value = F.silu(self.value_proj(cat))
        return self.drop(gate * value)                   # delta (not x+delta)


# ═══════════════════════════════════════════════════════════════════════════
# Multi-scale fusion
# ═══════════════════════════════════════════════════════════════════════════

class MultiScaleFusion(nn.Module):
    """Parallel HierarchicalScaleBlocks at multiple chunk sizes,
    fused via learned per-scale gating (not just linear concat).
    """

    def __init__(
        self,
        d_model: int,
        chunk_sizes: Sequence[int] = (16, 64, 256),
        gnn_depth: int = 2,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.scales = nn.ModuleList([
            HierarchicalScaleBlock(d_model, c, gnn_depth, dropout)
            for c in chunk_sizes
        ])
        # Per-scale learned gate
        self.scale_gates = nn.Linear(d_model, len(chunk_sizes))

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """x : (B, L, D) → delta : (B, L, D)"""
        deltas = torch.stack(
            [scale(x, mask) for scale in self.scales], dim=-1
        )                                                # (B, L, D, K)
        gates = F.softmax(self.scale_gates(x), dim=-1)   # (B, L, K)
        gates = gates.unsqueeze(2)                        # (B, L, 1, K)
        fused = (deltas * gates).sum(dim=-1)              # (B, L, D)
        return fused


# ═══════════════════════════════════════════════════════════════════════════
# Global register tokens (long-range pathway)
# ═══════════════════════════════════════════════════════════════════════════

class GlobalCondenser(nn.Module):
    """A small set of *k* global register tokens that cross-attend to the
    full sequence, then broadcast information back.

    Complexity: O(k · L) — linear in L for fixed k.

    Inspired by Perceiver's latent array and ViT register tokens,
    but bidirectional: tokens → registers → tokens.
    """

    def __init__(
        self,
        d_model: int,
        n_registers: int = 8,
        n_heads: int = 4,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.n_registers = n_registers
        self.registers = nn.Parameter(
            torch.randn(1, n_registers, d_model) * 0.02
        )

        # Registers attend to sequence (cross-attention)
        self.gather_q = nn.Linear(d_model, d_model, bias=False)
        self.gather_kv = nn.Linear(d_model, 2 * d_model, bias=False)

        # Sequence attends to registers (cross-attention)
        self.scatter_q = nn.Linear(d_model, d_model, bias=False)
        self.scatter_kv = nn.Linear(d_model, 2 * d_model, bias=False)

        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.scale = self.d_head ** -0.5

        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        self.norm_reg = nn.LayerNorm(d_model)
        self.norm_out = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)

    def _cross_attn(
        self,
        q_input: torch.Tensor,   # (B, Lq, D)
        kv_input: torch.Tensor,  # (B, Lkv, D)
        q_proj: nn.Linear,
        kv_proj: nn.Linear,
        kv_mask: Optional[torch.Tensor] = None,  # (B, Lkv)
    ) -> torch.Tensor:
        B = q_input.size(0)
        H, Dh = self.n_heads, self.d_head

        q = q_proj(q_input).reshape(B, -1, H, Dh).transpose(1, 2)
        kv = kv_proj(kv_input).reshape(B, -1, H, 2 * Dh).transpose(1, 2)
        k, v = kv[..., :Dh], kv[..., Dh:]

        attn = torch.einsum("bhid,bhjd->bhij", q, k) * self.scale
        if kv_mask is not None:
            attn = attn.masked_fill(
                kv_mask[:, None, None, :] == 0, float("-inf")
            )
        attn = F.softmax(attn, dim=-1)
        attn = torch.nan_to_num(attn, nan=0.0)

        out = torch.einsum("bhij,bhjd->bhid", attn, v)
        return out.transpose(1, 2).reshape(B, -1, H * Dh)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """x : (B, L, D) → delta : (B, L, D)"""
        B = x.size(0)
        regs = self.registers.expand(B, -1, -1)         # (B, k, D)

        # Gather: registers ← sequence
        regs = regs + self._cross_attn(
            regs, x, self.gather_q, self.gather_kv, kv_mask=mask
        )
        regs = self.norm_reg(regs)

        # Scatter: sequence ← registers
        delta = self._cross_attn(
            x, regs, self.scatter_q, self.scatter_kv
        )
        delta = self.norm_out(self.out_proj(delta))
        return self.drop(delta)


# ═══════════════════════════════════════════════════════════════════════════
# Feed-forward (SwiGLU)
# ═══════════════════════════════════════════════════════════════════════════

class SwiGLU(nn.Module):
    """SwiGLU feed-forward (Shazeer 2020, used in PaLM/LLaMA)."""

    def __init__(self, d_model: int, expansion: int = 4, dropout: float = 0.0):
        super().__init__()
        d_inner = d_model * expansion
        self.w_gate = nn.Linear(d_model, d_inner, bias=False)
        self.w_up = nn.Linear(d_model, d_inner, bias=False)
        self.w_down = nn.Linear(d_inner, d_model, bias=False)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop(self.w_down(F.silu(self.w_gate(x)) * self.w_up(x)))


# ═══════════════════════════════════════════════════════════════════════════
# HFN Layer (one full block of the architecture)
# ═══════════════════════════════════════════════════════════════════════════

class HFNLayer(nn.Module):
    """One layer of the Hierarchical Fractal Network.

    Pre-norm residual pattern:
        x = x + LocalChunkAttention(norm(x))
        x = x + MultiScaleFusion(norm(x))
        x = x + GlobalCondenser(norm(x))
        x = x + SwiGLU(norm(x))

    The three middle sub-layers capture:
        - Local: token-to-token within chunk window
        - Hierarchical: multi-scale chunk-level structure
        - Global: unbounded-range via register tokens
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int = 4,
        local_chunk_size: int = 64,
        scale_chunk_sizes: Sequence[int] = (16, 64, 256),
        gnn_depth: int = 2,
        n_registers: int = 8,
        ffn_expansion: int = 4,
        dropout: float = 0.0,
        layer_idx: int = 0,
    ):
        super().__init__()

        # Sub-layer 1: local attention (alternating shift)
        self.norm1 = nn.LayerNorm(d_model)
        self.local_attn = LocalChunkAttention(
            d_model, n_heads, local_chunk_size, dropout,
            shift=(layer_idx % 2 == 1),
        )

        # Sub-layer 2: multi-scale hierarchical
        self.norm2 = nn.LayerNorm(d_model)
        self.multi_scale = MultiScaleFusion(
            d_model, scale_chunk_sizes, gnn_depth, dropout,
        )

        # Sub-layer 3: global register pathway
        self.norm3 = nn.LayerNorm(d_model)
        self.global_path = GlobalCondenser(
            d_model, n_registers, n_heads, dropout,
        )

        # Sub-layer 4: feed-forward
        self.norm4 = nn.LayerNorm(d_model)
        self.ffn = SwiGLU(d_model, ffn_expansion, dropout)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        x = x + self.local_attn(self.norm1(x), mask)
        x = x + self.multi_scale(self.norm2(x), mask)
        x = x + self.global_path(self.norm3(x), mask)
        x = x + self.ffn(self.norm4(x))
        return x


# ═══════════════════════════════════════════════════════════════════════════
# Complete model
# ═══════════════════════════════════════════════════════════════════════════

class HFN(nn.Module):
    """Hierarchical Fractal Network — complete sequence model.

    Parameters
    ----------
    vocab_size : int
        Vocabulary size for token embedding.
    d_model : int
        Hidden dimension.
    n_layers : int
        Number of HFNLayers.
    n_heads : int
        Attention heads for local attention and global condenser.
    local_chunk_size : int
        Window size for local attention.
    scale_chunk_sizes : sequence of int
        Chunk sizes for multi-scale hierarchical processing.
    gnn_depth : int
        GNN depth at each scale.
    n_registers : int
        Number of global register tokens.
    ffn_expansion : int
        FFN hidden dim multiplier.
    dropout : float
    max_len : int
        Maximum sequence length (for absolute position fallback).
    """

    def __init__(
        self,
        vocab_size: int = 32000,
        d_model: int = 512,
        n_layers: int = 12,
        n_heads: int = 8,
        local_chunk_size: int = 64,
        scale_chunk_sizes: Sequence[int] = (16, 64, 256),
        gnn_depth: int = 2,
        n_registers: int = 8,
        ffn_expansion: int = 4,
        dropout: float = 0.1,
        max_len: int = 8192,
    ):
        super().__init__()
        self.d_model = d_model

        # Embeddings
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_len, d_model)  # absolute fallback
        self.emb_drop = nn.Dropout(dropout)
        self.emb_norm = nn.LayerNorm(d_model)

        # Layers
        self.layers = nn.ModuleList([
            HFNLayer(
                d_model=d_model,
                n_heads=n_heads,
                local_chunk_size=local_chunk_size,
                scale_chunk_sizes=scale_chunk_sizes,
                gnn_depth=gnn_depth,
                n_registers=n_registers,
                ffn_expansion=ffn_expansion,
                dropout=dropout,
                layer_idx=i,
            )
            for i in range(n_layers)
        ])

        # Output head
        self.final_norm = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

        # Weight tying
        self.lm_head.weight = self.tok_emb.weight

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
        input_ids : (B, L)  long tensor of token indices
        mask      : (B, L)  float, 1=valid 0=pad

        Returns
        -------
        logits : (B, L, vocab_size)
        """
        B, L = input_ids.shape
        positions = torch.arange(L, device=input_ids.device).unsqueeze(0)

        x = self.tok_emb(input_ids) + self.pos_emb(positions)
        x = self.emb_drop(self.emb_norm(x))

        for layer in self.layers:
            x = layer(x, mask)

        x = self.final_norm(x)
        return self.lm_head(x)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ═══════════════════════════════════════════════════════════════════════════
# Convenience: config presets
# ═══════════════════════════════════════════════════════════════════════════

def hfn_small(vocab_size: int = 32000, **kw) -> HFN:
    """~25M params — ablation/debugging scale."""
    return HFN(vocab_size, d_model=256, n_layers=6, n_heads=4,
               scale_chunk_sizes=(8, 32, 128), **kw)


def hfn_base(vocab_size: int = 32000, **kw) -> HFN:
    """~110M params — BERT-base equivalent."""
    return HFN(vocab_size, d_model=768, n_layers=12, n_heads=12,
               scale_chunk_sizes=(16, 64, 256), **kw)


def hfn_large(vocab_size: int = 32000, **kw) -> HFN:
    """~350M params — GPT-2 medium equivalent."""
    return HFN(vocab_size, d_model=1024, n_layers=24, n_heads=16,
               scale_chunk_sizes=(16, 64, 256, 1024), gnn_depth=3, **kw)
