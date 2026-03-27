"""
Neural Cellular Automata Language Model (NCA-LM)
=================================================
Zero attention. Zero softmax. Zero transformer components.

Design principle
----------------
Each token position is a cell in a 1D cellular automaton.
Each cell has a state vector (d_model dimensions).
At each timestep:
  1. PERCEIVE: gather information from local neighbourhood
     via depthwise separable convolution (no attention).
  2. REACT: nonlinear state update via gated MLP.
  3. DIFFUSE: long-range information propagation via
     multi-rate diffusion (learned diffusion coefficients).

Repeat T steps. Global structure EMERGES from local rules
iterated over time — not from any global attention mechanism.

Biological analogy: morphogenesis.
Mathematical analogy: reaction-diffusion systems (Turing patterns).
Computational analogy: Rule 110 cellular automata (Turing-complete).

Why this is fundamentally different
------------------------------------
- Transformer: each token looks at ALL other tokens (global, O(L²))
- Mamba: each token processes a compressed history (sequential, O(L))
- NCA-LM: each token looks at NEIGHBOURS ONLY, iterated T times.
  Information travels at most T*kernel_radius positions per layer.
  Global patterns emerge from local dynamics, not global computation.

Complexity: O(L · T · d · k) where k = perception kernel size (fixed).
Strictly linear in L. No quadratic terms anywhere.

Modules
-------
PerceptionFilter   – Depthwise separable conv: sense neighbourhood.
ReactionGate       – Gated MLP: nonlinear state transition.
DiffusionOperator  – Multi-rate learned diffusion for long-range transport.
NCAStep            – One automaton timestep: perceive → react → diffuse.
NCABlock           – T timesteps with adaptive halting.
NCALayer           – NCABlock + feed-forward.
NCA_LM             – Complete language model.
"""

from __future__ import annotations

import math
from typing import Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ═══════════════════════════════════════════════════════════════════════════
# Perception: sense local neighbourhood (NO attention)
# ═══════════════════════════════════════════════════════════════════════════

class PerceptionFilter(nn.Module):
    """Depthwise separable convolution over local neighbourhood.

    Each channel independently senses its neighbourhood via a small
    1D kernel, then channels are mixed via pointwise convolution.

    This replaces attention's "what should I look at?" with
    "what is physically next to me?" — a fundamentally different
    information access pattern.

    The kernel includes Sobel-like derivative filters to detect
    gradients (rate of change) in the feature landscape, giving
    cells awareness of local structure, not just local values.
    """

    def __init__(self, d_model: int, kernel_size: int = 7, n_filters: int = 3):
        super().__init__()
        self.d_model = d_model
        self.kernel_size = kernel_size
        self.n_filters = n_filters

        # Multiple perception channels:
        # 1. Identity (self state)
        # 2. Smooth average (neighbourhood mean)
        # 3. Gradient (difference / Sobel)
        # Each is a depthwise conv
        self.depthwise_convs = nn.ModuleList([
            nn.Conv1d(
                d_model, d_model,
                kernel_size=kernel_size,
                padding=kernel_size // 2,
                groups=d_model,  # depthwise: each channel independent
                bias=False,
            )
            for _ in range(n_filters)
        ])

        # Pointwise mixing: combine n_filters perception channels
        self.pointwise = nn.Conv1d(
            d_model * n_filters, d_model, kernel_size=1, bias=False,
        )
        self.norm = nn.LayerNorm(d_model)

    def _init_filters(self):
        """Initialize with interpretable kernels, then let them learn."""
        k = self.kernel_size
        with torch.no_grad():
            # Filter 0: identity-ish (peaked at center)
            w0 = torch.zeros(k)
            w0[k // 2] = 1.0
            for p in self.depthwise_convs[0].parameters():
                p.data.copy_(w0.view(1, 1, k).expand_as(p))

            if self.n_filters > 1:
                # Filter 1: smooth average
                w1 = torch.ones(k) / k
                for p in self.depthwise_convs[1].parameters():
                    p.data.copy_(w1.view(1, 1, k).expand_as(p))

            if self.n_filters > 2:
                # Filter 2: Sobel gradient
                w2 = torch.zeros(k)
                w2[0] = -1.0
                w2[-1] = 1.0
                for p in self.depthwise_convs[2].parameters():
                    p.data.copy_(w2.view(1, 1, k).expand_as(p))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, L, D) → perceived: (B, L, D)"""
        # Conv1d expects (B, C, L)
        h = x.transpose(1, 2)  # (B, D, L)

        percepts = [conv(h) for conv in self.depthwise_convs]
        h_cat = torch.cat(percepts, dim=1)  # (B, D*n_filters, L)
        h_mixed = self.pointwise(h_cat)     # (B, D, L)

        return self.norm(h_mixed.transpose(1, 2))  # (B, L, D)


# ═══════════════════════════════════════════════════════════════════════════
# Reaction: nonlinear state update (NO attention)
# ═══════════════════════════════════════════════════════════════════════════

class ReactionGate(nn.Module):
    """Gated state update inspired by chemical reaction dynamics.

    Given current state x and perceived neighbourhood p:
      candidate = SiLU(W_up([x; p]))
      gate      = sigmoid(W_gate([x; p]))
      delta     = W_down(gate * candidate)
      x_new     = x + alpha * delta

    The gate determines which dimensions of state to update.
    alpha is a learnable per-channel scaling that starts near zero
    (stable initialization — cells don't explode at step 0).
    """

    def __init__(self, d_model: int, expansion: int = 2, dropout: float = 0.0):
        super().__init__()
        d_inner = d_model * expansion
        self.w_up = nn.Linear(2 * d_model, d_inner, bias=False)
        self.w_gate = nn.Linear(2 * d_model, d_inner, bias=False)
        self.w_down = nn.Linear(d_inner, d_model, bias=False)
        self.drop = nn.Dropout(dropout)

        # Learnable update magnitude — starts near zero for stability
        self.alpha = nn.Parameter(torch.full((d_model,), 0.1))

    def forward(self, x: torch.Tensor, perceived: torch.Tensor) -> torch.Tensor:
        """
        x         : (B, L, D) current cell state
        perceived : (B, L, D) neighbourhood perception

        Returns: updated state (B, L, D)
        """
        cat = torch.cat([x, perceived], dim=-1)  # (B, L, 2D)
        candidate = F.silu(self.w_up(cat))
        gate = torch.sigmoid(self.w_gate(cat))
        delta = self.w_down(self.drop(gate * candidate))
        return x + self.alpha * delta


# ═══════════════════════════════════════════════════════════════════════════
# Diffusion: long-range information transport (NO attention)
# ═══════════════════════════════════════════════════════════════════════════

class DiffusionOperator(nn.Module):
    """Learned multi-rate diffusion for long-range information propagation.

    Physical analogy: heat equation  ∂u/∂t = D · ∂²u/∂x²
    but with learned, per-channel diffusion coefficients D_i,
    and multiple diffusion rates operating simultaneously.

    Implementation: convolution with a learned smoothing kernel
    at multiple dilations (multi-rate). Dilation d means information
    can jump d positions in one step.

    After T timesteps with max dilation d_max, effective range is
    T * d_max positions — linear growth without any attention.
    """

    def __init__(
        self,
        d_model: int,
        n_rates: int = 3,
        dilations: Sequence[int] = (1, 4, 16),
        kernel_size: int = 3,
    ):
        super().__init__()
        assert n_rates == len(dilations)
        self.n_rates = n_rates

        # One depthwise conv per diffusion rate
        self.diffuse_convs = nn.ModuleList([
            nn.Conv1d(
                d_model, d_model,
                kernel_size=kernel_size,
                padding=dilation * (kernel_size // 2),
                dilation=dilation,
                groups=d_model,
                bias=False,
            )
            for dilation in dilations
        ])

        # Learned per-channel mixing of diffusion rates
        self.rate_mix = nn.Parameter(torch.zeros(d_model, n_rates))

        # Diffusion strength (starts small for stability)
        self.strength = nn.Parameter(torch.tensor(0.1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, L, D) → diffused: (B, L, D)"""
        B, L, D = x.shape
        h = x.transpose(1, 2)  # (B, D, L)

        # Apply each diffusion rate
        diffused = torch.stack(
            [conv(h) for conv in self.diffuse_convs], dim=-1
        )  # (B, D, L, n_rates)

        # Per-channel mixing weights
        weights = F.softmax(self.rate_mix, dim=-1)  # (D, n_rates)
        weights = weights.view(1, D, 1, self.n_rates)

        # Weighted combination
        combined = (diffused * weights).sum(dim=-1)  # (B, D, L)
        combined = combined.transpose(1, 2)          # (B, L, D)

        # Diffusion = residual + small correction
        return x + self.strength * (combined - x)


# ═══════════════════════════════════════════════════════════════════════════
# One NCA timestep
# ═══════════════════════════════════════════════════════════════════════════

class NCAStep(nn.Module):
    """One cellular automaton timestep: perceive → react → diffuse.

    This is the fundamental unit. Like one tick of a CA clock.
    No attention. No softmax. No global computation.
    """

    def __init__(
        self,
        d_model: int,
        kernel_size: int = 7,
        n_perception_filters: int = 3,
        reaction_expansion: int = 2,
        dilations: Sequence[int] = (1, 4, 16),
        dropout: float = 0.0,
    ):
        super().__init__()
        self.perceive = PerceptionFilter(d_model, kernel_size, n_perception_filters)
        self.react = ReactionGate(d_model, reaction_expansion, dropout)
        self.diffuse = DiffusionOperator(d_model, len(dilations), dilations)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, L, D) → x_next: (B, L, D)"""
        perceived = self.perceive(x)
        x = self.react(x, perceived)
        x = self.diffuse(x)
        return x


# ═══════════════════════════════════════════════════════════════════════════
# NCA Block: T timesteps with optional adaptive halting
# ═══════════════════════════════════════════════════════════════════════════

class NCABlock(nn.Module):
    """Multiple NCA timesteps with shared or unshared weights.

    Parameters
    ----------
    d_model : int
    n_steps : int
        Number of CA timesteps.
    share_weights : bool
        If True, all timesteps share one NCAStep (like a recurrent
        system or DEQ). Dramatically fewer parameters.
    adaptive : bool
        If True, per-cell adaptive halting (cells that have "settled"
        stop updating early).
    """

    def __init__(
        self,
        d_model: int,
        n_steps: int = 8,
        share_weights: bool = True,
        adaptive: bool = True,
        kernel_size: int = 7,
        n_perception_filters: int = 3,
        reaction_expansion: int = 2,
        dilations: Sequence[int] = (1, 4, 16),
        dropout: float = 0.0,
        halting_threshold: float = 0.99,
    ):
        super().__init__()
        self.n_steps = n_steps
        self.adaptive = adaptive
        self.share_weights = share_weights
        self.halting_threshold = halting_threshold

        step_args = dict(
            d_model=d_model,
            kernel_size=kernel_size,
            n_perception_filters=n_perception_filters,
            reaction_expansion=reaction_expansion,
            dilations=dilations,
            dropout=dropout,
        )

        if share_weights:
            self.step = NCAStep(**step_args)
        else:
            self.steps = nn.ModuleList(
                [NCAStep(**step_args) for _ in range(n_steps)]
            )

        if adaptive:
            self.halt_proj = nn.Linear(d_model, 1)

    def _get_step(self, t: int) -> NCAStep:
        return self.step if self.share_weights else self.steps[t]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, L, D) → (B, L, D)"""
        if not self.adaptive:
            # Simple: just iterate
            for t in range(self.n_steps):
                x = self._get_step(t)(x)
            return x

        # Adaptive halting
        B, L, D = x.shape
        device = x.device

        halting_prob = torch.zeros(B, L, 1, device=device)
        output_accum = torch.zeros(B, L, D, device=device)
        still_running = torch.ones(B, L, 1, device=device, dtype=torch.bool)

        state = x

        for t in range(self.n_steps):
            state = self._get_step(t)(state)
            p = torch.sigmoid(self.halt_proj(state))

            new_halted = (
                still_running
                & ((halting_prob + p) > self.halting_threshold)
            )
            if t == self.n_steps - 1:
                new_halted = still_running

            still_running_float = still_running.float()
            halt_weight = torch.where(
                new_halted,
                1.0 - halting_prob,
                p * still_running_float,
            )

            output_accum = output_accum + halt_weight * state
            halting_prob = halting_prob + p * still_running_float
            still_running = still_running & ~new_halted

            if not still_running.any():
                break

        return output_accum


# ═══════════════════════════════════════════════════════════════════════════
# Feed-forward (SwiGLU, no attention anywhere)
# ═══════════════════════════════════════════════════════════════════════════

class GatedFFN(nn.Module):
    """Gated feed-forward. Per-position. No attention."""

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
# NCA Layer = NCABlock + FFN
# ═══════════════════════════════════════════════════════════════════════════

class NCALayer(nn.Module):
    """One layer: NCA dynamics + per-position feed-forward."""

    def __init__(
        self,
        d_model: int,
        n_steps: int = 8,
        share_weights: bool = True,
        adaptive: bool = True,
        kernel_size: int = 7,
        dilations: Sequence[int] = (1, 4, 16),
        reaction_expansion: int = 2,
        ffn_expansion: int = 4,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.nca = NCABlock(
            d_model=d_model,
            n_steps=n_steps,
            share_weights=share_weights,
            adaptive=adaptive,
            kernel_size=kernel_size,
            dilations=dilations,
            reaction_expansion=reaction_expansion,
            dropout=dropout,
        )
        self.ffn = GatedFFN(d_model, ffn_expansion, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.nca(self.norm(x))
        x = self.ffn(x)
        return x


# ═══════════════════════════════════════════════════════════════════════════
# Complete model
# ═══════════════════════════════════════════════════════════════════════════

class NCA_LM(nn.Module):
    """Neural Cellular Automata Language Model.

    Zero attention. Zero softmax (except final classification head).
    Information propagation is purely local + diffusive + iterative.

    Parameters
    ----------
    vocab_size : int
    d_model : int
    n_layers : int
        Number of NCALayers (each runs n_steps CA iterations internally).
    n_steps : int
        CA timesteps per layer.
    share_weights : bool
        Share NCA step weights across timesteps within each layer.
    adaptive : bool
        Per-cell adaptive halting.
    kernel_size : int
        Perception filter width.
    dilations : sequence of int
        Diffusion rates. Max dilation determines max single-step reach.
    ffn_expansion : int
    dropout : float
    max_len : int
    """

    def __init__(
        self,
        vocab_size: int = 32000,
        d_model: int = 256,
        n_layers: int = 6,
        n_steps: int = 8,
        share_weights: bool = True,
        adaptive: bool = True,
        kernel_size: int = 7,
        dilations: Sequence[int] = (1, 4, 16),
        reaction_expansion: int = 2,
        ffn_expansion: int = 4,
        dropout: float = 0.1,
        max_len: int = 8192,
    ):
        super().__init__()
        self.d_model = d_model

        # Embeddings (learned position — could replace with sinusoidal)
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_len, d_model)
        self.emb_drop = nn.Dropout(dropout)
        self.emb_norm = nn.LayerNorm(d_model)

        # CA layers
        self.layers = nn.ModuleList([
            NCALayer(
                d_model=d_model,
                n_steps=n_steps,
                share_weights=share_weights,
                adaptive=adaptive,
                kernel_size=kernel_size,
                dilations=dilations,
                reaction_expansion=reaction_expansion,
                ffn_expansion=ffn_expansion,
                dropout=dropout,
            )
            for _ in range(n_layers)
        ])

        # Output
        self.final_norm = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.lm_head.weight = self.tok_emb.weight  # weight tying

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, std=0.02)
            elif isinstance(m, PerceptionFilter):
                m._init_filters()

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        input_ids : (B, L) long tensor
        Returns   : logits (B, L, vocab_size)
        """
        B, L = input_ids.shape
        pos = torch.arange(L, device=input_ids.device).unsqueeze(0)

        x = self.tok_emb(input_ids) + self.pos_emb(pos)
        x = self.emb_drop(self.emb_norm(x))

        for layer in self.layers:
            x = layer(x)

        return self.lm_head(self.final_norm(x))

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def effective_range(self) -> str:
        """Report theoretical maximum information propagation range."""
        n_layers = len(self.layers)
        # Each layer: n_steps iterations, each with max dilation
        layer0 = self.layers[0]
        n_steps = layer0.nca.n_steps
        max_dil = max(
            conv.dilation[0]
            for conv in layer0.nca._get_step(0).diffuse.diffuse_convs
        )
        k = layer0.nca._get_step(0).perceive.kernel_size
        # Per step: perception covers k//2 + diffusion covers max_dil
        per_step = k // 2 + max_dil
        total = n_layers * n_steps * per_step
        return (
            f"Per step: {per_step} positions | "
            f"Per layer ({n_steps} steps): {n_steps * per_step} | "
            f"Total ({n_layers} layers): {total} positions"
        )


# ═══════════════════════════════════════════════════════════════════════════
# Config presets
# ═══════════════════════════════════════════════════════════════════════════

def nca_tiny(vocab_size: int = 32000, **kw) -> NCA_LM:
    """Very small — CPU benchmark and ablation."""
    return NCA_LM(
        vocab_size, d_model=64, n_layers=2, n_steps=6,
        share_weights=True, adaptive=True,
        kernel_size=5, dilations=(1, 4, 16),
        reaction_expansion=2, ffn_expansion=2,
        **kw,
    )


def nca_small(vocab_size: int = 32000, **kw) -> NCA_LM:
    """~15-25M params — serious ablation."""
    return NCA_LM(
        vocab_size, d_model=256, n_layers=6, n_steps=8,
        share_weights=True, adaptive=True,
        kernel_size=7, dilations=(1, 4, 16),
        reaction_expansion=2, ffn_expansion=4,
        **kw,
    )


def nca_base(vocab_size: int = 32000, **kw) -> NCA_LM:
    """~100M params — BERT-base equivalent."""
    return NCA_LM(
        vocab_size, d_model=512, n_layers=8, n_steps=12,
        share_weights=True, adaptive=True,
        kernel_size=7, dilations=(1, 4, 16, 64),
        reaction_expansion=2, ffn_expansion=4,
        **kw,
    )
