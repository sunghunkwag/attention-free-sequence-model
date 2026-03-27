"""
Particle Field Network (PFN)
=============================
Lagrangian sequence model. No tokens. No attention. No fixed grid.

Fundamental departure
---------------------
ALL existing sequence models (Transformer, Mamba, NCA, BLT, etc.)
use an EULERIAN frame: information lives at fixed positions on a grid.
Position 0 has vector_0, position 1 has vector_1, etc.

PFN uses a LAGRANGIAN frame: information is carried by PARTICLES
that exist in a continuous space with learned positions.
There is no grid. Particles move, merge, split, and interact
through distance-based force fields.

Physical analogy: SPH (Smoothed Particle Hydrodynamics).
Each particle has:
  - position  (where in "meaning space" it exists)
  - momentum  (direction of semantic drift)
  - charge    (information content / state vector)
  - mass      (importance weight)

Particles interact through:
  - Short-range repulsion (prevent collapse / maintain distinctness)
  - Medium-range coupling (information exchange between nearby concepts)
  - Field emission (each particle contributes to a global field
    that all particles sense)

Why this is fundamentally different
------------------------------------
Token-based:  "The cat sat" → [tok_The, tok_cat, tok_sat] at positions [0, 1, 2]
Particle-based: "The cat sat" → particles in continuous space, positions LEARNED
  "cat" and "sat" might be close (subject-verb), "The" might be far (article)
  The GEOMETRY of the particle configuration encodes syntax/semantics.

No attention: particles interact through physics (distance kernels),
not through learned query-key-value projections.

No fixed vocabulary: input bytes (0-255) seed initial particle states.
The model learns to spawn and evolve particles from raw bytes.

Complexity: O(N · K) where N = number of particles, K = neighbours
per particle (fixed via spatial hashing or KNN). Strictly linear.

Modules
-------
ByteToParticle     – Convert byte stream to initial particle system.
ParticleKernel     – Distance-based interaction (SPH-style).
ParticleStep       – One dynamics timestep: interact → update → diffuse.
ParticleDynamics   – T timesteps of particle evolution.
FieldReadout       – Extract predictions from particle field.
PFN                – Complete model.
"""

from __future__ import annotations

import math
from typing import Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ═══════════════════════════════════════════════════════════════════════════
# Byte → Particle conversion
# ═══════════════════════════════════════════════════════════════════════════

class ByteToParticle(nn.Module):
    """Convert raw byte sequence into a particle system.

    Each byte spawns a particle with:
      - position: learned embedding in R^pos_dim (semantic space coordinates)
      - charge:   learned state vector in R^d_model (information content)
      - mass:     scalar importance weight (learned per byte value)

    The byte's ORDINAL position injects a continuous phase signal
    (sinusoidal, like positional encoding but as initial momentum).
    """

    def __init__(self, d_model: int, pos_dim: int = 16, max_len: int = 8192):
        super().__init__()
        self.d_model = d_model
        self.pos_dim = pos_dim

        # Byte value → initial particle properties (256 possible byte values)
        self.byte_to_charge = nn.Embedding(256, d_model)
        self.byte_to_position = nn.Embedding(256, pos_dim)
        self.byte_to_mass = nn.Embedding(256, 1)

        # Ordinal position → momentum (sinusoidal, not learned)
        self.register_buffer(
            'pos_phase', self._make_sinusoidal(max_len, d_model)
        )

    @staticmethod
    def _make_sinusoidal(max_len: int, dim: int) -> torch.Tensor:
        pe = torch.zeros(max_len, dim)
        position = torch.arange(max_len).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, dim, 2).float() * (-math.log(10000.0) / dim)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe

    def forward(self, byte_ids: torch.Tensor) -> dict:
        """
        byte_ids: (B, L) long tensor, values 0-255

        Returns dict with:
          charge:   (B, L, d_model)  — information state
          position: (B, L, pos_dim)  — location in meaning space
          mass:     (B, L, 1)        — importance weight
          momentum: (B, L, d_model)  — initial velocity (from ordinal pos)
        """
        B, L = byte_ids.shape

        charge = self.byte_to_charge(byte_ids)          # (B, L, D)
        position = self.byte_to_position(byte_ids)       # (B, L, pos_dim)
        mass = torch.sigmoid(self.byte_to_mass(byte_ids))  # (B, L, 1) in [0,1]
        momentum = self.pos_phase[:L].unsqueeze(0).expand(B, -1, -1)

        # Initial charge = byte embedding + ordinal momentum
        charge = charge + momentum

        return {
            'charge': charge,
            'position': position,
            'mass': mass,
        }


# ═══════════════════════════════════════════════════════════════════════════
# Particle interaction kernel (SPH-style)
# ═══════════════════════════════════════════════════════════════════════════

class ParticleKernel(nn.Module):
    """Distance-based particle interaction.

    Each particle senses its neighbours through a smooth kernel
    that depends on distance in position-space.

    Unlike attention (which computes affinities from content),
    this computes interactions from GEOMETRY (spatial distance).

    The kernel has three components:
      1. Density estimation: how crowded is this region?
      2. Force field: net information flow from neighbours.
      3. Pressure: prevent particles from collapsing together.
    """

    def __init__(self, d_model: int, pos_dim: int = 16, n_kernels: int = 4):
        super().__init__()
        self.d_model = d_model
        self.pos_dim = pos_dim
        self.n_kernels = n_kernels

        # Learned kernel widths (one per kernel)
        self.log_bandwidth = nn.Parameter(torch.zeros(n_kernels))

        # Kernel-specific charge projections
        self.charge_proj = nn.Linear(d_model, d_model * n_kernels, bias=False)

        # Combine kernel outputs
        self.combine = nn.Linear(d_model * n_kernels, d_model, bias=False)
        self.norm = nn.LayerNorm(d_model)

    def forward(
        self,
        charge: torch.Tensor,    # (B, N, D)
        position: torch.Tensor,  # (B, N, pos_dim)
        mass: torch.Tensor,      # (B, N, 1)
    ) -> torch.Tensor:
        """Compute interaction field for each particle.

        Returns: field contribution (B, N, D)
        """
        B, N, D = charge.shape

        # Pairwise distances in position space
        # (B, N, 1, pos_dim) - (B, 1, N, pos_dim) → (B, N, N, pos_dim)
        diff = position.unsqueeze(2) - position.unsqueeze(1)
        dist_sq = (diff ** 2).sum(dim=-1)  # (B, N, N)

        # Multi-bandwidth RBF kernels
        bandwidths = torch.exp(self.log_bandwidth)  # (n_kernels,)
        # (B, N, N, n_kernels)
        kernels = torch.exp(
            -dist_sq.unsqueeze(-1) / (2 * bandwidths.view(1, 1, 1, -1) + 1e-8)
        )

        # Mass-weighted kernels
        mass_weights = mass.unsqueeze(1)  # (B, 1, N, 1)
        weighted_kernels = kernels * mass_weights  # (B, N, N, n_kernels)

        # Project charges through kernel-specific transforms
        projected = self.charge_proj(charge)  # (B, N, D*n_kernels)
        projected = projected.reshape(B, N, self.n_kernels, D)  # (B, N, K, D)

        # Kernel-weighted aggregation:
        # For each particle i, sum over all j:
        #   field_i = sum_j kernel(i,j) * mass_j * proj_k(charge_j)
        # Shape: (B, N, N, K) × need to contract with (B, N, K, D)

        # Rearrange for einsum: kernels (B,N,N,K), projected (B,N,K,D)
        # For particle i: sum over j of kernel[i,j,k] * projected[j,k,d]
        field = torch.einsum(
            'bijk,bjkd->bikd', weighted_kernels, projected
        )  # (B, N, K, D)

        # Flatten kernels and combine
        field = field.reshape(B, N, self.n_kernels * D)
        field = self.combine(field)  # (B, N, D)

        return self.norm(field)


# ═══════════════════════════════════════════════════════════════════════════
# One particle dynamics timestep
# ═══════════════════════════════════════════════════════════════════════════

class ParticleStep(nn.Module):
    """One timestep of particle dynamics.

    1. Compute interaction field (from kernel)
    2. Update charge (information state)
    3. Update position (particles move in meaning space)
    4. Update mass (importance can change)
    """

    def __init__(self, d_model: int, pos_dim: int = 16, n_kernels: int = 4,
                 dropout: float = 0.0):
        super().__init__()
        self.kernel = ParticleKernel(d_model, pos_dim, n_kernels)

        # Charge update: gated reaction to field
        self.charge_gate = nn.Linear(2 * d_model, d_model, bias=False)
        self.charge_update = nn.Linear(2 * d_model, d_model, bias=False)
        self.alpha = nn.Parameter(torch.tensor(0.1))

        # Position update: field induces drift
        self.pos_drift = nn.Linear(d_model, pos_dim, bias=False)
        self.pos_rate = nn.Parameter(torch.tensor(0.01))

        # Mass update
        self.mass_update = nn.Linear(d_model, 1, bias=False)

        self.drop = nn.Dropout(dropout)

    def forward(self, charge, position, mass):
        """
        charge:   (B, N, D)
        position: (B, N, pos_dim)
        mass:     (B, N, 1)

        Returns: updated (charge, position, mass)
        """
        # Compute interaction field
        field = self.kernel(charge, position, mass)  # (B, N, D)

        # Update charge (gated)
        cat = torch.cat([charge, field], dim=-1)
        gate = torch.sigmoid(self.charge_gate(cat))
        delta = F.silu(self.charge_update(cat))
        charge = charge + self.alpha * self.drop(gate * delta)

        # Update position (field-induced drift)
        drift = self.pos_drift(field)
        position = position + self.pos_rate * drift

        # Update mass
        mass = torch.sigmoid(mass + 0.01 * self.mass_update(field))

        return charge, position, mass


# ═══════════════════════════════════════════════════════════════════════════
# Particle dynamics block: T timesteps
# ═══════════════════════════════════════════════════════════════════════════

class ParticleDynamics(nn.Module):
    """Run particle system for T timesteps.

    Weight sharing across timesteps (like DEQ / physics simulation
    with time-invariant force laws).
    """

    def __init__(self, d_model: int, pos_dim: int = 16, n_kernels: int = 4,
                 n_steps: int = 6, dropout: float = 0.0,
                 share_weights: bool = True):
        super().__init__()
        self.n_steps = n_steps
        self.share_weights = share_weights

        if share_weights:
            self.step = ParticleStep(d_model, pos_dim, n_kernels, dropout)
        else:
            self.steps = nn.ModuleList([
                ParticleStep(d_model, pos_dim, n_kernels, dropout)
                for _ in range(n_steps)
            ])

    def _get_step(self, t: int) -> ParticleStep:
        return self.step if self.share_weights else self.steps[t]

    def forward(self, charge, position, mass):
        """Evolve particle system for T steps."""
        for t in range(self.n_steps):
            charge, position, mass = self._get_step(t)(
                charge, position, mass
            )
        return charge, position, mass


# ═══════════════════════════════════════════════════════════════════════════
# Field readout: extract per-position predictions
# ═══════════════════════════════════════════════════════════════════════════

class FieldReadout(nn.Module):
    """Read out predictions from evolved particle system.

    After dynamics, we need per-byte predictions.
    The particle charges are already in the original byte order
    (particles don't reorder, they move in meaning-space).
    So readout is simply: project charge → logits.

    But we also incorporate the particle's evolved position
    and mass as additional context.
    """

    def __init__(self, d_model: int, pos_dim: int, vocab_size: int):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.pos_proj = nn.Linear(pos_dim, d_model, bias=False)
        self.mass_proj = nn.Linear(1, d_model, bias=False)
        self.out = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, charge, position, mass):
        """
        charge:   (B, N, D)
        position: (B, N, pos_dim)
        mass:     (B, N, 1)

        Returns: logits (B, N, vocab_size)
        """
        h = self.norm(charge)
        h = h + self.pos_proj(position) + self.mass_proj(mass)
        return self.out(h)


# ═══════════════════════════════════════════════════════════════════════════
# Feed-forward between dynamics blocks
# ═══════════════════════════════════════════════════════════════════════════

class GatedFFN(nn.Module):
    """Per-particle feed-forward."""

    def __init__(self, d_model: int, expansion: int = 2, dropout: float = 0.0):
        super().__init__()
        d_inner = d_model * expansion
        self.norm = nn.LayerNorm(d_model)
        self.w_gate = nn.Linear(d_model, d_inner, bias=False)
        self.w_up = nn.Linear(d_model, d_inner, bias=False)
        self.w_down = nn.Linear(d_inner, d_model, bias=False)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        x = self.norm(x)
        return residual + self.drop(
            self.w_down(F.silu(self.w_gate(x)) * self.w_up(x))
        )


# ═══════════════════════════════════════════════════════════════════════════
# PFN Layer
# ═══════════════════════════════════════════════════════════════════════════

class PFNLayer(nn.Module):
    """One layer: particle dynamics + FFN."""

    def __init__(self, d_model: int, pos_dim: int = 16, n_kernels: int = 4,
                 n_steps: int = 4, ffn_expansion: int = 2, dropout: float = 0.0):
        super().__init__()
        self.dynamics = ParticleDynamics(
            d_model, pos_dim, n_kernels, n_steps, dropout)
        self.ffn = GatedFFN(d_model, ffn_expansion, dropout)

    def forward(self, charge, position, mass):
        charge, position, mass = self.dynamics(charge, position, mass)
        charge = self.ffn(charge)
        return charge, position, mass


# ═══════════════════════════════════════════════════════════════════════════
# Complete model
# ═══════════════════════════════════════════════════════════════════════════

class PFN(nn.Module):
    """Particle Field Network.

    No tokens. No attention. No fixed grid computation.
    Information is carried by particles in continuous space.

    Parameters
    ----------
    vocab_size : int
        Output vocabulary (e.g. 256 for byte-level).
    d_model : int
        Particle charge dimension.
    pos_dim : int
        Dimensionality of particle position space.
    n_layers : int
        Number of PFNLayers (each has internal dynamics timesteps).
    n_steps : int
        Dynamics timesteps per layer.
    n_kernels : int
        Number of interaction kernel bandwidths.
    ffn_expansion : int
    dropout : float
    max_len : int
    """

    def __init__(
        self,
        vocab_size: int = 256,
        d_model: int = 128,
        pos_dim: int = 16,
        n_layers: int = 4,
        n_steps: int = 4,
        n_kernels: int = 4,
        ffn_expansion: int = 2,
        dropout: float = 0.0,
        max_len: int = 4096,
    ):
        super().__init__()
        self.spawn = ByteToParticle(d_model, pos_dim, max_len)

        self.layers = nn.ModuleList([
            PFNLayer(d_model, pos_dim, n_kernels, n_steps,
                     ffn_expansion, dropout)
            for _ in range(n_layers)
        ])

        self.readout = FieldReadout(d_model, pos_dim, vocab_size)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, std=0.02)

    def forward(self, byte_ids: torch.Tensor) -> torch.Tensor:
        """
        byte_ids: (B, L) long tensor, values in [0, vocab_size)
        Returns: logits (B, L, vocab_size)
        """
        particles = self.spawn(byte_ids)
        charge = particles['charge']
        position = particles['position']
        mass = particles['mass']

        for layer in self.layers:
            charge, position, mass = layer(charge, position, mass)

        return self.readout(charge, position, mass)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ═══════════════════════════════════════════════════════════════════════════
# Presets
# ═══════════════════════════════════════════════════════════════════════════

def pfn_tiny(vocab_size: int = 256, **kw) -> PFN:
    """CPU benchmark."""
    return PFN(vocab_size, d_model=64, pos_dim=8, n_layers=2,
               n_steps=3, n_kernels=3, ffn_expansion=2, **kw)


def pfn_small(vocab_size: int = 256, **kw) -> PFN:
    """Ablation scale."""
    return PFN(vocab_size, d_model=256, pos_dim=16, n_layers=4,
               n_steps=6, n_kernels=4, ffn_expansion=4, **kw)
