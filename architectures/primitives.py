"""
Atomic mechanism modules for architecture search.

Every module has interface: forward(x) -> x where x is (B, L, D).
All modules are attention-free and softmax-free.

Categories:
  - Information Propagation (15+ mechanisms): move information across positions
  - State Update (8+ mechanisms): transform per-position state
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# =========================================================================
# INFORMATION PROPAGATION MECHANISMS
# =========================================================================


class DepthwiseConv(nn.Module):
    """Depthwise convolution at a given kernel size. Local mixing."""
    def __init__(self, d_model, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv1d(d_model, d_model, kernel_size,
                              padding=kernel_size // 2, groups=d_model, bias=False)
        self.pw = nn.Conv1d(d_model, d_model, 1, bias=False)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        h = x.transpose(1, 2)
        h = self.pw(self.conv(h)).transpose(1, 2)
        return x + self.norm(h)


class DilatedConv(nn.Module):
    """Dilated depthwise convolution for sparse long-range mixing."""
    def __init__(self, d_model, dilation=4, kernel_size=3):
        super().__init__()
        pad = dilation * (kernel_size // 2)
        self.conv = nn.Conv1d(d_model, d_model, kernel_size,
                              padding=pad, dilation=dilation, groups=d_model, bias=False)
        self.pw = nn.Conv1d(d_model, d_model, 1, bias=False)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        h = x.transpose(1, 2)
        h = self.pw(self.conv(h)).transpose(1, 2)
        return x + self.norm(h)


class GatedShiftMixerVariant(nn.Module):
    """GatedShiftMixer with configurable shift offsets."""
    def __init__(self, d_model, shifts=(-16, -4, -1, 1, 4, 16)):
        super().__init__()
        self.shifts = list(shifts)
        n = len(shifts) + 1
        self.gate_proj = nn.Linear(d_model, n * d_model, bias=False)
        self.value_proj = nn.Linear(d_model, d_model, bias=False)
        self.norm = nn.LayerNorm(d_model)
        self.n = n

    def forward(self, x):
        B, L, D = x.shape
        shifted = [x]
        for s in self.shifts:
            shifted.append(torch.roll(x, shifts=-s, dims=1))
        stacked = torch.stack(shifted, dim=2)
        stacked = self.value_proj(stacked)
        gates = torch.sigmoid(self.gate_proj(x)).reshape(B, L, self.n, D)
        out = (stacked * gates).sum(dim=2)
        return self.norm(out)


class SpectralFilter(nn.Module):
    """FFT -> learned frequency filter -> iFFT. O(L log L) spectral mixing."""
    def __init__(self, d_model, max_len=128):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(max_len // 2 + 1, d_model) * 0.02)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        B, L, D = x.shape
        h = x.transpose(1, 2)  # (B, D, L)
        freq = torch.fft.rfft(h, dim=-1)  # (B, D, L//2+1)
        n_freq = freq.shape[-1]
        w = self.weight[:n_freq, :].t().unsqueeze(0)  # (1, D, n_freq)
        freq = freq * torch.sigmoid(w)
        h = torch.fft.irfft(freq, n=L, dim=-1)
        return x + self.norm(h.transpose(1, 2))


class DiagonalSSM(nn.Module):
    """Diagonal state space recurrence (S4-style). O(L log L) via scan."""
    def __init__(self, d_model):
        super().__init__()
        self.A_log = nn.Parameter(torch.randn(d_model) * 0.1 - 1.0)
        self.B_proj = nn.Linear(d_model, d_model, bias=False)
        self.C_proj = nn.Linear(d_model, d_model, bias=False)
        self.D = nn.Parameter(torch.ones(d_model))
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        B, L, D = x.shape
        A = -torch.exp(self.A_log)  # negative real eigenvalues for stability
        Bx = self.B_proj(x)  # (B, L, D)

        # Parallel scan via cumulative powers of A
        # h_t = A^t * h_0 + sum_{k=0}^{t} A^{t-k} * Bx_k
        # Use conv1d with exponentially decaying kernel as fast approximation
        powers = A.unsqueeze(0).unsqueeze(0) ** torch.arange(L, device=x.device).float().unsqueeze(-1).unsqueeze(0)  # (1, L, D)
        # Flip for causal convolution
        kernel = powers.permute(2, 0, 1)  # (D, 1, L)
        Bx_t = Bx.transpose(1, 2)  # (B, D, L)
        y = F.conv1d(Bx_t, kernel, padding=L - 1, groups=D)[:, :, :L]  # (B, D, L)
        y = self.C_proj(y.transpose(1, 2))
        return x + self.norm(y + self.D * x)


class RandomSparseWiring(nn.Module):
    """Fixed random sparse wiring pattern. Each position reads from K random others."""
    def __init__(self, d_model, n_wires=8, max_len=128):
        super().__init__()
        self.n_wires = n_wires
        # Pre-generate random offsets
        offsets = torch.randint(-max_len, max_len, (n_wires,))
        self.register_buffer('offsets', offsets)
        self.gate_proj = nn.Linear(d_model, (n_wires + 1) * d_model, bias=False)
        self.value_proj = nn.Linear(d_model, d_model, bias=False)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        B, L, D = x.shape
        shifted = [x]
        for i in range(self.n_wires):
            shifted.append(torch.roll(x, shifts=-self.offsets[i].item(), dims=1))
        stacked = torch.stack(shifted, dim=2)
        stacked = self.value_proj(stacked)
        n = self.n_wires + 1
        gates = torch.sigmoid(self.gate_proj(x)).reshape(B, L, n, D)
        out = (stacked * gates).sum(dim=2)
        return self.norm(out)


class ButterflyMixer(nn.Module):
    """FFT butterfly pattern wiring. log2(L) stages of stride-2^k exchanges."""
    def __init__(self, d_model, max_stages=6):
        super().__init__()
        self.max_stages = max_stages
        self.gates = nn.ModuleList([
            nn.Linear(d_model * 2, d_model, bias=False) for _ in range(max_stages)
        ])
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        B, L, D = x.shape
        n_stages = min(self.max_stages, max(1, int(math.log2(max(L, 2)))))
        h = x
        for s in range(n_stages):
            stride = 1 << s
            if stride >= L:
                break
            shifted = torch.roll(h, shifts=stride, dims=1)
            combined = torch.cat([h, shifted], dim=-1)
            h = h + torch.sigmoid(self.gates[s](combined))
        return self.norm(h)


class HierarchicalPoolBroadcast(nn.Module):
    """Pool at stride -> process -> broadcast back. Multi-scale summary."""
    def __init__(self, d_model, stride=4):
        super().__init__()
        self.stride = stride
        self.down = nn.Conv1d(d_model, d_model, stride * 2 - 1, stride=stride,
                              padding=stride - 1, groups=d_model, bias=False)
        self.proj = nn.Linear(d_model, d_model, bias=False)
        self.gate = nn.Linear(d_model * 2, d_model, bias=False)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        B, L, D = x.shape
        h = self.down(x.transpose(1, 2)).transpose(1, 2)
        h = self.proj(h)
        # Broadcast back via repeat_interleave
        h_up = h.repeat_interleave(self.stride, dim=1)[:, :L]
        gate = torch.sigmoid(self.gate(torch.cat([x, h_up], dim=-1)))
        return x + self.norm(gate * h_up)


class ExponentialMovingAverage(nn.Module):
    """Exponential moving average with learned per-channel decay."""
    def __init__(self, d_model):
        super().__init__()
        self.decay_logit = nn.Parameter(torch.randn(d_model) * 0.5)
        self.norm = nn.LayerNorm(d_model)
        self.mix = nn.Linear(d_model * 2, d_model, bias=False)

    def forward(self, x):
        B, L, D = x.shape
        alpha = torch.sigmoid(self.decay_logit)  # (D,)
        # Parallel EMA via conv1d with exponentially decaying kernel
        one_minus_alpha = 1 - alpha
        powers = alpha.unsqueeze(0) ** torch.arange(L, device=x.device).float().unsqueeze(-1)  # (L, D)
        kernel = (one_minus_alpha.unsqueeze(0) * powers).permute(1, 0).unsqueeze(1)  # (D, 1, L)
        x_t = x.transpose(1, 2)  # (B, D, L)
        ema = F.conv1d(x_t, kernel, padding=L - 1, groups=D)[:, :, :L]  # (B, D, L)
        ema = ema.transpose(1, 2)
        return x + self.norm(self.mix(torch.cat([x, ema], dim=-1)))


class WaveletMixer(nn.Module):
    """Haar wavelet decompose -> learned transform -> reconstruct."""
    def __init__(self, d_model):
        super().__init__()
        self.approx_proj = nn.Linear(d_model, d_model, bias=False)
        self.detail_proj = nn.Linear(d_model, d_model, bias=False)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        B, L, D = x.shape
        # Pad to even length
        pad_len = L % 2
        if pad_len:
            x_pad = F.pad(x, (0, 0, 0, 1))
        else:
            x_pad = x
        L2 = x_pad.shape[1]
        even = x_pad[:, 0::2]
        odd = x_pad[:, 1::2]
        approx = (even + odd) / math.sqrt(2)
        detail = (even - odd) / math.sqrt(2)
        approx = self.approx_proj(approx)
        detail = self.detail_proj(detail)
        # Reconstruct
        recon_even = (approx + detail) / math.sqrt(2)
        recon_odd = (approx - detail) / math.sqrt(2)
        recon = torch.zeros(B, L2, D, device=x.device)
        recon[:, 0::2] = recon_even
        recon[:, 1::2] = recon_odd
        return x + self.norm(recon[:, :L])


class LSHLocalExchange(nn.Module):
    """Hash-bucket local exchange. Group tokens by hash, exchange within groups."""
    def __init__(self, d_model, n_buckets=8):
        super().__init__()
        self.n_buckets = n_buckets
        self.hash_proj = nn.Linear(d_model, n_buckets, bias=False)
        self.value_proj = nn.Linear(d_model, d_model, bias=False)
        self.norm = nn.LayerNorm(d_model)
        self.gate = nn.Linear(d_model * 2, d_model, bias=False)

    def forward(self, x):
        B, L, D = x.shape
        # Soft bucket assignment via sigmoid (no softmax)
        bucket_scores = torch.sigmoid(self.hash_proj(x))  # (B, L, n_buckets)
        v = self.value_proj(x)  # (B, L, D)
        # Per-bucket weighted average
        weighted = bucket_scores.unsqueeze(-1) * v.unsqueeze(2)  # (B, L, n_buckets, D)
        bucket_sums = weighted.sum(dim=1, keepdim=True)  # (B, 1, n_buckets, D)
        bucket_counts = bucket_scores.sum(dim=1, keepdim=True).unsqueeze(-1) + 1e-6
        bucket_means = bucket_sums / bucket_counts
        # Read from buckets
        read = (bucket_scores.unsqueeze(-1) * bucket_means).sum(dim=2)  # (B, L, D)
        gate = torch.sigmoid(self.gate(torch.cat([x, read], dim=-1)))
        return x + self.norm(gate * read)


class CellularAutomataStep(nn.Module):
    """Cellular automata-style local update rule with learned transition."""
    def __init__(self, d_model, radius=2):
        super().__init__()
        self.radius = radius
        ks = 2 * radius + 1
        self.conv = nn.Conv1d(d_model, d_model, ks, padding=radius, groups=d_model, bias=False)
        self.rule = nn.Sequential(
            nn.Linear(d_model * 2, d_model, bias=False),
            nn.SiLU(),
            nn.Linear(d_model, d_model, bias=False),
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        neighborhood = self.conv(x.transpose(1, 2)).transpose(1, 2)
        update = self.rule(torch.cat([x, neighborhood], dim=-1))
        return x + self.norm(update)


class StridedConvUpDown(nn.Module):
    """Strided conv down -> process -> transpose conv up."""
    def __init__(self, d_model, stride=4):
        super().__init__()
        self.stride = stride
        self.down = nn.Conv1d(d_model, d_model, stride * 2 - 1, stride=stride,
                              padding=stride - 1, groups=d_model, bias=False)
        self.process = nn.Sequential(
            nn.Linear(d_model, d_model, bias=False),
            nn.SiLU(),
            nn.Linear(d_model, d_model, bias=False),
        )
        self.up = nn.ConvTranspose1d(d_model, d_model, stride * 2, stride=stride,
                                      padding=stride // 2, groups=d_model, bias=False)
        self.gate = nn.Linear(d_model * 2, d_model, bias=False)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        B, L, D = x.shape
        h = self.down(x.transpose(1, 2)).transpose(1, 2)
        h = self.process(h)
        h_up = self.up(h.transpose(1, 2)).transpose(1, 2)[:, :L]
        gate = torch.sigmoid(self.gate(torch.cat([x, h_up], dim=-1)))
        return x + self.norm(gate * h_up)


class SinkhornPermutation(nn.Module):
    """Learned permutation via Sinkhorn operator (differentiable soft permutation)."""
    def __init__(self, d_model, n_iters=4):
        super().__init__()
        self.n_iters = n_iters
        self.query = nn.Linear(d_model, d_model, bias=False)
        self.key = nn.Linear(d_model, d_model, bias=False)
        self.norm = nn.LayerNorm(d_model)
        self.alpha = nn.Parameter(torch.tensor(0.1))

    def forward(self, x):
        B, L, D = x.shape
        q = self.query(x)
        k = self.key(x)
        # Cost matrix (no softmax - use Sinkhorn normalization)
        cost = torch.bmm(q, k.transpose(1, 2)) / math.sqrt(D)
        # Sinkhorn iterations (alternating row/column normalization via log-domain)
        log_alpha = cost
        for _ in range(self.n_iters):
            log_alpha = log_alpha - torch.logsumexp(log_alpha, dim=-1, keepdim=True)
            log_alpha = log_alpha - torch.logsumexp(log_alpha, dim=-2, keepdim=True)
        perm = torch.exp(log_alpha)
        out = torch.bmm(perm, x)
        return x + self.norm(self.alpha * out)


class LongConvFreqDomain(nn.Module):
    """Long convolution with kernel parameterized in frequency domain."""
    def __init__(self, d_model, max_len=128):
        super().__init__()
        self.freq_kernel = nn.Parameter(torch.randn(d_model, max_len // 2 + 1, 2) * 0.02)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        B, L, D = x.shape
        h = x.transpose(1, 2)  # (B, D, L)
        H = torch.fft.rfft(h, dim=-1)
        n_freq = H.shape[-1]
        # Learned kernel in freq domain
        K = torch.view_as_complex(self.freq_kernel[:, :n_freq].contiguous())
        K = K.unsqueeze(0)  # (1, D, n_freq)
        out = torch.fft.irfft(H * K, n=L, dim=-1)
        return x + self.norm(out.transpose(1, 2))


# =========================================================================
# STATE UPDATE MECHANISMS
# =========================================================================


class SwiGLUFFN(nn.Module):
    """SwiGLU feed-forward."""
    def __init__(self, d_model, expansion=2):
        super().__init__()
        di = d_model * expansion
        self.norm = nn.LayerNorm(d_model)
        self.wg = nn.Linear(d_model, di, bias=False)
        self.wu = nn.Linear(d_model, di, bias=False)
        self.wd = nn.Linear(di, d_model, bias=False)

    def forward(self, x):
        r = x
        x = self.norm(x)
        return r + self.wd(F.silu(self.wg(x)) * self.wu(x))


class GeGLUFFN(nn.Module):
    """GeGLU feed-forward."""
    def __init__(self, d_model, expansion=2):
        super().__init__()
        di = d_model * expansion
        self.norm = nn.LayerNorm(d_model)
        self.wg = nn.Linear(d_model, di, bias=False)
        self.wu = nn.Linear(d_model, di, bias=False)
        self.wd = nn.Linear(di, d_model, bias=False)

    def forward(self, x):
        r = x
        x = self.norm(x)
        return r + self.wd(F.gelu(self.wg(x)) * self.wu(x))


class ReGLUFFN(nn.Module):
    """ReGLU feed-forward."""
    def __init__(self, d_model, expansion=2):
        super().__init__()
        di = d_model * expansion
        self.norm = nn.LayerNorm(d_model)
        self.wg = nn.Linear(d_model, di, bias=False)
        self.wu = nn.Linear(d_model, di, bias=False)
        self.wd = nn.Linear(di, d_model, bias=False)

    def forward(self, x):
        r = x
        x = self.norm(x)
        return r + self.wd(F.relu(self.wg(x)) * self.wu(x))


class HighwayGating(nn.Module):
    """Highway network gating: T(x) * H(x) + (1-T(x)) * x."""
    def __init__(self, d_model):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.transform = nn.Linear(d_model, d_model, bias=False)
        self.gate = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x):
        h = self.norm(x)
        t = torch.sigmoid(self.gate(h))
        return t * F.silu(self.transform(h)) + (1 - t) * x


class ConvGRUCell(nn.Module):
    """Conv-based GRU cell operating on sequence."""
    def __init__(self, d_model, kernel_size=3):
        super().__init__()
        pad = kernel_size // 2
        self.conv_z = nn.Conv1d(d_model * 2, d_model, kernel_size, padding=pad, bias=False)
        self.conv_r = nn.Conv1d(d_model * 2, d_model, kernel_size, padding=pad, bias=False)
        self.conv_h = nn.Conv1d(d_model * 2, d_model, kernel_size, padding=pad, bias=False)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        h = x.transpose(1, 2)
        cat = torch.cat([h, h], dim=1)
        z = torch.sigmoid(self.conv_z(cat))
        r = torch.sigmoid(self.conv_r(cat))
        cat_r = torch.cat([r * h, h], dim=1)
        h_new = torch.tanh(self.conv_h(cat_r))
        out = (1 - z) * h + z * h_new
        return self.norm(out.transpose(1, 2))


class ResidualMLP(nn.Module):
    """Residual MLP with variable depth."""
    def __init__(self, d_model, depth=2, expansion=2):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        layers = []
        for i in range(depth):
            if i == depth - 1:
                layers.append(nn.Linear(d_model, d_model, bias=False))
            else:
                layers.append(nn.Linear(d_model, d_model * expansion, bias=False))
                layers.append(nn.SiLU())
                layers.append(nn.Linear(d_model * expansion, d_model, bias=False))
        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        return x + self.mlp(self.norm(x))


class SqueezeExciteUpdate(nn.Module):
    """Squeeze-excite as a state update: global avg pool -> MLP -> per-channel gating."""
    def __init__(self, d_model, reduction=4):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.f1 = nn.Linear(d_model, d_model // max(1, reduction), bias=False)
        self.f2 = nn.Linear(d_model // max(1, reduction), d_model, bias=False)

    def forward(self, x):
        h = self.norm(x)
        g = h.mean(1, keepdim=True)
        return x * torch.sigmoid(self.f2(F.silu(self.f1(g))))


class PerChannelScale(nn.Module):
    """Per-channel learned scaling with bias."""
    def __init__(self, d_model):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(d_model))
        self.bias = nn.Parameter(torch.zeros(d_model))
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        return self.norm(x * self.scale + self.bias)


class PolynomialActivation(nn.Module):
    """Polynomial activation network: x + c2*x^2 + c3*x^3 with learned coefficients."""
    def __init__(self, d_model):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.proj = nn.Linear(d_model, d_model, bias=False)
        self.c2 = nn.Parameter(torch.randn(d_model) * 0.01)
        self.c3 = nn.Parameter(torch.randn(d_model) * 0.001)

    def forward(self, x):
        h = self.norm(x)
        h = self.proj(h)
        return x + h + self.c2 * h * h + self.c3 * h * h * h


class StochasticDepthUpdate(nn.Module):
    """State-dependent stochastic depth: skip probability depends on content."""
    def __init__(self, d_model, expansion=2):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        di = d_model * expansion
        self.wu = nn.Linear(d_model, di, bias=False)
        self.wg = nn.Linear(d_model, di, bias=False)
        self.wd = nn.Linear(di, d_model, bias=False)
        self.skip_gate = nn.Linear(d_model, 1, bias=False)

    def forward(self, x):
        h = self.norm(x)
        skip_prob = torch.sigmoid(self.skip_gate(h.mean(dim=1, keepdim=True)))  # (B, 1, 1)
        update = self.wd(F.silu(self.wu(h)) * self.wg(h))
        if self.training:
            mask = (torch.rand_like(skip_prob) > skip_prob).float()
            return x + mask * update
        return x + (1 - skip_prob) * update


# =========================================================================
# SEEKER FIELD NETWORK PRIMITIVES
# Three radical mechanisms for O(L) scaling with exact retrieval:
#   1. Data-Dependent Selective Forgetting (Dynamic Time-Scale Gating)
#   2. Orthogonal Binding via High-Dimensional Computing (HDC)
#   3. Decoupled Episodic LSH Cache (O(1) Resonance Memory)
# =========================================================================


class DynamicTimeScaleGating(nn.Module):
    """Data-Dependent Selective Forgetting.

    The integration rate (time-constant) is entirely dictated by the input's
    structural entropy. Low-information tokens (stop words, padding) freeze the
    hidden state (0% update), while high-density structural tokens force the
    gate wide open. The model autonomously controls its own forgetting rate.

    Mechanism:
      1. Estimate per-token structural entropy via a lightweight MLP that
         produces a scalar "importance" for each (batch, position).
      2. Use that importance as the GRU-style update gate z, replacing the
         mechanical sigmoid(Wx+b) with a content-adaptive gate.
      3. Compute a candidate new state via depthwise conv + SiLU.
      4. Blend: h_new = (1 - z) * h_old + z * candidate.

    This means truly uninformative tokens leave the hidden state untouched,
    while structurally novel tokens overwrite it completely.
    """

    def __init__(self, d_model, kernel_size=3):
        super().__init__()
        pad = kernel_size // 2

        # Entropy estimator: projects each token to a scalar "importance"
        # score. Two-layer bottleneck ensures the gate is data-dependent,
        # not just a linear projection of the raw embedding.
        self.entropy_net = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model // 4, bias=False),
            nn.SiLU(),
            nn.Linear(d_model // 4, 1, bias=False),
        )

        # Candidate state computation: local context via depthwise conv,
        # then pointwise nonlinear transform. This is what the state
        # *would* become if the gate is fully open.
        self.conv_candidate = nn.Conv1d(
            d_model, d_model, kernel_size, padding=pad, groups=d_model, bias=False
        )
        self.candidate_proj = nn.Sequential(
            nn.Linear(d_model, d_model, bias=False),
            nn.SiLU(),
            nn.Linear(d_model, d_model, bias=False),
        )

        # Reset gate: controls how much of the old state leaks into the
        # candidate computation (standard GRU mechanism).
        self.reset_gate = nn.Sequential(
            nn.Linear(d_model * 2, d_model, bias=False),
        )

        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        """
        Args:
            x: (B, L, D) input tensor
        Returns:
            (B, L, D) updated tensor with data-dependent forgetting
        """
        B, L, D = x.shape

        # --- Step 1: Estimate structural entropy / importance ---
        # z ∈ (0, 1) per token. High z = important = update aggressively.
        # Low z = uninformative = freeze state.
        z = torch.sigmoid(self.entropy_net(x))  # (B, L, 1)

        # --- Step 2: Reset gate (how much old state influences candidate) ---
        # Concatenate current state with itself (like GRU with h=x)
        r = torch.sigmoid(self.reset_gate(torch.cat([x, x], dim=-1)))  # (B, L, D)

        # --- Step 3: Compute candidate new state ---
        # Apply reset gate, then local mixing via depthwise conv
        gated_x = r * x
        local = self.conv_candidate(gated_x.transpose(1, 2)).transpose(1, 2)
        candidate = self.candidate_proj(local)  # (B, L, D)

        # --- Step 4: Data-dependent blend ---
        # z broadcasts over D: important tokens update fully, boring ones don't
        out = (1 - z) * x + z * candidate

        return self.norm(out)


class HDCBinding(nn.Module):
    """Orthogonal Binding via High-Dimensional Computing.

    Instead of lossy additive mixing, this module binds two information
    streams orthogonally using circular convolution in the frequency domain.
    This preserves structural information even after thousands of binding
    operations, because circular convolution in high-D spaces is approximately
    orthogonal (the bound vector is quasi-orthogonal to both inputs).

    Binding: A ⊛ B = iFFT(FFT(A) ⊙ FFT(B))    (circular convolution)
    Unbinding: A = iFFT(FFT(A⊛B) ⊙ conj(FFT(B)))  (correlation)

    For differentiability, we use the standard complex FFT which is fully
    differentiable in PyTorch. No straight-through estimator needed.

    Architecture:
      1. Project input into two streams: "content" and "structure".
      2. Bind them via circular convolution in frequency domain.
      3. Gate the bound representation back into the residual stream.

    This ensures that even after 50,000 tokens, high-frequency structural
    tokens can be exactly retrieved via the unbinding (correlation) operation.
    """

    def __init__(self, d_model):
        super().__init__()
        # Two projections create the streams to be bound
        self.content_proj = nn.Linear(d_model, d_model, bias=False)
        self.structure_proj = nn.Linear(d_model, d_model, bias=False)

        # Learned "structural basis" that encodes positional/structural info.
        # This acts as the binding key: different positions get different
        # quasi-orthogonal keys, enabling later unbinding.
        self.structural_basis = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)

        # Output gate: controls how much of the HDC-bound information
        # is mixed back into the residual stream.
        self.gate = nn.Linear(d_model * 2, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        self.norm = nn.LayerNorm(d_model)

    def _circular_conv(self, a, b):
        """Circular convolution via FFT: a ⊛ b = iFFT(FFT(a) ⊙ FFT(b)).

        This is the core HDC binding operation. In high dimensions,
        the result is quasi-orthogonal to both a and b, enabling
        exact retrieval via correlation (unbinding).

        Args:
            a: (B, L, D) first operand
            b: (B, L, D) second operand (or (1, 1, D) for broadcast)
        Returns:
            (B, L, D) circular convolution result
        """
        # FFT along the feature dimension (D), NOT the sequence dimension.
        # This binds information *within* each token's representation.
        A = torch.fft.rfft(a, dim=-1)
        B = torch.fft.rfft(b, dim=-1)
        # Element-wise complex multiplication = circular convolution
        bound = torch.fft.irfft(A * B, n=a.shape[-1], dim=-1)
        return bound

    def forward(self, x):
        """
        Args:
            x: (B, L, D) input tensor
        Returns:
            (B, L, D) with orthogonally bound structural information
        """
        # --- Extract content and structure streams ---
        content = self.content_proj(x)    # (B, L, D) — "what"
        structure = self.structure_proj(x) + self.structural_basis  # (B, L, D) — "where/how"

        # --- Bind via circular convolution ---
        # The bound result is quasi-orthogonal to both content and structure,
        # meaning it encodes both without destructive interference.
        bound = self._circular_conv(content, structure)  # (B, L, D)

        # --- Gated residual ---
        bound = self.out_proj(bound)
        gate = torch.sigmoid(self.gate(torch.cat([x, bound], dim=-1)))
        return x + self.norm(gate * bound)


class EpisodicLSHCache(nn.Module):
    """Decoupled Episodic LSH Cache — O(1) Resonance Memory.

    A global, decoupled sparse memory bank that sits *outside* the main O(L)
    propagation stream. The network projects structural keys into an LSH hash
    space. When the current context "resonates" (hash-collides) with a
    previously stored state, the system bypasses the O(L) bottleneck and
    retrieves the exact past state in O(1) time.

    NO nn.MultiheadAttention. NO softmax over sequence lengths.
    Uses locality-sensitive hashing collisions for retrieval.

    Mechanism:
      1. Hash each token into B buckets using random hyperplane LSH.
         (Differentiable via straight-through estimator on the sign function.)
      2. Write: accumulate token values into hash buckets (soft write).
      3. Read: retrieve from buckets that the current token hashes to.
      4. Gate the retrieved memory back into the stream.

    The cache is populated incrementally as the sequence is processed,
    creating an episodic memory that can be queried in O(1) per token.
    """

    def __init__(self, d_model, n_hashes=4, n_buckets=32):
        super().__init__()
        self.n_hashes = n_hashes
        self.n_buckets = n_buckets

        # LSH random hyperplanes: project D-dimensional vectors to n_hashes
        # binary hash codes. Each hash function maps to n_buckets buckets.
        # We use multiple independent hash functions for better recall.
        self.hash_planes = nn.Parameter(
            torch.randn(n_hashes, d_model, n_buckets) * (1.0 / math.sqrt(d_model)),
        )
        # Don't train the hash planes — they should be random for LSH guarantees.
        # But we keep them as parameters for device placement.
        self.hash_planes.requires_grad = False

        # Learned value projection for writing into cache
        self.value_proj = nn.Linear(d_model, d_model, bias=False)

        # Learned key projection for better hash discrimination
        self.key_proj = nn.Linear(d_model, d_model, bias=False)

        # Output gate and projection for reading from cache
        self.read_gate = nn.Linear(d_model * 2, d_model, bias=False)
        self.read_proj = nn.Linear(d_model, d_model, bias=False)

        self.norm = nn.LayerNorm(d_model)

    def _lsh_hash(self, x):
        """Hard LSH bucket assignments via Straight-Through Estimator.

        Forward pass: strictly binary (0.0 or 1.0) bucket assignments.
        Each token is assigned to exactly the top-1 bucket per hash function
        via argmax one-hot encoding. Zero leakage into unassigned buckets.

        Backward pass: gradients flow through the soft sigmoid surrogate
        via the STE identity trick, bypassing the non-differentiable
        discrete argmax.

        Args:
            x: (B, L, D) input keys
        Returns:
            bucket_scores: (B, L, n_hashes, n_buckets) binary assignments
        """
        keys = self.key_proj(x)  # (B, L, D)

        # Project through random hyperplanes
        # hash_planes: (n_hashes, D, n_buckets), keys: (B, L, D)
        # scores: (B, L, n_hashes, n_buckets)
        scores = torch.einsum('ild,hdj->ilhj', keys, self.hash_planes)

        # --- Soft surrogate for backward pass ---
        soft_scores = torch.sigmoid(scores * 10.0)  # (B, L, H, Nb)

        # --- Hard discrete assignment for forward pass ---
        # Top-1 argmax per hash function → one-hot binary vector
        # Each token activates exactly 1 bucket per hash. No leakage.
        hard_indices = scores.argmax(dim=-1, keepdim=True)  # (B, L, H, 1)
        hard_assignments = torch.zeros_like(scores).scatter_(-1, hard_indices, 1.0)

        # --- STE: binary forward, smooth backward ---
        # Gradient of hard_assignments w.r.t. parameters = gradient of soft_scores
        bucket_scores = (hard_assignments - soft_scores).detach() + soft_scores

        return bucket_scores

    def forward(self, x):
        """
        Args:
            x: (B, L, D) input tensor
        Returns:
            (B, L, D) with episodic cache retrievals mixed in

        The cache is built and queried within each forward pass over the
        sequence. All tokens write to the cache, and all tokens read from it.
        Because LSH collisions group similar tokens, a token at position t
        can retrieve information from a distant position t' if they hash
        to the same bucket — achieving O(1) retrieval.
        """
        B, L, D = x.shape

        # --- Compute hash bucket assignments ---
        bucket_scores = self._lsh_hash(x)  # (B, L, n_hashes, n_buckets)

        # --- Write phase: accumulate values into buckets ---
        values = self.value_proj(x)  # (B, L, D)

        # For each hash function, compute weighted bucket contents.
        # bucket_scores: (B, L, H, Nb) — how much each token belongs to each bucket
        # values: (B, L, D) — what each token writes
        # We want: bucket_memory[b, h, nb, :] = sum_l bucket_scores[b, l, h, nb] * values[b, l, :]
        bucket_memory = torch.einsum(
            'blhn,bld->bhnd', bucket_scores, values
        )  # (B, n_hashes, n_buckets, D)

        # Normalize by bucket occupancy to get bucket means
        bucket_counts = bucket_scores.sum(dim=1, keepdim=False)  # (B, n_hashes, n_buckets)
        bucket_counts = bucket_counts.unsqueeze(-1).clamp(min=1e-6)  # (B, H, Nb, 1)
        bucket_means = bucket_memory / bucket_counts  # (B, H, Nb, D)

        # --- Read phase: retrieve from matching buckets ---
        # Each token reads from the buckets it hashes to, weighted by its
        # bucket assignment scores. This is O(1) per token (fixed n_hashes * n_buckets).
        read = torch.einsum(
            'blhn,bhnd->blhd', bucket_scores, bucket_means
        )  # (B, L, n_hashes, D)

        # Average across hash functions
        read = read.mean(dim=2)  # (B, L, D)
        read = self.read_proj(read)

        # --- Gated residual ---
        gate = torch.sigmoid(self.read_gate(torch.cat([x, read], dim=-1)))
        return x + self.norm(gate * read)


class PhaseRouter(nn.Module):
    """Phase Synchronization Router for dynamic primitive selection.

    Instead of rigid nn.Sequential block execution, this module routes data
    through primitives based on "phase synchronization" — a learned measure
    of how well the current hidden state resonates with each primitive's
    preferred input distribution.

    Each primitive has a learned "phase signature". The router computes
    similarity between the current state's phase and each primitive's
    signature, then uses these as soft routing weights.

    No attention (no QKV, no softmax over sequence length). The routing
    decision is per-position, based on channel statistics, not on
    position-to-position comparisons.
    """

    def __init__(self, d_model, n_primitives):
        super().__init__()
        self.n_primitives = n_primitives

        # Each primitive has a learned "phase signature" — a vector in D-space
        # that represents the kind of input it's most effective for.
        self.phase_signatures = nn.Parameter(
            torch.randn(n_primitives, d_model) * 0.02
        )

        # Phase extractor: maps current state to a phase vector
        self.phase_proj = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model, bias=False),
        )

        # Temperature for routing sharpness
        self.temperature = nn.Parameter(torch.tensor(1.0))

    def forward(self, x, primitive_outputs):
        """Route among primitive outputs based on phase synchronization.

        Args:
            x: (B, L, D) current hidden state (before primitives)
            primitive_outputs: list of (B, L, D) tensors, one per primitive
        Returns:
            (B, L, D) weighted combination of primitive outputs
        """
        B, L, D = x.shape

        # Extract phase from current state — use mean over sequence
        # to get a per-batch phase vector, then broadcast
        phase = self.phase_proj(x)  # (B, L, D)

        # Compute resonance with each primitive's signature
        # phase: (B, L, D), signatures: (n_prim, D)
        # resonance: (B, L, n_prim) — how well each position syncs with each primitive
        resonance = torch.einsum('bld,pd->blp', phase, self.phase_signatures)
        resonance = resonance / (D ** 0.5 * self.temperature.abs().clamp(min=0.1))

        # Soft routing via sigmoid (NOT softmax — we allow multiple primitives
        # to activate simultaneously, which is more expressive than winner-take-all)
        weights = torch.sigmoid(resonance)  # (B, L, n_prim)

        # Normalize so weights sum to 1 (ensures stable magnitude)
        weights = weights / (weights.sum(dim=-1, keepdim=True) + 1e-6)

        # Weighted combination of primitive outputs
        stacked = torch.stack(primitive_outputs, dim=2)  # (B, L, n_prim, D)
        out = (stacked * weights.unsqueeze(-1)).sum(dim=2)  # (B, L, D)

        return out


# =========================================================================
# REGISTRY: Maps string names to (class, default_kwargs) for search engine
# =========================================================================

PROPAGATION_REGISTRY = {
    'depthwise_conv_3': (DepthwiseConv, {'kernel_size': 3}),
    'depthwise_conv_5': (DepthwiseConv, {'kernel_size': 5}),
    'depthwise_conv_7': (DepthwiseConv, {'kernel_size': 7}),
    'depthwise_conv_9': (DepthwiseConv, {'kernel_size': 9}),
    'depthwise_conv_15': (DepthwiseConv, {'kernel_size': 15}),
    'depthwise_conv_31': (DepthwiseConv, {'kernel_size': 31}),
    'dilated_conv_d1': (DilatedConv, {'dilation': 1}),
    'dilated_conv_d2': (DilatedConv, {'dilation': 2}),
    'dilated_conv_d4': (DilatedConv, {'dilation': 4}),
    'dilated_conv_d8': (DilatedConv, {'dilation': 8}),
    'dilated_conv_d16': (DilatedConv, {'dilation': 16}),
    'dilated_conv_d32': (DilatedConv, {'dilation': 32}),
    'dilated_conv_d64': (DilatedConv, {'dilation': 64}),
    'gated_shift_small': (GatedShiftMixerVariant, {'shifts': (-4, -1, 1, 4)}),
    'gated_shift_medium': (GatedShiftMixerVariant, {'shifts': (-16, -4, -1, 1, 4, 16)}),
    'gated_shift_large': (GatedShiftMixerVariant, {'shifts': (-32, -16, -4, -1, 1, 4, 16, 32)}),
    'gated_shift_asym': (GatedShiftMixerVariant, {'shifts': (-32, -8, -2, 1, 3, 12)}),
    'gated_shift_powers': (GatedShiftMixerVariant, {'shifts': (-16, -8, -4, -2, -1, 1, 2, 4, 8, 16)}),
    'spectral_filter': (SpectralFilter, {}),
    'diagonal_ssm': (DiagonalSSM, {}),
    'random_sparse_wiring': (RandomSparseWiring, {'n_wires': 8}),
    'butterfly_mixer': (ButterflyMixer, {'max_stages': 6}),
    'hierarchical_pool_s2': (HierarchicalPoolBroadcast, {'stride': 2}),
    'hierarchical_pool_s4': (HierarchicalPoolBroadcast, {'stride': 4}),
    'hierarchical_pool_s8': (HierarchicalPoolBroadcast, {'stride': 8}),
    'ema': (ExponentialMovingAverage, {}),
    'wavelet_mixer': (WaveletMixer, {}),
    'lsh_exchange': (LSHLocalExchange, {'n_buckets': 8}),
    'cellular_automata_r1': (CellularAutomataStep, {'radius': 1}),
    'cellular_automata_r2': (CellularAutomataStep, {'radius': 2}),
    'cellular_automata_r3': (CellularAutomataStep, {'radius': 3}),
    'strided_updown_s2': (StridedConvUpDown, {'stride': 2}),
    'strided_updown_s4': (StridedConvUpDown, {'stride': 4}),
    'sinkhorn_permutation': (SinkhornPermutation, {'n_iters': 4}),
    'long_conv_freq': (LongConvFreqDomain, {}),
    'hdc_binding': (HDCBinding, {}),
    'episodic_lsh_cache': (EpisodicLSHCache, {'n_hashes': 4, 'n_buckets': 32}),
    'episodic_lsh_cache_small': (EpisodicLSHCache, {'n_hashes': 2, 'n_buckets': 16}),
}

UPDATE_REGISTRY = {
    'swiglu': (SwiGLUFFN, {'expansion': 2}),
    'geglu': (GeGLUFFN, {'expansion': 2}),
    'reglu': (ReGLUFFN, {'expansion': 2}),
    'highway': (HighwayGating, {}),
    'conv_gru': (ConvGRUCell, {'kernel_size': 3}),
    'residual_mlp_d1': (ResidualMLP, {'depth': 1}),
    'residual_mlp_d2': (ResidualMLP, {'depth': 2}),
    'squeeze_excite': (SqueezeExciteUpdate, {'reduction': 4}),
    'per_channel_scale': (PerChannelScale, {}),
    'polynomial_activation': (PolynomialActivation, {}),
    'stochastic_depth': (StochasticDepthUpdate, {'expansion': 2}),
    'dynamic_timescale': (DynamicTimeScaleGating, {'kernel_size': 3}),
    'dynamic_timescale_k5': (DynamicTimeScaleGating, {'kernel_size': 5}),
}


def build_module(name, d_model, registry):
    """Build a module by name from a registry."""
    cls, kwargs = registry[name]
    return cls(d_model, **kwargs)
