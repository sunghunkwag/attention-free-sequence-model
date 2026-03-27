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
}


def build_module(name, d_model, registry):
    """Build a module by name from a registry."""
    cls, kwargs = registry[name]
    return cls(d_model, **kwargs)
