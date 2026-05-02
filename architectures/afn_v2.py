"""[EXPERIMENTAL] 
AFN v2 — fixes from v1 failure analysis.

Changes from v1:
1. SparseRouting REMOVED (sort breaks gradient)
2. Replaced with SqueezeExcite: global avg pool → MLP → per-channel gate
   O(L) global context, full gradient flow, zero attention.
3. Dilations widened: (1, 4, 16) → covers 16 positions per step
4. NCA steps increased per layer
5. ContentGatedConv: kernel weights modulated by cell state
"""

import math
from typing import Sequence
import torch
import torch.nn as nn
import torch.nn.functional as F


class PerceptionFilter(nn.Module):
    def __init__(self, d, k=7, nf=3):
        super().__init__()
        self.convs = nn.ModuleList([
            nn.Conv1d(d, d, k, padding=k//2, groups=d, bias=False) for _ in range(nf)])
        self.pw = nn.Conv1d(d*nf, d, 1, bias=False)
        self.norm = nn.LayerNorm(d)
    def forward(self, x):
        h = x.transpose(1,2)
        h = torch.cat([c(h) for c in self.convs], dim=1)
        return self.norm(self.pw(h).transpose(1,2))


class ReactionGate(nn.Module):
    def __init__(self, d, exp=2, drop=0.0):
        super().__init__()
        di = d*exp
        self.wu = nn.Linear(2*d, di, bias=False)
        self.wg = nn.Linear(2*d, di, bias=False)
        self.wd = nn.Linear(di, d, bias=False)
        self.drop = nn.Dropout(drop)
        self.alpha = nn.Parameter(torch.full((d,), 0.1))
    def forward(self, x, p):
        cat = torch.cat([x, p], -1)
        return x + self.alpha * self.wd(self.drop(
            torch.sigmoid(self.wg(cat)) * F.silu(self.wu(cat))))


class MultiRateDiffusion(nn.Module):
    def __init__(self, d, dilations=(1,4,16), ks=3):
        super().__init__()
        self.convs = nn.ModuleList([
            nn.Conv1d(d, d, ks, padding=dil*(ks//2), dilation=dil,
                      groups=d, bias=False) for dil in dilations])
        self.sel = nn.Linear(d, len(dilations), bias=False)
        self.strength = nn.Parameter(torch.tensor(0.1))
    def forward(self, x):
        B,L,D = x.shape
        h = x.transpose(1,2)
        diffs = torch.stack([c(h) for c in self.convs], dim=-1)
        w = torch.softmax(self.sel(x), dim=-1).unsqueeze(1)
        combined = (diffs * w).sum(-1).transpose(1,2)
        return x + self.strength * (combined - x)


class NCAStep(nn.Module):
    def __init__(self, d, k=7, dilations=(1,4,16), exp=2, drop=0.0):
        super().__init__()
        self.perceive = PerceptionFilter(d, k)
        self.react = ReactionGate(d, exp, drop)
        self.diffuse = MultiRateDiffusion(d, dilations)
    def forward(self, x):
        return self.diffuse(self.react(x, self.perceive(x)))


class SqueezeExcite(nn.Module):
    """Global context via average pooling + MLP + gating.
    
    O(L) complexity. Full gradient flow. Zero attention.
    Gives every cell awareness of the global state.
    """
    def __init__(self, d, reduction=4):
        super().__init__()
        self.norm = nn.LayerNorm(d)
        self.fc1 = nn.Linear(d, d // reduction, bias=False)
        self.fc2 = nn.Linear(d // reduction, d, bias=False)
    def forward(self, x):
        # x: (B, L, D)
        h = self.norm(x)
        # Global average pool
        glob = h.mean(dim=1, keepdim=True)  # (B, 1, D)
        # MLP → gate
        gate = torch.sigmoid(self.fc2(F.silu(self.fc1(glob))))  # (B, 1, D)
        return x * gate  # broadcast to all positions


class ContentGatedConv(nn.Module):
    """Convolution where kernel weights are modulated by content.
    
    Standard conv: same kernel everywhere.
    This: kernel amplitude modulated per-position by cell state.
    Gives content-dependent information routing without attention.
    """
    def __init__(self, d, ks=15):
        super().__init__()
        self.conv = nn.Conv1d(d, d, ks, padding=ks//2, groups=d, bias=False)
        self.gate_proj = nn.Linear(d, d, bias=False)
        self.norm = nn.LayerNorm(d)
    def forward(self, x):
        h = self.conv(x.transpose(1,2)).transpose(1,2)
        gate = torch.sigmoid(self.gate_proj(x))
        return self.norm(x + gate * h)


class CoarseNCA(nn.Module):
    def __init__(self, d, stride=4, n_steps=2, k=5, dilations=(1,4), drop=0.0):
        super().__init__()
        self.stride = stride
        self.down = nn.Conv1d(d, d, stride*2-1, stride=stride,
                              padding=stride-1, groups=d, bias=False)
        self.dp = nn.Conv1d(d, d, 1, bias=False)
        self.dn = nn.LayerNorm(d)
        self.step = NCAStep(d, k, dilations, 2, drop)
        self.n_steps = n_steps
        self.up = nn.ConvTranspose1d(d, d, stride*2, stride=stride,
                                      padding=stride//2, groups=d, bias=False)
        self.up_proj = nn.Conv1d(d, d, 1, bias=False)
        self.gate = nn.Linear(2*d, d, bias=False)
    def forward(self, x):
        B,L,D = x.shape
        h = self.dn(self.dp(self.down(x.transpose(1,2))).transpose(1,2))
        for _ in range(self.n_steps):
            h = self.step(h)
        hu = self.up_proj(self.up(h.transpose(1,2))).transpose(1,2)[:,:L]
        cat = torch.cat([x, hu], -1)
        return torch.sigmoid(self.gate(cat)) * hu


class GatedFFN(nn.Module):
    def __init__(self, d, exp=2, drop=0.0):
        super().__init__()
        di = d*exp
        self.norm = nn.LayerNorm(d)
        self.wg = nn.Linear(d, di, bias=False)
        self.wu = nn.Linear(d, di, bias=False)
        self.wd = nn.Linear(di, d, bias=False)
        self.drop = nn.Dropout(drop)
    def forward(self, x):
        r = x; x = self.norm(x)
        return r + self.drop(self.wd(F.silu(self.wg(x)) * self.wu(x)))


class AFN2Layer(nn.Module):
    def __init__(self, d, nca_steps=4, nca_k=7, dilations=(1,4,16),
                 coarse_stride=4, coarse_steps=2, conv_ks=15,
                 ffn_exp=2, drop=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(d)
        self.nca = NCAStep(d, nca_k, dilations, 2, drop)
        self.nca_steps = nca_steps
        
        self.norm2 = nn.LayerNorm(d)
        self.se = SqueezeExcite(d)
        
        self.norm3 = nn.LayerNorm(d)
        self.cgconv = ContentGatedConv(d, conv_ks)
        
        self.norm4 = nn.LayerNorm(d)
        self.coarse = CoarseNCA(d, coarse_stride, coarse_steps,
                                 dilations=dilations[:2], drop=drop)
        
        self.ffn = GatedFFN(d, ffn_exp, drop)

    def forward(self, x):
        # Local NCA dynamics
        h = self.norm1(x)
        for _ in range(self.nca_steps):
            h = self.nca(h)
        x = x + (h - self.norm1(x))
        
        # Global squeeze-excite
        x = self.se(self.norm2(x))
        
        # Content-gated wide conv
        x = self.cgconv(self.norm3(x))
        
        # Coarse-scale NCA
        x = x + self.coarse(self.norm4(x))
        
        # FFN
        x = self.ffn(x)
        return x


class AFN2(nn.Module):
    def __init__(self, vocab_size=256, d_model=64, n_layers=2,
                 nca_steps=4, nca_k=7, dilations=(1,4,16),
                 coarse_stride=4, coarse_steps=2, conv_ks=15,
                 ffn_exp=2, drop=0.0, max_len=4096):
        super().__init__()
        self.te = nn.Embedding(vocab_size, d_model)
        self.pe = nn.Embedding(max_len, d_model)
        self.en = nn.LayerNorm(d_model)
        self.ed = nn.Dropout(drop)
        self.layers = nn.ModuleList([
            AFN2Layer(d_model, nca_steps, nca_k, dilations,
                      coarse_stride, coarse_steps, conv_ks, ffn_exp, drop)
            for _ in range(n_layers)])
        self.fn = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)
        self.head.weight = self.te.weight
        self._init()
    def _init(self):
        for m in self.modules():
            if isinstance(m, nn.Linear): nn.init.xavier_uniform_(m.weight)
            elif isinstance(m, nn.Embedding): nn.init.normal_(m.weight, std=0.02)
    def forward(self, x):
        B,L = x.shape
        p = torch.arange(L, device=x.device).unsqueeze(0)
        h = self.ed(self.en(self.te(x) + self.pe(p)))
        for layer in self.layers: h = layer(h)
        return self.head(self.fn(h))
    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
