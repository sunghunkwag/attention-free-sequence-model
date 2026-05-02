"""[EXPERIMENTAL] AFN v4 with Fourier-domain gated sequence mixing (O(L log L))."""
import torch
import torch.nn as nn
from architectures.afn_v3 import NCAStep, CoarseNCA, GatedFFN, SqueezeExcite


class FourierGateMixer(nn.Module):
    def __init__(self, d_model: int, max_len: int = 4096):
        super().__init__()
        self.freq_gate = nn.Parameter(torch.zeros(max_len // 2 + 1, d_model))
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, l, d = x.shape
        xf = torch.fft.rfft(x.transpose(1, 2), dim=-1)
        g = torch.sigmoid(self.freq_gate[: xf.shape[-1], :].transpose(0, 1)).unsqueeze(0)
        y = torch.fft.irfft(xf * g, n=l, dim=-1).transpose(1, 2)
        return self.norm(y)


class AFN4Layer(nn.Module):
    def __init__(self, d, nca_steps=3, nca_k=5, dilations=(1,4,16), coarse_stride=4, coarse_steps=1, ffn_exp=2, drop=0.0, max_len=4096):
        super().__init__()
        self.norm1 = nn.LayerNorm(d)
        self.nca = NCAStep(d, nca_k, dilations, 2, drop)
        self.nca_steps = nca_steps
        self.norm2 = nn.LayerNorm(d)
        self.fourier = FourierGateMixer(d, max_len=max_len)
        self.norm3 = nn.LayerNorm(d)
        self.se = SqueezeExcite(d)
        self.norm4 = nn.LayerNorm(d)
        self.coarse = CoarseNCA(d, coarse_stride, coarse_steps, dilations=dilations[:2], drop=drop)
        self.ffn = GatedFFN(d, ffn_exp, drop)

    def forward(self, x):
        h = self.norm1(x)
        for _ in range(self.nca_steps):
            h = self.nca(h)
        x = x + h
        update = self.fourier(self.norm2(x))
        x = x + update
        x = x + self.se(self.norm3(x))
        x = x + self.coarse(self.norm4(x))
        return self.ffn(x)


class AFN4(nn.Module):
    def __init__(self, vocab_size=256, d_model=64, n_layers=2, nca_steps=3, nca_k=5, dilations=(1,4,16), coarse_stride=4, coarse_steps=1, ffn_exp=2, drop=0.0, max_len=4096):
        super().__init__()
        self.te = nn.Embedding(vocab_size, d_model)
        self.pe = nn.Embedding(max_len, d_model)
        self.en = nn.LayerNorm(d_model)
        self.ed = nn.Dropout(drop)
        self.layers = nn.ModuleList([AFN4Layer(d_model, nca_steps, nca_k, dilations, coarse_stride, coarse_steps, ffn_exp, drop, max_len) for _ in range(n_layers)])
        self.fn = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)
        self.head.weight = self.te.weight

    def forward(self, x):
        b, l = x.shape
        p = torch.arange(l, device=x.device).unsqueeze(0)
        h = self.ed(self.en(self.te(x) + self.pe(p)))
        for layer in self.layers:
            h = layer(h)
        return self.head(self.fn(h))

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
