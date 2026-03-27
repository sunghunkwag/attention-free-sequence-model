"""
AFN v3 — adds GatedShiftMixer for content-preserving long-range transfer.

v2 failure: diffusion blurs content. SqueezeExcite loses identity.
v3 fix: GatedShiftMixer = shift register with content-dependent gating.

No attention. Each position sees fixed-offset copies of the sequence
and learns (via content-dependent gating) which offsets to listen to.
Content travels without mixing.
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
            nn.Conv1d(d,d,k,padding=k//2,groups=d,bias=False) for _ in range(nf)])
        self.pw = nn.Conv1d(d*nf,d,1,bias=False)
        self.norm = nn.LayerNorm(d)
    def forward(self, x):
        h = x.transpose(1,2)
        return self.norm(self.pw(torch.cat([c(h) for c in self.convs],1)).transpose(1,2))


class ReactionGate(nn.Module):
    def __init__(self, d, exp=2, drop=0.0):
        super().__init__()
        di=d*exp
        self.wu=nn.Linear(2*d,di,bias=False);self.wg=nn.Linear(2*d,di,bias=False)
        self.wd=nn.Linear(di,d,bias=False);self.drop=nn.Dropout(drop)
        self.alpha=nn.Parameter(torch.full((d,),0.1))
    def forward(self, x, p):
        cat=torch.cat([x,p],-1)
        return x+self.alpha*self.wd(self.drop(torch.sigmoid(self.wg(cat))*F.silu(self.wu(cat))))


class MultiRateDiffusion(nn.Module):
    def __init__(self, d, dilations=(1,4,16), ks=3):
        super().__init__()
        self.convs=nn.ModuleList([nn.Conv1d(d,d,ks,padding=dil*(ks//2),dilation=dil,groups=d,bias=False) for dil in dilations])
        self.sel=nn.Linear(d,len(dilations),bias=False)
        self.strength=nn.Parameter(torch.tensor(0.1))
    def forward(self, x):
        B,L,D=x.shape;h=x.transpose(1,2)
        diffs=torch.stack([c(h) for c in self.convs],dim=-1)
        w=torch.softmax(self.sel(x),dim=-1).unsqueeze(1)
        combined=(diffs*w).sum(-1).transpose(1,2)
        return x+self.strength*(combined-x)


class NCAStep(nn.Module):
    def __init__(self, d, k=7, dilations=(1,4,16), exp=2, drop=0.0):
        super().__init__()
        self.perceive=PerceptionFilter(d,k)
        self.react=ReactionGate(d,exp,drop)
        self.diffuse=MultiRateDiffusion(d,dilations)
    def forward(self, x):
        return self.diffuse(self.react(x, self.perceive(x)))


class GatedShiftMixer(nn.Module):
    """Content-preserving long-range transfer via gated shift register.
    
    Creates shifted copies of the input at fixed offsets.
    Each position selects which shifts to accept based on its content.
    
    Unlike attention:
    - No Q/K/V projections
    - No softmax
    - Fixed wiring (shift offsets are hardcoded, not learned)
    - Selection is per-channel gating, not per-token weighting
    
    Like attention:
    - Content at position i can directly access content at position i±offset
    - No information loss from mixing/averaging
    
    O(L · n_shifts · D) — linear in L.
    """
    def __init__(self, d_model: int, shifts: Sequence[int] = (-32,-16,-4,-1,1,4,16,32)):
        super().__init__()
        self.shifts = list(shifts)
        n = len(shifts) + 1  # +1 for self
        
        # Content-dependent gate: which shifts to listen to
        self.gate_proj = nn.Linear(d_model, n * d_model, bias=False)
        
        # Value projection per shift (lightweight)
        self.value_proj = nn.Linear(d_model, d_model, bias=False)
        
        self.norm = nn.LayerNorm(d_model)
        self.n = n
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, L, D) → (B, L, D)"""
        B, L, D = x.shape
        
        # Create shifted copies
        shifted = [x]  # self
        for s in self.shifts:
            shifted.append(torch.roll(x, shifts=-s, dims=1))
        
        # Stack: (B, L, n, D)
        stacked = torch.stack(shifted, dim=2)
        
        # Project values
        stacked = self.value_proj(stacked)  # (B, L, n, D)
        
        # Content-dependent gates: (B, L, n*D) → reshape → (B, L, n, D)
        gates = torch.sigmoid(self.gate_proj(x)).reshape(B, L, self.n, D)
        
        # Gated selection
        out = (stacked * gates).sum(dim=2)  # (B, L, D)
        
        return self.norm(out)


class SqueezeExcite(nn.Module):
    def __init__(self, d, r=4):
        super().__init__()
        self.norm=nn.LayerNorm(d)
        self.f1=nn.Linear(d,d//r,bias=False);self.f2=nn.Linear(d//r,d,bias=False)
    def forward(self, x):
        h=self.norm(x);g=h.mean(1,keepdim=True)
        return x*torch.sigmoid(self.f2(F.silu(self.f1(g))))


class CoarseNCA(nn.Module):
    def __init__(self, d, stride=4, n_steps=2, k=5, dilations=(1,4), drop=0.0):
        super().__init__()
        self.stride=stride
        self.down=nn.Conv1d(d,d,stride*2-1,stride=stride,padding=stride-1,groups=d,bias=False)
        self.dp=nn.Conv1d(d,d,1,bias=False);self.dn=nn.LayerNorm(d)
        self.step=NCAStep(d,k,dilations,2,drop);self.n_steps=n_steps
        self.up=nn.ConvTranspose1d(d,d,stride*2,stride=stride,padding=stride//2,groups=d,bias=False)
        self.up_proj=nn.Conv1d(d,d,1,bias=False)
        self.gate=nn.Linear(2*d,d,bias=False)
    def forward(self, x):
        B,L,D=x.shape;h=self.dn(self.dp(self.down(x.transpose(1,2))).transpose(1,2))
        for _ in range(self.n_steps): h=self.step(h)
        hu=self.up_proj(self.up(h.transpose(1,2))).transpose(1,2)[:,:L]
        return torch.sigmoid(self.gate(torch.cat([x,hu],-1)))*hu


class GatedFFN(nn.Module):
    def __init__(self, d, exp=2, drop=0.0):
        super().__init__()
        di=d*exp;self.norm=nn.LayerNorm(d)
        self.wg=nn.Linear(d,di,bias=False);self.wu=nn.Linear(d,di,bias=False)
        self.wd=nn.Linear(di,d,bias=False);self.drop=nn.Dropout(drop)
    def forward(self, x):
        r=x;x=self.norm(x);return r+self.drop(self.wd(F.silu(self.wg(x))*self.wu(x)))


class AFN3Layer(nn.Module):
    def __init__(self, d, nca_steps=3, nca_k=5, dilations=(1,4,16),
                 shifts=(-32,-16,-4,-1,1,4,16,32),
                 coarse_stride=4, coarse_steps=1, ffn_exp=2, drop=0.0):
        super().__init__()
        self.norm1=nn.LayerNorm(d)
        self.nca=NCAStep(d,nca_k,dilations,2,drop);self.nca_steps=nca_steps
        
        self.norm2=nn.LayerNorm(d)
        self.shift_mixer=GatedShiftMixer(d,shifts)
        
        self.norm3=nn.LayerNorm(d)
        self.se=SqueezeExcite(d)
        
        self.norm4=nn.LayerNorm(d)
        self.coarse=CoarseNCA(d,coarse_stride,coarse_steps,dilations=dilations[:2],drop=drop)
        
        self.ffn=GatedFFN(d,ffn_exp,drop)

    def forward(self, x):
        h=self.norm1(x)
        for _ in range(self.nca_steps): h=self.nca(h)
        x=x+(h-self.norm1(x))
        
        x=x+self.shift_mixer(self.norm2(x))  # content-preserving long-range
        x=self.se(self.norm3(x))               # global conditioning
        x=x+self.coarse(self.norm4(x))         # multi-scale
        x=self.ffn(x)
        return x


class AFN3(nn.Module):
    def __init__(self, vocab_size=256, d_model=64, n_layers=2,
                 nca_steps=3, nca_k=5, dilations=(1,4,16),
                 shifts=(-32,-16,-4,-1,1,4,16,32),
                 coarse_stride=4, coarse_steps=1,
                 ffn_exp=2, drop=0.0, max_len=4096):
        super().__init__()
        self.te=nn.Embedding(vocab_size,d_model);self.pe=nn.Embedding(max_len,d_model)
        self.en=nn.LayerNorm(d_model);self.ed=nn.Dropout(drop)
        self.layers=nn.ModuleList([
            AFN3Layer(d_model,nca_steps,nca_k,dilations,shifts,
                      coarse_stride,coarse_steps,ffn_exp,drop)
            for _ in range(n_layers)])
        self.fn=nn.LayerNorm(d_model);self.head=nn.Linear(d_model,vocab_size,bias=False)
        self.head.weight=self.te.weight
        for m in self.modules():
            if isinstance(m,nn.Linear):nn.init.xavier_uniform_(m.weight)
            elif isinstance(m,nn.Embedding):nn.init.normal_(m.weight,std=0.02)
    def forward(self, x):
        B,L=x.shape;p=torch.arange(L,device=x.device).unsqueeze(0)
        h=self.ed(self.en(self.te(x)+self.pe(p)))
        for layer in self.layers:h=layer(h)
        return self.head(self.fn(h))
    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
