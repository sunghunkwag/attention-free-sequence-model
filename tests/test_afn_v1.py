"""Tests + Benchmark for Adaptive Field Network."""
import pytest, torch, time
import torch.nn.functional as F
from architectures.afn_v1 import (AFN, AFNLayer, CoarseNCA, GatedFFN, MultiRateDiffusion,
                  NCAStep, PerceptionFilter, ReactionGate, SparseRouting,
                  afn_tiny)

B, D = 2, 32
torch.manual_seed(42)
def _randn(B,L,D): return torch.randn(B,L,D)
def _ids(B,L,V=100): return torch.randint(0,V,(B,L))


class TestPerceptionFilter:
    def test_shape(self):
        assert PerceptionFilter(D)(_randn(B,20,D)).shape == (B,20,D)
    def test_grad(self):
        x = _randn(1,16,D).requires_grad_(True)
        PerceptionFilter(D)(x).sum().backward()
        assert x.grad is not None

class TestReactionGate:
    def test_shape(self):
        rg = ReactionGate(D)
        assert rg(_randn(B,10,D), _randn(B,10,D)).shape == (B,10,D)

class TestMultiRateDiffusion:
    def test_shape(self):
        d = MultiRateDiffusion(D, dilations=(1,4))
        assert d(_randn(B,20,D)).shape == (B,20,D)
    def test_content_adaptive(self):
        d = MultiRateDiffusion(D, dilations=(1,4))
        x = _randn(B,16,D)
        w = torch.softmax(d.rate_selector(x), dim=-1)
        assert torch.allclose(w.sum(dim=-1), torch.ones(B,16), atol=1e-6)

class TestNCAStep:
    def test_shape(self):
        s = NCAStep(D, kernel_size=5, dilations=(1,4))
        assert s(_randn(B,20,D)).shape == (B,20,D)
    def test_grad(self):
        s = NCAStep(D, kernel_size=5, dilations=(1,4))
        x = _randn(1,12,D).requires_grad_(True)
        s(x).sum().backward()
        assert x.grad is not None

class TestSparseRouting:
    def test_shape(self):
        sr = SparseRouting(D, bucket_size=4, n_routes=2)
        assert sr(_randn(B,20,D)).shape == (B,20,D)
    def test_no_attention(self):
        sr = SparseRouting(D)
        for n, m in sr.named_modules():
            assert not isinstance(m, torch.nn.MultiheadAttention)
    def test_grad(self):
        sr = SparseRouting(D, bucket_size=4)
        x = _randn(1,16,D).requires_grad_(True)
        sr(x).sum().backward()
        assert x.grad is not None
    def test_content_dependent(self):
        """Different content should produce different routing."""
        sr = SparseRouting(D, bucket_size=4)
        sr.eval()
        x1 = torch.randn(1, 16, D)
        x2 = torch.randn(1, 16, D) * 5
        y1 = sr(x1); y2 = sr(x2)
        assert not torch.allclose(y1, y2)

class TestCoarseNCA:
    def test_shape(self):
        c = CoarseNCA(D, stride=4, n_steps=2, dilations=(1,4))
        assert c(_randn(B,32,D)).shape == (B,32,D)
    def test_short_seq(self):
        c = CoarseNCA(D, stride=4, n_steps=2, dilations=(1,))
        assert c(_randn(B,8,D)).shape == (B,8,D)
    def test_grad(self):
        c = CoarseNCA(D, stride=4, n_steps=2, dilations=(1,))
        x = _randn(1,16,D).requires_grad_(True)
        c(x).sum().backward()
        assert x.grad is not None

class TestAFNLayer:
    def test_shape(self):
        layer = AFNLayer(D, nca_steps=2, nca_kernel=5, nca_dilations=(1,4),
                          bucket_size=4, coarse_stride=4, coarse_steps=1)
        assert layer(_randn(B,32,D)).shape == (B,32,D)
    def test_grad_all(self):
        layer = AFNLayer(D, nca_steps=2, nca_kernel=5, nca_dilations=(1,4),
                          bucket_size=4, coarse_stride=4, coarse_steps=1)
        x = _randn(1,16,D).requires_grad_(True)
        layer(x).sum().backward()
        assert x.grad is not None
        for name, p in layer.named_parameters():
            if p.requires_grad:
                assert p.grad is not None, f"No grad: {name}"

class TestAFN:
    def test_output_shape(self):
        m = afn_tiny(vocab_size=100)
        assert m(_ids(B,32,100)).shape == (B,32,100)
    def test_grad_e2e(self):
        m = afn_tiny(vocab_size=100)
        m(_ids(1,16,100)).sum().backward()
        assert m.tok_emb.weight.grad is not None
    def test_weight_tying(self):
        m = afn_tiny(vocab_size=100)
        assert m.head.weight is m.tok_emb.weight
    def test_no_attention(self):
        m = afn_tiny(vocab_size=100)
        for n, mod in m.named_modules():
            assert not isinstance(mod, torch.nn.MultiheadAttention), n
    def test_seq_one(self):
        m = afn_tiny(vocab_size=100)
        assert m(_ids(B,1,100)).shape == (B,1,100)
    def test_param_count(self):
        m = afn_tiny(vocab_size=100)
        n = m.count_parameters()
        assert 10_000 < n < 10_000_000, f"Params: {n}"


# ═══════════════════════════════════════════════════════════════════════════
# Benchmark: AFN vs Transformer vs NCA-LM
# ═══════════════════════════════════════════════════════════════════════════

class TransformerBaseline(torch.nn.Module):
    def __init__(self, vs, d=64, nl=2, nh=4, ml=64):
        super().__init__()
        self.te = torch.nn.Embedding(vs, d)
        self.pe = torch.nn.Embedding(ml, d)
        self.ls = torch.nn.ModuleList()
        for _ in range(nl):
            self.ls.append(torch.nn.ModuleDict({
                'n1': torch.nn.LayerNorm(d),
                'a': torch.nn.MultiheadAttention(d, nh, batch_first=True),
                'n2': torch.nn.LayerNorm(d),
                'f': torch.nn.Sequential(torch.nn.Linear(d,4*d), torch.nn.GELU(),
                                          torch.nn.Linear(4*d,d)),
            }))
        self.fn = torch.nn.LayerNorm(d)
        self.h = torch.nn.Linear(d, vs, bias=False)
    def forward(self, x):
        B,L = x.shape
        p = torch.arange(L, device=x.device).unsqueeze(0)
        h = self.te(x) + self.pe(p)
        cm = torch.triu(torch.ones(L,L,device=x.device),1).bool()
        for l in self.ls:
            h2 = l['n1'](h); h2,_ = l['a'](h2,h2,h2,attn_mask=cm); h = h+h2
            h = h + l['f'](l['n2'](h))
        return self.h(self.fn(h))
    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def gen_nested_depth(bs, L=48, md=4):
    inputs = torch.full((bs,L), 2, dtype=torch.long)
    targets = torch.full((bs,L), 0, dtype=torch.long)
    for b in range(bs):
        s, d, ds = [], 0, []
        for i in range(L):
            co = d < md and (L-i) > (d+1)
            cc = d > 0
            if co and cc:
                if torch.rand(1).item() < 0.55: s.append(0); d += 1
                else: s.append(1); d -= 1
            elif co: s.append(0); d += 1
            elif cc: s.append(1); d -= 1
            else: break
            ds.append(d)
        n = len(s)
        inputs[b,:n] = torch.tensor(s); targets[b,:n] = torch.tensor(ds)
    return inputs, targets


def gen_multiscale_copy(bs, vs=16, L=64):
    inputs = torch.randint(2, vs, (bs, L))
    targets = torch.full((bs, L), 0, dtype=torch.long)
    for b in range(bs):
        pat = torch.randint(2, vs, (4,))
        inputs[b, 0:4] = pat; inputs[b, 16] = 1
        targets[b, 32:36] = pat
    return inputs, targets


def gen_parity(bs, ng=8, gs=4):
    L = ng * gs
    inputs = torch.randint(0, 2, (bs, L))
    targets = torch.full((bs, L), 2, dtype=torch.long)
    for b in range(bs):
        gp = []
        for g in range(ng):
            s,e = g*gs, (g+1)*gs
            p = inputs[b,s:e].sum().item() % 2; gp.append(p)
            targets[b, e-1] = p
        targets[b, -1] = sum(gp) % 2
    return inputs, targets


def train_eval(model, name, task_fn, steps=150, bs=16, ign=-100):
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)
    model.train()
    t0 = time.time()
    for step in range(steps):
        inp, tgt = task_fn(bs)
        logits = model(inp)
        lf = logits.reshape(-1, logits.size(-1))
        tf = tgt.reshape(-1)
        mask = tf != ign
        if not mask.any(): continue
        loss = F.cross_entropy(lf[mask], tf[mask])
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
    dt = time.time() - t0
    model.eval()
    with torch.no_grad():
        inp, tgt = task_fn(64)
        logits = model(inp)
        lf = logits.reshape(-1, logits.size(-1)); tf = tgt.reshape(-1)
        mask = tf != ign
        fl = F.cross_entropy(lf[mask], tf[mask]).item()
        fa = (lf[mask].argmax(-1) == tf[mask]).float().mean().item()
    return fl, fa, dt


def run_benchmark():
    torch.manual_seed(42)
    tasks = [
        ('Nested Depth', lambda bs: gen_nested_depth(bs), 5, -100, 150),
        ('MultiScale Copy', lambda bs: gen_multiscale_copy(bs), 16, 0, 150),
        ('Hier. Parity', lambda bs: gen_parity(bs), 3, 2, 150),
    ]

    print(f"\n{'='*85}")
    print("BENCHMARK: AFN vs Transformer vs NCA-LM")
    print(f"{'='*85}")

    from architectures.nca_lm import NCA_LM

    all_results = []

    for tname, tfn, vout, ign, steps in tasks:
        si, _ = tfn(2)
        ev = max(si.max().item()+1, vout)

        models = {
            'Transformer': TransformerBaseline(ev, d=64, nl=2, nh=4, ml=64),
            'NCA-LM': NCA_LM(ev, d_model=64, n_layers=2, n_steps=3,
                              share_weights=True, adaptive=False,
                              kernel_size=5, dilations=(1,4),
                              reaction_expansion=2, ffn_expansion=2,
                              dropout=0.0, max_len=64),
            'AFN': afn_tiny(ev),
        }

        print(f"\n{'─'*60}")
        print(f"Task: {tname}")
        for mn, m in models.items():
            print(f"  {mn}: {m.count_parameters():,} params")

        for mn, m in models.items():
            torch.manual_seed(42)
            fl, fa, dt = train_eval(m, mn, tfn, steps=steps, ign=ign)
            print(f"  {mn:<15} loss={fl:.4f} acc={fa:.4f} time={dt:.1f}s")
            all_results.append((tname, mn, m.count_parameters(), fl, fa, dt))

    print(f"\n\n{'='*85}")
    print("SUMMARY")
    print(f"{'='*85}")
    print(f"{'Task':<20} {'Model':<15} {'Params':>10} {'Loss':>10} {'Acc':>10} {'Time':>8}")
    print('─'*85)
    for tname, mn, p, fl, fa, dt in all_results:
        print(f"{tname:<20} {mn:<15} {p:>10,} {fl:>10.4f} {fa:>10.4f} {dt:>7.1f}s")


if __name__ == "__main__":
    import sys
    if "--bench" in sys.argv:
        run_benchmark()
    else:
        pytest.main([__file__, "-v", "-k", "not benchmark"])
