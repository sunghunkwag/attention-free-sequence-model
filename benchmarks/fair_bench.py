"""
Fair benchmark: AFN vs NCA-LM vs Transformer
ALL models matched at ~105-110K parameters.
"""
import time, torch, torch.nn as nn, torch.nn.functional as F
from architectures.afn_v1 import AFN
from architectures.nca_lm import NCA_LM


class Transformer(nn.Module):
    def __init__(self, vs, d=64, nl=2, nh=4, ml=64):
        super().__init__()
        self.te = nn.Embedding(vs, d); self.pe = nn.Embedding(ml, d)
        self.ls = nn.ModuleList()
        for _ in range(nl):
            self.ls.append(nn.ModuleDict({
                'n1': nn.LayerNorm(d),
                'a': nn.MultiheadAttention(d, nh, batch_first=True),
                'n2': nn.LayerNorm(d),
                'f': nn.Sequential(nn.Linear(d,4*d), nn.GELU(), nn.Linear(4*d,d)),
            }))
        self.fn = nn.LayerNorm(d); self.h = nn.Linear(d, vs, bias=False)
    def forward(self, x):
        B,L = x.shape; p = torch.arange(L, device=x.device).unsqueeze(0)
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
            co = d < md and (L-i) > (d+1); cc = d > 0
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
        inputs[b, 0:4] = pat; inputs[b, 16] = 1; targets[b, 32:36] = pat
    return inputs, targets

def gen_parity(bs, ng=8, gs=4):
    L = ng * gs
    inputs = torch.randint(0, 2, (bs, L))
    targets = torch.full((bs, L), 2, dtype=torch.long)
    for b in range(bs):
        gp = []
        for g in range(ng):
            s2,e = g*gs, (g+1)*gs
            p = inputs[b,s2:e].sum().item() % 2; gp.append(p); targets[b, e-1] = p
        targets[b, -1] = sum(gp) % 2
    return inputs, targets


def train_eval(model, task_fn, steps=200, bs=16, ign=-100):
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    model.train(); t0 = time.time()
    for step in range(steps):
        inp, tgt = task_fn(bs)
        logits = model(inp)
        lf = logits.reshape(-1, logits.size(-1)); tf = tgt.reshape(-1)
        mask = tf != ign
        if not mask.any(): continue
        loss = F.cross_entropy(lf[mask], tf[mask])
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sched.step()
    dt = time.time() - t0
    model.eval()
    with torch.no_grad():
        inp, tgt = task_fn(128)
        logits = model(inp)
        lf = logits.reshape(-1, logits.size(-1)); tf = tgt.reshape(-1)
        mask = tf != ign
        fl = F.cross_entropy(lf[mask], tf[mask]).item()
        fa = (lf[mask].argmax(-1) == tf[mask]).float().mean().item()
    return fl, fa, dt


def main():
    tasks = [
        ('Nested Depth',    lambda bs: gen_nested_depth(bs),    5,  -100, 200),
        ('MultiScale Copy', lambda bs: gen_multiscale_copy(bs), 16, 0,    200),
        ('Hier. Parity',    lambda bs: gen_parity(bs),          3,  2,    200),
    ]

    print(f"\n{'='*90}")
    print("FAIR BENCHMARK — ALL MODELS ~105-110K PARAMS")
    print(f"{'='*90}")

    all_r = []

    for tname, tfn, vout, ign, steps in tasks:
        si, _ = tfn(2)
        ev = max(si.max().item()+1, vout)

        models = {
            'Transformer': Transformer(ev, d=64, nl=2, nh=4, ml=64),
            'NCA-LM': NCA_LM(ev, d_model=52, n_layers=2, n_steps=3,
                              share_weights=True, adaptive=False,
                              kernel_size=5, dilations=(1,4),
                              reaction_expansion=2, ffn_expansion=2,
                              dropout=0.0, max_len=64),
            'AFN': AFN(ev, d_model=36, n_layers=2, nca_steps=3,
                       nca_kernel=5, nca_dilations=(1,4),
                       bucket_size=4, n_routes=2, coarse_stride=4,
                       coarse_steps=2, ffn_expansion=2, max_len=64),
        }

        print(f"\n{'─'*70}")
        print(f"Task: {tname}")
        for mn, m in models.items():
            print(f"  {mn:<15} {m.count_parameters():>8,} params")

        for mn, m in models.items():
            torch.manual_seed(42)
            fl, fa, dt = train_eval(m, tfn, steps=steps, ign=ign)
            print(f"  {mn:<15} loss={fl:.4f}  acc={fa:.4f}  time={dt:.1f}s")
            all_r.append((tname, mn, m.count_parameters(), fl, fa, dt))

    print(f"\n\n{'='*90}")
    print("SUMMARY — PARAM-MATCHED")
    print(f"{'='*90}")
    print(f"{'Task':<20} {'Model':<15} {'Params':>10} {'Loss':>10} {'Acc':>10} {'Time':>8}")
    print('─'*90)
    for t, mn, p, fl, fa, dt in all_r:
        best_acc = max(r[4] for r in all_r if r[0] == t)
        marker = ' ★' if fa == best_acc else ''
        print(f"{t:<20} {mn:<15} {p:>10,} {fl:>10.4f} {fa:>10.4f} {dt:>7.1f}s{marker}")


if __name__ == "__main__":
    main()
