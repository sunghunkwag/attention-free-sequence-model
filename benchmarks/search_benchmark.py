"""
Final benchmark: Best Discovered vs AFN v3 vs Transformer — param-matched.

Run from repo root:
    python benchmarks/search_benchmark.py
"""
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from architectures.afn_v3 import AFN3
from architectures.best_discovered import BestDiscovered


class TransformerBaseline(nn.Module):
    def __init__(self, vocab_size, d_model=64, n_layers=2, n_heads=4, max_len=64):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_len, d_model)
        self.layers = nn.ModuleList()
        for _ in range(n_layers):
            self.layers.append(nn.ModuleDict({
                'norm1': nn.LayerNorm(d_model),
                'attn': nn.MultiheadAttention(d_model, n_heads, batch_first=True),
                'norm2': nn.LayerNorm(d_model),
                'ff': nn.Sequential(
                    nn.Linear(d_model, 4 * d_model), nn.GELU(),
                    nn.Linear(4 * d_model, d_model)),
            }))
        self.final_norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, x):
        B, L = x.shape
        pos = torch.arange(L, device=x.device).unsqueeze(0)
        h = self.tok_emb(x) + self.pos_emb(pos)
        causal = torch.triu(torch.ones(L, L, device=x.device), diagonal=1).bool()
        for layer in self.layers:
            h2 = layer['norm1'](h)
            h2, _ = layer['attn'](h2, h2, h2, attn_mask=causal)
            h = h + h2
            h = h + layer['ff'](layer['norm2'](h))
        return self.head(self.final_norm(h))

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def gen_nested_depth(batch_size, seq_len=48, max_depth=4):
    inputs = torch.full((batch_size, seq_len), 2, dtype=torch.long)
    targets = torch.full((batch_size, seq_len), 0, dtype=torch.long)
    for b in range(batch_size):
        seq, depth, depths = [], 0, []
        for i in range(seq_len):
            can_open = depth < max_depth and (seq_len - i) > (depth + 1)
            can_close = depth > 0
            if can_open and can_close:
                if torch.rand(1).item() < 0.55:
                    seq.append(0); depth += 1
                else:
                    seq.append(1); depth -= 1
            elif can_open:
                seq.append(0); depth += 1
            elif can_close:
                seq.append(1); depth -= 1
            else:
                break
            depths.append(depth)
        n = len(seq)
        inputs[b, :n] = torch.tensor(seq)
        targets[b, :n] = torch.tensor(depths)
    return inputs, targets


def gen_multiscale_copy(batch_size, vocab_size=16, seq_len=64):
    inputs = torch.randint(2, vocab_size, (batch_size, seq_len))
    targets = torch.full((batch_size, seq_len), 0, dtype=torch.long)
    for b in range(batch_size):
        pattern = torch.randint(2, vocab_size, (4,))
        inputs[b, 0:4] = pattern
        inputs[b, 16] = 1
        targets[b, 32:36] = pattern
    return inputs, targets


def gen_hierarchical_parity(batch_size, n_groups=8, group_size=4):
    seq_len = n_groups * group_size
    inputs = torch.randint(0, 2, (batch_size, seq_len))
    targets = torch.full((batch_size, seq_len), 2, dtype=torch.long)
    for b in range(batch_size):
        group_parities = []
        for g in range(n_groups):
            start, end = g * group_size, (g + 1) * group_size
            parity = inputs[b, start:end].sum().item() % 2
            group_parities.append(parity)
            targets[b, end - 1] = parity
        targets[b, -1] = sum(group_parities) % 2
    return inputs, targets


def train_and_evaluate(model, task_fn, n_steps=200, batch_size=16, ignore_index=-100):
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)
    model.train()
    t0 = time.time()

    for step in range(n_steps):
        inputs, targets = task_fn(batch_size)
        logits = model(inputs)
        logits_flat = logits.reshape(-1, logits.size(-1))
        targets_flat = targets.reshape(-1)
        mask = targets_flat != ignore_index
        if not mask.any():
            continue
        loss = F.cross_entropy(logits_flat[mask], targets_flat[mask])
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

    train_time = time.time() - t0

    model.eval()
    with torch.no_grad():
        inputs, targets = task_fn(128)
        logits = model(inputs)
        logits_flat = logits.reshape(-1, logits.size(-1))
        targets_flat = targets.reshape(-1)
        mask = targets_flat != ignore_index
        final_loss = F.cross_entropy(logits_flat[mask], targets_flat[mask]).item()
        preds = logits_flat[mask].argmax(dim=-1)
        final_acc = (preds == targets_flat[mask]).float().mean().item()

    return final_loss, final_acc, train_time


def main():
    tasks = [
        ('Nested Depth',    gen_nested_depth,       5,  -100, 200),
        ('MultiScale Copy', gen_multiscale_copy,    16,  0,   200),
        ('Hier. Parity',    gen_hierarchical_parity, 3,  2,   200),
    ]

    print()
    print("=" * 100)
    print("BEST DISCOVERED vs AFN v3 vs TRANSFORMER — PARAM-MATCHED, ZERO ATTENTION")
    print("=" * 100)

    results = []

    for task_name, task_fn, vocab_out, ignore_idx, n_steps in tasks:
        sample_in, _ = task_fn(2)
        effective_vocab = max(sample_in.max().item() + 1, vocab_out)

        transformer = TransformerBaseline(effective_vocab, d_model=64,
                                          n_layers=2, n_heads=4, max_len=64)
        afn3 = AFN3(effective_vocab, d_model=32, n_layers=2, nca_steps=3,
                     nca_k=5, dilations=(1, 4, 16),
                     shifts=(-32, -16, -4, -1, 1, 4, 16, 32),
                     coarse_stride=4, coarse_steps=1, ffn_exp=2, max_len=64)
        best_disc = BestDiscovered(vocab_size=effective_vocab, d_model=50, max_len=128)

        print(f"\n{'─' * 80}")
        print(f"Task: {task_name}")
        print(f"  Transformer    {transformer.count_parameters():>8,} params")
        print(f"  AFN v3         {afn3.count_parameters():>8,} params")
        print(f"  Best Discovered{best_disc.count_parameters():>8,} params")

        for model_name, model in [('Transformer', transformer),
                                   ('AFN v3', afn3),
                                   ('Best Discovered', best_disc)]:
            torch.manual_seed(42)
            loss, acc, dt = train_and_evaluate(
                model, task_fn, n_steps=n_steps, ignore_index=ignore_idx)
            print(f"  {model_name:<17} loss={loss:.4f}  acc={acc:.4f}  time={dt:.1f}s")
            results.append((task_name, model_name, model.count_parameters(),
                           loss, acc, dt))

    # Summary
    print(f"\n\n{'=' * 100}")
    print("FINAL RESULTS")
    print(f"{'=' * 100}")
    print(f"{'Task':<20} {'Model':<18} {'Params':>10} {'Loss':>10} {'Acc':>10} {'Time':>8}")
    print("─" * 85)

    for task, model, params, loss, acc, dt in results:
        best_acc = max(r[4] for r in results if r[0] == task)
        marker = " <<< WINNER" if abs(acc - best_acc) < 1e-6 else ""
        print(f"{task:<20} {model:<18} {params:>10,} {loss:>10.4f} "
              f"{acc:>10.4f} {dt:>7.1f}s{marker}")

    # Win count per model
    print(f"\n{'─' * 85}")
    model_names = ['Transformer', 'AFN v3', 'Best Discovered']
    task_names = ['Nested Depth', 'MultiScale Copy', 'Hier. Parity']
    for mn in model_names:
        wins = sum(1 for task in task_names
                   if max((r[4], r[1]) for r in results if r[0] == task)[1] == mn)
        avg_acc = sum(r[4] for r in results if r[1] == mn) / len(task_names)
        print(f"  {mn:<18} wins: {wins}/3  avg_acc: {avg_acc:.4f}")


if __name__ == "__main__":
    main()
