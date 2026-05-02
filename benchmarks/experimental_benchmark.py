"""Small reproducible benchmark including AFN v3, AFN v4, and BestDiscovered."""
import argparse
import json
import time
from pathlib import Path
import sys

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from architectures.afn_v3 import AFN3
from architectures.afn_v4 import AFN4
from architectures.best_discovered import BestDiscovered


def gen_multiscale_copy(batch_size, vocab_size=16, seq_len=64):
    inputs = torch.randint(2, vocab_size, (batch_size, seq_len))
    targets = torch.full((batch_size, seq_len), 0, dtype=torch.long)
    for b in range(batch_size):
        pattern = torch.randint(2, vocab_size, (4,))
        inputs[b, 0:4] = pattern
        inputs[b, 16] = 1
        targets[b, 32:36] = pattern
    return inputs, targets


def train_and_eval(model, steps, batch_size):
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)
    model.train()
    t0 = time.time()
    for _ in range(steps):
        x, y = gen_multiscale_copy(batch_size)
        logits = model(x)
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
        opt.zero_grad()
        loss.backward()
        opt.step()
    dt = time.time() - t0
    model.eval()
    with torch.no_grad():
        x, y = gen_multiscale_copy(64)
        logits = model(x)
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1)).item()
        acc = (logits.argmax(-1) == y).float().mean().item()
    return loss, acc, dt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--steps', type=int, default=20)
    ap.add_argument('--batch-size', type=int, default=8)
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    models = {
        'best_discovered': BestDiscovered(vocab_size=16, d_model=50, max_len=64),
        'afn_v3': AFN3(vocab_size=16, d_model=32, n_layers=2, max_len=64),
        'afn_v4': AFN4(vocab_size=16, d_model=32, n_layers=2, max_len=64),
    }

    rows = []
    for name, model in models.items():
        loss, acc, dt = train_and_eval(model, args.steps, args.batch_size)
        rows.append({
            'model': name,
            'params': model.count_parameters(),
            'loss': loss,
            'accuracy': acc,
            'train_time_sec': dt,
            'steps': args.steps,
            'batch_size': args.batch_size,
            'seed': args.seed,
        })

    out_dir = Path('benchmarks/generated')
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / 'experimental_results.json').write_text(json.dumps(rows, indent=2), encoding='utf-8')

    md = ['# Experimental Benchmark Results', '',
          '| Model | Params | Loss | Accuracy | Train Time (s) |',
          '|---|---:|---:|---:|---:|']
    for r in rows:
        md.append(f"| {r['model']} | {r['params']:,} | {r['loss']:.4f} | {r['accuracy']:.4f} | {r['train_time_sec']:.2f} |")
    (out_dir / 'experimental_results.md').write_text('\n'.join(md) + '\n', encoding='utf-8')


if __name__ == '__main__':
    main()
