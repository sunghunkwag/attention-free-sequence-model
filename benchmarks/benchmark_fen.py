"""
Benchmark: FEN vs Transformer on synthetic multi-scale tasks.

Tasks
-----
1. Multi-Scale Copy:
   Input:  [local_pattern (4 tokens)] [noise (12 tokens)] [global_marker (1)] [noise (15 tokens)]
   Target: Reproduce local_pattern at specific positions AND respond to global_marker.
   Tests: local chunk-level memory + global long-range dependency simultaneously.

2. Hierarchical Parity:
   Input:  sequence of 0/1 tokens in groups of 4.
   Target: For each group, output parity of that group (local).
           Final token: output parity of ALL group parities (global).
   Tests: local computation + hierarchical aggregation.

3. Nested Bracket Matching:
   Input:  nested brackets at multiple depths.
   Target: For each position, predict the depth.
   Tests: multi-scale structure tracking.

Environment: CPU, ~4GB RAM. Models kept very small (~500K-2M params).
"""

import time
import json
import sys
from dataclasses import dataclass
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ═══════════════════════════════════════════════════════════════════════════
# Minimal Transformer baseline (same param budget)
# ═══════════════════════════════════════════════════════════════════════════

class TransformerBaseline(nn.Module):
    """Small Transformer for fair comparison."""

    def __init__(self, vocab_size: int, d_model: int = 128, n_layers: int = 4,
                 n_heads: int = 4, max_len: int = 128, dropout: float = 0.0):
        super().__init__()
        self.d_model = d_model
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_len, d_model)
        self.layers = nn.ModuleList()
        for _ in range(n_layers):
            self.layers.append(nn.ModuleDict({
                'norm1': nn.LayerNorm(d_model),
                'attn': nn.MultiheadAttention(d_model, n_heads,
                                               dropout=dropout, batch_first=True),
                'norm2': nn.LayerNorm(d_model),
                'ff': nn.Sequential(
                    nn.Linear(d_model, 4 * d_model),
                    nn.GELU(),
                    nn.Linear(4 * d_model, d_model),
                ),
            }))
        self.final_norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, L = x.shape
        pos = torch.arange(L, device=x.device).unsqueeze(0)
        h = self.tok_emb(x) + self.pos_emb(pos)
        # Causal mask
        causal = torch.triu(torch.ones(L, L, device=x.device), diagonal=1).bool()
        for layer in self.layers:
            h2 = layer['norm1'](h)
            h2, _ = layer['attn'](h2, h2, h2, attn_mask=causal)
            h = h + h2
            h = h + layer['ff'](layer['norm2'](h))
        return self.head(self.final_norm(h))

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ═══════════════════════════════════════════════════════════════════════════
# Import FEN (small config adjusted for benchmark)
# ═══════════════════════════════════════════════════════════════════════════

from architectures.fen import FEN


def make_fen(vocab_size: int) -> FEN:
    """FEN sized to roughly match Transformer baseline params."""
    return FEN(
        vocab_size=vocab_size,
        d_model=64,
        n_layers=2,
        n_heads=4,
        fine_chunk_size=8,
        coarse_chunk_sizes=(16,),
        max_iterations=2,
        share_weights=True,
        ffn_expansion=2,
        dropout=0.0,
        max_len=64,
    )


# ═══════════════════════════════════════════════════════════════════════════
# Task 1: Multi-Scale Copy
# ═══════════════════════════════════════════════════════════════════════════

def generate_multiscale_copy(batch_size: int, vocab_size: int = 16,
                              seq_len: int = 64) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Sequence structure:
      [pattern(4)] [noise(12)] [COPY_CMD] [noise(15)] [RECALL_ZONE(4)] [noise(rest)]

    Target: at RECALL_ZONE positions, predict the original pattern tokens.
    At all other positions, predict a BLANK token.

    This requires:
    - Local: remembering a 4-token pattern
    - Global: responding to COPY_CMD that's 12+ positions away from pattern

    Vocab: 0=BLANK, 1=COPY_CMD, 2..vocab_size-1=content tokens
    """
    BLANK = 0
    COPY_CMD = 1
    content_range = vocab_size - 2  # tokens 2..vocab_size-1

    inputs = torch.randint(2, vocab_size, (batch_size, seq_len))
    targets = torch.full((batch_size, seq_len), BLANK, dtype=torch.long)

    pattern_start = 0
    pattern_len = 4
    copy_cmd_pos = 16
    recall_start = 32

    for b in range(batch_size):
        # Place pattern
        pattern = torch.randint(2, vocab_size, (pattern_len,))
        inputs[b, pattern_start:pattern_start + pattern_len] = pattern

        # Place copy command
        inputs[b, copy_cmd_pos] = COPY_CMD

        # Target: reproduce pattern at recall zone
        targets[b, recall_start:recall_start + pattern_len] = pattern

    return inputs, targets


# ═══════════════════════════════════════════════════════════════════════════
# Task 2: Hierarchical Parity
# ═══════════════════════════════════════════════════════════════════════════

def generate_hierarchical_parity(batch_size: int, n_groups: int = 8,
                                   group_size: int = 4) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Input:  binary sequence of length n_groups * group_size
    Target: at last position of each group → parity of that group (0 or 1)
            at very last position → parity of all group parities (meta-parity)

    Vocab: 0, 1 for input; 0, 1 for target; 2 = ignore position
    This tests hierarchical aggregation: local parity + global meta-parity.
    """
    seq_len = n_groups * group_size
    inputs = torch.randint(0, 2, (batch_size, seq_len))
    targets = torch.full((batch_size, seq_len), 2, dtype=torch.long)  # 2 = ignore

    for b in range(batch_size):
        group_parities = []
        for g in range(n_groups):
            start = g * group_size
            end = start + group_size
            group = inputs[b, start:end]
            parity = group.sum().item() % 2
            group_parities.append(parity)
            # Target at last position of group
            targets[b, end - 1] = parity

        # Meta-parity at very last position
        meta_parity = sum(group_parities) % 2
        targets[b, -1] = meta_parity

    return inputs, targets


# ═══════════════════════════════════════════════════════════════════════════
# Task 3: Nested Depth Prediction
# ═══════════════════════════════════════════════════════════════════════════

def generate_nested_depth(batch_size: int, seq_len: int = 48,
                           max_depth: int = 4) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Input:  sequence of ( and ) tokens forming valid nested brackets
    Target: at each position, the current depth after processing that token

    Vocab: 0 = (, 1 = ), 2 = padding
    Target vocab: 0..max_depth

    Tests multi-scale structure: local bracket matching + global depth tracking.
    """
    OPEN = 0
    CLOSE = 1
    PAD = 2

    inputs = torch.full((batch_size, seq_len), PAD, dtype=torch.long)
    targets = torch.full((batch_size, seq_len), 0, dtype=torch.long)

    for b in range(batch_size):
        # Generate valid bracket sequence
        seq = []
        depth = 0
        depths = []
        for i in range(seq_len):
            can_open = depth < max_depth and (seq_len - i) > (depth + 1)
            can_close = depth > 0

            if can_open and can_close:
                if torch.rand(1).item() < 0.55:  # slight bias to open
                    seq.append(OPEN)
                    depth += 1
                else:
                    seq.append(CLOSE)
                    depth -= 1
            elif can_open:
                seq.append(OPEN)
                depth += 1
            elif can_close:
                seq.append(CLOSE)
                depth -= 1
            else:
                break
            depths.append(depth)

        actual_len = len(seq)
        inputs[b, :actual_len] = torch.tensor(seq, dtype=torch.long)
        targets[b, :actual_len] = torch.tensor(depths, dtype=torch.long)

    return inputs, targets


# ═══════════════════════════════════════════════════════════════════════════
# Training loop
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class TrainResult:
    model_name: str
    task_name: str
    losses: List[float]
    accuracies: List[float]
    final_loss: float
    final_accuracy: float
    param_count: int
    total_time: float
    steps: int


def train_on_task(
    model: nn.Module,
    model_name: str,
    task_fn,
    task_name: str,
    vocab_size_out: int,
    n_steps: int = 300,
    batch_size: int = 16,
    lr: float = 3e-4,
    ignore_index: int = -100,
    eval_every: int = 10,
) -> TrainResult:
    """Train model on synthetic task, track loss and accuracy."""
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, n_steps)

    losses = []
    accuracies = []
    model.train()
    start_time = time.time()

    for step in range(n_steps):
        inputs, targets = task_fn(batch_size)

        logits = model(inputs)  # (B, L, V)

        # Reshape for cross-entropy
        logits_flat = logits.reshape(-1, logits.size(-1))
        targets_flat = targets.reshape(-1)

        # Map ignore tokens
        mask = targets_flat != ignore_index
        if not mask.any():
            continue

        loss = F.cross_entropy(
            logits_flat[mask], targets_flat[mask]
        )

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        if step % eval_every == 0:
            with torch.no_grad():
                preds = logits_flat[mask].argmax(dim=-1)
                acc = (preds == targets_flat[mask]).float().mean().item()
                losses.append(loss.item())
                accuracies.append(acc)

    total_time = time.time() - start_time

    # Final evaluation (larger batch)
    model.eval()
    with torch.no_grad():
        inputs, targets = task_fn(64)
        logits = model(inputs)
        logits_flat = logits.reshape(-1, logits.size(-1))
        targets_flat = targets.reshape(-1)
        mask = targets_flat != ignore_index
        final_loss = F.cross_entropy(logits_flat[mask], targets_flat[mask]).item()
        preds = logits_flat[mask].argmax(dim=-1)
        final_acc = (preds == targets_flat[mask]).float().mean().item()

    return TrainResult(
        model_name=model_name,
        task_name=task_name,
        losses=losses,
        accuracies=accuracies,
        final_loss=final_loss,
        final_accuracy=final_acc,
        param_count=sum(p.numel() for p in model.parameters() if p.requires_grad),
        total_time=total_time,
        steps=n_steps,
    )


# ═══════════════════════════════════════════════════════════════════════════
# Run all experiments
# ═══════════════════════════════════════════════════════════════════════════

def run_all():
    torch.manual_seed(42)
    results: List[TrainResult] = []

    # ── Task configs ──
    tasks = [
        {
            'name': 'MultiScale Copy',
            'fn': lambda bs: generate_multiscale_copy(bs, vocab_size=16, seq_len=64),
            'vocab_out': 16,
            'ignore_index': 0,  # BLANK = don't care
            'n_steps': 200,
        },
        {
            'name': 'Hierarchical Parity',
            'fn': lambda bs: generate_hierarchical_parity(bs, n_groups=8, group_size=4),
            'vocab_out': 3,  # 0, 1, 2(ignore)
            'ignore_index': 2,
            'n_steps': 200,
        },
        {
            'name': 'Nested Depth',
            'fn': lambda bs: generate_nested_depth(bs, seq_len=48, max_depth=4),
            'vocab_out': 5,  # depths 0-4
            'ignore_index': -100,  # no ignore (all positions valid)
            'n_steps': 200,
        },
    ]

    for task in tasks:
        print(f"\n{'='*60}")
        print(f"Task: {task['name']}")
        print(f"{'='*60}")

        # Determine input vocab from task
        sample_in, sample_tgt = task['fn'](2)
        input_vocab = max(sample_in.max().item() + 1, 3)
        print(f"Input vocab: {input_vocab}, Output vocab: {task['vocab_out']}")
        print(f"Sequence shape: {sample_in.shape}")

        # ── Build models ──
        # For tasks where input/output vocab differ, we use output vocab
        # and let unused tokens go to waste (simpler than separate heads)
        effective_vocab = max(input_vocab, task['vocab_out'])

        transformer = TransformerBaseline(
            vocab_size=effective_vocab, d_model=64, n_layers=4,
            n_heads=4, max_len=64,
        )
        fen = make_fen(vocab_size=effective_vocab)

        t_params = transformer.count_parameters()
        f_params = fen.count_parameters()
        print(f"Transformer: {t_params:,} params")
        print(f"FEN:         {f_params:,} params")
        print(f"Ratio:       {f_params/t_params:.2f}x")

        # ── Train Transformer ──
        print(f"\nTraining Transformer on {task['name']}...")
        torch.manual_seed(42)
        r_t = train_on_task(
            transformer, "Transformer", task['fn'], task['name'],
            task['vocab_out'], n_steps=task['n_steps'],
            ignore_index=task['ignore_index'],
        )
        results.append(r_t)
        print(f"  Final loss: {r_t.final_loss:.4f}, Accuracy: {r_t.final_accuracy:.4f}")
        print(f"  Time: {r_t.total_time:.1f}s")

        # ── Train FEN ──
        print(f"\nTraining FEN on {task['name']}...")
        torch.manual_seed(42)
        r_f = train_on_task(
            fen, "FEN", task['fn'], task['name'],
            task['vocab_out'], n_steps=task['n_steps'],
            ignore_index=task['ignore_index'],
        )
        results.append(r_f)
        print(f"  Final loss: {r_f.final_loss:.4f}, Accuracy: {r_f.final_accuracy:.4f}")
        print(f"  Time: {r_f.total_time:.1f}s")

        # ── Compare ──
        print(f"\n{'─'*40}")
        delta_acc = r_f.final_accuracy - r_t.final_accuracy
        delta_loss = r_t.final_loss - r_f.final_loss
        print(f"FEN vs Transformer:")
        print(f"  Accuracy: {'+' if delta_acc >= 0 else ''}{delta_acc:.4f}")
        print(f"  Loss:     {'+' if delta_loss >= 0 else ''}{delta_loss:.4f} (positive = FEN better)")
        print(f"  Speed:    {r_t.total_time/r_f.total_time:.2f}x (Transformer/FEN)")

    # ── Summary table ──
    print(f"\n\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")
    print(f"{'Task':<25} {'Model':<15} {'Params':>10} {'Loss':>10} {'Acc':>10} {'Time':>8}")
    print(f"{'─'*80}")
    for r in results:
        print(f"{r.task_name:<25} {r.model_name:<15} {r.param_count:>10,} "
              f"{r.final_loss:>10.4f} {r.final_accuracy:>10.4f} {r.total_time:>7.1f}s")

    # Save raw data
    raw = []
    for r in results:
        raw.append({
            'model': r.model_name,
            'task': r.task_name,
            'params': r.param_count,
            'final_loss': r.final_loss,
            'final_acc': r.final_accuracy,
            'time': r.total_time,
            'steps': r.steps,
            'loss_curve': r.losses,
            'acc_curve': r.accuracies,
        })
    with open('benchmark_results.json', 'w') as f:
        json.dump(raw, f, indent=2)
    print("\nResults saved to benchmark_results.json")

    return results


if __name__ == "__main__":
    run_all()
