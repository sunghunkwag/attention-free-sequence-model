"""
Automated architecture search engine.

Discovers mechanism combinations that outperform GatedShiftMixer
by randomly sampling from the primitives library and evaluating
on 3 synthetic benchmark tasks.

Usage:
    python -m architectures.search_engine [--n_trials 3000]
"""

import json
import os
import random
import time
import traceback
from typing import List, Dict, Any, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from architectures.primitives import (
    PROPAGATION_REGISTRY, UPDATE_REGISTRY, build_module
)


# =========================================================================
# Task generators (copied from benchmarks/final_benchmark.py)
# =========================================================================

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


TASKS = {
    'nested_depth': {
        'fn': gen_nested_depth,
        'vocab_out': 5,
        'ignore_index': -100,
        'seq_len': 48,
    },
    'multiscale_copy': {
        'fn': gen_multiscale_copy,
        'vocab_out': 16,
        'ignore_index': 0,
        'seq_len': 64,
    },
    'hierarchical_parity': {
        'fn': gen_hierarchical_parity,
        'vocab_out': 3,
        'ignore_index': 2,
        'seq_len': 32,
    },
}


# =========================================================================
# Searchable Architecture
# =========================================================================

class SearchableBlock(nn.Module):
    """One block: sequence of propagation mechanisms + update mechanism."""
    def __init__(self, d_model: int, prop_names: List[str], update_name: str):
        super().__init__()
        self.props = nn.ModuleList([
            build_module(name, d_model, PROPAGATION_REGISTRY)
            for name in prop_names
        ])
        self.update = build_module(update_name, d_model, UPDATE_REGISTRY)

    def forward(self, x):
        for prop in self.props:
            x = prop(x)
        x = self.update(x)
        return x


class SearchableModel(nn.Module):
    """Complete searchable model: embedding + blocks + head."""
    def __init__(self, vocab_size: int, d_model: int, blocks: List[dict], max_len: int = 128):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_len, d_model)
        self.emb_norm = nn.LayerNorm(d_model)

        self.blocks = nn.ModuleList([
            SearchableBlock(d_model, b['props'], b['update'])
            for b in blocks
        ])
        self.final_norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)
        self.head.weight = self.tok_emb.weight

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, std=0.02)

    def forward(self, x):
        B, L = x.shape
        pos = torch.arange(L, device=x.device).unsqueeze(0)
        h = self.emb_norm(self.tok_emb(x) + self.pos_emb(pos))
        for block in self.blocks:
            h = block(h)
        return self.head(self.final_norm(h))

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# =========================================================================
# Architecture Search Engine
# =========================================================================

# Mechanisms that use O(L^2) or worse — avoid for large L but OK for search sizes
EXPENSIVE_PROPS = {'sinkhorn_permutation'}


class ArchitectureSearch:
    """Automated architecture search over mechanism primitives."""

    def __init__(self, target_params: int = 105_000, seed: int = 42):
        self.target_params = target_params
        self.seed = seed
        # Exclude O(L^2) mechanisms from random search for speed
        self.prop_names = [k for k in PROPAGATION_REGISTRY.keys()
                           if k not in EXPENSIVE_PROPS]
        self.update_names = list(UPDATE_REGISTRY.keys())
        self.results: List[Dict[str, Any]] = []

    def random_architecture(self) -> Tuple[List[dict], int]:
        """Sample a random architecture config. Returns (blocks_config, d_model)."""
        n_layers = random.randint(1, 4)
        blocks = []
        for _ in range(n_layers):
            n_props = random.randint(2, 4)
            props = random.sample(self.prop_names, min(n_props, len(self.prop_names)))
            update = random.choice(self.update_names)
            blocks.append({'props': props, 'update': update})

        # Binary search for d_model to hit target params
        d_model = self._find_d_model(blocks)
        return blocks, d_model

    def _find_d_model(self, blocks: List[dict], vocab_size: int = 16) -> int:
        """Find d_model that gets close to target_params via binary search."""
        lo, hi = 8, 128

        def get_params(d):
            try:
                model = SearchableModel(vocab_size, d, blocks, max_len=128)
                p = model.count_parameters()
                del model
                return p
            except Exception:
                return None

        # Binary search for closest d_model
        best_d = 32
        best_diff = float('inf')

        while lo <= hi:
            mid = (lo + hi) // 2
            # Ensure even
            mid = mid if mid % 2 == 0 else mid + 1
            p = get_params(mid)
            if p is None:
                hi = mid - 2
                continue
            diff = abs(p - self.target_params)
            if diff < best_diff:
                best_diff = diff
                best_d = mid
            if p < self.target_params:
                lo = mid + 2
            else:
                hi = mid - 2

        # Also check neighbors
        for d in [best_d - 2, best_d + 2]:
            if 8 <= d <= 128:
                p = get_params(d)
                if p is not None and abs(p - self.target_params) < best_diff:
                    best_diff = abs(p - self.target_params)
                    best_d = d

        return best_d

    def _build_model(self, blocks: List[dict], d_model: int, vocab_size: int) -> SearchableModel:
        """Build model from config."""
        return SearchableModel(vocab_size, d_model, blocks, max_len=128)

    def shape_test(self, model: nn.Module, d_model: int) -> bool:
        """Test that model handles (B=2, L=48, D) correctly."""
        try:
            x = torch.randint(0, 10, (2, 48))
            out = model(x)
            assert out.shape[0] == 2
            assert out.shape[1] == 48
            return True
        except Exception:
            return False

    def gradient_test(self, model: nn.Module) -> bool:
        """Test that gradients flow through the model."""
        try:
            x = torch.randint(0, 10, (2, 32))
            out = model(x)
            loss = out.sum()
            loss.backward()
            has_grad = False
            for p in model.parameters():
                if p.grad is not None and p.grad.abs().sum() > 0:
                    has_grad = True
                    break
            return has_grad
        except Exception:
            return False

    def attention_free_test(self, model: nn.Module) -> bool:
        """Check that no nn.MultiheadAttention exists in the model."""
        for m in model.modules():
            if isinstance(m, nn.MultiheadAttention):
                return False
        return True

    def quick_eval(self, model: nn.Module, task_fn, steps: int = 100,
                   batch_size: int = 8, ignore_index: int = -100) -> Tuple[float, float]:
        """Fast 100-step train + eval. Returns (loss, accuracy)."""
        optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)
        model.train()

        last_loss = None
        for step in range(steps):
            inputs, targets = task_fn(batch_size)
            logits = model(inputs)
            logits_flat = logits.reshape(-1, logits.size(-1))
            targets_flat = targets.reshape(-1)
            mask = targets_flat != ignore_index
            if not mask.any():
                continue
            loss = F.cross_entropy(logits_flat[mask], targets_flat[mask])
            # Early stop: if loss is still very high at step 50, give up
            if step == 50 and loss.item() > 5.0:
                break
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        # Eval
        model.eval()
        with torch.no_grad():
            inputs, targets = task_fn(32)
            logits = model(inputs)
            logits_flat = logits.reshape(-1, logits.size(-1))
            targets_flat = targets.reshape(-1)
            mask = targets_flat != ignore_index
            if not mask.any():
                return 10.0, 0.0
            final_loss = F.cross_entropy(logits_flat[mask], targets_flat[mask]).item()
            preds = logits_flat[mask].argmax(dim=-1)
            final_acc = (preds == targets_flat[mask]).float().mean().item()
        return final_loss, final_acc

    def full_eval(self, model: nn.Module, task_fn, steps: int = 200,
                  batch_size: int = 16, ignore_index: int = -100) -> Tuple[float, float]:
        """Full 200-step evaluation with larger eval batch."""
        optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)
        model.train()

        for step in range(steps):
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

        model.eval()
        with torch.no_grad():
            inputs, targets = task_fn(128)
            logits = model(inputs)
            logits_flat = logits.reshape(-1, logits.size(-1))
            targets_flat = targets.reshape(-1)
            mask = targets_flat != ignore_index
            if not mask.any():
                return 10.0, 0.0
            final_loss = F.cross_entropy(logits_flat[mask], targets_flat[mask]).item()
            preds = logits_flat[mask].argmax(dim=-1)
            final_acc = (preds == targets_flat[mask]).float().mean().item()
        return final_loss, final_acc

    def search(self, n_trials: int = 3000, log_path: str = 'results/search_log.json'):
        """Main search loop."""
        random.seed(self.seed)
        torch.manual_seed(self.seed)

        os.makedirs(os.path.dirname(log_path), exist_ok=True)

        # Load existing results if resuming
        if os.path.exists(log_path):
            with open(log_path, 'r') as f:
                self.results = json.load(f)
            start_trial = len(self.results)
            print(f"Resuming from trial {start_trial}")
        else:
            start_trial = 0

        t_start = time.time()

        for trial in range(start_trial, n_trials):
            trial_result = self._run_trial(trial)
            self.results.append(trial_result)

            # Incremental save every trial
            if trial % 10 == 0 or trial == n_trials - 1:
                with open(log_path, 'w') as f:
                    json.dump(self.results, f, indent=1)

            # Progress report every 50 trials
            if (trial + 1) % 50 == 0:
                elapsed = time.time() - t_start
                completed = [r for r in self.results if r.get('status') == 'evaluated']
                if completed:
                    best = max(completed, key=lambda r: r.get('avg_acc', 0))
                    print(f"[{trial+1}/{n_trials}] elapsed={elapsed:.0f}s "
                          f"evaluated={len(completed)} "
                          f"best_avg_acc={best.get('avg_acc', 0):.4f} "
                          f"best_arch={best.get('arch_id', '?')}")
                else:
                    print(f"[{trial+1}/{n_trials}] elapsed={elapsed:.0f}s "
                          f"no successful evaluations yet")

        # Final save
        with open(log_path, 'w') as f:
            json.dump(self.results, f, indent=1)

        # Full eval on top 10%
        self._promote_top_performers(log_path)

        total_time = time.time() - t_start
        print(f"\nSearch complete: {n_trials} trials in {total_time:.0f}s")
        return self.results

    def _run_trial(self, trial_id: int) -> Dict[str, Any]:
        """Run a single trial. Train one model on all 3 tasks jointly."""
        result = {'trial': trial_id, 'status': 'failed', 'timestamp': time.time()}

        try:
            blocks, d_model = self.random_architecture()
            result['blocks'] = blocks
            result['d_model'] = d_model
            result['arch_id'] = f"arch_{trial_id:04d}"

            effective_vocab = 16

            torch.manual_seed(42 + trial_id)
            model = self._build_model(blocks, d_model, effective_vocab)
            result['n_params'] = model.count_parameters()

            # Shape test
            if not self.shape_test(model, d_model):
                result['fail_reason'] = 'shape_test'
                return result

            # Gradient test
            model.zero_grad()
            if not self.gradient_test(model):
                result['fail_reason'] = 'gradient_test'
                del model
                return result

            # Attention-free test
            if not self.attention_free_test(model):
                result['fail_reason'] = 'attention_free_test'
                del model
                return result

            # Joint training on all 3 tasks (round-robin), 100 steps total
            optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)
            task_list = list(TASKS.items())
            model.train()

            for step in range(100):
                task_name, task_cfg = task_list[step % len(task_list)]
                inputs, targets = task_cfg['fn'](8)
                # Pad inputs to max seq len (64) for uniform processing
                B, L = inputs.shape
                if L < 64:
                    inputs = F.pad(inputs, (0, 64 - L), value=0)
                logits = model(inputs)
                logits = logits[:, :L]
                logits_flat = logits.reshape(-1, logits.size(-1))
                targets_flat = targets.reshape(-1)
                mask = targets_flat != task_cfg['ignore_index']
                if not mask.any():
                    continue
                loss = F.cross_entropy(logits_flat[mask], targets_flat[mask])
                if step == 30 and loss.item() > 5.0:
                    result['fail_reason'] = 'early_stop_high_loss'
                    del model
                    return result
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

            # Eval on each task separately
            model.eval()
            task_results = {}
            total_acc = 0.0
            total_loss = 0.0

            with torch.no_grad():
                for task_name, task_cfg in TASKS.items():
                    inputs, targets = task_cfg['fn'](32)
                    B, L = inputs.shape
                    if L < 64:
                        inputs = F.pad(inputs, (0, 64 - L), value=0)
                    logits = model(inputs)[:, :L]
                    logits_flat = logits.reshape(-1, logits.size(-1))
                    targets_flat = targets.reshape(-1)
                    mask = targets_flat != task_cfg['ignore_index']
                    if not mask.any():
                        task_results[task_name] = {'loss': 10.0, 'acc': 0.0}
                        continue
                    loss_val = F.cross_entropy(logits_flat[mask], targets_flat[mask]).item()
                    preds = logits_flat[mask].argmax(dim=-1)
                    acc_val = (preds == targets_flat[mask]).float().mean().item()
                    task_results[task_name] = {'loss': loss_val, 'acc': acc_val}
                    total_acc += acc_val
                    total_loss += loss_val

            result['task_results'] = task_results
            result['avg_acc'] = total_acc / len(TASKS)
            result['total_loss'] = total_loss
            result['status'] = 'evaluated'
            del model

        except Exception as e:
            result['fail_reason'] = str(e)[:200]
            result['traceback'] = traceback.format_exc()[-500:]

        return result

    def _promote_top_performers(self, log_path: str):
        """Run full eval (200 steps) on top 10% of quick-eval performers."""
        evaluated = [r for r in self.results if r.get('status') == 'evaluated']
        if not evaluated:
            print("No evaluated architectures to promote.")
            return

        evaluated.sort(key=lambda r: r.get('avg_acc', 0), reverse=True)
        n_top = max(1, len(evaluated) // 10)
        top = evaluated[:n_top]

        print(f"\nPromoting top {n_top} architectures to full eval (200 steps)...")

        max_vocab = 16
        for i, result in enumerate(top):
            blocks = result['blocks']
            d_model = result['d_model']

            full_task_results = {}
            total_acc = 0.0
            total_loss = 0.0

            for task_name, task_cfg in TASKS.items():
                torch.manual_seed(42)
                try:
                    model = self._build_model(blocks, d_model, max_vocab)
                    loss, acc = self.full_eval(
                        model, task_cfg['fn'],
                        steps=200, batch_size=16,
                        ignore_index=task_cfg['ignore_index']
                    )
                    full_task_results[task_name] = {'loss': loss, 'acc': acc}
                    total_acc += acc
                    total_loss += loss
                    del model
                except Exception as e:
                    full_task_results[task_name] = {'loss': 10.0, 'acc': 0.0, 'error': str(e)[:100]}

            result['full_task_results'] = full_task_results
            result['full_avg_acc'] = total_acc / len(TASKS)
            result['full_total_loss'] = total_loss
            result['status'] = 'full_evaluated'

            if (i + 1) % 10 == 0:
                print(f"  Full eval {i+1}/{n_top}: avg_acc={result['full_avg_acc']:.4f}")

        # Save updated results
        with open(log_path, 'w') as f:
            json.dump(self.results, f, indent=1)

    def extract_best(self, output_path: str = 'architectures/best_discovered.py'):
        """Extract the best architecture to a standalone file."""
        full_evaluated = [r for r in self.results if r.get('status') == 'full_evaluated']
        if not full_evaluated:
            # Fall back to quick-eval results
            full_evaluated = [r for r in self.results if r.get('status') == 'evaluated']

        if not full_evaluated:
            print("No successful architectures found.")
            return None

        # Sort by full_avg_acc (or avg_acc), tiebreak by total_loss
        best = max(full_evaluated, key=lambda r: (
            r.get('full_avg_acc', r.get('avg_acc', 0)),
            -r.get('full_total_loss', r.get('total_loss', 999))
        ))

        blocks = best['blocks']
        d_model = best['d_model']

        # Generate code
        all_props = set()
        all_updates = set()
        for b in blocks:
            all_props.update(b['props'])
            all_updates.add(b['update'])

        # Map mechanism names to class names
        prop_imports = set()
        for name in all_props:
            cls, _ = PROPAGATION_REGISTRY[name]
            prop_imports.add(cls.__name__)
        for name in all_updates:
            cls, _ = UPDATE_REGISTRY[name]
            prop_imports.add(cls.__name__)

        acc_info = best.get('full_avg_acc', best.get('avg_acc', 0))
        task_info = best.get('full_task_results', best.get('task_results', {}))

        code = f'''"""
Best discovered architecture from automated search.

Architecture ID: {best.get('arch_id', 'unknown')}
Trial: {best.get('trial', '?')}
Parameters: {best.get('n_params', '?')}
d_model: {d_model}
Average accuracy (across 3 tasks): {acc_info:.4f}

Per-task results:
'''
        for task_name, task_res in task_info.items():
            code += f"  {task_name}: loss={task_res['loss']:.4f}, acc={task_res['acc']:.4f}\n"

        code += f"""
Mechanism composition:
"""
        for i, b in enumerate(blocks):
            code += f"  Block {i}: propagation={b['props']}, update={b['update']}\n"

        code += f'''
This architecture was discovered via random search over {len(self.results)} trials.
All mechanisms are attention-free and softmax-free (except final classification head).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from architectures.primitives import (
    PROPAGATION_REGISTRY, UPDATE_REGISTRY, build_module
)


class DiscoveredBlock(nn.Module):
    """One block of the discovered architecture."""
    def __init__(self, d_model, prop_names, update_name):
        super().__init__()
        self.props = nn.ModuleList([
            build_module(name, d_model, PROPAGATION_REGISTRY)
            for name in prop_names
        ])
        self.update = build_module(update_name, d_model, UPDATE_REGISTRY)

    def forward(self, x):
        for prop in self.props:
            x = prop(x)
        x = self.update(x)
        return x


class BestDiscovered(nn.Module):
    """Best architecture discovered by automated search."""

    BLOCKS_CONFIG = {json.dumps(blocks)}
    D_MODEL = {d_model}

    def __init__(self, vocab_size=16, d_model={d_model}, max_len=128):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_len, d_model)
        self.emb_norm = nn.LayerNorm(d_model)

        self.blocks = nn.ModuleList([
            DiscoveredBlock(d_model, b["props"], b["update"])
            for b in self.BLOCKS_CONFIG
        ])
        self.final_norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)
        self.head.weight = self.tok_emb.weight

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, std=0.02)

    def forward(self, x):
        B, L = x.shape
        pos = torch.arange(L, device=x.device).unsqueeze(0)
        h = self.emb_norm(self.tok_emb(x) + self.pos_emb(pos))
        for block in self.blocks:
            h = block(h)
        return self.head(self.final_norm(h))

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
'''

        with open(output_path, 'w') as f:
            f.write(code)

        print(f"Best architecture saved to {output_path}")
        return best

    def generate_analysis(self, output_path: str = 'results/search_analysis.md'):
        """Generate analysis report."""
        evaluated = [r for r in self.results if r.get('status') in ('evaluated', 'full_evaluated')]
        failed = [r for r in self.results if r.get('status') == 'failed']

        if not evaluated:
            with open(output_path, 'w') as f:
                f.write("# Search Analysis\n\nNo successful evaluations.\n")
            return

        evaluated.sort(key=lambda r: r.get('full_avg_acc', r.get('avg_acc', 0)), reverse=True)

        lines = []
        lines.append("# Architecture Search Analysis")
        lines.append(f"\nTotal trials: {len(self.results)}")
        lines.append(f"Evaluated: {len(evaluated)}")
        lines.append(f"Failed: {len(failed)}")

        # Failure breakdown
        if failed:
            fail_reasons = {}
            for r in failed:
                reason = r.get('fail_reason', 'unknown')
                fail_reasons[reason] = fail_reasons.get(reason, 0) + 1
            lines.append("\n## Failure Breakdown")
            for reason, count in sorted(fail_reasons.items(), key=lambda x: -x[1]):
                lines.append(f"- {reason}: {count}")

        # Top 30 architectures
        lines.append("\n## Top 30 Architectures")
        lines.append(f"\n{'Rank':<6} {'Arch ID':<12} {'Avg Acc':<10} {'Total Loss':<12} {'Params':<10} {'Mechanisms'}")
        lines.append("-" * 100)
        for i, r in enumerate(evaluated[:30]):
            acc = r.get('full_avg_acc', r.get('avg_acc', 0))
            loss = r.get('full_total_loss', r.get('total_loss', 999))
            params = r.get('n_params', '?')
            mechs = []
            for b in r.get('blocks', []):
                mechs.extend(b.get('props', []))
                mechs.append(f"[{b.get('update', '?')}]")
            mech_str = ', '.join(mechs)
            lines.append(f"{i+1:<6} {r.get('arch_id', '?'):<12} {acc:<10.4f} {loss:<12.4f} {str(params):<10} {mech_str}")

        # Mechanism frequency in top performers
        lines.append("\n## Mechanism Frequency in Top 10%")
        n_top = max(1, len(evaluated) // 10)
        top_mechs = {}
        for r in evaluated[:n_top]:
            for b in r.get('blocks', []):
                for p in b.get('props', []):
                    top_mechs[p] = top_mechs.get(p, 0) + 1
                u = b.get('update', '')
                top_mechs[u] = top_mechs.get(u, 0) + 1

        for mech, count in sorted(top_mechs.items(), key=lambda x: -x[1]):
            lines.append(f"- {mech}: {count}")

        # Mechanisms that never appear in top 10%
        all_mechs = set(PROPAGATION_REGISTRY.keys()) | set(UPDATE_REGISTRY.keys())
        absent = all_mechs - set(top_mechs.keys())
        if absent:
            lines.append("\n## Mechanisms Absent from Top 10%")
            for m in sorted(absent):
                lines.append(f"- {m}")

        # Per-task analysis
        lines.append("\n## Per-Task Analysis")
        for task_name in TASKS:
            lines.append(f"\n### {task_name}")
            task_sorted = sorted(evaluated,
                                  key=lambda r: r.get('full_task_results', r.get('task_results', {})).get(task_name, {}).get('acc', 0),
                                  reverse=True)
            lines.append(f"{'Rank':<6} {'Arch ID':<12} {'Acc':<10} {'Loss':<10} {'Key Mechanisms'}")
            lines.append("-" * 80)
            for i, r in enumerate(task_sorted[:10]):
                tr = r.get('full_task_results', r.get('task_results', {})).get(task_name, {})
                mechs = []
                for b in r.get('blocks', []):
                    mechs.extend(b.get('props', []))
                lines.append(f"{i+1:<6} {r.get('arch_id', '?'):<12} {tr.get('acc', 0):<10.4f} {tr.get('loss', 99):<10.4f} {', '.join(mechs[:6])}")

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w') as f:
            f.write('\n'.join(lines))
        print(f"Analysis saved to {output_path}")


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_trials', type=int, default=3000)
    parser.add_argument('--target_params', type=int, default=105_000)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    search = ArchitectureSearch(target_params=args.target_params, seed=args.seed)
    search.search(n_trials=args.n_trials)
    search.extract_best()
    search.generate_analysis()


if __name__ == '__main__':
    main()
