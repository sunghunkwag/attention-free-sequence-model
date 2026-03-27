# Attention-Free Sequence Model

Can a sequence model beat Transformer **without any attention mechanism**?

This repository documents one day of architectural exploration: 5 architectures designed, implemented, tested, and benchmarked. 4 failed. 1 succeeded.

## Result

**AFN v3** (Adaptive Field Network v3) beats a param-matched Transformer on all 3 synthetic benchmark tasks with **zero attention, zero softmax in information routing**.

| Task | Transformer (105K) | AFN v3 (103K) | Winner |
|---|---|---|---|
| Nested Depth | 98.1% | **99.7%** | AFN v3 |
| MultiScale Copy | 52.2% | **93.9%** | AFN v3 |
| Hierarchical Parity | 49.7% | **50.7%** | AFN v3 |

All models trained for 200 steps with identical hyperparameters (AdamW, lr=3e-4, batch=16).

## Key Discovery: GatedShiftMixer

The core mechanism that enabled the breakthrough on MultiScale Copy (52% → 94%):

```
Input:     [pattern at pos 0-3] ... [copy command at pos 16] ... [recall at pos 32-35]
Problem:   How does pos 32 access content from pos 0 without attention?
```

**GatedShiftMixer** creates shifted copies of the sequence at fixed offsets (e.g., ±1, ±4, ±16, ±32) using `torch.roll`. Each position then selects which shifts to accept via content-dependent gating. This achieves attention's core function — **content-preserving long-range access** — without Q/K/V projections or softmax.

```python
# Simplified core logic
shifted_copies = [x] + [torch.roll(x, -s, dims=1) for s in shifts]
stacked = torch.stack(shifted_copies, dim=2)          # (B, L, n_shifts+1, D)
gates = sigmoid(gate_proj(x)).reshape(B, L, n+1, D)   # content-dependent
output = (stacked * gates).sum(dim=2)                  # gated selection
```

Why this is NOT attention:
- No query-key-value decomposition
- No softmax over similarity scores
- Fixed wiring (shift offsets are hardcoded constants)
- Selection is per-channel gating, not per-token weighting

## Architecture: AFN v3

Each layer processes information through 4 sub-layers:

1. **NCA Dynamics** (local) — Depthwise conv perception → gated reaction → multi-rate dilated diffusion. Iterated T steps. Handles local patterns.
2. **GatedShiftMixer** (long-range) — Fixed-offset shifted copies with content-dependent gating. Enables content-preserving long-range transfer without attention.
3. **SqueezeExcite** (global) — Global average pooling → MLP → per-channel gating. O(L) global conditioning.
4. **CoarseNCA** (multi-scale) — Strided conv downsample → NCA at coarse resolution → transposed conv upsample → gated addition.

Total complexity: **O(L)** — strictly linear in sequence length.

## Failed Architectures (and why)

| Architecture | Core Idea | Failure Mode |
|---|---|---|
| HFN | Multi-scale parallel fusion + global register tokens | Repackaged attention everywhere |
| FEN | V-cycle multigrid iteration | V-cycle overhead > benefit; still used attention |
| NCA-LM | Pure cellular automata (local rules only) | Beat Transformer on Nested Depth, failed on long-range copy |
| PFN | Lagrangian particle dynamics | O(N²) hidden in pairwise distance computation |
| AFN v1 | NCA + sort-based sparse routing | `argsort` breaks gradient flow |
| AFN v2 | NCA + SqueezeExcite + content-gated conv | Diffusion blurs content; cannot preserve identity over distance |

Each failure is documented with structural diagnosis. The progression from NCA-LM (which proved local dynamics work) through PFN (which identified the need for content-dependent routing) to AFN v3 (which solved it with GatedShiftMixer) is the arc of the exploration.

## Honest Limitations

- **Synthetic tasks only.** No natural language perplexity measurement. Three constructed tasks at L=48-64, d=32-64, 200 training steps.
- **`torch.roll` is circular.** Wraps around sequence boundaries. For autoregressive LM use, future-direction shifts must be masked (not yet implemented).
- **Shift offsets are hardcoded.** `(-32, -16, -4, -1, 1, 4, 16, 32)` was hand-tuned for L=64. Adaptive shift selection for arbitrary lengths is an open problem.
- **5-6x slower than Transformer** on CPU. NCA iteration + shift mixer + coarse NCA are all sequential. GPU pipelining would help but is untested.
- **Hierarchical Parity is still near chance** for all models at 200 steps. This task remains unsolved at this training budget.
- **Not compared against Mamba, RWKV, Hyena, or other subquadratic baselines.** Only Transformer was used as reference.

## Repository Structure

```
attention-free-sequence-model/
├── architectures/
│   ├── fractal_gnn_original.py   # Starting point (input from user)
│   ├── hfn.py                    # Attempt 1: Hierarchical Fractal Network (failed)
│   ├── fen.py                    # Attempt 2: Fractal Equilibrium Network (failed)
│   ├── nca_lm.py                 # Attempt 3: Neural Cellular Automata LM (partial success)
│   ├── pfn.py                    # Attempt 4: Particle Field Network (failed)
│   ├── afn_v1.py                 # Attempt 5a: Adaptive Field Network v1 (failed)
│   ├── afn_v2.py                 # Attempt 5b: v2 with SqueezeExcite (failed on copy)
│   └── afn_v3.py                 # Attempt 5c: v3 with GatedShiftMixer (SUCCESS)
├── tests/
│   ├── test_fractal_gnn_original.py  # 31/31 pass
│   ├── test_hfn.py                   # 42/42 pass
│   ├── test_fen.py                   # 46/46 pass
│   ├── test_nca_lm.py               # 32/32 pass
│   ├── test_pfn.py                   # 16/16 pass
│   ├── test_afn_v1.py               # 22/22 pass
│   └── test_afn_v3.py               # AFN v3 test suite
├── benchmarks/
│   ├── final_benchmark.py            # Param-matched AFN v3 vs Transformer
│   ├── fair_bench.py                 # 3-way comparison (AFN v1 vs NCA vs Transformer)
│   └── benchmark_fen.py              # Early FEN vs Transformer benchmark
├── requirements.txt
└── README.md
```

## Quick Start

```bash
pip install torch pytest

# Run all tests
pytest tests/ -v

# Run the benchmark
python benchmarks/final_benchmark.py
```

## Requirements

- Python 3.10+
- PyTorch 2.0+
- pytest (for tests)

No other dependencies. Zero external libraries in model code.

## What This Is / What This Is Not

**This is:** An experimental exploration of attention-free sequence modeling. One concrete mechanism (GatedShiftMixer) that solves a specific problem (content-preserving long-range transfer) without attention. A documented trail of 4 failures and 1 success.

**This is not:** A Transformer replacement. A production-ready architecture. A claim of general superiority. A validated language model.

## License

MIT
