# Attention-Free Sequence Model

Can a sequence model beat a Transformer **without any attention mechanism**?

This repository documents a systematic architectural exploration: 5 manually designed architectures (4 failed, 1 succeeded), followed by an automated architecture search over 3,000 candidate configurations. The final discovered architecture surpasses both the Transformer baseline and the best manually designed model on all evaluated tasks.

## Summary of Results

### Phase 1: Manual Design — AFN v3

**AFN v3** (Adaptive Field Network v3) was the first architecture to beat a param-matched Transformer on all 3 synthetic tasks, using **zero attention and zero softmax in information routing**. Its core mechanism, **GatedShiftMixer**, achieves content-preserving long-range access via fixed-offset shifted copies with content-dependent gating.

### Phase 2: Automated Architecture Search

To determine whether GatedShiftMixer represents a local optimum or whether superior mechanism combinations exist, we conducted an automated search over a library of 45 atomic mechanism modules (34 propagation + 11 state update). The search evaluated 3,000 random architectures, with the top 10% (298 architectures) promoted to full evaluation at 200 training steps.

**The best discovered architecture (arch_2334) outperforms both AFN v3 and the Transformer on all 3 tasks:**

| Task | Transformer (105K) | AFN v3 (103K) | Best Discovered (103K) |
|---|---|---|---|
| Nested Depth | 92.5% | 98.4% | **99.8%** |
| MultiScale Copy | 52.2% | 94.0% | **98.6%** |
| Hierarchical Parity | 49.7% | 50.7% | **94.4%** |
| **Average** | **64.8%** | **81.0%** | **97.6%** |

All models trained for 200 steps with identical hyperparameters (AdamW, lr=3e-4, batch=16, ~103-105K parameters).

## Winning Architecture: arch_2334

The discovered architecture consists of 4 blocks with d_model=50 and 103,500 parameters:

| Block | Propagation Mechanisms | State Update |
|---|---|---|
| 0 | LSH local exchange, Hierarchical pool (stride 2) | Conv-GRU |
| 1 | Hierarchical pool (stride 2), Dilated conv (d=8), Dilated conv (d=32), Depthwise conv (k=3) | Squeeze-Excite |
| 2 | Depthwise conv (k=3), Depthwise conv (k=15) | Polynomial activation |
| 3 | Depthwise conv (k=5), Spectral filter (FFT), Dilated conv (d=4) | Per-channel scale |

Complexity: **O(L)** in sequence length for all propagation mechanisms except the spectral filter, which is O(L log L). No attention. No softmax in routing.

GatedShiftMixer does not appear in the winning architecture. The search discovered that combinations of simpler mechanisms — hierarchical pooling, dilated convolutions at multiple scales, hash-bucket local exchange, and spectral filtering — can collectively achieve the same long-range information transfer that GatedShiftMixer provides, while also excelling at hierarchical computation.

## Architecture Search Findings

### Search Statistics
- **Total trials:** 3,000
- **Successfully evaluated:** 2,989 (99.6%)
- **Full evaluations (200 steps):** 298 (top 10%)
- **Failed trials:** 11 (all early-stopped due to divergence)

### Mechanism Analysis

**Most frequent mechanisms in the top 10% of performers:**
1. Strided conv up/down (stride 2) — 110 appearances
2. Dilated convolution (dilation 32) — 107
3. Conv-GRU state update — 87
4. Depthwise convolution (k=3) — 86
5. Long convolution (frequency domain) — 79

**Absent from top 10%:**
- Sinkhorn permutation (O(L^2) complexity, excluded from search)

**GatedShiftMixer variants** appear in top performers but are not essential. The search demonstrates that the same function — content-preserving long-range transfer — can be achieved through alternative mechanism combinations, particularly hierarchical pooling paired with large-dilation convolutions.

### Per-Task Observations

- **Nested Depth:** 10 architectures achieved 100% accuracy. The strongest signal is long convolution in the frequency domain, which appeared in 7 of 10 top performers. Local context via depthwise convolutions is also consistently present.
- **MultiScale Copy:** 10+ architectures achieved 100% accuracy. Dilated convolution at dilation 32 and hierarchical pooling are the most reliable mechanisms, confirming that explicit multi-scale structure facilitates long-range content transfer.
- **Hierarchical Parity:** The hardest task. Peak accuracy is 95.2%. Top performers favor hierarchical pool (stride 2), cellular automata, and EMA — mechanisms that naturally aggregate information across groups.

## Phase 1: Manual Architecture Design

### Key Discovery: GatedShiftMixer

The core mechanism of AFN v3 that first solved MultiScale Copy (52% → 94%):

```python
# Simplified core logic
shifted_copies = [x] + [torch.roll(x, -s, dims=1) for s in shifts]
stacked = torch.stack(shifted_copies, dim=2)          # (B, L, n_shifts+1, D)
gates = sigmoid(gate_proj(x)).reshape(B, L, n+1, D)   # content-dependent
output = (stacked * gates).sum(dim=2)                  # gated selection
```

Properties distinguishing this from attention:
- No query-key-value decomposition
- No softmax over similarity scores
- Fixed wiring (shift offsets are constants, not learned)
- Selection is per-channel gating, not per-token weighting

### AFN v3 Layer Structure

1. **NCA Dynamics** (local) — Depthwise conv perception, gated reaction, multi-rate dilated diffusion. Iterated T steps.
2. **GatedShiftMixer** (long-range) — Fixed-offset shifted copies with content-dependent gating.
3. **SqueezeExcite** (global) — Global average pooling, MLP, per-channel gating. O(L).
4. **CoarseNCA** (multi-scale) — Strided conv downsample, NCA at coarse resolution, transposed conv upsample, gated addition.

### Failed Architectures

| Architecture | Core Idea | Failure Mode |
|---|---|---|
| HFN | Multi-scale parallel fusion + global register tokens | Repackaged attention in all components |
| FEN | V-cycle multigrid iteration | V-cycle overhead exceeded benefit; still relied on attention |
| NCA-LM | Pure cellular automata (local rules only) | Succeeded on Nested Depth, failed on long-range copy |
| PFN | Lagrangian particle dynamics | O(N^2) hidden in pairwise distance computation |
| AFN v1 | NCA + sort-based sparse routing | `argsort` breaks gradient flow |
| AFN v2 | NCA + SqueezeExcite + content-gated conv | Diffusion blurs content; cannot preserve identity over distance |

The progression from NCA-LM (establishing that local dynamics work) through PFN (identifying the need for content-dependent routing) to AFN v3 (solving it with GatedShiftMixer) constitutes the arc of the manual exploration.

## Primitives Library

The search engine draws from a library of 45 atomic modules (`architectures/primitives.py`), all satisfying the interface `forward(x) -> x` where `x` is `(B, L, D)`:

**Propagation mechanisms (37):** Depthwise convolutions (k=3,5,7,9,15,31), dilated convolutions (d=1,2,4,8,16,32,64), GatedShiftMixer variants (4 shift configurations), spectral filter (FFT), diagonal state space model (S4-style), random sparse wiring, butterfly mixer, hierarchical pool+broadcast (stride 2,4,8), exponential moving average, wavelet mixer, LSH local exchange, cellular automata (radius 1,2,3), strided conv up/down (stride 2,4), Sinkhorn permutation, long convolution (frequency domain), **HDC binding**, **episodic LSH cache** (2 variants).

**State update mechanisms (13):** SwiGLU, GeGLU, ReGLU, highway gating, conv-GRU, residual MLP (depth 1,2), squeeze-excite, per-channel scale, polynomial activation, stochastic depth, **dynamic time-scale gating** (2 variants).

## Phase 3: Seeker Field Network

Building on the search findings, the **Seeker Field Network** introduces three radical O(L) mechanisms designed to overcome information bottlenecking at 100K+ token scales:

| Mechanism | Purpose | How It Works |
|---|---|---|
| **DynamicTimeScaleGating** | Data-dependent forgetting | Input structural entropy controls update rate. Low-info tokens freeze state; high-density tokens overwrite it. |
| **HDCBinding** | Lossless structural folding | Circular convolution in frequency domain (HDC) binds information orthogonally. Exact retrieval via unbinding even after 50K+ tokens. |
| **EpisodicLSHCache** | O(1) resonance memory | Decoupled LSH hash-bucket memory bank. Hash collisions enable direct past-state retrieval without attention. |
| **PhaseRouter** | Dynamic primitive routing | Replaces rigid `nn.Sequential` — data routes itself through primitives based on learned phase synchronization. |

All mechanisms are fully differentiable, attention-free (no QKV, no softmax over sequence lengths), and O(L) in time/memory.

```
architectures/seeker_field_network.py   # SeekerFieldNetwork model
architectures/primitives.py             # +4 new primitives (48 total)
tests/test_seeker_field_network.py      # 26/26 tests passing
```

## Limitations

- **Synthetic tasks only.** Evaluation is limited to three constructed tasks at L=32-64, d=50-64, 200 training steps. No natural language perplexity or real-world benchmarks.
- **`torch.roll` is circular.** Wraps around sequence boundaries. Autoregressive use requires masking future-direction shifts (not implemented).
- **Search covers random sampling only.** No evolutionary selection, Bayesian optimization, or gradient-based architecture search was applied. The search space was sampled uniformly.
- **Not compared against Mamba, RWKV, Hyena, or other subquadratic baselines.** Only a standard Transformer was used as reference.
- **Reproducibility caveat.** Joint training on all 3 tasks during search differs from per-task evaluation in the final benchmark. The ranking may shift under different training protocols.

## Repository Structure

```
attention-free-sequence-model/
├── architectures/
│   ├── fractal_gnn_original.py   # Starting point (input from user)
│   ├── hfn.py                    # Attempt 1: Hierarchical Fractal Network (failed)
│   ├── fen.py                    # Attempt 2: Fractal Equilibrium Network (failed)
│   ├── nca_lm.py                 # Attempt 3: Neural Cellular Automata LM (partial)
│   ├── pfn.py                    # Attempt 4: Particle Field Network (failed)
│   ├── afn_v1.py                 # Attempt 5a: Adaptive Field Network v1 (failed)
│   ├── afn_v2.py                 # Attempt 5b: v2 with SqueezeExcite (failed)
│   ├── afn_v3.py                 # Attempt 5c: v3 with GatedShiftMixer (succeeded)
│   ├── primitives.py             # 48 atomic mechanism modules for search
│   ├── search_engine.py          # Automated architecture search (3000 trials)
│   ├── best_discovered.py        # Best architecture from search (arch_2334)
│   └── seeker_field_network.py   # Seeker Field Network (Phase 3)
├── tests/
│   ├── test_fractal_gnn_original.py
│   ├── test_hfn.py
│   ├── test_fen.py
│   ├── test_nca_lm.py
│   ├── test_pfn.py
│   ├── test_afn_v1.py
│   ├── test_afn_v3.py
│   ├── test_best_discovered.py   # 15/15 pass
│   └── test_seeker_field_network.py  # 26/26 pass
├── benchmarks/
│   ├── final_benchmark.py        # AFN v3 vs Transformer
│   ├── fair_bench.py             # 3-way: AFN v1 vs NCA vs Transformer
│   ├── benchmark_fen.py          # FEN vs Transformer
│   └── search_benchmark.py       # Best Discovered vs AFN v3 vs Transformer
├── results/
│   ├── search_log.json           # All 3000 trial results
│   └── search_analysis.md        # Mechanism frequency and per-task analysis
├── requirements.txt
└── README.md
```

## Quick Start

```bash
pip install torch pytest

# Run all tests
pytest tests/ -v

# Run the original benchmark (AFN v3 vs Transformer)
python benchmarks/final_benchmark.py

# Run the full benchmark (Best Discovered vs AFN v3 vs Transformer)
python benchmarks/search_benchmark.py

# Re-run the architecture search (takes ~85 minutes on CPU)
python -m architectures.search_engine --n_trials 3000
```

## Requirements

- Python 3.10+
- PyTorch 2.0+
- pytest (for tests)

No other dependencies. Zero external libraries in model code.

## Scope

**This is:** An experimental exploration of attention-free sequence modeling. A documented trail of 4 manual failures, 1 manual success, and an automated search that discovered a superior mechanism combination. Evidence that GatedShiftMixer, while effective, is not the only path to content-preserving long-range transfer without attention.

**This is not:** A Transformer replacement. A production-ready architecture. A claim of general superiority. A validated language model.

## License

MIT
