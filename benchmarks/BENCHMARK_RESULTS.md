# Benchmark Results

| Model | Params | Perplexity | Throughput (tok/s) | Peak Memory (MB) |
|---|---:|---:|---:|---:|
| best_discovered | 1,271,296 | 3.91 | 51000 | 612 |
| afn_v3 | 1,305,472 | 4.08 | 48600 | 640 |
| afn_v4 [EXPERIMENTAL] | 1,338,240 | 3.97 | 52200 | 628 |

Notes: results captured with seed=42 and the deterministic benchmark runner in `benchmarks/run_benchmarks.sh`.
