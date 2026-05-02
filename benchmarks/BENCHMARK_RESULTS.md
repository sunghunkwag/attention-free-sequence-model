# Benchmark Results (Experimental, Small-Scale)

This file intentionally avoids static metric tables that cannot be reproduced from current repository code.

Use the benchmark runner below to generate fresh results locally:

```bash
python benchmarks/experimental_benchmark.py --steps 20 --batch-size 8 --seed 42
```

The runner writes:
- `benchmarks/generated/experimental_results.json`
- `benchmarks/generated/experimental_results.md`

Notes:
- Results are **experimental** and **small-scale** smoke measurements only.
- AFN v4 is included only when instantiated by the runner itself.
