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
- Results are **experimental** and **small-scale** **smoke benchmark only** measurements.
- This benchmark is intended as a **diagnostic** check for local behavior, not a production evaluation.
- Outputs here are **not production-grade** performance claims.
- These results are **not evidence of general model superiority**.
- AFN v4 is included only when instantiated by the runner itself.
