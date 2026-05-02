#!/usr/bin/env bash
set -euo pipefail
export PYTHONHASHSEED=42
python - <<'PY'
import random, torch
random.seed(42); torch.manual_seed(42)
from benchmarks.final_benchmark import main as final_main
final_main()
PY
python benchmarks/fair_bench.py
python benchmarks/search_benchmark.py
