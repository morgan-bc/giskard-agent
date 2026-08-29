"""Shared test setup."""

import sys
from pathlib import Path

# Make benchmark scripts importable from tests (gaia_scorer, evaluate_gaia).
_BENCH_GAIA = str(Path(__file__).resolve().parents[1] / "benchmarks" / "gaia")
if _BENCH_GAIA not in sys.path:
    sys.path.insert(0, _BENCH_GAIA)
