"""eval-watch: drift detection (PSI/KL) + 3-scenario quality gates with bootstrap CI.

Pure python, zero dependencies. PSI/KL over discretized score distributions and a
bootstrap confidence interval for the metric of interest, with
pessimistic/base/optimistic gate thresholds (never a single number).
"""
from __future__ import annotations
import math
import random
from collections import Counter
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def psi(expected: list[float], actual: list[float], bins: int = 10, range_: tuple[float, float] | None = None) -> float:
    """Population Stability Index between two score distributions (0 = identical)."""
    lo, hi = range_ or (min(expected + actual), max(expected + actual))
    if hi == lo:
        return 0.0
    edges = [lo + (hi - lo) * i / bins for i in range(bins + 1)]
    e = _histogram(expected, edges, lo, hi, bins)
    a = _histogram(actual, edges, lo, hi, bins)
    out = 0.0
    for p_exp, p_act in zip(e, a):
        if p_act == 0:
            continue
        out += (p_act - p_exp) * math.log((p_act + 1e-9) / (p_exp + 1e-9))
    return out


def kl_divergence(expected: list[float], actual: list[float], bins: int = 10,
                  range_: tuple[float, float] | None = None) -> float:
    """KL(p_actual || p_expected); large values signal drift."""
    lo, hi = range_ or (min(expected + actual), max(expected + actual))
    if hi == lo:
        return 0.0
    edges = [lo + (hi - lo) * i / bins for i in range(bins + 1)]
    e = _histogram(expected, edges, lo, hi, bins)
    a = _histogram(actual, edges, lo, hi, bins)
    return sum(
        (pa * math.log((pa + 1e-9) / (pe + 1e-9)))
        for pa, pe in zip(a, e)
        if pa > 0
    )


def _histogram(values: list[float], edges: list[float], lo: float, hi: float, bins: int) -> list[float]:
    counts = [0] * bins
    for v in values:
        idx = bins - 1 if v >= hi else max(0, int((v - lo) / (hi - lo) * bins))
        idx = min(idx, bins - 1)
        counts[idx] += 1
    n = len(values)
    return [c / n for c in counts]


def bootstrap_ci(values: list[float], n_samples: int = 1000,
                 rng: random.Random | None = None, z: float = 1.96) -> tuple[float, float, float]:
    """(pessimistic, base, optimistic) for the metric mean via bootstrap CI."""
    rng = rng or random.Random(42)
    n = len(values)
    means = [sum(rng.choices(values, k=n)) / n for _ in range(n_samples)]
    means.sort()
    lo, hi = means[int(0.025 * n_samples)], means[int(0.975 * n_samples)]
    return lo, sum(values) / n, hi


def gate_decision(base: float, pessimistic: float, target_pessimistic: float) -> str:
    """3-scenario gate: pessimistic must clear the floor for regulated domains."""
    if pessimistic >= target_pessimistic:
        return "PASS"
    if base >= target_pessimistic:
        return "CONDITIONAL"
    return "BLOCK"


def run_demo() -> None:
    rng = random.Random(0)
    baseline = [200 + (rng.random() - 0.5) * 10 for _ in range(1000)]     # latency ms, stable
    drifted = [203 + (rng.random() - 0.5) * 11 for _ in range(1000)]      # ~+3ms shift after deploy

    p_psi = psi(baseline, drifted)
    p_kl = kl_divergence(baseline, drifted)
    lo, base, hi = bootstrap_ci([m - 200 for m in drifted])       # shift above baseline

    print("\n=== eval-watch: drift + 3-scenario gate ===")
    print(f"PSI={p_psi:.4f} (0=identical)  KL={p_kl:.4f}")
    print(f"Latency delta vs baseline: pessim={lo:+.1f}ms base={base:+.1f}ms optim={hi:+.1f}ms")
    print(f"Gate (target pessim > 15ms): {gate_decision(base, lo, 15)}")
    if p_psi > 0.1:
        print(">> DRIFT ALERT: PSI>0.1 — investigate before releasing.")


if __name__ == "__main__":
    run_demo()