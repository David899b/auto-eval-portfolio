"""Tests for eval-watch math (pure python)."""
import sys
import random
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from eval_watch.eval_watch import (
    psi,
    kl_divergence,
    bootstrap_ci,
    gate_decision,
)


def test_psi_identical_is_zero():
    data = [0.5, 0.5, 0.6, 0.7, 0.4, 0.55, 0.65, 0.45]
    assert abs(psi(data, data) - 0.0) < 1e-6


def test_psi_drift_is_positive():
    data = [round(v, 3) for v in range(0, 100)]
    shifted = [v + 25 for v in data]
    assert psi(data, shifted) > 0.05


def test_kl_divergence_shift():
    data = [round(v, 3) for v in range(0, 100)]
    shifted = [v + 25 for v in data]
    assert kl_divergence(data, shifted) > 0


def test_bootstrap_ci_nesting():
    rng = random.Random(1)
    values = [10 + i % 3 for i in range(200)]
    lo, base, hi = bootstrap_ci(values, n_samples=200, rng=rng)
    assert lo <= base <= hi


def test_gate_decisions():
    assert gate_decision(0.97, 0.964, 0.95) == "PASS"
    assert gate_decision(0.96, 0.94, 0.95) == "CONDITIONAL"
    assert gate_decision(0.90, 0.88, 0.95) == "BLOCK"


def _run() -> int:
    """Shared runner: report failures and exit non-zero so CI can gate on it."""
    tests = [
        (n, f)
        for n, f in sorted(globals().items())
        if n.startswith("test_") and callable(f)
    ]
    failures = 0
    for name, fn in tests:
        try:
            fn()
        except Exception as exc:  # noqa: BLE001
            failures += 1
            print(f"FAIL {name}: {type(exc).__name__}: {exc}")
        else:
            print(f"PASS {name}")
    print(f"\n{len(tests) - failures}/{len(tests)} passed")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(_run())
