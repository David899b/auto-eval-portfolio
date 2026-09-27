"""Statistical primitives for evaluation — stdlib only, deterministic, tested.

Every number an agent or harness reports must be reproducible and auditable, so
this module takes an explicit seed wherever randomness is involved and avoids
third-party dependencies entirely. A metric you cannot reproduce is a metric you
cannot defend in a review.
"""
from __future__ import annotations

import math
import random
from collections import Counter
from dataclasses import dataclass
from enum import Enum


class Scenario(str, Enum):
    """Three-scenario reporting: never ship a single number."""

    PESSIMISTIC = "pessimistic"
    BASE = "base"
    OPTIMISTIC = "optimistic"


def total_variation_distance(p: list[float], q: list[float]) -> float:
    """TV distance between two categorical distributions. 0 = identical, 1 = disjoint.

    This — not PSI — is the right tool for label-mix drift. PSI bins a numeric
    axis, so a class-mix vector like ``[0.9, 0.1]`` vs ``[0.1, 0.9]`` collapses
    onto the same bins and reports zero drift, which is a dangerous false
    negative. TV compares the distributions element-wise and cannot do that.
    """
    if len(p) != len(q):
        raise ValueError("distributions must have the same support")
    if not p:
        raise ValueError("empty distribution")
    return 0.5 * sum(abs(a - b) for a, b in zip(p, q))


def jensen_shannon_divergence(p: list[float], q: list[float]) -> float:
    """JSD in nats, bounded to [0, ln 2]. Symmetric and finite at zero support."""
    if len(p) != len(q):
        raise ValueError("distributions must have the same support")
    if not p:
        raise ValueError("empty distribution")
    m = [(a + b) / 2.0 for a, b in zip(p, q)]
    kl_pm = sum(a * math.log(a / mid) for a, mid in zip(p, m) if a > 0)
    kl_qm = sum(b * math.log(b / mid) for b, mid in zip(q, m) if b > 0)
    return 0.5 * kl_pm + 0.5 * kl_qm


def drift_score(
    golden: list[float], production: list[float]
) -> dict[str, float]:
    """Drift verdict inputs for a categorical label mix.

    Returns several distances on purpose: no single number decides a release,
    and the report has to show which measure fired.
    """
    tvd = total_variation_distance(golden, production)
    return {
        "total_variation_distance": tvd,
        "jensen_shannon_divergence": jensen_shannon_divergence(golden, production),
        "classes_moved": sum(1 for a, b in zip(golden, production) if abs(a - b) > 0.05),
    }


def percentile(values: list[float], q: float) -> float:
    """Linear-interpolation percentile. ``q`` in [0, 1]."""
    if not values:
        raise ValueError("percentile of empty sequence")
    if not 0.0 <= q <= 1.0:
        raise ValueError(f"q must be in [0, 1], got {q}")
    ordered = sorted(values)
    if len(ordered) == 1:
        return float(ordered[0])
    pos = q * (len(ordered) - 1)
    low = math.floor(pos)
    high = math.ceil(pos)
    if low == high:
        return float(ordered[low])
    return float(ordered[low] + (ordered[high] - ordered[low]) * (pos - low))


def mean(values: list[float]) -> float:
    if not values:
        raise ValueError("mean of empty sequence")
    return sum(values) / len(values)


def confusion_counts(
    labels_a: list[str], labels_b: list[str]
) -> tuple[Counter, Counter, Counter, int]:
    """Build (joint, marginal_a, marginal_b, n) for two labelers."""
    if len(labels_a) != len(labels_b):
        raise ValueError("label sequences must be the same length")
    joint: Counter = Counter()
    for a, b in zip(labels_a, labels_b):
        joint[(a, b)] += 1
    return (
        joint,
        Counter(labels_a),
        Counter(labels_b),
        len(labels_a),
    )


def cohens_kappa(labels_a: list[str], labels_b: list[str]) -> float:
    """Cohen's kappa for two raters on nominal labels.

    1.0 = perfect agreement, 0.0 = chance-level, < 0 = worse than chance.
    This is the number that decides whether a "golden set" is actually golden:
    if two independent labelers only agree by chance, the set is noise.
    """
    joint, marg_a, marg_b, n = confusion_counts(labels_a, labels_b)
    if n == 0:
        raise ValueError("cohens_kappa of empty sequences")
    observed = sum(c for (a, b), c in joint.items() if a == b) / n
    expected = sum((marg_a[label] / n) * (marg_b[label] / n) for label in marg_a)
    if expected >= 1.0:
        # Both raters used a single identical label: agreement is perfect by
        # construction but chance agreement is also 1, so kappa is undefined.
        return 1.0 if observed == 1.0 else 0.0
    return (observed - expected) / (1.0 - expected)


def wilson_interval(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for a proportion — behaves at 0% and 100%."""
    if n == 0:
        return (0.0, 1.0)
    p = successes / n
    denom = 1.0 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    spread = (z / denom) * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2))
    return (max(0.0, centre - spread), min(1.0, centre + spread))


def _binned(values: list[float], bins: int) -> list[float]:
    lo, hi = min(values), max(values)
    if hi == lo:
        return [1.0] + [0.0] * (bins - 1)
    width = (hi - lo) / bins
    counts = [0.0] * bins
    for v in values:
        idx = min(bins - 1, int((v - lo) / width))
        counts[idx] += 1.0
    total = sum(counts)
    return [c / total for c in counts]


def population_stability_index(
    expected: list[float], actual: list[float], bins: int = 10, epsilon: float = 1e-6
) -> float:
    """PSI between two samples. 0 = identical; > 0.1 is the conventional alert line.

    Uses the union range so neither distribution is clipped, and floors empty
    bins at ``epsilon`` so a support change returns a large-but-finite value
    instead of infinity.
    """
    if not expected or not actual:
        raise ValueError("PSI needs two non-empty samples")
    lo, hi = min(min(expected), min(actual)), max(max(expected), max(actual))
    if hi == lo:
        return 0.0
    width = (hi - lo) / bins
    exp_counts = [0.0] * bins
    act_counts = [0.0] * bins
    for v in expected:
        exp_counts[min(bins - 1, int((v - lo) / width))] += 1.0
    for v in actual:
        act_counts[min(bins - 1, int((v - lo) / width))] += 1.0
    exp_total, act_total = sum(exp_counts), sum(act_counts)
    total = 0.0
    for e, a in zip(exp_counts, act_counts):
        e_pct = max(e / exp_total, epsilon)
        a_pct = max(a / act_total, epsilon)
        total += (a_pct - e_pct) * math.log(a_pct / e_pct)
    return total


def kl_divergence(expected: list[float], actual: list[float], epsilon: float = 1e-9) -> float:
    """KL(actual || expected) over the union range, in nats."""
    if not expected or not actual:
        raise ValueError("KL needs two non-empty samples")
    lo, hi = min(min(expected), min(actual)), max(max(expected), max(actual))
    if hi == lo:
        return 0.0
    width = (hi - lo) / 10
    exp_counts = [0.0] * 10
    act_counts = [0.0] * 10
    for v in expected:
        exp_counts[min(9, int((v - lo) / width))] += 1.0
    for v in actual:
        act_counts[min(9, int((v - lo) / width))] += 1.0
    exp_total, act_total = sum(exp_counts), sum(act_counts)
    total = 0.0
    for e, a in zip(exp_counts, act_counts):
        p = max(a / act_total, epsilon)
        q = max(e / exp_total, epsilon)
        total += p * math.log(p / q)
    return total


def bootstrap_ci(
    values: list[float],
    n_bootstrap: int = 1000,
    confidence: float = 0.95,
    seed: int = 0,
    statistic: str = "mean",
    direction: str = "min",
) -> dict[Scenario, float]:
    """Seeded bootstrap CI mapped to pessimistic / base / optimistic.

    Seeded on purpose: an evaluation whose 3 scenarios move on every run cannot
    be diffed against a previous run, which makes regressions invisible.

    ``direction`` decides which tail of the interval is the *pessimistic* one,
    and getting it wrong silently inverts the gate:

    * ``"min"`` — lower is worse (F1, accuracy, agreement). The bad news is at
      the bottom of the interval.
    * ``"max"`` — higher is worse (latency, cost, error rate). The bad news is
      at the *top*. Without this, a latency gate reads the optimistic end as
      pessimistic and waves through exactly the regressions it exists to catch.
    """
    if len(values) < 2:
        nan = float("nan")
        return {s: nan for s in Scenario}
    rng = random.Random(seed)
    fn = {"mean": mean, "p95": lambda xs: percentile(xs, 0.95)}[statistic]
    boots: list[float] = []
    n = len(values)
    for _ in range(n_bootstrap):
        sample = [values[rng.randrange(n)] for _ in range(n)]
        boots.append(fn(sample))
    alpha = (1.0 - confidence) / 2.0
    low, high = percentile(boots, alpha), percentile(boots, 1.0 - alpha)
    pessimistic, optimistic = (low, high) if direction == "min" else (high, low)
    return {
        Scenario.PESSIMISTIC: pessimistic,
        Scenario.BASE: percentile(boots, 0.5),
        Scenario.OPTIMISTIC: optimistic,
    }


@dataclass(frozen=True)
class GateDecision:
    """Outcome of a release gate: never a bare boolean, always a reason."""

    decision: str  # PASS | CONDITIONAL | BLOCK
    reason: str
    scenarios: dict[Scenario, float]

    @property
    def is_blocking(self) -> bool:
        return self.decision == "BLOCK"


def gate_on_scenarios(
    scenarios: dict[Scenario, float],
    limit: float,
    direction: str = "max",
    margin: float = 0.0,
) -> GateDecision:
    """Gate a metric on its *worst* scenario, not its average.

    direction="max": higher is worse (latency, cost, error rate).
    direction="min": lower is worse (F1, ANLS, schema-ok rate).
    ``margin`` widens the accept band for the conditional case.
    """
    pessimistic = scenarios[Scenario.PESSIMISTIC]
    if direction == "max":
        if pessimistic > limit:
            return GateDecision("BLOCK", f"pessimistic {pessimistic:.4g} > limit {limit:.4g}", scenarios)
        if pessimistic > limit - margin:
            return GateDecision(
                "CONDITIONAL",
                f"pessimistic {pessimistic:.4g} within {margin:.4g} of limit {limit:.4g}",
                scenarios,
            )
        return GateDecision("PASS", f"pessimistic {pessimistic:.4g} <= limit {limit:.4g}", scenarios)
    if pessimistic < limit:
        return GateDecision("BLOCK", f"pessimistic {pessimistic:.4g} < floor {limit:.4g}", scenarios)
    if pessimistic < limit + margin:
        return GateDecision(
            "CONDITIONAL",
            f"pessimistic {pessimistic:.4g} within {margin:.4g} of floor {limit:.4g}",
            scenarios,
        )
    return GateDecision("PASS", f"pessimistic {pessimistic:.4g} >= floor {limit:.4g}", scenarios)
