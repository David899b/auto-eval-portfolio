"""Tests for the statistical primitives. Zero-dependency, runnable either way:

    python tests/test_stats.py        # no pytest needed
    pytest tests/test_stats.py        # if installed
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from ai_eval.stats import (  # noqa: E402
    Scenario,
    bootstrap_ci,
    cohens_kappa,
    drift_score,
    gate_on_scenarios,
    jensen_shannon_divergence,
    kl_divergence,
    mean,
    percentile,
    population_stability_index,
    total_variation_distance,
    wilson_interval,
)


def test_percentile_endpoints():
    xs = [1.0, 2.0, 3.0, 4.0]
    assert percentile(xs, 0.0) == 1.0
    assert percentile(xs, 1.0) == 4.0
    assert percentile(xs, 0.5) == 2.5


def test_percentile_interpolates():
    # p95 of 1..100 is the value a latency SLO is actually measured against
    assert abs(percentile([float(i) for i in range(1, 101)], 0.95) - 95.05) < 0.01


def test_percentile_rejects_bad_input():
    for bad in ([], ):
        try:
            percentile(bad, 0.5)
        except ValueError:
            pass
        else:
            raise AssertionError("empty sequence must raise")
    try:
        percentile([1.0], 1.5)
    except ValueError:
        pass
    else:
        raise AssertionError("q out of range must raise")


def test_mean():
    assert mean([1.0, 2.0, 3.0]) == 2.0


def test_kappa_perfect_and_chance():
    a = ["x", "y", "x", "y", "x"]
    assert abs(cohens_kappa(a, list(a)) - 1.0) < 1e-9
    # second rater always picks the other class -> worse than chance
    b = ["y", "x", "y", "x", "y"]
    assert cohens_kappa(a, b) < 0.0


def test_kappa_detects_the_real_failure_mode():
    """Two labelers that disagree a lot must NOT be able to pass a 0.8 target."""
    a = ["pos", "neg"] * 25
    b = ["pos", "pos", "neg", "neg"] * 12 + ["pos", "neg"]
    kappa = cohens_kappa(a, b)
    assert 0.0 < kappa < 0.5
    assert kappa < 0.8


def test_kappa_length_mismatch_raises():
    try:
        cohens_kappa(["a"], ["a", "b"])
    except ValueError:
        pass
    else:
        raise AssertionError("mismatched lengths must raise")


def test_psi_identical_is_zero():
    xs = [float(i % 50) for i in range(500)]
    assert abs(population_stability_index(xs, xs)) < 1e-9


def test_psi_grows_with_drift():
    baseline = [200 + (i % 10) * 0.5 for i in range(500)]
    small = [202 + (i % 10) * 0.5 for i in range(500)]
    large = [260 + (i % 10) * 0.5 for i in range(500)]
    assert population_stability_index(baseline, small) < population_stability_index(
        baseline, large
    )


def test_psi_non_overlapping_support_is_finite_and_alerting():
    """A total distribution change must not return inf (that breaks dashboards)."""
    value = population_stability_index([1.0] * 50, [99.0] * 50)
    assert value == value  # not NaN
    assert value < float("inf")
    assert value > 0.1  # past the conventional alert line


def test_kl_zero_for_identical_and_positive_for_shift():
    xs = [float(i % 20) for i in range(400)]
    assert abs(kl_divergence(xs, xs)) < 1e-6
    assert kl_divergence(xs, [v + 5.0 for v in xs]) > 0.0


def test_bootstrap_is_seed_reproducible():
    """The whole point: 3 scenarios must be diffable between runs."""
    values = [0.90, 0.93, 0.88, 0.95, 0.91, 0.89, 0.94, 0.92]
    first = bootstrap_ci(values, n_bootstrap=200, seed=7)
    second = bootstrap_ci(values, n_bootstrap=200, seed=7)
    assert first == second
    different_seed = bootstrap_ci(values, n_bootstrap=200, seed=8)
    assert different_seed != first  # proves the seed is actually load-bearing


def test_bootstrap_scenarios_are_ordered():
    values = [0.90, 0.93, 0.88, 0.95, 0.91, 0.89, 0.94, 0.92]
    s = bootstrap_ci(values, n_bootstrap=300, seed=1)
    assert s[Scenario.PESSIMISTIC] <= s[Scenario.BASE] <= s[Scenario.OPTIMISTIC]
    assert set(s) == set(Scenario)


def test_bootstrap_direction_max_flips_which_tail_is_pessimistic():
    """Regression: for "higher is worse" metrics the bad news is the *top* tail.

    Without this, a latency gate reads the optimistic end of the interval as the
    pessimistic scenario and waves through precisely the regressions it exists
    to catch.
    """
    values = [180.0, 500.0, 520.0, 540.0] * 5
    lower_is_worse = bootstrap_ci(values, n_bootstrap=400, seed=3, direction="min")
    higher_is_worse = bootstrap_ci(values, n_bootstrap=400, seed=3, direction="max")

    assert lower_is_worse[Scenario.PESSIMISTIC] < higher_is_worse[Scenario.PESSIMISTIC]
    assert higher_is_worse[Scenario.PESSIMISTIC] == lower_is_worse[Scenario.OPTIMISTIC]
    assert higher_is_worse[Scenario.OPTIMISTIC] == lower_is_worse[Scenario.PESSIMISTIC]
    # the median is direction-independent
    assert higher_is_worse[Scenario.BASE] == lower_is_worse[Scenario.BASE]


def test_gate_blocks_a_latency_regression_despite_a_healthy_average():
    """A latency sample whose mean clears the SLO but whose tail does not.

    Mean of the sample is ~436ms, under the 500ms limit, so a mean-based gate
    passes it. The pessimistic tail of the bootstrap is ~502ms. This is the
    exact shape of bug a release gate exists to catch, and the reason the
    direction has to be threaded through bootstrap_ci.
    """
    s = bootstrap_ci(
        [180.0, 500.0, 520.0, 540.0] * 5,
        n_bootstrap=400,
        seed=3,
        direction="max",
    )
    assert s[Scenario.BASE] < 500.0, "mean should look healthy"
    assert s[Scenario.PESSIMISTIC] > 500.0, "tail should breach"
    assert gate_on_scenarios(s, limit=500.0, direction="max").decision == "BLOCK"


def test_bootstrap_p95_statistic():
    values = [float(i) for i in range(1, 51)]
    s = bootstrap_ci(values, n_bootstrap=200, seed=2, statistic="p95")
    assert s[Scenario.PESSIMISTIC] > 40.0


def test_bootstrap_too_few_values_is_nan_not_crash():
    s = bootstrap_ci([0.9])
    assert all(v != v for v in s.values())


def test_total_variation_distance_edges():
    assert total_variation_distance([0.5, 0.5], [0.5, 0.5]) == 0.0
    assert abs(total_variation_distance([1.0, 0.0], [0.0, 1.0]) - 1.0) < 1e-9
    assert abs(total_variation_distance([0.6, 0.4], [0.4, 0.6]) - 0.2) < 1e-9


def test_tvd_mismatched_support_raises():
    try:
        total_variation_distance([0.5, 0.5], [1.0])
    except ValueError:
        pass
    else:
        raise AssertionError("mismatched support must raise")


def test_jsd_is_symmetric_zero_and_bounded():
    p, q = [0.7, 0.2, 0.1], [0.4, 0.4, 0.2]
    assert abs(jensen_shannon_divergence(p, p)) < 1e-12
    assert abs(jensen_shannon_divergence(p, q) - jensen_shannon_divergence(q, p)) < 1e-12
    assert 0.0 < jensen_shannon_divergence(p, q) < 0.693147


def test_jsd_handles_disjoint_support_finitely():
    value = jensen_shannon_divergence([1.0, 0.0], [0.0, 1.0])
    assert value == value and value < float("inf")
    assert abs(value - 0.693147) < 1e-6


def test_categorical_drift_survives_an_inverted_mix():
    """REGRESSION GUARD.

    An earlier version compared class-mix vectors with PSI. PSI bins a numeric
    axis, so [0.5, 0.5] and [0.5, 0.5] land in the same bins and PSI reports
    ZERO drift for a completely inverted class mix — a false negative in a
    tool whose entire job is to catch drift. TV distance cannot do that.
    """
    golden = [0.9, 0.1]
    inverted = [0.1, 0.9]
    assert population_stability_index(golden, inverted) < 0.01  # the trap
    assert total_variation_distance(golden, inverted) > 0.5  # the fix

    scores = drift_score(golden, inverted)
    assert scores["total_variation_distance"] > 0.5
    assert scores["classes_moved"] == 2


def test_drift_score_is_quiet_on_identical_mixes():
    scores = drift_score([0.5, 0.3, 0.2], [0.5, 0.3, 0.2])
    assert scores["total_variation_distance"] == 0.0
    assert scores["classes_moved"] == 0


def test_wilson_interval_edges():
    # Wilson deliberately does NOT reach 0.0/1.0 at the extremes; that is the
    # property that makes it usable when a run has 0 failures or 0 successes.
    lo, hi = wilson_interval(0, 100)
    assert lo < 0.01 and 0.0 < hi < 0.05
    lo, hi = wilson_interval(100, 100)
    assert hi > 0.99 and 0.95 < lo < 1.0
    assert wilson_interval(0, 0) == (0.0, 1.0)


def test_gate_blocks_on_pessimistic_not_average():
    """The average passes, the worst case does not -> must BLOCK."""
    s = {
        Scenario.PESSIMISTIC: 0.86,
        Scenario.BASE: 0.95,
        Scenario.OPTIMISTIC: 0.99,
    }
    decision = gate_on_scenarios(s, limit=0.90, direction="min")
    assert decision.decision == "BLOCK"
    assert decision.is_blocking
    assert "pessimistic" in decision.reason


def test_gate_pass_and_conditional_bands():
    good = {Scenario.PESSIMISTIC: 0.97, Scenario.BASE: 0.98, Scenario.OPTIMISTIC: 0.99}
    assert gate_on_scenarios(good, limit=0.90, direction="min").decision == "PASS"

    # conditional band for direction="min" is [limit, limit + margin)
    edge = {Scenario.PESSIMISTIC: 0.905, Scenario.BASE: 0.98, Scenario.OPTIMISTIC: 0.99}
    assert (
        gate_on_scenarios(edge, limit=0.90, direction="min", margin=0.02).decision
        == "CONDITIONAL"
    )
    # just under the floor is still a BLOCK, the margin must not soften it
    under = {Scenario.PESSIMISTIC: 0.895, Scenario.BASE: 0.98, Scenario.OPTIMISTIC: 0.99}
    assert gate_on_scenarios(under, limit=0.90, direction="min", margin=0.02).decision == "BLOCK"


def test_gate_max_direction():
    s = {Scenario.PESSIMISTIC: 220.0, Scenario.BASE: 165.0, Scenario.OPTIMISTIC: 140.0}
    assert gate_on_scenarios(s, limit=200.0, direction="max").decision == "BLOCK"
    ok = {Scenario.PESSIMISTIC: 120.0, Scenario.BASE: 165.0, Scenario.OPTIMISTIC: 190.0}
    assert gate_on_scenarios(ok, limit=200.0, direction="max").decision == "PASS"


TESTS = [v for k, v in sorted(globals().items()) if k.startswith("test_")]

if __name__ == "__main__":
    failures = 0
    for fn in TESTS:
        try:
            fn()
        except Exception as exc:  # noqa: BLE001
            failures += 1
            print(f"FAIL {fn.__name__}: {type(exc).__name__}: {exc}")
        else:
            print(f"PASS {fn.__name__}")
    print(f"\n{len(TESTS) - failures}/{len(TESTS)} passed")
    raise SystemExit(1 if failures else 0)
