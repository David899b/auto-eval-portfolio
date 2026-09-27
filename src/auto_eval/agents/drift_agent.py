"""Drift Agent — turns distribution drift into a release decision.

Composes the shared primitives in :mod:`ai_eval.stats` rather than
reimplementing them, so the numbers here and the numbers in
``projects/eval-watch`` cannot silently disagree.

A drift agent that prints a number without a verdict is a dashboard. The whole
value is the gate: drift is only interesting when it changes what you do next.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Sequence

from ai_eval.stats import (
    Scenario,
    bootstrap_ci,
    drift_score,
    gate_on_scenarios,
    kl_divergence,
    population_stability_index,
)
from ai_eval.stats import GateDecision


@dataclass
class DriftAgent:
    """Monitors a metric distribution against a baseline and gates releases."""

    psi_alert: float = 0.10
    kl_alert: float = 0.10
    seed: int = 0

    def check_distribution(
        self,
        baseline: list[float],
        current: list[float],
        bins: int = 10,
    ) -> dict:
        """PSI + KL for a numeric metric, with an explicit alert verdict."""
        if not baseline or not current:
            raise ValueError("drift check needs a non-empty baseline and current sample")
        psi = population_stability_index(baseline, current, bins=bins)
        kl = kl_divergence(baseline, current)
        alerting = psi > self.psi_alert or kl > self.kl_alert
        return {
            "psi": psi,
            "kl": kl,
            "psi_alert_line": self.psi_alert,
            "kl_alert_line": self.kl_alert,
            "drift_detected": alerting,
            "severity": self._severity(psi),
            "n_baseline": len(baseline),
            "n_current": len(current),
        }

    @staticmethod
    def _severity(psi: float) -> str:
        if psi < 0.05:
            return "stable"
        if psi < 0.10:
            return "watch"
        if psi < 0.25:
            return "significant"
        return "severe"

    def check_label_mix(
        self,
        golden: Sequence[str],
        production: Sequence[str],
        alert_threshold: float = 0.10,
    ) -> dict:
        """Drift on a categorical mix (the common case for classifiers).

        Uses TV distance, not PSI: binning a proportion vector loses the class
        identity and can report zero drift for a fully inverted mix.
        """
        golden_counts = Counter(golden)
        prod_counts = Counter(production)
        classes = sorted(set(golden_counts) | set(prod_counts))
        golden_pct = [golden_counts.get(c, 0) / len(golden) for c in classes]
        prod_pct = [prod_counts.get(c, 0) / len(production) for c in classes]

        scores = drift_score(golden_pct, prod_pct)
        tvd = scores["total_variation_distance"]
        moved = [
            {
                "class": c,
                "golden_pct": round(golden_pct[i] * 100, 2),
                "production_pct": round(prod_pct[i] * 100, 2),
            }
            for i, c in enumerate(classes)
            if abs(golden_pct[i] - prod_pct[i]) > 0.05
        ]
        return {
            "total_variation_distance": tvd,
            "jensen_shannon_divergence": scores["jensen_shannon_divergence"],
            "alert_threshold": alert_threshold,
            "drift_detected": tvd > alert_threshold,
            "moved_classes": moved,
            "classes": classes,
        }

    def gate_latency(
        self,
        baseline_latencies: list[float],
        current_latencies: list[float],
        slo_p95_ms: float,
        margin_ms: float = 10.0,
    ) -> dict:
        """Latency SLO gate on the p95, judged on the 3 scenarios.

        Uses the p95 rather than the mean on purpose: a release can hold the
        average and still break the tail that users actually feel.
        """
        scenarios = bootstrap_ci(
            current_latencies,
            n_bootstrap=1000,
            seed=self.seed,
            statistic="p95",
            direction="max",
        )
        decision: GateDecision = gate_on_scenarios(
            scenarios, limit=slo_p95_ms, direction="max", margin=margin_ms
        )
        baseline_p95 = bootstrap_ci(
            baseline_latencies,
            n_bootstrap=1000,
            seed=self.seed,
            statistic="p95",
            direction="max",
        )[Scenario.BASE]
        current_p95 = scenarios[Scenario.BASE]
        return {
            "slo_p95_ms": slo_p95_ms,
            "margin_ms": margin_ms,
            "baseline_p95_ms": baseline_p95,
            "scenarios_ms": scenarios,
            "delta_p95_ms": current_p95 - baseline_p95,
            "decision": decision.decision,
            "reason": decision.reason,
            "is_blocking": decision.is_blocking,
        }
