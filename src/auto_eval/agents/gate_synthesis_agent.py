"""Gate Synthesis Agent — one release decision out of many metrics.

The failure mode this exists to prevent: a release passes because *most*
metrics improved while the one that matters regressed. This agent takes every
metric, reduces each to three scenarios, and refuses to average them away.

Decision order (documented on purpose, so a reviewer can argue with it):

* any metric BLOCKs            -> BLOCK
* else any metric CONDITIONAL  -> CONDITIONAL
* else                         -> PASS
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ai_eval.stats import Scenario, bootstrap_ci, gate_on_scenarios

_SEVERITY_ORDER = {"PASS": 0, "CONDITIONAL": 1, "BLOCK": 2}


@dataclass
class MetricGate:
    """One metric, its scenarios and the limit it must respect."""

    name: str
    values: list[float]
    limit: float
    direction: str = "min"  # "min": higher is better (F1, ANLS)
    margin: float = 0.0
    unit: str = ""

    def evaluate(self, seed: int = 0, n_bootstrap: int = 1000) -> dict:
        # direction must be forwarded: for "higher is worse" metrics the
        # pessimistic scenario is the top of the interval, not the bottom.
        scenarios = bootstrap_ci(
            self.values, n_bootstrap=n_bootstrap, seed=seed, direction=self.direction
        )
        decision = gate_on_scenarios(
            scenarios, limit=self.limit, direction=self.direction, margin=self.margin
        )
        return {
            "metric": self.name,
            "unit": self.unit,
            "limit": self.limit,
            "direction": self.direction,
            "scenarios": scenarios,
            "decision": decision.decision,
            "reason": decision.reason,
        }


@dataclass
class GateSynthesisAgent:
    """Combines per-metric gates into a single auditable release verdict."""

    seed: int = 0
    n_bootstrap: int = 1000
    history: list[dict] = field(default_factory=list)

    def synthesize(self, gates: list[MetricGate]) -> dict:
        if not gates:
            raise ValueError("need at least one metric gate to make a decision")

        results = [g.evaluate(seed=self.seed, n_bootstrap=self.n_bootstrap) for g in gates]
        worst = max((r["decision"] for r in results), key=lambda d: _SEVERITY_ORDER[d])

        blocking = [r["metric"] for r in results if r["decision"] == "BLOCK"]
        conditional = [r["metric"] for r in results if r["decision"] == "CONDITIONAL"]

        decision = {
            "decision": worst,
            "n_metrics": len(results),
            "blocking_metrics": blocking,
            "conditional_metrics": conditional,
            "metrics": results,
            "rationale": self._rationale(worst, blocking, conditional),
        }
        self.history.append(
            {
                "decision": worst,
                "blocking_metrics": blocking,
                "n_metrics": len(results),
            }
        )
        return decision

    @staticmethod
    def _rationale(worst: str, blocking: list[str], conditional: list[str]) -> str:
        if worst == "BLOCK":
            return f"blocked by {', '.join(blocking)}: the pessimistic scenario breaches its limit"
        if worst == "CONDITIONAL":
            return f"ship with sign-off: {', '.join(conditional)} sit inside the accept margin"
        return "all metrics pass on their pessimistic scenario"

    def run_rate(self) -> dict:
        """Release pass rate over recorded history — the number a team is judged on."""
        if not self.history:
            return {"n_releases": 0, "pass_rate": None, "block_rate": None}
        n = len(self.history)
        blocked = sum(1 for h in self.history if h["decision"] == "BLOCK")
        conditional = sum(1 for h in self.history if h["decision"] == "CONDITIONAL")
        return {
            "n_releases": n,
            "pass_rate": round(sum(1 for h in self.history if h["decision"] == "PASS") / n, 4),
            "conditional_rate": round(conditional / n, 4),
            "block_rate": round(blocked / n, 4),
            "top_blocking_metrics": self._top_blockers(),
        }

    def _top_blockers(self) -> list[dict]:
        counter: dict[str, int] = {}
        for h in self.history:
            for metric in h["blocking_metrics"]:
                counter[metric] = counter.get(metric, 0) + 1
        ranked = sorted(counter.items(), key=lambda kv: (-kv[1], kv[0]))
        return [{"metric": m, "blocks": c} for m, c in ranked]

    def explain(self, decision: dict) -> str:
        """Human-readable summary for a PR comment or a release note."""
        lines = [f"Release gate: {decision['decision']}", f"Why: {decision['rationale']}"]
        for m in decision["metrics"]:
            s = m["scenarios"]
            lines.append(
                f"  - {m['metric']}: {s[Scenario.PESSIMISTIC]:.4g} / {s[Scenario.BASE]:.4g} / "
                f"{s[Scenario.OPTIMISTIC]:.4g} {m['unit']} (limit {m['limit']:.4g}) -> {m['decision']}"
            )
        return "\n".join(lines)

    @staticmethod
    def scenarios_as_dict(decision: dict) -> dict[str, Any]:
        """JSON-safe view of a decision (enums are not serialisable)."""
        out = dict(decision)
        out["metrics"] = [
            {**m, "scenarios": {s.value: v for s, v in m["scenarios"].items()}}
            for m in decision["metrics"]
        ]
        return out
