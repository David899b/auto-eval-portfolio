"""Tests for the evaluation harness.

These are the tests the repo never had. The first one is a regression guard
for a bug that shipped: three of five metrics were silently NaN because the
bootstrap was handed a single pre-aggregated number instead of per-item values.

    python tests/test_harness.py
    pytest tests/test_harness.py
"""
from __future__ import annotations

import asyncio
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from ai_eval.harness import (  # noqa: E402
    EvaluationHarness,
    GoldenItem,
    GoldenSetManager,
    MetricName,
    ModelAdapter,
    SchemaJudge,
    Verdict,
    _dump,
)
from ai_eval.stats import Scenario  # noqa: E402


@dataclass
class Doc:
    """Stand-in for a pydantic model: same ``model_validate``/``model_dump`` API."""

    label: str = "a"
    score: float = 0.0

    def model_dump(self) -> dict:
        return {"label": self.label, "score": self.score}

    @classmethod
    def model_validate(cls, data: dict) -> "Doc":
        if "label" not in data:
            raise ValueError("label is required")
        if not data["label"]:
            raise ValueError("label must be non-empty")
        return cls(label=data["label"], score=float(data.get("score", 0.0)))


def _items(n: int = 30) -> list[GoldenItem]:
    return [
        GoldenItem(
            id=f"item-{i}",
            input_data={"text": f"doc {i}"},
            expected_output=Doc(label="pos" if i % 2 == 0 else "neg", score=0.9),
            split="test",
        )
        for i in range(n)
    ]


class _StaticAdapter(ModelAdapter):
    """Always predicts correctly, with a fixed cost and a small latency jitter."""

    def __init__(self, correct: bool = True, jitter: int = 0):
        self.correct = correct
        self.jitter = jitter
        self.calls = 0

    async def predict(self, input_data: dict, schema: type) -> tuple[Doc, dict]:
        self.calls += 1
        expected_label = "pos" if int(input_data["text"].split()[-1]) % 2 == 0 else "neg"
        label = expected_label if self.correct else ("pos" if expected_label == "neg" else "neg")
        return Doc(label=label, score=0.9), {
            "cost_usd": 0.001,
            "tokens": 100,
            "_jitter": self.jitter,
        }

    def get_model_id(self) -> str:
        return "static-adapter"


def _harness(tmp: str, adapter: ModelAdapter, n: int = 30) -> EvaluationHarness:
    mgr = GoldenSetManager(Path(tmp))
    mgr.create_from_items(
        items=_items(n),
        version="1.0",
        annotators=["annotator-a", "annotator-b"],
        inter_rater_kappa=0.91,
        schema_version="v1",
    )
    return EvaluationHarness(
        golden_set_manager=mgr,
        model_adapter=adapter,
        judge=SchemaJudge(schema=Doc),
        gates={
            MetricName.CLASSIFICATION_ACCURACY: {Scenario.PESSIMISTIC: 0.80},
            MetricName.EXTRACTION_F1: {Scenario.PESSIMISTIC: 0.90},
        },
        seed=0,
    )


def test_harness_imports_without_third_party_deps():
    """The repo must be runnable with a bare Python install."""
    import ai_eval.harness as mod

    source = Path(mod.__file__).read_text()
    for banned in ("import pandas", "import numpy", "from rich", "import pydantic"):
        assert banned not in source, f"{banned} is back in the harness"


def test_dump_is_duck_typed_not_pydantic_bound():
    assert _dump(Doc(label="x")) == {"label": "x", "score": 0.0}
    assert _dump({"already": "plain"}) == {"already": "plain"}
    assert _dump(7) == 7


def test_no_metric_is_nan_on_a_healthy_run():
    """REGRESSION GUARD.

    The old code bootstrapped single pre-aggregated values for derivation
    rate, latency p95 and cost, and the ``len < 2`` guard returned NaN for
    every one of them — while the README still published those numbers.
    """
    with tempfile.TemporaryDirectory() as tmp:
        h = _harness(tmp, _StaticAdapter())
        report = asyncio.run(h.evaluate())

        assert set(report.metrics) >= {
            MetricName.CLASSIFICATION_ACCURACY,
            MetricName.EXTRACTION_F1,
            MetricName.DERIVATION_RATE,
            MetricName.LATENCY_P95_MS,
            MetricName.COST_PER_1K_TOKENS,
        }
        for metric, scenarios in report.metrics.items():
            for scenario in Scenario:
                value = scenarios[scenario]
                assert value == value, f"{metric.value}/{scenario.value} is NaN"
                assert value < float("inf"), f"{metric.value}/{scenario.value} is inf"


def test_scenarios_are_ordered_worst_case_first():
    """Scenarios must read worst-first, and "worst" depends on the metric.

    For accuracy/F1 (higher is better) the pessimistic case is the bottom of the
    interval. For latency and cost (higher is worse) it is the top. Asserting a
    single ascending order for every metric is what allowed bootstrap_ci to map
    the optimistic tail of a latency interval to PESSIMISTIC and report a
    regression as an improvement.
    """
    higher_is_worse = {MetricName.LATENCY_P95_MS, MetricName.COST_PER_1K_TOKENS}
    with tempfile.TemporaryDirectory() as tmp:
        h = _harness(tmp, _StaticAdapter())
        report = asyncio.run(h.evaluate())
        for metric, scenarios in report.metrics.items():
            p, b, o = (
                scenarios[Scenario.PESSIMISTIC],
                scenarios[Scenario.BASE],
                scenarios[Scenario.OPTIMISTIC],
            )
            if metric in higher_is_worse:
                assert p >= b >= o, f"{metric.value} scenarios not worst-first"
            else:
                assert p <= b <= o, f"{metric.value} scenarios out of order"
        acc = report.metrics[MetricName.CLASSIFICATION_ACCURACY]
        for scenario in Scenario:
            assert 0.0 <= acc[scenario] <= 1.0
        cost = report.metrics[MetricName.COST_PER_1K_TOKENS]
        # 0.001 per call -> 1.0 per 1k calls, allow CI spread
        assert 0.5 < cost[Scenario.BASE] < 1.5


def test_evaluation_is_reproducible_from_seed():
    """Seeded metrics must not move between identical runs.

    Latency is deliberately excluded: it is wall-clock measured, so demanding
    reproducibility there would be demanding a lie. Everything derived from the
    data has to be stable, or a regression cannot be diffed.
    """
    wall_clock = {MetricName.LATENCY_P95_MS}
    with tempfile.TemporaryDirectory() as tmp:
        first = asyncio.run(_harness(tmp, _StaticAdapter()).evaluate())
        second = asyncio.run(_harness(tmp, _StaticAdapter()).evaluate())
        for metric in first.metrics:
            if metric in wall_clock:
                continue
            assert first.metrics[metric] == second.metrics[metric], (
                f"{metric.value} moved between identical runs"
            )


def test_wrong_predictions_lower_accuracy_and_trip_the_gate():
    with tempfile.TemporaryDirectory() as tmp:
        good = asyncio.run(_harness(tmp, _StaticAdapter(correct=True)).evaluate())
        bad = asyncio.run(_harness(tmp, _StaticAdapter(correct=False)).evaluate())
        g = good.metrics[MetricName.CLASSIFICATION_ACCURACY][Scenario.PESSIMISTIC]
        b = bad.metrics[MetricName.CLASSIFICATION_ACCURACY][Scenario.PESSIMISTIC]
        assert g > b
        assert bad.gate_results.get("classification_accuracy_pessimistic_gate") is False
        assert good.gate_results.get("classification_accuracy_pessimistic_gate") is True


def test_schema_violation_fails_extraction_f1():
    with tempfile.TemporaryDirectory() as tmp:
        class BrokenAdapter(_StaticAdapter):
            async def predict(self, input_data: dict, schema: type):
                self.calls += 1
                return Doc(label="", score=0.0), {"cost_usd": 0.001}

        h = _harness(tmp, BrokenAdapter())
        report = asyncio.run(h.evaluate())
        assert report.metrics[MetricName.EXTRACTION_F1][Scenario.PESSIMISTIC] < 0.5
        assert report.gate_results.get("extraction_f1_pessimistic_gate") is False


def test_golden_set_freeze_is_content_addressed():
    with tempfile.TemporaryDirectory() as tmp:
        mgr = GoldenSetManager(Path(tmp))
        items = _items(10)
        meta_a = mgr.create_from_items(items, "1.0", ["a", "b"], 0.9, "v1")
        meta_b = mgr.create_from_items(items, "1.0", ["a", "b"], 0.9, "v1")
        assert meta_a.hash_sha256 == meta_b.hash_sha256

        mutated = _items(10)
        mutated[0] = GoldenItem(
            id=mutated[0].id,
            input_data=mutated[0].input_data,
            expected_output=Doc(label="flipped", score=0.1),
        )
        meta_c = mgr.create_from_items(mutated, "1.0", ["a", "b"], 0.9, "v1")
        assert meta_c.hash_sha256 != meta_a.hash_sha256


def test_golden_set_roundtrips_through_disk():
    with tempfile.TemporaryDirectory() as tmp:
        mgr = GoldenSetManager(Path(tmp))
        mgr.create_from_items(_items(12), "1.0", ["a", "b"], 0.88, "v1")
        meta, items = mgr.load()
        assert meta.version == "1.0"
        assert len(items) == 12
        assert mgr.verify_integrity() is True

        # without a schema the round trip is plain dicts
        assert isinstance(items[0].expected_output, dict)


def test_golden_set_rehydrates_typed_items_on_load():
    """A frozen set that cannot be loaded back typed is a decorative freeze."""
    with tempfile.TemporaryDirectory() as tmp:
        mgr = GoldenSetManager(Path(tmp))
        mgr.create_from_items(_items(12), "1.0", ["a", "b"], 0.88, "v1")
        _, items = mgr.load(schema=Doc)
        assert all(isinstance(i.expected_output, Doc) for i in items)
        assert items[0].expected_output.label in {"pos", "neg"}


def test_harness_only_evaluates_the_requested_split():
    with tempfile.TemporaryDirectory() as tmp:
        mgr = GoldenSetManager(Path(tmp))
        items = _items(10)
        items[0] = GoldenItem(
            id=items[0].id,
            input_data=items[0].input_data,
            expected_output=items[0].expected_output,
            split="train",
        )
        mgr.create_from_items(items, "1.0", ["a", "b"], 0.9, "v1")
        h = EvaluationHarness(mgr, _StaticAdapter(), SchemaJudge(Doc), {})
        report = asyncio.run(h.evaluate(split="test"))
        assert all(p.item_id != "item-0" for p in report.per_item)
        assert report.total_items == 9


def test_print_summary_runs_without_rich(capsys=None):
    with tempfile.TemporaryDirectory() as tmp:
        h = _harness(tmp, _StaticAdapter())
        report = asyncio.run(h.evaluate())
        report.print_summary()  # must not raise
        report.to_dict()  # must be JSON-serialisable


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
