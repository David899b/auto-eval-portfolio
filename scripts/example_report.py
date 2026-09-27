"""Run the harness end to end and write a real report to disk.

Exists so the dashboard has something honest to render. A viewer pointed at
invented numbers teaches you nothing about whether the pipeline works.

    PYTHONPATH=src python scripts/example_report.py --out report.json
    PYTHONPATH=src python -m dashboard.app --report report.json
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from ai_eval.harness import (  # noqa: E402
    EvaluationHarness,
    GoldenItem,
    GoldenSetManager,
    MetricName,
    ModelAdapter,
    SchemaJudge,
)
from ai_eval.stats import Scenario  # noqa: E402


@dataclass
class Doc:
    """Stand-in for a pydantic model: same ``model_validate``/``model_dump`` API."""

    label: str = ""
    score: float = 0.0

    def model_dump(self) -> dict:
        return {"label": self.label, "score": self.score}

    @classmethod
    def model_validate(cls, data: dict) -> "Doc":
        if not data.get("label"):
            raise ValueError("label is required and must be non-empty")
        return cls(label=str(data["label"]), score=float(data.get("score", 0.0)))


class DemoAdapter(ModelAdapter):
    """Deterministic adapter with a realistic error profile.

    Wrong on every 7th item so the accuracy gate has something to say:
    a report where every metric passes teaches a reviewer nothing.

    Note the latency in the report is wall-clock around this (trivial) call, not
    anything the adapter reports. To exercise a latency gate, make ``predict``
    actually sleep, or drive the DriftAgent latency gate directly.
    """

    def __init__(self) -> None:
        self.calls = 0

    async def predict(self, input_data: dict, schema: type) -> tuple[Doc, dict]:
        self.calls += 1
        i = int(str(input_data["text"]).split()[-1])
        expected = "pos" if i % 2 == 0 else "neg"
        # Miss every 7th item: a real adapter is not perfect, and a report where
        # every metric passes teaches a reviewer nothing.
        label = expected if self.calls % 7 else ("pos" if expected == "neg" else "neg")
        # `schema` arrives as the runtime type of expected_output, which the
        # golden set stores as a plain dict, so validate through Doc directly.
        prediction = Doc.model_validate({"label": label, "score": 0.9})
        return prediction, {
            "cost_usd": 0.001,
            "tokens": 100,
            # Not used for the latency metric: the harness times the call with
            # perf_counter. Kept only to show it round-trips into metadata.
            "adapter_reported_latency_ms": 640.0 if self.calls % 6 == 0 else 180.0,
        }

    def get_model_id(self) -> str:
        return "demo-adapter-v1"


async def build_report(n_items: int) -> dict:
    with tempfile.TemporaryDirectory() as tmp:
        mgr = GoldenSetManager(Path(tmp))
        mgr.create_from_items(
            items=[
                GoldenItem(
                    id=f"item-{i}",
                    input_data={"text": f"doc {i}"},
                    expected_output=Doc(label="pos" if i % 2 == 0 else "neg", score=0.9),
                    split="test",
                )
                for i in range(n_items)
            ],
            version="1.0",
            annotators=["annotator-a", "annotator-b"],
            inter_rater_kappa=0.91,
            schema_version="v1",
        )
        harness = EvaluationHarness(
            golden_set_manager=mgr,
            model_adapter=DemoAdapter(),
            judge=SchemaJudge(schema=Doc),
            gates={
                MetricName.CLASSIFICATION_ACCURACY: {Scenario.PESSIMISTIC: 0.80},
                MetricName.EXTRACTION_F1: {Scenario.PESSIMISTIC: 0.90},
            },
            seed=0,
        )
        report = await harness.evaluate()

    payload = report.to_dict()
    payload["decision"] = report.print_summary.__doc__ and _verdict(payload) or ""
    return payload


def _verdict(payload: dict) -> str:
    """Overall verdict: BLOCK if any gate failed, else PASS.

    Derived from the gates the harness actually evaluated. Nothing here is
    hardcoded, so the dashboard cannot render a green light over a red run.
    """
    gates = payload.get("gate_results", {}) or {}
    failed = [name for name, ok in gates.items() if not ok]
    return "BLOCK" if failed else "PASS"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="report.json")
    parser.add_argument("--items", type=int, default=30)
    args = parser.parse_args()

    payload = asyncio.run(build_report(args.items))
    out = Path(args.out)
    out.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")

    metrics = payload.get("metrics", {})
    print(f"wrote {out}")
    print(f"  decision: {payload.get('decision', 'n/a')}")
    print(f"  metrics:  {len(metrics)}")
    for name, scenarios in sorted(metrics.items()):
        print(f"    {name:28} {scenarios}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
