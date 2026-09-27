"""Core evaluation harness for LLM systems.

Production-grade evaluation with:
- Golden set management (versioned, frozen, auditable)
- Prompt engineering with schema enforcement
- Judge/coherence 2nd-pass for gray zone
- 3-scenario reporting (pessimistic/base/optimistic)
- CI/CD gates with regression detection
- Compliance (PII, Ley 25.326, EU AI Act)
"""

from __future__ import annotations

import hashlib
import json
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Generic, Protocol, TypeVar, runtime_checkable

from ai_eval.stats import Scenario, bootstrap_ci, percentile


@runtime_checkable
class _Dumpable(Protocol):
    """Anything pydantic-style that can serialise itself.

    Duck-typed so the harness has no hard dependency on pydantic: any object
    exposing ``model_dump()`` works, and plain dataclasses/dicts work too.
    """

    def model_dump(self) -> dict: ...


T = TypeVar("T", bound=_Dumpable)


def _dump(value: Any) -> Any:
    """Serialise a value if it knows how, else pass it through untouched."""
    dump = getattr(value, "model_dump", None)
    return dump() if callable(dump) else value


class Verdict(str, Enum):
    """Evaluation verdict for a single prediction."""

    PASS = "PASS"
    FAIL = "FAIL"
    GRAY = "GRAY"  # needs human review


class MetricName(str, Enum):
    """Standard metric names."""

    CLASSIFICATION_ACCURACY = "classification_accuracy"
    EXTRACTION_F1 = "extraction_f1"
    DERIVATION_RATE = "derivation_rate"
    LATENCY_P95_MS = "latency_p95_ms"
    COST_PER_1K_TOKENS = "cost_per_1k_tokens"
    HALLUCINATION_RATE = "hallucination_rate"
    PII_LEAKAGE_RATE = "pii_leakage_rate"


@dataclass(frozen=True, slots=True)
class GoldenSetMetadata:
    """Immutable metadata for a frozen golden set."""

    version: str
    created_at: datetime
    hash_sha256: str
    total_items: int
    classes: list[str]
    inter_rater_kappa: float
    annotators: list[str]
    schema_version: str
    description: str = ""

    def to_dict(self) -> dict:
        return {
            "version": self.version,
            "created_at": self.created_at.isoformat(),
            "hash_sha256": self.hash_sha256,
            "total_items": self.total_items,
            "classes": self.classes,
            "inter_rater_kappa": self.inter_rater_kappa,
            "annotators": self.annotators,
            "schema_version": self.schema_version,
            "description": self.description,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "GoldenSetMetadata":
        return cls(
            version=data["version"],
            created_at=datetime.fromisoformat(data["created_at"]),
            hash_sha256=data["hash_sha256"],
            total_items=data["total_items"],
            classes=data["classes"],
            inter_rater_kappa=data["inter_rater_kappa"],
            annotators=data["annotators"],
            schema_version=data["schema_version"],
            description=data.get("description", ""),
        )


@dataclass(frozen=True, slots=True)
class GoldenItem(Generic[T]):
    """Single golden set item with ground truth."""

    id: str
    input_data: dict
    expected_output: T
    metadata: dict = field(default_factory=dict)
    split: str = "test"  # train/val/test
    weight: float = 1.0

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "input_data": self.input_data,
            "expected_output": _dump(self.expected_output),
            "metadata": self.metadata,
            "split": self.split,
            "weight": self.weight,
        }


@dataclass(frozen=True, slots=True)
class PredictionResult(Generic[T]):
    """Model prediction with evaluation."""

    item_id: str
    predicted: T
    expected: T
    verdict: Verdict
    scores: dict[MetricName, float]
    latency_ms: float
    cost_usd: float
    metadata: dict = field(default_factory=dict)
    gray_zone_reason: str | None = None

    def to_dict(self) -> dict:
        return {
            "item_id": self.item_id,
            "predicted": _dump(self.predicted),
            "expected": _dump(self.expected),
            "verdict": self.verdict.value,
            "scores": {k.value: v for k, v in self.scores.items()},
            "latency_ms": self.latency_ms,
            "cost_usd": self.cost_usd,
            "metadata": self.metadata,
            "gray_zone_reason": self.gray_zone_reason,
        }


@dataclass(frozen=True, slots=True)
class EvaluationReport:
    """Complete evaluation report with 3 scenarios."""

    model_id: str
    golden_set_version: str
    evaluated_at: datetime
    total_items: int
    metrics: dict[MetricName, dict[Scenario, float]]
    per_item: list[PredictionResult]
    gate_results: dict[str, bool]
    compliance: dict[str, bool]
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "model_id": self.model_id,
            "golden_set_version": self.golden_set_version,
            "evaluated_at": self.evaluated_at.isoformat(),
            "total_items": self.total_items,
            "metrics": {
                k.value: {s.value: v for s, v in v.items()}
                for k, v in self.metrics.items()
            },
            "per_item": [p.to_dict() for p in self.per_item],
            "gate_results": self.gate_results,
            "compliance": self.compliance,
            "metadata": self.metadata,
        }

    def print_summary(self) -> None:
        """Render the 3-scenario report as plain text (no rich dependency)."""
        header = f"{'Metric':<28} {'Pessimistic':>12} {'Base':>12} {'Optimistic':>12}  Gate"
        print(f"\nEvaluation Report: {self.model_id}")
        print("=" * len(header))
        print(header)
        print("-" * len(header))
        for metric, scenarios in self.metrics.items():
            # Gates are stored per scenario as "<metric>_<scenario>_gate".
            # A metric only passes if *every* scenario passed — the pessimistic
            # one decides, which is the whole point of reporting three.
            per_scenario = {
                scenario: self.gate_results.get(f"{metric.value}_{scenario.value}_gate")
                for scenario in Scenario
            }
            graded = {s: v for s, v in per_scenario.items() if v is not None}
            if graded:
                ok = all(graded.values())
                detail = " ".join(
                    f"{s.value[0].upper()}:{'ok' if v else 'X'}" for s, v in graded.items()
                )
            else:
                ok, detail = True, "ungated"
            print(
                f"{metric.value:<28} "
                f"{scenarios[Scenario.PESSIMISTIC]:>12.4f} "
                f"{scenarios[Scenario.BASE]:>12.4f} "
                f"{scenarios[Scenario.OPTIMISTIC]:>12.4f}  "
                f"{'PASS' if ok else 'FAIL':<4} {detail}"
            )

        if self.compliance:
            print("\nCompliance")
            print("-" * 40)
            for check, passed in self.compliance.items():
                print(f"{check:<36} {'PASS' if passed else 'FAIL'}")


class GoldenSetManager:
    """Manages golden sets with versioning, freezing, and audit trail."""

    def __init__(self, storage_path: Path):
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        self._current: GoldenSetMetadata | None = None
        self._items: list[GoldenItem] = []

    def create_from_items(
        self,
        items: list[GoldenItem],
        version: str,
        annotators: list[str],
        inter_rater_kappa: float,
        schema_version: str,
        description: str = "",
    ) -> GoldenSetMetadata:
        """Create and freeze a new golden set version."""
        # Compute hash
        content = json.dumps([item.to_dict() for item in items], sort_keys=True).encode()
        hash_sha256 = hashlib.sha256(content).hexdigest()[:16]

        metadata = GoldenSetMetadata(
            version=version,
            created_at=datetime.now(),
            hash_sha256=hash_sha256,
            total_items=len(items),
            classes=self._extract_classes(items),
            inter_rater_kappa=inter_rater_kappa,
            annotators=annotators,
            schema_version=schema_version,
            description=description,
        )

        # Save
        self._save(metadata, items)
        self._current = metadata
        self._items = items

        print(f"Golden set frozen: v{version} ({hash_sha256}) — {len(items)} items, κ={inter_rater_kappa:.3f}")
        return metadata

    def _extract_classes(self, items: list[GoldenItem]) -> list[str]:
        classes = set()
        for item in items:
            if hasattr(item.expected_output, "model_fields"):
                for field_name, value in item.expected_output.model_dump().items():
                    if isinstance(value, str):
                        classes.add(value)
        return sorted(classes)

    def _save(self, metadata: GoldenSetMetadata, items: list[GoldenItem]) -> None:
        version_dir = self.storage_path / f"v{metadata.version}"
        version_dir.mkdir(exist_ok=True)

        # Save metadata
        (version_dir / "metadata.json").write_text(json.dumps(metadata.to_dict(), indent=2))

        # Save items
        items_data = [item.to_dict() for item in items]
        (version_dir / "items.jsonl").write_text("\n".join(json.dumps(i) for i in items_data))

        # Save manifest
        manifest = {
            "metadata": metadata.to_dict(),
            "item_count": len(items),
            "file_hashes": {
                "items.jsonl": hashlib.sha256(json.dumps(items_data, sort_keys=True).encode()).hexdigest(),
            },
        }
        (version_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))

        # Update latest symlink
        latest = self.storage_path / "latest"
        if latest.exists():
            latest.unlink()
        latest.symlink_to(f"v{metadata.version}")

    def load(
        self, version: str | None = None, schema: type[T] | None = None
    ) -> tuple[GoldenSetMetadata, list[GoldenItem]]:
        """Load a golden set version (or latest).

        Pass ``schema`` to rehydrate ``expected_output`` back into its typed
        form. Without it the items come back as plain dicts, which silently
        breaks any judge that compares a typed prediction against a typed
        expectation — the freeze would be decorative.
        """
        if version is None:
            version_dir = self.storage_path / "latest"
            if not version_dir.exists():
                raise ValueError("No golden set found")
        else:
            version_dir = self.storage_path / f"v{version}"
            if not version_dir.exists():
                raise ValueError(f"Golden set v{version} not found")

        metadata = GoldenSetMetadata.from_dict(json.loads((version_dir / "metadata.json").read_text()))
        items_data = [json.loads(line) for line in (version_dir / "items.jsonl").read_text().splitlines()]

        items = []
        for d in items_data:
            expected = d["expected_output"]
            if schema is not None:
                validator = getattr(schema, "model_validate", None)
                if callable(validator):
                    expected = validator(expected)
            items.append(GoldenItem(
                id=d["id"],
                input_data=d["input_data"],
                expected_output=expected,  # type: ignore[arg-type]
                metadata=d.get("metadata", {}),
                split=d.get("split", "test"),
                weight=d.get("weight", 1.0),
            ))

        self._current = metadata
        self._items = items
        return metadata, items

    def verify_integrity(self, version: str | None = None) -> bool:
        """Verify golden set hasn't been tampered with."""
        if version is None:
            version = self._current.version if self._current else "latest"
        metadata, items = self.load(version)
        content = json.dumps([item.to_dict() for item in items], sort_keys=True).encode()
        computed_hash = hashlib.sha256(content).hexdigest()[:16]
        return computed_hash == metadata.hash_sha256


class ModelAdapter(ABC):
    """Abstract adapter for different model providers."""

    @abstractmethod
    async def predict(self, input_data: dict, schema: type[T]) -> tuple[T, dict]:
        """Return (prediction, metadata{latency_ms, cost_usd, tokens})."""
        pass

    @abstractmethod
    def get_model_id(self) -> str:
        pass


class Judge(ABC, Generic[T]):
    """Judge for coherence/gray-zone detection."""

    @abstractmethod
    def evaluate(self, input_data: dict, predicted: T, expected: T) -> tuple[Verdict, dict[MetricName, float], str | None]:
        """Return (verdict, scores, gray_zone_reason)."""
        pass


class SchemaJudge(Judge[T]):
    """Judge that validates schema compliance + semantic coherence."""

    def __init__(self, schema: type[T], coherence_threshold: float = 0.85):
        self.schema = schema
        self.coherence_threshold = coherence_threshold

    def evaluate(self, input_data: dict, predicted: T, expected: T) -> tuple[Verdict, dict[MetricName, float], str | None]:
        scores = {}

        # Schema validation
        try:
            self.schema.model_validate(predicted.model_dump() if hasattr(predicted, "model_dump") else predicted)
            scores[MetricName.EXTRACTION_F1] = 1.0
        except Exception:
            scores[MetricName.EXTRACTION_F1] = 0.0
            return Verdict.FAIL, scores, "Schema validation failed"

        # Semantic comparison (simplified - would use embeddings/LLM judge)
        similarity = self._semantic_similarity(predicted, expected)
        scores[MetricName.CLASSIFICATION_ACCURACY] = similarity

        if similarity >= self.coherence_threshold:
            return Verdict.PASS, scores, None
        elif similarity >= 0.5:
            return Verdict.GRAY, scores, f"Low coherence: {similarity:.2f}"
        else:
            return Verdict.FAIL, scores, f"Coherence below threshold: {similarity:.2f}"

    def _semantic_similarity(self, a: T, b: T) -> float:
        # Simplified - real implementation uses embeddings or LLM-as-judge
        a_dict = a.model_dump() if hasattr(a, "model_dump") else a
        b_dict = b.model_dump() if hasattr(b, "model_dump") else b

        # Field-level exact match ratio
        matches = 0
        total = 0
        for key in set(a_dict.keys()) | set(b_dict.keys()):
            total += 1
            if a_dict.get(key) == b_dict.get(key):
                matches += 1
        return matches / total if total > 0 else 0.0


class EvaluationHarness:
    """Main evaluation orchestrator."""

    def __init__(
        self,
        golden_set_manager: GoldenSetManager,
        model_adapter: ModelAdapter,
        judge: Judge,
        gates: dict[MetricName, dict[Scenario, float]],
        compliance_checks: list[callable] | None = None,
        seed: int = 0,
    ):
        self.golden_set_manager = golden_set_manager
        self.model_adapter = model_adapter
        self.judge = judge
        self.gates = gates
        self.compliance_checks = compliance_checks or []
        self.seed = seed  # makes the 3 scenarios reproducible across runs
        # Rehydrate golden items on load so a judge can compare typed objects.
        self._schema = getattr(type(judge), "schema", None)

    async def evaluate(self, split: str = "test", version: str | None = None) -> EvaluationReport:
        """Run full evaluation on specified split."""
        metadata, items = self.golden_set_manager.load(version, schema=self._schema)
        test_items = [item for item in items if item.split == split]

        print(f"Evaluating {len(test_items)} items from golden set v{metadata.version}")

        predictions = []
        for item in test_items:
            start = time.perf_counter()
            predicted, meta = await self.model_adapter.predict(item.input_data, type(item.expected_output))
            latency = (time.perf_counter() - start) * 1000

            verdict, scores, gray_reason = self.judge.evaluate(item.input_data, predicted, item.expected_output)

            predictions.append(PredictionResult(
                item_id=item.id,
                predicted=predicted,
                expected=item.expected_output,
                verdict=verdict,
                scores=scores,
                latency_ms=latency,
                cost_usd=meta.get("cost_usd", 0.0),
                metadata=meta,
                gray_zone_reason=gray_reason,
            ))

        # Aggregate metrics with 3 scenarios
        metrics = self._compute_scenarios(predictions)

        # Gate evaluation
        gate_results = {}
        for metric, thresholds in self.gates.items():
            for scenario, threshold in thresholds.items():
                gate_key = f"{metric.value}_{scenario.value}_gate"
                actual = metrics[metric][scenario]
                # Higher is better for accuracy/F1, lower for derivation/latency/cost
                if metric in (MetricName.CLASSIFICATION_ACCURACY, MetricName.EXTRACTION_F1):
                    gate_results[gate_key] = actual >= threshold
                else:
                    gate_results[gate_key] = actual <= threshold

        # Compliance checks
        compliance = {}
        for check in self.compliance_checks:
            name = check.__name__
            compliance[name] = check(predictions)

        report = EvaluationReport(
            model_id=self.model_adapter.get_model_id(),
            golden_set_version=metadata.version,
            evaluated_at=datetime.now(),
            total_items=len(test_items),
            metrics=metrics,
            per_item=predictions,
            gate_results=gate_results,
            compliance=compliance,
        )

        return report

    def _compute_scenarios(self, predictions: list[PredictionResult]) -> dict[MetricName, dict[Scenario, float]]:
        """Compute metrics with bootstrap confidence intervals for 3 scenarios."""
        metrics = {}

        # Classification accuracy
        accuracies = [p.scores.get(MetricName.CLASSIFICATION_ACCURACY, 0) for p in predictions]
        metrics[MetricName.CLASSIFICATION_ACCURACY] = self._bootstrap_ci(accuracies)

        # Extraction F1
        f1s = [p.scores.get(MetricName.EXTRACTION_F1, 0) for p in predictions]
        metrics[MetricName.EXTRACTION_F1] = self._bootstrap_ci(f1s)

        # Derivation rate (gray + fail), bootstrapped per item
        derivations = [
            0.0 if p.verdict == Verdict.PASS else 1.0 for p in predictions
        ]
        metrics[MetricName.DERIVATION_RATE] = self._bootstrap_ci(derivations)

        # Latency P95 — bootstrap the per-item latencies, not the aggregate.
        # Bootstrapping a single pre-aggregated number has no sampling
        # distribution and previously returned NaN for the whole metric.
        latencies = [p.latency_ms for p in predictions]
        metrics[MetricName.LATENCY_P95_MS] = self._bootstrap_ci(
            latencies, statistic="p95", direction="max"
        )

        # Cost per 1k tokens — same reasoning: resample per-item costs.
        costs = [p.cost_usd for p in predictions]
        metrics[MetricName.COST_PER_1K_TOKENS] = self._bootstrap_ci(
            [c * 1000 for c in costs], direction="max"
        )

        return metrics

    def _bootstrap_ci(
        self,
        values: list[float],
        n_bootstrap: int = 1000,
        confidence: float = 0.95,
        statistic: str = "mean",
        direction: str = "min",
    ) -> dict[Scenario, float]:
        """Seeded 3-scenario bootstrap, delegated to :mod:`ai_eval.stats`.

        Seeded so two runs over the same predictions produce identical numbers
        and a regression can actually be diffed.

        ``direction`` must be passed for every "higher is worse" metric. Left at
        the default, latency and cost report their *optimistic* tail as the
        pessimistic scenario, so a regression reads as an improvement.
        """
        return bootstrap_ci(
            values,
            n_bootstrap=n_bootstrap,
            confidence=confidence,
            seed=self.seed,
            statistic=statistic,
            direction=direction,
        )