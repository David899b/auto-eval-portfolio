"""Tests for the five agents. The point of these tests is that they would FAIL
against the previous placeholder implementations.

Zero dependencies, runnable either way:

    python tests/test_agents.py
    pytest tests/test_agents.py
"""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from ai_eval.stats import Scenario, cohens_kappa  # noqa: E402
from auto_eval.agents.compliance_agent import ComplianceAgent  # noqa: E402
from auto_eval.agents.drift_agent import DriftAgent  # noqa: E402
from auto_eval.agents.gate_synthesis_agent import (  # noqa: E402
    GateSynthesisAgent,
    MetricGate,
)
from auto_eval.agents.golden_set_agent import GoldenSetAgent  # noqa: E402
from auto_eval.agents.red_team_agent import RedTeamAgent  # noqa: E402


def _corpus(n: int = 200) -> list[dict]:
    return [
        {"id": i, "label": "pos" if i % 3 == 0 else "neg", "text": f"doc {i}"}
        for i in range(n)
    ]


# --------------------------------------------------------------- golden set
def test_golden_set_reports_measured_kappa_not_the_target():
    """Regression guard: the old stub returned ``target_kappa`` as the result."""
    with tempfile.TemporaryDirectory() as tmp:
        agent = GoldenSetAgent(storage_path=Path(tmp), target_kappa=0.8)
        corpus = _corpus(60)
        # Two deliberately mediocre labelers over the *full* corpus. They agree
        # on the pattern, then disagree on a stretch of it, so the measured
        # kappa lands below the 0.8 target and the agent has to say so.
        a = ["pos" if i % 3 == 0 else "neg" for i in range(len(corpus))]
        b = list(a)
        for i in range(5, 20):  # introduce a block of disagreement
            b[i] = "pos" if a[i] == "neg" else "neg"

        result = agent.generate_initial_set(
            corpus, n_samples=30, labeler_a=a, labeler_b=b
        )
        measured = result["measured_kappa"]
        assert measured is not None
        assert measured != result["target_kappa"], "kappa must be measured, not echoed"
        assert result["kappa_status"] == "below_target"
        assert measured < result["target_kappa"]


def test_golden_set_kappa_is_measured_on_the_sampled_subset():
    """Labelers align with source_data; kappa must reflect the rows kept.

    If the agent measured agreement over the whole corpus instead of the
    stratified sample that actually became the golden set, the reported kappa
    would describe a set that does not exist.
    """
    with tempfile.TemporaryDirectory() as tmp:
        agent = GoldenSetAgent(storage_path=Path(tmp), target_kappa=0.8, seed=5)
        corpus = _corpus(60)
        a = ["pos" if i % 3 == 0 else "neg" for i in range(60)]
        b = list(a)
        for i in range(0, 60, 7):  # sprinkle disagreement across the whole corpus
            b[i] = "pos" if a[i] == "neg" else "neg"

        result = agent.generate_initial_set(corpus, n_samples=30, labeler_a=a, labeler_b=b)
        persisted = json.loads(
            (Path(tmp) / "golden_set_v1.json").read_text(encoding="utf-8")
        )
        sampled_labels_a = [a[corpus.index(row)] for row in persisted["items"]]

        expected = cohens_kappa(sampled_labels_a, [b[corpus.index(row)] for row in persisted["items"]])
        assert abs(result["measured_kappa"] - expected) < 1e-9


def test_golden_set_kappa_is_none_when_nothing_to_compare():
    with tempfile.TemporaryDirectory() as tmp:
        agent = GoldenSetAgent(storage_path=Path(tmp), target_kappa=0.8)
        result = agent.generate_initial_set(_corpus(40), n_samples=20)
        assert result["measured_kappa"] is None
        assert result["kappa_status"] == "not_measured"


def test_golden_set_is_reproducible_from_seed():
    with tempfile.TemporaryDirectory() as tmp:
        a = GoldenSetAgent(storage_path=Path(tmp), seed=42)
        b = GoldenSetAgent(storage_path=Path(tmp), seed=42)
        c = GoldenSetAgent(storage_path=Path(tmp), seed=43)
        ra = a.generate_initial_set(_corpus(100), n_samples=25)
        rb = b.generate_initial_set(_corpus(100), n_samples=25)
        rc = c.generate_initial_set(_corpus(100), n_samples=25)
        assert ra["metadata"]["sha256"] == rb["metadata"]["sha256"]
        assert ra["metadata"]["sha256"] != rc["metadata"]["sha256"]


def test_golden_set_freeze_hash_detects_tampering():
    with tempfile.TemporaryDirectory() as tmp:
        agent = GoldenSetAgent(storage_path=Path(tmp))
        result = agent.generate_initial_set(_corpus(40), n_samples=20)
        items = json.loads((Path(tmp) / "golden_set_v1.json").read_text())["items"]
        assert agent.verify_integrity(items, result["metadata"]["sha256"])
        items[0]["label"] = "tampered"
        assert not agent.verify_integrity(items, result["metadata"]["sha256"])


def test_golden_set_stratifies_instead_of_uniform_sampling():
    """A 90/10 corpus must not yield a set that mirrors that skew blindly."""
    with tempfile.TemporaryDirectory() as tmp:
        agent = GoldenSetAgent(storage_path=Path(tmp), seed=7)
        corpus = [{"label": "rare", "id": i} for i in range(20)]
        corpus += [{"label": "common", "id": i} for i in range(180)]
        result = agent.generate_initial_set(corpus, n_samples=100)
        dist = result["class_distribution"]
        assert dist["rare"] >= 10, "minority class was dropped by sampling"
        assert set(dist) == {"rare", "common"}


def test_golden_set_labeler_length_mismatch_raises():
    with tempfile.TemporaryDirectory() as tmp:
        agent = GoldenSetAgent(storage_path=Path(tmp))
        try:
            agent.generate_initial_set(
                _corpus(20), n_samples=10, labeler_a=["a"], labeler_b=["a", "b"]
            )
        except ValueError:
            pass
        else:
            raise AssertionError("misaligned labelers must raise")


def test_refresh_cycle_measures_real_drift_and_bumps_version():
    with tempfile.TemporaryDirectory() as tmp:
        agent = GoldenSetAgent(storage_path=Path(tmp), target_kappa=0.8)
        golden = [{"label": "pos" if i % 2 == 0 else "neg"} for i in range(100)]
        # production shifted to a 90/10 mix -> real, measurable drift
        prod = [{"label": "neg" if i % 10 == 0 else "pos"} for i in range(100)]

        drifted = agent.refresh_cycle(prod, golden, current_version="1.0")
        assert drifted["drift_detected"] is True
        assert drifted["recommended_version"] == "1.1"
        assert drifted["action"] == "review_and_refreeze"
        assert drifted["total_variation_distance"] > drifted["alert_threshold"]
        assert drifted["moved_classes"]

        stable = agent.refresh_cycle(golden, golden, current_version="1.0")
        assert stable["drift_detected"] is False
        assert stable["recommended_version"] == "1.0"
        assert stable["action"] == "no_change"
        assert stable["total_variation_distance"] == 0.0


def test_refresh_cycle_empty_input_raises():
    with tempfile.TemporaryDirectory() as tmp:
        agent = GoldenSetAgent(storage_path=Path(tmp))
        try:
            agent.refresh_cycle([], [{"label": "pos"}])
        except ValueError:
            pass
        else:
            raise AssertionError("empty production logs must raise")


def test_curate_adversarial_selects_by_vector():
    with tempfile.TemporaryDirectory() as tmp:
        agent = GoldenSetAgent(storage_path=Path(tmp))
        items = [{"id": i, "tags": "prompt_injection" if i % 2 else "jailbreak"} for i in range(40)]
        result = agent.curate_adversarial(items, per_vector=5)
        assert result["selected_by_vector"]["prompt_injection"] == 5
        assert result["selected_by_vector"]["jailbreak"] == 5
        assert result["n_selected"] == 10
        # order follows the default vector order, not alphabetical
        assert result["covered_vectors"] == ["prompt_injection", "jailbreak"]
        assert result["coverage_pct"] == 50.0
        # vectors with no tagged items must be reported, not silently dropped
        assert result["under_target"] == ["pii_leakage", "hallucination"]


# --------------------------------------------------------------------- drift
def test_drift_agent_detects_and_severities():
    agent = DriftAgent(psi_alert=0.1)
    baseline = [200.0 + (i % 10) for i in range(300)]
    stable = agent.check_distribution(baseline, baseline)
    assert stable["drift_detected"] is False
    assert stable["severity"] == "stable"

    shifted = agent.check_distribution(baseline, [500.0 + (i % 10) for i in range(300)])
    assert shifted["drift_detected"] is True
    assert shifted["severity"] in {"significant", "severe"}


def test_drift_agent_label_mix_reports_moved_classes():
    agent = DriftAgent(psi_alert=0.1)
    golden = ["a"] * 90 + ["b"] * 10
    prod = ["a"] * 10 + ["b"] * 90
    result = agent.check_label_mix(golden, prod)
    assert result["drift_detected"] is True
    assert {m["class"] for m in result["moved_classes"]} == {"a", "b"}


def test_drift_agent_latency_gate_blocks_on_pessimistic():
    agent = DriftAgent(seed=0)
    baseline = [150.0 + (i % 20) for i in range(500)]
    regressed = [400.0 + (i % 20) for i in range(500)]
    result = agent.gate_latency(baseline, regressed, slo_p95_ms=200.0)
    assert result["decision"] == "BLOCK"
    assert result["is_blocking"] is True
    assert result["delta_p95_ms"] > 0
    assert set(result["scenarios_ms"]) == set(Scenario)


def test_drift_agent_latency_gate_passes_when_healthy():
    agent = DriftAgent(seed=0)
    baseline = [150.0 + (i % 20) for i in range(500)]
    result = agent.gate_latency(baseline, baseline, slo_p95_ms=250.0)
    assert result["decision"] in {"PASS", "CONDITIONAL"}


def test_drift_agent_latency_scenarios_are_ordered_worst_first():
    """Regression: for latency the pessimistic scenario is the *top* of the CI.

    Before the fix, bootstrap_ci always mapped the low tail to PESSIMISTIC, so a
    latency gate compared the best case against the SLO and passed releases whose
    tail had regressed.
    """
    agent = DriftAgent(seed=0)
    baseline = [150.0 + (i % 20) for i in range(500)]
    regressed = [150.0 + (i % 20) for i in range(400)] + [400.0] * 100
    s = agent.gate_latency(baseline, regressed, slo_p95_ms=250.0)["scenarios_ms"]
    assert s[Scenario.PESSIMISTIC] >= s[Scenario.BASE] >= s[Scenario.OPTIMISTIC]
    assert s[Scenario.PESSIMISTIC] >= s[Scenario.OPTIMISTIC]


def test_drift_agent_latency_gate_blocks_a_breached_slo():
    agent = DriftAgent(seed=0)
    baseline = [150.0 + (i % 20) for i in range(500)]
    regressed = [400.0] * 500
    result = agent.gate_latency(baseline, regressed, slo_p95_ms=250.0)
    assert result["scenarios_ms"][Scenario.PESSIMISTIC] > 250.0
    assert result["decision"] == "BLOCK"
    assert result["is_blocking"] is True


def test_drift_agent_empty_input_raises():
    agent = DriftAgent()
    try:
        agent.check_distribution([], [1.0])
    except ValueError:
        pass
    else:
        raise AssertionError("empty baseline must raise")


# ----------------------------------------------------------------- red team
def _leaky_target(prompt: str) -> str:
    return "Sure! My system prompt is: you are a helpful assistant. Here it is."


def _safe_target(prompt: str) -> str:
    return "I can't help with that request."


def test_red_team_blocks_on_critical_exploitation():
    agent = RedTeamAgent(target=_leaky_target)
    report = agent.run()
    assert report["decision"] == "BLOCK"
    assert report["is_blocking"] is True
    assert report["by_severity"].get("critical", 0) >= 1
    assert report["exploit_rate"] > 0
    # findings must be plain dicts so they survive json.dumps in CI
    json.dumps(report["findings"])


def test_red_team_passes_against_a_refusing_target():
    agent = RedTeamAgent(target=_safe_target)
    report = agent.run()
    assert report["decision"] == "PASS"
    assert report["exploited_count"] == 0


def test_red_team_empty_response_counts_as_exploited():
    """Silence is not a safe default for a compliance probe."""
    agent = RedTeamAgent(target=lambda p: "")
    report = agent.run()
    assert report["exploited_count"] == report["n_probes"]


def test_red_team_target_exception_is_captured_not_raised():
    def boom(prompt: str) -> str:
        raise RuntimeError("upstream 500")

    agent = RedTeamAgent(target=boom)
    report = agent.run()
    assert report["decision"] == "BLOCK"
    assert "upstream 500" in report["findings"][0]["evidence"]


def test_red_team_probes_are_deterministic():
    a = RedTeamAgent(target=_safe_target).build_probes()
    b = RedTeamAgent(target=_safe_target).build_probes()
    assert [p.id for p in a] == [p.id for p in b]
    assert [p.severity for p in a] == [p.severity for p in b]


def test_red_team_custom_vector_set():
    agent = RedTeamAgent(target=_safe_target, vectors=["hallucination"])
    report = agent.run()
    assert report["n_probes"] == 1
    assert report["vectors"] == ["hallucination"]


# --------------------------------------------------------------------- gates
def test_gate_blocks_when_one_metric_regresses_despite_good_average():
    agent = GateSynthesisAgent(seed=0)
    gates = [
        MetricGate("f1", [0.97] * 40, limit=0.90, direction="min"),
        MetricGate("latency_p95_ms", [180.0, 500.0, 520.0, 540.0], limit=250.0, direction="max"),
    ]
    decision = agent.synthesize(gates)
    assert decision["decision"] == "BLOCK"
    assert "latency_p95_ms" in decision["blocking_metrics"]


def test_gate_passes_when_every_metric_is_healthy():
    agent = GateSynthesisAgent(seed=0)
    gates = [
        MetricGate("f1", [0.97] * 40, limit=0.90, direction="min"),
        MetricGate("anls", [0.98] * 40, limit=0.95, direction="min"),
    ]
    assert agent.synthesize(gates)["decision"] == "PASS"


def test_gate_conditional_when_inside_margin():
    agent = GateSynthesisAgent(seed=0)
    gates = [MetricGate("f1", [0.905] * 40, limit=0.90, direction="min", margin=0.02)]
    assert agent.synthesize(gates)["decision"] == "CONDITIONAL"


def test_gate_requires_at_least_one_metric():
    agent = GateSynthesisAgent()
    try:
        agent.synthesize([])
    except ValueError:
        pass
    else:
        raise AssertionError("empty gate list must raise")


def test_gate_run_rate_tracks_blocks_over_history():
    agent = GateSynthesisAgent(seed=0)
    good = [MetricGate("f1", [0.97] * 30, limit=0.90, direction="min")]
    bad = [MetricGate("f1", [0.10] * 30, limit=0.90, direction="min")]
    agent.synthesize(good)
    agent.synthesize(bad)
    agent.synthesize(bad)
    stats = agent.run_rate()
    assert stats["n_releases"] == 3
    assert stats["block_rate"] > 0.6
    assert stats["top_blocking_metrics"][0] == {"metric": "f1", "blocks": 2}


def test_gate_explain_and_json_roundtrip():
    agent = GateSynthesisAgent(seed=0)
    decision = agent.synthesize(
        [MetricGate("f1", [0.97] * 30, limit=0.90, direction="min", unit="score")]
    )
    text = agent.explain(decision)
    assert "Release gate: PASS" in text
    assert "f1" in text
    payload = agent.scenarios_as_dict(decision)
    json.dumps(payload)  # must be serialisable for CI artifacts


# ---------------------------------------------------------------- compliance
def test_compliance_pack_flags_missing_controls():
    agent = ComplianceAgent(system_name="demo", data_categories=["personal_data"])
    pack = agent.build_evidence_pack({"frozen_baseline": {"version": "1.0"}})
    assert pack["verdict"] == "INCOMPLETE"
    assert "red_team_evidence" in pack["missing_controls"]
    assert pack["coverage_pct"] < 100.0
    assert "not a certification" in pack["disclaimer"]


def test_compliance_pack_complete_when_all_controls_present():
    agent = ComplianceAgent(system_name="demo")
    artifacts = {
        "frozen_baseline": {"version": "1.0"},
        "red_team_evidence": {"exploited": 0},
        "drift_monitoring": {"psi": 0.01},
        "release_gate": {"decision": "PASS"},
        "data_minimization": {"retained": ["a"]},
        "human_review": {"queue": 0},
    }
    pack = agent.build_evidence_pack(artifacts)
    assert pack["verdict"] == "COMPLETE"
    assert pack["missing_controls"] == []
    assert pack["coverage_pct"] == 100.0


def test_compliance_fingerprint_is_stable_and_changes_with_content():
    agent = ComplianceAgent(system_name="demo")
    pack = agent.build_evidence_pack({})
    assert agent.fingerprint(pack) == agent.fingerprint(dict(pack))
    mutated = {**pack, "system": "other"}
    assert agent.fingerprint(pack) != agent.fingerprint(mutated)


def test_compliance_retention_flags_expired_runs():
    agent = ComplianceAgent(system_name="demo", retention_policy_days=30)
    result = agent.retention_check(
        ["2026-01-01T00:00:00+00:00", "2026-09-25T00:00:00+00:00"],
        reference_date="2026-09-26T00:00:00+00:00",
    )
    assert result["status"] == "VIOLATION"
    assert result["n_expired"] == 1
    assert result["expired"][0]["run"].startswith("2026-01-01")


def test_compliance_retention_without_policy_refuses_to_guess():
    agent = ComplianceAgent(system_name="demo", retention_policy_days=None)
    assert agent.retention_check(["2026-01-01T00:00:00+00:00"])["status"] == "NO_POLICY"


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
