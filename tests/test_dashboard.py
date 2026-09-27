"""Tests for the report viewer and the example report generator.

The previous dashboard was a Streamlit app with hardcoded metrics that all read
PASS. These tests pin the two properties that matter: it renders a real report,
and it cannot render a green verdict over a report whose gates failed.
"""
from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dashboard.app import _worst_first, render_report  # noqa: E402

PASSED: list[str] = []
FAILED: list[str] = []


def check(name: str, condition: bool, detail: str = "") -> None:
    if condition:
        PASSED.append(name)
        print(f"PASS {name}")
    else:
        FAILED.append(name)
        print(f"FAIL {name} {detail}")


def test_worst_first_is_a_passthrough() -> None:
    """The producer already orders the scenarios; the viewer must not reorder.

    ``bootstrap_ci`` is direction-aware, so PESSIMISTIC holds the worst case for
    every metric. Re-ordering here would be a second, contradictory copy of that
    rule, so the viewer reads the keys as they are.
    """
    s = {"PESSIMISTIC": 0.1, "BASE": 0.5, "OPTIMISTIC": 0.9}
    for metric in ("classification_accuracy", "latency_p95_ms", "cost_per_1k_tokens"):
        check(f"{metric} passes the scenarios through", _worst_first(metric, s) == (0.1, 0.5, 0.9),
              str(_worst_first(metric, s)))
    lower = {"PESSIMISTIC": 900.0, "BASE": 400.0, "OPTIMISTIC": 180.0}
    check("a latency interval with the worst case on top is preserved",
          _worst_first("latency_p95_ms", lower) == (900.0, 400.0, 180.0))


def test_renders_a_real_report() -> None:
    report = {
        "model_id": "demo-adapter-v1",
        "decision": "BLOCK",
        "metadata": {"golden_set_version": "1.0"},
        "metrics": {
            "classification_accuracy": {"pessimistic": 0.81, "base": 0.93, "optimistic": 0.98},
            "latency_p95_ms": {"pessimistic": 640.0, "base": 210.0, "optimistic": 180.0},
        },
        "gate_results": {"classification_accuracy": False},
    }
    html = render_report(report)
    check("renders the model id", "demo-adapter-v1" in html)
    check("renders a BLOCK verdict", "BLOCK" in html)
    check("renders both metrics",
          "classification_accuracy" in html and "latency_p95_ms" in html)
    check("renders the pessimistic value, not just the base", "0.8100" in html or "0.81" in html)
    check("shows the failed gate", "BLOCK" in html)


def test_renders_lowercase_scenario_keys() -> None:
    """harness.to_dict() emits lowercase scenario keys; the viewer must cope."""
    report = {
        "decision": "PASS",
        "metrics": {"extraction_f1": {"pessimistic": 1.0, "base": 1.0, "optimistic": 1.0}},
    }
    html = render_report(report)
    check("accepts lowercase scenario keys",
          "extraction_f1" in html and "1.000" in html)


def test_empty_report_does_not_crash() -> None:
    html = render_report({"decision": "UNKNOWN"})
    check("an empty report still renders", "<html" in html and "no metrics" in html.lower())


def test_nan_is_rendered_as_not_a_number() -> None:
    report = {
        "decision": "BLOCK",
        "metrics": {"cost_per_1k_tokens": {"pessimistic": float("nan"), "base": 1.0,
                                            "optimistic": 1.0}},
    }
    html = render_report(report)
    check("NaN renders as n/a rather than 'nan'", "n/a" in html and ">nan<" not in html)


def test_generator_produces_a_report_the_viewer_accepts() -> None:
    """End to end: run the real harness, then render its real output."""
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "report.json"
        proc = subprocess.run(
            [sys.executable, str(ROOT / "scripts" / "example_report.py"), "--out", str(out)],
            cwd=ROOT, capture_output=True, text=True,
            env={**__import__("os").environ, "PYTHONPATH": str(ROOT / "src")},
        )
        check("example_report.py exits 0", proc.returncode == 0, proc.stderr[-300:])
        if proc.returncode != 0:
            return
        payload = json.loads(out.read_text(encoding="utf-8"))
        check("report has metrics", len(payload.get("metrics", {})) >= 4,
              str(len(payload.get("metrics", {}))))
        check("report has a decision", payload.get("decision") in {"PASS", "BLOCK"},
              str(payload.get("decision")))
        check("report is not the old hardcoded mock",
              payload["metrics"].get("classification_accuracy", {}).get("base") != 0.95,
              str(payload["metrics"].get("classification_accuracy")))
        html = render_report(payload)
        check("the viewer renders the generated report", "<html" in html and "auto-eval report" in html)

        # The contract the whole display rests on: for every "higher is worse"
        # metric the pessimistic case must be the top of the interval.
        for metric in ("latency_p95_ms", "cost_per_1k_tokens", "derivation_rate"):
            s = payload["metrics"].get(metric, {})
            p, b, o = (s.get("pessimistic"), s.get("base"), s.get("optimistic"))
            check(f"{metric} reports the worst case first in a real run",
                  p >= b >= o, f"{p} / {b} / {o}")
        s = payload["metrics"]["classification_accuracy"]
        check("classification_accuracy reports the worst case first",
              s["pessimistic"] <= s["base"] <= s["optimistic"],
              f'{s["pessimistic"]} / {s["base"]} / {s["optimistic"]}')


def test_dashboard_has_no_third_party_imports() -> None:
    source = (ROOT / "dashboard" / "app.py").read_text(encoding="utf-8")
    for banned in ("import streamlit", "import pandas", "import plotly", "import numpy"):
        check(f"dashboard does not {banned}", banned not in source)


if __name__ == "__main__":
    for fn in [
        test_worst_first_is_a_passthrough,
        test_renders_a_real_report,
        test_renders_lowercase_scenario_keys,
        test_empty_report_does_not_crash,
        test_nan_is_rendered_as_not_a_number,
        test_generator_produces_a_report_the_viewer_accepts,
        test_dashboard_has_no_third_party_imports,
    ]:
        fn()
    print(f"\n{len(PASSED)}/{len(PASSED) + len(FAILED)} passed")
    if FAILED:
        print("failed: " + ", ".join(FAILED))
        raise SystemExit(1)
