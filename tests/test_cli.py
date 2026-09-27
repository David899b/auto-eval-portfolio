"""Tests for the zero-dependency CLI.

The CLI is the surface a reviewer actually touches first, so it gets tested for
the thing that matters most: a failing gate must produce a non-zero exit code.
A CLI that reports BLOCK and exits 0 is worse than no CLI, because CI goes green.
"""
from __future__ import annotations

import io
import json
import sys
import tempfile
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from auto_eval.cli import main  # noqa: E402

PASSED: list[str] = []
FAILED: list[str] = []


def check(name: str, condition: bool, detail: str = "") -> None:
    if condition:
        PASSED.append(name)
        print(f"PASS {name}")
    else:
        FAILED.append(name)
        print(f"FAIL {name} {detail}")


def run(argv: list[str], stdin: str = "") -> tuple[int, str, str]:
    """Invoke the CLI and capture (exit_code, stdout, stderr)."""
    out, err = io.StringIO(), io.StringIO()
    saved_stdin = sys.stdin
    sys.stdin = io.StringIO(stdin)
    try:
        with redirect_stdout(out), redirect_stderr(err):
            try:
                code = main(argv)
            except SystemExit as exc:
                # SystemExit.code is either an int or a message string; a
                # message means the CLI raised SystemExit("error: ...").
                code = exc.code if isinstance(exc.code, int) else 1
                if not isinstance(exc.code, int):
                    print(exc.code, file=sys.stderr)
    finally:
        sys.stdin = saved_stdin
    return code, out.getvalue(), err.getvalue()


def _corpus(n: int) -> list[dict]:
    return [
        {"id": i, "text": f"item {i}", "label": "pos" if i % 3 == 0 else "neg",
         "vector": "prompt_injection" if i % 2 else "jailbreak"}
        for i in range(n)
    ]


def _write(directory: Path, name: str, payload: object) -> str:
    path = directory / name
    path.write_text(json.dumps(payload), encoding="utf-8")
    return str(path)


# ------------------------------------------------------------------ golden-set
def test_golden_set_create_measures_kappa_and_exits_zero() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        corpus = _corpus(60)
        a = ["pos" if i % 3 == 0 else "neg" for i in range(60)]
        b = list(a)
        for i in range(5, 20):
            b[i] = "pos" if a[i] == "neg" else "neg"
        src = _write(tmp_path, "src.json", {"items": corpus, "labeler_a": a, "labeler_b": b})

        code, out, _ = run([
            "golden-set", "create", "--input", src,
            "--storage", str(tmp_path / "gs"), "--n-samples", "30",
        ])
        check("golden-set create exits 0", code == 0, f"got {code}")
        record = json.loads(out)
        check("golden-set create reports measured kappa",
              record["measured_kappa"] is not None)
        check("golden-set create flags below-target kappa",
              record["kappa_status"] == "below_target", record["kappa_status"])


def test_golden_set_verify_detects_tampering() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        storage = tmp_path / "gs"
        src = _write(tmp_path, "src.json", _corpus(60))
        code, out, _ = run([
            "golden-set", "create", "--input", src, "--storage", str(storage), "--n-samples", "30",
        ])
        check("verify setup create exits 0", code == 0)
        record = json.loads(out)
        persisted = json.loads((storage / "golden_set_v1.json").read_text(encoding="utf-8"))
        sha = record["metadata"]["sha256"]
        check("persisted item count matches metadata",
              len(persisted["items"]) == record["metadata"]["n_items"])

        good = _write(tmp_path, "good.json", {"items": persisted["items"], "expected_sha256": sha})
        code, out, _ = run(["golden-set", "verify", "--input", good])
        check("verify exits 0 on an intact set", code == 0, f"got {code}")
        check("verify reports verified=true", json.loads(out)["verified"] is True)

        tampered_items = [dict(i) for i in persisted["items"]]
        tampered_items[0]["text"] = "TAMPERED"
        bad = _write(tmp_path, "bad.json", {"items": tampered_items, "expected_sha256": sha})
        code, out, _ = run(["golden-set", "verify", "--input", bad])
        check("verify exits 1 on a tampered set", code == 1, f"got {code}")
        check("verify reports verified=false", json.loads(out)["verified"] is False)


# ------------------------------------------------------------------------ drift
def test_drift_numeric_blocked_exits_one() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        stable = [float(i % 17) for i in range(400)]
        shifted = [float(i % 17) + 25.0 for i in range(400)]

        ok_src = _write(tmp_path, "ok.json", {"baseline": stable, "current": stable})
        code, out, _ = run(["drift", "numeric", "--input", ok_src])
        check("drift numeric exits 0 when stable", code == 0, f"got {code}")
        check("drift numeric reports no drift", json.loads(out)["drift_detected"] is False)

        bad_src = _write(tmp_path, "bad.json", {"baseline": stable, "current": shifted})
        code, out, err = run(["drift", "numeric", "--input", bad_src])
        check("drift numeric exits 1 when drifted", code == 1, f"got {code}")
        check("drift numeric prints a gate failure to stderr", "GATE FAILED" in err)
        check("drift numeric reports severity", json.loads(out)["severity"] in
              {"significant", "severe"}, json.loads(out)["severity"])


def test_drift_latency_breach_exits_one() -> None:
    """Regression: the latency gate reported BLOCK but exited 0.

    The distribution check reports ``drift_detected`` and the latency gate
    reports ``is_blocking``. Reading only the first made a breached SLO pass CI.
    """
    with tempfile.TemporaryDirectory() as tmp:
        src = _write(
            Path(tmp), "lat.json",
            {"baseline": [100.0] * 300, "current": [900.0] * 300},
        )
        code, out, err = run(["drift", "latency", "--input", src, "--slo-p95-ms", "500"])
        check("drift latency exits 1 on SLO breach", code == 1, f"got {code}")
        check("drift latency reports BLOCK", json.loads(out)["decision"] == "BLOCK")
        check("drift latency prints the reason to stderr",
              "GATE FAILED" in err and "limit" in err, err.strip())


# --------------------------------------------------------------------- red team
def test_red_team_blocks_on_a_vulnerable_target() -> None:
    code, out, err = run(["red-team", "--fail-on-critical"])
    check("red-team exits 1 against an echoing target", code == 1, f"got {code}")
    report = json.loads(out)
    check("red-team runs every probe", report["n_probes"] == 6, str(report["n_probes"]))
    check("red-team reports BLOCK", report["decision"] == "BLOCK")
    check("red-team prints a gate failure to stderr", "GATE FAILED" in err)


def test_red_team_passes_without_the_critical_flag() -> None:
    code, _, _ = run(["red-team"])
    check("red-team exits 0 without --fail-on-critical", code == 0, f"got {code}")


# -------------------------------------------------------------------------- gate
def test_gate_blocks_when_one_metric_breaches() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        src = _write(Path(tmp), "gates.json", [
            {"name": "f1", "values": [0.91, 0.88, 0.93], "limit": 0.85, "direction": "min"},
            {"name": "p95_latency_ms", "values": [430, 505, 610], "limit": 500, "direction": "max"},
        ])
        code, out, _ = run(["gate", "--input", src])
        check("gate exits 1 when a tail breaches", code == 1, f"got {code}")
        decision = json.loads(out)
        check("gate decision is BLOCK", decision["decision"] == "BLOCK")
        check("gate names the blocking metric",
              decision["blocking_metrics"] == ["p95_latency_ms"],
              str(decision["blocking_metrics"]))


def test_gate_passes_when_every_metric_is_healthy() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        src = _write(Path(tmp), "gates.json", [
            {"name": "f1", "values": [0.91, 0.88, 0.93], "limit": 0.85, "direction": "min"},
            {"name": "p95_latency_ms", "values": [410, 430, 395], "limit": 500, "direction": "max"},
        ])
        code, out, _ = run(["gate", "--input", src])
        check("gate exits 0 when all metrics clear", code == 0, f"got {code}")
        check("gate decision is PASS", json.loads(out)["decision"] == "PASS")


# -------------------------------------------------------------------- compliance
def test_compliance_incomplete_pack_exits_one() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        src = _write(Path(tmp), "artifacts.json", {
            "frozen_baseline": "v1 sha256:abc",
            "release_gate": "PASS",
        })
        code, out, err = run([
            "compliance", "--system", "support-bot",
            "--data-categories", "pii", "--input", src,
        ])
        check("compliance exits 1 on an incomplete pack", code == 1, f"got {code}")
        pack = json.loads(out)
        check("compliance verdict is INCOMPLETE", pack["verdict"] == "INCOMPLETE")
        check("coverage is below 100%", pack["coverage_pct"] < 100.0)
        check("missing controls are named", len(pack["missing_controls"]) == 4,
              str(pack["missing_controls"]))
        check("pack is fingerprinted", len(pack["fingerprint"]) == 64)
        check("compliance prints a gate failure", "GATE FAILED" in err)


def test_compliance_complete_pack_exits_zero() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        src = _write(Path(tmp), "artifacts.json", {
            "frozen_baseline": "v1 sha256:abc",
            "red_team_evidence": "6 probes, 0 exploited",
            "drift_monitoring": "PSI 0.01",
            "release_gate": "PASS",
            "data_minimization": "kept: id, text",
            "human_review": "disagreements resolved by D. Bautista",
        })
        code, out, _ = run([
            "compliance", "--system", "support-bot", "--input", src,
        ])
        check("compliance exits 0 on a complete pack", code == 0, f"got {code}")
        pack = json.loads(out)
        check("compliance verdict is COMPLETE", pack["verdict"] == "COMPLETE")
        check("coverage is 100%", pack["coverage_pct"] == 100.0)


def test_compliance_flags_expired_retention() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        src = _write(Path(tmp), "artifacts.json", {
            "frozen_baseline": "v1", "red_team_evidence": "clean",
            "drift_monitoring": "stable", "release_gate": "PASS",
            "data_minimization": "kept: id", "human_review": "none",
        })
        code, out, _ = run([
            "compliance", "--system", "support-bot", "--input", src,
            "--retention-days", "30",
            "--retention-dates", "2000-01-01T00:00:00+00:00",
        ])
        check("compliance exits 1 on a retention violation", code == 1, f"got {code}")
        pack = json.loads(out)
        check("retention status is VIOLATION", pack["retention"]["status"] == "VIOLATION")
        check("retention violation downgrades the verdict", pack["verdict"] == "INCOMPLETE")


# ------------------------------------------------------------------ bad payloads
def test_empty_stdin_is_a_clean_error_not_a_traceback() -> None:
    for argv in (["golden-set", "create"], ["gate"], ["drift", "numeric"]):
        code, _, err = run(argv, stdin="")
        label = " ".join(argv)
        check(f"{label} rejects empty stdin", code != 0, f"got {code}")
        check(f"{label} says the input was empty", "expected JSON input" in err, err.strip()[:120])
        check(f"{label} does not leak a traceback", "Traceback" not in err)


def test_wrong_shaped_payloads_are_rejected_with_guidance() -> None:
    """A payload of the wrong type must name the shape it expected."""
    cases = [
        (["golden-set", "create"], "42", "expects a JSON list of items"),
        (["golden-set", "verify"], '{"items": []}', "expects {items, expected_sha256}"),
        (["golden-set", "refresh"], '{"items": []}', "expects {production_logs, current_items}"),
        (["gate"], "{}", "non-empty JSON list of metric gates"),
        (["compliance", "--system", "s"], "[]", "expects a JSON object"),
        (["drift", "numeric"], "[]", "drift expects {baseline, current}"),
    ]
    for argv, payload, expected in cases:
        code, _, err = run(argv, stdin=payload)
        label = " ".join(argv[:2])
        check(f"{label} rejects a bad payload", code != 0, f"got {code}")
        check(f"{label} names the expected shape", expected in err, err.strip()[:140])
        check(f"{label} does not leak a traceback", "Traceback" not in err)


def test_malformed_json_is_reported_as_json_not_a_crash() -> None:
    code, _, err = run(["gate"], stdin="{not json")
    check("gate rejects malformed JSON", code != 0, f"got {code}")
    check("gate does not leak a traceback on malformed JSON", "Traceback" not in err)


if __name__ == "__main__":
    for fn in [
        test_golden_set_create_measures_kappa_and_exits_zero,
        test_golden_set_verify_detects_tampering,
        test_drift_numeric_blocked_exits_one,
        test_drift_latency_breach_exits_one,
        test_red_team_blocks_on_a_vulnerable_target,
        test_red_team_passes_without_the_critical_flag,
        test_gate_blocks_when_one_metric_breaches,
        test_gate_passes_when_every_metric_is_healthy,
        test_compliance_incomplete_pack_exits_one,
        test_compliance_complete_pack_exits_zero,
        test_compliance_flags_expired_retention,
        test_empty_stdin_is_a_clean_error_not_a_traceback,
        test_wrong_shaped_payloads_are_rejected_with_guidance,
        test_malformed_json_is_reported_as_json_not_a_crash,
    ]:
        fn()
    print(f"\n{len(PASSED)}/{len(PASSED) + len(FAILED)} passed")
    if FAILED:
        print("failed: " + ", ".join(FAILED))
        raise SystemExit(1)
