"""Tests for redteam-dx probe logic (pure python)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from redteam_dx.redteam_dx import RedTeamDX, ProbeSuites, demo_target


def test_suites_defined():
    assert "prompt_injection" in ProbeSuites
    assert "jailbreak" in ProbeSuites
    assert "pii_leakage" in ProbeSuites
    assert "hallucination" in ProbeSuites


def test_refusing_target_is_clean():
    rt = RedTeamDX(lambda p: "I can't help with that.")
    report = rt.run()
    assert report["exploited_count"] == 0
    assert all(sev == 0 for sev in report["by_severity"].values())


def test_leaky_target_detected():
    rt = RedTeamDX(demo_target)
    report = rt.run()
    assert report["exploited_count"] >= 1
    assert report["by_severity"]["critical"] >= 1


def test_findings_iterable_of_dicts():
    rt = RedTeamDX(demo_target)
    report = rt.run()
    for f in report["findings"]:
        assert "probe" in f and "exploited" in f and "evidence" in f


def _run() -> int:
    """Shared runner: report failures and exit non-zero so CI can gate on it."""
    tests = [
        (n, f)
        for n, f in sorted(globals().items())
        if n.startswith("test_") and callable(f)
    ]
    failures = 0
    for name, fn in tests:
        try:
            fn()
        except Exception as exc:  # noqa: BLE001
            failures += 1
            print(f"FAIL {name}: {type(exc).__name__}: {exc}")
        else:
            print(f"PASS {name}")
    print(f"\n{len(tests) - failures}/{len(tests)} passed")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(_run())
