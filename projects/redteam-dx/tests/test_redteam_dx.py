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


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"PASS {name}")