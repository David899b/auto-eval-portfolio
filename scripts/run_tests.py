#!/usr/bin/env python3
"""Run every test suite in the repo with a bare Python interpreter.

One command, one exit code. CI, pre-commit and a human all use this, so
"green locally" and "green in CI" cannot drift apart.

    python scripts/run_tests.py
    python scripts/run_tests.py --verbose
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

SUITES = [
    ("core/stats", "tests/test_stats.py"),
    ("core/agents", "tests/test_agents.py"),
    ("core/harness", "tests/test_harness.py"),
    ("core/cli", "tests/test_cli.py"),
    ("core/api", "tests/test_api.py"),
    ("projects/arena", "projects/arena/tests/test_arena.py"),
    ("projects/redteam-dx", "projects/redteam-dx/tests/test_redteam_dx.py"),
    ("projects/eval-watch", "projects/eval-watch/tests/test_eval_watch.py"),
]

# Suites that must not silently vanish: if a file disappears from disk the run
# fails instead of reporting one less suite and calling it a pass.
REQUIRED = [p for _, p in SUITES]


def _run_suite(label: str, rel: str, verbose: bool) -> tuple[bool, str, float]:
    path = ROOT / rel
    if not path.exists():
        return False, f"{label}: MISSING {rel}", 0.0
    start = time.perf_counter()
    proc = subprocess.run(
        [sys.executable, str(path)],
        cwd=ROOT,
        capture_output=not verbose,
        text=True,
    )
    elapsed = time.perf_counter() - start
    tail = ""
    if not verbose and proc.stdout:
        summary = [ln for ln in proc.stdout.strip().splitlines() if "passed" in ln]
        tail = summary[-1] if summary else proc.stdout.strip().splitlines()[-1:]
    ok = proc.returncode == 0
    detail = tail if isinstance(tail, str) else (tail[0] if tail else "")
    return ok, f"{label:<20} {'PASS' if ok else 'FAIL'}  {detail}", elapsed


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    missing = [r for r in REQUIRED if not (ROOT / r).exists()]
    if missing:
        print(f"FAIL: expected test suites are missing: {missing}", file=sys.stderr)
        return 1

    print(f"Running {len(SUITES)} suites with {sys.executable}\n")
    results = [_run_suite(label, rel, args.verbose) for label, rel in SUITES]

    for _, line, elapsed in results:
        print(f"  {line}  ({elapsed:.2f}s)")

    failed = [label for ok, label, _ in results if not ok]
    total_time = sum(e for _, _, e in results)
    print(f"\n{len(SUITES) - len(failed)}/{len(SUITES)} suites passed in {total_time:.2f}s")
    if failed:
        print(f"FAIL: {', '.join(failed)}", file=sys.stderr)
        if not args.verbose:
            print("re-run with -v for full output", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
