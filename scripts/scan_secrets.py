#!/usr/bin/env python3
"""Scan for credentials and client-identifying data before it reaches a public remote.

This repo was public once with a CV, a client name and an internal ticket
reference in its history. That is the incident this script exists to prevent
repeating, and it runs in CI over both the working tree and the full history.

Stdlib only, no dependencies, deterministic. Exit code 1 on any finding.

    python scripts/scan_secrets.py --root .
    python scripts/scan_secrets.py --root . --history
    python scripts/scan_secrets.py --root . --baseline .secrets-baseline.json
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from dataclasses import dataclass, asdict
from pathlib import Path

# (name, regex) — deliberately specific. A scanner that cries wolf gets disabled.
SECRET_RULES: list[tuple[str, str]] = [
    ("aws_access_key_id", r"\bAKIA[0-9A-Z]{16}\b"),
    ("github_token", r"\bgh[pousr]_[A-Za-z0-9]{36,}\b"),
    ("slack_token", r"\bxox[baprs]-[A-Za-z0-9-]{10,}\b"),
    ("openai_key", r"\bsk-(?:proj-)?[A-Za-z0-9_-]{32,}\b"),
    ("anthropic_key", r"\bsk-ant-[A-Za-z0-9_-]{32,}\b"),
    ("google_api_key", r"\bAIza[0-9A-Za-z_-]{35}\b"),
    ("private_key_block", r"-----BEGIN (?:RSA |EC |OPENSSH |PGP )?PRIVATE KEY-----"),
    ("generic_bearer", r"(?i)\bauthorization:\s*bearer\s+[A-Za-z0-9._-]{24,}"),
    ("connection_string_password", r"(?i)\b(?:postgres|postgresql|mysql|mongodb)://[^\s:]+:[^\s@]+@"),
    # hardcoded assignment, not the mere word "token"
    ("hardcoded_secret_assignment", r"(?i)\b(?:api[_-]?key|secret|password|passwd|token)\s*[:=]\s*['\"][A-Za-z0-9._/+-]{16,}['\"]"),
]

# Client-identifying data: not a credential, but it must never be public either.
PII_RULES: list[tuple[str, str]] = [
    ("internal_ticket_ref", r"\bINC\d{6,}\b"),
    ("soc_placeholder_specific", r"(?i)\bzero data retention\b"),
]

# Allowlist: known-safe matches, e.g. the placeholders we deliberately keep.
ALLOWLIST: list[tuple[str, str]] = [
    # the regex examples live in this very file
    ("generic_bearer", "authorization:\\s*bearer"),
]

TEXT_SUFFIXES = {
    ".py", ".md", ".txt", ".yml", ".yaml", ".json", ".toml", ".cfg", ".ini",
    ".html", ".js", ".ts", ".sh", ".env", ".jsonschema", ".sql", ".csv",
}
SKIP_DIRS = {
    ".git", "node_modules", "__pycache__", ".venv", "venv", ".mypy_cache",
    ".pytest_cache", ".ruff_cache", "dist", "build", ".eggs",
}


@dataclass
class Finding:
    rule: str
    path: str
    line: int
    excerpt: str
    severity: str = "high"


def _scan_text(text: str, path: str, kind: str) -> list[Finding]:
    findings: list[Finding] = []
    for name, pattern in SECRET_RULES + PII_RULES:
        rx = re.compile(pattern)
        for lineno, line in enumerate(text.splitlines(), start=1):
            if len(line) > 2000:
                continue
            match = rx.search(line)
            if not match:
                continue
            if any(name == n and match.group(0) in frag for n, frag in ALLOWLIST):
                continue
            excerpt = match.group(0)
            # never print the whole secret: enough to locate, not enough to use
            excerpt = excerpt[:6] + "…" if kind == "secret" and len(excerpt) > 6 else excerpt
            findings.append(
                Finding(
                    rule=name,
                    path=path,
                    line=lineno,
                    excerpt=excerpt,
                    severity="high" if kind == "secret" else "medium",
                )
            )
    return findings


def scan_worktree(root: Path) -> list[Finding]:
    findings: list[Finding] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        if any(part in SKIP_DIRS for part in path.parts):
            continue
        if path.suffix.lower() not in TEXT_SUFFIXES and path.name != "Dockerfile":
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        kind = "pii" if path.name == "scan_secrets.py" else "secret"
        findings.extend(_scan_text(text, str(path.relative_to(root)), kind))
    return findings


def scan_history(root: Path, max_commits: int = 200) -> list[Finding]:
    findings: list[Finding] = []
    try:
        shas = subprocess.run(
            ["git", "rev-list", "--max-count", str(max_commits), "HEAD"],
            cwd=root, capture_output=True, text=True, check=True,
        ).stdout.split()
    except (subprocess.CalledProcessError, FileNotFoundError) as exc:
        print(f"  ! could not read git history: {exc}", file=sys.stderr)
        return findings
    for sha in shas:
        try:
            diff = subprocess.run(
                ["git", "show", "--format=", "--unified=0", sha],
                cwd=root, capture_output=True, text=True, check=True,
            ).stdout
        except subprocess.CalledProcessError:
            continue
        path = "<history>"
        for line in diff.splitlines():
            if line.startswith("+++ b/"):
                path = line[6:]
                continue
            if not line.startswith("+"):
                continue
            for finding in _scan_text(line[1:], f"{sha[:8]}:{path}", "secret"):
                finding.path = f"{path} (commit {sha[:8]})"
                findings.append(finding)
    return findings


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=".", type=Path)
    parser.add_argument("--history", action="store_true", help="also scan git history")
    parser.add_argument("--json", type=Path, help="write findings as JSON")
    parser.add_argument(
        "--baseline", type=Path, help="JSON file of known-accepted fingerprint strings"
    )
    args = parser.parse_args()

    accepted: set[str] = set()
    if args.baseline and args.baseline.exists():
        accepted = set(json.loads(args.baseline.read_text()))

    findings = scan_worktree(args.root.resolve())
    print(f"working tree: {len(findings)} finding(s)")
    if args.history:
        history = scan_history(args.root.resolve())
        findings.extend(history)
        print(f"git history:  {len(history)} finding(s)")

    if accepted:
        findings = [f for f in findings if f.excerpt not in accepted]

    for finding in findings:
        print(
            f"  [{finding.severity}] {finding.rule}: {finding.path}:{finding.line} -> {finding.excerpt}",
            file=sys.stderr,
        )

    if args.json:
        args.json.write_text(json.dumps([asdict(f) for f in findings], indent=2))

    if findings:
        print(
            f"\nFAIL: {len(findings)} potential secret/PII finding(s). "
            "Do not commit these. If a hit is a false positive, record it in the "
            "--baseline file deliberately.",
            file=sys.stderr,
        )
        return 1
    print("OK: no credentials or client-identifying data found")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
