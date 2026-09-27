"""Compliance Agent — evidence, not vibes.

For any system handling personal data, the question is never "are we
compliant?" but "can you produce the evidence on demand?". This agent turns a
run's artifacts into a checkable evidence pack: what was evaluated, against
which frozen baseline, and with which gate outcome.

Scope note: this generates *evidence artifacts*. It is not a certification and
does not replace a legal review.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

# Controls this agent can produce evidence for, mapped to what the artifact is.
CONTROL_ARTIFACTS: dict[str, str] = {
    "frozen_baseline": "versioned golden set with SHA-256 freeze",
    "red_team_evidence": "adversarial scan with per-finding severity",
    "drift_monitoring": "PSI/KL drift record per release",
    "release_gate": "3-scenario gate decision per release",
    "data_minimization": "fields retained vs discarded, per dataset",
    "human_review": "disagreement queue resolved by a named reviewer",
}


@dataclass
class ComplianceAgent:
    """Builds a reproducible evidence pack for one evaluation run."""

    system_name: str
    data_categories: list[str] = field(default_factory=list)
    retention_policy_days: int | None = None

    def build_evidence_pack(self, run_artifacts: dict[str, Any]) -> dict:
        """Assemble the pack and flag any control with no artifact attached.

        A missing artifact is reported as ``MISSING`` rather than assumed
        compliant — an absent control is the finding.
        """
        controls: list[dict] = []
        for control, expected in CONTROL_ARTIFACTS.items():
            artifact = run_artifacts.get(control)
            controls.append(
                {
                    "control": control,
                    "expected_artifact": expected,
                    "status": "PRESENT" if artifact else "MISSING",
                    "artifact": artifact,
                }
            )

        missing = [c["control"] for c in controls if c["status"] == "MISSING"]
        return {
            "system": self.system_name,
            "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "data_categories": self.data_categories,
            "retention_policy_days": self.retention_policy_days,
            "n_controls": len(controls),
            "coverage_pct": round(
                (len(controls) - len(missing)) / len(controls) * 100, 2
            )
            if controls
            else 0.0,
            "missing_controls": missing,
            "verdict": "COMPLETE" if not missing else "INCOMPLETE",
            "controls": controls,
            "disclaimer": (
                "Evidence artifact generator, not a certification. Does not replace "
                "a legal or data-protection review."
            ),
        }

    def fingerprint(self, pack: dict) -> str:
        """Content hash of the pack, so the evidence itself is tamper-evident."""
        payload = json.dumps(pack, sort_keys=True, default=str).encode()
        return hashlib.sha256(payload).hexdigest()

    def retention_check(self, run_dates: list[str], reference_date: str | None = None) -> dict:
        """Flag runs older than the retention policy.

        Keeping personal data past its stated retention window is one of the
        most common — and most avoidable — findings in a data-protection review.
        """
        if self.retention_policy_days is None:
            return {
                "status": "NO_POLICY",
                "reason": "retention_policy_days not configured; cannot evaluate",
            }
        if not run_dates:
            return {"status": "NO_RUNS", "n_runs": 0, "expired": []}

        reference = datetime.fromisoformat(
            (reference_date or datetime.now(timezone.utc).isoformat(timespec="seconds"))
            .replace("Z", "+00:00")
        )
        expired: list[dict] = []
        for run in run_dates:
            ts = datetime.fromisoformat(run.replace("Z", "+00:00"))
            age_days = (reference - ts).days
            if age_days > self.retention_policy_days:
                expired.append({"run": run, "age_days": age_days})
        return {
            "status": "VIOLATION" if expired else "COMPLIANT",
            "policy_days": self.retention_policy_days,
            "n_runs": len(run_dates),
            "n_expired": len(expired),
            "expired": expired,
        }
