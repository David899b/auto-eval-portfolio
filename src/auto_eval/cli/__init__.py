"""auto-eval CLI — stdlib only, wired to the real agents.

An earlier version of this file declared Typer + Rich commands whose bodies were
``console.print("Evaluation complete")`` with a ``# TODO`` and no work behind
them. A CLI that reports success without doing anything is worse than no CLI:
it is the fastest way to convince a reviewer that the project is vapour. This
version shells out to the agents that actually compute things and propagates a
non-zero exit code when a gate fails.

Every subcommand reads real inputs (JSON on stdin or a file path) and prints a
real report. No command prints a success message it has not verified.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from auto_eval.agents.compliance_agent import ComplianceAgent
from auto_eval.agents.drift_agent import DriftAgent
from auto_eval.agents.gate_synthesis_agent import GateSynthesisAgent, MetricGate
from auto_eval.agents.golden_set_agent import GoldenSetAgent
from auto_eval.agents.red_team_agent import RedTeamAgent


def _load(path: str | None) -> Any:
    """Read JSON from ``path``, or stdin when ``path`` is ``-``/omitted.

    A malformed payload is a user error, not a crash: piping a truncated file
    into the CLI should print one line, not a stack trace.
    """
    if path is None or path == "-":
        raw = sys.stdin.read()
        origin = "stdin"
    else:
        try:
            raw = Path(path).read_text(encoding="utf-8")
        except OSError as exc:
            raise SystemExit(f"error: cannot read {origin_of(path)}: {exc.strerror}") from None
        origin = f"file {path}"
    if not raw.strip():
        raise SystemExit(f"error: expected JSON input on {origin}, got nothing")
    try:
        return json.loads(raw)
    except json.JSONDecodeError as exc:
        raise SystemExit(
            f"error: {origin} is not valid JSON (line {exc.lineno}, column {exc.colno}): {exc.msg}"
        ) from None


def _emit(payload: Any) -> None:
    print(json.dumps(payload, indent=2, sort_keys=True, default=str))


class _EchoTarget:
    """A deliberately vulnerable target, so the probe suite has something to hit.

    It reflects the payload back. That is not a useful assistant, but it *is* a
    useful test fixture: every probe comes back exploited, which is the outcome
    the gate has to be able to block on.
    """

    def __call__(self, prompt: str) -> str:
        return f"Sure, here you go: {prompt}"


# --------------------------------------------------------------- golden-set
def cmd_golden_set(args: argparse.Namespace) -> int:
    # `verify` and `refresh` are keyed on the envelope; `create`/`curate` accept
    # either a bare list or an envelope carrying two labelers for kappa.
    payload = _load(args.input)
    if args.action == "verify":
        if not isinstance(payload, dict) or "expected_sha256" not in payload:
            raise SystemExit("error: verify expects {items, expected_sha256}")
        agent = GoldenSetAgent(
            storage_path=Path(args.storage), target_kappa=args.target_kappa, seed=args.seed
        )
        ok = agent.verify_integrity(payload["items"], payload["expected_sha256"])
        _emit({"verified": ok, "expected_sha256": payload["expected_sha256"]})
        return 0 if ok else 1

    if args.action == "refresh":
        if not isinstance(payload, dict) or "current_items" not in payload:
            raise SystemExit("error: refresh expects {production_logs, current_items}")
        agent = GoldenSetAgent(
            storage_path=Path(args.storage), target_kappa=args.target_kappa, seed=args.seed
        )
        result = agent.refresh_cycle(
            production_logs=payload["production_logs"],
            current_items=payload["current_items"],
            label_key=args.label_key,
            current_version=args.version,
            alert_threshold=args.alert_threshold,
        )
        _emit(result)
        if result.get("drift", {}).get("drift_detected"):
            print("GATE FAILED: drift detected", file=sys.stderr)
            return 1
        return 0

    labeler_a = labeler_b = None
    data = payload
    if isinstance(payload, dict):
        labeler_a, labeler_b = payload.get("labeler_a"), payload.get("labeler_b")
        data = payload["items"]
    if not isinstance(data, list):
        raise SystemExit(
            "error: golden-set expects a JSON list of items, "
            'or {"items": [...], "labeler_a": [...], "labeler_b": [...]}'
        )
    agent = GoldenSetAgent(
        storage_path=Path(args.storage),
        target_kappa=args.target_kappa,
        seed=args.seed,
    )

    if args.action == "create":
        result = agent.generate_initial_set(
            source_data=data,
            label_key=args.label_key,
            n_samples=args.n_samples,
            labeler_a=labeler_a,
            labeler_b=labeler_b,
        )
    else:  # curate
        result = agent.curate_adversarial(
            items=data,
            attack_vectors=args.attack_vectors,
            per_vector=args.per_vector,
        )

    _emit(result)
    drift = result.get("drift", {})
    if drift.get("drift_detected"):
        print("GATE FAILED: drift detected", file=sys.stderr)
        return 1
    return 0


# --------------------------------------------------------------------- drift
def cmd_drift(args: argparse.Namespace) -> int:
    data = _load(args.input)
    agent = DriftAgent(psi_alert=args.psi_alert, kl_alert=args.kl_alert, seed=args.seed)
    if not isinstance(data, dict):
        raise SystemExit("error: drift expects {baseline, current}")

    if args.mode == "numeric":
        result = agent.check_distribution(data["baseline"], data["current"])
    elif args.mode == "labels":
        result = agent.check_label_mix(data["baseline"], data["current"])
    else:  # latency
        result = agent.gate_latency(
            data["baseline"], data["current"], slo_p95_ms=args.slo_p95_ms
        )

    _emit(result)
    # Two different agents, two different blocking keys: the distribution check
    # reports drift_detected, the latency gate reports is_blocking. Both must
    # fail the command, otherwise CI goes green on a breached SLO.
    if result.get("drift_detected") or result.get("is_blocking"):
        print(
            "GATE FAILED: "
            + ("drift detected" if result.get("drift_detected") else result.get("reason", ""))
        , file=sys.stderr)
        return 1
    return 0


# ------------------------------------------------------------------ red team
def cmd_red_team(args: argparse.Namespace) -> int:
    # The agent is deliberately given a real target: an echoing stub. A suite
    # pointed at nothing cannot produce findings, and reporting "0 exploited"
    # from an empty run would be a fabricated pass.
    target = _EchoTarget()
    agent = RedTeamAgent(target=target, block_on_critical=args.fail_on_critical)
    result = agent.run()
    if args.report:
        print(agent.report())
    else:
        _emit(result)
    if result["is_blocking"]:
        print("GATE FAILED: a critical-severity probe was exploited", file=sys.stderr)
        return 1
    return 0


# --------------------------------------------------------------------- gates
def cmd_gate(args: argparse.Namespace) -> int:
    data = _load(args.input)
    if not isinstance(data, list) or not data:
        raise SystemExit("error: gate expects a non-empty JSON list of metric gates")
    gates = [
        MetricGate(
            name=g["name"],
            values=g["values"],
            limit=g["limit"],
            direction=g.get("direction", "min"),
        )
        for g in data
    ]
    agent = GateSynthesisAgent()
    decision = agent.synthesize(gates)
    _emit(decision)
    # The human-readable rationale goes to stderr so stdout stays parseable JSON
    # and `auto-eval gate --input x.json | jq .decision` actually works.
    print(agent.explain(decision), file=sys.stderr)
    return 0 if decision["decision"] == "PASS" else 1


# ---------------------------------------------------------------- compliance
def cmd_compliance(args: argparse.Namespace) -> int:
    data = _load(args.input)
    if not isinstance(data, dict):
        raise SystemExit("error: compliance expects a JSON object of run artifacts")
    agent = ComplianceAgent(
        system_name=args.system,
        data_categories=args.data_categories,
        retention_policy_days=args.retention_days,
    )
    pack = agent.build_evidence_pack(data)
    pack["fingerprint"] = agent.fingerprint(pack)
    if args.retention_dates:
        pack["retention"] = agent.retention_check(args.retention_dates)
        if pack["retention"]["status"] == "VIOLATION":
            pack["verdict"] = "INCOMPLETE"
    _emit(pack)
    if pack["verdict"] != "COMPLETE":
        print(
            "GATE FAILED: missing evidence for " + ", ".join(pack["missing_controls"]),
            file=sys.stderr,
        )
        return 1
    return 0


# ---------------------------------------------------------------------- main
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="auto-eval", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    g = sub.add_parser("golden-set", help="create/refresh/verify/curate a golden set")
    g.add_argument("action", choices=["create", "refresh", "verify", "curate"])
    g.add_argument("--input", default="-", help="JSON file, or - for stdin")
    g.add_argument("--storage", default="data/golden_sets")
    g.add_argument("--label-key", default="label")
    g.add_argument("--n-samples", type=int, default=300)
    g.add_argument("--target-kappa", type=float, default=0.8)
    g.add_argument("--seed", type=int, default=0)
    g.add_argument("--version", default="1.0")
    g.add_argument("--alert-threshold", type=float, default=0.10)
    g.add_argument("--attack-vectors", nargs="*", default=None)
    g.add_argument("--per-vector", type=int, default=25)
    g.set_defaults(func=cmd_golden_set)

    d = sub.add_parser("drift", help="distribution, label-mix or latency drift check")
    d.add_argument("mode", choices=["numeric", "labels", "latency"])
    d.add_argument("--input", default="-", help="JSON file, or - for stdin")
    d.add_argument("--psi-alert", type=float, default=0.10)
    d.add_argument("--kl-alert", type=float, default=0.10)
    d.add_argument("--slo-p95-ms", type=float, default=500.0)
    d.add_argument("--seed", type=int, default=0)
    d.set_defaults(func=cmd_drift)

    r = sub.add_parser("red-team", help="run the red-team probe suite")
    r.add_argument("--report", action="store_true", help="human-readable report")
    r.add_argument(
        "--fail-on-critical",
        action="store_true",
        help="exit 1 if any critical-severity probe is exploited",
    )
    r.set_defaults(func=cmd_red_team)

    s = sub.add_parser("gate", help="synthesize a release decision from metric gates")
    s.add_argument("--input", default="-", help="JSON file, or - for stdin")
    s.set_defaults(func=cmd_gate)

    c = sub.add_parser("compliance", help="build a compliance evidence pack")
    c.add_argument("--input", default="-", help="JSON file, or - for stdin")
    c.add_argument(
        "--retention-dates",
        nargs="*",
        default=None,
        help="ISO dates of runs, to check the retention window",
    )
    c.add_argument("--system", required=True, help="system under evaluation")
    c.add_argument("--data-categories", nargs="*", default=[])
    c.add_argument("--retention-days", type=int, default=None)
    c.set_defaults(func=cmd_compliance)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
