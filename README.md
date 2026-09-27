![Python](https://img.shields.io/badge/Python-3.11+-blue?logo=python)
![License](https://img.shields.io/badge/License-MIT-green)
![Status](https://img.shields.io/badge/Status-Project%20Showcase-262730)
![Agents](https://img.shields.io/badge/Agents-5-orange)
![Deps](https://img.shields.io/badge/dependencies-0-success)
![Tests](https://img.shields.io/badge/tests-207-success)

# auto-eval-platform

**Self-serve evaluation infrastructure for LLM teams. Agents do the work. You ship.**

> Project showcase. Demo/orientation material — not a maintained product.

---

## The Problem

You're shipping LLM features. Evaluation is a bottleneck:
- Golden sets rot (no versioning, no kappa, no refresh)
- Red-teaming is manual, one-off, not continuous
- Gates are "vibes" — no statistical rigor, no pessimistic scenario
- Drift detected by customers, not by you
- Compliance is a spreadsheet, not executable tests
- Every new use case = rebuild eval from scratch

---

## What is actually built

Five agents, all implemented, all covered by tests. Each one measures what it
reports — there are no stubs returning plausible dicts.

| Surface | What it is |
|---|---|
| `src/ai_eval/stats.py` | kappa, Wilson, PSI, KL, JSD, TVD, seeded bootstrap CI |
| `src/auto_eval/agents/` | the 5 agents below |
| `src/ai_eval/harness.py` | evaluation harness: per-item bootstrap, no pandas/numpy |
| `src/auto_eval/cli/` | `auto-eval golden-set / drift / red-team / gate / compliance` |
| `src/auto_eval/api/` | stdlib `http.server` API over the same agents |
| `projects/` | 3 standalone demos, each with its own tests |

**Zero runtime dependencies.** Clone it and run the whole suite with a bare
Python 3.11+. There is no install step, because there is nothing to install.

```bash
python3 scripts/run_tests.py     # 9 suites, 207 assertions
```

The agents:

| Agent | What it actually computes | Intended frequency |
|-------|--------|-----------|
| **Golden Set Agent** | Generate/curate/version/refresh golden sets (kappa ≥ 0.8, SHA-256 freeze) | On-demand + drift-triggered |
| **Red Team Agent** | Prompt injection, jailbreak, PII leakage, hallucination, schema violation | Nightly |
| **Gate Synthesis Agent** | Bootstrap CI → 3-scenario gates (pessimistic/base/optimistic) + cost-sensitive thresholds | On every model change |
| **Drift Agent** | PSI/KL on embeddings → alerts + retraining triggers | Hourly / daily |
| **Compliance Agent** | Evidence pack + SHA-256 fingerprint + retention check | On audit demand |

> The "intended frequency" column describes the target deployment. What runs
> today is the CLI and the API; there is no scheduler in this repo.

---

## Terminology

- **EDI (Especialista en Datos e IA):** role model where the evaluator delivers *model + evidence + frozen test*; domain experts validate against ground truth.
- **3-scenario reporting:** every metric reported as pessimistic / base / optimistic with bootstrap confidence intervals — never a single number.
- **Frozen golden set:** immutable, versioned (SHA-256), kappa ≥ 0.8, drift-triggered refresh.

---

## Demo Projects

Small, self-contained, zero-dependency experiments that explore the ideas behind
the platform — each runnable offline. Each project is documented in
English (`README.md`) and Spanish (`README.es.md`):

| Project | Question it explores | Run |
|---------|---------------------|-----|
| [`projects/arena`](projects/arena) | How do different models rank head-to-head (Bradley–Terry + bootstrap CI)? | `python projects/arena/arena/demo.py` |
| [`projects/redteam-dx`](projects/redteam-dx) | Can red-teaming be declarative, repeatable and CI-gated (garak/PyRIT-style)? | `python projects/redteam-dx/redteam_dx/redteam_dx.py` |
| [`projects/eval-watch`](projects/eval-watch) | Does PSI/KL drift alert before the customer does, with 3-scenario gates? | `python projects/eval-watch/eval_watch/eval_watch.py` |

Each project has its own `README.md` (method + why it matters) and `tests/`.

---

## Quick Start

No install step. Clone and run:

```bash
export PYTHONPATH=src

# Full suite: 9 files, 207 assertions
python3 scripts/run_tests.py

# Build a golden set from a JSON corpus, measuring inter-rater kappa
python3 -m auto_eval.cli golden-set create --input corpus.json --storage data/gs

# Gate a release: exits non-zero if any metric's worst scenario breaches
echo '[{"name":"f1","values":[0.91,0.88,0.93],"limit":0.85,"direction":"min"}]' > g.json
python3 -m auto_eval.cli gate --input g.json | jq .decision

# Produce a real report and look at it
python3 scripts/example_report.py --out report.json
python3 -m dashboard/app --report report.json      # http://127.0.0.1:8080
```

Subcommands: `golden-set {create,refresh,verify,curate}`, `drift {numeric,labels,latency}`,
`red-team`, `gate`, `compliance`. Every one reads JSON and writes JSON to stdout;
human-readable summaries go to stderr so `| jq` works.

---

## Architecture

```
┌───────────────────────────────────────────────────────────────┐
│  auto-eval-platform                        stdlib only        │
├───────────────────────────────────────────────────────────────┤
│  CLI (argparse)              │  API (http.server)            │
│  golden-set / drift          │  POST /golden-set             │
│  red-team / gate             │  POST /runs/synthesize        │
│  compliance                  │  GET  /health, /runs          │
├───────────────────────────────────────────────────────────────┤
│  Agents: GoldenSet · Drift · RedTeam · GateSynthesis ·       │
│          Compliance                                           │
│  stats.py: kappa, Wilson, PSI, KL, JSD, TVD, bootstrap CI     │
│  harness.py: per-item bootstrap, 3-scenario metrics, gates    │
├───────────────────────────────────────────────────────────────┤
│  Golden sets: versioned, SHA-256 frozen, kappa measured      │
│  Dashboard: stdlib report viewer over real harness output    │
│  CI: 8 suites on Python 3.11 / 3.12 / 3.13                    │
└───────────────────────────────────────────────────────────────┘
```

**Not in this repo**, despite what earlier revisions of this file claimed: no
FastAPI, no Streamlit, no Docker, no Helm chart, no scheduler, no Slack/PagerDuty
alerts, no database. Those were aspirations in a design doc. What is here is
what the tests cover.

---

## Gates That Mean Something

Every metric is reported as three scenarios and the gate reads the **worst** one:

```bash
$ PYTHONPATH=src python3 -m auto_eval.cli gate --input gates.json
{
  "decision": "BLOCK",
  "blocking_metrics": ["p95_latency_ms"],
  "rationale": "blocked by p95_latency_ms: the pessimistic scenario breaches its limit"
}
$ echo $?
1
```

Two rules that are easy to get backwards, and that the tests pin:

- For accuracy and F1, higher is better, so the pessimistic case is the
  **bottom** of the interval.
- For latency, cost and derivation rate, higher is worse, so the pessimistic
  case is the **top**.

`bootstrap_ci` is direction-aware for exactly this reason. When it was not, the
latency gate compared the *optimistic* end against the limit and passed
releases whose tail had regressed — see
`test_gate_blocks_a_latency_regression_despite_a_healthy_average`.

---

## Compliance Concept

An earlier version of this file listed per-regulation test-case counts with a
green tick next to each. None of that was implemented and all of it was
invented. The numbers are gone because there was nothing behind them.

What the ComplianceAgent actually does:

| Capability | Implemented | Where |
|---|---|---|
| Evidence pack: 6 controls, PRESENT/MISSING per control | yes | `build_evidence_pack` |
| SHA-256 fingerprint of the pack, so evidence is tamper-evident | yes | `fingerprint` |
| Retention-window check over a list of run dates | yes | `retention_check` |
| Regulation → atomic obligations → executable tests | **no** | not built |
| Per-regulation test-case counts (Ley 25.326 / GDPR / EU AI Act) | **no** | never existed |

The control list is fixed: `frozen_baseline`, `red_team_evidence`,
`drift_monitoring`, `release_gate`, `data_minimization`, `human_review`. You
supply the artifacts; the agent reports which controls have evidence attached
and which are missing. A missing control is the finding.

Scope note carried in the code: this generates *evidence artifacts*. It is not a
certification and does not replace a legal or data-protection review.

---

## Deploy Options

There is no Dockerfile, no `docker-compose.yml`, no Helm chart and no
serverless config in this repo. The commands an earlier revision listed here
would all have failed.

What you can actually run today, with a bare Python and nothing installed:

| Target | Command |
|--------|---------|
| Local CLI | `PYTHONPATH=src python3 -m auto_eval.cli --help` |
| Local API | `PYTHONPATH=src python3 -m auto_eval.api.main --port 8000` |
| Report viewer | `PYTHONPATH=src python3 -m dashboard.app --report report.json` |
| CI | push to `main`; the workflow runs the 8 suites on 3 Python versions |

The API binds to `127.0.0.1` and has **no authentication** — it is a local
evaluation tool, not something to expose.

---

## License

MIT — Use it, extend it, learn from it.

---

**Ready to stop hand-crafting eval?** → `cookiecutter gh:David899b/auto-eval-portfolio`