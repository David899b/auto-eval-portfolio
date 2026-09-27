![Python](https://img.shields.io/badge/Python-3.11+-blue?logo=python)
![License](https://img.shields.io/badge/License-MIT-green)
![Status](https://img.shields.io/badge/Status-Project%20Showcase-262730)
![Agents](https://img.shields.io/badge/Agents-5-orange)
![API](https://img.shields.io/badge/API-FastAPI-009688?logo=fastapi)
![Dashboard](https://img.shields.io/badge/Dashboard-Streamlit-FF4B4B?logo=streamlit)

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

## The Concept: Agent-Native Evaluation Platform

`auto-eval-platform` is an architecture sketch for **5 agents** that run continuously:

| Agent | Purpose | Frequency |
|-------|--------|-----------|
| **Golden Set Agent** | Generate/curate/version/refresh golden sets (kappa ≥ 0.8, SHA-256 freeze) | On-demand + drift-triggered |
| **Red Team Agent** | Prompt injection, jailbreak, PII leakage, hallucination, schema violation | Nightly |
| **Gate Synthesis Agent** | Bootstrap CI → 3-scenario gates (pessimistic/base/optimistic) + cost-sensitive thresholds | On every model change |
| **Drift Agent** | PSI/KL on embeddings → alerts + retraining triggers | Hourly / daily |
| **Compliance Agent** | Regulations/DPAs → executable test suites + audit trails | On regulation change |

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

```bash
# 1. Create new eval project (2 minutes)
cookiecutter gh:David899b/auto-eval-platform --directory templates/cookiecutter-eval-project

# 2. Configure
cd my-eval-project
cp configs/config.yaml.example configs/config.yaml
# Edit: model endpoint, golden set path, gates, compliance regs

# 3. Run evaluation
python -m auto_eval.cli evaluate --split test

# 4. Dashboard (demo app)
streamlit run dashboard/app.py
```

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      auto-eval-platform                      │
├─────────────────────────────────────────────────────────────┤
│  API (FastAPI)          │  Agents (Background Workers)      │
│  ├── /evaluate          │  ├── GoldenSetAgent               │
│  ├── /runs/{id}         │  ├── RedTeamAgent (nightly cron)  │
│  ├── /projects/{id}     │  ├── GateSynthesisAgent           │
│  └── /dashboard         │  ├── DriftAgent (hourly cron)     │
│                         │  └── ComplianceAgent              │
├─────────────────────────────────────────────────────────────┤
│  Golden Sets (Versioned, Frozen, SHA-256, Kappa ≥ 0.8)      │
│  Harness (Instructor + Judge + Bootstrap CI + Gates)        │
│  CI/CD (GitHub Actions / GitLab CI)                         │
│  Dashboard (Streamlit) + Alerts (Slack/Email/PagerDuty)     │
└─────────────────────────────────────────────────────────────┘
```

---

## Gates That Mean Something

| Metric | Pessimistic | Base | Optimistic | Gate Type |
|--------|-------------|------|------------|-----------|
| Accuracy | 0.92 | 0.95 | 0.97 | MIN |
| F1 | 0.89 | 0.93 | 0.96 | MIN |
| Derivation Rate | 0.12 | 0.08 | 0.05 | MAX |
| Latency P95 (ms) | 1800 | 1200 | 800 | MAX |
| Cost per 1k tokens | $0.045 | $0.032 | $0.025 | MAX |

*Pessimistic must pass for regulated domains.*

---

## Compliance Concept

| Regulation | Test Cases | Audit Trail |
|------------|------------|-------------|
| Ley 25.326 (Argentina) | 12 | ✅ |
| GDPR (EU) | 18 | ✅ |
| EU AI Act (prep) | 22 | ✅ |
| Custom DPA/SCC | auto-generated | ✅ |

Compliance pipeline: regulation → atomic obligations → executable pytest suite → SHA-256 audit trail.

---

## Deploy Options

| Target | Command |
|--------|---------|
| Kubernetes (Helm) | `helm install auto-eval-platform ./helm` |
| Docker Compose | `docker compose up -d` |
| Serverless (AWS Lambda/GCP Cloud Run) | `serverless deploy` |
| VM / Bare Metal | `docker compose -f docker-compose.prod.yml up -d` |

---

## License

MIT — Use it, extend it, learn from it.

---

**Ready to stop hand-crafting eval?** → `cookiecutter gh:David899b/auto-eval-platform`