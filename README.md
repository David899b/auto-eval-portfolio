# auto-eval-platform

**Self-serve evaluation infrastructure for LLM teams. Agents do the work. You ship.**

---

## 🎯 The Problem

You're shipping LLM features. Evaluation is a bottleneck:
- Golden sets rot (no versioning, no kappa, no refresh)
- Red-teaming is manual, one-off, not continuous
- Gates are "vibes" — no statistical rigor, no pessimistic scenario
- Drift detected by customers, not by you
- Compliance is a spreadsheet, not executable tests
- Every new use case = rebuild eval from scratch

---

## 🤖 The Solution: Agent-Native Evaluation Platform

`auto-eval-platform` deploys **5 agents** that run continuously:

| Agent | What It Does | Frequency |
|-------|-------------|-----------|
| 🎯 **Golden Set Agent** | Genera, curra, versiona, refresca golden sets (kappa ≥ 0.8, SHA-256 freeze) | On-demand + drift-triggered |
| 🔴 **Red Team Agent** | Prompt injection, jailbreak, PII leakage, hallucination, schema violation | Nightly |
| 🎯 **Gate Synthesis Agent** | Bootstrap CI → 3-scenario gates (pessimistic/base/optimistic) + cost-sensitive thresholds | On every model change |
| 📈 **Drift Agent** | PSI/KL en embeddings → alertas + auto-retraining triggers | Hourly / daily |
| ⚖️ **Compliance Agent** | Ingesta DPAs/regulaciones → test suites ejecutables + audit trails | On regulation change |

**Deploy once. Teams self-serve via API. You never touch eval again.**

---

## 🚀 Quick Start

```bash
# 1. Create new eval project (2 minutes)
cookiecutter gh:David899b/auto-eval-platform --directory templates/cookiecutter-eval-project

# 2. Configure
cd my-eval-project
cp configs/config.yaml.example configs/config.yaml
# Edit: model endpoint, golden set path, gates, compliance regs

# 3. Run evaluation
python -m auto_eval.cli evaluate --split test

# 4. Dashboard
streamlit run dashboard/app.py
```

---

## 🏗️ Architecture

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
│  Dashboard (Streamlit) + Alertas (Slack/Email/PagerDuty)    │
└─────────────────────────────────────────────────────────────┘
```

---

## 🎯 Gates That Mean Something

| Metric | Pessimistic | Base | Optimistic | Gate Type |
|--------|-------------|------|------------|-----------|
| Classification Accuracy | 0.92 | 0.95 | 0.97 | MIN |
| Extraction F1 | 0.89 | 0.93 | 0.96 | MIN |
| Derivation Rate | 0.12 | 0.08 | 0.05 | MAX |
| Latency P95 (ms) | 1800 | 1200 | 800 | MAX |
| Cost per 1k tokens | $0.045 | $0.032 | $0.025 | MAX |

**Pessimistic must pass for regulated domains.** No "it works on my machine."

---

## ⚖️ Compliance Built-In

| Regulation | Test Cases | Audit Trail |
|------------|------------|-------------|
| Ley 25.326 (Argentina) | 12 | ✅ |
| GDPR (EU) | 18 | ✅ |
| EU AI Act (Prep) | 22 | ✅ |
| Custom DPA/SCC | Auto-generated | ✅ |

**Compliance Agent** ingiere tu DPA/regulación → genera test suite pytest ejecutable + audit trail inmutable.

---

## 📦 Deploy Options

| Target | Command |
|--------|---------|
| Kubernetes (Helm) | `helm install auto-eval-platform ./helm` |
| Docker Compose | `docker compose up -d` |
| Serverless (AWS Lambda/GCP Cloud Run) | `serverless deploy` |
| VM / Bare Metal | `docker compose -f docker-compose.prod.yml up -d` |

---

## 💰 Business Model

| Track | Price | What You Get |
|-------|-------|--------------|
| **Track A — Fractional EDI** | $12k/mo (20h/wk) | Yo hago la evaluación: golden sets, harness, gates, dashboard |
| **Track B — Platform Deploy** | $50k (6-8 wks) + $8k/mo | Plataforma deployada en tu infra, agents corriendo, handoff a 1 persona |

**Zero meetings by design.** Entregas via PR + dashboard + runbook. Sync = 0-1/mes.

---

## 📊 Proof

| Project | Metrics | Compliance |
|---------|---------|------------|
| **Production Document-AI** | Field F1 0.971, ANLS 0.984, Schema OK 100% | Policy verified, DPA aligned, Ley 25.326 |
| **Public-Utility MVP** | 4-week build, kappa≥0.8, self-hosted Qwen | Ley 25.326 ready |

---

## 🛠️ Stack

`Python 3.11+` • `FastAPI` • `Instructor/Pydantic` • `vLLM/Ollama` • `MLflow` • `Evidently` • `pytest` • `GitHub Actions` • `Docker` • `Kubernetes` • `Streamlit` • `Plotly` • `Cookiecutter`

---

## 📄 License

MIT — Use it, extend it, sell it. Built by [David Bautista](https://github.com/David899b) (Principal AI Engineer, EDI).

---

**Ready to stop hand-crafting eval?** → `cookiecutter gh:David899b/auto-eval-platform`
