# redteam-dx — Declarative adversarial scanning for LLM apps

Automated red-teaming over a **probe suite** (prompt injection, jailbreak,
PII leakage, hallucination), graded with a **MITRE ATLAS-inspired severity model**
(critical/high/medium/low) and per-finding evidence lines.

## Why this matters
Red-teaming is usually a one-off burst. This design makes it **declarative and
repeatable**: add a probe as data, run it on every model release, and let a CI job
block the merge if `critical > 0`. Same idea driving garak, PyRIT and promptfoo.

## Method
- **Probes are data** (`ProbeSuites`): id, vector, payload, baseline severity.
- **Target is a callable**: `complete(prompt) -> str`. Swap the stub for a real
  client (Ollama local, OpenAI, Anthropic) — nothing else changes.
- **Detector heuristics:** refusal markers, empty/terse replies, content surfaced.
  In a real deployment you'd pair this with an LLM judge for aggressive scenarios.
- **Report:** `findings` per probe with evidence + aggregate `by_severity`.

## Run
```bash
python redteam_dx/redteam_dx.py
```
Prints a scan against the offline stub. Point `RedTeamDX.target` at your model to
scan the real thing.

## Read more
- garak (NVIDIA LLM vulnerability scanner) — probe/plug architecture
- PyRIT (Microsoft) — orchestrators, converters, scoring
- promptfoo — declarative YAML red team config + CI integration
- MITRE ATLAS — AI adversary tactics & techniques