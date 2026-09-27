# arena — Multi-model head-to-head comparison

Compare several LLMs (different providers AND local models) on a shared prompt set,
reporting an arena-style ranking with **bootstrap confidence intervals**.

## Why this matters
"Which model is better?" is a pairwise question, not a single-number question. One
F1 on one golden set can hide a two-place ranking shift. This mini-tool brings the
**Chatbot Arena / Bradley–Terry** method to a controlled, reproducible eval: you
decide the prompts, the judge, and the model set.

## Method
- **Battles:** round-robin random pairings over your prompt set, judged by a
  referee (LLM judge or deterministic rule).
- **Win rate:** ties count as half a win.
- **Bradley–Terry:** gradient ascent maximizes pairwise likelihood → strength per
  model, recentered to elo-like scale.
- **Bootstrap CI:** resample the match list with replacement 500×, recompute
  strengths, report 2.5–97.5 percentile.

## Run
```bash
python arena/demo.py
```
The demo uses deterministic offline responders so it works with zero dependencies
and prints a reproducible ranking. Replace `_mock_responder` with real clients
(OpenAI, Anthropic, local Ollama/vLLM) to run against your actual models.

## What to watch
- Wide CI ⇒ unstable ranking ⇒ you need more battles or better prompts.
- Judge bias is the top confounder: measure judge agreement (kappa) before trusting a table.

## Read more
- Chatbot Arena methodology (Bradley–Terry + Elo)
- Bootstrap confidence intervals for BT models