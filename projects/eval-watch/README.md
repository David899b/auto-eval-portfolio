# eval-watch — Drift detection + 3-scenario quality gates

Continuously compares a metric's distribution (latency, accuracy, cost) against a
frozen baseline, raises a **drift alert** via PSI/KL, and decides on releases with
a **3-scenario bootstrap gate** (pessimistic / base / optimistic).

## Why this matters
Single-number thresholds lie. A release can pass "avg latency ok" while the p95
degrades under load. PSI catches the *distribution shift* your mean hides; the
3-scenario gate forces you to ship only when the pessimistic case also holds —
the discipline that regulated domains require.

## Method
- **PSI**: discrete distribution distance over binned scores — 0 = identical,
  >0.1 conventional alert zone.
- **KL divergence**: `KL(actual || expected)`, sensitive to tail shifts.
- **Bootstrap CI** on the metric → pessim./base/optim. → gate `PASS | CONDITIONAL | BLOCK`.

## Run
```bash
python eval_watch/eval_watch.py
```
Offline demo: baseline vs. a drifted distribution, prints PSI/KL + gate decision.

## Where it fits
Slot this behind your model endpoint or harness runner, log each run to the
baseline store, alert on PSI > 0.1, block merges when the gate says `BLOCK`.

## Read more
- PSI in credit-risk monitoring (standard >0.1 alert)
- KL divergence for ML drift (Evidently's approach)