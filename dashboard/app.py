"""A zero-dependency report viewer for evaluation runs.

The previous version of this file was a Streamlit app whose numbers were
hardcoded in the source — every metric already reading "PASS" — behind four
third-party imports. A dashboard that displays invented results is worse than no
dashboard: it looks like evidence.

This version reads a real report produced by :mod:`ai_eval.harness` and renders
it with :mod:`http.server`. Same zero-dependency rule as the rest of the repo.

Generate a report to look at:

    PYTHONPATH=src python -m auto_eval.api.main            # terminal 1
    PYTHONPATH=src python scripts/example_report.py         # writes report.json
    PYTHONPATH=src python -m dashboard.app --port 8080     # terminal 2

Or point it at any harness report:

    python -m dashboard.app --report path/to/report.json
"""
from __future__ import annotations

import argparse
import html
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

# Which way is "bad" for each metric. This is the same distinction the gate uses,
# and getting it wrong is how a latency regression gets rendered in green.
HIGHER_IS_WORSE = {"latency_p95_ms", "derivation_rate", "cost_per_1k_tokens", "cost_usd"}

DECISION_STYLE = {
    "PASS": ("#0f9960", "✅"),
    "SHIP": ("#0f9960", "✅"),
    "CONDITIONAL": ("#c47f00", "⚠️"),
    "BLOCK": ("#d1242f", "⛔"),
}


def _worst_first(metric: str, scenarios: dict[str, Any]) -> tuple[float, float, float]:
    """Return (pessimistic, base, optimistic) for display.

    This is a passthrough, and that is deliberate. The PESSIMISTIC key already
    holds the worst case for every metric, because ``bootstrap_ci`` is
    direction-aware: for accuracy it is the bottom of the interval, for latency
    and cost the top. Re-ordering by direction *here* would be a second,
    contradictory implementation of the same rule, and the two would eventually
    disagree. The invariant to rely on is the one the producer guarantees.
    """
    p = float(scenarios.get("PESSIMISTIC", scenarios.get("pessimistic", 0.0)))
    b = float(scenarios.get("BASE", scenarios.get("base", 0.0)))
    o = float(scenarios.get("OPTIMISTIC", scenarios.get("optimistic", 0.0)))
    return p, b, o


def _bar(frac: float, colour: str) -> str:
    frac = max(0.0, min(1.0, frac))
    return (
        f'<div class="bar"><div class="fill" style="width:{frac * 100:.1f}%;'
        f'background:{colour}"></div></div>'
    )


def _fmt(value: float) -> str:
    if value != value:  # NaN
        return "n/a"
    if abs(value) >= 100:
        return f"{value:,.0f}"
    if abs(value) >= 1:
        return f"{value:.3f}"
    return f"{value:.4f}"


def render_report(report: dict) -> str:
    """Render a harness report as a standalone HTML page."""
    decision = str(report.get("decision", "UNKNOWN")).upper()
    colour, icon = DECISION_STYLE.get(decision, ("#8b949e", "•"))
    metrics = report.get("metrics", {}) or {}

    # Provenance first: whose model, which frozen baseline, how many items.
    provenance = [
        ("model", report.get("model_id")),
        ("golden set", report.get("golden_set_version")),
        ("items", report.get("total_items")),
        ("evaluated", report.get("evaluated_at")),
    ]
    prov_html = "".join(
        f'<div><span class="k">{html.escape(k)}</span>'
        f'<span class="v">{html.escape(str(v))}</span></div>'
        for k, v in provenance
        if v not in (None, "")
    )

    rows: list[str] = []
    for name, scenarios in sorted(metrics.items()):
        p, b, o = _worst_first(name, scenarios)
        # Scale bars against the worst scenario in the row so relative
        # differences are visible even when the absolute scale differs.
        peak = max(abs(p), abs(b), abs(o)) or 1.0
        gate = report.get("gates", {}).get(name)
        gate_html = ""
        if gate:
            gcolour, gicon = DECISION_STYLE.get(str(gate).upper(), ("#8b949e", "•"))
            gate_html = f'<span class="gate" style="color:{gcolour}">{gicon} {html.escape(str(gate))}</span>'
        rows.append(
            f"""<tr>
  <td class="metric">{html.escape(name)}</td>
  <td class="num">{_fmt(p)}</td>
  <td>{_bar(abs(p) / peak, colour)}</td>
  <td class="num strong">{_fmt(b)}</td>
  <td class="num">{_fmt(o)}</td>
  <td>{gate_html}</td>
</tr>"""
        )

    if not rows:
        rows.append(
            '<tr><td colspan="6" class="empty">This report contains no metrics. '
            "Point <code>--report</code> at a harness report.</td></tr>"
        )

    meta = "".join(
        f"<div><span class='k'>{html.escape(str(k))}</span>"
        f"<span class='v'>{html.escape(str(v))}</span></div>"
        for k, v in (report.get("metadata", {}) or {}).items()
    )

    return f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>auto-eval report</title>
<style>
:root {{ --bg:#0b0f14; --fg:#e6edf3; --muted:#8b949e; --card:#111820;
         --border:#21262d; --acc:#00d4aa; }}
* {{ box-sizing:border-box; margin:0; padding:0 }}
body {{ background:var(--bg); color:var(--fg); font:15px/1.6 system-ui,-apple-system,"Segoe UI",sans-serif; padding:2.5rem 1.5rem }}
.wrap {{ max-width:940px; margin:0 auto }}
h1 {{ font-size:1.5rem; margin-bottom:.25rem }}
.sub {{ color:var(--muted); font-size:.9rem; margin-bottom:2rem }}
.verdict {{ display:inline-flex; align-items:center; gap:.6rem; padding:.6rem 1.1rem;
  border:1px solid {colour}; border-radius:8px; color:{colour}; font-weight:600;
  font-size:1.05rem; margin-bottom:1.5rem }}
table {{ width:100%; border-collapse:collapse; background:var(--card);
  border:1px solid var(--border); border-radius:8px; overflow:hidden }}
th,td {{ padding:.7rem .9rem; text-align:left; border-bottom:1px solid var(--border) }}
th {{ color:var(--muted); font-weight:500; font-size:.8rem; text-transform:uppercase;
  letter-spacing:.04em }}
tr:last-child td {{ border-bottom:none }}
.num {{ font-variant-numeric:tabular-nums; color:var(--muted); width:110px }}
.strong {{ color:var(--fg); font-weight:600 }}
.metric {{ font-family:ui-monospace,SFMono-Regular,Menlo,monospace; font-size:.85rem }}
.bar {{ background:#21262d; border-radius:3px; height:7px; min-width:90px }}
.fill {{ height:100%; border-radius:3px }}
.gate {{ font-weight:600; font-size:.85rem }}
.empty {{ color:var(--muted); text-align:center; padding:2.5rem }}
.meta {{ margin-top:2rem; display:grid; grid-template-columns:repeat(auto-fit,minmax(190px,1fr)); gap:.8rem }}
.meta>div {{ background:var(--card); border:1px solid var(--border); border-radius:6px; padding:.7rem .9rem }}
.k {{ display:block; color:var(--muted); font-size:.72rem; text-transform:uppercase; letter-spacing:.05em }}
.v {{ font-family:ui-monospace,Menlo,monospace; font-size:.82rem; word-break:break-all }}
.note {{ margin-top:2rem; color:var(--muted); font-size:.8rem; border-top:1px solid var(--border); padding-top:1rem }}
code {{ background:#21262d; padding:.1rem .35rem; border-radius:3px; font-size:.85em }}
</style></head><body><div class="wrap">
<h1>auto-eval report</h1>
<p class="sub">Every metric as pessimistic / base / optimistic. No single-number verdicts.</p>
{f'<div class="meta">{prov_html}</div>' if prov_html else ''}
<div class="verdict">{icon} {html.escape(decision)}</div>
<table>
<thead><tr><th>Metric</th><th>Pessimistic</th><th></th><th>Base</th>
<th>Optimistic</th><th>Gate</th></tr></thead>
<tbody>{"".join(rows)}</tbody>
</table>
{f'<div class="meta">{meta}</div>' if meta else ''}
<p class="note">Rendered by <code>dashboard/app.py</code> — stdlib only, no mock data.
Bars are scaled per row against that row's worst scenario, so compare within a
row, not across rows. For latency and cost the pessimistic case is the
<em>top</em> of the interval.</p>
</div></body></html>"""


class _Handler(BaseHTTPRequestHandler):
    report_html: str = ""

    def log_message(self, fmt: str, *args: Any) -> None:  # noqa: A003
        """Silence per-request logging."""

    def do_GET(self) -> None:  # noqa: N802
        body = self.report_html.encode()
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def main() -> int:
    parser = argparse.ArgumentParser(description="Serve an auto-eval report")
    parser.add_argument("--report", required=True, help="path to a harness report JSON")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8080)
    args = parser.parse_args()

    report = json.loads(Path(args.report).read_text(encoding="utf-8"))
    handler = type("BoundHandler", (_Handler,), {"report_html": render_report(report)})
    server = ThreadingHTTPServer((args.host, args.port), handler)
    print(f"report on http://{args.host}:{args.port}  (ctrl-c to stop)")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
