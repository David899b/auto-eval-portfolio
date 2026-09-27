"""A stdlib-only HTTP API over the evaluation harness.

This replaces an earlier FastAPI + Pydantic module whose handlers were
``pass``/empty-dict stubs that answered ``{"status": "completed"}`` without
running anything. A stub that reports success is worse than a 501, so this
version either does the work or says it could not.

Deliberately built on :mod:`http.server` rather than FastAPI: the whole point of
this package is that a reviewer can clone it and run it with a bare interpreter.
Pulling in a web framework to serve three read-only endpoints would break that
for no benefit. A real deployment would put this behind a real ASGI server; that
is a packaging decision, not a design one.

Routes:
    GET  /health
    POST /golden-set     build + freeze a golden set, report measured kappa
    GET  /runs           list recorded gate decisions
    POST /runs/synthesize  turn metric gates into one release decision
"""
from __future__ import annotations

import json
import tempfile
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from ai_eval.stats import Scenario  # noqa: F401  (documented in the payload schema)
from auto_eval.agents.gate_synthesis_agent import GateSynthesisAgent, MetricGate
from auto_eval.agents.golden_set_agent import GoldenSetAgent

MAX_BODY_BYTES = 8 * 1024 * 1024  # a golden set is KB-scale; 8MB is already generous


class _Handler(BaseHTTPRequestHandler):
    server_version = "auto-eval/0.2"
    # Injected by make_server
    agent: GateSynthesisAgent
    storage: Path

    def log_message(self, fmt: str, *args: Any) -> None:  # noqa: A003
        """Quieter than the default, which logs every request to stderr."""

    # ------------------------------------------------------------------ helpers
    def _send(self, code: int, payload: dict | list) -> None:
        body = json.dumps(payload, indent=2, default=str).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _read_json(self) -> Any:
        length = int(self.headers.get("Content-Length") or 0)
        if length <= 0:
            raise ValueError("request body is empty; expected JSON")
        if length > MAX_BODY_BYTES:
            raise ValueError(f"request body exceeds {MAX_BODY_BYTES} bytes")
        return json.loads(self.rfile.read(length).decode("utf-8"))

    # -------------------------------------------------------------------- routes
    def do_GET(self) -> None:  # noqa: N802
        if self.path == "/health":
            self._send(200, {"status": "ok", "service": "auto-eval"})
            return
        if self.path == "/runs":
            self._send(200, {"runs": self.agent.history, "n_runs": len(self.agent.history)})
            return
        self._send(404, {"error": f"no route for GET {self.path}"})

    def do_POST(self) -> None:  # noqa: N802
        try:
            payload = self._read_json()
        except ValueError as exc:
            self._send(400, {"error": str(exc)})
            return

        if self.path == "/golden-set":
            self._build_golden_set(payload)
            return
        if self.path == "/runs/synthesize":
            self._synthesize(payload)
            return
        self._send(404, {"error": f"no route for POST {self.path}"})

    def _build_golden_set(self, payload: Any) -> None:
        if not isinstance(payload, dict) or "items" not in payload:
            self._send(400, {
                "error": 'expected {"items": [...], "labeler_a": [...], "labeler_b": [...]}'
            })
            return
        items = payload["items"]
        if not isinstance(items, list) or not items:
            self._send(400, {"error": "items must be a non-empty list"})
            return

        agent = GoldenSetAgent(
            storage_path=self.storage,
            target_kappa=float(payload.get("target_kappa", 0.8)),
            seed=int(payload.get("seed", 0)),
        )
        try:
            record = agent.generate_initial_set(
                source_data=items,
                label_key=payload.get("label_key", "label"),
                n_samples=int(payload.get("n_samples", min(300, len(items)))),
                labeler_a=payload.get("labeler_a"),
                labeler_b=payload.get("labeler_b"),
            )
        except (ValueError, KeyError, TypeError) as exc:
            self._send(422, {"error": f"{type(exc).__name__}: {exc}"})
            return

        status = 201 if record["kappa_status"] != "below_target" else 200
        self._send(status, record)

    def _synthesize(self, payload: Any) -> None:
        if not isinstance(payload, list) or not payload:
            self._send(400, {"error": "expected a non-empty list of metric gates"})
            return
        try:
            gates = [
                MetricGate(
                    name=g["name"],
                    values=[float(v) for v in g["values"]],
                    limit=float(g["limit"]),
                    direction=g.get("direction", "min"),
                    margin=float(g.get("margin", 0.0)),
                    unit=g.get("unit", ""),
                )
                for g in payload
            ]
        except (KeyError, TypeError, ValueError) as exc:
            self._send(422, {"error": f"malformed gate: {exc}"})
            return

        try:
            decision = self.agent.synthesize(gates)
        except ValueError as exc:
            self._send(422, {"error": str(exc)})
            return
        # A BLOCK is a successful evaluation of the request, not an HTTP error.
        self._send(200, decision)


def make_server(host: str = "127.0.0.1", port: int = 8000, storage: Path | None = None) -> ThreadingHTTPServer:
    """Build the server. Defaults to loopback: this has no auth by design."""
    handler = type(
        "BoundHandler",
        (_Handler,),
        {
            "agent": GateSynthesisAgent(),
            "storage": storage or Path(tempfile.gettempdir()) / "auto_eval_api",
        },
    )
    return ThreadingHTTPServer((host, port), handler)


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Run the auto-eval API")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    server = make_server(args.host, args.port)
    print(f"auto-eval API on http://{args.host}:{args.port} (no auth; loopback only)")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
