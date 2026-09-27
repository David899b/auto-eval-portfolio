"""Tests for the stdlib HTTP API.

The previous implementation was FastAPI + Pydantic stubs whose handlers
returned ``{"status": "completed"}`` without doing any work. These tests pin the
behaviour that matters: real computation over real input, honest status codes,
and no fabricated success.
"""
from __future__ import annotations

import json
import sys
import tempfile
import threading
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from auto_eval.api.main import make_server  # noqa: E402

PASSED: list[str] = []
FAILED: list[str] = []


def check(name: str, condition: bool, detail: str = "") -> None:
    if condition:
        PASSED.append(name)
        print(f"PASS {name}")
    else:
        FAILED.append(name)
        print(f"FAIL {name} {detail}")


def request(url: str, payload: object = None) -> tuple[int, object]:
    """Return (status_code, parsed_body). Never raises on a 4xx/5xx."""
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"} if data else {},
        method="POST" if data is not None else "GET",
    )
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            return resp.status, json.loads(resp.read())
    except urllib.error.HTTPError as exc:
        body = exc.read()
        try:
            return exc.code, json.loads(body)
        except json.JSONDecodeError:
            return exc.code, {"raw": body.decode("utf-8", "replace")}


def _corpus(n: int) -> list[dict]:
    return [
        {"id": i, "text": f"item {i}", "label": "pos" if i % 3 == 0 else "neg"}
        for i in range(n)
    ]


def main() -> int:
    with tempfile.TemporaryDirectory() as tmp:
        server = make_server("127.0.0.1", 0, storage=Path(tmp))
        port = server.server_address[1]
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        base = f"http://127.0.0.1:{port}"

        try:
            # ------------------------------------------------------------ health
            status, body = request(f"{base}/health")
            check("GET /health returns 200", status == 200, str(status))
            check("GET /health reports ok", body.get("status") == "ok", str(body))

            # ------------------------------------------------------- unknown route
            status, body = request(f"{base}/nope")
            check("unknown GET returns 404", status == 404, str(status))
            check("404 body names the path", "nope" in str(body.get("error", "")), str(body))

            # ------------------------------------------------------ golden set: real
            corpus = _corpus(60)
            status, record = request(
                f"{base}/golden-set",
                {"items": corpus, "n_samples": 30, "target_kappa": 0.8, "seed": 1},
            )
            check("POST /golden-set returns 201 on a fresh set", status == 201, str(status))
            check("golden set reports n_items", record["metadata"]["n_items"] == 30,
                  str(record["metadata"].get("n_items")))
            check("golden set is frozen with a sha256",
                  len(record["metadata"]["sha256"]) == 64, str(record["metadata"].get("sha256")))
            check("kappa is reported as not measured when no labelers were sent",
                  record["measured_kappa"] is None and record["kappa_status"] == "not_measured",
                  f'{record["measured_kappa"]} / {record["kappa_status"]}')

            # ------------------------------------- golden set: kappa is really measured
            a = ["pos" if i % 3 == 0 else "neg" for i in range(60)]
            b = list(a)
            for i in range(5, 20):
                b[i] = "pos" if a[i] == "neg" else "neg"
            status, record = request(
                f"{base}/golden-set",
                {"items": corpus, "labeler_a": a, "labeler_b": b, "n_samples": 30, "seed": 1},
            )
            check("POST /golden-set with labelers returns 200 below target", status == 200, str(status))
            check("kappa is measured, not the target",
                  record["measured_kappa"] is not None
                  and record["measured_kappa"] != record["target_kappa"],
                  str(record.get("measured_kappa")))
            check("below-target kappa is flagged", record["kappa_status"] == "below_target",
                  record["kappa_status"])

            # ---------------------------------------------------- bad input: 4xx not 500
            status, body = request(f"{base}/golden-set", {"nope": 1})
            check("golden-set without items returns 400", status == 400, str(status))
            check("400 body names the expected shape", "items" in str(body.get("error", "")), str(body))

            status, body = request(f"{base}/golden-set", {"items": []})
            check("golden-set with empty items returns 400", status == 400, str(status))

            status, body = request(
                f"{base}/golden-set", {"items": corpus, "labeler_a": a}  # only one labeler
            )
            check("one-sided labelers return 422", status == 422, str(status))
            check("422 explains the pairing rule", "together" in str(body.get("error", "")), str(body))

            status, body = request(
                f"{base}/golden-set", {"items": corpus, "labeler_a": [1, 2], "labeler_b": [1, 2]}
            )
            check("mismatched labeler length returns 422", status == 422, str(status))

            # --------------------------------------------------- gate synthesis: real
            status, body = request(
                f"{base}/runs/synthesize",
                [
                    {"name": "f1", "values": [0.91, 0.88, 0.93], "limit": 0.85, "direction": "min"},
                    {"name": "p95_latency_ms", "values": [430, 505, 610], "limit": 500,
                     "direction": "max"},
                ],
            )
            check("POST /runs/synthesize returns 200 for a BLOCK", status == 200, str(status))
            check("a BLOCK is 200, not 5xx", body["decision"] == "BLOCK", str(body.get("decision")))
            check("the blocking metric is named",
                  body["blocking_metrics"] == ["p95_latency_ms"], str(body.get("blocking_metrics")))

            status, body = request(
                f"{base}/runs/synthesize",
                [{"name": "f1", "values": [0.97] * 30, "limit": 0.90, "direction": "min"}],
            )
            check("a healthy gate returns 200 PASS",
                  status == 200 and body["decision"] == "PASS", f'{status} {body.get("decision")}')

            status, body = request(f"{base}/runs/synthesize", [])
            check("empty gate list returns 400", status == 400, str(status))

            status, body = request(f"{base}/runs/synthesize", [{"values": [1, 2]}])
            check("gate without a name returns 422", status == 422, str(status))

            # ------------------------------------------------- run history is real
            status, body = request(f"{base}/runs")
            check("GET /lists recorded runs", status == 200, str(status))
            check("history has the 2 synthesized runs", body["n_runs"] == 2, str(body.get("n_runs")))
            check("history records the BLOCK",
                  any(r["decision"] == "BLOCK" for r in body["runs"]), str(body.get("runs")))

            # ------------------------------------------------------ no auth by design
            status, _ = request(f"{base}/runs")
            check("no auth layer exists (documented, loopback only)", status == 200)
        finally:
            server.shutdown()
            server.server_close()

    print(f"\n{len(PASSED)}/{len(PASSED) + len(FAILED)} passed")
    if FAILED:
        print("failed: " + ", ".join(FAILED))
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
