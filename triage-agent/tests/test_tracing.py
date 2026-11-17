"""
tests/test_tracing.py — traces POSTed to /api/v1/traces with per-agent spans.

A local HTTP server stands in for the eval platform and records every request.
The REAL compiled graph runs with a stub LLM that fires real LangChain callbacks
(100 input / 20 output / 150 total tokens per call).
"""

from __future__ import annotations

import json
import sys
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

import tracing  # noqa: E402
from tests.stub_llm import StubLLM  # noqa: E402
from tests.test_multiturn import stubbed, summary, triage  # noqa: E402


class _Platform:
    def __init__(self, status: int = 200):
        self.requests: list[dict] = []
        self.status = status
        outer = self

        class H(BaseHTTPRequestHandler):
            def do_POST(self):
                body = self.rfile.read(int(self.headers["Content-Length"]))
                outer.requests.append({"path": self.path, "headers": dict(self.headers), "json": json.loads(body)})
                self.send_response(outer.status)
                self.end_headers()
                self.wfile.write(b"{}")

            def log_message(self, *a):
                pass

        self.server = HTTPServer(("127.0.0.1", 0), H)
        self.url = f"http://127.0.0.1:{self.server.server_address[1]}"
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def close(self):
        self.server.shutdown()


@pytest.fixture
def platform(monkeypatch):
    p = _Platform()
    monkeypatch.setenv("TRACE_API_URL", p.url)
    monkeypatch.setenv("TRACE_API_KEY", "test-key-123")
    monkeypatch.delenv("TRACE_API_KEY_HEADER", raising=False)
    tracing.reset_tracer()
    yield p
    tracing.reset_tracer()
    p.close()


def _three_turns(llm_model="gemini-3.1-pro-preview"):
    from graph import create_graph, run_turn
    llm = StubLLM(structured={"PatientSummary": [summary(0.3), summary(0.9)], "TriageResult": [triage(conf=0.9)]},
                  questions=["How many breaths per minute?"], model=llm_model)
    app, tid = create_graph(), str(uuid.uuid4())
    with stubbed(llm):
        r1 = run_turn(app, tid, "3 year old with cough")
        r2 = run_turn(app, tid, "36 breaths, drinking well")
    assert tracing.get_tracer().flush(5)
    return tid, r1, r2


def test_one_trace_per_turn_posted_to_api_v1_traces_with_bearer_key(platform):
    tid, r1, r2 = _three_turns()
    assert len(platform.requests) == 2
    for req in platform.requests:
        assert req["path"] == "/api/v1/traces"
        h = {k.lower(): v for k, v in req["headers"].items()}
        assert h["authorization"] == "Bearer test-key-123"
        assert h["content-type"] == "application/json"
    t1, t2 = (r["json"] for r in platform.requests)
    assert t1["session_id"] == t2["session_id"] == tid
    assert t1["input"] == "3 year old with cough" and t1["output"] == r1["message"]
    assert t2["metadata"]["phase"] == "done" and t2["metadata"]["triage_level"] == "urgent"
    assert t1["trace_id"] != t2["trace_id"]


def test_spans_per_agent_with_latency_tokens_and_cost(platform):
    _three_turns()
    t2 = platform.requests[1]["json"]  # completing turn: intake, retrieval, triage all ran
    agents = {s["name"]: s for s in t2["spans"] if s["type"] == "agent"}
    assert {"supervisor", "intake", "retrieval", "triage", "output"} <= set(agents)
    llm_spans = [s for s in t2["spans"] if s["type"] == "llm"]
    # completing turn: intake extraction, retrieval query gen (stub summary has no
    # search_queries), triage = 3 calls (relevance judge is patched out in stubbed();
    # Phase 5 made the completion message fixed text)
    assert len(llm_spans) == t2["metadata"]["llm_calls"] == 3
    for s in llm_spans:
        assert s["parent_span_id"] in {a["span_id"] for a in agents.values()}
        assert s["usage"] == {"input_tokens": 100, "output_tokens": 20, "reasoning_tokens": 30, "total_tokens": 150}
        assert s["latency_ms"] >= 0 and s["model"] == "gemini-3.1-pro-preview"
        assert s["cost_usd"] == pytest.approx((100 * 2.0 + 50 * 12.0) / 1e6)
    assert agents["intake"]["llm_calls"] == 1 and agents["triage"]["llm_calls"] == 1
    assert agents["retrieval"]["llm_calls"] == 1
    assert agents["intake"]["usage"]["total_tokens"] == 150
    assert agents["supervisor"]["llm_calls"] == 0 and agents["supervisor"]["cost_usd"] == 0.0
    assert t2["usage"]["total_tokens"] == 450
    assert t2["cost_usd"] == pytest.approx(3 * (100 * 2.0 + 50 * 12.0) / 1e6)
    assert t2["latency_ms"] >= max(s["latency_ms"] for s in t2["spans"])


def test_cost_is_null_when_a_model_has_no_known_price(platform):
    _three_turns(llm_model="gemini-3.8-flash")
    t = platform.requests[1]["json"]
    assert t["cost_usd"] is None
    assert all(s["cost_usd"] is None for s in t["spans"] if s["type"] == "llm")


def test_custom_key_header(platform, monkeypatch):
    monkeypatch.setenv("TRACE_API_KEY_HEADER", "X-API-Key")
    tracing.reset_tracer()
    _three_turns()
    h = {k.lower(): v for k, v in platform.requests[0]["headers"].items()}  # header names are case-insensitive
    assert h["x-api-key"] == "test-key-123" and "authorization" not in h


def test_disabled_without_env(monkeypatch):
    monkeypatch.delenv("TRACE_API_URL", raising=False)
    monkeypatch.delenv("TRACE_API_KEY", raising=False)
    tracing.reset_tracer()
    t = tracing.get_tracer()
    assert t.enabled is False and t.new_turn("s", "x") is None


def test_platform_down_never_breaks_or_blocks_triage(monkeypatch):
    # Nothing listens on this port
    monkeypatch.setenv("TRACE_API_URL", "http://127.0.0.1:9")
    monkeypatch.setenv("TRACE_API_KEY", "k")
    monkeypatch.setenv("TRACE_TIMEOUT_S", "0.5")
    tracing.reset_tracer()
    t0 = time.perf_counter()
    tid, r1, r2 = _three_turns()
    assert r2["is_complete"]
    # flush() above waited for the retries; the turns themselves did not
    assert time.perf_counter() - t0 < 30
    assert tracing.get_tracer().stats["failed"] == 2
    tracing.reset_tracer()


def test_server_error_retried_then_counted_failed(monkeypatch):
    p = _Platform(status=503)
    monkeypatch.setenv("TRACE_API_URL", p.url)
    monkeypatch.setenv("TRACE_API_KEY", "k")
    tracing.reset_tracer()
    tr = tracing.get_tracer()
    tr.submit({"trace_id": "x"})
    assert tr.flush(10)
    assert len(p.requests) == 3 and tr.stats["failed"] == 1   # 3 attempts on 5xx
    p.status = 400
    tr.submit({"trace_id": "y"})
    assert tr.flush(10)
    assert len(p.requests) == 4 and tr.stats["failed"] == 2   # no retry on 4xx
    p.close()
    tracing.reset_tracer()


def test_llm_error_recorded_as_error_span(platform):
    from graph import create_graph, run_turn
    llm = StubLLM(structured={"PatientSummary": [summary(0.3)]}, questions=["q?"], fail_kinds=("question",))
    app = create_graph()
    with stubbed(llm):
        with pytest.raises(RuntimeError):
            run_turn(app, str(uuid.uuid4()), "child with cough")
    assert tracing.get_tracer().flush(5)
    t = platform.requests[0]["json"]
    assert t["status"] == "error"
    err = [s for s in t["spans"] if s["type"] == "llm" and s["status"] == "error"]
    assert len(err) == 1 and "stub failure" in err[0]["error"]
