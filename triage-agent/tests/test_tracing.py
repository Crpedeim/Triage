"""
tests/test_tracing.py — one trace per user turn, POSTed to /api/v1/traces.

A local HTTP server stands in for the eval platform and records every request
(it answers 202 with the documented body). The REAL compiled graph runs with a
stub LLM that fires real LangChain callbacks (100 input / 20 output / 150
total tokens per call, so 30 reasoning tokens).
"""

from __future__ import annotations

import json
import socket
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


class FakePlatform:
    """Records requests; answers with (status, body) from `responses` (last one repeats)."""

    def __init__(self, responses=None, delay_s: float = 0.0):
        self.requests: list[dict] = []
        self.responses = list(responses or [(202, {"id": "t1", "evalsQueued": 0, "costUsd": 0, "spans": 0})])
        self.delay_s = delay_s
        outer = self

        class H(BaseHTTPRequestHandler):
            def do_POST(self):
                body = self.rfile.read(int(self.headers["Content-Length"]))
                outer.requests.append({"path": self.path, "headers": dict(self.headers), "json": json.loads(body)})
                if outer.delay_s:
                    time.sleep(outer.delay_s)
                status, resp = outer.responses.pop(0) if len(outer.responses) > 1 else outer.responses[0]
                data = json.dumps(resp).encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def log_message(self, *a):
                pass

        self.server = HTTPServer(("127.0.0.1", 0), H)
        self.url = f"http://127.0.0.1:{self.server.server_address[1]}"
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def close(self):
        self.server.shutdown()


@pytest.fixture
def platform(monkeypatch):
    p = FakePlatform()
    monkeypatch.setenv("TRACE_API_URL", p.url)
    monkeypatch.setenv("TRACE_API_KEY", "evp_live_test")
    tracing.reset_tracer()
    yield p
    tracing.reset_tracer()
    p.close()


def two_turns(llm_model="gemini-3.1-pro-preview", trace_metadata=None, flush_s=10):
    from graph import create_graph, run_turn
    llm = StubLLM(structured={"PatientSummary": [summary(0.3), summary(0.9)], "TriageResult": [triage(conf=0.9)]},
                  questions=["How many breaths per minute?"], model=llm_model)
    app, tid = create_graph(), str(uuid.uuid4())
    times = []
    with stubbed(llm):
        t0 = time.perf_counter(); r1 = run_turn(app, tid, "3 year old with cough", trace_metadata=trace_metadata)
        times.append(time.perf_counter() - t0)
        t0 = time.perf_counter(); r2 = run_turn(app, tid, "36 breaths, drinking well", trace_metadata=trace_metadata)
        times.append(time.perf_counter() - t0)
    drained = tracing.get_tracer().flush(flush_s)
    return tid, r1, r2, times, drained


def test_one_trace_per_turn_with_bearer_key_and_session_metadata(platform):
    tid, r1, r2, _, drained = two_turns(trace_metadata={"evalCase": "A"})
    assert drained and len(platform.requests) == 2
    for req in platform.requests:
        assert req["path"] == "/api/v1/traces"
        h = {k.lower(): v for k, v in req["headers"].items()}
        assert h["authorization"] == "Bearer evp_live_test"
        assert h["content-type"] == "application/json"
    t1, t2 = (r["json"] for r in platform.requests)
    assert (t1["name"], t2["name"]) == ("triage.turn", "triage.final")
    assert t1["metadata"]["sessionId"] == t2["metadata"]["sessionId"] == tid
    assert (t1["metadata"]["turn"], t2["metadata"]["turn"]) == (1, 2)
    assert t1["metadata"]["final"] is False and t2["metadata"]["final"] is True
    assert "triageLevel" not in t1["metadata"]
    assert t2["metadata"]["triageLevel"] == "urgent" and t2["metadata"]["phase"] == "done"
    assert t1["metadata"]["evalCase"] == t2["metadata"]["evalCase"] == "A"
    # non-final turn: the user's message and the reply
    assert t1["input"] == "3 year old with cough" and t1["output"] == r1["message"]
    assert tracing.get_tracer().stats["sent"] == 2


def test_final_turn_carries_conversation_guideline_text_and_card(platform):
    _, r1, r2, _, _ = two_turns()
    t2 = platform.requests[1]["json"]
    assert t2["input"].startswith("CONVERSATION")
    assert "User: 3 year old with cough" in t2["input"] and "User: 36 breaths, drinking well" in t2["input"]
    assert f"Assistant: {r1['message']}" in t2["input"]
    assert "GUIDELINE TEXT GIVEN TO THE TRIAGE AGENT:" in t2["input"]
    from tests.test_multiturn import DOC
    assert DOC.content.strip()[:60] in t2["input"]          # the chunk text itself, not just its title
    assert "TRIAGE LEVEL: 🟠 URGENT" in t2["output"]
    assert "RECOMMENDED ACTIONS:" in t2["output"] and "BASED ON:" in t2["output"]
    assert t2["metadata"]["assistantMessage"] == r2["message"]
    assert t2["metadata"]["guidelinesCited"] == ["ASSESS AND CLASSIFY COUGH OR DIFFICULT BREATHING"]


def test_span_tree_agents_and_llm_calls(platform):
    two_turns()
    t2 = platform.requests[1]["json"]
    spans = t2["spans"]
    by_id = {s["id"]: s for s in spans}
    root = spans[0]
    assert root["id"] == "turn" and "parentId" not in root and root["name"] == "triage.final"
    assert root["latencyMs"] == t2["latencyMs"]
    agents = [s for s in spans if s["kind"] in ("agent", "router", "output")]
    assert {s["name"] for s in agents} >= {"supervisor", "intake", "retrieval", "relevance_judge", "triage", "output"}
    assert all(s["parentId"] == "turn" for s in agents)
    llm = [s for s in spans if s["kind"] == "llm"]
    # completing turn: intake extraction, retrieval query generation (the stub
    # summary has no search_queries), triage = 3 calls; the judge is patched out
    assert len(llm) == t2["metadata"]["llmCalls"] == 3
    assert sorted(s["metadata"]["agent"] for s in llm) == ["intake", "retrieval", "triage"]
    for s in llm:
        parent = by_id[s["parentId"]]
        assert parent["kind"] == "agent" and parent["name"] == s["metadata"]["agent"]
        assert s["model"] == "gemini-3.1-pro-preview" and s["status"] == "ok"
        assert (s["promptTokens"], s["completionTokens"]) == (100, 50)   # 20 output + 30 reasoning
        assert s["metadata"]["reasoningTokens"] == 30
    # tokens only on the LLM leaves (the platform would otherwise count them twice)
    assert all("promptTokens" not in s and "model" not in s for s in spans if s["kind"] != "llm")
    assert (t2["promptTokens"], t2["completionTokens"]) == (300, 150)
    assert t2["model"] == "gemini-3.1-pro-preview"
    # startedAt ordering and containment: children start inside their parent
    for s in spans[1:]:
        assert s["startedAt"] >= by_id[s["parentId"]]["startedAt"]
    assert max(s["latencyMs"] for s in spans) == root["latencyMs"]


def test_no_cost_is_sent(platform):
    two_turns()
    def keys(o):
        if isinstance(o, dict):
            for k, v in o.items():
                yield k
                yield from keys(v)
        elif isinstance(o, list):
            for v in o:
                yield from keys(v)
    for req in platform.requests:
        assert not [k for k in keys(req["json"]) if "cost" in k.lower()]


def test_disabled_without_env(monkeypatch):
    monkeypatch.delenv("TRACE_API_URL", raising=False)
    monkeypatch.delenv("TRACE_API_KEY", raising=False)
    tracing.reset_tracer()
    t = tracing.get_tracer()
    assert t.enabled is False and t.new_turn("s", "x") is None
    t.submit_turn(None, "x", {})          # no-op, no error


def _closed_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def test_platform_down_never_breaks_or_blocks_triage(monkeypatch):
    monkeypatch.setenv("TRACE_API_URL", f"http://127.0.0.1:{_closed_port()}")
    monkeypatch.setenv("TRACE_API_KEY", "k")
    monkeypatch.setenv("TRACE_TIMEOUT_S", "0.5")
    tracing.reset_tracer()
    _, r1, r2, times, drained = two_turns(flush_s=60)
    assert r2["is_complete"] and r2["state"]["triage_result"].triage_level.value == "urgent"
    assert drained
    st = tracing.get_tracer().stats
    assert st["failed"] == 2 and st["sent"] == 0 and st["retries"] == 4   # 3 attempts per trace
    # the turns did not wait for the (failing) sends
    assert max(times) < 5
    tracing.reset_tracer()


def test_slow_platform_does_not_slow_turns(monkeypatch):
    p = FakePlatform(delay_s=2.0)
    monkeypatch.setenv("TRACE_API_URL", p.url)
    monkeypatch.setenv("TRACE_API_KEY", "k")
    tracing.reset_tracer()
    _, _, r2, times, drained = two_turns(flush_s=30)
    assert r2["is_complete"] and drained and len(p.requests) == 2
    assert max(times) < 2.0      # each send took 2 s; no turn waited for one
    p.close()
    tracing.reset_tracer()


def test_tracing_bug_never_breaks_triage(platform, monkeypatch):
    def boom(*a, **k):
        raise KeyError("bug in build_payload")
    monkeypatch.setattr(tracing.TurnTrace, "build_payload", boom)
    _, _, r2, _, _ = two_turns()
    assert r2["is_complete"]
    assert tracing.get_tracer().stats["failed"] == 2 and platform.requests == []


def test_llm_error_recorded_as_error_span(platform):
    from graph import create_graph, run_turn
    llm = StubLLM(structured={"PatientSummary": [summary(0.3)]}, questions=["q?"], fail_kinds=("question",))
    app = create_graph()
    with stubbed(llm):
        with pytest.raises(RuntimeError):
            run_turn(app, str(uuid.uuid4()), "child with cough")
    assert tracing.get_tracer().flush(5)
    t = platform.requests[0]["json"]
    assert t["status"] == "error" and t["spans"][0]["status"] == "error"
    err = [s for s in t["spans"] if s["kind"] == "llm" and s["status"] == "error"]
    assert len(err) == 1 and "stub failure" in err[0]["metadata"]["error"]
    assert err[0]["metadata"]["agent"] == "intake"
