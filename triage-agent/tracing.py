"""
tracing.py — Send one trace per user turn, with spans per agent and per LLM call,
to an LLM eval platform via  POST {TRACE_API_URL}/api/v1/traces.

Configuration (environment):
  TRACE_API_URL         base URL of the platform, e.g. https://evals.example.com
                        (tracing is OFF unless this and TRACE_API_KEY are set)
  TRACE_API_KEY         API key, sent as  Authorization: Bearer <key>
  TRACE_API_KEY_HEADER  optional: send the raw key in this header instead
                        (e.g. X-API-Key)
  TRACE_TIMEOUT_S       per-request timeout (default 5)
  TRACE_SERVICE_NAME    reported service name (default "triage-ai")

Delivery: traces go on a bounded in-memory queue and a daemon thread POSTs
them, with 3 attempts and exponential backoff on network errors / 5xx / 429.
Tracing never raises into, or waits on, the triage request: when the queue is
full the trace is dropped and counted (Tracer.stats). Tracer.flush() drains
the queue (used at shutdown and in tests).

PAYLOAD — ASSUMED SHAPE. The platform's exact schema was not available when
this was written. The field names below are this module's own convention;
adapt build_payload() if the platform expects different names.

{
  "trace_id": "uuid", "name": "triage_turn", "session_id": "<thread_id>",
  "service": "triage-ai",
  "start_time": "ISO-8601 UTC", "end_time": "...", "latency_ms": 1234.5,
  "input": "<user message>", "output": "<assistant reply>",
  "status": "ok" | "error",
  "usage": {"input_tokens": n, "output_tokens": n, "reasoning_tokens": n, "total_tokens": n},
  "cost_usd": 0.0123 | null,          # null when any model used has no known price
  "metadata": {"phase": "...", "triage_level": "...", "triage_confidence": 0.8,
               "models": ["..."], "llm_calls": n},
  "spans": [
    {"span_id": "...", "parent_span_id": null, "name": "intake", "type": "agent",
     "start_time": "...", "end_time": "...", "latency_ms": ..., "status": "ok",
     "usage": {...}, "cost_usd": ... | null, "llm_calls": n},
    {"span_id": "...", "parent_span_id": "<agent span id>", "name": "llm",
     "type": "llm", "model": "gemini-3.8-flash", ... same timing/usage/cost fields,
     "error": "..." (only on failure)}
  ]
}
"""

from __future__ import annotations

import atexit
import json
import logging
import os
import queue
import threading
import time
import urllib.error
import urllib.request
import uuid
from datetime import datetime, timezone

from langchain_core.callbacks import BaseCallbackHandler

from pricing import call_cost

log = logging.getLogger("triage.tracing")

AGENT_NODES = {"supervisor", "intake", "retrieval", "relevance_check", "triage", "output"}


def _iso(ts: float) -> str:
    return datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()


class TurnTrace(BaseCallbackHandler):
    """LangChain callback that records agent (node) spans and LLM-call spans for one turn."""

    def __init__(self, session_id: str, user_input: str) -> None:
        self.trace_id = str(uuid.uuid4())
        self.session_id = session_id
        self.user_input = user_input
        self.start = time.time()
        self.end: float | None = None
        self.spans: list[dict] = []
        self._open: dict = {}           # run_id -> span
        self._agent_by_name: dict = {}  # node name -> currently open agent span
        self._lock = threading.Lock()

    # ── agent spans (LangGraph nodes) ────────────────────────────────────
    def on_chain_start(self, serialized, inputs, *, run_id, **kw):
        name = kw.get("name") or (serialized or {}).get("name")
        if name not in AGENT_NODES:
            return
        span = {"span_id": str(uuid.uuid4()), "parent_span_id": None, "name": name, "type": "agent",
                "_t0": time.time(), "status": "ok"}
        with self._lock:
            self._open[run_id] = span
            self._agent_by_name[name] = span

    def on_chain_end(self, outputs, *, run_id, **kw):
        self._close(run_id, "ok")

    def on_chain_error(self, error, *, run_id, **kw):
        self._close(run_id, "error", str(error)[:500])

    # ── LLM-call spans ───────────────────────────────────────────────────
    def on_chat_model_start(self, serialized, messages, *, run_id, **kw):
        inv = kw.get("invocation_params") or {}
        node = (kw.get("metadata") or {}).get("langgraph_node")
        with self._lock:
            parent = self._agent_by_name.get(node)
            self._open[run_id] = {
                "span_id": str(uuid.uuid4()),
                "parent_span_id": parent["span_id"] if parent else None,
                "name": "llm", "type": "llm", "agent": node,
                "model": inv.get("model") or inv.get("model_name") or inv.get("model_id"),
                "_t0": time.time(), "status": "ok",
            }

    def on_llm_end(self, response, *, run_id, **kw):
        um = {}
        try:
            um = response.generations[0][0].message.usage_metadata or {}
        except Exception:
            pass
        with self._lock:
            span = self._open.get(run_id)
            if span is not None:
                inp, out, tot = um.get("input_tokens", 0), um.get("output_tokens", 0), um.get("total_tokens", 0)
                span["usage"] = {"input_tokens": inp, "output_tokens": out,
                                 "reasoning_tokens": max(0, tot - inp - out), "total_tokens": tot}
                span["cost_usd"] = call_cost(span["model"], inp, out, max(0, tot - inp - out))
        self._close(run_id, "ok")

    def on_llm_error(self, error, *, run_id, **kw):
        with self._lock:
            span = self._open.get(run_id)
            if span is not None:
                span["usage"] = {"input_tokens": 0, "output_tokens": 0, "reasoning_tokens": 0, "total_tokens": 0}
                span["cost_usd"] = None
        self._close(run_id, "error", str(error)[:500])

    # ── helpers ──────────────────────────────────────────────────────────
    def _close(self, run_id, status, error=None):
        with self._lock:
            span = self._open.pop(run_id, None)
            if span is None:
                return
            t1 = time.time()
            span["start_time"], span["end_time"] = _iso(span["_t0"]), _iso(t1)
            span["latency_ms"] = round((t1 - span.pop("_t0")) * 1000, 1)
            span["status"] = status
            if error:
                span["error"] = error
            if span["type"] == "agent" and self._agent_by_name.get(span["name"]) is span:
                del self._agent_by_name[span["name"]]
            self.spans.append(span)

    def build_payload(self, output: str, state: dict | None, status: str = "ok",
                      service: str = "triage-ai") -> dict:
        self.end = self.end or time.time()
        llm_spans = [s for s in self.spans if s["type"] == "llm"]
        # Roll LLM usage/cost up into each agent span
        for agent in (s for s in self.spans if s["type"] == "agent"):
            kids = [s for s in llm_spans if s["parent_span_id"] == agent["span_id"]]
            agent["llm_calls"] = len(kids)
            agent["usage"] = _sum_usage(kids)
            agent["cost_usd"] = _sum_cost(kids)
        state = state or {}
        tr = state.get("triage_result")
        return {
            "trace_id": self.trace_id, "name": "triage_turn", "session_id": self.session_id,
            "service": service,
            "start_time": _iso(self.start), "end_time": _iso(self.end),
            "latency_ms": round((self.end - self.start) * 1000, 1),
            "input": self.user_input, "output": output,
            "status": "error" if status != "ok" or any(s["status"] == "error" for s in self.spans) else "ok",
            "usage": _sum_usage(llm_spans),
            "cost_usd": _sum_cost(llm_spans),
            "metadata": {
                "phase": state.get("phase"),
                "triage_level": tr.triage_level.value if tr is not None else None,
                "triage_confidence": state.get("triage_confidence"),
                "models": sorted({s["model"] for s in llm_spans if s.get("model")}),
                "llm_calls": len(llm_spans),
            },
            "spans": sorted(self.spans, key=lambda s: s["start_time"]),
        }


def _sum_usage(spans: list[dict]) -> dict:
    keys = ("input_tokens", "output_tokens", "reasoning_tokens", "total_tokens")
    return {k: sum((s.get("usage") or {}).get(k, 0) for s in spans) for k in keys}


def _sum_cost(spans: list[dict]) -> float | None:
    """None when any successful call has an unknown price: never report a partial total as the total."""
    costs = [s.get("cost_usd") for s in spans if s["status"] == "ok"]
    if not costs:
        return 0.0
    if any(c is None for c in costs):
        return None
    return round(sum(costs), 8)


class Tracer:
    """Background sender. One instance per process (get_tracer())."""

    def __init__(self, base_url: str | None, api_key: str | None, key_header: str | None = None,
                 timeout_s: float = 5.0, service: str = "triage-ai", max_queue: int = 1000) -> None:
        self.base_url = (base_url or "").rstrip("/")
        self.api_key = api_key
        self.key_header = key_header
        self.timeout_s = timeout_s
        self.service = service
        self.enabled = bool(self.base_url and self.api_key)
        self.stats = {"sent": 0, "failed": 0, "dropped": 0}
        self._q: queue.Queue = queue.Queue(maxsize=max_queue)
        self._worker: threading.Thread | None = None

    @property
    def endpoint(self) -> str:
        return f"{self.base_url}/api/v1/traces"

    def new_turn(self, session_id: str, user_input: str) -> TurnTrace | None:
        return TurnTrace(session_id, user_input) if self.enabled else None

    def submit(self, payload: dict) -> None:
        if not self.enabled:
            return
        try:
            self._q.put_nowait(payload)
        except queue.Full:
            self.stats["dropped"] += 1
            log.warning("trace queue full; dropped trace %s", payload.get("trace_id"))
            return
        if self._worker is None or not self._worker.is_alive():
            self._worker = threading.Thread(target=self._run, name="trace-sender", daemon=True)
            self._worker.start()

    def flush(self, timeout_s: float = 10.0) -> bool:
        """Wait until queued traces are sent (or failed). Returns True if drained."""
        deadline = time.time() + timeout_s
        while time.time() < deadline:
            if self._q.unfinished_tasks == 0:
                return True
            time.sleep(0.02)
        return self._q.unfinished_tasks == 0

    def _run(self) -> None:
        while True:
            payload = self._q.get()
            try:
                self._send(payload)
            finally:
                self._q.task_done()

    def _send(self, payload: dict) -> None:
        body = json.dumps(payload, default=str).encode()
        headers = {"Content-Type": "application/json", "User-Agent": f"{self.service}-tracing/1"}
        if self.key_header:
            headers[self.key_header] = self.api_key
        else:
            headers["Authorization"] = f"Bearer {self.api_key}"
        for attempt in range(3):
            try:
                req = urllib.request.Request(self.endpoint, data=body, headers=headers, method="POST")
                with urllib.request.urlopen(req, timeout=self.timeout_s) as resp:
                    if 200 <= resp.status < 300:
                        self.stats["sent"] += 1
                        return
            except urllib.error.HTTPError as e:
                if e.code < 500 and e.code != 429:   # client error: retrying won't help
                    self.stats["failed"] += 1
                    log.warning("trace %s rejected: HTTP %s", payload.get("trace_id"), e.code)
                    return
            except Exception as e:  # network error / timeout
                log.debug("trace send attempt %d failed: %s", attempt + 1, e)
            time.sleep(0.2 * (2 ** attempt))
        self.stats["failed"] += 1
        log.warning("trace %s not delivered after 3 attempts", payload.get("trace_id"))


_TRACER: Tracer | None = None


def get_tracer() -> Tracer:
    global _TRACER
    if _TRACER is None:
        _TRACER = Tracer(
            base_url=os.getenv("TRACE_API_URL"),
            api_key=os.getenv("TRACE_API_KEY"),
            key_header=os.getenv("TRACE_API_KEY_HEADER") or None,
            timeout_s=float(os.getenv("TRACE_TIMEOUT_S", "5")),
            service=os.getenv("TRACE_SERVICE_NAME", "triage-ai"),
        )
        atexit.register(_TRACER.flush, 5.0)
    return _TRACER


def reset_tracer() -> None:
    """For tests: re-read the environment on next get_tracer()."""
    global _TRACER
    _TRACER = None
