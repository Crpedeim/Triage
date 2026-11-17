"""
tracing.py — one trace per user turn, sent to the LLM eval platform as
``POST {TRACE_API_URL}/api/v1/traces``.

The payload follows the platform's contract, docs/TRACING_API.md in the eval
platform repo (request schema: src/lib/trace-input.ts). tests/test_tracing_contract.py
checks it against that document's examples, rules and response bodies.

Configuration (environment). Tracing is OFF unless both are set:
  TRACE_API_URL     platform origin, e.g. https://eval-web-....run.app
  TRACE_API_KEY     project API key (evp_live_...), sent as  Authorization: Bearer <key>
  TRACE_TIMEOUT_S   per-request timeout in seconds (default 5)

What one turn sends:
  name      "triage.turn" for a turn that ends with a question, "triage.final"
            for the turn that produces the triage card (evaluators are scoped
            to a trace name, so the judges run on "triage.final" only)
  input     the user's message; on the final turn, the whole conversation plus
            the guideline text the triage agent was given (an LLM judge sees
            only input and output)
  output    the assistant's reply; on the final turn, the plain-text triage card
  model     the triage model on the final turn, else the last model called
  promptTokens / completionTokens   sums over the LLM spans (completion
            includes reasoning tokens); latencyMs = wall time of the turn
  metadata  sessionId, turn (1-based user-turn number), final, phase,
            llmCalls, models; triageLevel and triageConfidence on the final
            turn; plus any caller metadata (the eval adds evalCase / evalLabel)
  spans     a tree: root span for the turn → one span per agent run
            (intake, retrieval, relevance_judge, triage; supervisor = "router",
            output) → one span per LLM call inside it (kind "llm", model,
            tokens, latency, status). Tokens sit only on the LLM spans, so the
            platform counts each call once.

Cost is NOT sent: the platform computes it from model and tokens.

Delivery: payloads go on a bounded in-memory queue and a daemon thread POSTs
them. Network errors, timeouts, 429 and 5xx are retried (3 attempts, backoff);
other 4xx are logged with the platform's error body and not retried. Tracing
never raises into, or waits on, the triage request: a full queue drops the
trace (counted in Tracer.stats), and every failure is only logged.
Tracer.flush() drains the queue (tests, the eval, shutdown).
"""

from __future__ import annotations

import atexit
import collections
import json
import logging
import os
import queue
import signal
import threading
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone

from langchain_core.callbacks import BaseCallbackHandler

log = logging.getLogger("triage.tracing")

TRACE_NAME_TURN = "triage.turn"
TRACE_NAME_FINAL = "triage.final"
MAX_SPANS = 500          # platform limit per request (TRACING_API.md)

# LangGraph node → (span name, span kind)
NODE_SPANS = {
    "supervisor": ("supervisor", "router"),
    "intake": ("intake", "agent"),
    "retrieval": ("retrieval", "agent"),
    "relevance_check": ("relevance_judge", "agent"),
    "triage": ("triage", "agent"),
    "output": ("output", "output"),
}


def iso_ms(ts: float) -> str:
    """ISO 8601 UTC with milliseconds, e.g. 2026-11-17T10:00:00.050Z."""
    return datetime.fromtimestamp(ts, tz=timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def _ms(seconds: float) -> int:
    return max(0, int(round(seconds * 1000)))


class TurnTrace(BaseCallbackHandler):
    """LangChain callback that records agent spans and LLM-call spans for one user turn."""

    def __init__(self, session_id: str, user_input: str, metadata: dict | None = None) -> None:
        self.session_id = session_id
        self.user_input = user_input
        self.extra_metadata = dict(metadata or {})
        self.start_wall = time.time()
        self.start_perf = time.perf_counter()
        self.end_perf: float | None = None
        self.spans: list[dict] = []         # closed spans
        self._open: dict = {}               # run_id -> span
        self._parent: dict = {}             # run_id -> parent run_id (every chain)
        self._agent_runs: dict = {}         # run_id -> agent span (open or closed)
        self._n = 0
        self._lock = threading.Lock()

    def _new_id(self) -> str:
        self._n += 1
        return f"s{self._n}"

    def _span(self, name: str, kind: str, **extra) -> dict:
        return {"id": self._new_id(), "parentId": "turn", "name": name, "kind": kind, "status": "ok",
                "_wall": time.time(), "_perf": time.perf_counter(), **extra}

    # ── agent spans (LangGraph nodes) ────────────────────────────────────
    def on_chain_start(self, serialized, inputs, *, run_id, parent_run_id=None, **kw):
        name = kw.get("name") or (serialized or {}).get("name")
        with self._lock:
            self._parent[run_id] = parent_run_id
            node = (kw.get("metadata") or {}).get("langgraph_node")
            if name in NODE_SPANS and (node is None or node == name):
                span_name, kind = NODE_SPANS[name]
                span = self._span(span_name, kind, _node=name)
                self._open[run_id] = span
                self._agent_runs[run_id] = span

    def on_chain_end(self, outputs, *, run_id, **kw):
        self._close(run_id, "ok")

    def on_chain_error(self, error, *, run_id, **kw):
        self._close(run_id, "error", str(error)[:500])

    # ── LLM-call spans ───────────────────────────────────────────────────
    def _agent_for(self, parent_run_id, node: str | None) -> dict | None:
        rid, hops = parent_run_id, 0
        while rid is not None and hops < 50:
            if rid in self._agent_runs:
                return self._agent_runs[rid]
            rid, hops = self._parent.get(rid), hops + 1
        # Fallback: the most recent span of that node
        for span in reversed(list(self._agent_runs.values())):
            if span.get("_node") == node:
                return span
        return None

    def on_chat_model_start(self, serialized, messages, *, run_id, parent_run_id=None, **kw):
        inv = kw.get("invocation_params") or {}
        node = (kw.get("metadata") or {}).get("langgraph_node")
        with self._lock:
            agent = self._agent_for(parent_run_id, node)
            model = inv.get("model") or inv.get("model_name") or inv.get("model_id")
            span = self._span("llm", "llm", model=model or None,
                              _agent=agent["name"] if agent else node)
            span["parentId"] = agent["id"] if agent else "turn"
            self._open[run_id] = span

    def on_llm_end(self, response, *, run_id, **kw):
        um = {}
        try:
            um = response.generations[0][0].message.usage_metadata or {}
        except Exception:
            pass
        with self._lock:
            span = self._open.get(run_id)
            if span is not None:
                inp = int(um.get("input_tokens") or 0)
                out = int(um.get("output_tokens") or 0)
                tot = int(um.get("total_tokens") or 0)
                # Gemini's total includes reasoning ("thinking") tokens that are not
                # in output_tokens; the platform bills them as output.
                span["promptTokens"] = inp
                span["completionTokens"] = max(out, tot - inp)
                span["_reasoning"] = max(0, tot - inp - out)
        self._close(run_id, "ok")

    def on_llm_error(self, error, *, run_id, **kw):
        self._close(run_id, "error", str(error)[:500])

    def _close(self, run_id, status, error=None):
        with self._lock:
            span = self._open.pop(run_id, None)
            if span is None:
                return
            span["latencyMs"] = _ms(time.perf_counter() - span["_perf"])
            span["status"] = status
            if error:
                span["_error"] = error
            self.spans.append(span)

    # ── payload ──────────────────────────────────────────────────────────
    def finish(self) -> None:
        if self.end_perf is None:
            self.end_perf = time.perf_counter()

    def build_payload(self, output: str, state: dict | None, status: str = "ok", final: bool = False) -> dict:
        """The POST /api/v1/traces body for this turn (TRACING_API.md)."""
        self.finish()
        state = state or {}
        latency = _ms(self.end_perf - self.start_perf)
        with self._lock:
            closed = list(self.spans)
            still_open = list(self._open.values())
        for s in still_open:                 # e.g. the turn raised mid-node
            s = dict(s, status="error", latencyMs=_ms(self.end_perf - s["_perf"]), _error="not finished")
            closed.append(s)
        llm = [s for s in closed if s["kind"] == "llm"]
        prompt_tokens = sum(s.get("promptTokens", 0) for s in llm)
        completion_tokens = sum(s.get("completionTokens", 0) for s in llm)
        any_error = status != "ok" or any(s["status"] == "error" for s in closed)
        name = TRACE_NAME_FINAL if final else TRACE_NAME_TURN

        spans = [{"id": "turn", "name": name, "kind": "turn", "status": "error" if any_error else "ok",
                  "startedAt": iso_ms(self.start_wall), "latencyMs": latency}]
        for s in sorted(closed, key=lambda s: s["_perf"]):
            out = {"id": s["id"], "parentId": s["parentId"], "name": s["name"], "kind": s["kind"],
                   "status": s["status"], "startedAt": iso_ms(s["_wall"]), "latencyMs": s["latencyMs"]}
            meta: dict = {}
            if s["kind"] == "llm":
                if s.get("model"):
                    out["model"] = s["model"]
                out["promptTokens"] = s.get("promptTokens", 0)
                out["completionTokens"] = s.get("completionTokens", 0)
                meta["agent"] = s.get("_agent")
                if s.get("_reasoning"):
                    meta["reasoningTokens"] = s["_reasoning"]
            else:
                meta["node"] = s.get("_node")
            if s.get("_error"):
                meta["error"] = s["_error"]
            out["metadata"] = meta
            spans.append(out)
        dropped = 0
        if len(spans) > MAX_SPANS:
            dropped = len(spans) - MAX_SPANS
            spans = spans[:MAX_SPANS]

        triage_models = [s["model"] for s in llm if s.get("_agent") == "triage" and s.get("model")]
        any_models = [s["model"] for s in llm if s.get("model")]
        model = (triage_models or any_models or ["no-llm-call"])[-1]

        tr = state.get("triage_result")
        metadata = {
            "app": "triage-ai",
            "sessionId": self.session_id,
            "turn": _user_turns(state),
            "final": final,
            "phase": state.get("phase"),
            "llmCalls": len(llm),
            "models": sorted(set(any_models)),
            **({"spansDropped": dropped} if dropped else {}),
        }
        if final:
            metadata["triageLevel"] = tr.triage_level.value if tr is not None else None
            metadata["triageConfidence"] = state.get("triage_confidence")
            metadata["assistantMessage"] = output
            metadata["guidelinesCited"] = list(tr.guidelines_cited) if tr is not None else []
        metadata.update(self.extra_metadata)

        if final:
            trace_input = final_input_text(state)
            trace_output = final_output_text(state)
        else:
            trace_input, trace_output = self.user_input, output
        return {
            "name": name,
            "input": trace_input,
            "output": trace_output,
            "model": model,
            "promptTokens": prompt_tokens,
            "completionTokens": completion_tokens,
            "latencyMs": latency,
            "status": "error" if any_error else "ok",
            "metadata": metadata,
            "spans": spans,
        }


def _text(msg) -> str:
    c = getattr(msg, "content", "")
    if isinstance(c, str):
        return c
    return " ".join(b.get("text", "") for b in c if isinstance(b, dict)).strip()


def _user_turns(state: dict) -> int:
    from langchain_core.messages import HumanMessage
    return sum(isinstance(m, HumanMessage) for m in state.get("messages") or [])


def final_input_text(state: dict) -> str:
    """Conversation + the guideline text the triage agent was given (what the judges need)."""
    from langchain_core.messages import AIMessage, HumanMessage
    lines = ["CONVERSATION (health worker / caregiver and the triage assistant):"]
    for m in state.get("messages") or []:
        if isinstance(m, HumanMessage):
            lines.append(f"User: {_text(m)}")
        elif isinstance(m, AIMessage):
            lines.append(f"Assistant: {_text(m)}")
    lines += ["", "GUIDELINE TEXT GIVEN TO THE TRIAGE AGENT:"]
    docs = state.get("retrieved_guidelines") or []
    if not docs:
        lines.append("(none retrieved)")
    for i, d in enumerate(docs, 1):
        md = d.get("metadata") or {}
        src = md.get("document") or md.get("source") or md.get("guideline_type") or "guideline"
        page = f", p. {md['pages']}" if md.get("pages") else ""
        lines.append(f"[{i}] {src}{page}: {md.get('section', 'unknown section')}")
        lines.append(str(d.get("content", "")).strip())
        lines.append("")
    return "\n".join(lines).rstrip()


def final_output_text(state: dict) -> str:
    from output import format_triage_card
    return format_triage_card(state)


class Tracer:
    """Background sender. One instance per process (get_tracer())."""

    def __init__(self, base_url: str | None, api_key: str | None, timeout_s: float = 5.0,
                 max_queue: int = 1000, backoff_s: float = 0.5) -> None:
        self.base_url = (base_url or "").rstrip("/")
        self.api_key = api_key
        self.timeout_s = timeout_s
        self.backoff_s = backoff_s
        self.enabled = bool(self.base_url and self.api_key)
        self.stats = {"submitted": 0, "sent": 0, "failed": 0, "dropped": 0, "retries": 0}
        # 202 responses: {"id", "costUsd", "evalsQueued", "spans", plus name/sessionId/turn/...}
        self.accepted: collections.deque = collections.deque(maxlen=10000)
        self._q: queue.Queue = queue.Queue(maxsize=max_queue)
        self._worker: threading.Thread | None = None
        self._worker_lock = threading.Lock()

    @property
    def endpoint(self) -> str:
        return f"{self.base_url}/api/v1/traces"

    def new_turn(self, session_id: str, user_input: str, metadata: dict | None = None) -> TurnTrace | None:
        return TurnTrace(session_id, user_input, metadata) if self.enabled else None

    def submit_turn(self, turn: TurnTrace | None, output: str, state: dict | None,
                    status: str = "ok", final: bool = False) -> None:
        """Build and queue a turn's trace. Never raises (a tracing bug must not break triage)."""
        if turn is None or not self.enabled:
            return
        try:
            self.submit(turn.build_payload(output, state, status=status, final=final))
        except Exception as e:
            self.stats["failed"] += 1
            log.warning("could not build trace: %s: %s", type(e).__name__, e)

    def submit(self, payload: dict) -> None:
        if not self.enabled:
            return
        self.stats["submitted"] += 1
        try:
            self._q.put_nowait(payload)
        except queue.Full:
            self.stats["dropped"] += 1
            log.warning("trace queue full; dropped a trace")
            return
        with self._worker_lock:
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
            except Exception as e:  # never let the sender thread die
                self.stats["failed"] += 1
                log.warning("trace sender error: %s", e)
            finally:
                self._q.task_done()

    def _send(self, payload: dict) -> None:
        body = json.dumps(payload, default=str).encode()
        headers = {"Content-Type": "application/json", "Authorization": f"Bearer {self.api_key}",
                   "User-Agent": "triage-ai-tracing/2"}
        md = payload.get("metadata") or {}
        label = f"{payload.get('name')} session={md.get('sessionId')} turn={md.get('turn')}"
        for attempt in range(3):
            if attempt:
                self.stats["retries"] += 1
                time.sleep(self.backoff_s * (2 ** (attempt - 1)))
            try:
                req = urllib.request.Request(self.endpoint, data=body, headers=headers, method="POST")
                with urllib.request.urlopen(req, timeout=self.timeout_s) as resp:
                    raw = resp.read()
                    self.stats["sent"] += 1
                    try:
                        info = json.loads(raw or b"{}")
                    except ValueError:
                        info = {}
                    self.accepted.append({**(info if isinstance(info, dict) else {}), "name": payload.get("name"),
                                          **{k: md.get(k) for k in ("sessionId", "turn", "evalCase", "evalLabel")}})
                    return
            except urllib.error.HTTPError as e:
                detail = ""
                try:
                    detail = e.read().decode("utf-8", "replace")[:500]
                except Exception:
                    pass
                if e.code < 500 and e.code != 429:   # 400 / 401 / 422: retrying won't help
                    self.stats["failed"] += 1
                    log.warning("trace rejected (%s): HTTP %s %s", label, e.code, detail)
                    return
                log.info("trace send attempt %d (%s): HTTP %s %s", attempt + 1, label, e.code, detail)
            except Exception as e:  # network error / timeout
                log.info("trace send attempt %d (%s) failed: %s", attempt + 1, label, e)
        self.stats["failed"] += 1
        log.warning("trace not delivered after 3 attempts (%s)", label)


_TRACER: Tracer | None = None
_TRACER_LOCK = threading.Lock()


def get_tracer() -> Tracer:
    global _TRACER
    with _TRACER_LOCK:
        if _TRACER is None:
            _TRACER = Tracer(
                base_url=os.getenv("TRACE_API_URL"),
                api_key=os.getenv("TRACE_API_KEY"),
                timeout_s=float(os.getenv("TRACE_TIMEOUT_S", "5")),
            )
            atexit.register(_TRACER.flush, 5.0)
        return _TRACER


def reset_tracer() -> None:
    """For tests: re-read the environment on next get_tracer()."""
    global _TRACER
    with _TRACER_LOCK:
        _TRACER = None


def install_sigterm_flush(timeout_s: float = 5.0) -> None:
    """On SIGTERM (Cloud Run scale-in), send queued traces before exiting. Main thread only."""
    previous = signal.getsignal(signal.SIGTERM)

    def handler(signum, frame):
        try:
            if _TRACER is not None and _TRACER.enabled:
                _TRACER.flush(timeout_s)
        finally:
            if callable(previous) and previous not in (signal.SIG_DFL, signal.SIG_IGN):
                previous(signum, frame)
            else:
                raise SystemExit(0)

    try:
        signal.signal(signal.SIGTERM, handler)
    except (ValueError, OSError):   # not the main thread / unsupported platform
        pass
