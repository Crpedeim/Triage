"""
eval/run_eval.py — End-to-end evaluation runner for TriageAI.

IMPORTANT: the cases in eval/cases.json are a SELF-WRITTEN set of patient
vignettes. Each expected level is derived from the project's own sample
guideline text (rag/ingest.py) and the level mapping in state.py. This is a
regression / sanity suite, NOT a validated clinical benchmark.

How a case runs:
  * The real compiled graph from graph.create_graph() is used.
  * A simulated caregiver (a separate LLM call, NOT part of the system) answers
    each intake question using only the hidden vignette. Its time and tokens are
    reported separately and excluded from system metrics.
  * A LangChain callback records every system LLM call (model, latency, tokens)
    and per-node wall time.

Drivers:
  --driver run_turn  (default) uses graph.run_turn(), the same entry point as the UI.
  --driver manual    resumes the graph with update_state + invoke(None). Needed
                     only for the pre-fix baseline, where run_turn() was broken.

Metrics (see summarise()):
  accuracy          predicted level == expected level
  under-triage      predicted level LESS severe than expected (or no result)
  over-triage       predicted level MORE severe than expected
  latency           system wall time per case; median and p95 (nearest-rank)
  tokens            sum of API usage total_tokens over system LLM calls
  cost              computed ONLY for models with a price in PRICES_PER_M; a model
                    without a known price makes total cost "unknown"

Usage:
  python -m eval.run_eval --label phase0 [--driver manual] [--concurrency 4] [--ids A B ...] [--set baseline6]
"""

from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import sys
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from langchain_core.callbacks import BaseCallbackHandler  # noqa: E402
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage  # noqa: E402

SEVERITY = {"self_care": 0, "standard": 1, "urgent": 2, "emergency": 3}
NODE_NAMES = {"supervisor", "intake", "retrieval", "relevance_check", "triage", "output"}

# Prices live in pricing.py (shared with tracing). Unpriced models → unknown cost.
from pricing import PRICES_PER_M  # noqa: E402

MAX_CAREGIVER_TURNS = int(os.getenv("EVAL_MAX_CAREGIVER_TURNS", "10"))
MAX_STEPS = int(os.getenv("EVAL_MAX_STEPS", "30"))


class Recorder(BaseCallbackHandler):
    """Records system LLM calls and per-node wall time. Thread-safe per instance."""

    def __init__(self) -> None:
        self.calls: list[dict] = []
        self.nodes: list[tuple[str, float]] = []
        self._t: dict = {}
        self._lock = threading.Lock()

    def on_chat_model_start(self, serialized, messages, *, run_id, **kw):
        inv = kw.get("invocation_params") or {}
        model = inv.get("model") or inv.get("model_name") or inv.get("model_id")
        with self._lock:
            self._t[run_id] = (time.perf_counter(), model, kw.get("tags") or [], kw.get("metadata") or {})

    def on_llm_end(self, response, *, run_id, **kw):
        with self._lock:
            t0, model, _tags, meta = self._t.pop(run_id, (time.perf_counter(), None, [], {}))
        um = {}
        try:
            um = response.generations[0][0].message.usage_metadata or {}
        except Exception:
            pass
        inp, out, tot = um.get("input_tokens", 0), um.get("output_tokens", 0), um.get("total_tokens", 0)
        with self._lock:
            self.calls.append(dict(
                model=model, node=meta.get("langgraph_node"),
                latency_s=round(time.perf_counter() - t0, 3),
                input_tokens=inp, output_tokens=out, total_tokens=tot,
                reasoning_tokens=max(0, tot - inp - out),
            ))

    def on_llm_error(self, error, *, run_id, **kw):
        with self._lock:
            t0, model, _tags, meta = self._t.pop(run_id, (time.perf_counter(), None, [], {}))
            self.calls.append(dict(model=model, node=meta.get("langgraph_node"),
                                   error=str(error)[:300], latency_s=round(time.perf_counter() - t0, 3)))

    def on_chain_start(self, serialized, inputs, *, run_id, **kw):
        name = kw.get("name") or (serialized or {}).get("name")
        if name in NODE_NAMES:
            with self._lock:
                self._t[run_id] = (time.perf_counter(), "node:" + name, [], {})

    def on_chain_end(self, outputs, *, run_id, **kw):
        with self._lock:
            entry = self._t.get(run_id)
            if entry and str(entry[1]).startswith("node:"):
                self._t.pop(run_id)
                self.nodes.append((entry[1][5:], round(time.perf_counter() - entry[0], 3)))


def _text(msg) -> str:
    c = msg.content
    if isinstance(c, str):
        return c
    return " ".join(b.get("text", "") for b in c if isinstance(b, dict)).strip()


class Caregiver:
    """Simulated caregiver. Answers ONLY from the hidden vignette. Not part of the system."""

    def __init__(self) -> None:
        from langchain_openai import ChatOpenAI
        model = os.getenv("EVAL_CAREGIVER_MODEL") or os.getenv("FAST_MODEL_NAME", "gemini-3.8-flash")
        self.llm = ChatOpenAI(model=model, temperature=0)
        self.model = model

    def answer(self, vignette: str, question: str, log: list) -> str:
        t0 = time.perf_counter()
        r = self.llm.invoke([
            SystemMessage(content=(
                "You are a parent/caregiver (or the patient) answering a community health worker. "
                "Answer ONLY what was asked, in 1-2 short plain sentences, using ONLY these facts. "
                "If the facts do not cover the question, say you don't know or haven't noticed. "
                f"FACTS: {vignette}")),
            HumanMessage(content=question),
        ])
        um = r.usage_metadata or {}
        log.append(dict(latency_s=round(time.perf_counter() - t0, 3), total_tokens=um.get("total_tokens", 0)))
        return _text(r)


def _is_unanswered_question(values: dict, answered: set) -> AIMessage | None:
    msgs = values.get("messages") or []
    last = msgs[-1] if msgs else None
    if isinstance(last, AIMessage) and last.id not in answered and not values.get("intake_complete"):
        return last
    return None


def run_case_manual(app, case, caregiver, rec):
    """Pre-fix driver: update_state + invoke(None) to get past interrupt_before=['intake']."""
    from state import create_initial_state
    cfg = {"configurable": {"thread_id": str(uuid.uuid4())}, "callbacks": [rec]}
    transcript, cg_log, answered = [("caregiver", case["opening"])], [], set()
    sys_time, turns, status = 0.0, 0, "ok"
    s = create_initial_state()
    s["messages"] = [HumanMessage(content=case["opening"])]
    t0 = time.perf_counter(); app.invoke(s, cfg); sys_time += time.perf_counter() - t0
    for _ in range(MAX_STEPS):
        snap = app.get_state(cfg)
        if not snap.next:
            break
        q = _is_unanswered_question(snap.values, answered)
        if q is not None:
            if turns >= MAX_CAREGIVER_TURNS:
                status = "stopped: caregiver turn limit"; break
            answered.add(q.id)
            transcript.append(("intake", _text(q)))
            a = caregiver.answer(case["vignette"], _text(q), cg_log)
            transcript.append(("caregiver", a))
            app.update_state(cfg, {"messages": [HumanMessage(content=a)]})
            turns += 1
        t0 = time.perf_counter()
        app.invoke(None, cfg)
        sys_time += time.perf_counter() - t0
    else:
        status = "stopped: step limit"
    return app.get_state(cfg).values, transcript, cg_log, sys_time, turns, status


def run_case_run_turn(app, case, caregiver, rec):
    """Production driver: graph.run_turn(), exactly what the UI calls."""
    from graph import run_turn
    thread_id = str(uuid.uuid4())
    transcript, cg_log = [("caregiver", case["opening"])], []
    sys_time, turns, status = 0.0, 0, "ok"
    msg = case["opening"]
    rec.turn_times = []
    for _ in range(MAX_CAREGIVER_TURNS + 1):
        t0 = time.perf_counter()
        r = run_turn(app, thread_id, msg, config={"callbacks": [rec]})
        dt = time.perf_counter() - t0
        sys_time += dt
        rec.turn_times.append(round(dt, 3))
        if r["is_complete"]:
            break
        q = r["message"]
        transcript.append(("intake", q))
        if turns >= MAX_CAREGIVER_TURNS:
            status = "stopped: caregiver turn limit"; break
        msg = caregiver.answer(case["vignette"], q, cg_log)
        transcript.append(("caregiver", msg))
        turns += 1
    values = r.get("state") or {}
    return values, transcript, cg_log, sys_time, turns, status


def run_case(case: dict, driver: str, caregiver: Caregiver) -> dict:
    from graph import create_graph
    from output import format_triage_card
    app = create_graph()
    rec = Recorder()
    fn = run_case_manual if driver == "manual" else run_case_run_turn
    try:
        values, transcript, cg_log, sys_time, turns, status = fn(app, case, caregiver, rec)
    except Exception as e:  # recorded, never hidden
        values, transcript, cg_log, sys_time, turns = {}, [], [], 0.0, 0
        status = f"error: {type(e).__name__}: {str(e)[:300]}"
    tr = values.get("triage_result")
    ps = values.get("patient_summary")
    level = tr.triage_level.value if tr else None
    ok_calls = [c for c in rec.calls if "error" not in c]
    per_model: dict = {}
    for c in ok_calls:
        m = per_model.setdefault(c["model"], dict(calls=0, input=0, output=0, reasoning=0, total=0, llm_s=0.0))
        m["calls"] += 1; m["input"] += c["input_tokens"]; m["output"] += c["output_tokens"]
        m["reasoning"] += c["reasoning_tokens"]; m["total"] += c["total_tokens"]; m["llm_s"] = round(m["llm_s"] + c["latency_s"], 3)
    exp = case["expected"]
    return dict(
        id=case["id"], set=case.get("set"), expected=exp, predicted=level,
        correct=level == exp,
        under_triage=(level is None) or SEVERITY[level] < SEVERITY[exp],
        over_triage=(level is not None) and SEVERITY[level] > SEVERITY[exp],
        basis=case.get("basis"), status=status, final_phase=values.get("phase"),
        cycle_count=values.get("cycle_count"), confidence=values.get("triage_confidence"),
        red_flags=(ps.red_flags if ps else None), completeness=(ps.completeness_score if ps else None),
        safety_override=bool(tr and "[SAFETY OVERRIDE]" in tr.reasoning),
        suspected=[(c.name, c.confidence) for c in (tr.suspected_conditions if tr else [])],
        guidelines_cited=(tr.guidelines_cited if tr else []),
        retrieved_sections=[g.get("metadata", {}).get("section") for g in values.get("retrieved_guidelines", [])],
        intake_questions_asked=turns, system_wall_s=round(sys_time, 3),
        llm_calls=len(rec.calls), llm_errors=[c for c in rec.calls if "error" in c],
        per_model=per_model, node_times=rec.nodes, calls=rec.calls,
        turn_times=getattr(rec, "turn_times", None),  # per run_turn() call (run_turn driver only)
        caregiver=dict(calls=len(cg_log), total_tokens=sum(c["total_tokens"] for c in cg_log),
                       wall_s=round(sum(c["latency_s"] for c in cg_log), 3)),
        transcript=transcript,
        triage_card=format_triage_card(values) if tr else None,
    )


def _p95(xs: list[float]) -> float | None:
    """Nearest-rank 95th percentile (for n < 20 this is the maximum)."""
    if not xs:
        return None
    s = sorted(xs)
    return s[max(0, math.ceil(0.95 * len(s)) - 1)]


def _cost(per_model: dict) -> dict:
    """Cost for priced models; total is None when any used model has no price."""
    lo = hi = 0.0
    unpriced = []
    for m, u in per_model.items():
        p = PRICES_PER_M.get(m)
        if not p:
            unpriced.append(m); continue
        lo += (u["input"] * p["input"] + u["output"] * p["output"]) / 1e6
        hi += (u["input"] * p["input"] + (u["output"] + u["reasoning"]) * p["output"]) / 1e6
    return dict(priced_low=round(lo, 6), priced_high=round(hi, 6), unpriced_models=unpriced,
                total_known=not unpriced)


def summarise(results: list[dict], meta: dict) -> dict:
    n = len(results)
    walls = [r["system_wall_s"] for r in results if not r["status"].startswith("error")]
    toks = [sum(m["total"] for m in r["per_model"].values()) for r in results]
    costs = [_cost(r["per_model"]) for r in results]
    return dict(
        meta=meta, n=n,
        accuracy=dict(correct=sum(r["correct"] for r in results), n=n),
        under_triage=dict(count=sum(r["under_triage"] for r in results), n=n,
                          ids=[r["id"] for r in results if r["under_triage"]]),
        over_triage=dict(count=sum(r["over_triage"] for r in results), n=n,
                         ids=[r["id"] for r in results if r["over_triage"]]),
        no_result=sum(r["predicted"] is None for r in results),
        errors=[(r["id"], r["status"]) for r in results if r["status"] != "ok"],
        llm_errors=sum(len(r["llm_errors"]) for r in results),
        latency_s=dict(median=round(statistics.median(walls), 2) if walls else None,
                       p95_nearest_rank=_p95(walls), min=min(walls) if walls else None,
                       max=max(walls) if walls else None, n=len(walls)),
        tokens_per_triage=dict(median=statistics.median(toks) if toks else None,
                               p95_nearest_rank=_p95(toks), total=sum(toks)),
        llm_calls_per_triage=dict(median=statistics.median([r["llm_calls"] for r in results]) if results else None),
        final_turn_latency_s=_final_turn(results),
        priced_cost_per_triage_usd=dict(
            low_median=round(statistics.median([c["priced_low"] for c in costs]), 6) if costs else None,
            high_median=round(statistics.median([c["priced_high"] for c in costs]), 6) if costs else None,
            unpriced_models=sorted({m for c in costs for m in c["unpriced_models"]}),
            note="low = reasoning tokens not billed; high = reasoning billed at output rate; "
                 "total cost is unknown while any model is unpriced"),
        confusion=_confusion(results),
    )


def _final_turn(results):
    """Latency of the completing turn (intake complete → retrieval → triage), when recorded."""
    xs = [r["turn_times"][-1] for r in results if r.get("turn_times")]
    if not xs:
        return None
    return dict(median=round(statistics.median(xs), 2), p95_nearest_rank=_p95(xs), n=len(xs))


def _confusion(results):
    out: dict = {}
    for r in results:
        out.setdefault(r["expected"], {}).setdefault(str(r["predicted"]), 0)
        out[r["expected"]][str(r["predicted"])] += 1
    return out


def _ensure_knowledge_base():
    from rag.store import count
    if count() == 0:
        from rag.ingest import ingest_sample_data
        ingest_sample_data()


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", required=True)
    ap.add_argument("--driver", choices=["run_turn", "manual"], default="run_turn")
    ap.add_argument("--concurrency", type=int, default=4)
    ap.add_argument("--ids", nargs="*")
    ap.add_argument("--set", dest="case_set")
    ap.add_argument("--cases", default=str(Path(__file__).parent / "cases.json"))
    args = ap.parse_args(argv)

    cases = json.load(open(args.cases))
    if args.case_set:
        cases = [c for c in cases if c.get("set") == args.case_set]
    if args.ids:
        cases = [c for c in cases if c["id"] in args.ids]

    _ensure_knowledge_base()
    # Mirror app.py: the server loads the embedding model before serving.
    from rag.embeddings import warm_up
    warm_up()
    from llm import get_llm_info
    caregiver = Caregiver()
    meta = dict(label=args.label, driver=args.driver, concurrency=args.concurrency,
                llm=get_llm_info(), caregiver_model=caregiver.model,
                started=time.strftime("%Y-%m-%dT%H:%M:%S%z"), n_cases=len(cases))
    print(json.dumps(meta), flush=True)

    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=args.concurrency) as ex:
        results = list(ex.map(lambda c: run_case(c, args.driver, caregiver), cases))
    meta["suite_wall_s"] = round(time.perf_counter() - t0, 2)

    for r in results:
        print(f"{r['id']:40s} exp={r['expected']:9s} pred={str(r['predicted']):9s} "
              f"{'OK ' if r['correct'] else 'UNDER' if r['under_triage'] else 'OVER '} "
              f"wall={r['system_wall_s']:7.2f}s calls={r['llm_calls']:2d} {r['status']}", flush=True)
    summary = summarise(results, meta)
    out_dir = Path(__file__).parent / "results"
    out_dir.mkdir(exist_ok=True)
    json.dump(dict(summary=summary, results=results), open(out_dir / f"{args.label}.json", "w"), indent=1, default=str)
    print(json.dumps({k: v for k, v in summary.items() if k != "confusion"}, indent=1, default=str))
    return summary


if __name__ == "__main__":
    main()
