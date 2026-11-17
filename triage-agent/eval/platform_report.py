"""
eval/platform_report.py — read what the eval platform stored for a traced eval
run and compare it with the suite's own results.

Reads the platform's Postgres in a READ ONLY transaction, for one project
only. The connection string comes from EVAL_PLATFORM_DB_URL (never printed).

    python -m eval.platform_report --project <projectId> --results eval/results/<label>.json \
        [--label <evalLabel>] [--sessions-from-live <live.json>] [--out report.json]

Reports:
  delivery     traces sent by the client (Tracer.stats / 202 bodies in the result
               file) vs traces stored on the platform for this eval label
  agents       latency per agent span and per LLM span, as stored by the platform
  cost         platform-computed cost (Trace.costUsd) summed per conversation =
               TOTAL COST PER TRIAGE; split by model from Span.costUsd; judge
               cost (EvalRun.costUsd) reported separately
  evaluators   pass rate and score per evaluator, plus unfinished runs
  safety       triage-safety-judge verdict next to the suite's label and
               under-triage flag, case by case; every disagreement listed
"""

from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import sys
from collections import defaultdict
from decimal import Decimal
from pathlib import Path


def p95(xs):
    if not xs:
        return None
    s = sorted(xs)
    return s[max(0, math.ceil(0.95 * len(s)) - 1)]


def stats(xs):
    xs = [float(x) for x in xs if x is not None]
    if not xs:
        return {"n": 0}
    return {"n": len(xs), "median": round(statistics.median(xs), 6), "p95_nearest_rank": round(p95(xs), 6),
            "min": round(min(xs), 6), "max": round(max(xs), 6), "total": round(sum(xs), 6)}


def fetch(conn, sql, params):
    cur = conn.execute(sql, params)
    cols = [c.name for c in cur.description]
    out = []
    for row in cur.fetchall():
        out.append({k: (float(v) if isinstance(v, Decimal) else v) for k, v in zip(cols, row)})
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--project", required=True)
    ap.add_argument("--results", help="eval result file (run_eval.py) with evalCase ids")
    ap.add_argument("--label", help="metadata.evalLabel to select (default: the result file's label)")
    ap.add_argument("--sessions", nargs="*", default=[], help="select by metadata.sessionId instead")
    ap.add_argument("--out")
    a = ap.parse_args(argv)
    sys.stdout.reconfigure(encoding="utf-8")

    import psycopg
    url = os.environ["EVAL_PLATFORM_DB_URL"]
    results = json.load(open(a.results, encoding="utf-8")) if a.results else None
    label = a.label or (results["summary"]["meta"]["label"] if results else None)

    with psycopg.connect(url, connect_timeout=20) as conn:
        conn.execute("SET TRANSACTION READ ONLY")
        if a.sessions:
            where, params = """t."projectId" = %s AND t.metadata->>'sessionId' = ANY(%s)""", [a.project, a.sessions]
        else:
            where, params = """t."projectId" = %s AND t.metadata->>'evalLabel' = %s""", [a.project, label]
        traces = fetch(conn, f"""
            SELECT t.id, t.name, t.model, t."promptTokens", t."completionTokens", t."costUsd", t."latencyMs",
                   t.status, t."createdAt", t.metadata->>'sessionId' AS session, (t.metadata->>'turn')::int AS turn,
                   t.metadata->>'evalCase' AS case_id, t.metadata->>'triageLevel' AS level,
                   t.metadata->>'expected' AS expected
            FROM "Trace" t WHERE {where} ORDER BY t."createdAt" """, params)
        ids = [t["id"] for t in traces]
        spans = fetch(conn, """
            SELECT s."traceId", s.name, s.kind, s.status, s.model, s."latencyMs", s."promptTokens",
                   s."completionTokens", s."costUsd", s.metadata->>'agent' AS agent
            FROM "Span" s WHERE s."traceId" = ANY(%s)""", [ids])
        runs = fetch(conn, """
            SELECT r."traceId", e.name AS evaluator, e.type, r.status, r.score, r.passed, r.reasoning, r.error,
                   r.model, r."costUsd", r."latencyMs", r.attempts
            FROM "EvalRun" r JOIN "Evaluator" e ON e.id = r."evaluatorId"
            WHERE r."traceId" = ANY(%s)""", [ids])
        conn.rollback()

    rep: dict = {"project": a.project, "label": label, "sessions": a.sessions or None}

    # ── delivery ─────────────────────────────────────────────────────────
    client = (results or {}).get("summary", {}).get("meta", {}).get("tracing") if results else None
    accepted_ids = {x.get("id") for x in (client or {}).get("accepted", []) if x.get("id")}
    rep["delivery"] = {
        "client_stats": (client or {}).get("stats"),
        "client_202_ids": len(accepted_ids),
        "platform_traces": len(traces),
        "platform_by_name": {n: sum(t["name"] == n for t in traces) for n in sorted({t["name"] for t in traces})},
        "accepted_but_missing_on_platform": sorted(accepted_ids - set(ids)) if client else None,
        "on_platform_not_in_client_202s": sorted(set(ids) - accepted_ids) if client else None,
        "platform_spans": len(spans),
    }

    # ── per-agent latency (platform spans) ──────────────────────────────
    agents = defaultdict(list)
    llm = defaultdict(list)
    for s in spans:
        if s["kind"] in ("agent", "router", "output"):
            agents[s["name"]].append(s["latencyMs"])
        elif s["kind"] == "llm":
            llm[f'{s["agent"]} / {s["model"]}'].append(s["latencyMs"])
    turn = defaultdict(list)
    for t in traces:
        turn[t["name"]].append(t["latencyMs"])
    rep["latency_ms"] = {
        "per_turn": {k: stats(v) for k, v in turn.items()},
        "agent_spans": {k: stats(v) for k, v in sorted(agents.items())},
        "llm_spans_by_agent_and_model": {k: stats(v) for k, v in sorted(llm.items())},
    }

    # ── cost per triage (platform-computed) ─────────────────────────────
    by_session = defaultdict(list)
    for t in traces:
        by_session[t["session"]].append(t)
    span_cost_by_session_model = defaultdict(lambda: defaultdict(float))
    trace_session = {t["id"]: t["session"] for t in traces}
    for s in spans:
        if s["model"]:
            span_cost_by_session_model[trace_session[s["traceId"]]][s["model"]] += s["costUsd"] or 0.0
    complete = {k: v for k, v in by_session.items() if any(t["name"] == "triage.final" for t in v)}
    per_triage = {k: sum(t["costUsd"] for t in v) for k, v in complete.items()}
    models = sorted({m for d in span_cost_by_session_model.values() for m in d})
    rep["cost_usd"] = {
        "per_triage_all_turns": stats(per_triage.values()),
        "per_triage_by_model": {m: stats([span_cost_by_session_model[k].get(m, 0.0) for k in complete]) for m in models},
        "per_trace_name": {n: stats([t["costUsd"] for t in traces if t["name"] == n]) for n in rep["delivery"]["platform_by_name"]},
        "triages": len(complete),
        "conversations_without_final_trace": sorted(k for k in by_session if k not in complete),
        "judge_cost_per_triage": stats([sum(r["costUsd"] or 0 for r in runs if trace_session[r["traceId"]] == k)
                                        for k in complete]),
        "tokens_per_triage": stats([sum(t["promptTokens"] + t["completionTokens"] for t in v) for v in complete.values()]),
    }

    # ── evaluators ───────────────────────────────────────────────────────
    ev = defaultdict(list)
    for r in runs:
        ev[r["evaluator"]].append(r)
    rep["evaluators"] = {
        name: {"runs": len(rs), "done": sum(r["status"] == "done" for r in rs),
               "unfinished_or_failed": {st: sum(r["status"] == st for r in rs) for st in ("queued", "running", "failed")},
               "passed": sum(bool(r["passed"]) for r in rs if r["status"] == "done"),
               "score": stats([r["score"] for r in rs if r["status"] == "done"]),
               "judge_latency_ms": stats([r["latencyMs"] for r in rs if r["latencyMs"] is not None])}
        for name, rs in sorted(ev.items())}

    # ── per case: suite vs platform judges ───────────────────────────────
    suite = {r["id"]: r for r in (results or {}).get("results", [])}
    finals = [t for t in traces if t["name"] == "triage.final"]
    runs_by_trace = defaultdict(dict)
    for r in runs:
        runs_by_trace[r["traceId"]][r["evaluator"]] = r
    rows = []
    for t in finals:
        s = suite.get(t["case_id"], {})
        jr = runs_by_trace[t["id"]]
        sj, gj, lv = jr.get("triage-safety-judge"), jr.get("triage-grounding-judge"), jr.get("triage-level-valid")
        rows.append({
            "case": t["case_id"], "session": t["session"], "trace": t["id"],
            "expected": s.get("expected", t["expected"]), "predicted": s.get("predicted", t["level"]),
            "trace_level": t["level"], "suite_correct": s.get("correct"), "suite_under_triage": s.get("under_triage"),
            "suite_over_triage": s.get("over_triage"),
            "safety_pass": sj["passed"] if sj and sj["status"] == "done" else None,
            "safety_score": sj["score"] if sj else None,
            "safety_reasoning": (sj or {}).get("reasoning"),
            "grounding_pass": gj["passed"] if gj and gj["status"] == "done" else None,
            "grounding_score": gj["score"] if gj else None,
            "grounding_reasoning": (gj or {}).get("reasoning"),
            "level_valid": lv["passed"] if lv and lv["status"] == "done" else None,
            "cost_usd_all_turns": per_triage.get(t["session"]),
        })
    rows.sort(key=lambda r: str(r["case"]))
    judged = [r for r in rows if r["safety_pass"] is not None and r["suite_under_triage"] is not None]
    # Agreement: the judge says "unsafe" exactly when the suite says under-triage
    agree = [r for r in judged if (not r["safety_pass"]) == bool(r["suite_under_triage"])]
    rep["safety_vs_suite"] = {
        "judged": len(judged),
        "agree": len(agree),
        "matrix": {
            "suite_under_and_judge_fail": sum(r["suite_under_triage"] and not r["safety_pass"] for r in judged),
            "suite_under_and_judge_pass": sum(r["suite_under_triage"] and r["safety_pass"] for r in judged),
            "suite_ok_and_judge_fail": sum((not r["suite_under_triage"]) and not r["safety_pass"] for r in judged),
            "suite_ok_and_judge_pass": sum((not r["suite_under_triage"]) and r["safety_pass"] for r in judged),
        },
        "disagreements": [{k: r[k] for k in ("case", "expected", "predicted", "suite_under_triage", "safety_score",
                                             "safety_reasoning")} for r in judged if r not in agree],
        "cases_not_judged": [r["case"] for r in rows if r not in judged],
        "suite_cases_without_final_trace": sorted(set(suite) - {r["case"] for r in rows}) if suite else None,
    }
    rep["cases"] = rows

    text = json.dumps(rep, indent=1, default=str)
    if a.out:
        Path(a.out).write_text(text, encoding="utf-8")
    brief = {k: v for k, v in rep.items() if k != "cases"}
    print(json.dumps(brief, indent=1, default=str))
    print("\ncase                          exp        pred       under  safety  ground  level  cost$")
    for r in rows:
        print(f'{str(r["case"])[:29]:29} {str(r["expected"]):10} {str(r["predicted"]):10} {str(r["suite_under_triage"]):6} '
              f'{str(r["safety_pass"]):7} {str(r["grounding_pass"]):7} {str(r["level_valid"]):6} {r["cost_usd_all_turns"]}')


if __name__ == "__main__":
    main()
