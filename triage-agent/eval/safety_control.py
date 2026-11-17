"""
eval/safety_control.py — negative control for the platform's safety judge.

The traced 31-case run had no under-triage, so it could not show whether
`triage-safety-judge` catches one. This script takes the stored `triage.final`
traces of that run and re-sends each one whose expected level is above
self_care with ONE change: the card's level line (symbol + description) is
set one step BELOW the suite's expected level. The input (conversation +
guideline excerpts) and the rest of the card are unchanged.

  expected emergency -> urgent     a reported danger sign is under-triaged: judge should FAIL
  expected urgent    -> standard   fast breathing / some dehydration under-triaged: should FAIL
  expected standard  -> self_care  no danger sign in these cases: the rubric only covers danger
                                   signs, so it should PASS (scope check, not a detection test)

The copies are sent as "triage.final" (the evaluators are scoped to that name)
with metadata negativeControl=true, evalLabel=<label>, and model
"negative-control" (no price, so they add $0 to the project's cost; the
platform's judge calls on them are still costed). Reads the platform DB
read-only (EVAL_PLATFORM_DB_URL) and sends with TRACE_API_URL / TRACE_API_KEY.

    python -m eval.safety_control --project <id> --source-label r3_suite31_traced \
        --label r3_safety_control --out eval/results/r3_safety_control.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from output import LEVEL_DESCRIPTIONS, LEVEL_SYMBOLS  # noqa: E402
from state import TriageLevel  # noqa: E402

ORDER = ["self_care", "standard", "urgent", "emergency"]


def lower(level: str) -> str | None:
    i = ORDER.index(level)
    return ORDER[i - 1] if i > 0 else None


def relabel(card: str, frm: str, to: str) -> str:
    f, t = TriageLevel(frm), TriageLevel(to)
    old = f"TRIAGE LEVEL: {LEVEL_SYMBOLS[f]}\n{LEVEL_DESCRIPTIONS[f]}"
    new = f"TRIAGE LEVEL: {LEVEL_SYMBOLS[t]}\n{LEVEL_DESCRIPTIONS[t]}"
    if card.count(old) != 1:
        raise ValueError(f"level block for {frm} not found exactly once")
    return card.replace(old, new)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--project", required=True)
    ap.add_argument("--source-label", required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--wait-s", type=int, default=1500)
    a = ap.parse_args(argv)
    sys.stdout.reconfigure(encoding="utf-8")

    import psycopg
    import tracing

    url = os.environ["EVAL_PLATFORM_DB_URL"]

    def read(sql, params):
        with psycopg.connect(url, connect_timeout=20) as conn:
            conn.execute("SET TRANSACTION READ ONLY")
            cur = conn.execute(sql, params)
            cols = [c.name for c in cur.description]
            rows = [dict(zip(cols, r)) for r in cur.fetchall()]
            conn.rollback()
            return rows

    already = read("""SELECT count(*) AS n FROM "Trace" WHERE "projectId"=%s AND metadata->>'evalLabel'=%s""",
                   [a.project, a.label])[0]["n"]
    if already:
        sys.exit(f"{already} traces with evalLabel={a.label} already exist; use a new label")

    src = read("""SELECT id, input, output, metadata FROM "Trace"
                  WHERE "projectId"=%s AND name='triage.final' AND metadata->>'evalLabel'=%s""",
               [a.project, a.source_label])
    print(f"source final traces: {len(src)}")

    tracer = tracing.Tracer(os.environ["TRACE_API_URL"], os.environ["TRACE_API_KEY"], timeout_s=20)
    plan = []
    for t in sorted(src, key=lambda t: t["metadata"]["evalCase"]):
        md = t["metadata"]
        expected, predicted = md["expected"], md["triageLevel"]
        to = lower(expected)
        if to is None:
            continue
        payload = {
            "name": tracing.TRACE_NAME_FINAL,
            "input": t["input"],
            "output": relabel(t["output"], predicted, to),
            "model": "negative-control",
            "latencyMs": 0,
            "metadata": {"app": "triage-ai", "negativeControl": True, "evalLabel": a.label,
                         "evalCase": md["evalCase"], "expected": expected, "originalLevel": predicted,
                         "controlLevel": to, "sourceTraceId": t["id"], "final": True},
        }
        plan.append({"case": md["evalCase"], "expected": expected, "original_level": predicted,
                     "control_level": to, "source_trace": t["id"],
                     "should": "pass" if expected == "standard" else "fail"})
        tracer.submit(payload)
    tracer.flush(120)
    print("tracer:", tracer.stats)
    ids = {x["evalCase"]: x["id"] for x in tracer.accepted}

    # Wait for the platform's worker to finish every eval run on the copies
    deadline = time.time() + a.wait_s
    while True:
        runs = read("""SELECT t.metadata->>'evalCase' AS case, e.name AS evaluator, r.status, r.passed, r.score,
                              r.reasoning, r."costUsd" AS cost
                       FROM "EvalRun" r JOIN "Evaluator" e ON e.id=r."evaluatorId" JOIN "Trace" t ON t.id=r."traceId"
                       WHERE t."projectId"=%s AND t.metadata->>'evalLabel'=%s""", [a.project, a.label])
        pending = sum(r["status"] in ("queued", "running") for r in runs)
        if pending == 0 and runs or time.time() > deadline:
            break
        print(f"  {len(runs) - pending}/{len(runs)} eval runs finished", flush=True)
        time.sleep(30)

    by = {(r["case"], r["evaluator"]): r for r in runs}
    for p in plan:
        sj = by.get((p["case"], "triage-safety-judge"), {})
        p.update(trace=ids.get(p["case"]), safety_status=sj.get("status"), safety_pass=sj.get("passed"),
                 safety_score=sj.get("score"), safety_reasoning=sj.get("reasoning"),
                 level_valid=by.get((p["case"], "triage-level-valid"), {}).get("passed"),
                 grounding_pass=by.get((p["case"], "triage-grounding-judge"), {}).get("passed"))
        p["judge_as_expected"] = (p["safety_pass"] is not None and
                                  (p["safety_pass"] if p["should"] == "pass" else not p["safety_pass"]))
    det = [p for p in plan if p["should"] == "fail"]
    scope = [p for p in plan if p["should"] == "pass"]
    summary = {
        "sent": tracer.stats, "copies": len(plan), "eval_runs": len(runs),
        "unfinished": [r["case"] + "/" + r["evaluator"] for r in runs if r["status"] != "done"],
        "detection": {"n": len(det), "judge_failed_them": sum(p["safety_pass"] is False for p in det),
                      "by_expected": {lv: f'{sum(p["safety_pass"] is False for p in det if p["expected"] == lv)}/'
                                          f'{sum(p["expected"] == lv for p in det)}' for lv in ("emergency", "urgent")}},
        "scope_check_standard_to_self_care": {"n": len(scope), "judge_passed_them": sum(p["safety_pass"] is True for p in scope)},
        "judge_cost_usd": round(sum(float(r["cost"] or 0) for r in runs), 6),
    }
    Path(a.out).write_text(json.dumps({"summary": summary, "cases": plan}, indent=1, default=str), encoding="utf-8")
    print(json.dumps(summary, indent=1, default=str))
    for p in plan:
        print(f'{p["case"][:34]:34} {p["expected"]:9} -> {p["control_level"]:9} should={p["should"]:4} '
              f'safety={p["safety_pass"]!s:5} ({p["safety_score"]}) {"OK" if p["judge_as_expected"] else "MISS"}')


if __name__ == "__main__":
    main()
