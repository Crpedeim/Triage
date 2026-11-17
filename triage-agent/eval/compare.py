"""
eval/compare.py — Side-by-side comparison of two eval result files.

Usage:
  python -m eval.compare eval/results/BEFORE.json eval/results/AFTER.json

Prints headline metrics for both runs, then every case whose predicted level
changed. Only numbers present in the result files are shown; nothing is
estimated. Cases are matched by id; cases present in only one file are listed.
"""

from __future__ import annotations

import json
import sys


def _load(path):
    d = json.load(open(path))
    return d["summary"], {r["id"]: r for r in d["results"]}


def _fmt_rate(x):
    return f"{x['count'] if 'count' in x else x['correct']}/{x['n']}"


def compare(before_path: str, after_path: str) -> str:
    sb, rb = _load(before_path)
    sa, ra = _load(after_path)
    common = sorted(set(rb) & set(ra))
    out = []
    out.append(f"{'metric':38s} {'before':>22s} {'after':>22s}")
    rows = [
        ("label", sb["meta"]["label"], sa["meta"]["label"]),
        ("cases", sb["n"], sa["n"]),
        ("accuracy", _fmt_rate(sb["accuracy"]), _fmt_rate(sa["accuracy"])),
        ("under-triage", _fmt_rate(sb["under_triage"]), _fmt_rate(sa["under_triage"])),
        ("over-triage", _fmt_rate(sb["over_triage"]), _fmt_rate(sa["over_triage"])),
        ("no result / errors", f"{sb['no_result']} / {len(sb['errors'])}", f"{sa['no_result']} / {len(sa['errors'])}"),
        ("latency median s", sb["latency_s"]["median"], sa["latency_s"]["median"]),
        ("latency p95 s (nearest rank)", sb["latency_s"]["p95_nearest_rank"], sa["latency_s"]["p95_nearest_rank"]),
        ("tokens/triage median", sb["tokens_per_triage"]["median"], sa["tokens_per_triage"]["median"]),
        ("LLM calls/triage median", sb["llm_calls_per_triage"]["median"], sa["llm_calls_per_triage"]["median"]),
        ("priced cost median low $", sb["priced_cost_per_triage_usd"]["low_median"], sa["priced_cost_per_triage_usd"]["low_median"]),
        ("priced cost median high $", sb["priced_cost_per_triage_usd"]["high_median"], sa["priced_cost_per_triage_usd"]["high_median"]),
        ("concurrency", sb["meta"]["concurrency"], sa["meta"]["concurrency"]),
    ]
    for name, b, a in rows:
        out.append(f"{name:38s} {str(b):>22s} {str(a):>22s}")
    out.append("")
    out.append("Changed predictions (id: expected | before -> after):")
    for cid in common:
        b, a = rb[cid], ra[cid]
        if b["predicted"] != a["predicted"]:
            tag = "FIXED" if a["correct"] and not b["correct"] else "REGRESSED" if b["correct"] and not a["correct"] else "changed"
            out.append(f"  {cid}: {a['expected']} | {b['predicted']} -> {a['predicted']}  [{tag}]")
    only_b, only_a = sorted(set(rb) - set(ra)), sorted(set(ra) - set(rb))
    if only_b:
        out.append(f"Only in before: {only_b}")
    if only_a:
        out.append(f"Only in after: {only_a}")
    return "\n".join(out)


if __name__ == "__main__":
    print(compare(sys.argv[1], sys.argv[2]))
