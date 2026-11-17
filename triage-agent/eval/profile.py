"""
eval/profile.py — Where does the time go? Built from a result file's recorded
per-node and per-LLM-call timings (no new measurements, no estimates).

Usage:  python -m eval.profile eval/results/<label>.json
"""

from __future__ import annotations

import json
import statistics
import sys
from collections import defaultdict


def profile(path: str) -> str:
    d = json.load(open(path))
    results = d["results"]
    out = [f"Profile of {d['summary']['meta']['label']} ({len(results)} cases)", ""]

    # Share of system wall time per node, summed over all cases
    node_total = defaultdict(float)
    node_runs = defaultdict(list)
    wall_total = sum(r["system_wall_s"] for r in results)
    for r in results:
        for name, secs in r["node_times"]:
            node_total[name] += secs
            node_runs[name].append(secs)
    out.append(f"{'node':12s} {'runs':>5s} {'median s':>9s} {'sum s':>9s} {'% of wall':>9s}")
    for name in sorted(node_total, key=node_total.get, reverse=True):
        runs = node_runs[name]
        out.append(f"{name:12s} {len(runs):5d} {statistics.median(runs):9.2f} {node_total[name]:9.1f} "
                   f"{100 * node_total[name] / wall_total:8.1f}%")
    out.append(f"{'(all nodes)':12s} {'':5s} {'':9s} {sum(node_total.values()):9.1f} "
               f"{100 * sum(node_total.values()) / wall_total:8.1f}%   system wall total {wall_total:.1f} s")
    out.append("")

    # LLM calls by (node, model)
    calls = [c for r in results for c in r["calls"] if "error" not in c]
    llm_total = sum(c["latency_s"] for c in calls)
    groups = defaultdict(list)
    for c in calls:
        groups[(c.get("node"), c.get("model"))].append(c)
    out.append(f"{'node':10s} {'model':24s} {'calls':>5s} {'median s':>9s} {'sum s':>8s} {'% LLM':>6s} "
               f"{'med in':>7s} {'med out':>7s} {'med reas':>8s}")
    for (node, model), cs in sorted(groups.items(), key=lambda kv: -sum(c["latency_s"] for c in kv[1])):
        lat = [c["latency_s"] for c in cs]
        out.append(f"{str(node):10s} {str(model):24s} {len(cs):5d} {statistics.median(lat):9.2f} {sum(lat):8.1f} "
                   f"{100 * sum(lat) / llm_total:5.1f}% {statistics.median(c['input_tokens'] for c in cs):7.0f} "
                   f"{statistics.median(c['output_tokens'] for c in cs):7.0f} "
                   f"{statistics.median(c['reasoning_tokens'] for c in cs):8.0f}")
    out.append(f"LLM time {llm_total:.1f} s = {100 * llm_total / wall_total:.1f}% of system wall time")
    out.append("")

    # The completing turn (retrieval + triage) vs interview turns
    turns = [r["intake_questions_asked"] for r in results]
    out.append(f"intake questions per case: median {statistics.median(turns)}, max {max(turns)}")
    return "\n".join(out)


if __name__ == "__main__":
    print(profile(sys.argv[1]))
