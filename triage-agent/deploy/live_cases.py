"""
deploy/live_cases.py — run eval cases through a DEPLOYED TriageAI URL.

Each case from eval/cases.json is a new conversation (new thread id) sent
through the app's public Gradio API (/on_send), exactly what the browser
calls. The caregiver is scripted, not an LLM: the opening line, then every
question is answered with the case's hidden vignette text. Records the wall
time of every turn (client side, including network) and the final triage
level from the card.

    python deploy/live_cases.py --url https://SERVICE.run.app --ids A_severe_pneumonia ...
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
import uuid
from pathlib import Path

LEVELS = {"EMERGENCY": "emergency", "URGENT": "urgent", "STANDARD": "standard", "SELF-CARE": "self_care",
          "SELF CARE": "self_care", "SELF_CARE": "self_care"}


def card_level(card: str) -> str | None:
    head = next((l for l in (card or "").splitlines() if l.strip()), "").upper()
    for k, v in LEVELS.items():
        if k in head:
            return v
    return None


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True)
    ap.add_argument("--ids", nargs="+", required=True)
    ap.add_argument("--max-turns", type=int, default=6)
    ap.add_argument("--out", default="")
    a = ap.parse_args()

    from gradio_client import Client

    cases = json.loads((Path(__file__).parent.parent / "eval" / "cases.json").read_text(encoding="utf-8"))
    cases = {c["id"]: c for c in (cases["cases"] if isinstance(cases, dict) else cases)}
    client = Client(a.url, verbose=False)
    report = []
    for cid in a.ids:
        case = cases[cid]
        tid = f"live-{cid[:12]}-{uuid.uuid4().hex[:8]}"
        history, turns, level, card = [], [], None, ""
        msg = case["opening"]
        for _ in range(a.max_turns):
            t = time.perf_counter()
            out = client.predict(msg, history, tid, api_name="/on_send")
            turns.append(round(time.perf_counter() - t, 2))
            history, card = out[0], out[3] or ""
            level = card_level(card)
            if level:
                break
            msg = case["vignette"]
        cited = [l.strip("-* ") for l in card.split("Based on:")[-1].splitlines()
                 if l.strip("-* ").startswith(("WHO", "curated"))][:3] if "Based on:" in card else []
        row = {"id": cid, "expected": case["expected"], "level": level, "correct": level == case["expected"],
               "turn_seconds": turns, "total_seconds": round(sum(turns), 2), "cited": cited}
        report.append(row)
        print(f"{cid:28} expected={case['expected']:9} got={level!s:9} {'OK ' if row['correct'] else 'MISS'} "
              f"turns={turns} total={row['total_seconds']}s")
        for c in cited:
            print(f"    cites: {c}")
    all_turns = [t for r in report for t in r["turn_seconds"]]
    print(f"\n{sum(r['correct'] for r in report)}/{len(report)} correct; per-turn seconds: n={len(all_turns)} "
          f"median={statistics.median(all_turns):.2f} min={min(all_turns):.2f} max={max(all_turns):.2f}")
    if a.out:
        Path(a.out).write_text(json.dumps({"url": a.url, "cases": report}, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
