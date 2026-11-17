"""
deploy/multiinstance_check.py — does one conversation survive being served by
different Cloud Run instances?

Each turn of one conversation (one thread_id) is sent through a DIFFERENT
gradio_client.Client. Every Client has its own cookie jar, so Cloud Run's
session-affinity cookie can pin each one to a different instance. Afterwards
the script reads Cloud Run's request logs and prints which instance served
each turn. If the turns ran on more than one instance and the conversation
still advanced (each answer builds on the previous ones, final restore shows
every message), the state was shared through the Postgres checkpointer.

    python deploy/multiinstance_check.py --url https://SERVICE.run.app \
        --project P --service S [--token "$(gcloud auth print-identity-token)"]

Uses case A of eval/cases.json (3-year-old, fast breathing, chest indrawing).
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
import uuid
from datetime import datetime, timezone

TURNS = [
    "3 year old child, cough for 4 days, breathing seems fast",
    "She can drink, no vomiting, no fits, she is awake and alert.",
    "50 breaths per minute, the lower chest pulls in, no harsh noise.",
]


def now() -> datetime:
    return datetime.now(timezone.utc)


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True, action="append",
                    help="repeat to rotate turns across URLs (e.g. two tagged revisions)")
    ap.add_argument("--project", required=True)
    ap.add_argument("--service", required=True)
    ap.add_argument("--token", default="", help="identity token for a private service")
    a = ap.parse_args()

    from gradio_client import Client

    headers = {"Authorization": f"Bearer {a.token}"} if a.token else None
    tid = f"multiinstance-{uuid.uuid4().hex[:12]}"
    history: list = []
    calls = []
    for i, msg in enumerate(TURNS + [None]):
        url = a.url[i % len(a.url)]
        client = Client(url, headers=headers, verbose=False)  # fresh cookie jar each time
        t0 = now()
        tic = time.perf_counter()
        if msg is None:
            out = client.predict(tid, api_name="/restore_session")
            history = out[0]
            label = "restore"
            card = out[3] or ""
            print("restored triage card:", next((l.strip("# *") for l in card.splitlines() if l.strip()), "(none)"))
        else:
            out = client.predict(msg, history, tid, api_name="/on_send")
            history = out[0]
            label = f"turn {i + 1}"
        dt = time.perf_counter() - tic
        calls.append({"label": label, "url": url.split("//")[1].split("---")[0], "start": t0, "end": now(), "seconds": round(dt, 2),
                      "messages": len(history), "last": (history[-1]["content"] if history else "")[:110],
                      "status": out[2] if msg is not None else out[2]})
        print(f"{label:8} {dt:6.2f}s messages={len(history)} | {calls[-1]['last']}")

    time.sleep(20)  # request logs arrive with a short delay
    flt = (f'resource.type="cloud_run_revision" AND resource.labels.service_name="{a.service}" '
           f'AND httpRequest.requestMethod="POST" AND httpRequest.requestUrl:"queue/join" '
           f'AND timestamp>="{calls[0]["start"].isoformat()}"')
    out = subprocess.run(["gcloud", "logging", "read", flt, "--project", a.project, "--limit=200",
                          "--format=json"], capture_output=True, text=True, check=True,
                         shell=sys.platform == "win32").stdout
    joins = sorted(((datetime.fromisoformat(e["timestamp"].replace("Z", "+00:00")[:26] + "+00:00"),
                     e["labels"].get("instanceId", "")[-8:]) for e in json.loads(out or "[]")))
    print("\nturn      instance  seconds  messages_after")
    instances = set()
    for c in calls:
        inst = [i for t, i in joins if c["start"] <= t <= c["end"]]
        instances.update(inst)
        print(f"{c['label']:8}  {','.join(inst) or '?':8}  {c['seconds']:7.2f}  {c['messages']:3}  via {c['url']}")
    print(f"\ndistinct instances: {len(instances)}  final messages restored: {calls[-1]['messages']}")
    print(f"final status: {calls[-1]['status'][:80] if isinstance(calls[-1]['status'], str) else calls[-1]['status']}")


if __name__ == "__main__":
    main()
