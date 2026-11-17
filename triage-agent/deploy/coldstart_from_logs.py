"""
deploy/coldstart_from_logs.py — instance cold-start times from Cloud Run logs.

For every instance of a Cloud Run service (optionally one revision) it pairs:
  t0  Cloud Run system log "Starting new instance"
  t1  the app's own "Running on local URL" line (Gradio is listening)
  t2  Cloud Run's "STARTUP TCP probe succeeded" (instance can take traffic)
and prints t1 - t0 and t2 - t0 in seconds. Nothing is estimated: every number
is the difference of two log timestamps written by Cloud Run or the app.

    python deploy/coldstart_from_logs.py --project P --service triage-ai-bench \
        [--revision triage-ai-bench-00002-abc] [--freshness 2h]
"""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
from collections import defaultdict
from datetime import datetime


def _ts(s: str) -> datetime:
    s = s.rstrip("Z")
    if "." in s:
        head, frac = s.split(".")
        s = f"{head}.{frac[:6]}"
    return datetime.fromisoformat(s)


def read_logs(project: str, service: str, revision: str | None, freshness: str) -> list[dict]:
    flt = f'resource.type="cloud_run_revision" AND resource.labels.service_name="{service}"'
    if revision:
        flt += f' AND resource.labels.revision_name="{revision}"'
    flt += (' AND (textPayload:"Starting new instance" OR textPayload:"Running on local URL"'
            ' OR textPayload:"STARTUP TCP probe succeeded")')
    cmd = ["gcloud", "logging", "read", flt, "--project", project, f"--freshness={freshness}",
           "--limit=1000", "--format=json"]
    out = subprocess.run(cmd, capture_output=True, text=True, check=True, shell=sys.platform == "win32").stdout
    return json.loads(out or "[]")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--project", required=True)
    ap.add_argument("--service", required=True)
    ap.add_argument("--revision")
    ap.add_argument("--freshness", default="2h")
    a = ap.parse_args()

    by_instance: dict[str, dict] = defaultdict(dict)
    for e in read_logs(a.project, a.service, a.revision, a.freshness):
        inst = (e.get("labels") or {}).get("instanceId")
        text = e.get("textPayload") or ""
        if not inst:
            continue
        rev = e["resource"]["labels"].get("revision_name")
        rec = by_instance[inst]
        rec["revision"] = rev
        t = _ts(e["timestamp"])
        if "Starting new instance" in text:
            rec["start"] = min(t, rec.get("start", t))
            rec["reason"] = text.split("Reason:")[-1].strip()[:60] if "Reason:" in text else ""
        elif "Running on local URL" in text:
            rec["listening"] = min(t, rec.get("listening", t))
        elif "STARTUP TCP probe succeeded" in text:
            rec["probe_ok"] = min(t, rec.get("probe_ok", t))

    rows = []
    for inst, r in by_instance.items():
        if "start" in r and ("listening" in r or "probe_ok" in r):
            listen = (r["listening"] - r["start"]).total_seconds() if "listening" in r else None
            probe = (r["probe_ok"] - r["start"]).total_seconds() if "probe_ok" in r else None
            rows.append((r["start"], r["revision"], inst[-8:], listen, probe, r.get("reason", "")))
    rows.sort()
    print("start (UTC)                 revision                         instance  listening_s  probe_ok_s  reason")
    for start, rev, inst, listen, probe, reason in rows:
        f = lambda v: f"{v:10.2f}" if v is not None else "         -"
        print(f"{start.isoformat()[:23]:27} {rev:32} {inst}  {f(listen)}  {f(probe)}  {reason}")
    probes = [r[4] for r in rows if r[4] is not None]
    if probes:
        print(f"\nn={len(probes)} probe_ok_s median={statistics.median(probes):.2f} "
              f"min={min(probes):.2f} max={max(probes):.2f}")


if __name__ == "__main__":
    main()
