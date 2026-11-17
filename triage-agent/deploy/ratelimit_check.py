"""
deploy/ratelimit_check.py — check the rate limits on a deployed revision.

Run it against a revision deployed with SMALL limits, e.g.
    RATE_LIMIT_SESSION=2/600  RATE_LIMIT_IP=3/3600
(a tagged, no-traffic revision, so the public URL is unaffected):

    python deploy/ratelimit_check.py --url https://TAG---SERVICE.run.app \
        --session-limit 2 --ip-limit 3

Checks, in order (only 3 messages reach the LLM):
  1. a message over MAX_MESSAGE_CHARS is refused ("too long"), costs nothing
  2. session limit: messages 1..N of one conversation pass, N+1 is refused
  3. IP limit: new conversations from the same IP pass until the IP limit,
     the next one is refused ("network address")
  4. spoofing: a new conversation that sends its own X-Forwarded-For header
     must STILL be refused, i.e. the limiter does not key on a client-supplied
     header entry
"""

from __future__ import annotations

import argparse
import random
import sys
import uuid


def last_bot(history) -> str:
    for m in reversed(history or []):
        if m.get("role") == "assistant":
            return str(m.get("content"))
    return ""


def send(url: str, msg: str, tid: str, headers=None):
    from gradio_client import Client

    c = Client(url, headers=headers, verbose=False)
    out = c.predict(msg, [], tid, api_name="/on_send")
    return last_bot(out[0])


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True)
    ap.add_argument("--session-limit", type=int, required=True)
    ap.add_argument("--ip-limit", type=int, required=True)
    ap.add_argument("--max-chars", type=int, default=2000)
    a = ap.parse_args()
    results = []

    def check(name, ok, detail):
        results.append(ok)
        print(f"[{'PASS' if ok else 'FAIL'}] {name}: {detail[:150]}")

    # 1. message length (refused before any limit is charged)
    r = send(a.url, "x" * (a.max_chars + 1), f"rl-long-{uuid.uuid4().hex[:8]}")
    check("over-length message refused", "too long" in r.lower(), r)

    # 2. session limit
    tid = f"rl-session-{uuid.uuid4().hex[:8]}"
    msgs = ["4 year old girl, mild cough for 2 days", "No fever, eating fine, no danger signs."]
    sent = 0
    for i in range(a.session_limit):
        r = send(a.url, msgs[i % len(msgs)], tid)
        sent += 1
        check(f"session message {i + 1} processed", "rate limit" not in r.lower(), r)
    r = send(a.url, "one more message", tid)
    check(f"session message {a.session_limit + 1} refused", "rate limit reached" in r.lower()
          and "conversation" in r.lower(), r)

    # 3. IP limit with new conversations (the refused session message did not charge the IP)
    while sent < a.ip_limit:
        r = send(a.url, "55 year old man, headache for one week", f"rl-ip-{uuid.uuid4().hex[:8]}")
        sent += 1
        check(f"IP message {sent} (new conversation) processed", "rate limit" not in r.lower(), r)
    r = send(a.url, "55 year old man, headache for one week", f"rl-ip-{uuid.uuid4().hex[:8]}")
    check(f"IP message {a.ip_limit + 1} refused", "rate limit reached" in r.lower()
          and "network address" in r.lower(), r)

    # 4. spoofed X-Forwarded-For must not get a fresh IP bucket
    fake = f"203.0.113.{random.randint(1, 254)}"
    r = send(a.url, "55 year old man, headache for one week", f"rl-spoof-{uuid.uuid4().hex[:8]}",
             headers={"X-Forwarded-For": fake})
    check(f"spoofed X-Forwarded-For {fake} still refused", "rate limit reached" in r.lower(), r)

    print(f"\n{sum(results)}/{len(results)} checks passed")
    sys.exit(0 if all(results) else 1)


if __name__ == "__main__":
    main()
