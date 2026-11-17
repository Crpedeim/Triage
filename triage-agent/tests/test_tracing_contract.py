"""
tests/test_tracing_contract.py — tracing.py against the platform contract
(tests/contract/TRACING_API.md, a copy of the eval platform's docs/TRACING_API.md).

1. The document's own request examples pass the contract checker, and its
   422 example (a span naming a missing parent) is reproduced word for word:
   the checker is faithful to the document.
2. Every payload tracing.py sends for a real (stubbed-LLM) conversation passes
   the same checker.
3. The sender handles each documented response: the 202 body, and the 400 /
   401 / 422 / 500 error bodies (500 is "safe to retry"; the others are not).
4. Optional, when EVAL_PLATFORM_REPO points at a checkout of the platform with
   node_modules: the same payloads parsed by the platform's real zod schema
   (src/lib/trace-input.ts), and its server-side cost for the doc's example.
"""

from __future__ import annotations

import copy
import json
import logging
import os
import shutil
import subprocess
import sys
from datetime import date
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

import tracing  # noqa: E402
from tests.contract.trace_contract import contract_errors, doc_examples  # noqa: E402
from tests.test_tracing import FakePlatform, platform, two_turns  # noqa: E402,F401

EX = doc_examples()


# ── 1. the checker matches the document ─────────────────────────────────
def test_doc_request_examples_are_valid():
    assert contract_errors(EX["multi_agent"]) == []
    assert contract_errors(EX["minimal"]) == []
    assert len(EX["multi_agent"]["spans"]) == 5


def test_doc_422_example_is_reproduced():
    bad = copy.deepcopy(EX["multi_agent"])
    bad["spans"][1]["parentId"] = "plannr"      # the typo in the document's example
    assert contract_errors(bad) == EX["response_422"]["details"]["fieldErrors"]["spans"]


def test_span_rules_from_the_document():
    t = copy.deepcopy(EX["multi_agent"])
    t["spans"][2]["id"] = "plan"
    assert 'duplicate span id "plan"' in contract_errors(t)
    t = copy.deepcopy(EX["multi_agent"])
    t["spans"][0]["parentId"] = "classify"       # root -> classify -> root
    assert contract_errors(t) == ["span parents form a cycle"] * 5
    t = copy.deepcopy(EX["minimal"])
    t["spans"] = [{"name": "x", "latencyMs": 1, "status": "degraded"}]
    assert contract_errors(t) == ['spans[0].status: "ok" or "error"']
    t = copy.deepcopy(EX["minimal"]); t["latencyMs"] = 1.5
    assert contract_errors(t) == ["latencyMs: integer >= 0"]
    t = copy.deepcopy(EX["minimal"]); del t["model"]
    assert contract_errors(t) == ["model: required string, >=1 char"]


# ── 2. what tracing.py sends ─────────────────────────────────────────────
def _payloads(platform):
    two_turns(trace_metadata={"evalCase": "A_severe_pneumonia"})
    return [r["json"] for r in platform.requests]


def test_sent_payloads_meet_the_contract(platform):
    payloads = _payloads(platform)
    assert len(payloads) == 2
    for p in payloads:
        assert contract_errors(p) == []
        # TriageAI's own conventions on top of the contract
        assert p["spans"][0]["id"] == "turn"
        assert all(s.get("parentId") for s in p["spans"][1:])
        assert all("costUsd" not in s for s in p["spans"]) and "costUsd" not in p
        assert all(isinstance(v, (str, int, float, bool, list, dict, type(None))) for v in p["metadata"].values())
        assert len(p["spans"]) <= 500


def test_span_cap_is_respected():
    turn = tracing.TurnTrace("s", "hi")
    for i in range(600):
        span = turn._span("llm", "llm", model="gemini-3.8-flash", _agent="intake")
        span.update(latencyMs=1, promptTokens=1, completionTokens=1)
        turn.spans.append(span)
    p = turn.build_payload("ok", {})
    assert len(p["spans"]) == 500 and p["metadata"]["spansDropped"] == 101
    assert contract_errors(p) == []


# ── 3. the documented responses ──────────────────────────────────────────
def _send_one(responses, payload=None):
    p = FakePlatform(responses=responses)
    tr = tracing.Tracer(p.url, "evp_live_test", timeout_s=5, backoff_s=0.01)
    tr.submit(payload or EX["minimal"])
    assert tr.flush(10)
    p.close()
    return tr, p


def test_202_body_recorded():
    tr, p = _send_one([(202, EX["response_202"])], EX["multi_agent"])
    assert len(p.requests) == 1 and tr.stats["sent"] == 1 and tr.stats["failed"] == 0
    got = tr.accepted[0]
    assert got["id"] == EX["response_202"]["id"] and got["costUsd"] == EX["response_202"]["costUsd"]
    assert got["spans"] == 5 and got["name"] == "triage.run"


@pytest.mark.parametrize("status", [400, 401, 422])
def test_client_errors_are_logged_and_not_retried(status, caplog):
    body = EX["response_422"] if status == 422 else EX["error_bodies"][status]
    with caplog.at_level(logging.WARNING, logger="triage.tracing"):
        tr, p = _send_one([(status, body)])
    assert len(p.requests) == 1 and tr.stats == {**tr.stats, "sent": 0, "failed": 1, "retries": 0}
    assert f"HTTP {status}" in caplog.text and body["error"] in caplog.text


def test_500_is_retried_then_counted_failed():
    tr, p = _send_one([(500, EX["error_bodies"][500])])
    assert len(p.requests) == 3 and tr.stats["failed"] == 1 and tr.stats["retries"] == 2


def test_500_then_202_delivers_once():
    tr, p = _send_one([(500, EX["error_bodies"][500]), (202, EX["response_202"])])
    assert len(p.requests) == 2 and tr.stats["sent"] == 1 and tr.stats["failed"] == 0


def test_error_bodies_found_in_document():
    assert EX["error_bodies"][400] == {"error": "invalid json"}
    assert EX["error_bodies"][401] == {"error": "invalid api key"}
    assert EX["error_bodies"][500] == {"error": "internal error"}


# ── 4. the platform's own schema (optional) ──────────────────────────────
PLATFORM_REPO = os.getenv("EVAL_PLATFORM_REPO")


@pytest.mark.skipif(not PLATFORM_REPO, reason="EVAL_PLATFORM_REPO not set")
def test_platform_zod_schema_accepts_our_payloads(platform, tmp_path):
    npx = shutil.which("npx")
    assert npx, "npx not found"
    payloads = _payloads(platform) + [EX["multi_agent"]]
    f = tmp_path / "payloads.json"
    f.write_text(json.dumps(payloads), encoding="utf-8")
    script = Path(__file__).parent / "contract" / "validate_with_platform.ts"
    r = subprocess.run([npx, "tsx", str(script), str(f)], cwd=PLATFORM_REPO, capture_output=True,
                       text=True, timeout=120, shell=sys.platform == "win32")
    assert r.returncode == 0, r.stderr
    out = json.loads(r.stdout.strip().splitlines()[-1])
    assert [o["ok"] for o in out] == [True] * len(payloads), out
    # stub model gemini-3.1-pro-preview: 3 calls x (100 in, 50 out) on the final turn
    assert out[1]["costUsd"] == pytest.approx(3 * (100 * 2.0 + 50 * 12.0) / 1e6)
    if date.today() < date(2027, 1, 1):          # the doc's flash price changes on 2027-01-01
        assert out[2]["costUsd"] == pytest.approx(EX["multi_agent_cost"])
