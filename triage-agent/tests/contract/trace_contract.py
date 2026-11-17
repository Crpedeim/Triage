"""
tests/contract/trace_contract.py — the eval platform's POST /api/v1/traces
contract, as written in tests/contract/TRACING_API.md (a copy of
docs/TRACING_API.md from the eval platform repo, main @ 2fe84bb).

contract_errors(payload) applies the document's field tables and span rules
and returns the problems with the document's own error messages (empty list =
the platform would accept the body). doc_examples() pulls the request
examples, the example responses and the error bodies out of the document
itself, so the tests follow the document rather than a hand copy.
"""

from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path

DOC = Path(__file__).with_name("TRACING_API.md")
MAX_SPANS = 500


def _is_int(v) -> bool:
    return isinstance(v, int) and not isinstance(v, bool)


def _str(v, lo=0, hi=None) -> bool:
    return isinstance(v, str) and len(v) >= lo and (hi is None or len(v) <= hi)


def _date(v) -> bool:
    if _is_int(v) or (isinstance(v, float) and v == v):
        return True
    if not isinstance(v, str):
        return False
    try:
        datetime.fromisoformat(v.replace("Z", "+00:00"))
        return True
    except ValueError:
        return False


def contract_errors(t) -> list[str]:
    if not isinstance(t, dict):
        return ["body must be a JSON object"]
    errs: list[str] = []
    # Trace fields
    if not _str(t.get("name"), 1):
        errs.append("name: required string, >=1 char")
    for f in ("input", "output"):
        if not _str(t.get(f)):
            errs.append(f"{f}: required string")
    if not _str(t.get("model"), 1):
        errs.append("model: required string, >=1 char")
    for f in ("promptTokens", "completionTokens", "latencyMs"):
        if f in t and not (_is_int(t[f]) and t[f] >= 0):
            errs.append(f"{f}: integer >= 0")
    if "status" in t and not isinstance(t["status"], str):
        errs.append("status: string")
    if "metadata" in t and not isinstance(t["metadata"], dict):
        errs.append("metadata: must be an object")
    spans = t.get("spans")
    if spans is None:
        return errs
    if not isinstance(spans, list):
        return errs + ["spans: array"]
    if len(spans) > MAX_SPANS:
        errs.append(f"spans: at most {MAX_SPANS}")
    # Span fields
    for i, s in enumerate(spans):
        p = f"spans[{i}]"
        if not isinstance(s, dict):
            errs.append(f"{p}: object"); continue
        for f, hi in (("id", 128), ("parentId", 128)):
            if f in s and not _str(s[f], 1, hi):
                errs.append(f"{p}.{f}: string, 1-{hi} chars")
        if not _str(s.get("name"), 1, 200):
            errs.append(f"{p}.name: required string, 1-200 chars")
        if "kind" in s and not _str(s["kind"], 1, 32):
            errs.append(f"{p}.kind: string, 1-32 chars")
        if "status" in s and s["status"] not in ("ok", "error"):
            errs.append(f"{p}.status: \"ok\" or \"error\"")
        if "model" in s and not _str(s["model"], 1):
            errs.append(f"{p}.model: string, >=1 char")
        if "startedAt" in s and not _date(s["startedAt"]):
            errs.append(f"{p}.startedAt: ISO 8601 string or epoch milliseconds")
        if not (_is_int(s.get("latencyMs")) and s["latencyMs"] >= 0):
            errs.append(f"{p}.latencyMs: required integer >= 0")
        for f in ("promptTokens", "completionTokens"):
            if f in s and not (_is_int(s[f]) and s[f] >= 0):
                errs.append(f"{p}.{f}: integer >= 0")
        if "metadata" in s and not isinstance(s["metadata"], dict):
            errs.append(f"{p}.metadata: object")
    if errs:
        return errs
    # Span rules (checked after the field types pass), with the document's messages
    ids: dict[str, int] = {}
    for s in spans:
        if "id" in s:
            if s["id"] in ids:
                errs.append(f'duplicate span id "{s["id"]}"')
            else:
                ids[s["id"]] = len(ids)
    by_id = {s["id"]: s for s in spans if "id" in s}
    for s in spans:
        if "parentId" not in s:
            continue
        if s["parentId"] not in by_id:
            errs.append(f'parentId "{s["parentId"]}" is not the id of a span in this request')
            continue
        seen = {s["id"]} if "id" in s else set()
        parent = s["parentId"]
        while parent is not None:
            if parent in seen:
                errs.append("span parents form a cycle")
                break
            seen.add(parent)
            parent = by_id.get(parent, {}).get("parentId")
    return errs


def _section(text: str, heading: str) -> str:
    start = text.index(heading)
    nxt = text.find("\n## ", start + len(heading))
    return text[start: nxt if nxt != -1 else len(text)]


def doc_examples() -> dict:
    """Request and response examples, verbatim from the document."""
    text = DOC.read_text(encoding="utf-8")
    multi = re.search(r"-d @- <<'JSON'\n(.*?)\nJSON\n", text, re.S).group(1)
    minimal = re.search(r"-d '(\{.*?\})'", _section(text, "## Minimal example"), re.S).group(1)
    example = _section(text, "## Example: a multi-agent run")
    response_202 = re.search(r"Response \(`202`\):\s*```json\n(.*?)\n```", example, re.S).group(1)
    example_422 = re.search(r"Example 422[^\n]*\n\s*```json\n(.*?)\n```", text, re.S).group(1)
    bodies = {}
    for code, body in re.findall(r"^\| `(\d{3})[^`]*` \| `(\{.*?\})`", _section(text, "## Responses"), re.M):
        try:
            bodies[int(code)] = json.loads(body)
        except ValueError:      # schematic rows ({"id": string, ...}, [...]) are not literal JSON
            pass
    cost = re.search(r"= `0\.0096 \+ 0\.0012 = ([0-9.]+)`", example)
    return {
        "multi_agent": json.loads(multi),
        "minimal": json.loads(minimal),
        "response_202": json.loads(response_202),
        "response_422": json.loads(example_422),
        "error_bodies": bodies,           # {400: {...}, 401: {...}, 500: {...}}
        "multi_agent_cost": float(cost.group(1)) if cost else None,
    }
