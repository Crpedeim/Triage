"""
tests/test_adk_retrieval.py — Phase 7 ADK retrieval agent: adapter conversions (offline).

Skipped unless google-adk is installed (requirements-adk.txt).
"""

from __future__ import annotations

import base64
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

pytest.importorskip("google.adk", reason="google-adk not installed (requirements-adk.txt)")

from google.adk.models.llm_request import LlmRequest  # noqa: E402
from google.genai import types  # noqa: E402

from agents import adk_retrieval as ar  # noqa: E402


def _req(contents, system="be helpful", tools=None):
    return LlmRequest(contents=contents, config=types.GenerateContentConfig(system_instruction=system, tools=tools))


def test_messages_include_system_user_assistant_tool_calls_and_tool_results():
    sig = b"\x01\x02signature"
    contents = [
        types.Content(role="user", parts=[types.Part(text="3 year old with cough")]),
        types.Content(role="model", parts=[types.Part(
            function_call=types.FunctionCall(id="call_1", name="search_guidelines", args={"query": "pneumonia"}),
            thought_signature=sig)]),
        types.Content(role="user", parts=[types.Part(function_response=types.FunctionResponse(
            id="call_1", name="search_guidelines", response={"results": [{"id": "a"}]}))]),
    ]
    msgs = ar.to_openai_messages(_req(contents))
    assert [m["role"] for m in msgs] == ["system", "user", "assistant", "tool"]
    tc = msgs[2]["tool_calls"][0]
    assert tc["id"] == "call_1" and tc["function"]["name"] == "search_guidelines"
    assert json.loads(tc["function"]["arguments"]) == {"query": "pneumonia"}
    # Gemini 3 thought signature is sent back (base64), else HTTP 400
    assert base64.b64decode(tc["extra_content"]["google"]["thought_signature"]) == sig
    assert msgs[3]["tool_call_id"] == "call_1" and json.loads(msgs[3]["content"]) == {"results": [{"id": "a"}]}


def test_response_tool_call_keeps_thought_signature_and_usage():
    sig = base64.b64encode(b"sig-bytes").decode()
    tc = SimpleNamespace(id="call_9", function=SimpleNamespace(name="search_guidelines", arguments='{"query": "x"}'),
                         model_extra={"extra_content": {"google": {"thought_signature": sig}}})
    resp = SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=None, tool_calls=[tc]))],
                           usage=SimpleNamespace(prompt_tokens=10, completion_tokens=5, total_tokens=40))
    out = ar.from_openai_response(resp)
    part = out.content.parts[0]
    assert part.function_call.name == "search_guidelines" and part.function_call.args == {"query": "x"}
    assert part.thought_signature == b"sig-bytes"
    assert out.usage_metadata.total_token_count == 40
    # and it survives the round trip back into a request
    back = ar.to_openai_messages(_req([out.content]))
    assert back[1]["tool_calls"][0]["extra_content"]["google"]["thought_signature"] == sig


def test_tools_use_adk_json_schema():
    from google.adk.tools import FunctionTool

    def search_guidelines(query: str) -> dict:
        """Search.

        Args:
            query: the query.
        """
        return {}

    decl = FunctionTool(search_guidelines)._get_declaration()
    tools = ar.to_openai_tools(_req([], tools=[types.Tool(function_declarations=[decl])]))
    assert tools[0]["function"]["name"] == "search_guidelines"
    assert tools[0]["function"]["parameters"]["properties"]["query"]["type"] == "string"
    assert tools[0]["function"]["parameters"]["required"] == ["query"]


def test_parse_ids_json_fenced_and_fallback():
    a, b = "a" * 32, "b" * 32
    assert ar._parse_ids(f'["{a}", "{b}"]') == [a, b]
    assert ar._parse_ids(f'```json\n["{a}"]\n```') == [a]
    assert ar._parse_ids(f"The relevant ids are {a} and {b}.") == [a, b]
    assert ar._parse_ids("") == []
