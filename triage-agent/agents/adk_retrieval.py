"""
agents/adk_retrieval.py — Retrieval as a ReAct tool-using agent built with
Google's Agent Development Kit (ADK). OPTIONAL (Phase 7 comparison).

Enable with  RETRIEVAL_BACKEND=adk  (default: the LangGraph retrieval node in
agents/retrieval.py). Needs requirements-adk.txt (google-adk 2.0.0).

HOW IT WORKS
  An ADK LlmAgent gets the patient case and one tool, search_guidelines(query).
  It decides what to search, reads the results, refines and searches again if
  the classification criteria are missing (at most MAX_TOOL_CALLS searches),
  then answers with the ids of the chunks relevant for classifying the case.
  The graph-level relevance judge (relevance_check, parallel with triage) still
  runs afterwards, so both retrieval backends are compared inside the same graph.

MODEL ADAPTER (OpenAICompatLlm)
  ADK's built-in Gemini client uses the native Gemini API (x-goog-api-key).
  This project calls Gemini through the OpenAI-compatible endpoint, and the
  LiteLLM route ADK offers pins openai==2.24.0 / pydantic==2.12.5, which
  conflict with langchain-openai and gradio. So this module implements ADK's
  documented extension point, a BaseLlm subclass, over the openai SDK. It also
  fires LangChain callbacks for every model call, so the eval runner and
  tracing record the ADK agent's latency and tokens like any other agent's.
"""

from __future__ import annotations

import asyncio
import base64
import json
import os
import re
import uuid
from typing import AsyncGenerator

from google.adk.agents import LlmAgent
from google.adk.models.base_llm import BaseLlm
from google.adk.models.llm_request import LlmRequest
from google.adk.models.llm_response import LlmResponse
from google.adk.runners import InMemoryRunner
from google.genai import types
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.outputs import ChatGeneration, LLMResult
from langchain_core.runnables.config import ensure_config, get_callback_manager_for_config

from rag.store import Document
from tools.vector_search import multi_query_search

MAX_TOOL_CALLS = int(os.getenv("ADK_MAX_TOOL_CALLS", "4"))

INSTRUCTION = """You are a clinical guideline retrieval agent for community health workers.
Goal: find the guideline chunks needed to CLASSIFY the patient case you are given
(the classification criteria, thresholds and actions that apply).

Work step by step:
1. Call search_guidelines with a focused clinical query for the main problem.
2. Read the results. If the criteria that classify this case are not in them, call
   search_guidelines again with a refined query (e.g. a danger sign, a threshold,
   a different body system). Do not repeat a query. At most 4 searches.
3. When you have what is needed, reply with ONLY a JSON array of the chunk ids that are
   relevant for classifying this patient, most relevant first, e.g. ["id1", "id2"].
   Use only ids returned by search_guidelines."""


# ─── Model adapter: ADK BaseLlm over an OpenAI-compatible endpoint ──────────

class OpenAICompatLlm(BaseLlm):
    """ADK model backed by any OpenAI-compatible chat endpoint (here: Gemini)."""

    model: str
    callback_manager: object | None = None   # LangChain callback manager of the calling node

    @classmethod
    def supported_models(cls) -> list[str]:
        return [r".*"]

    async def generate_content_async(self, llm_request: LlmRequest, stream: bool = False
                                     ) -> AsyncGenerator[LlmResponse, None]:
        from openai import AsyncOpenAI

        messages = to_openai_messages(llm_request)
        tools = to_openai_tools(llm_request)
        runs = []
        if self.callback_manager is not None:
            lc_msgs = [HumanMessage(content=json.dumps(m, default=str)[:2000]) for m in messages]
            runs = self.callback_manager.on_chat_model_start(
                {"name": "OpenAICompatLlm"}, [lc_msgs], invocation_params={"model": self.model})
        try:
            client = AsyncOpenAI()  # OPENAI_API_KEY / OPENAI_BASE_URL from the environment
            kwargs = {"model": self.model, "messages": messages, "temperature": 0}
            if tools:
                kwargs["tools"] = tools
            resp = await client.chat.completions.create(**kwargs)
        except Exception as e:
            for r in runs:
                r.on_llm_error(e)
            raise
        llm_response = from_openai_response(resp)
        u = resp.usage
        for r in runs:
            r.on_llm_end(LLMResult(generations=[[ChatGeneration(message=AIMessage(
                content=_response_text(llm_response) or "tool_call",
                usage_metadata={"input_tokens": getattr(u, "prompt_tokens", 0) or 0,
                                "output_tokens": getattr(u, "completion_tokens", 0) or 0,
                                "total_tokens": getattr(u, "total_tokens", 0) or 0}))]]))
        yield llm_response


def to_openai_messages(req: LlmRequest) -> list[dict]:
    """ADK/genai Contents → OpenAI chat messages (system, user, assistant+tool_calls, tool)."""
    out: list[dict] = []
    sys_inst = req.config.system_instruction if req.config else None
    if sys_inst:
        text = sys_inst if isinstance(sys_inst, str) else " ".join(
            p.text for p in getattr(sys_inst, "parts", []) or [] if getattr(p, "text", None))
        out.append({"role": "system", "content": text})
    for content in req.contents or []:
        parts = content.parts or []
        if content.role == "model":
            text = "".join(p.text for p in parts if p.text)
            call_parts = [p for p in parts if p.function_call]
            msg: dict = {"role": "assistant", "content": text or None}
            if call_parts:
                msg["tool_calls"] = [_tool_call_out(p) for p in call_parts]
            out.append(msg)
        else:
            responses = [p.function_response for p in parts if p.function_response]
            for fr in responses:
                out.append({"role": "tool", "tool_call_id": fr.id or "", "content": json.dumps(fr.response, default=str)})
            text = "".join(p.text for p in parts if p.text)
            if text:
                out.append({"role": "user", "content": text})
    return out


def _tool_call_out(part: types.Part) -> dict:
    c = part.function_call
    out = {"id": c.id or f"call_{uuid.uuid4().hex[:8]}", "type": "function",
           "function": {"name": c.name, "arguments": json.dumps(c.args or {})}}
    if part.thought_signature:
        out["extra_content"] = {"google": {"thought_signature": base64.b64encode(part.thought_signature).decode()}}
    return out


def to_openai_tools(req: LlmRequest) -> list[dict]:
    tools = []
    for tool in (req.config.tools if req.config and req.config.tools else []):
        for fd in getattr(tool, "function_declarations", None) or []:
            schema = getattr(fd, "parameters_json_schema", None)
            if schema is None and fd.parameters is not None:
                schema = _genai_schema_to_json(fd.parameters.model_dump(exclude_none=True))
            tools.append({"type": "function", "function": {
                "name": fd.name, "description": fd.description or "",
                "parameters": schema or {"type": "object", "properties": {}}}})
    return tools


def _genai_schema_to_json(s: dict) -> dict:
    out = {k: v for k, v in s.items() if k not in ("type", "properties", "items")}
    if "type" in s:
        out["type"] = str(s["type"]).lower().replace("type.", "")
    if "properties" in s:
        out["properties"] = {k: _genai_schema_to_json(v) for k, v in s["properties"].items()}
    if "items" in s:
        out["items"] = _genai_schema_to_json(s["items"])
    return out


def from_openai_response(resp) -> LlmResponse:
    msg = resp.choices[0].message
    parts: list[types.Part] = []
    if msg.content:
        parts.append(types.Part(text=msg.content))
    for tc in msg.tool_calls or []:
        try:
            args = json.loads(tc.function.arguments or "{}")
        except json.JSONDecodeError:
            args = {}
        # Gemini 3 returns a "thought signature" with each function call and
        # rejects the next request (HTTP 400) unless it is sent back. Keep it on
        # the genai Part (bytes) so ADK stores it in the session history.
        sig = ((getattr(tc, "model_extra", None) or {}).get("extra_content") or {}).get("google", {}).get("thought_signature")
        parts.append(types.Part(function_call=types.FunctionCall(id=tc.id, name=tc.function.name, args=args),
                                thought_signature=base64.b64decode(sig) if sig else None))
    u = resp.usage
    usage = types.GenerateContentResponseUsageMetadata(
        prompt_token_count=getattr(u, "prompt_tokens", None),
        candidates_token_count=getattr(u, "completion_tokens", None),
        total_token_count=getattr(u, "total_tokens", None)) if u else None
    return LlmResponse(content=types.Content(role="model", parts=parts), usage_metadata=usage)


def _response_text(r: LlmResponse) -> str:
    if not r.content or not r.content.parts:
        return ""
    return "".join(p.text for p in r.content.parts if p.text)


# ─── The ReAct retrieval agent ───────────────────────────────────────────────

def adk_retrieve(summary, patient_context: str, guideline_type: str | None,
                 model: str | None = None) -> tuple[list[Document], list[str], int]:
    """
    Run the ADK agent for one patient. Returns (selected docs, queries it issued,
    number of tool calls). Never raises for agent-level problems: if the agent
    gives no usable answer, every chunk it retrieved is returned.
    """
    seen: dict[str, Document] = {}
    queries: list[str] = []

    def search_guidelines(query: str) -> dict:
        """Search the clinical guideline knowledge base.

        Args:
            query: a focused clinical search query, e.g. "IMNCI classify cough fast breathing".

        Returns:
            The best matching guideline chunks, each with id, section and text.
        """
        if len(queries) >= MAX_TOOL_CALLS:
            return {"error": f"search limit of {MAX_TOOL_CALLS} reached; answer now with the chunk ids"}
        queries.append(query)
        docs = multi_query_search([query], n_results_per_query=4, guideline_type=guideline_type) or \
            multi_query_search([query], n_results_per_query=4, guideline_type=None)
        for d in docs:
            seen.setdefault(d.id, d)
        return {"results": [{"id": d.id, "section": d.section, "text": d.content[:700]} for d in docs]}

    llm = OpenAICompatLlm(model=model or os.getenv("FAST_MODEL_NAME", "gemini-3.8-flash"),
                          callback_manager=get_callback_manager_for_config(ensure_config()))
    agent = LlmAgent(name="guideline_retriever", model=llm, instruction=INSTRUCTION, tools=[search_guidelines])
    final_text = asyncio.run(_run(agent, f"PATIENT CASE:\n{patient_context}"))

    ids = _parse_ids(final_text)
    selected = [seen[i] for i in ids if i in seen] or sorted(seen.values(), key=lambda d: d.score)
    return selected, queries, len(queries)


async def _run(agent: LlmAgent, text: str) -> str:
    runner = InMemoryRunner(agent=agent, app_name="triage_retrieval")
    session = await runner.session_service.create_session(app_name="triage_retrieval", user_id="triage")
    final = ""
    async for event in runner.run_async(user_id="triage", session_id=session.id,
                                        new_message=types.Content(role="user", parts=[types.Part(text=text)])):
        if event.is_final_response() and event.content and event.content.parts:
            final = "".join(p.text for p in event.content.parts if p.text) or final
    return final


def _parse_ids(text: str) -> list[str]:
    t = re.sub(r"^```(?:json)?\s*|\s*```$", "", (text or "").strip())
    try:
        v = json.loads(t)
        if isinstance(v, list):
            return [str(x) for x in v]
    except (json.JSONDecodeError, TypeError):
        pass
    return re.findall(r"[0-9a-f]{32}", text or "")
