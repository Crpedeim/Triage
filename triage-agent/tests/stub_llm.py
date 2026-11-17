"""
tests/stub_llm.py — Scripted stand-in for a LangChain chat model.

Used to run the REAL compiled graph (real supervisor, real routing, real
checkpointer) without network calls. Each structured-output schema has its own
queue of results; plain .invoke() answers by recognising the system prompt.
Every call is logged so tests can assert which agents ran and how often.
"""

from __future__ import annotations

from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, LLMResult
from langchain_core.runnables.config import ensure_config, get_callback_manager_for_config

# Token usage reported for every stub call (lets tracing/eval code be tested)
STUB_USAGE = {"input_tokens": 100, "output_tokens": 20, "total_tokens": 150}


class StubLLM:
    def __init__(self, structured: dict | None = None, questions: list[str] | None = None,
                 model: str = "stub-model", fail_kinds: tuple[str, ...] = ()):
        self.model = model
        self.fail_kinds = fail_kinds
        # schema name -> list of results (last one repeats when exhausted)
        self.structured = {k: list(v) for k, v in (structured or {}).items()}
        self.questions = list(questions or [])
        self.log: list[dict] = []

    # ── LangChain surface used by the agents ──────────────────────────────
    def with_structured_output(self, schema, **_kw):
        return _Structured(self, schema)

    def _traced(self, messages, kind: str, fn):
        """Fire on_chat_model_start / on_llm_end like a real chat model, via the ambient config."""
        cm = get_callback_manager_for_config(ensure_config())
        runs = cm.on_chat_model_start({"name": "StubLLM"}, [list(messages)], invocation_params={"model": self.model})
        try:
            if kind in self.fail_kinds:
                raise RuntimeError(f"stub failure: {kind}")
            out = fn()
        except Exception as e:
            for r in runs:
                r.on_llm_error(e)
            raise
        content = out.content if isinstance(out, AIMessage) else "structured-output"
        for r in runs:
            r.on_llm_end(LLMResult(generations=[[ChatGeneration(
                message=AIMessage(content=content, usage_metadata=dict(STUB_USAGE)))]]))
        return out

    def invoke(self, messages, *args, **kwargs):
        system = str(messages[0].content) if messages else ""
        kind = ("relevance_judge" if "retrieval evaluator" in system else "query_gen" if "search query" in system
                else "completion" if "intake is complete" in system else "question")
        return self._traced(messages, kind, lambda: self._plain(messages, system))

    def _plain(self, messages, system):
        if "retrieval evaluator" in system:
            kind, text = "relevance_judge", '{"score": 4, "missing_aspects": [], "reasoning": "stub"}'
        elif "search query" in system:
            kind, text = "query_gen", '["stub clinical query"]'
        elif "intake is complete" in system:
            kind, text = "completion", "Thank you, I have enough information and am analysing the case now."
        else:
            kind = "question"
            text = self.questions.pop(0) if len(self.questions) > 1 else (self.questions[0] if self.questions else "How old is the patient?")
        self.log.append({"kind": kind, "system": system, "messages": messages})
        return AIMessage(content=text)

    # ── helpers for assertions ────────────────────────────────────────────
    def count(self, kind: str) -> int:
        return sum(1 for c in self.log if c["kind"] == kind)

    def calls(self, kind: str) -> list[dict]:
        return [c for c in self.log if c["kind"] == kind]


class _Structured:
    def __init__(self, parent: StubLLM, schema):
        self.parent, self.schema = parent, schema

    def invoke(self, messages, *args, **kwargs):
        return self.parent._traced(messages, f"structured:{self.schema.__name__}", lambda: self._invoke(messages))

    def _invoke(self, messages):
        name = self.schema.__name__
        queue = self.parent.structured.get(name)
        if not queue:
            raise AssertionError(f"StubLLM has no scripted result for {name}")
        result = queue.pop(0) if len(queue) > 1 else queue[0]
        self.parent.log.append({"kind": f"structured:{name}", "system": str(messages[0].content), "messages": messages})
        if callable(result):
            result = result()
        return result.model_copy(deep=True)
