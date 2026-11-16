"""
tests/stub_llm.py — Scripted stand-in for a LangChain chat model.

Used to run the REAL compiled graph (real supervisor, real routing, real
checkpointer) without network calls. Each structured-output schema has its own
queue of results; plain .invoke() answers by recognising the system prompt.
Every call is logged so tests can assert which agents ran and how often.
"""

from __future__ import annotations

from langchain_core.messages import AIMessage


class StubLLM:
    def __init__(self, structured: dict | None = None, questions: list[str] | None = None):
        # schema name -> list of results (last one repeats when exhausted)
        self.structured = {k: list(v) for k, v in (structured or {}).items()}
        self.questions = list(questions or [])
        self.log: list[dict] = []

    # ── LangChain surface used by the agents ──────────────────────────────
    def with_structured_output(self, schema, **_kw):
        return _Structured(self, schema)

    def invoke(self, messages, *args, **kwargs):
        system = str(messages[0].content) if messages else ""
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
        name = self.schema.__name__
        queue = self.parent.structured.get(name)
        if not queue:
            raise AssertionError(f"StubLLM has no scripted result for {name}")
        result = queue.pop(0) if len(queue) > 1 else queue[0]
        self.parent.log.append({"kind": f"structured:{name}", "system": str(messages[0].content), "messages": messages})
        if callable(result):
            result = result()
        return result.model_copy(deep=True)
