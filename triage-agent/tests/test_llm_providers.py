"""tests/test_llm_providers.py — provider wiring (no network calls)."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))


def test_vertexai_provider_builds_gemini_on_vertex(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "vertexai")
    monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "demo-project")
    monkeypatch.setenv("GOOGLE_CLOUD_LOCATION", "us-central1")
    monkeypatch.delenv("MODEL_NAME", raising=False)
    monkeypatch.delenv("FAST_MODEL_NAME", raising=False)
    from langchain_google_genai import ChatGoogleGenerativeAI

    from llm import get_fast_llm, get_llm
    main, fast = get_llm(), get_fast_llm()
    assert isinstance(main, ChatGoogleGenerativeAI) and isinstance(fast, ChatGoogleGenerativeAI)
    assert main.vertexai is True and main.project == "demo-project" and main.location == "us-central1"
    assert main.model.endswith("gemini-3.1-pro-preview") and fast.model.endswith("gemini-3.8-flash")
    assert main.temperature == 0.0


def test_vertexai_default_location_is_global(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "vertexai")
    monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "demo-project")
    monkeypatch.delenv("GOOGLE_CLOUD_LOCATION", raising=False)
    from llm import get_llm
    assert get_llm().location == "global"


def test_vertexai_requires_project(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "vertexai")
    monkeypatch.delenv("GOOGLE_CLOUD_PROJECT", raising=False)
    from llm import get_llm
    with pytest.raises(EnvironmentError, match="GOOGLE_CLOUD_PROJECT"):
        get_llm()


def test_vertexai_supports_structured_output_binding(monkeypatch):
    """The agents call .with_structured_output(PatientSummary/TriageResult); binding must work offline."""
    monkeypatch.setenv("LLM_PROVIDER", "vertexai")
    monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "demo-project")
    from llm import get_llm
    from state import PatientSummary, TriageResult
    for schema in (PatientSummary, TriageResult):
        assert get_llm().with_structured_output(schema) is not None
