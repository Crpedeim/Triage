"""
llm.py — LLM Provider Abstraction Layer

This is the ONLY place in the codebase where a specific LLM provider is mentioned.
Every agent calls get_llm() or get_fast_llm() and gets back a LangChain BaseChatModel.
Swap providers by changing LLM_PROVIDER in your .env — zero code changes anywhere else.

Supported providers (set LLM_PROVIDER in .env):
    anthropic   → Claude Sonnet 4 (default, best reasoning)
    openai      → GPT-4o-mini (fast, cheap)
    groq        → Llama 3.3 70B via Groq (fastest inference, free tier)
    together    → Llama 3.3 70B via Together AI (good free tier)
    ollama      → Any local model via Ollama (fully offline, no API key)

WHY THIS ABSTRACTION EXISTS:
LangChain normalises all chat models behind BaseChatModel. The .invoke(), .stream(),
and .with_structured_output() methods work identically regardless of provider.
This means you can develop locally on Ollama (free), test on Groq (fast free tier),
and submit on Anthropic (best quality) — all by changing one env variable.

USAGE IN AGENTS:
    from llm import get_llm, get_fast_llm

    # Main reasoning model (used by Triage Agent - needs best quality)
    llm = get_llm()

    # Fast model (used by Intake Agent - speed matters for conversation)
    fast_llm = get_fast_llm()

    # With structured output (Pydantic schema)
    structured = get_llm().with_structured_output(PatientSummary)

    # Check current provider (for logging/debugging)
    from llm import current_provider
    print(current_provider())  # "anthropic"
"""

from __future__ import annotations

import os
from functools import lru_cache
from typing import Literal

from dotenv import load_dotenv
from langchain_core.language_models import BaseChatModel

load_dotenv()

# ─── Supported provider type ───
Provider = Literal["anthropic", "openai", "groq", "together", "ollama"]

# ─── Model name overrides per provider ───
# These are the defaults. Override via MODEL_NAME in .env for any provider.
DEFAULT_MODELS: dict[str, str] = {
    "anthropic": "claude-sonnet-4-5",
    "openai": "gpt-4o-mini",
    "groq": "llama-3.3-70b-versatile",
    "together": "meta-llama/Llama-3.3-70B-Instruct-Turbo",
    "ollama": "llama3.2",          # run: ollama pull llama3.2
}

# Fast/cheap model overrides per provider (used for simpler tasks like conversation)
FAST_MODELS: dict[str, str] = {
    "anthropic": "claude-haiku-4-5-20251001",
    "openai": "gpt-4o-mini",
    "groq": "llama-3.1-8b-instant",
    "together": "meta-llama/Llama-3.2-11B-Vision-Instruct-Turbo",
    "ollama": "llama3.2",
}


def current_provider() -> str:
    """Return the currently configured provider name."""
    return os.getenv("LLM_PROVIDER", "anthropic").lower()


def _get_provider() -> Provider:
    """Read and validate the provider from environment."""
    provider = current_provider()
    supported = list(DEFAULT_MODELS.keys())
    if provider not in supported:
        raise ValueError(
            f"LLM_PROVIDER='{provider}' is not supported. "
            f"Choose one of: {supported}"
        )
    return provider  # type: ignore


def _build_llm(model_name: str, provider: Provider, temperature: float = 0.0) -> BaseChatModel:
    """
    Instantiate the right LangChain chat model for the given provider.

    temperature=0.0 is intentional for clinical reasoning — we want
    deterministic outputs, not creative ones. The structured output
    extraction also benefits from low temperature.
    """
    if provider == "anthropic":
        from langchain_anthropic import ChatAnthropic
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise EnvironmentError(
                "ANTHROPIC_API_KEY not set. Add it to your .env file."
            )
        return ChatAnthropic(
            model=model_name,
            temperature=temperature,
            anthropic_api_key=api_key,
            max_tokens=4096,
        )

    elif provider == "openai":
        from langchain_openai import ChatOpenAI
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise EnvironmentError(
                "OPENAI_API_KEY not set. Add it to your .env file."
            )
        return ChatOpenAI(
            model=model_name,
            temperature=temperature,
            openai_api_key=api_key,
        )

    elif provider == "groq":
        # pip install langchain-groq
        try:
            from langchain_groq import ChatGroq
        except ImportError:
            raise ImportError(
                "Groq provider requires: pip install langchain-groq"
            )
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise EnvironmentError(
                "GROQ_API_KEY not set. Get a free key at console.groq.com"
            )
        return ChatGroq(
            model=model_name,
            temperature=temperature,
            groq_api_key=api_key,
        )

    elif provider == "together":
        # pip install langchain-together
        try:
            from langchain_together import ChatTogether
        except ImportError:
            raise ImportError(
                "Together provider requires: pip install langchain-together"
            )
        api_key = os.getenv("TOGETHER_API_KEY")
        if not api_key:
            raise EnvironmentError(
                "TOGETHER_API_KEY not set. Get a free key at api.together.xyz"
            )
        return ChatTogether(
            model=model_name,
            temperature=temperature,
            together_api_key=api_key,
        )

    elif provider == "ollama":
        # pip install langchain-ollama
        # Ollama must be running locally: https://ollama.ai
        try:
            from langchain_ollama import ChatOllama
        except ImportError:
            raise ImportError(
                "Ollama provider requires: pip install langchain-ollama\n"
                "Also install Ollama from https://ollama.ai and run: ollama pull llama3.2"
            )
        base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        return ChatOllama(
            model=model_name,
            temperature=temperature,
            base_url=base_url,
        )

    raise ValueError(f"Unknown provider: {provider}")


def get_llm(temperature: float = 0.0) -> BaseChatModel:
    """
    Get the main reasoning LLM for the current provider.

    Use this for:
    - Triage Agent (complex clinical reasoning)
    - Retrieval Agent (query formulation + relevance evaluation)
    - Any structured output extraction

    The model is NOT cached (no @lru_cache) because some providers
    have per-request config. If you need caching, wrap the call yourself.
    """
    provider = _get_provider()
    model_name = os.getenv("MODEL_NAME", DEFAULT_MODELS[provider])
    return _build_llm(model_name, provider, temperature)


def get_fast_llm(temperature: float = 0.0) -> BaseChatModel:
    """
    Get the fast/cheap LLM for the current provider.

    Use this for:
    - Intake Agent (conversational, speed matters)
    - Simple classification tasks
    - Any task where speed > reasoning quality

    For providers where there's no meaningful "fast" model (e.g., Ollama),
    this returns the same model as get_llm().
    """
    provider = _get_provider()
    model_name = os.getenv("FAST_MODEL_NAME", FAST_MODELS[provider])
    return _build_llm(model_name, provider, temperature)


def get_llm_info() -> dict:
    """
    Return info about the currently configured LLM for logging/debugging.

    Useful to print at startup so you know exactly what model is running.
    """
    provider = _get_provider()
    return {
        "provider": provider,
        "main_model": os.getenv("MODEL_NAME", DEFAULT_MODELS[provider]),
        "fast_model": os.getenv("FAST_MODEL_NAME", FAST_MODELS[provider]),
        "temperature": 0.0,
    }
