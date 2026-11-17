"""
agents/retrieval.py — Retrieval Agent and Relevance Check nodes

PURPOSE:
Find the clinical guideline chunks for the current patient case, and judge
whether they are relevant enough to triage on.

TWO NODES (Phase 5 latency redesign):
  retrieval_agent        ONE retrieval attempt per run:
                           attempt 1 uses the search queries the intake
                           extraction already produced (no extra LLM call);
                           later attempts ask the LLM to reformulate, excluding
                           queries already tried and naming what was missing.
  relevance_check_agent  LLM-as-judge score (1-5) for the retrieved chunks.

SELF-CORRECTING LOOP (now at graph level, and parallel):
  retrieval → { relevance_check ∥ triage } → supervisor
  The judge and the triage agent run IN PARALLEL on the same guidelines. If the
  judge scores them < RELEVANCE_THRESHOLD and attempts remain, the supervisor
  discards that (speculative) triage result and routes back to retrieval to
  reformulate. If the judge accepts them, the triage result is used directly,
  so the judge no longer adds a sequential LLM call. When the judge rejects,
  the outcome is the same as the old sequential loop (retrieve → judge →
  reformulate) at the cost of one extra triage call.

NOTE: neither node talks to the user.
"""

from __future__ import annotations

import json
import os
import re

from langchain_core.messages import HumanMessage, SystemMessage

from llm import get_fast_llm
from prompts.retrieval_prompt import query_generation_prompt
from state import TriageState
from rag.store import Document
from tools.vector_search import (
    evaluate_relevance,
    multi_query_search,
)

MAX_RETRIEVAL_ATTEMPTS = int(os.getenv("MAX_RETRIEVAL_ATTEMPTS", "3"))
RELEVANCE_THRESHOLD = 3


def retrieval_agent(state: TriageState) -> dict:
    """One retrieval attempt. Returns retrieved_guidelines, retrieval_queries, retrieval_attempts."""
    summary = state.get("patient_summary")
    attempt = state.get("retrieval_attempts", 0)
    tried: list[str] = list(state.get("retrieval_queries") or [])

    if summary is None:
        return {"retrieved_guidelines": [], "retrieval_queries": tried, "retrieval_sufficient": False,
                "retrieval_attempts": attempt + 1, "retrieval_retry": False, "retrieval_exhausted": True}

    # ─── Queries ─────────────────────────────────────────────────────────
    queries: list[str] = []
    if attempt == 0:
        # Produced by the intake extraction call: no extra round trip
        queries = [q for q in (summary.search_queries or []) if q.strip() and q not in tried][:4]
    if not queries:
        queries = _llm_queries(summary, tried, state.get("retrieval_missing") or [])
    if not queries:
        queries = [q for q in _fallback_queries(summary) if q not in tried] or _fallback_queries(summary)

    # ─── Search (age-aware filter, then unfiltered if empty) ─────────────
    guideline_type_filter = _get_guideline_type(summary)
    docs = multi_query_search(queries=queries, n_results_per_query=4, guideline_type=guideline_type_filter)
    if not docs:
        docs = multi_query_search(queries=queries, n_results_per_query=4, guideline_type=None)

    seen = list(state.get("retrieval_seen_ids") or [])
    new_ids = [d.id for d in docs if d.id not in seen]
    return {
        "retrieved_guidelines": [doc.to_dict() for doc in docs],
        "retrieval_queries": tried + [q for q in queries if q not in tried],
        "retrieval_attempts": attempt + 1,
        "retrieval_seen_ids": seen + new_ids,
        # A retry that surfaces no chunk the judge has not already seen cannot
        # change the outcome: the supervisor stops retrying (measured: the
        # adult chest-pain / stroke cases retried 3x against the same 2 chunks).
        "retrieval_exhausted": attempt > 0 and not new_ids,
        "retrieval_retry": False,
        # Judged by relevance_check_agent, which runs in parallel with triage
        "retrieval_score": None,
        "retrieval_sufficient": False,
    }


def relevance_check_agent(state: TriageState) -> dict:
    """LLM-as-judge for the current guidelines. Runs in parallel with the triage agent."""
    summary = state.get("patient_summary")
    docs = [Document.from_dict(d) for d in (state.get("retrieved_guidelines") or [])]
    if summary is None:
        return {"retrieval_score": 1, "retrieval_sufficient": False, "retrieval_missing": ["no patient summary"]}
    evaluation = evaluate_relevance(_build_patient_context(summary), docs)
    score = int(evaluation.get("score", 1))
    return {
        "retrieval_score": score,
        "retrieval_sufficient": score >= RELEVANCE_THRESHOLD,
        "retrieval_missing": list(evaluation.get("missing_aspects") or []),
    }


def _llm_queries(summary, tried: list[str], missing: list[str]) -> list[str]:
    """Ask the LLM for (re)formulated queries, excluding ones already tried."""
    llm = get_fast_llm()
    prompt = query_generation_prompt(summary)
    if tried:
        prompt += ("\n\nThese queries were already tried and did not retrieve the right guideline section; "
                   "write DIFFERENT ones:\n" + "\n".join(f"- {q}" for q in tried))
    if missing:
        prompt += "\n\nThe previous results were missing: " + "; ".join(missing)
    response = llm.invoke([
        SystemMessage(content="You are a clinical search query specialist. "
                              "Generate precise clinical search queries. "
                              "Return ONLY a JSON array of strings."),
        HumanMessage(content=prompt),
    ])
    return [q for q in _parse_queries(response.content) if q not in tried][:4]


def _get_guideline_type(summary) -> str | None:
    """
    Return the guideline type filter based on patient age.
    Pediatric cases use IMNCI; adults use WHO PEN/Primary Care.
    None means no filter (search all).
    """
    if summary.is_pediatric:
        return "imnci"
    elif summary.age is not None and summary.age >= 18:
        return "who_pen"
    return None  # Unknown age — search all


def _parse_queries(response_text: str) -> list[str]:
    """
    Parse the LLM's JSON response into a list of query strings.
    Handles common LLM output quirks (markdown fences, extra whitespace).
    """
    text = response_text.strip()
    # Strip markdown fences
    text = re.sub(r'^```(?:json)?\s*', '', text)
    text = re.sub(r'\s*```$', '', text)
    text = text.strip()

    try:
        parsed = json.loads(text)
        if isinstance(parsed, list):
            return [str(q).strip() for q in parsed if str(q).strip()]
    except (json.JSONDecodeError, TypeError):
        pass

    # Fallback: extract quoted strings
    return re.findall(r'"([^"]{5,})"', text)


def _fallback_queries(summary) -> list[str]:
    """
    Build simple search queries without an LLM call.
    Used when query generation completely fails.
    """
    queries = []

    # Base query from chief complaint
    if summary.chief_complaint:
        queries.append(summary.chief_complaint[:80])

    # Symptom-based queries
    for symptom in summary.symptoms[:2]:
        queries.append(f"{symptom.name} classification management")

    # Red flag query
    if summary.red_flags:
        queries.append(f"{summary.red_flags[0]} danger sign classification")

    # Age-specific IMNCI query
    if summary.is_pediatric:
        queries.append("IMNCI classify sick child danger signs")
    elif summary.age and summary.age >= 18:
        queries.append("WHO primary care adult patient classification")

    return queries[:4]


def _build_patient_context(summary) -> str:
    """
    Build a brief patient context string for the relevance evaluator.
    """
    age_str = "unknown age"
    if summary.age_months and summary.age_months < 24:
        age_str = f"{summary.age_months} months"
    elif summary.age:
        age_str = f"{summary.age} years"

    symptoms = ", ".join([s.name for s in summary.symptoms]) or "unspecified"
    red_flags = ", ".join(summary.red_flags) or "none"

    return (
        f"{age_str} patient. "
        f"Chief complaint: {summary.chief_complaint}. "
        f"Symptoms: {symptoms}. "
        f"Red flags: {red_flags}."
    )
