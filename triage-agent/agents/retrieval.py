"""
agents/retrieval.py — Retrieval Agent Node

PURPOSE:
Find the relevant clinical guideline chunks for the current patient case.
Uses a self-correcting loop: retrieve → evaluate → refine if needed.

INPUTS FROM STATE:
  patient_summary   — structured patient data from Intake Agent

OUTPUTS TO STATE:
  retrieved_guidelines   — list of serialized Document dicts (top relevant chunks)
  retrieval_queries      — queries used (for debugging/display)
  retrieval_sufficient   — True if good guidelines were found

SELF-CORRECTING LOOP:
  Attempt 1: Generate queries → multi_query_search → evaluate relevance
  If score < 3:  reformulate queries → search again
  If score >= 3: proceed
  After MAX_RETRIEVAL_ATTEMPTS: proceed with best available results

WHY SELF-CORRECTION MATTERS:
  Without it, a query like "cough in children" might retrieve generic
  advice instead of the IMNCI pneumonia classification table.
  The LLM-as-judge catches this and triggers a reformulation like
  "IMNCI classify cough fast breathing chest indrawing" which retrieves
  the right section.

NOTE: The agent does NOT talk to the user. It operates entirely on
the PatientSummary from shared state.
"""

from __future__ import annotations

import json
import os
import re

from langchain_core.messages import HumanMessage, SystemMessage

from llm import get_fast_llm
from prompts.retrieval_prompt import query_generation_prompt
from state import TriageState
from tools.vector_search import (
    evaluate_relevance,
    format_for_triage_agent,
    multi_query_search,
)

MAX_RETRIEVAL_ATTEMPTS = int(os.getenv("MAX_RETRIEVAL_ATTEMPTS", "3"))
RELEVANCE_THRESHOLD = 3


def retrieval_agent(state: TriageState) -> dict:
    """
    Retrieval Agent node function for LangGraph.

    Returns dict with keys: retrieved_guidelines, retrieval_queries, retrieval_sufficient
    """
    summary = state.get("patient_summary")

    # Edge case: no patient summary yet (should not happen in normal flow)
    if summary is None:
        return {
            "retrieved_guidelines": [],
            "retrieval_queries": [],
            "retrieval_sufficient": False,
        }

    llm = get_fast_llm()
    all_queries: list[str] = []
    best_docs = []
    best_score = 0

    # Determine age-aware guideline filter
    guideline_type_filter = _get_guideline_type(summary)

    for attempt in range(MAX_RETRIEVAL_ATTEMPTS):

        # ─── Step 1: Generate search queries ──────────────────────────────
        query_prompt = query_generation_prompt(summary)
        response = llm.invoke([
            SystemMessage(content="You are a clinical search query specialist. "
                                  "Generate precise clinical search queries. "
                                  "Return ONLY a JSON array of strings."),
            HumanMessage(content=query_prompt),
        ])

        queries = _parse_queries(response.content)
        if not queries:
            # Fallback: build simple queries from patient summary
            queries = _fallback_queries(summary)

        # Deduplicate against already-tried queries
        new_queries = [q for q in queries if q not in all_queries]
        if not new_queries:
            break
        all_queries.extend(new_queries)

        # ─── Step 2: Multi-query search ───────────────────────────────────
        docs = multi_query_search(
            queries=new_queries,
            n_results_per_query=4,
            guideline_type=guideline_type_filter,
        )

        if not docs:
            # Try without the type filter (store may not have that type yet)
            docs = multi_query_search(
                queries=new_queries,
                n_results_per_query=4,
                guideline_type=None,
            )

        if not docs:
            continue

        # ─── Step 3: Evaluate relevance ───────────────────────────────────
        patient_context = _build_patient_context(summary)
        evaluation = evaluate_relevance(patient_context, docs)
        score = evaluation.get("score", 0)

        # Keep track of best results across attempts
        if score > best_score:
            best_score = score
            best_docs = docs

        if score >= RELEVANCE_THRESHOLD:
            # Good enough — stop searching
            break

        # Score too low — log what's missing and try again with different queries
        # The next iteration will generate new queries since old ones are in all_queries

    return {
        "retrieved_guidelines": [doc.to_dict() for doc in best_docs],
        "retrieval_queries": all_queries,
        "retrieval_sufficient": best_score >= RELEVANCE_THRESHOLD or len(best_docs) > 0,
    }


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
