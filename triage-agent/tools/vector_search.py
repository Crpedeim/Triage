"""
tools/vector_search.py — Vector Search Tool for the Retrieval Agent

The Retrieval Agent calls search_guidelines() to find relevant clinical
guideline chunks for a given patient presentation.

This module provides two things:
1. search_guidelines() — the core function used by the Retrieval Agent
2. evaluate_relevance() — the LLM-as-judge function for the self-correcting loop

SELF-CORRECTING RETRIEVAL LOOP (from the design doc):
    The Retrieval Agent does NOT blindly pass results forward.
    After each search, it calls evaluate_relevance() to score the results.
    If the score is below threshold, it reformulates the query and searches again.
    This loop runs up to MAX_RETRIEVAL_ATTEMPTS times.

    retrieve → evaluate → if good: return | if not: reformulate → retrieve again

WHY THIS MATTERS:
    A naive RAG system with "cough in children" might retrieve generic cough
    management advice instead of the IMNCI pneumonia classification table.
    The self-correction loop catches this: the judge says "these results don't
    contain classification criteria" and the agent reformulates to
    "IMNCI classify cough fast breathing chest indrawing".
"""

from __future__ import annotations

import os
from typing import Optional

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage

from rag.embeddings import embed_query
from rag.store import Document, search

load_dotenv()

RETRIEVAL_TOP_K = int(os.getenv("RETRIEVAL_TOP_K", "5"))
MAX_RETRIEVAL_ATTEMPTS = int(os.getenv("MAX_RETRIEVAL_ATTEMPTS", "3"))
RELEVANCE_THRESHOLD = 3  # Out of 5 — below this triggers reformulation


def search_guidelines(
    query: str,
    n_results: int = RETRIEVAL_TOP_K,
    guideline_type: Optional[str] = None,
) -> list[Document]:
    """
    Search the clinical guidelines vector store for chunks matching the query.

    This is the TOOL that the Retrieval Agent calls. It handles:
    - Embedding the query
    - Querying ChromaDB
    - Optional filtering by guideline type

    Args:
        query:          Natural language search query.
                        The Retrieval Agent formulates this as a clinical query,
                        not just the raw patient description.
                        Example: "IMNCI pneumonia classification chest indrawing child"
        n_results:      Number of chunks to return (default: RETRIEVAL_TOP_K from env)
        guideline_type: Optional filter by guideline type.
                        Values: "imnci", "who_pen", "who_primary_care", "nrhm"
                        None = search all guidelines.

    Returns:
        List of Document objects sorted by relevance (best match first).
        Empty list if the vector store is empty (not yet ingested).

    Example:
        results = search_guidelines("fast breathing pneumonia child age 3")
        for doc in results:
            print(f"[{doc.section}] score={doc.score:.3f}")
            print(doc.content[:200])
    """
    if not query.strip():
        return []

    # Embed the query using the same model used during ingestion
    query_vector = embed_query(query)

    # Build optional metadata filter
    where = None
    if guideline_type:
        where = {"guideline_type": guideline_type}

    results = search(query_vector, n_results=n_results, where=where)
    return results


def multi_query_search(
    queries: list[str],
    n_results_per_query: int = 3,
    guideline_type: Optional[str] = None,
) -> list[Document]:
    """
    Search with multiple queries and merge/deduplicate results.

    The Retrieval Agent generates multiple clinical search queries
    from the patient summary to ensure broad coverage.
    This function runs all queries and deduplicates by document ID.

    Args:
        queries:              List of search queries to run in parallel.
        n_results_per_query:  Results per query (total = queries * n_results, before dedup)
        guideline_type:       Optional filter.

    Returns:
        Deduplicated list of Documents sorted by best score per document.
        If the same chunk is retrieved by multiple queries, the best
        (lowest) score is kept.

    Example:
        queries = [
            "IMNCI classify cough pneumonia child",
            "chest indrawing danger sign classification",
            "pediatric pneumonia triage respiratory rate age 1-5"
        ]
        results = multi_query_search(queries)
    """
    # Run all queries and collect results
    seen_ids: dict[str, Document] = {}

    for query in queries:
        results = search_guidelines(query, n_results=n_results_per_query, guideline_type=guideline_type)
        for doc in results:
            if doc.id not in seen_ids or doc.score < seen_ids[doc.id].score:
                seen_ids[doc.id] = doc

    # Sort by score (best first)
    merged = sorted(seen_ids.values(), key=lambda d: d.score)
    return merged


def evaluate_relevance(
    patient_context: str,
    retrieved_docs: list[Document],
) -> dict:
    """
    Use an LLM to evaluate whether retrieved chunks are relevant to the patient case.

    This is the "LLM-as-judge" step in the self-correcting retrieval loop.
    It returns a relevance score and identifies what information is still missing.

    Args:
        patient_context: A brief description of the patient case
                         (from PatientSummary: chief_complaint + key symptoms)
        retrieved_docs:  The retrieved Document chunks to evaluate.

    Returns:
        {
            "score": int,              # 1-5 (5 = highly relevant)
            "sufficient": bool,        # True if score >= RELEVANCE_THRESHOLD
            "missing_aspects": list,   # What clinical aspects are not covered
            "reasoning": str,          # Explanation of the score
        }

    SCORING RUBRIC (in the system prompt):
        5: All retrieved docs are directly relevant. Clinical classification
           criteria, thresholds, and action recommendations are present.
        4: Most docs are relevant. Main classification is present but some
           detail is missing.
        3: Some relevant content but key classification criteria missing.
        2: Mostly irrelevant. Only tangentially related content.
        1: No relevant content. Wrong disease area entirely.

    NOTE: This function makes an LLM API call. It is called within the
    Retrieval Agent's self-correcting loop, not at the start. The pattern is:
        search → if score >= 3: return | else: reformulate → search again
    """
    from llm import get_fast_llm

    if not retrieved_docs:
        return {
            "score": 1,
            "sufficient": False,
            "missing_aspects": ["No documents retrieved"],
            "reasoning": "Vector store returned no results.",
        }

    llm = get_fast_llm()

    # Format retrieved docs for the judge
    docs_text = "\n\n---\n\n".join([
        f"[Source: {doc.source_file} | Section: {doc.section}]\n{doc.content[:500]}"
        for doc in retrieved_docs[:5]  # Judge only first 5 to save tokens
    ])

    system = """You are a clinical knowledge retrieval evaluator.
You will be given a patient case description and a set of retrieved clinical guideline chunks.
Your job is to evaluate whether the retrieved chunks contain the information needed to triage this patient.

Respond ONLY with a JSON object in this exact format (no markdown, no explanation):
{
  "score": <int 1-5>,
  "missing_aspects": [<list of strings>],
  "reasoning": "<one sentence>"
}

SCORING:
5: All retrieved chunks are directly relevant. Classification criteria, thresholds, and action recommendations are present.
4: Most chunks are relevant. Main classification is present but some detail is missing.
3: Some relevant content but key classification criteria are missing.
2: Mostly irrelevant. Only tangentially related content retrieved.
1: No relevant content. Wrong clinical area entirely."""

    user = f"""PATIENT CASE:
{patient_context}

RETRIEVED GUIDELINE CHUNKS:
{docs_text}

Evaluate the relevance of the retrieved chunks for triaging this patient case."""

    response = llm.invoke([
        SystemMessage(content=system),
        HumanMessage(content=user),
    ])

    # Parse the JSON response
    import json
    import re
    try:
        # Strip any accidental markdown
        raw = response.content.strip()
        raw = re.sub(r'^```json\s*', '', raw)
        raw = re.sub(r'\s*```$', '', raw)
        parsed = json.loads(raw)
        score = int(parsed.get("score", 1))
        return {
            "score": score,
            "sufficient": score >= RELEVANCE_THRESHOLD,
            "missing_aspects": parsed.get("missing_aspects", []),
            "reasoning": parsed.get("reasoning", ""),
        }
    except (json.JSONDecodeError, KeyError, ValueError):
        # If parsing fails, assume sufficient to avoid infinite loops
        return {
            "score": 3,
            "sufficient": True,
            "missing_aspects": [],
            "reasoning": "Evaluation parsing failed; proceeding with retrieved content.",
        }


def format_for_triage_agent(docs: list[Document]) -> str:
    """
    Format retrieved documents into a structured string for the Triage Agent's prompt.

    The Triage Agent receives this formatted string as context.
    Clear formatting is critical — the LLM must be able to distinguish
    between guideline sections and know which source each piece of advice comes from.

    Args:
        docs: List of Document objects from the retrieval pipeline.

    Returns:
        Formatted string with section headers and source citations.

    Example output:
        === GUIDELINE 1 ===
        Source: imnci_chart_booklet.pdf
        Section: ASSESS AND CLASSIFY COUGH OR DIFFICULT BREATHING
        ---
        SEVERE PNEUMONIA OR VERY SEVERE DISEASE: Any general danger sign,
        or chest indrawing, or stridor in a calm child...
    """
    if not docs:
        return "No relevant guidelines retrieved."

    formatted_parts = []
    for i, doc in enumerate(docs, start=1):
        part = (
            f"=== GUIDELINE {i} ===\n"
            f"Source: {doc.source_file}\n"
            f"Section: {doc.section}\n"
            f"---\n"
            f"{doc.content}"
        )
        formatted_parts.append(part)

    return "\n\n".join(formatted_parts)
