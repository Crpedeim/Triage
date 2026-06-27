"""
prompts/retrieval_prompt.py — Retrieval Agent prompts

Two prompts:
1. query_generation_prompt() — generates clinical search queries from PatientSummary
2. RELEVANCE_SYSTEM — instructs the LLM-as-judge relevance evaluator
   (used in tools/vector_search.py evaluate_relevance())
"""

from __future__ import annotations

from state import PatientSummary


def query_generation_prompt(summary: PatientSummary) -> str:
    """
    Build the prompt for generating multiple clinical search queries.

    The Retrieval Agent calls this to generate 3-5 diverse queries
    from the PatientSummary. Each query targets a different aspect:
    - Classification criteria
    - Danger sign definitions
    - Age-specific thresholds
    - Treatment/action protocols

    Returns:
        A prompt that asks the LLM to produce a JSON list of query strings.
    """
    # Build patient context for the prompt
    age_str = "unknown age"
    if summary.age_months is not None and summary.age_months < 24:
        age_str = f"{summary.age_months} months old"
    elif summary.age is not None:
        age_str = f"{summary.age} years old"

    symptoms_str = ", ".join([s.name for s in summary.symptoms]) if summary.symptoms else "not specified"
    red_flags_str = ", ".join(summary.red_flags) if summary.red_flags else "none identified"

    guideline_focus = "IMNCI" if (
        summary.age is not None and summary.age < 5
    ) or (
        summary.age_months is not None and summary.age_months < 60
    ) else "WHO Primary Care and WHO PEN"

    return f"""You are a clinical knowledge retrieval specialist.

Given this patient presentation, generate 3-5 search queries to find relevant \
clinical guidelines for triage.

PATIENT:
- Age: {age_str}
- Chief complaint: {summary.chief_complaint}
- Symptoms: {symptoms_str}
- Red flags: {red_flags_str}
- Primary guideline source: {guideline_focus}

Generate queries that cover:
1. The primary classification criteria for this presentation
2. Danger sign definitions and thresholds relevant to this case
3. Age-specific clinical thresholds (if pediatric)
4. The action/treatment protocol for the likely classification

Rules:
- Use clinical terminology, not patient language ("tachypnea" not "fast breathing")
- Include guideline source in queries ({guideline_focus})
- Each query should target a DIFFERENT aspect of the clinical question
- Queries should be 5-10 words each

Respond ONLY with a JSON array of strings. No markdown, no explanation.
Example: ["IMNCI classify pneumonia child chest indrawing", "fast breathing threshold age 1-5 IMNCI"]"""


RELEVANCE_SYSTEM = """You are a clinical knowledge retrieval evaluator.

You will be given a patient case description and retrieved clinical guideline chunks.
Evaluate whether the retrieved chunks contain the information needed to triage this patient.

Respond ONLY with a JSON object (no markdown, no explanation):
{{
  "score": <int 1-5>,
  "missing_aspects": [<list of strings describing what's missing>],
  "reasoning": "<one sentence>"
}}

SCORING RUBRIC:
5: Retrieved chunks directly address the classification criteria for this presentation.
   Contains: classification thresholds, danger sign definitions, and action recommendations.
4: Most relevant content retrieved. Main classification criteria present but some details missing.
3: Some relevant content but missing key thresholds or classification criteria.
2: Mostly tangential content. Related disease area but wrong classification criteria.
1: Irrelevant content. Wrong disease area entirely.

Score 3+ = sufficient to proceed with triage.
Score 1-2 = reformulate queries and re-search."""
