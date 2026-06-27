"""
agents/triage.py — Triage Agent Node

PURPOSE:
Reason through the patient case using retrieved guidelines and produce
a structured triage decision with confidence score.

INPUTS FROM STATE:
  patient_summary       — structured patient data from Intake Agent
  retrieved_guidelines  — serialized Document dicts from Retrieval Agent
  cycle_count           — number of intake→triage cycles so far

OUTPUTS TO STATE:
  triage_result         — TriageResult (classification + reasoning + actions)
  triage_confidence     — float 0.0-1.0
  follow_up_questions   — list of specific questions if confidence < threshold

RED FLAG OVERRIDE (SAFETY GUARDRAIL):
  If PatientSummary.red_flags is non-empty, triage_level is forced to
  "urgent" or "emergency" in post-processing, REGARDLESS of LLM output.
  This is implemented as Python logic, not left to the LLM's judgment.
  A clinical system must never under-triage a danger sign case.

STRUCTURED OUTPUT:
  The Triage Agent uses LLM structured output to produce a TriageResult
  object directly. This means the LLM must output valid JSON matching
  the TriageResult schema. Invalid output triggers a retry.
"""

from __future__ import annotations

from langchain_core.messages import HumanMessage, SystemMessage

from llm import get_llm
from prompts.triage_prompt import TRIAGE_SYSTEM, triage_prompt
from rag.store import Document
from state import (
    CONFIDENCE_THRESHOLD,
    PatientSummary,
    SuspectedCondition,
    TriageLevel,
    TriageResult,
    TriageState,
)


def triage_agent(state: TriageState) -> dict:
    """
    Triage Agent node function for LangGraph.

    Returns dict with keys: triage_result, triage_confidence, follow_up_questions
    """
    summary: PatientSummary | None = state.get("patient_summary")
    raw_guidelines: list[dict] = state.get("retrieved_guidelines", [])
    cycle_count: int = state.get("cycle_count", 0)

    # Deserialize guidelines from state (they're stored as dicts)
    guidelines = [Document.from_dict(g) for g in raw_guidelines]

    # Edge case: no summary (should not happen in normal flow)
    if summary is None:
        return _error_result("No patient summary available for triage.")

    llm = get_llm()

    # ─── Build the triage prompt ──────────────────────────────────────────
    user_prompt = triage_prompt(
        summary=summary,
        guidelines=guidelines,
        cycle_count=cycle_count,
    )

    # ─── LLM call with structured output ─────────────────────────────────
    structured_llm = llm.with_structured_output(TriageResult)

    try:
        result: TriageResult = structured_llm.invoke([
            SystemMessage(content=TRIAGE_SYSTEM),
            HumanMessage(content=user_prompt),
        ])
    except Exception as e:
        # Structured output failed — return a low-confidence result
        # that will trigger the feedback loop
        return _error_result(f"Triage reasoning failed: {str(e)[:100]}")

    # ─── Post-processing: Red Flag Override (SAFETY GUARDRAIL) ───────────
    # If danger signs are present, escalate regardless of LLM output.
    # This is the critical safety layer.
    result = _apply_red_flag_override(result, summary)

    # ─── Extract confidence and follow-up questions ───────────────────────
    # Confidence is embedded in the TriageResult's reasoning
    # We extract it from the suspected_conditions scores
    confidence = _compute_confidence(result, summary)

    follow_up_questions: list[str] = []
    if confidence < CONFIDENCE_THRESHOLD and cycle_count < state.get("max_cycles", 3) - 1:
        # Generate specific follow-up questions to resolve uncertainty
        follow_up_questions = _extract_follow_up_questions(result, summary)

    return {
        "triage_result": result,
        "triage_confidence": confidence,
        "follow_up_questions": follow_up_questions,
    }


# ─── Helper functions ────────────────────────────────────────────────────────

def _apply_red_flag_override(result: TriageResult, summary: PatientSummary) -> TriageResult:
    """
    Post-processing safety guardrail.

    If any red flag danger signs are present in the PatientSummary,
    ensure triage_level is at least URGENT. This prevents the LLM from
    accidentally under-triaging a dangerous case.

    Returns a new TriageResult with the corrected triage_level.
    """
    if not summary.has_red_flags:
        return result

    # Danger signs present — must be urgent or emergency
    if result.triage_level not in (TriageLevel.EMERGENCY, TriageLevel.URGENT):
        # Build an escalated result
        red_flags_str = ", ".join(summary.red_flags)
        escalated_reasoning = (
            f"[SAFETY OVERRIDE] Red flag danger signs present: {red_flags_str}. "
            f"Triage level escalated to URGENT per IMNCI protocol. "
            f"Original assessment: {result.reasoning}"
        )

        # Generate referral note if not already present
        referral_note = result.referral_note or (
            f"Patient presents with danger sign(s): {red_flags_str}. "
            f"Requires urgent clinical assessment. "
            f"Chief complaint: {summary.chief_complaint}."
        )

        return TriageResult(
            triage_level=TriageLevel.URGENT,
            suspected_conditions=result.suspected_conditions,
            recommended_actions=[
                f"URGENT: Refer to nearest health facility immediately (danger sign present: {red_flags_str})",
                *result.recommended_actions,
            ],
            referral_note=referral_note,
            reasoning=escalated_reasoning,
            guidelines_cited=result.guidelines_cited,
        )

    return result


def _compute_confidence(result: TriageResult, summary: PatientSummary) -> float:
    """
    Compute overall confidence for this triage result.

    Logic:
    - Start with the top suspected condition's confidence
    - Boost if: red flags clearly present (high certainty of urgency)
    - Penalize if: no suspected conditions, or completeness was low
    - Cap at 1.0

    This gives a scalar confidence that the Supervisor uses for routing.
    """
    if not result.suspected_conditions:
        return 0.4  # Low confidence — couldn't identify conditions

    top_confidence = max(c.confidence for c in result.suspected_conditions)

    # Boost: red flags clearly identified → we're confident in the urgency
    if summary.has_red_flags:
        top_confidence = min(1.0, top_confidence + 0.15)

    # Boost: if guidelines were cited, reasoning is grounded
    if result.guidelines_cited:
        top_confidence = min(1.0, top_confidence + 0.05)

    # Penalty: very low patient summary completeness
    if summary.completeness_score < 0.4:
        top_confidence = max(0.1, top_confidence - 0.2)

    return round(top_confidence, 3)


def _extract_follow_up_questions(
    result: TriageResult,
    summary: PatientSummary,
) -> list[str]:
    """
    Generate specific follow-up questions to increase confidence.

    These are generated from the ruling_out fields of suspected conditions
    and from known missing information in the PatientSummary.
    The questions are passed back to the Intake Agent in the next cycle.
    """
    questions = []

    # Pull questions from suspected condition ruling_out fields
    for condition in result.suspected_conditions[:2]:
        for ruling_out_symptom in condition.ruling_out[:1]:
            questions.append(f"Is the following sign present: {ruling_out_symptom}?")

    # Add questions for common missing clinical decision points
    if not summary.vital_signs or summary.vital_signs.respiratory_rate is None:
        if any("cough" in s.name.lower() or "breath" in s.name.lower()
               for s in summary.symptoms):
            questions.append(
                "Can you count the child's breaths in one minute? "
                "Is it more than 40 breaths per minute?"
            )

    if not summary.red_flags:
        questions.append(
            "Is the child able to drink or breastfeed? "
            "Has the child had any convulsions?"
        )

    if summary.age is None and summary.age_months is None:
        questions.insert(0, "What is the patient's age?")

    return questions[:3]  # Return at most 3 follow-up questions


def _error_result(reason: str) -> dict:
    """
    Return a low-confidence triage result when the agent fails.
    Forces the feedback loop to get more information.
    """
    return {
        "triage_result": TriageResult(
            triage_level=TriageLevel.STANDARD,
            suspected_conditions=[
                SuspectedCondition(
                    name="Unable to classify",
                    confidence=0.1,
                    supporting_symptoms=[],
                    ruling_out=[],
                )
            ],
            recommended_actions=["Collect more patient information", "Refer to health center"],
            reasoning=f"Triage assessment could not be completed: {reason}",
            guidelines_cited=[],
        ),
        "triage_confidence": 0.1,
        "follow_up_questions": ["Please describe the patient's main symptoms in detail."],
    }
