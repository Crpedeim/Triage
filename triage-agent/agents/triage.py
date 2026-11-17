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

SAFETY FLOOR (deterministic, safety.triage_floor):
  After the LLM call the final level is max(LLM level, rule-based floor).
  Danger signs (IMNCI general danger signs, chest indrawing, stridor, severe
  dehydration, stiff neck, adult RED_FLAG_KEYWORDS, hypertensive crisis) set
  an EMERGENCY floor; fast breathing for age or some dehydration set URGENT.
  This is Python, not the LLM's judgement, and it only ever escalates.

CONFIDENCE (_compute_confidence):
  Starts from the LLM's top-condition confidence and is reduced (never
  boosted) for an ambiguous differential, unanswered checklist items, no
  guideline citation, or low intake completeness. Capped at
  CONFIDENCE_CAP (0.95): the earlier +0.15 red-flag / +0.05 citation
  boosts pushed every real case to 1.0.

STRUCTURED OUTPUT:
  The Triage Agent uses LLM structured output to produce a TriageResult
  object directly. This means the LLM must output valid JSON matching
  the TriageResult schema. Invalid output triggers a retry.
"""

from __future__ import annotations

from langchain_core.messages import HumanMessage, SystemMessage

import safety

from llm import get_llm
from prompts.triage_prompt import TRIAGE_SYSTEM, triage_prompt
from rag.store import Document
from state import (
    AMBIGUITY_GAP,
    CONFIDENCE_THRESHOLD,
    PatientSummary,
    SuspectedCondition,
    TriageLevel,
    TriageResult,
    TriageState,
)

CONFIDENCE_CAP = 0.95


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
        # The LLM call failed (API error, quota, invalid structured output).
        # Fail SAFE: at least URGENT, and the deterministic danger-sign floor
        # still applies (R2 Phase 2: a quota outage turned emergencies into
        # STANDARD cards before this).
        return _error_result(f"Triage reasoning failed: {str(e)[:100]}", summary)

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

LEVEL_ACTIONS = {
    TriageLevel.EMERGENCY: "EMERGENCY: Refer to hospital IMMEDIATELY. Give pre-referral treatment per protocol.",
    TriageLevel.URGENT: "URGENT: Refer to a health facility TODAY.",
}


def _apply_red_flag_override(result: TriageResult, summary: PatientSummary) -> TriageResult:
    """
    Post-processing safety guardrail.

    final level = max(LLM level, safety.triage_floor(summary)). Danger signs
    force EMERGENCY; fast breathing / some dehydration force at least URGENT.
    Never downgrades. Returns a new TriageResult when it escalates.
    """
    floor, reasons = safety.triage_floor(summary)
    if floor is None or safety.max_level(result.triage_level, floor) == result.triage_level:
        return result

    reason_str = "; ".join(reasons)
    escalated_reasoning = (
        f"[SAFETY OVERRIDE] {reason_str}. Triage level escalated from "
        f"{result.triage_level.value.upper()} to {floor.value.upper()} by deterministic rule. "
        f"Original assessment: {result.reasoning}"
    )
    referral_note = result.referral_note or (
        f"Danger finding(s): {reason_str}. Chief complaint: {summary.chief_complaint}."
    )
    return TriageResult(
        triage_level=floor,
        suspected_conditions=result.suspected_conditions,
        recommended_actions=[f"{LEVEL_ACTIONS[floor]} ({reason_str})", *result.recommended_actions],
        referral_note=referral_note,
        reasoning=escalated_reasoning,
        guidelines_cited=result.guidelines_cited,
    )


def _compute_confidence(result: TriageResult, summary: PatientSummary) -> float:
    """
    Confidence that the classification is well supported by the evidence.

    p = top suspected-condition confidence (0.4 if the model gave none)
      - ambiguity:  if the top two are within AMBIGUITY_GAP, subtract the shortfall
      × coverage:   (0.6 + 0.4 × fraction of applicable checklist items answered)
      - 0.10        if no guideline section was cited
      - 0.20        if intake completeness < 0.4
    clamped to [0.05, CONFIDENCE_CAP]. No boosts: red flags change the LEVEL
    (via the safety floor), not how certain the reasoning is.
    """
    conds = sorted(result.suspected_conditions, key=lambda c: c.confidence, reverse=True)
    p = conds[0].confidence if conds else 0.4

    if len(conds) >= 2:
        gap = conds[0].confidence - conds[1].confidence
        if gap < AMBIGUITY_GAP:
            p -= (AMBIGUITY_GAP - gap)

    p *= 0.6 + 0.4 * safety.checklist_coverage(summary)

    if not result.guidelines_cited:
        p -= 0.10
    if summary.completeness_score < 0.4:
        p -= 0.20

    return round(min(CONFIDENCE_CAP, max(0.05, p)), 3)


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

    # Checklist items the caregiver could not answer the first time
    for item in safety.unassessed_items(summary)[:2]:
        if item != "respiratory_rate":
            questions.append(f"Can you check again whether this is present: {item.replace('_', ' ')}?")

    if summary.age is None and summary.age_months is None:
        questions.insert(0, "What is the patient's age?")

    return questions[:3]  # Return at most 3 follow-up questions


def _error_result(reason: str, summary: PatientSummary | None = None) -> dict:
    """
    Result when the automated assessment could not be completed.

    Fails SAFE: the level is at least URGENT (a patient who could not be
    assessed is sent to a health facility today, never told it is routine),
    and when a summary exists the deterministic danger-sign floor is applied,
    so danger signs still give EMERGENCY. Low confidence keeps the feedback
    loop (a retry) available.
    """
    result = TriageResult(
        triage_level=TriageLevel.URGENT,
        suspected_conditions=[
            SuspectedCondition(
                name="Unable to classify (automated assessment failed)",
                confidence=0.1,
                supporting_symptoms=[],
                ruling_out=[],
            )
        ],
        recommended_actions=[
            "Automated assessment could not be completed: refer to a health facility TODAY "
            "for clinical assessment.",
            "Collect more patient information",
        ],
        reasoning=f"Triage assessment could not be completed: {reason}",
        guidelines_cited=[],
    )
    if summary is not None:
        result = _apply_red_flag_override(result, summary)
    return {
        "triage_result": result,
        "triage_confidence": 0.1,
        "follow_up_questions": ["Please describe the patient's main symptoms in detail."],
    }
