"""
prompts/intake_prompt.py — Intake Agent prompts

Two prompts:
1. EXTRACTION_SYSTEM — instructs the LLM to extract a PatientSummary
   from the conversation history. Used with structured output.
2. question_prompt() — builds the prompt for generating the next
   follow-up question given the current PatientSummary gaps.
"""

from __future__ import annotations

from state import PatientSummary


EXTRACTION_SYSTEM = """You are a clinical intake assistant helping an ASHA (community health) worker \
assess a patient in rural India.

YOUR TASK:
Extract all available patient information from the conversation and return it \
as a structured PatientSummary. Be precise — only record what has been explicitly \
stated. Do not infer or assume.

CRITICAL RULES:
- chief_complaint: Required. Summarize in one sentence.
- completeness_score: Rate 0.0–1.0 based on how much clinical information is available.
  - 0.0–0.3: Only chief complaint known, no details
  - 0.4–0.6: Some symptoms known but key details missing (duration, severity, danger signs)
  - 0.7–0.85: Core symptoms documented with enough detail for triage
  - 0.85–1.0: Comprehensive — vitals, duration, associated symptoms, danger sign check done

RED FLAGS — if ANY are mentioned, add to red_flags list:
  chest indrawing, unable to drink, unable to breastfeed, vomits everything,
  convulsions, seizures, lethargic, unconscious, altered consciousness,
  severe dehydration, stridor when calm, severe malnutrition, chest pain,
  difficulty breathing at rest, signs of stroke, severe bleeding

LANGUAGE NOTE: The health worker may describe symptoms in simple or informal language.
Map these to clinical terms:
  "breathing fast" → tachypnea
  "chest pulling in" → chest indrawing
  "not eating/drinking" → reduced oral intake
  "floppy" or "not responding" → lethargic
  "fits" → convulsions

All field values must be in ENGLISH regardless of what language was used in conversation."""


def question_prompt(
    summary: PatientSummary,
    follow_up_questions: list[str],
    conversation_length: int,
) -> str:
    """
    Build the prompt for generating the next targeted question.

    Args:
        summary:              Current PatientSummary extracted from conversation.
        follow_up_questions:  Specific questions from Triage Agent (if cycling back).
        conversation_length:  Number of turns so far (helps calibrate urgency).

    Returns:
        A user-turn prompt that tells the LLM what to ask next.
    """
    # Build context about what we know and what's missing
    known_parts = []
    missing_parts = []

    if summary.age is not None:
        known_parts.append(f"age {summary.age} years")
    else:
        missing_parts.append("age")

    if summary.chief_complaint:
        known_parts.append(f"chief complaint: {summary.chief_complaint}")

    if summary.symptoms:
        symptom_names = [s.name for s in summary.symptoms]
        known_parts.append(f"symptoms: {', '.join(symptom_names)}")

    if summary.vital_signs and summary.vital_signs.respiratory_rate:
        known_parts.append(f"respiratory rate: {summary.vital_signs.respiratory_rate}/min")
    else:
        if any("cough" in s.name.lower() or "breath" in s.name.lower()
               for s in summary.symptoms):
            missing_parts.append("respiratory rate")

    if summary.red_flags:
        known_parts.append(f"RED FLAGS: {', '.join(summary.red_flags)}")

    # Triage Agent follow-up questions take absolute priority
    if follow_up_questions:
        priority_q = follow_up_questions[0]
        return f"""The triage assessment needs more information to be confident.

Ask this specific follow-up question:
"{priority_q}"

Phrase it naturally for a community health worker. Keep it short and clear. \
Ask ONLY this one question — do not ask anything else."""

    # Otherwise generate the most important missing question
    return f"""Based on the clinical intake so far, ask ONE focused follow-up question \
to fill the most critical gap.

WHAT WE KNOW: {', '.join(known_parts) if known_parts else 'only chief complaint'}
WHAT IS MISSING: {', '.join(missing_parts) if missing_parts else 'additional symptom details'}
COMPLETENESS: {summary.completeness_score:.0%}
CONVERSATION TURNS: {conversation_length}

QUESTION PRIORITY (ask in this order if still missing):
1. Patient age (critical for IMNCI thresholds)
2. Duration of illness
3. Respiratory rate (if respiratory symptoms present)
4. Danger signs: can the patient drink/breastfeed? Any convulsions? Lethargic?
5. Fever — temperature if available, or how high
6. Associated symptoms

Rules:
- Ask ONLY ONE question
- Keep it simple — the health worker may have low medical literacy
- Be specific, not vague ("Is the child breathing more than 40 times per minute?" \
not "How is the breathing?")
- Do NOT repeat questions already answered in the conversation"""


COMPLETION_SYSTEM = """You are a clinical intake assistant.
The patient intake is complete. Write a brief, warm confirmation message (1 sentence) \
telling the health worker you have enough information and are now analyzing the case.
Do not ask any more questions. Do not give medical advice yet."""
