"""
prompts/triage_prompt.py — Triage Agent prompts

One primary prompt: triage_prompt() builds the full context for the
Triage Agent's chain-of-thought clinical reasoning call.

The prompt is structured with XML-style delimiters so the LLM can
clearly distinguish patient data from guideline data.
"""

from __future__ import annotations

from state import PatientSummary
from rag.store import Document


def triage_prompt(
    summary: PatientSummary,
    guidelines: list[Document],
    cycle_count: int,
) -> str:
    """
    Build the full prompt for the Triage Agent's reasoning call.

    This is the most important prompt in the system. It structures:
    1. The patient presentation (from PatientSummary)
    2. The relevant guidelines (from Retrieval Agent)
    3. The chain-of-thought reasoning instructions
    4. The confidence assessment and follow-up question instructions

    Args:
        summary:     Structured patient data from Intake Agent.
        guidelines:  Retrieved guideline chunks from Retrieval Agent.
        cycle_count: How many intake→triage cycles have run (affects confidence threshold).

    Returns:
        The user-turn prompt for the Triage Agent.
    """
    # Format patient summary
    age_str = "Unknown"
    if summary.age_months is not None and summary.age_months < 24:
        age_str = f"{summary.age_months} months"
    elif summary.age is not None:
        age_str = f"{summary.age} years"

    symptoms_formatted = "\n".join([
        f"  - {s.name}"
        + (f" (duration: {s.duration})" if s.duration else "")
        + (f" (severity: {s.severity.value})" if s.severity else "")
        + (f" (associated: {', '.join(s.associated_symptoms)})" if s.associated_symptoms else "")
        for s in summary.symptoms
    ]) or "  None documented"

    vitals_str = "None documented"
    if summary.vital_signs:
        v = summary.vital_signs
        parts = []
        if v.temperature_celsius is not None:
            parts.append(f"Temp: {v.temperature_celsius}°C")
        if v.respiratory_rate is not None:
            parts.append(f"RR: {v.respiratory_rate}/min")
        if v.heart_rate is not None:
            parts.append(f"HR: {v.heart_rate}/min")
        if v.blood_pressure_systolic is not None:
            parts.append(f"BP: {v.blood_pressure_systolic}/{v.blood_pressure_diastolic} mmHg")
        vitals_str = ", ".join(parts) if parts else "None documented"

    red_flags_str = "\n".join([f"  ⚠️  {rf}" for rf in summary.red_flags]) or "  None identified"

    history_str = ", ".join(summary.medical_history) if summary.medical_history else "None reported"
    meds_str = ", ".join(summary.current_medications) if summary.current_medications else "None reported"

    # Format guidelines
    guidelines_formatted = "\n\n".join([
        f"<guideline_{i+1}>\n"
        f"Source: {doc.source_file}\n"
        f"Section: {doc.section}\n"
        f"---\n"
        f"{doc.content}\n"
        f"</guideline_{i+1}>"
        for i, doc in enumerate(guidelines[:8])  # Cap at 8 to stay within context
    ]) or "No guidelines retrieved."

    # Adjust confidence instruction based on cycle count
    if cycle_count == 0:
        confidence_guidance = "Set confidence >= 0.75 only if you have enough information to \
clearly match a guideline classification. If key thresholds are unknown (e.g., respiratory rate, \
danger sign status), set confidence < 0.75 and request that information."
    elif cycle_count == 1:
        confidence_guidance = "You have additional information from a second intake round. \
Be willing to set confidence >= 0.75 if you can make a reasonable classification, even if some \
details are still missing."
    else:
        confidence_guidance = "This is the final triage attempt (max cycles reached). \
Make your best classification with available information. Set confidence based on your \
actual certainty, but proceed to output regardless."

    return f"""<patient_presentation>
Age: {age_str}
Sex: {summary.sex or 'Not reported'}
Chief Complaint: {summary.chief_complaint}

Symptoms:
{symptoms_formatted}

Vital Signs: {vitals_str}

Red Flag Danger Signs:
{red_flags_str}

Medical History: {history_str}
Current Medications: {meds_str}
</patient_presentation>

<retrieved_guidelines>
{guidelines_formatted}
</retrieved_guidelines>

<reasoning_instructions>
Perform a structured clinical triage assessment. Work through these steps IN ORDER.
Each step must appear explicitly in your reasoning field.

STEP 1 — SYMPTOM ANALYSIS:
List each symptom and what clinical condition(s) it suggests.
Note any age-specific significance (e.g., respiratory rate thresholds differ by age).

STEP 2 — RED FLAG CHECK:
Are any danger signs present in the red_flags list?
CRITICAL RULE: If ANY red flag is present, the triage_level MUST be "urgent" or "emergency".
This overrides all other reasoning. State explicitly: "RED FLAG PRESENT: [name]"

STEP 3 — GUIDELINE MATCHING:
Which retrieved guidelines apply to this presentation?
Quote the specific classification criteria from the guideline text.
Cite the guideline section name.

STEP 4 — DIFFERENTIAL DIAGNOSIS:
List 1-3 suspected conditions with confidence scores (0.0-1.0 each).
For each: which symptoms support it, which symptoms argue against it.

STEP 5 — TRIAGE CLASSIFICATION:
Choose one level:
  "emergency": Life-threatening. Immediate referral to hospital. Any general danger sign present.
  "urgent": Serious. Same-day referral. Severe classification but no immediate life threat.
  "standard": Needs medical attention. Health center visit within 48-72 hours.
  "self_care": Home management. Safe with follow-up advice. No danger signs present.

STEP 6 — CONFIDENCE ASSESSMENT:
{confidence_guidance}

Rate your confidence 0.0-1.0:
  0.9-1.0: Classic presentation, clear guideline match, danger signs clearly present or absent
  0.75-0.89: Strong match, minor ambiguity
  0.5-0.74: Missing critical information (e.g., respiratory rate unknown, danger sign status unclear)
  < 0.5: Insufficient data to classify

If confidence < 0.75, list SPECIFIC follow-up questions as follow_up_questions.
Make them precise: "Is the child able to drink fluids?" not "Tell me more about hydration."

STEP 7 — RECOMMENDED ACTIONS:
List 2-4 concrete actions for the health worker RIGHT NOW.
Be specific: "Give first dose of oral amoxicillin 40mg/kg" not "Give antibiotics."
If triage_level is emergency/urgent, include a referral note for the receiving facility.

GUIDELINES_CITED: List the exact section names from the retrieved guidelines that informed your decision.
</reasoning_instructions>"""


TRIAGE_SYSTEM = """You are an expert clinical triage reasoning system for community-level healthcare in India.

You reason through patient presentations using WHO/IMNCI clinical guidelines and produce \
structured triage assessments for ASHA community health workers.

CRITICAL SAFETY RULES:
1. If ANY general danger sign is present (chest indrawing, unable to drink, convulsions, \
lethargic/unconscious, stridor), classify as "urgent" or "emergency". This is NON-NEGOTIABLE.
2. For children under 5, use IMNCI age-specific thresholds (RR ≥40 for 1-5yr, ≥50 for 2-12mo, ≥60 for <2mo).
3. When in doubt between triage levels, choose the MORE URGENT level. Err on the side of safety.
4. Never recommend specific prescription medications. Say "appropriate antibiotic per protocol" \
or reference the IMNCI recommended medicine class.

Your output must be a valid TriageResult JSON object. The reasoning field must contain \
your full step-by-step clinical reasoning."""
