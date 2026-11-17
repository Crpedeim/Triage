"""
agents/intake.py — Intake Agent Node

PURPOSE:
Conducts an adaptive symptom interview with the health worker.
Extracts structured PatientSummary from the conversation.
Signals completion when enough information is collected.

INPUTS FROM STATE:
  messages             — full conversation history
  follow_up_questions  — specific questions from Triage Agent (if cycling back)

OUTPUTS TO STATE:
  messages             — updated with new AI question (or completion message)
  patient_summary      — extracted and validated PatientSummary
  intake_complete      — True when enough information is collected
  awaiting_user        — True when a question was asked (the run ends and
                         waits for the user's reply)
  follow_up_questions  — remaining queued follow-up questions

FLOW INSIDE THE AGENT:
  1. Extract PatientSummary from conversation (structured LLM call), then
     derive canonical red flags deterministically (safety.derive_red_flags)
  2. If triage queued follow-up questions, ask the next one (even if the
     summary already looks complete — triage asked for it specifically)
  3. If the age is known and a danger-sign checklist group applies that has
     not been asked or answered, ask that group's FIXED question (no LLM call)
  4. Else if complete (completeness >= threshold, or question budget used up)
     → completion message, intake_complete=True
  5. Else → generate the next question with QUESTION_SYSTEM (never the
     extraction prompt) and validate it is a plain question

SAFETY: intake cannot complete before every applicable checklist group has
been asked (safety.applicable_groups). A sign the caregiver could not answer
stays "unknown"; it is not re-asked, lowers confidence, and is listed on the
triage card as not assessed.

WHY TWO LLM CALLS:
  Call 1 uses structured output to extract the PatientSummary schema.
  Call 2 generates the next natural-language question with its own prompt.
  Reusing the extraction prompt for call 2 once produced a JSON patient summary
  that was shown to the user as if it were a question.
"""

from __future__ import annotations

import re

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

import safety
from llm import get_fast_llm
from prompts.intake_prompt import (
    COMPLETION_SYSTEM,
    EXTRACTION_SYSTEM,
    FALLBACK_QUESTION,
    QUESTION_SYSTEM,
    question_prompt,
)
from state import PatientSummary, TriageState

# Hard cap on questions per intake round so an interview always terminates
MAX_INTAKE_QUESTIONS = 8


def intake_agent(state: TriageState) -> dict:
    """
    Intake Agent node function for LangGraph.

    Returns a partial state update (see module docstring).
    """
    messages = state["messages"]
    follow_up_questions = list(state.get("follow_up_questions") or [])
    checklist_asked = list(state.get("checklist_asked") or [])

    llm = get_fast_llm()

    # ─── Step 1: Extract PatientSummary from conversation ─────────────────
    extraction_messages = [SystemMessage(content=EXTRACTION_SYSTEM)] + list(messages)
    structured_llm = llm.with_structured_output(PatientSummary)

    try:
        summary: PatientSummary = structured_llm.invoke(extraction_messages)
    except Exception:
        summary = PatientSummary(
            chief_complaint=_extract_chief_complaint_fallback(messages),
            completeness_score=0.1,
        )
    summary = _with_canonical_red_flags(summary)

    # ─── Step 2: follow-up questions queued by the triage agent ───────────
    # ─── Step 3: danger-sign checklist group (fixed question, no LLM) ─────
    if not follow_up_questions:
        group = safety.next_unasked_group(summary, checklist_asked)
        if group is not None:
            return {
                "messages": [AIMessage(content=group.question)],
                "patient_summary": summary,
                "intake_complete": False,
                "awaiting_user": True,
                "checklist_asked": checklist_asked + [group.id],
            }

    # ─── Step 4: complete only if no follow-ups are pending ───────────────
    asked_so_far = sum(1 for m in messages if isinstance(m, AIMessage))
    out_of_budget = asked_so_far >= MAX_INTAKE_QUESTIONS
    if (summary.is_complete or out_of_budget) and not follow_up_questions:
        completion_response = llm.invoke([
            SystemMessage(content=COMPLETION_SYSTEM),
            HumanMessage(content=f"Patient summary: {summary.chief_complaint}. "
                                  f"Completeness: {summary.completeness_score:.0%}"),
        ])
        return {
            "messages": [AIMessage(content=_text(completion_response.content))],
            "patient_summary": summary,
            "intake_complete": True,
            "awaiting_user": False,
        }

    # ─── Step 4: Generate the next question ───────────────────────────────
    prompt_text = question_prompt(
        summary=summary,
        follow_up_questions=follow_up_questions,
        conversation_length=len(messages),
    )
    question_response = llm.invoke([
        SystemMessage(content=QUESTION_SYSTEM),
        *list(messages),
        HumanMessage(content=prompt_text),
    ])

    default = follow_up_questions[0] if follow_up_questions else FALLBACK_QUESTION
    question = clean_question(_text(question_response.content), default=default)

    return {
        "messages": [AIMessage(content=question)],
        "patient_summary": summary,
        "intake_complete": False,
        "awaiting_user": True,
        # Consume the follow-up question we just asked
        "follow_up_questions": follow_up_questions[1:],
    }


# Code fences, a leading JSON object/array, or a JSON key ("name": ...)
_STRUCTURED_OUTPUT_RE = re.compile(r"```|^\s*[\{\[]|\"[A-Za-z_][\w ]*\"\s*:", re.MULTILINE)


def _with_canonical_red_flags(summary: PatientSummary) -> PatientSummary:
    """
    Replace free-text red flags with canonical RED_FLAG_KEYWORDS entries derived
    deterministically (safety.derive_red_flags); keep any LLM flag that does not
    map to a keyword so nothing the model noticed is silently dropped.
    """
    canonical = safety.derive_red_flags(summary)
    unmapped = [f for f in summary.red_flags if not safety.canonical_red_flags_in([f]) and not safety._negated(f)]
    return summary.model_copy(update={"red_flags": canonical + unmapped})


def clean_question(text: str, default: str = FALLBACK_QUESTION) -> str:
    """
    Return text if it reads as a plain-language question for the user.

    Falls back to `default` when the model returned nothing, or something that
    looks like JSON / code / a field dump instead of a question.
    """
    t = (text or "").strip()
    if not t or _STRUCTURED_OUTPUT_RE.search(t):
        return default
    return t


def _text(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return " ".join(b.get("text", "") for b in content if isinstance(b, dict)).strip()
    return str(content)


def _extract_chief_complaint_fallback(messages: list) -> str:
    """
    Fallback: extract chief complaint from first human message without LLM.
    Used only when structured output completely fails.
    """
    for msg in messages:
        if isinstance(msg, HumanMessage):
            text = msg.content if isinstance(msg.content, str) else str(msg.content)
            return text[:200]
    return "Patient complaint not captured"
