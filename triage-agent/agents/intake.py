"""
agents/intake.py — Intake Agent Node

PURPOSE:
Conducts an adaptive symptom interview with the health worker.
Extracts structured PatientSummary from the conversation.
Signals completion when enough information is collected.

INPUTS FROM STATE:
  messages             — full conversation history
  follow_up_questions  — specific questions from Triage Agent (if cycling back)
  patient_summary      — previous summary (if cycling back, may be partial)
  cycle_count          — how many triage cycles have run

OUTPUTS TO STATE:
  messages             — updated with new AI question (or completion message)
  patient_summary      — extracted and validated PatientSummary
  intake_complete      — True when completeness_score >= threshold

FLOW INSIDE THE AGENT:
  1. Extract PatientSummary from conversation (structured LLM call)
  2. Check if complete (completeness_score >= COMPLETENESS_THRESHOLD)
  3a. If complete → write completion message, set intake_complete=True
  3b. If not complete → generate next question, add to messages

WHY TWO LLM CALLS:
  Call 1 uses structured output to extract the PatientSummary schema.
  Call 2 generates the next natural-language question.
  Combining them into one call would require a complex output schema and
  the extraction quality drops — structured tasks and generation tasks
  work better separately.

HUMAN-IN-THE-LOOP NOTE:
  When the agent generates a question (step 3b), it adds an AIMessage to
  the conversation and returns. LangGraph will INTERRUPT the graph here,
  waiting for the user's next HumanMessage. When the user responds, the
  graph resumes and routes back to this agent, which re-runs from step 1
  with the updated conversation.
"""

from __future__ import annotations

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from llm import get_fast_llm
from prompts.intake_prompt import (
    COMPLETION_SYSTEM,
    EXTRACTION_SYSTEM,
    question_prompt,
)
from state import COMPLETENESS_THRESHOLD, PatientSummary, TriageState


def intake_agent(state: TriageState) -> dict:
    """
    Intake Agent node function for LangGraph.

    Takes the current graph state, runs the intake logic, and returns
    a partial state update dict. LangGraph merges this into the full state.

    Returns dict with keys: messages, patient_summary, intake_complete
    """
    messages = state["messages"]
    follow_up_questions = state.get("follow_up_questions", [])
    cycle_count = state.get("cycle_count", 0)

    llm = get_fast_llm()

    # ─── Step 1: Extract PatientSummary from conversation ─────────────────
    # Use structured output so we always get a valid Pydantic model back.
    # The LLM reads the full conversation and extracts what it knows.
    # On first call with no messages, returns a near-empty summary.

    extraction_messages = [SystemMessage(content=EXTRACTION_SYSTEM)] + list(messages)
    structured_llm = llm.with_structured_output(PatientSummary)

    try:
        summary: PatientSummary = structured_llm.invoke(extraction_messages)
    except Exception:
        # Fallback: if structured output fails (rare), use empty summary
        summary = PatientSummary(
            chief_complaint=_extract_chief_complaint_fallback(messages),
            completeness_score=0.1,
        )

    # ─── Step 2: Check completeness ───────────────────────────────────────
    if summary.is_complete:
        # Intake is done — write a brief confirmation and signal completion
        completion_response = llm.invoke([
            SystemMessage(content=COMPLETION_SYSTEM),
            HumanMessage(content=f"Patient summary: {summary.chief_complaint}. "
                                  f"Completeness: {summary.completeness_score:.0%}"),
        ])
        completion_msg = AIMessage(content=completion_response.content)
        return {
            "messages": [completion_msg],
            "patient_summary": summary,
            "intake_complete": True,
        }

    # ─── Step 3: Generate the next question ───────────────────────────────
    # Build the question prompt based on what's missing
    prompt_text = question_prompt(
        summary=summary,
        follow_up_questions=follow_up_questions,
        conversation_length=len(messages),
    )

    question_response = llm.invoke([
        SystemMessage(content=EXTRACTION_SYSTEM),
        *list(messages),
        HumanMessage(content=prompt_text),
    ])

    question_msg = AIMessage(content=question_response.content)

    # If we consumed a follow-up question from the Triage Agent, remove it
    # from the queue so we don't ask the same question twice
    remaining_follow_ups = follow_up_questions[1:] if follow_up_questions else []

    return {
        "messages": [question_msg],
        "patient_summary": summary,
        "intake_complete": False,
        "follow_up_questions": remaining_follow_ups,
    }


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
