"""
agents/supervisor.py — Supervisor Node (Deterministic Router)

PURPOSE:
Inspect shared state and decide which agent acts next.
This is pure Python — NO LLM call. Routing decisions are too critical
to leave to an LLM.

ROUTING LOGIC (route_next):
  awaiting_user=True               → "end"   (intake asked a question; this run
                                              ends and the next user message
                                              starts a new run on the saved state)
  intake_complete=False            → "intake"
  intake complete, no guidelines   → "retrieval"
  guidelines present, no result    → "triage"
  result, confidence >= threshold  → "output"
  result, confidence < threshold,
    cycles left                    → "intake"  (FEEDBACK LOOP)
  result, confidence < threshold,
    no cycles left                 → "output"  (best effort)

FEEDBACK LOOP:
  When the supervisor sends a low-confidence case back to intake it clears the
  previous triage result and retrieval output. After intake has asked the
  triage agent's follow-up questions and completes again, the normal path
  (retrieval → triage) re-runs on the new information. Without the reset the
  stale low-confidence result would keep routing back to intake forever.

This function is also used as the conditional edge function in the LangGraph
graph definition (graph.add_conditional_edges).
"""

from __future__ import annotations

from state import CONFIDENCE_THRESHOLD, Phase, TriageState


def supervisor(state: TriageState) -> dict:
    """
    Supervisor node: inspects state and returns routing decision.

    Called by LangGraph as a node. Returns a partial state update. The
    conditional edge then calls route_next() on the UPDATED state, so the
    update and the edge always agree.
    """
    next_agent = route_next(state)
    update: dict = {"current_agent": next_agent}

    if next_agent == "intake":
        if _is_low_confidence_loop_back(state):
            update.update(
                phase=Phase.INTAKE.value,
                intake_complete=False,
                cycle_count=state.get("cycle_count", 0) + 1,
                # Clear stale outputs so triage (and retrieval) re-run after
                # the follow-up questions are answered.
                triage_result=None,
                triage_confidence=0.0,
                retrieved_guidelines=[],
                retrieval_queries=[],
                retrieval_sufficient=False,
            )
        else:
            update["phase"] = Phase.INTAKE.value

    elif next_agent == "retrieval":
        update["phase"] = Phase.RETRIEVAL.value

    elif next_agent == "triage":
        update["phase"] = Phase.TRIAGE.value

    elif next_agent == "output":
        update["phase"] = Phase.DONE.value

    # "end": leave phase unchanged (still intake, waiting for the user)
    return update


def _is_low_confidence_loop_back(state: TriageState) -> bool:
    return (
        state.get("triage_result") is not None
        and state.get("triage_confidence", 0.0) < CONFIDENCE_THRESHOLD
    )


def route_next(state: TriageState) -> str:
    """
    Core routing function. Returns the name of the next node to execute.

    Return values: "intake" | "retrieval" | "triage" | "output" | "end"
    """
    if state.get("awaiting_user", False):
        return "end"

    intake_complete: bool = state.get("intake_complete", False)
    retrieved_guidelines: list = state.get("retrieved_guidelines", [])
    retrieval_sufficient: bool = state.get("retrieval_sufficient", False)
    triage_result = state.get("triage_result")
    triage_confidence: float = state.get("triage_confidence", 0.0)
    cycle_count: int = state.get("cycle_count", 0)
    max_cycles: int = state.get("max_cycles", 3)

    # Triage has produced a result: accept it, or loop back for more information.
    if triage_result is not None:
        if triage_confidence >= CONFIDENCE_THRESHOLD:
            return "output"
        if cycle_count < max_cycles - 1:
            return "intake"
        return "output"

    if not intake_complete:
        return "intake"

    if not retrieval_sufficient and not retrieved_guidelines:
        return "retrieval"

    return "triage"


def has_diagnostic_ambiguity(state: TriageState) -> bool:
    """
    Check if the Triage Agent's differential is genuinely ambiguous.

    Used in Enhancement 3 (multi-specialist consultation) to decide
    whether to trigger a specialist debate.

    Ambiguity = top 2 conditions have confidence gap < AMBIGUITY_GAP.
    No ambiguity = one condition clearly dominates.
    """
    from state import AMBIGUITY_GAP

    triage_result = state.get("triage_result")
    if triage_result is None:
        return False

    conditions = triage_result.suspected_conditions
    if len(conditions) < 2:
        return False

    sorted_conditions = sorted(conditions, key=lambda c: c.confidence, reverse=True)
    gap = sorted_conditions[0].confidence - sorted_conditions[1].confidence
    return gap < AMBIGUITY_GAP
