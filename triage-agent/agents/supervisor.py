"""
agents/supervisor.py — Supervisor Node (Deterministic Router)

PURPOSE:
Inspect shared state and decide which agent acts next.
This is pure Python — NO LLM call. Routing decisions are too critical
to leave to an LLM.

ROUTING LOGIC:
  intake_complete=False            → route to "intake"
  intake_complete=True,
    retrieval not done             → route to "retrieval"
  retrieval done,
    triage not done                → route to "triage"
  triage done,
    confidence >= threshold        → route to "output"
  triage done,
    confidence < threshold,
    cycles < max_cycles            → route back to "intake" (FEEDBACK LOOP)
  triage done,
    confidence < threshold,
    cycles >= max_cycles           → route to "output" (best-effort)

RETURNS:
  current_agent  — which agent runs next (string)
  phase          — updated phase string
  cycle_count    — incremented if looping back to intake
  intake_complete — reset to False if looping back

This function is also used as the conditional edge function in the LangGraph
graph definition (graph.add_conditional_edges).
"""

from __future__ import annotations

from state import CONFIDENCE_THRESHOLD, Phase, TriageState


def supervisor(state: TriageState) -> dict:
    """
    Supervisor node: inspects state and returns routing decision.

    Called by LangGraph as a node. Returns a partial state update
    with current_agent and phase fields updated.
    """
    routing = route_next(state)
    next_agent = routing

    # Build the state update
    update: dict = {"current_agent": next_agent}

    if next_agent == "intake":
        # Either starting or cycling back
        current_phase = state.get("phase", Phase.INTAKE.value)
        if current_phase == Phase.TRIAGE.value:
            # We're cycling back — this is the feedback loop
            update["phase"] = Phase.INTAKE.value
            update["intake_complete"] = False
            update["cycle_count"] = state.get("cycle_count", 0) + 1
        else:
            update["phase"] = Phase.INTAKE.value

    elif next_agent == "retrieval":
        update["phase"] = Phase.RETRIEVAL.value

    elif next_agent == "triage":
        update["phase"] = Phase.TRIAGE.value

    elif next_agent == "output":
        update["phase"] = Phase.DONE.value

    return update


def route_next(state: TriageState) -> str:
    """
    Core routing function. Returns the name of the next node to execute.

    This function is passed to graph.add_conditional_edges() as the
    routing function. It reads state and returns a string key that
    maps to a node name in the graph.

    Return values: "intake" | "retrieval" | "triage" | "output"
    """
    intake_complete: bool = state.get("intake_complete", False)
    retrieval_sufficient: bool = state.get("retrieval_sufficient", False)
    retrieved_guidelines: list = state.get("retrieved_guidelines", [])
    triage_result = state.get("triage_result")
    triage_confidence: float = state.get("triage_confidence", 0.0)
    cycle_count: int = state.get("cycle_count", 0)
    max_cycles: int = state.get("max_cycles", 3)
    phase: str = state.get("phase", Phase.INTAKE.value)

    # ── Case 1: Intake not done → go to intake ───────────────────────────
    if not intake_complete:
        return "intake"

    # ── Case 2: Intake done, retrieval not done → go to retrieval ────────
    if intake_complete and not retrieval_sufficient and not retrieved_guidelines:
        return "retrieval"

    # ── Case 3: Retrieval done, triage not done → go to triage ───────────
    if (intake_complete and
            (retrieval_sufficient or retrieved_guidelines) and
            triage_result is None):
        return "triage"

    # ── Case 4: Triage done with HIGH confidence → output ─────────────────
    if triage_result is not None and triage_confidence >= CONFIDENCE_THRESHOLD:
        return "output"

    # ── Case 5: Triage done with LOW confidence (FEEDBACK LOOP) ───────────
    if triage_result is not None and triage_confidence < CONFIDENCE_THRESHOLD:
        if cycle_count < max_cycles - 1:
            # Loop back to intake with specific follow-up questions
            return "intake"
        else:
            # Max cycles reached — output best-effort result
            return "output"

    # ── Case 6: Retrieval done, need to re-run triage (after feedback) ────
    if phase == Phase.INTAKE.value and intake_complete and retrieved_guidelines:
        return "triage"

    # Default: go to intake
    return "intake"


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
