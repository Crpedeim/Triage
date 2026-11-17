"""
graph.py — LangGraph StateGraph Definition

This is where the four agents are wired together into a runnable graph.
After reading this file you should be able to answer:
  - Which node runs first?
  - How does the supervisor decide routing?
  - How does the graph wait for the user between turns?
  - How does state persist between user messages?

GRAPH STRUCTURE:
  Entry point → supervisor
  supervisor → intake | retrieval | triage | output | END  (conditional edges)
  intake     → supervisor  (unconditional)
  retrieval  → relevance_check AND triage  (parallel fan-out, Phase 5)
  relevance_check → supervisor
  triage     → supervisor  (unconditional)
  output     → END

  Pattern: hub-and-spoke. Every agent reports back to supervisor.
  Supervisor decides who goes next. Agents never route directly to each other.

HUMAN-IN-THE-LOOP (one graph run per user turn):
  There is no interrupt. Each user message is one graph run on the thread's
  checkpointed state:

  1. run_turn() invokes the graph with {"messages": [HumanMessage], "awaiting_user": False}.
     The checkpointer merges this into the saved state for that thread_id.
  2. supervisor → intake. Intake extracts the PatientSummary.
  3. If more information is needed, intake appends its question (AIMessage)
     and sets awaiting_user=True → supervisor routes to END. The run finishes
     and run_turn() returns the question.
  4. The next user message starts step 1 again on the saved state.
  5. Once intake is complete the same run continues
     supervisor → retrieval → triage → output → END.
  6. Low-confidence feedback loop: triage → supervisor → intake (asks the
     triage agent's follow-up question, awaiting_user=True) → END.

  Earlier versions used interrupt_before=["intake"] and re-invoked with new
  input on every turn. That restarted at the supervisor and paused before
  intake each time, so intake never ran. A one-run-per-turn design needs no
  resume step and never re-executes LLM calls.

CHECKPOINTER:
  State is checkpointed per thread_id (one per user session). Between turns
  the checkpointer restores the exact graph state.
  checkpointing.make_checkpointer() returns LangGraph's PostgresSaver when
  CHECKPOINT_DB_URL is set (sessions survive restarts and are shared across
  instances) and an in-memory saver otherwise (dev/tests).

USAGE:
  from graph import create_graph, run_turn

  app = create_graph()

  # New session
  thread_id = "session_123"
  response = run_turn(app, thread_id, "3 year old child has cough for 4 days")
  print(response["message"])   # "How long has the child been breathing fast?"

  # Continue session
  response = run_turn(app, thread_id, "breathing fast for 2 days, about 50 per minute")
  print(response["message"])   # next question or triage card

  # Check if done
  if response["phase"] == "done":
      print(response["triage_card"])
"""

from __future__ import annotations

from typing import Any

from langchain_core.messages import HumanMessage
from checkpointing import make_checkpointer
from langgraph.graph import END, StateGraph

from agents.intake import intake_agent
from agents.retrieval import relevance_check_agent, retrieval_agent
from agents.supervisor import route_next, supervisor
from agents.triage import triage_agent
import tracing
from output import format_for_gradio, format_agent_status
from state import Phase, TriageState, create_initial_state


def _format_output(state: TriageState) -> dict:
    """
    Output node — the terminal node in the graph.

    Runs after the supervisor decides the assessment is complete.
    Does not modify state — just signals that graph execution should end.
    Returns an empty dict (no state changes needed — output is read
    directly from state by the calling code).
    """
    return {}  # No state changes — caller reads triage_result from state


def create_graph(checkpointer=None):
    """
    Build and compile the TriageAI LangGraph StateGraph.

    Args:
        checkpointer: LangGraph checkpointer for state persistence.
                      Defaults to checkpointing.make_checkpointer():
                      PostgresSaver when CHECKPOINT_DB_URL is set (sessions
                      survive restarts), otherwise in-memory.

    Returns:
        Compiled LangGraph app (CompiledStateGraph) ready to invoke.

    NODES:
        supervisor  — routing logic (pure Python, runs on every turn)
        intake      — symptom interviewer (LLM, talks to user)
        retrieval   — guideline searcher (LLM + vector store)
        triage      — clinical reasoner (LLM, structured output)
        output      — terminal node (no-op)

    EDGES:
        All agents → supervisor (unconditional return to hub)
        supervisor → {intake|retrieval|triage|output} (conditional)
        output → END
    """
    if checkpointer is None:
        checkpointer = make_checkpointer()

    graph = StateGraph(TriageState)

    # ─── Add nodes ───────────────────────────────────────────────────────
    graph.add_node("supervisor", supervisor)
    graph.add_node("intake", intake_agent)
    graph.add_node("retrieval", retrieval_agent)
    graph.add_node("relevance_check", relevance_check_agent)
    graph.add_node("triage", triage_agent)
    graph.add_node("output", _format_output)

    # ─── Entry point ─────────────────────────────────────────────────────
    # Every invocation starts at the supervisor, which reads state
    # and decides which agent runs first.
    graph.set_entry_point("supervisor")

    # ─── Unconditional edges (agents → supervisor) ────────────────────────
    # Every agent, after completing its work, returns to the supervisor.
    # The supervisor then re-reads state and decides the next step.
    graph.add_edge("intake", "supervisor")
    graph.add_edge("triage", "supervisor")

    # ─── Parallel fan-out after retrieval (Phase 5) ───────────────────────
    # relevance_check (LLM judge) and triage run in the SAME superstep on the
    # same guidelines; the supervisor runs once after both have finished and
    # discards the triage result if the judge rejected the guidelines.
    graph.add_edge("retrieval", "relevance_check")
    graph.add_edge("retrieval", "triage")
    graph.add_edge("relevance_check", "supervisor")

    # ─── Terminal edge ────────────────────────────────────────────────────
    graph.add_edge("output", END)

    # ─── Conditional edges (supervisor → agents) ──────────────────────────
    # route_next() is imported from agents/supervisor.py.
    # It reads state and returns a string key ("intake", "retrieval", etc.)
    # The mapping dict translates that key to the actual node name.
    graph.add_conditional_edges(
        "supervisor",
        route_next,
        {
            "intake":    "intake",
            "retrieval": "retrieval",
            "triage":    "triage",
            "output":    "output",
            "end":       END,      # intake asked a question; wait for the user
        },
    )

    # ─── Compile ─────────────────────────────────────────────────────────
    # No interrupts: a run ends at END when intake is awaiting the user.
    return graph.compile(checkpointer=checkpointer)


def run_turn(
    app,
    thread_id: str,
    user_message: str,
    config: dict | None = None,
    trace_metadata: dict | None = None,
) -> dict[str, Any]:
    """
    Process one user message and return the system's response.

    This is the main interface for the Gradio app and any other frontend.
    Each call is one graph run on the thread's checkpointed state.

    Args:
        app:          Compiled LangGraph app (from create_graph()).
        thread_id:    Unique session identifier. Same ID = same conversation.
        user_message: The user's text input.
        config:       Optional LangGraph config (e.g. {"callbacks": [...]}).
                      thread_id always comes from the thread_id argument.
        trace_metadata: Extra metadata for this turn's trace (e.g. the eval
                      case id). Ignored when tracing is off.

    Returns:
        {
            "message":     str,   # AI response (question or completion message)
            "phase":       str,   # Current phase ("intake"|"retrieval"|"triage"|"done")
            "status":      str,   # Human-readable agent status line
            "triage_card": str,   # Formatted triage card (only when phase=="done")
            "is_complete": bool,  # True when triage is complete
            "state":       dict,  # Full state for debugging
        }
    """
    thread_config: dict = dict(config or {})
    thread_config["configurable"] = {**(thread_config.get("configurable") or {}), "thread_id": thread_id}

    # Tracing (no-op unless TRACE_API_URL and TRACE_API_KEY are set)
    tracer = tracing.get_tracer()
    turn_trace = tracer.new_turn(thread_id, user_message, trace_metadata)
    if turn_trace is not None:
        thread_config["callbacks"] = list(thread_config.get("callbacks") or []) + [turn_trace]

    current_state = _get_current_state(app, thread_config)
    was_done = bool(current_state) and current_state.get("phase") == Phase.DONE.value
    user_input = {"messages": [HumanMessage(content=user_message)], "awaiting_user": False}

    try:
        if current_state is None:
            # New session: full initial state + first message
            initial = create_initial_state()
            initial.update(user_input)
            app.invoke(initial, config=thread_config)
        else:
            # Existing session: merge the reply into the saved state and run one turn
            app.invoke(user_input, config=thread_config)
    except Exception:
        if turn_trace is not None:
            turn_trace.finish()
            tracer.submit_turn(turn_trace, output="", state=_get_current_state(app, thread_config), status="error")
        raise

    # ─── Read the updated state ───────────────────────────────────────────
    state = _get_current_state(app, thread_config)

    if state is None:
        return {
            "message": "Session error. Please start a new conversation.",
            "phase": "intake",
            "status": "⚠️ Error",
            "triage_card": "",
            "is_complete": False,
            "state": {},
        }

    # ─── Extract the last AI message (the agent's question/response) ──────
    last_ai_message = _get_last_ai_message(state)
    phase = state.get("phase", Phase.INTAKE.value)
    is_complete = phase == Phase.DONE.value

    if turn_trace is not None:
        turn_trace.finish()
        tracer.submit_turn(turn_trace, output=last_ai_message, state=state,
                           final=is_complete and not was_done)

    return {
        "message": last_ai_message,
        "phase": phase,
        "status": format_agent_status(state),
        "triage_card": format_for_gradio(state) if is_complete else "",
        "is_complete": is_complete,
        "state": dict(state),
    }


def _get_current_state(app, config: dict) -> dict | None:
    """
    Retrieve the current graph state for a thread.
    Returns None if no state exists yet (new session).
    """
    try:
        snapshot = app.get_state(config)
        if snapshot and snapshot.values:
            return snapshot.values
    except Exception:
        pass
    return None


def _get_last_ai_message(state: dict) -> str:
    """
    Extract the most recent AIMessage content from the message history.
    This is the question or completion message the agent just produced.
    """
    from langchain_core.messages import AIMessage
    messages = state.get("messages", [])
    for msg in reversed(messages):
        if isinstance(msg, AIMessage):
            content = msg.content
            if isinstance(content, str):
                return content
            elif isinstance(content, list):
                # Some LLM providers return content as list of blocks
                text_blocks = [b.get("text", "") for b in content if isinstance(b, dict)]
                return " ".join(text_blocks).strip()
    return "I'm ready to help. Please describe the patient's condition."
