"""
graph.py — LangGraph StateGraph Definition

This is where the four agents are wired together into a runnable graph.
After reading this file you should be able to answer:
  - Which node runs first?
  - How does the supervisor decide routing?
  - Where does the human-in-the-loop interrupt happen?
  - How does state persist between user messages?

GRAPH STRUCTURE:
  Entry point → supervisor
  supervisor → intake | retrieval | triage | output  (conditional edges)
  intake     → supervisor  (unconditional)
  retrieval  → supervisor  (unconditional)
  triage     → supervisor  (unconditional)
  output     → END

  Pattern: hub-and-spoke. Every agent reports back to supervisor.
  Supervisor decides who goes next. Agents never route directly to each other.

HUMAN-IN-THE-LOOP:
  interrupt_before=["intake"] tells LangGraph to pause execution BEFORE
  running the intake node each time. This is how we "wait for user input":

  1. User sends message → invoke graph with HumanMessage in state
  2. Supervisor routes to intake
  3. Graph PAUSES (interrupt_before fires)
  4. Resume → intake runs → generates question → adds AIMessage to state
  5. Supervisor routes back to intake (intake_complete=False)
  6. Graph PAUSES again
  7. UI reads last AIMessage (the question), shows it to user
  8. User answers → add HumanMessage to state, resume from checkpoint
  9. Repeat until intake_complete=True
  10. From intake_complete=True: supervisor→retrieval→triage→output
      (no more interrupts — these run straight through)

CHECKPOINTER:
  MemorySaver stores graph state in memory keyed by thread_id.
  Each user session gets a unique thread_id. Between HTTP requests,
  the checkpointer restores the exact graph state so execution
  continues from where it left off.

  For production: replace MemorySaver with SqliteSaver or PostgresSaver.
  The rest of the code is identical.

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
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph

from agents.intake import intake_agent
from agents.retrieval import retrieval_agent
from agents.supervisor import route_next, supervisor
from agents.triage import triage_agent
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
                      Defaults to MemorySaver (in-memory, per-process).
                      For production: SqliteSaver("triage.db") or PostgresSaver.

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
        checkpointer = MemorySaver()

    graph = StateGraph(TriageState)

    # ─── Add nodes ───────────────────────────────────────────────────────
    graph.add_node("supervisor", supervisor)
    graph.add_node("intake", intake_agent)
    graph.add_node("retrieval", retrieval_agent)
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
    graph.add_edge("retrieval", "supervisor")
    graph.add_edge("triage", "supervisor")

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
        },
    )

    # ─── Compile ─────────────────────────────────────────────────────────
    # interrupt_before=["intake"]: pause BEFORE intake runs each time.
    # This gives the caller a chance to read the last AI message
    # (the question) and present it to the user before continuing.
    app = graph.compile(
        checkpointer=checkpointer,
        interrupt_before=["intake"],
    )

    return app


def run_turn(
    app,
    thread_id: str,
    user_message: str,
    config: dict | None = None,
) -> dict[str, Any]:
    """
    Process one user message and return the system's response.

    This is the main interface for the Gradio app and any other frontend.
    It handles the interrupt/resume cycle transparently.

    Args:
        app:          Compiled LangGraph app (from create_graph()).
        thread_id:    Unique session identifier. Same ID = same conversation.
        user_message: The user's text input.
        config:       Optional LangGraph config overrides.

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
    thread_config = {"configurable": {"thread_id": thread_id}}
    if config:
        thread_config.update(config)

    # ─── Check if this is a new or existing session ───────────────────────
    current_state = _get_current_state(app, thread_config)
    is_new_session = current_state is None

    if is_new_session:
        # New session: invoke with initial state + first message
        initial = create_initial_state()
        initial["messages"] = [HumanMessage(content=user_message)]
        app.invoke(initial, config=thread_config)
    else:
        # Existing session: add user message and resume from checkpoint
        app.invoke(
            {"messages": [HumanMessage(content=user_message)]},
            config=thread_config,
        )

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
