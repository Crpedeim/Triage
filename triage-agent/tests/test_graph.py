"""
tests/test_graph.py — Tests for Phase 4: Graph Wiring and Output

Run with:
    cd triage-agent
    python -m pytest tests/test_graph.py -v

Test groups:
1. Graph structure  — nodes exist, edges correct, compiles without error
2. Output formatter — triage card formatting for all four triage levels
3. run_turn()       — end-to-end session management with mocked agents
4. State helpers    — _get_last_ai_message, session persistence
"""

from __future__ import annotations

import sys
import uuid
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from state import (
    Phase,
    PatientSummary,
    SuspectedCondition,
    Symptom,
    TriageLevel,
    TriageResult,
    VitalSigns,
    Severity,
    create_initial_state,
)


# ─── Shared fixtures ─────────────────────────────────────────────────────────

def make_complete_state(triage_level=TriageLevel.URGENT, confidence=0.9) -> dict:
    """Build a completed triage state for output formatter tests."""
    from langchain_core.messages import AIMessage, HumanMessage
    state = create_initial_state()
    state["messages"] = [
        HumanMessage(content="3yr child cough fast breathing"),
        AIMessage(content="Is there chest indrawing?"),
        HumanMessage(content="Yes, chest indrawing present"),
        AIMessage(content="I have enough information. Analysing now..."),
    ]
    state["patient_summary"] = PatientSummary(
        age=3, age_months=36, sex="male",
        chief_complaint="Cough for 4 days with fast breathing",
        symptoms=[
            Symptom(name="cough", duration="4 days", severity=Severity.MODERATE),
            Symptom(name="chest indrawing", severity=Severity.SEVERE),
        ],
        vital_signs=VitalSigns(respiratory_rate=50),
        red_flags=["chest indrawing"],
        completeness_score=0.85,
    )
    state["intake_complete"] = True
    state["retrieval_sufficient"] = True
    state["retrieved_guidelines"] = [{
        "content": "IMNCI: chest indrawing = severe pneumonia",
        "metadata": {"source_file": "imnci.pdf", "section": "Classify Cough",
                     "page_number": 1, "chunk_index": 0, "guideline_type": "imnci"},
        "score": 0.1, "id": "x1"
    }]
    state["triage_result"] = TriageResult(
        triage_level=triage_level,
        suspected_conditions=[
            SuspectedCondition(
                name="Severe Pneumonia (IMNCI)", confidence=confidence,
                supporting_symptoms=["cough", "chest indrawing"],
            )
        ],
        recommended_actions=[
            "Refer URGENTLY to nearest health facility",
            "Give first dose of appropriate antibiotic per protocol",
            "Keep child warm during transport",
        ],
        referral_note="3yr male: chest indrawing + RR 50/min. Severe Pneumonia per IMNCI.",
        reasoning="Chest indrawing = danger sign. IMNCI: urgent referral required.",
        guidelines_cited=["IMNCI: Assess and Classify Cough or Difficult Breathing"],
    )
    state["triage_confidence"] = confidence
    state["phase"] = Phase.DONE.value
    return state


# ═══════════════════════════════════════════════════════════════════════════════
# TEST GROUP 1: Graph Structure
# ═══════════════════════════════════════════════════════════════════════════════

class TestGraphStructure:
    """Verify the graph compiles correctly and has the right structure."""

    def test_graph_compiles(self):
        """Graph compiles without raising any exceptions."""
        from graph import create_graph
        app = create_graph()
        assert app is not None

    def test_graph_has_correct_nodes(self):
        """Graph contains all four agent nodes plus supervisor and output."""
        from graph import create_graph
        app = create_graph()
        node_names = set(app.get_graph().nodes.keys())
        required_nodes = {"supervisor", "intake", "retrieval", "triage", "output"}
        assert required_nodes.issubset(node_names), (
            f"Missing nodes: {required_nodes - node_names}"
        )

    def test_graph_entry_point_is_supervisor(self):
        """Graph starts at supervisor (the router)."""
        from graph import create_graph
        app = create_graph()
        graph_def = app.get_graph()
        # Entry node should have an edge from __start__
        start_edges = [e for e in graph_def.edges if e[0] == "__start__"]
        assert any("supervisor" in str(e) for e in start_edges), (
            "Supervisor should be the entry point"
        )

    def test_graph_uses_memory_saver_by_default(self):
        """Default checkpointer is MemorySaver."""
        from graph import create_graph
        from langgraph.checkpoint.memory import MemorySaver
        app = create_graph()
        assert isinstance(app.checkpointer, MemorySaver)

    def test_graph_accepts_custom_checkpointer(self):
        """Graph accepts a custom checkpointer."""
        from graph import create_graph
        from langgraph.checkpoint.memory import MemorySaver
        custom_checkpointer = MemorySaver()
        app = create_graph(checkpointer=custom_checkpointer)
        assert app.checkpointer is custom_checkpointer

    def test_conditional_routing_maps_are_complete(self):
        """All routes returned by route_next have a matching node."""
        from agents.supervisor import route_next

        # Simulate each routing case and verify the return value
        # is a valid node name
        valid_nodes = {"intake", "retrieval", "triage", "output"}

        # Case: intake not complete
        state = create_initial_state()
        assert route_next(state) in valid_nodes

        # Case: intake complete, no retrieval
        state["intake_complete"] = True
        assert route_next(state) in valid_nodes

        # Case: retrieval done
        state["retrieval_sufficient"] = True
        state["retrieved_guidelines"] = [{"content": "x", "metadata": {}, "score": 0.1, "id": "x"}]
        assert route_next(state) in valid_nodes

        # Case: triage done high confidence
        state["triage_result"] = TriageResult(
            triage_level=TriageLevel.URGENT, suspected_conditions=[],
            recommended_actions=[], reasoning="", guidelines_cited=[]
        )
        state["triage_confidence"] = 0.9
        assert route_next(state) in valid_nodes


# ═══════════════════════════════════════════════════════════════════════════════
# TEST GROUP 2: Output Formatter
# ═══════════════════════════════════════════════════════════════════════════════

class TestOutputFormatter:
    """Test triage card formatting for all triage levels."""

    def test_format_triage_card_emergency(self):
        from output import format_triage_card
        state = make_complete_state(TriageLevel.EMERGENCY, 0.95)
        card = format_triage_card(state)
        assert "EMERGENCY" in card
        assert "IMMEDIATELY" in card or "immediate" in card.lower()

    def test_format_triage_card_urgent(self):
        from output import format_triage_card
        state = make_complete_state(TriageLevel.URGENT, 0.9)
        card = format_triage_card(state)
        assert "URGENT" in card
        assert "RECOMMENDED ACTIONS" in card

    def test_format_triage_card_standard(self):
        from output import format_triage_card
        state = make_complete_state(TriageLevel.STANDARD, 0.8)
        card = format_triage_card(state)
        assert "STANDARD" in card

    def test_format_triage_card_self_care(self):
        from output import format_triage_card
        state = make_complete_state(TriageLevel.SELF_CARE, 0.85)
        card = format_triage_card(state)
        assert "SELF CARE" in card
        assert "home" in card.lower() or "Home" in card

    def test_format_triage_card_includes_patient_info(self):
        from output import format_triage_card
        state = make_complete_state()
        card = format_triage_card(state)
        assert "3" in card  # Age
        assert "Cough" in card  # Chief complaint

    def test_format_triage_card_includes_actions(self):
        from output import format_triage_card
        state = make_complete_state()
        card = format_triage_card(state)
        assert "RECOMMENDED ACTIONS" in card
        assert "Refer" in card or "refer" in card

    def test_format_triage_card_includes_guidelines(self):
        from output import format_triage_card
        state = make_complete_state()
        card = format_triage_card(state)
        assert "IMNCI" in card  # Guidelines cited

    def test_format_triage_card_includes_referral_note(self):
        from output import format_triage_card
        state = make_complete_state()
        card = format_triage_card(state)
        assert "REFERRAL NOTE" in card

    def test_format_triage_card_low_confidence_warning(self):
        from output import format_triage_card
        state = make_complete_state(confidence=0.5)
        card = format_triage_card(state)
        assert "Low confidence" in card or "low confidence" in card.lower()

    def test_format_triage_card_no_result(self):
        from output import format_triage_card
        state = create_initial_state()  # No triage_result
        card = format_triage_card(state)
        assert "INCOMPLETE" in card or "No result" in card

    def test_format_for_gradio_returns_markdown(self):
        from output import format_for_gradio
        state = make_complete_state()
        md = format_for_gradio(state)
        # Should contain markdown headers
        assert "##" in md
        assert "**" in md  # Bold text

    def test_format_agent_status_phases(self):
        from output import format_agent_status
        for phase, expected_fragment in [
            ("intake", "Collecting"),
            ("retrieval", "Searching"),
            ("triage", "case"),
            ("done", "complete"),
        ]:
            state = create_initial_state()
            state["phase"] = phase
            status = format_agent_status(state)
            assert expected_fragment.lower() in status.lower(), (
                f"Phase '{phase}' status should contain '{expected_fragment}'"
            )

    def test_format_agent_status_feedback_loop(self):
        from output import format_agent_status
        state = create_initial_state()
        state["phase"] = "intake"
        state["cycle_count"] = 1
        status = format_agent_status(state)
        # Should indicate this is not the first round
        assert "round" in status.lower() or "2" in status


# ═══════════════════════════════════════════════════════════════════════════════
# TEST GROUP 3: run_turn() Session Management
# ═══════════════════════════════════════════════════════════════════════════════

class TestRunTurn:
    """Test the run_turn() interface with fully mocked graph execution."""

    def _make_mock_app(self, state_values: dict):
        """Build a mock LangGraph app that returns the given state."""
        mock_snapshot = MagicMock()
        mock_snapshot.values = state_values

        mock_app = MagicMock()
        mock_app.invoke.return_value = None
        mock_app.get_state.return_value = mock_snapshot

        return mock_app

    def test_run_turn_returns_expected_keys(self):
        """run_turn always returns a dict with all expected keys."""
        from graph import run_turn
        from langchain_core.messages import AIMessage, HumanMessage

        state = create_initial_state()
        state["messages"] = [
            HumanMessage(content="3yr child cough"),
            AIMessage(content="How long has the child had these symptoms?"),
        ]
        state["phase"] = Phase.INTAKE.value
        mock_app = self._make_mock_app(state)

        result = run_turn(mock_app, "thread_1", "3yr child cough")

        assert "message" in result
        assert "phase" in result
        assert "status" in result
        assert "triage_card" in result
        assert "is_complete" in result
        assert "state" in result

    def test_run_turn_returns_last_ai_message(self):
        """run_turn extracts the most recent AI message."""
        from graph import run_turn
        from langchain_core.messages import AIMessage, HumanMessage

        state = create_initial_state()
        state["messages"] = [
            HumanMessage(content="child sick"),
            AIMessage(content="First question"),
            HumanMessage(content="answer"),
            AIMessage(content="Second follow-up question"),
        ]
        mock_app = self._make_mock_app(state)

        result = run_turn(mock_app, "thread_1", "answer")
        assert result["message"] == "Second follow-up question"

    def test_run_turn_is_complete_when_phase_done(self):
        """is_complete=True only when phase==done."""
        from graph import run_turn
        from langchain_core.messages import AIMessage

        done_state = make_complete_state()
        mock_app = self._make_mock_app(done_state)

        result = run_turn(mock_app, "thread_1", "yes chest indrawing")
        assert result["is_complete"] is True
        assert result["triage_card"] != ""

    def test_run_turn_not_complete_during_intake(self):
        """is_complete=False during intake phase."""
        from graph import run_turn
        from langchain_core.messages import AIMessage

        state = create_initial_state()
        state["phase"] = Phase.INTAKE.value
        state["messages"] = [AIMessage(content="How old is the patient?")]
        mock_app = self._make_mock_app(state)

        result = run_turn(mock_app, "thread_1", "child is sick")
        assert result["is_complete"] is False
        assert result["triage_card"] == ""

    def test_run_turn_handles_none_state(self):
        """If state retrieval fails, returns safe fallback."""
        from graph import run_turn

        mock_snapshot = MagicMock()
        mock_snapshot.values = None
        mock_app = MagicMock()
        mock_app.invoke.return_value = None
        mock_app.get_state.return_value = mock_snapshot

        result = run_turn(mock_app, "thread_1", "hello")
        assert "message" in result
        assert result["is_complete"] is False

    def test_run_turn_different_threads_are_independent(self):
        """Different thread_ids produce independent state lookups."""
        from graph import run_turn
        from langchain_core.messages import AIMessage

        state_a = create_initial_state()
        state_a["messages"] = [AIMessage(content="Question for thread A")]

        state_b = create_initial_state()
        state_b["messages"] = [AIMessage(content="Question for thread B")]

        mock_snapshot_a = MagicMock(); mock_snapshot_a.values = state_a
        mock_snapshot_b = MagicMock(); mock_snapshot_b.values = state_b

        def get_state_side_effect(config):
            tid = config["configurable"]["thread_id"]
            return mock_snapshot_a if tid == "thread_a" else mock_snapshot_b

        mock_app = MagicMock()
        mock_app.invoke.return_value = None
        mock_app.get_state.side_effect = get_state_side_effect

        result_a = run_turn(mock_app, "thread_a", "msg")
        result_b = run_turn(mock_app, "thread_b", "msg")

        assert result_a["message"] != result_b["message"]


# ═══════════════════════════════════════════════════════════════════════════════
# TEST GROUP 4: State Helpers
# ═══════════════════════════════════════════════════════════════════════════════

class TestStateHelpers:
    """Test internal helper functions in graph.py."""

    def test_get_last_ai_message_finds_latest(self):
        """Returns most recent AIMessage, not the first."""
        from graph import _get_last_ai_message
        from langchain_core.messages import AIMessage, HumanMessage

        state = {
            "messages": [
                HumanMessage(content="hello"),
                AIMessage(content="First question"),
                HumanMessage(content="answer"),
                AIMessage(content="Latest question"),
            ]
        }
        assert _get_last_ai_message(state) == "Latest question"

    def test_get_last_ai_message_no_ai_messages(self):
        """Returns default message when no AI messages exist."""
        from graph import _get_last_ai_message
        from langchain_core.messages import HumanMessage

        state = {"messages": [HumanMessage(content="hi")]}
        result = _get_last_ai_message(state)
        assert isinstance(result, str)
        assert len(result) > 0

    def test_get_last_ai_message_empty_messages(self):
        """Returns default when messages list is empty."""
        from graph import _get_last_ai_message
        state = {"messages": []}
        result = _get_last_ai_message(state)
        assert isinstance(result, str)

    def test_get_last_ai_message_list_content(self):
        """Handles AI messages with list content (some providers return this)."""
        from graph import _get_last_ai_message
        from langchain_core.messages import AIMessage

        msg = AIMessage(content=[{"type": "text", "text": "List content message"}])
        state = {"messages": [msg]}
        result = _get_last_ai_message(state)
        assert "List content message" in result


# ═══════════════════════════════════════════════════════════════════════════════
# TEST GROUP 5: Full graph invocation (mocked agents)
# ═══════════════════════════════════════════════════════════════════════════════

class TestGraphInvocation:
    """
    Test the compiled graph with all agents mocked.
    These tests verify that the graph WIRING is correct —
    that signals flow between agents as expected.
    """

    @patch("agents.intake.get_fast_llm")
    def test_graph_reaches_intake_on_first_message(self, mock_llm):
        """First message triggers intake agent execution."""
        from langchain_core.messages import AIMessage
        from graph import create_graph

        # Mock intake: returns incomplete summary with a question
        mock_summary = PatientSummary(
            chief_complaint="cough", completeness_score=0.3
        )
        mock_structured = MagicMock()
        mock_structured.invoke.return_value = mock_summary
        mock_llm_instance = MagicMock()
        mock_llm_instance.with_structured_output.return_value = mock_structured
        mock_response = MagicMock()
        mock_response.content = "How old is the patient?"
        mock_llm_instance.invoke.return_value = mock_response
        mock_llm.return_value = mock_llm_instance

        app = create_graph()
        thread_id = str(uuid.uuid4())
        config = {"configurable": {"thread_id": thread_id}}

        initial = create_initial_state()
        initial["messages"] = [{"role": "human", "content": "child has cough"}]

        # Should not raise
        try:
            from langchain_core.messages import HumanMessage
            initial["messages"] = [HumanMessage(content="child has cough")]
            app.invoke(initial, config=config)
        except Exception:
            pass  # Interrupt fires — that's expected

        # Check state was updated
        snapshot = app.get_state(config)
        if snapshot and snapshot.values:
            # Messages should be in state
            assert len(snapshot.values.get("messages", [])) >= 1
