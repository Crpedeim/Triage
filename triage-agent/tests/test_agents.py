"""
tests/test_agents.py — Tests for Phase 3: Individual Agents

Run with:
    cd triage-agent
    python -m pytest tests/test_agents.py -v

All LLM calls are mocked — no API key required.

Test groups:
1. Supervisor routing logic (pure Python, no mocking needed)
2. Intake Agent (mocked LLM structured output + question generation)
3. Retrieval Agent (mocked LLM query generation + real/mock vector search)
4. Triage Agent (mocked LLM structured output)
5. Red flag override (safety guardrail)
6. Full agent→supervisor routing flows
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from state import (
    COMPLETENESS_THRESHOLD,
    CONFIDENCE_THRESHOLD,
    PatientSummary,
    Phase,
    Severity,
    SuspectedCondition,
    Symptom,
    TriageLevel,
    TriageResult,
    VitalSigns,
    create_initial_state,
)


# ─── Shared mock data ────────────────────────────────────────────────────────

def make_pneumonia_summary(complete: bool = True) -> PatientSummary:
    return PatientSummary(
        age=3,
        age_months=36,
        sex="male",
        chief_complaint="Cough for 4 days with fast breathing",
        symptoms=[
            Symptom(name="cough", duration="4 days", severity=Severity.MODERATE),
            Symptom(name="tachypnea", severity=Severity.MODERATE,
                    associated_symptoms=["respiratory rate 50/min"]),
            Symptom(name="chest indrawing", severity=Severity.SEVERE),
        ],
        vital_signs=VitalSigns(temperature_celsius=38.2, respiratory_rate=50),
        red_flags=["chest indrawing"],
        completeness_score=0.85 if complete else 0.35,
    )


def make_mild_cough_summary() -> PatientSummary:
    return PatientSummary(
        age=4,
        sex="female",
        chief_complaint="Cough for 2 days",
        symptoms=[Symptom(name="cough", duration="2 days", severity=Severity.MILD)],
        vital_signs=VitalSigns(respiratory_rate=30),
        red_flags=[],
        completeness_score=0.8,
    )


def make_triage_result(level: TriageLevel, confidence: float) -> TriageResult:
    return TriageResult(
        triage_level=level,
        suspected_conditions=[
            SuspectedCondition(name="Test condition", confidence=confidence,
                               supporting_symptoms=["cough"])
        ],
        recommended_actions=["Test action"],
        reasoning="Test reasoning step 1. Step 2. Step 3.",
        guidelines_cited=["IMNCI: Test Section"],
    )


def make_state_with_summary(summary: PatientSummary = None, **overrides) -> dict:
    state = create_initial_state()
    if summary:
        state["patient_summary"] = summary
    state.update(overrides)
    return state


# ═══════════════════════════════════════════════════════════════════════════════
# TEST GROUP 1: Supervisor (no mocking needed — pure Python)
# ═══════════════════════════════════════════════════════════════════════════════

class TestSupervisorRouting:
    """Test all routing branches in the supervisor."""

    def test_routes_to_intake_when_not_complete(self):
        from agents.supervisor import route_next
        state = create_initial_state()
        assert route_next(state) == "intake"

    def test_routes_to_retrieval_after_intake_complete(self):
        from agents.supervisor import route_next
        state = create_initial_state()
        state["intake_complete"] = True
        state["patient_summary"] = make_mild_cough_summary()
        assert route_next(state) == "retrieval"

    def test_routes_to_triage_after_retrieval(self):
        from agents.supervisor import route_next
        state = create_initial_state()
        state["intake_complete"] = True
        state["patient_summary"] = make_mild_cough_summary()
        state["retrieval_sufficient"] = True
        state["retrieved_guidelines"] = [{"content": "test", "metadata": {}, "score": 0.1, "id": "x"}]
        assert route_next(state) == "triage"

    def test_routes_to_output_on_high_confidence(self):
        from agents.supervisor import route_next
        state = create_initial_state()
        state["intake_complete"] = True
        state["retrieval_sufficient"] = True
        state["retrieved_guidelines"] = [{"content": "test", "metadata": {}, "score": 0.1, "id": "x"}]
        state["triage_result"] = make_triage_result(TriageLevel.URGENT, 0.9)
        state["triage_confidence"] = 0.9
        assert route_next(state) == "output"

    def test_feedback_loop_on_low_confidence(self):
        """Low confidence with cycles remaining → loop back to intake."""
        from agents.supervisor import route_next
        state = create_initial_state()
        state["intake_complete"] = True
        state["retrieval_sufficient"] = True
        state["retrieved_guidelines"] = [{"content": "test", "metadata": {}, "score": 0.1, "id": "x"}]
        state["triage_result"] = make_triage_result(TriageLevel.STANDARD, 0.5)
        state["triage_confidence"] = 0.5  # Below threshold
        state["cycle_count"] = 0
        state["max_cycles"] = 3
        assert route_next(state) == "intake"

    def test_output_on_max_cycles_reached(self):
        """Low confidence but max cycles reached → output anyway."""
        from agents.supervisor import route_next
        state = create_initial_state()
        state["intake_complete"] = True
        state["retrieval_sufficient"] = True
        state["retrieved_guidelines"] = [{"content": "test", "metadata": {}, "score": 0.1, "id": "x"}]
        state["triage_result"] = make_triage_result(TriageLevel.STANDARD, 0.5)
        state["triage_confidence"] = 0.5
        state["cycle_count"] = 2  # cycle_count (2) >= max_cycles (3) - 1
        state["max_cycles"] = 3
        assert route_next(state) == "output"

    def test_supervisor_node_updates_phase(self):
        """Supervisor node returns phase updates alongside agent name."""
        from agents.supervisor import supervisor
        state = create_initial_state()
        state["intake_complete"] = True
        state["patient_summary"] = make_mild_cough_summary()
        result = supervisor(state)
        assert result["current_agent"] == "retrieval"
        assert result["phase"] == Phase.RETRIEVAL.value

    def test_supervisor_resets_intake_on_feedback_loop(self):
        """When cycling back, supervisor resets intake_complete and increments cycle_count."""
        from agents.supervisor import supervisor
        state = create_initial_state()
        state["phase"] = Phase.TRIAGE.value
        state["intake_complete"] = True
        state["retrieval_sufficient"] = True
        state["retrieved_guidelines"] = [{"content": "x", "metadata": {}, "score": 0.1, "id": "x"}]
        state["triage_result"] = make_triage_result(TriageLevel.STANDARD, 0.5)
        state["triage_confidence"] = 0.5
        state["cycle_count"] = 0
        state["max_cycles"] = 3
        result = supervisor(state)
        assert result["current_agent"] == "intake"
        assert result["intake_complete"] is False
        assert result["cycle_count"] == 1

    def test_has_diagnostic_ambiguity_true(self):
        """Two similar-confidence conditions = ambiguous."""
        from agents.supervisor import has_diagnostic_ambiguity
        state = create_initial_state()
        state["triage_result"] = TriageResult(
            triage_level=TriageLevel.STANDARD,
            suspected_conditions=[
                SuspectedCondition(name="Pneumonia", confidence=0.6, supporting_symptoms=[]),
                SuspectedCondition(name="Bronchiolitis", confidence=0.55, supporting_symptoms=[]),
            ],
            recommended_actions=[],
            reasoning="Test",
            guidelines_cited=[],
        )
        assert has_diagnostic_ambiguity(state) is True

    def test_has_diagnostic_ambiguity_false(self):
        """One dominant condition = not ambiguous."""
        from agents.supervisor import has_diagnostic_ambiguity
        state = create_initial_state()
        state["triage_result"] = TriageResult(
            triage_level=TriageLevel.URGENT,
            suspected_conditions=[
                SuspectedCondition(name="Severe Pneumonia", confidence=0.9, supporting_symptoms=[]),
                SuspectedCondition(name="Bronchiolitis", confidence=0.3, supporting_symptoms=[]),
            ],
            recommended_actions=[],
            reasoning="Test",
            guidelines_cited=[],
        )
        assert has_diagnostic_ambiguity(state) is False


# ═══════════════════════════════════════════════════════════════════════════════
# TEST GROUP 2: Intake Agent
# ═══════════════════════════════════════════════════════════════════════════════

class TestIntakeAgent:
    """Test Intake Agent with mocked LLM."""

    def _make_mock_llm(self, summary: PatientSummary, question: str = "How long has the cough lasted?"):
        """Build a mock LLM that returns the given summary and question."""
        mock_structured = MagicMock()
        mock_structured.invoke.return_value = summary

        mock_llm = MagicMock()
        mock_llm.with_structured_output.return_value = mock_structured

        # For the question generation call
        mock_response = MagicMock()
        mock_response.content = question
        mock_llm.invoke.return_value = mock_response

        return mock_llm

    @patch("agents.intake.get_fast_llm")
    def test_asks_question_when_incomplete(self, mock_get_llm):
        """Incomplete summary → agent generates a question."""
        from agents.intake import intake_agent
        from langchain_core.messages import HumanMessage

        incomplete_summary = make_pneumonia_summary(complete=False)
        mock_llm = self._make_mock_llm(incomplete_summary, "How old is the child?")
        mock_get_llm.return_value = mock_llm

        state = create_initial_state()
        state["messages"] = [HumanMessage(content="Child has cough")]
        result = intake_agent(state)

        assert result["intake_complete"] is False
        assert result["patient_summary"] is not None
        assert len(result["messages"]) > 0

    @patch("agents.intake.get_fast_llm")
    def test_signals_complete_when_sufficient(self, mock_get_llm):
        """Complete summary → agent signals intake_complete=True."""
        from agents.intake import intake_agent
        from langchain_core.messages import HumanMessage

        complete_summary = make_pneumonia_summary(complete=True)
        mock_llm = self._make_mock_llm(complete_summary)
        mock_get_llm.return_value = mock_llm

        state = create_initial_state()
        state["messages"] = [HumanMessage(content="3yr child cough fast breathing chest indrawing")]
        result = intake_agent(state)

        assert result["intake_complete"] is True
        assert result["patient_summary"].completeness_score >= COMPLETENESS_THRESHOLD

    @patch("agents.intake.get_fast_llm")
    def test_uses_follow_up_questions_from_triage(self, mock_get_llm):
        """When Triage Agent provided follow-up questions, Intake asks those first."""
        from agents.intake import intake_agent
        from langchain_core.messages import HumanMessage

        incomplete_summary = make_pneumonia_summary(complete=False)
        mock_llm = self._make_mock_llm(incomplete_summary, "Is the child able to drink fluids?")
        mock_get_llm.return_value = mock_llm

        state = create_initial_state()
        state["messages"] = [HumanMessage(content="Child is sick")]
        state["follow_up_questions"] = ["Is the child able to drink fluids?"]
        result = intake_agent(state)

        # Should have consumed the first follow-up question
        assert result.get("follow_up_questions", []) == []

    @patch("agents.intake.get_fast_llm")
    def test_handles_llm_failure_gracefully(self, mock_get_llm):
        """If structured output fails, agent returns low-completeness summary."""
        from agents.intake import intake_agent
        from langchain_core.messages import HumanMessage

        # Simulate structured output failure
        mock_structured = MagicMock()
        mock_structured.invoke.side_effect = Exception("LLM error")
        mock_llm = MagicMock()
        mock_llm.with_structured_output.return_value = mock_structured
        mock_response = MagicMock()
        mock_response.content = "What are the child's symptoms?"
        mock_llm.invoke.return_value = mock_response
        mock_get_llm.return_value = mock_llm

        state = create_initial_state()
        state["messages"] = [HumanMessage(content="Child is sick")]
        result = intake_agent(state)

        # Should not crash, should return something
        assert "intake_complete" in result
        assert result["intake_complete"] is False

    @patch("agents.intake.get_fast_llm")
    def test_returns_valid_state_keys(self, mock_get_llm):
        """Agent always returns the expected state keys."""
        from agents.intake import intake_agent
        from langchain_core.messages import HumanMessage

        summary = make_mild_cough_summary()
        summary.completeness_score = 0.4  # Incomplete
        mock_llm = self._make_mock_llm(summary)
        mock_get_llm.return_value = mock_llm

        state = create_initial_state()
        state["messages"] = [HumanMessage(content="Child has cough")]
        result = intake_agent(state)

        assert "messages" in result
        assert "patient_summary" in result
        assert "intake_complete" in result


# ═══════════════════════════════════════════════════════════════════════════════
# TEST GROUP 3: Retrieval Agent
# ═══════════════════════════════════════════════════════════════════════════════

class TestRetrievalAgent:
    """Test Retrieval Agent with mocked LLM and mock vector search."""

    def _mock_query_llm(self, queries: list[str]):
        """Build a mock LLM that returns the given queries as JSON."""
        import json
        mock_response = MagicMock()
        mock_response.content = json.dumps(queries)
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = mock_response
        return mock_llm

    @patch("agents.retrieval.get_fast_llm")
    @patch("agents.retrieval.multi_query_search")
    @patch("agents.retrieval.evaluate_relevance")
    def test_returns_guidelines_on_good_retrieval(self, mock_eval, mock_search, mock_get_llm):
        """Good retrieval (score >= 3) → returns guidelines and sufficient=True."""
        from agents.retrieval import retrieval_agent
        from rag.store import Document

        mock_get_llm.return_value = self._mock_query_llm([
            "IMNCI pneumonia chest indrawing child",
            "fast breathing classification age 1-5",
        ])
        mock_search.return_value = [
            Document(
                content="SEVERE PNEUMONIA: chest indrawing present",
                metadata={"source_file": "imnci.pdf", "section": "Classify Cough",
                          "page_number": 1, "chunk_index": 0, "guideline_type": "imnci"},
                score=0.2,
                id="abc123",
            )
        ]
        mock_eval.return_value = {"score": 4, "sufficient": True, "missing_aspects": [], "reasoning": "Relevant"}

        state = create_initial_state()
        state["patient_summary"] = make_pneumonia_summary()
        result = retrieval_agent(state)

        assert result["retrieval_sufficient"] is True
        assert len(result["retrieved_guidelines"]) > 0
        assert len(result["retrieval_queries"]) > 0

    @patch("agents.retrieval.get_fast_llm")
    @patch("agents.retrieval.multi_query_search")
    @patch("agents.retrieval.evaluate_relevance")
    def test_retries_on_low_relevance(self, mock_eval, mock_search, mock_get_llm):
        """Low relevance score → agent tries different queries."""
        from agents.retrieval import retrieval_agent
        from rag.store import Document

        call_count = [0]

        def mock_llm_side_effect(*args, **kwargs):
            import json
            call_count[0] += 1
            mock_response = MagicMock()
            mock_response.content = json.dumps([f"query attempt {call_count[0]}"])
            return mock_response

        mock_llm = MagicMock()
        mock_llm.invoke.side_effect = mock_llm_side_effect
        mock_get_llm.return_value = mock_llm

        mock_search.return_value = [
            Document(content="irrelevant content", metadata={"source_file": "test.pdf", "section": "test",
                     "page_number": 1, "chunk_index": 0, "guideline_type": "imnci"}, score=0.9, id="xyz")
        ]
        mock_eval.return_value = {"score": 1, "sufficient": False,
                                   "missing_aspects": ["classification criteria"], "reasoning": "Not relevant"}

        state = create_initial_state()
        state["patient_summary"] = make_pneumonia_summary()
        result = retrieval_agent(state)

        # Should have tried multiple query sets
        assert len(result["retrieval_queries"]) > 1

    @patch("agents.retrieval.get_fast_llm")
    @patch("agents.retrieval.multi_query_search")
    def test_handles_empty_vector_store(self, mock_search, mock_get_llm):
        """Empty vector store → returns empty guidelines gracefully."""
        from agents.retrieval import retrieval_agent

        mock_get_llm.return_value = self._mock_query_llm(["test query"])
        mock_search.return_value = []  # Empty store

        state = create_initial_state()
        state["patient_summary"] = make_pneumonia_summary()
        result = retrieval_agent(state)

        assert isinstance(result["retrieved_guidelines"], list)
        assert isinstance(result["retrieval_queries"], list)

    @patch("agents.retrieval.get_fast_llm")
    @patch("agents.retrieval.multi_query_search")
    @patch("agents.retrieval.evaluate_relevance")
    def test_serializes_documents_to_dicts(self, mock_eval, mock_search, mock_get_llm):
        """Documents are serialized to dicts for LangGraph state."""
        from agents.retrieval import retrieval_agent
        from rag.store import Document

        mock_get_llm.return_value = self._mock_query_llm(["IMNCI pneumonia"])
        mock_search.return_value = [
            Document(
                content="Test guideline content",
                metadata={"source_file": "imnci.pdf", "section": "Test Section",
                          "page_number": 1, "chunk_index": 0, "guideline_type": "imnci"},
                score=0.1,
                id="test_id",
            )
        ]
        mock_eval.return_value = {"score": 4, "sufficient": True, "missing_aspects": [], "reasoning": "Good"}

        state = create_initial_state()
        state["patient_summary"] = make_pneumonia_summary()
        result = retrieval_agent(state)

        # All guidelines should be dicts (not Document objects)
        for g in result["retrieved_guidelines"]:
            assert isinstance(g, dict)
            assert "content" in g
            assert "metadata" in g

    def test_handles_none_summary(self):
        """No patient summary → returns empty result gracefully."""
        from agents.retrieval import retrieval_agent
        state = create_initial_state()
        # patient_summary is None by default
        result = retrieval_agent(state)
        assert result["retrieved_guidelines"] == []
        assert result["retrieval_sufficient"] is False


# ═══════════════════════════════════════════════════════════════════════════════
# TEST GROUP 4: Triage Agent
# ═══════════════════════════════════════════════════════════════════════════════

class TestTriageAgent:
    """Test Triage Agent with mocked LLM structured output."""

    def _mock_triage_llm(self, result: TriageResult):
        """Build a mock LLM that returns the given TriageResult."""
        mock_structured = MagicMock()
        mock_structured.invoke.return_value = result
        mock_llm = MagicMock()
        mock_llm.with_structured_output.return_value = mock_structured
        return mock_llm

    def _make_state_with_guidelines(self, summary: PatientSummary) -> dict:
        state = create_initial_state()
        state["patient_summary"] = summary
        state["intake_complete"] = True
        state["retrieval_sufficient"] = True
        state["retrieved_guidelines"] = [{
            "content": "SEVERE PNEUMONIA: chest indrawing → urgent referral",
            "metadata": {"source_file": "imnci.pdf", "section": "Classify Cough",
                         "page_number": 1, "chunk_index": 0, "guideline_type": "imnci"},
            "score": 0.1,
            "id": "test_id",
        }]
        return state

    @patch("agents.triage.get_llm")
    def test_scenario_1_high_confidence_urgent(self, mock_get_llm):
        """Scenario 1: Severe pneumonia → URGENT, high confidence."""
        from agents.triage import triage_agent

        expected_result = make_triage_result(TriageLevel.URGENT, 0.92)
        mock_get_llm.return_value = self._mock_triage_llm(expected_result)

        state = self._make_state_with_guidelines(make_pneumonia_summary())
        result = triage_agent(state)

        assert result["triage_result"] is not None
        assert result["triage_result"].triage_level == TriageLevel.URGENT
        assert result["triage_confidence"] >= CONFIDENCE_THRESHOLD

    @patch("agents.triage.get_llm")
    def test_scenario_2_self_care(self, mock_get_llm):
        """Scenario 2: Mild cough → SELF_CARE."""
        from agents.triage import triage_agent

        expected_result = make_triage_result(TriageLevel.SELF_CARE, 0.85)
        mock_get_llm.return_value = self._mock_triage_llm(expected_result)

        state = self._make_state_with_guidelines(make_mild_cough_summary())
        result = triage_agent(state)

        assert result["triage_result"].triage_level == TriageLevel.SELF_CARE

    @patch("agents.triage.get_llm")
    def test_low_confidence_generates_follow_up_questions(self, mock_get_llm):
        """Low confidence → follow_up_questions are generated."""
        from agents.triage import triage_agent

        low_confidence_result = TriageResult(
            triage_level=TriageLevel.STANDARD,
            suspected_conditions=[
                SuspectedCondition(
                    name="Possible Pneumonia",
                    confidence=0.55,
                    supporting_symptoms=["cough"],
                    ruling_out=["respiratory rate not measured"],
                )
            ],
            recommended_actions=["Collect more information"],
            reasoning="Insufficient data for confident classification",
            guidelines_cited=[],
        )
        mock_get_llm.return_value = self._mock_triage_llm(low_confidence_result)

        state = self._make_state_with_guidelines(make_mild_cough_summary())
        state["cycle_count"] = 0  # First cycle, so follow-ups are appropriate
        result = triage_agent(state)

        assert result["triage_confidence"] < CONFIDENCE_THRESHOLD
        assert len(result["follow_up_questions"]) > 0

    @patch("agents.triage.get_llm")
    def test_handles_llm_failure_gracefully(self, mock_get_llm):
        """LLM failure → returns low-confidence error result, not exception."""
        from agents.triage import triage_agent

        mock_structured = MagicMock()
        mock_structured.invoke.side_effect = Exception("API error")
        mock_llm = MagicMock()
        mock_llm.with_structured_output.return_value = mock_structured
        mock_get_llm.return_value = mock_llm

        state = self._make_state_with_guidelines(make_pneumonia_summary())
        result = triage_agent(state)

        assert "triage_result" in result
        assert "triage_confidence" in result
        assert result["triage_confidence"] < CONFIDENCE_THRESHOLD

    @patch("agents.triage.get_llm")
    def test_no_summary_returns_error(self, mock_get_llm):
        """No patient summary → error result, not exception."""
        from agents.triage import triage_agent

        mock_get_llm.return_value = MagicMock()
        state = create_initial_state()
        result = triage_agent(state)

        assert "triage_result" in result
        assert result["triage_confidence"] < 0.5


# ═══════════════════════════════════════════════════════════════════════════════
# TEST GROUP 5: Red Flag Override (Safety Guardrail)
# ═══════════════════════════════════════════════════════════════════════════════

class TestRedFlagOverride:
    """Test the safety guardrail that overrides triage level for danger signs."""

    def test_escalates_self_care_to_urgent_when_red_flag(self):
        """If LLM says SELF_CARE but red flags present → must escalate to URGENT."""
        from agents.triage import _apply_red_flag_override

        # LLM incorrectly returned SELF_CARE despite chest indrawing
        wrong_result = make_triage_result(TriageLevel.SELF_CARE, 0.9)
        summary_with_flag = make_pneumonia_summary()  # Has chest indrawing in red_flags

        corrected = _apply_red_flag_override(wrong_result, summary_with_flag)

        assert corrected.triage_level in (TriageLevel.URGENT, TriageLevel.EMERGENCY)

    def test_escalates_standard_to_urgent_when_red_flag(self):
        """STANDARD with red flags → URGENT."""
        from agents.triage import _apply_red_flag_override

        wrong_result = make_triage_result(TriageLevel.STANDARD, 0.7)
        summary_with_flag = make_pneumonia_summary()
        corrected = _apply_red_flag_override(wrong_result, summary_with_flag)
        assert corrected.triage_level in (TriageLevel.URGENT, TriageLevel.EMERGENCY)

    def test_does_not_downgrade_emergency(self):
        """EMERGENCY stays EMERGENCY even with red flags."""
        from agents.triage import _apply_red_flag_override

        correct_result = make_triage_result(TriageLevel.EMERGENCY, 0.95)
        summary_with_flag = make_pneumonia_summary()
        result = _apply_red_flag_override(correct_result, summary_with_flag)
        assert result.triage_level == TriageLevel.EMERGENCY

    def test_no_change_without_red_flags(self):
        """No red flags → triage level unchanged."""
        from agents.triage import _apply_red_flag_override

        self_care_result = make_triage_result(TriageLevel.SELF_CARE, 0.85)
        clean_summary = make_mild_cough_summary()
        result = _apply_red_flag_override(self_care_result, clean_summary)
        assert result.triage_level == TriageLevel.SELF_CARE

    def test_override_adds_red_flag_to_actions(self):
        """Overridden result includes the red flag in recommended_actions."""
        from agents.triage import _apply_red_flag_override

        wrong_result = make_triage_result(TriageLevel.SELF_CARE, 0.9)
        summary_with_flag = make_pneumonia_summary()
        corrected = _apply_red_flag_override(wrong_result, summary_with_flag)

        # The override should mention the danger sign in actions
        combined_actions = " ".join(corrected.recommended_actions).lower()
        assert "refer" in combined_actions or "urgent" in combined_actions

    def test_override_preserves_original_reasoning(self):
        """Override preserves original reasoning but adds safety note."""
        from agents.triage import _apply_red_flag_override

        original_reasoning = "Original LLM reasoning step 1. Step 2."
        wrong_result = TriageResult(
            triage_level=TriageLevel.SELF_CARE,
            suspected_conditions=[],
            recommended_actions=["Rest and fluids"],
            reasoning=original_reasoning,
            guidelines_cited=[],
        )
        corrected = _apply_red_flag_override(wrong_result, make_pneumonia_summary())
        assert original_reasoning in corrected.reasoning
        assert "SAFETY OVERRIDE" in corrected.reasoning


# ═══════════════════════════════════════════════════════════════════════════════
# TEST GROUP 6: Confidence Computation
# ═══════════════════════════════════════════════════════════════════════════════

class TestConfidenceComputation:
    """Test the confidence scoring logic."""

    def test_high_confidence_with_clear_diagnosis(self):
        from agents.triage import _compute_confidence

        result = make_triage_result(TriageLevel.URGENT, 0.9)
        result.guidelines_cited = ["IMNCI Section"]
        summary = make_mild_cough_summary()  # No red flags
        confidence = _compute_confidence(result, summary)
        assert confidence >= 0.85  # High from 0.9 base + guidelines cited boost

    def test_low_confidence_with_empty_conditions(self):
        from agents.triage import _compute_confidence

        result = TriageResult(
            triage_level=TriageLevel.STANDARD,
            suspected_conditions=[],  # No conditions identified
            recommended_actions=[],
            reasoning="No conditions identified",
            guidelines_cited=[],
        )
        summary = make_mild_cough_summary()
        confidence = _compute_confidence(result, summary)
        assert confidence == 0.4

    def test_red_flag_boosts_confidence(self):
        from agents.triage import _compute_confidence

        result = make_triage_result(TriageLevel.URGENT, 0.75)
        summary = make_pneumonia_summary()  # Has chest indrawing red flag
        confidence = _compute_confidence(result, summary)
        assert confidence > 0.75  # Boosted by red flag presence

    def test_low_completeness_penalizes_confidence(self):
        from agents.triage import _compute_confidence

        result = make_triage_result(TriageLevel.STANDARD, 0.7)
        summary = make_pneumonia_summary(complete=False)  # completeness_score = 0.35
        confidence = _compute_confidence(result, summary)
        assert confidence <= 0.7  # Penalized for low completeness


# ═══════════════════════════════════════════════════════════════════════════════
# TEST GROUP 7: Agent Routing Flows (Integration)
# ═══════════════════════════════════════════════════════════════════════════════

class TestAgentRoutingFlows:
    """
    Test complete routing sequences through the supervisor.
    These don't call actual agents — they simulate state changes
    and verify the supervisor routes correctly at each step.
    """

    def test_happy_path_flow(self):
        """Happy path: intake → retrieval → triage → output."""
        from agents.supervisor import route_next

        # Start
        state = create_initial_state()
        assert route_next(state) == "intake"

        # After intake
        state["intake_complete"] = True
        state["patient_summary"] = make_mild_cough_summary()
        assert route_next(state) == "retrieval"

        # After retrieval
        state["retrieval_sufficient"] = True
        state["retrieved_guidelines"] = [{"content": "x", "metadata": {}, "score": 0.1, "id": "x"}]
        assert route_next(state) == "triage"

        # After triage (high confidence)
        state["triage_result"] = make_triage_result(TriageLevel.SELF_CARE, 0.88)
        state["triage_confidence"] = 0.88
        assert route_next(state) == "output"

    def test_feedback_loop_flow(self):
        """Feedback loop: intake → retrieval → triage (low conf) → intake → triage → output."""
        from agents.supervisor import route_next

        state = create_initial_state()
        state["intake_complete"] = True
        state["patient_summary"] = make_mild_cough_summary()
        state["retrieval_sufficient"] = True
        state["retrieved_guidelines"] = [{"content": "x", "metadata": {}, "score": 0.1, "id": "x"}]

        # First triage: low confidence
        state["triage_result"] = make_triage_result(TriageLevel.STANDARD, 0.5)
        state["triage_confidence"] = 0.5
        state["cycle_count"] = 0
        assert route_next(state) == "intake"  # Loop back

        # After second intake
        state["intake_complete"] = True
        state["cycle_count"] = 1
        state["phase"] = Phase.INTAKE.value
        # Reset triage result to trigger new triage
        state["triage_result"] = None
        assert route_next(state) == "triage"

        # Second triage: high confidence
        state["triage_result"] = make_triage_result(TriageLevel.STANDARD, 0.8)
        state["triage_confidence"] = 0.8
        assert route_next(state) == "output"

    def test_max_cycles_escape(self):
        """After max cycles, always go to output regardless of confidence."""
        from agents.supervisor import route_next

        state = create_initial_state()
        state["intake_complete"] = True
        state["retrieval_sufficient"] = True
        state["retrieved_guidelines"] = [{"content": "x", "metadata": {}, "score": 0.1, "id": "x"}]
        state["triage_result"] = make_triage_result(TriageLevel.STANDARD, 0.4)
        state["triage_confidence"] = 0.4  # Still low
        state["cycle_count"] = 2
        state["max_cycles"] = 3
        # At max cycles: cycle_count (2) >= max_cycles (3) - 1 → output
        assert route_next(state) == "output"
