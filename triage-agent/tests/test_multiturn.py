"""
tests/test_multiturn.py — Multi-turn flow on the REAL compiled graph.

These tests use graph.create_graph() and graph.run_turn() unmodified, with a
real MemorySaver checkpointer. Only the LLM (StubLLM) and the vector search are
replaced, so routing, state merging and persistence are all exercised for real.

Regression coverage for three production bugs:
  1. run_turn() never reached the intake agent (interrupt_before=["intake"] +
     re-invoking with new input restarted at the supervisor every turn).
  2. After the low-confidence feedback loop, route_next kept returning
     "intake" so triage never re-ran.
  3. The intake question call reused EXTRACTION_SYSTEM and once showed a raw
     JSON patient summary to the user.
"""

from __future__ import annotations

import sys
import uuid
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).parent.parent))

from rag.store import Document  # noqa: E402
from state import (  # noqa: E402
    DangerSign,
    DangerSignFinding,
    FindingStatus,
    PatientSummary,
    SuspectedCondition,
    Symptom,
    TriageLevel,
    TriageResult,
    VitalSigns,
)
from tests.stub_llm import StubLLM  # noqa: E402

DOC = Document(
    content="CLASSIFY COUGH OR DIFFICULT BREATHING: fast breathing = PNEUMONIA.",
    metadata={"source_file": "sample_imnci.txt", "section": "ASSESS AND CLASSIFY COUGH OR DIFFICULT BREATHING",
              "guideline_type": "imnci"},
    score=0.2, id="doc-1",
)


# Phase 2: a 3-year-old with cough needs the general-danger and breathing
# checklist groups answered before intake may complete (safety.py).
CHECKLIST_ALL_ABSENT = [DangerSignFinding(sign=s, status=FindingStatus.ABSENT) for s in (
    DangerSign.UNABLE_TO_DRINK_OR_BREASTFEED, DangerSign.VOMITS_EVERYTHING, DangerSign.CONVULSIONS,
    DangerSign.LETHARGIC_OR_UNCONSCIOUS, DangerSign.CHEST_INDRAWING, DangerSign.STRIDOR_WHEN_CALM)]


def summary(completeness: float, **kw) -> PatientSummary:
    base = dict(age=3, chief_complaint="3 year old with cough for 4 days",
                symptoms=[Symptom(name="cough", duration="4 days")],
                vital_signs=VitalSigns(respiratory_rate=36), completeness_score=completeness,
                danger_sign_findings=list(CHECKLIST_ALL_ABSENT))
    base.update(kw)
    return PatientSummary(**base)


def triage(level=TriageLevel.URGENT, conf=0.9, cited=("ASSESS AND CLASSIFY COUGH OR DIFFICULT BREATHING",),
           ruling_out=()) -> TriageResult:
    return TriageResult(
        triage_level=level,
        suspected_conditions=[SuspectedCondition(name="Pneumonia", confidence=conf, ruling_out=list(ruling_out))],
        recommended_actions=["Give an appropriate oral antibiotic per protocol"],
        reasoning="stub reasoning", guidelines_cited=list(cited),
    )


@contextmanager
def stubbed(llm: StubLLM):
    with patch("agents.intake.get_fast_llm", return_value=llm), \
         patch("agents.retrieval.get_fast_llm", return_value=llm), \
         patch("agents.triage.get_llm", return_value=llm), \
         patch("agents.retrieval.multi_query_search", return_value=[DOC]), \
         patch("agents.retrieval.evaluate_relevance", return_value={"score": 4}):
        yield


def test_run_turn_real_graph_three_turns_reaches_intake_and_completes():
    """3 user turns through run_turn(): intake runs every turn, turn 3 completes the triage."""
    from graph import create_graph, run_turn

    llm = StubLLM(
        structured={"PatientSummary": [summary(0.3), summary(0.5), summary(0.9)],
                    "TriageResult": [triage(conf=0.9)]},
        questions=["How many breaths does the child take in one minute?",
                   "Is the child able to drink or breastfeed?"],
    )
    app, tid = create_graph(), str(uuid.uuid4())
    with stubbed(llm):
        r1 = run_turn(app, tid, "My 3 year old has had a cough for 4 days.")
        r2 = run_turn(app, tid, "About 46 breaths a minute.")
        r3 = run_turn(app, tid, "Yes, she is drinking normally.")

    # Turn 1 and 2: intake ran and its question is what the user sees
    assert r1["message"] == "How many breaths does the child take in one minute?"
    assert r2["message"] == "Is the child able to drink or breastfeed?"
    assert not r1["is_complete"] and not r2["is_complete"]
    assert r1["phase"] == r2["phase"] == "intake"
    # Turn 3: intake completes and the same run goes retrieval → triage → output
    assert r3["is_complete"] and r3["phase"] == "done"
    assert "URGENT" in r3["triage_card"]
    assert llm.errors == []
    # Exactly one extraction per turn; one triage call in total
    assert llm.count("structured:PatientSummary") == 3
    assert llm.count("structured:TriageResult") == 1
    # The intake on turn 2 saw the user's turn-2 answer (state persisted across runs)
    turn2_extraction = llm.calls("structured:PatientSummary")[1]["messages"]
    assert any("46 breaths" in str(m.content) for m in turn2_extraction)
    # Conversation history accumulated in the checkpointer
    contents = [str(m.content) for m in r3["state"]["messages"]]
    assert contents[0] == "My 3 year old has had a cough for 4 days."
    assert "About 46 breaths a minute." in contents and "Yes, she is drinking normally." in contents


def test_run_turn_sessions_are_isolated_by_thread_id():
    from graph import create_graph, run_turn

    llm = StubLLM(structured={"PatientSummary": [summary(0.3)]}, questions=["How old is the patient?"])
    app = create_graph()
    with stubbed(llm):
        run_turn(app, "thread-a", "child has cough")
        r_b = run_turn(app, "thread-b", "adult has headache")
    msgs_b = [str(m.content) for m in r_b["state"]["messages"]]
    assert "child has cough" not in msgs_b
    assert msgs_b[0] == "adult has headache"


def test_low_confidence_loop_reruns_triage():
    """Low confidence → intake asks triage's follow-up → user answers → triage RE-RUNS and completes."""
    from graph import create_graph, run_turn

    llm = StubLLM(
        structured={
            "PatientSummary": [summary(0.9), summary(0.9)],
            # first triage: low confidence (0.3, nothing cited) → loop; second: confident
            "TriageResult": [triage(TriageLevel.STANDARD, conf=0.3, cited=(), ruling_out=("chest indrawing",)),
                             triage(TriageLevel.URGENT, conf=0.9)],
        },
        questions=["Does the lower chest pull in when the child breathes in?"],
    )
    app, tid = create_graph(), str(uuid.uuid4())
    with stubbed(llm):
        r1 = run_turn(app, tid, "3 year old, cough 4 days, 46 breaths per minute, drinking well")
        assert not r1["is_complete"], "low confidence must loop back for more information"
        assert r1["phase"] == "intake"
        assert r1["state"]["cycle_count"] == 1
        assert r1["state"]["triage_result"] is None, "stale result must be cleared on loop-back"
        queued = len(r1["state"]["follow_up_questions"]) + 1  # +1: the one just asked
        assert r1["message"] == "Does the lower chest pull in when the child breathes in?"
        answers = 0
        r2 = r1
        while not r2["is_complete"] and answers < 5:
            r2 = run_turn(app, tid, "No.")
            answers += 1

    assert r2["is_complete"] and r2["phase"] == "done"
    assert llm.errors == [], "no LLM request may be rejected (e.g. ending with a model turn)"
    assert answers == queued, "one user answer per queued follow-up question"
    assert llm.count("structured:TriageResult") == 2, "triage must re-run after the feedback loop"
    assert r2["state"]["triage_confidence"] >= 0.75
    assert r2["state"]["triage_result"].triage_level == TriageLevel.URGENT


def test_low_confidence_loop_terminates_at_max_cycles():
    """Confidence never rises → triage runs max_cycles times, then outputs best effort."""
    from graph import create_graph, run_turn

    llm = StubLLM(
        structured={"PatientSummary": [summary(0.9)],
                    "TriageResult": [triage(TriageLevel.STANDARD, conf=0.3, cited=(), ruling_out=("stridor",))]},
        questions=["Is there a harsh noise when the child breathes in?"],
    )
    app, tid = create_graph(), str(uuid.uuid4())
    replies = ["3 year old, cough, 46 breaths per minute"] + ["No."] * 10
    with stubbed(llm):
        for msg in replies:
            r = run_turn(app, tid, msg)
            if r["is_complete"]:
                break
    assert r["is_complete"] and r["phase"] == "done"
    assert llm.errors == []
    assert llm.count("structured:TriageResult") == 3  # max_cycles = 3
    assert r["state"]["cycle_count"] == 2


def test_supervisor_loop_back_clears_stale_triage_and_retrieval():
    from agents.supervisor import route_next, supervisor
    from state import Phase, create_initial_state

    s = create_initial_state()
    s.update(phase=Phase.TRIAGE.value, intake_complete=True, retrieval_sufficient=True,
             retrieved_guidelines=[DOC.to_dict()], triage_result=triage(conf=0.3), triage_confidence=0.3)
    upd = supervisor(s)
    assert upd["current_agent"] == "intake"
    assert upd["triage_result"] is None and upd["retrieved_guidelines"] == []
    assert upd["cycle_count"] == 1 and upd["intake_complete"] is False
    # The conditional edge evaluates the UPDATED state and must agree
    s.update(upd)
    assert route_next(s) == "intake"
    # After intake completes again: retrieval, then triage — not intake forever
    s["intake_complete"] = True
    assert route_next(s) == "retrieval"
    s.update(retrieved_guidelines=[DOC.to_dict()], retrieval_sufficient=True)
    assert route_next(s) == "triage"


def test_route_next_ends_run_while_awaiting_user():
    from agents.supervisor import route_next
    from state import create_initial_state

    s = create_initial_state()
    s["awaiting_user"] = True
    assert route_next(s) == "end"


def test_intake_question_uses_question_prompt_not_extraction_prompt():
    from agents.intake import intake_agent
    from langchain_core.messages import HumanMessage
    from prompts.intake_prompt import EXTRACTION_SYSTEM, QUESTION_SYSTEM
    from state import create_initial_state

    llm = StubLLM(structured={"PatientSummary": [summary(0.3)]}, questions=["How old is the child?"])
    s = create_initial_state()
    s["messages"] = [HumanMessage(content="child has cough")]
    with patch("agents.intake.get_fast_llm", return_value=llm):
        out = intake_agent(s)
    q_call = llm.calls("question")[0]
    assert q_call["system"] == QUESTION_SYSTEM
    assert q_call["system"] != EXTRACTION_SYSTEM
    assert out["awaiting_user"] is True
    assert out["messages"][0].content == "How old is the child?"


def test_intake_never_shows_json_to_the_user():
    """If the model still returns a JSON summary instead of a question, the user gets a plain question."""
    from agents.intake import intake_agent
    from langchain_core.messages import HumanMessage
    from prompts.intake_prompt import FALLBACK_QUESTION
    from state import create_initial_state

    leaked = '```json\n{\n  "chief_complaint": "18-month-old with fever",\n  "patient_age": "18 months"\n}\n```'
    for bad in [leaked, '{"chief_complaint": "fever"}', "", '"age": 2, "sex": "female"']:
        llm = StubLLM(structured={"PatientSummary": [summary(0.3)]}, questions=[bad])
        s = create_initial_state()
        s["messages"] = [HumanMessage(content="baby has fever and fits")]
        with patch("agents.intake.get_fast_llm", return_value=llm):
            out = intake_agent(s)
        shown = out["messages"][0].content
        assert shown == FALLBACK_QUESTION
        assert "{" not in shown and "```" not in shown


def test_clean_question_keeps_normal_questions():
    from agents.intake import clean_question
    for q in ["How many days has she had the fever?",
              "Please check: does the lower chest pull in when the child breathes in?",
              "Is the child able to drink [water or breast milk]?"]:
        assert clean_question(q) == q


def test_follow_up_questions_are_asked_even_when_summary_looks_complete():
    """After a loop-back, intake must ask triage's follow-up instead of completing immediately."""
    from agents.intake import intake_agent
    from langchain_core.messages import HumanMessage
    from state import create_initial_state

    llm = StubLLM(structured={"PatientSummary": [summary(0.95)]}, questions=["Is there stridor when calm?"])
    s = create_initial_state()
    s["messages"] = [HumanMessage(content="3 year old with cough")]
    s["follow_up_questions"] = ["Is the following sign present: stridor?"]
    with patch("agents.intake.get_fast_llm", return_value=llm):
        out = intake_agent(s)
    assert out["intake_complete"] is False and out["awaiting_user"] is True
    assert out["follow_up_questions"] == []
    assert llm.count("completion") == 0


# ─── Phase 5: relevance judge runs in parallel with triage ──────────────────

def test_judge_and_triage_run_in_parallel():
    """With judge and triage each taking 0.4 s, the completing turn takes ~0.4 s, not ~0.8 s."""
    import time as _time
    from graph import create_graph, run_turn

    def slow_judge(*a, **k):
        _time.sleep(0.4)
        return {"score": 4, "missing_aspects": []}

    def slow_triage():
        _time.sleep(0.4)
        return triage(conf=0.9)

    llm = StubLLM(structured={"PatientSummary": [summary(0.9, search_queries=["IMNCI classify cough"])],
                              "TriageResult": [slow_triage]})
    app = create_graph()
    with stubbed(llm), patch("agents.retrieval.evaluate_relevance", side_effect=slow_judge):
        t0 = _time.perf_counter()
        r = run_turn(app, str(uuid.uuid4()), "3 year old, cough, 36/min, drinking well, alert")
        elapsed = _time.perf_counter() - t0
    assert r["is_complete"]
    assert elapsed < 0.75, f"judge and triage did not overlap ({elapsed:.2f} s)"
    assert llm.count("query_gen") == 0, "attempt 1 must use the extraction's search_queries"


def test_judge_rejection_discards_speculative_triage_and_reformulates():
    from graph import create_graph, run_turn

    scores = iter([2, 4])  # attempt 1 rejected, attempt 2 accepted
    llm = StubLLM(structured={"PatientSummary": [summary(0.9, search_queries=["first query"])],
                              "TriageResult": [triage(TriageLevel.STANDARD, conf=0.9), triage(TriageLevel.URGENT, conf=0.9)]})
    app = create_graph()
    with stubbed(llm), patch("agents.retrieval.evaluate_relevance",
                             side_effect=lambda *a, **k: {"score": next(scores), "missing_aspects": ["thresholds"]}):
        r = run_turn(app, str(uuid.uuid4()), "3 year old, cough, 36/min, drinking well, alert")
    st = r["state"]
    assert r["is_complete"]
    assert st["retrieval_attempts"] == 2
    assert st["retrieval_queries"][0] == "first query" and len(st["retrieval_queries"]) == 2
    assert llm.count("query_gen") == 1, "reformulation asks the LLM for new queries"
    assert llm.count("structured:TriageResult") == 2
    assert st["triage_result"].triage_level == TriageLevel.URGENT, "result from the accepted guidelines is used"


def test_judge_rejects_every_attempt_stops_at_max_attempts():
    """Every attempt brings new chunks but the judge keeps rejecting → stop at MAX_RETRIEVAL_ATTEMPTS."""
    from graph import create_graph, run_turn
    from rag.store import Document

    ids = iter(range(100))
    new_docs = lambda *a, **k: [Document(content="c", metadata={"section": "s"}, id=f"d{next(ids)}")]  # noqa: E731
    llm = StubLLM(structured={"PatientSummary": [summary(0.9, search_queries=["q1"])],
                              "TriageResult": [triage(conf=0.9)]})
    app = create_graph()
    with stubbed(llm), patch("agents.retrieval.multi_query_search", side_effect=new_docs), \
         patch("agents.retrieval.evaluate_relevance", side_effect=lambda *a, **k: {"score": 1, "missing_aspects": []}):
        r = run_turn(app, str(uuid.uuid4()), "3 year old, cough, 36/min, drinking well, alert")
    assert r["is_complete"]
    assert r["state"]["retrieval_attempts"] == 3
    assert llm.count("structured:TriageResult") == 3


def test_intake_extraction_retries_once_before_falling_back():
    """A single transient extraction failure must not degrade the summary (seen twice in the Phase 5 eval)."""
    from agents.intake import intake_agent
    from langchain_core.messages import HumanMessage
    from state import create_initial_state

    calls = {"n": 0}

    def flaky():
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("Error code: 400")
        return summary(0.9)

    llm = StubLLM(structured={"PatientSummary": [flaky, flaky]})
    s = create_initial_state()
    s["messages"] = [HumanMessage(content="3 year old with cough")]
    with patch("agents.intake.get_fast_llm", return_value=llm):
        out = intake_agent(s)
    assert calls["n"] == 2
    assert out["patient_summary"].completeness_score == 0.9, "used the retried result, not the 0.1 fallback"


def test_retrieval_retry_stops_when_no_new_chunks():
    """Judge rejects, but the reformulated search returns the same chunks → stop after attempt 2."""
    from graph import create_graph, run_turn

    llm = StubLLM(structured={"PatientSummary": [summary(0.9, search_queries=["q1"])],
                              "TriageResult": [triage(conf=0.9)]})
    app = create_graph()
    with stubbed(llm), patch("agents.retrieval.evaluate_relevance",
                             side_effect=lambda *a, **k: {"score": 1, "missing_aspects": []}):
        r = run_turn(app, str(uuid.uuid4()), "3 year old, cough, 36/min, drinking well, alert")
    st = r["state"]
    assert r["is_complete"]
    assert st["retrieval_attempts"] == 2 and st["retrieval_exhausted"] is True
    assert llm.count("structured:TriageResult") == 2


def test_retrieval_retry_continues_when_new_chunks_appear():
    from graph import create_graph, run_turn
    from rag.store import Document

    batches = iter([[DOC], [Document(content="other", metadata={"section": "S2"}, id="doc-2")],
                    [Document(content="third", metadata={"section": "S3"}, id="doc-3")]])
    llm = StubLLM(structured={"PatientSummary": [summary(0.9, search_queries=["q1"])],
                              "TriageResult": [triage(conf=0.9)]})
    app = create_graph()
    with stubbed(llm), \
         patch("agents.retrieval.multi_query_search", side_effect=lambda *a, **k: next(batches)), \
         patch("agents.retrieval.evaluate_relevance", side_effect=lambda *a, **k: {"score": 1, "missing_aspects": []}):
        r = run_turn(app, str(uuid.uuid4()), "3 year old, cough, 36/min, drinking well, alert")
    assert r["state"]["retrieval_attempts"] == 3  # every retry brought a new chunk → full 3 attempts


def test_extraction_request_ends_with_user_turn_even_after_ai_message():
    from agents.intake import intake_agent
    from langchain_core.messages import AIMessage, HumanMessage
    from state import create_initial_state

    llm = StubLLM(structured={"PatientSummary": [summary(0.9)]})
    s = create_initial_state()
    s["messages"] = [HumanMessage(content="3 year old with cough"), AIMessage(content="Thank you, assessing now.")]
    s["follow_up_questions"] = ["Is there stridor?"]
    with patch("agents.intake.get_fast_llm", return_value=llm):
        intake_agent(s)
    assert llm.errors == []
    sent = llm.calls("structured:PatientSummary")[0]["messages"]
    assert isinstance(sent[-1], HumanMessage)
