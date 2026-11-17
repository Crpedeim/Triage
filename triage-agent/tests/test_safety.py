"""
tests/test_safety.py — Phase 2 safety layer.

Covers: the fixed danger-sign checklist intake must cover before completing,
deterministic red flags from RED_FLAG_KEYWORDS, the fast-breathing thresholds
from RESPIRATORY_RATE_THRESHOLDS, the rule-based triage floor (EMERGENCY for
danger signs), and non-saturating confidence. Includes a real-graph run that
reproduces the Phase 0 under-triage of case A (chest indrawing never asked).
"""

from __future__ import annotations

import sys
import uuid
from pathlib import Path
from unittest.mock import patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

import safety  # noqa: E402
from state import (  # noqa: E402
    RED_FLAG_KEYWORDS,
    DangerSign,
    DangerSignFinding,
    FindingStatus,
    PatientSummary,
    SuspectedCondition,
    Symptom,
    TriageLevel,
    TriageResult,
    VitalSigns,
    create_initial_state,
)
from tests.stub_llm import StubLLM  # noqa: E402
from tests.test_multiturn import stubbed  # noqa: E402

P, A = FindingStatus.PRESENT, FindingStatus.ABSENT
GENERAL = [DangerSign.UNABLE_TO_DRINK_OR_BREASTFEED, DangerSign.VOMITS_EVERYTHING,
           DangerSign.CONVULSIONS, DangerSign.LETHARGIC_OR_UNCONSCIOUS]
BREATH = [DangerSign.CHEST_INDRAWING, DangerSign.STRIDOR_WHEN_CALM]


def F(**statuses) -> list[DangerSignFinding]:
    return [DangerSignFinding(sign=DangerSign(k), status=v) for k, v in statuses.items()]


def all_absent(signs) -> dict:
    return {s.value: A for s in signs}


def child(age=3, complaint="cough for 4 days", symptoms=("cough",), rr=None, findings=None,
          completeness=0.9, **kw) -> PatientSummary:
    return PatientSummary(age=age, chief_complaint=complaint,
                          symptoms=[Symptom(name=s) for s in symptoms],
                          vital_signs=VitalSigns(respiratory_rate=rr) if rr is not None else None,
                          danger_sign_findings=findings or [], completeness_score=completeness, **kw)


def result(level=TriageLevel.STANDARD, confs=(0.95,), cited=("IMNCI",)) -> TriageResult:
    return TriageResult(triage_level=level,
                        suspected_conditions=[SuspectedCondition(name=f"c{i}", confidence=c) for i, c in enumerate(confs)],
                        recommended_actions=["action"], reasoning="llm reasoning", guidelines_cited=list(cited))


# ─── Checklist applicability ─────────────────────────────────────────────────

def test_child_with_cough_needs_general_and_breathing_groups():
    ids = [g.id for g in safety.applicable_groups(child())]
    assert ids == ["general_danger_signs", "breathing"]


def test_child_with_diarrhoea_and_fever_groups():
    s = child(age=2, complaint="diarrhoea for 3 days and fever", symptoms=("diarrhea", "fever"))
    assert [g.id for g in safety.applicable_groups(s)] == ["general_danger_signs", "dehydration", "fever"]


def test_adult_gets_adult_group_and_unknown_age_gets_none():
    assert [g.id for g in safety.applicable_groups(child(age=55, complaint="headache"))] == ["adult_danger_signs"]
    assert safety.applicable_groups(PatientSummary(chief_complaint="cough")) == []


def test_age_months_takes_precedence_and_infant_is_under_five():
    s = PatientSummary(age=0, age_months=7, chief_complaint="cough", symptoms=[Symptom(name="cough")])
    assert safety.is_under_five(s) is True and safety.age_in_months(s) == 7


# ─── Intake must cover the checklist before completing ───────────────────────

def _intake(summary_obj, asked=(), followups=()):
    from agents.intake import intake_agent
    from langchain_core.messages import HumanMessage
    llm = StubLLM(structured={"PatientSummary": [summary_obj]}, questions=["How long has the child been ill?"])
    st = create_initial_state()
    st["messages"] = [HumanMessage(content="child has cough")]
    st["checklist_asked"] = list(asked)
    st["follow_up_questions"] = list(followups)
    with patch("agents.intake.get_fast_llm", return_value=llm):
        return intake_agent(st), llm


def test_intake_does_not_complete_until_checklist_groups_asked():
    s = child(completeness=0.95)  # LLM thinks it's complete, but no danger signs were asked
    out, llm = _intake(s)
    assert out["intake_complete"] is False and out["awaiting_user"] is True
    assert out["messages"][0].content == safety.GENERAL_DANGER.question
    assert out["checklist_asked"] == ["general_danger_signs"]
    assert llm.count("question") == 0, "checklist questions are fixed text, no LLM call"

    out2, _ = _intake(s, asked=["general_danger_signs"])
    assert out2["messages"][0].content == safety.BREATHING.question
    assert out2["checklist_asked"] == ["general_danger_signs", "breathing"]

    out3, llm3 = _intake(s, asked=["general_danger_signs", "breathing"])
    assert out3["intake_complete"] is True
    from prompts.intake_prompt import COMPLETION_MESSAGE
    assert out3["messages"][0].content == COMPLETION_MESSAGE
    assert llm3.count("completion") == 0  # Phase 5: fixed text, no LLM call


def test_group_already_answered_is_not_asked():
    s = child(rr=36, findings=F(**all_absent(GENERAL + BREATH)), completeness=0.9)
    out, _ = _intake(s)
    assert out["intake_complete"] is True


def test_unanswered_item_is_not_reasked_and_is_reported():
    # Asked, caregiver didn't know about stridor → still unknown, but intake completes
    s = child(rr=36, findings=F(**all_absent(GENERAL), chest_indrawing=A), completeness=0.9)
    out, _ = _intake(s, asked=["general_danger_signs", "breathing"])
    assert out["intake_complete"] is True
    assert safety.unassessed_items(out["patient_summary"]) == ["stridor_when_calm"]
    from output import format_triage_card
    st = create_initial_state()
    st.update(patient_summary=out["patient_summary"], triage_result=result(), triage_confidence=0.8)
    assert "DANGER SIGNS NOT ASSESSED" in format_triage_card(st)
    assert "stridor when calm" in format_triage_card(st)


def test_coverage_counts_respiratory_rate():
    s = child(findings=F(**all_absent(GENERAL + BREATH)))
    assert safety.checklist_coverage(s) == pytest.approx(6 / 7)  # RR not counted yet
    assert safety.checklist_coverage(child(rr=30, findings=F(**all_absent(GENERAL + BREATH)))) == 1.0


# ─── Deterministic red flags (RED_FLAG_KEYWORDS) ─────────────────────────────

def test_every_red_flag_keyword_is_recognised():
    for kw in RED_FLAG_KEYWORDS:
        assert kw in safety.canonical_red_flags_in([kw]), kw


def test_lay_phrases_and_negation():
    assert safety.canonical_red_flags_in(["had two fits today"]) == {"convulsions"}
    assert safety.canonical_red_flags_in(["chest pulls in when breathing"]) == {"chest indrawing"}
    assert safety.canonical_red_flags_in(["very floppy"]) == {"lethargic"}
    assert safety.canonical_red_flags_in(["no convulsions", "not lethargic", "chest indrawing absent"]) == set()


def test_derive_red_flags_from_findings_and_text_absent_wins():
    s = child(findings=F(convulsions=P), red_flags=["lethargy"])
    assert safety.derive_red_flags(s) == ["convulsions", "lethargic"]
    s2 = child(findings=F(lethargic_or_unconscious=A), red_flags=["lethargy"])
    assert safety.derive_red_flags(s2) == []


# ─── Fast breathing (RESPIRATORY_RATE_THRESHOLDS) ───────────────────────────

@pytest.mark.parametrize("months,rr,fast", [
    (1, 59, False), (1, 60, True),     # < 2 months: >= 60
    (6, 49, False), (6, 50, True),     # 2-12 months: >= 50
    (36, 39, False), (36, 40, True),   # 1-5 years: >= 40
    (72, 60, False),                   # >= 5 years: rule not applicable
])
def test_fast_breathing_thresholds(months, rr, fast):
    s = PatientSummary(age=months // 12, age_months=months, chief_complaint="cough",
                       vital_signs=VitalSigns(respiratory_rate=rr))
    assert safety.has_fast_breathing(s) is fast


# ─── Triage floor ────────────────────────────────────────────────────────────

@pytest.mark.parametrize("summary_kw,expected", [
    (dict(findings=F(convulsions=P)), TriageLevel.EMERGENCY),
    (dict(findings=F(unable_to_drink_or_breastfeed=P)), TriageLevel.EMERGENCY),
    (dict(findings=F(chest_indrawing=P)), TriageLevel.EMERGENCY),
    (dict(findings=F(stridor_when_calm=P)), TriageLevel.EMERGENCY),
    (dict(rr=46), TriageLevel.URGENT),                                           # pneumonia
    (dict(rr=30), None),                                                         # no pneumonia
    (dict(complaint="diarrhoea", symptoms=("diarrhea",),
          findings=F(sunken_eyes=P, skin_pinch_very_slow=P)), TriageLevel.EMERGENCY),  # severe dehydration
    (dict(complaint="diarrhoea", symptoms=("diarrhea",),
          findings=F(sunken_eyes=P)), None),                                     # one sign only
    (dict(complaint="diarrhoea", symptoms=("diarrhea",),
          findings=F(restless_irritable=P, drinks_eagerly=P)), TriageLevel.URGENT),  # some dehydration
    (dict(complaint="fever", symptoms=("fever",), findings=F(stiff_neck=P)), TriageLevel.EMERGENCY),
])
def test_triage_floor_child_rules(summary_kw, expected):
    floor, reasons = safety.triage_floor(child(**summary_kw))
    assert floor == expected
    assert bool(reasons) == (expected is not None)


def test_triage_floor_adult_rules():
    adult = lambda **kw: PatientSummary(age=55, chief_complaint=kw.pop("cc", "headache for a week"), **kw)  # noqa: E731
    assert safety.triage_floor(adult(danger_sign_findings=F(chest_pain=P)))[0] == TriageLevel.EMERGENCY
    crisis = adult(vital_signs=VitalSigns(blood_pressure_systolic=190, blood_pressure_diastolic=125))
    assert safety.triage_floor(crisis)[0] == TriageLevel.EMERGENCY          # PEN crisis with headache
    grade2 = adult(vital_signs=VitalSigns(blood_pressure_systolic=160, blood_pressure_diastolic=100))
    assert safety.triage_floor(grade2)[0] is None                          # grade 2, no emergency features
    no_symptoms = adult(cc="routine check", vital_signs=VitalSigns(blood_pressure_systolic=190, blood_pressure_diastolic=125))
    assert safety.triage_floor(no_symptoms)[0] is None                     # crisis needs symptoms


# ─── Override ────────────────────────────────────────────────────────────────

def test_override_escalates_danger_sign_to_emergency_not_just_urgent():
    from agents.triage import _apply_red_flag_override
    for llm_level in (TriageLevel.SELF_CARE, TriageLevel.STANDARD, TriageLevel.URGENT):
        out = _apply_red_flag_override(result(llm_level), child(findings=F(chest_indrawing=P)))
        assert out.triage_level == TriageLevel.EMERGENCY
        assert out.reasoning.startswith("[SAFETY OVERRIDE]")
        assert out.recommended_actions[0].startswith("EMERGENCY")
        assert out.referral_note


def test_override_fast_breathing_raises_standard_to_urgent_only():
    from agents.triage import _apply_red_flag_override
    out = _apply_red_flag_override(result(TriageLevel.STANDARD), child(rr=46))
    assert out.triage_level == TriageLevel.URGENT


def test_override_never_downgrades():
    from agents.triage import _apply_red_flag_override
    r = result(TriageLevel.EMERGENCY)
    assert _apply_red_flag_override(r, child(rr=46)) is r
    r2 = result(TriageLevel.URGENT)
    assert _apply_red_flag_override(r2, child(rr=30)) is r2


# ─── Confidence no longer saturates ─────────────────────────────────────────

def test_confidence_capped_below_one_even_with_red_flags_and_citations():
    from agents.triage import CONFIDENCE_CAP, _compute_confidence
    s = child(rr=50, findings=F(chest_indrawing=P, **all_absent(GENERAL), stridor_when_calm=A), red_flags=["chest indrawing"])
    c = _compute_confidence(result(confs=(1.0,)), s)
    assert c == CONFIDENCE_CAP < 1.0


def test_confidence_reflects_ambiguity_coverage_and_citations():
    from agents.triage import _compute_confidence
    full = child(rr=30, findings=F(**all_absent(GENERAL + BREATH)))
    base = _compute_confidence(result(confs=(0.9,)), full)
    ambiguous = _compute_confidence(result(confs=(0.9, 0.85)), full)
    uncovered = _compute_confidence(result(confs=(0.9,)), child())
    uncited = _compute_confidence(result(confs=(0.9,), cited=()), full)
    assert base == 0.9
    assert ambiguous == pytest.approx(0.8)          # gap 0.05 < 0.15 → -0.10
    assert uncovered == pytest.approx(0.54)         # 0.9 × 0.6 (no checklist answers)
    assert uncited == pytest.approx(0.8)
    assert len({base, ambiguous, uncovered}) == 3


# ─── Real graph: Phase 0 case A no longer under-triaged ─────────────────────

def test_case_a_chest_indrawing_reported_via_checklist_becomes_emergency():
    """
    Phase 0 case A: intake never asked about chest indrawing and the LLM said
    'standard'. Now the breathing checklist question is asked; the caregiver's
    answer is extracted as chest_indrawing=present; the floor forces EMERGENCY
    even though the (stub) LLM still says 'standard'.
    """
    from graph import create_graph, run_turn

    s1 = child(rr=None, completeness=0.6)
    s2 = child(rr=None, completeness=0.8, findings=F(**all_absent(GENERAL)))
    s3 = child(rr=50, completeness=0.9,
               findings=F(**all_absent(GENERAL), chest_indrawing=P, stridor_when_calm=A))
    llm = StubLLM(structured={"PatientSummary": [s1, s2, s3],
                              "TriageResult": [result(TriageLevel.STANDARD, confs=(0.9,))]},
                  questions=["unused"])
    app, tid = create_graph(), str(uuid.uuid4())
    with stubbed(llm):
        r1 = run_turn(app, tid, "My 3 year old has had a cough for 4 days and seems to be breathing fast.")
        assert r1["message"] == safety.GENERAL_DANGER.question
        r2 = run_turn(app, tid, "He drinks fine, no vomiting, no fits, he is alert.")
        assert r2["message"] == safety.BREATHING.question
        r3 = run_turn(app, tid, "50 breaths a minute. Yes, the lower chest pulls in. No noise.")
    assert r3["is_complete"]
    tr = r3["state"]["triage_result"]
    assert tr.triage_level == TriageLevel.EMERGENCY
    assert "[SAFETY OVERRIDE]" in tr.reasoning
    assert "chest indrawing" in r3["state"]["patient_summary"].red_flags
    assert r3["state"]["triage_confidence"] < 1.0
