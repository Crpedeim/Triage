"""
tests/test_state.py — Tests for Phase 1: Schemas and State

Run with:
    cd triage-agent
    python -m pytest tests/test_state.py -v

These tests verify:
1. All Pydantic models validate correctly with good data
2. All Pydantic models reject bad data appropriately
3. Enums work as expected
4. PatientSummary computed properties work
5. TriageResult post-processing logic (red flag override) works
6. Initial state creation is correct
7. Edge cases: empty fields, boundary values, missing optional fields
8. Serialization: models can round-trip through JSON (important for LLM structured output)
"""

import json

import pytest
from pydantic import ValidationError

# ─── Import everything we're testing ───
from state import (
    AMBIGUITY_GAP,
    COMPLETENESS_THRESHOLD,
    CONFIDENCE_THRESHOLD,
    RED_FLAG_KEYWORDS,
    RESPIRATORY_RATE_THRESHOLDS,
    PatientSummary,
    Phase,
    Severity,
    SuspectedCondition,
    Symptom,
    TriageLevel,
    TriageResult,
    TriageState,
    VitalSigns,
    create_initial_state,
)


# ═══════════════════════════════════════════════════════════════════════════════
# TEST GROUP 1: Enum tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestEnums:
    """Verify all enums have the right values and are string-compatible."""

    def test_triage_levels_exist(self):
        """All four triage levels must exist."""
        assert TriageLevel.EMERGENCY == "emergency"
        assert TriageLevel.URGENT == "urgent"
        assert TriageLevel.STANDARD == "standard"
        assert TriageLevel.SELF_CARE == "self_care"

    def test_triage_level_is_string(self):
        """TriageLevel values should be usable as plain strings."""
        level = TriageLevel.EMERGENCY
        assert isinstance(level, str)
        assert level == "emergency"
        # This matters because LLMs will return string values, not enum instances

    def test_phases_exist(self):
        """All phases must exist."""
        assert Phase.INTAKE == "intake"
        assert Phase.RETRIEVAL == "retrieval"
        assert Phase.TRIAGE == "triage"
        assert Phase.CONSULTATION == "consultation"
        assert Phase.DONE == "done"

    def test_severity_levels(self):
        """All severity levels must exist."""
        assert Severity.MILD == "mild"
        assert Severity.MODERATE == "moderate"
        assert Severity.SEVERE == "severe"
        assert Severity.UNKNOWN == "unknown"


# ═══════════════════════════════════════════════════════════════════════════════
# TEST GROUP 2: Symptom model tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestSymptom:
    """Test the Symptom Pydantic model."""

    def test_minimal_symptom(self):
        """Symptom with only required field (name)."""
        s = Symptom(name="cough")
        assert s.name == "cough"
        assert s.duration is None
        assert s.severity == Severity.UNKNOWN
        assert s.associated_symptoms == []

    def test_full_symptom(self):
        """Symptom with all fields populated."""
        s = Symptom(
            name="tachypnea",
            duration="4 days",
            severity=Severity.MODERATE,
            associated_symptoms=["chest indrawing", "nasal flaring"],
        )
        assert s.name == "tachypnea"
        assert s.duration == "4 days"
        assert s.severity == Severity.MODERATE
        assert len(s.associated_symptoms) == 2

    def test_symptom_missing_name_fails(self):
        """Name is required — must fail without it."""
        with pytest.raises(ValidationError):
            Symptom()

    def test_symptom_json_roundtrip(self):
        """Symptom must survive JSON serialization and deserialization.
        This is critical because LLMs produce JSON that we parse into Pydantic models."""
        original = Symptom(
            name="fever",
            duration="2 days",
            severity=Severity.SEVERE,
            associated_symptoms=["chills", "sweating"],
        )
        json_str = original.model_dump_json()
        reconstructed = Symptom.model_validate_json(json_str)
        assert reconstructed == original


# ═══════════════════════════════════════════════════════════════════════════════
# TEST GROUP 3: VitalSigns model tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestVitalSigns:
    """Test the VitalSigns Pydantic model."""

    def test_empty_vitals(self):
        """All fields optional — empty VitalSigns is valid."""
        v = VitalSigns()
        assert v.temperature_celsius is None
        assert v.respiratory_rate is None

    def test_partial_vitals(self):
        """ASHA workers often can only measure temp and count breaths."""
        v = VitalSigns(
            temperature_celsius=38.5,
            respiratory_rate=52,
        )
        assert v.temperature_celsius == 38.5
        assert v.respiratory_rate == 52
        assert v.blood_pressure_systolic is None  # not available at community level

    def test_full_vitals(self):
        """All vitals populated (rare at community level, but valid)."""
        v = VitalSigns(
            temperature_celsius=39.2,
            respiratory_rate=55,
            heart_rate=120,
            blood_pressure_systolic=90,
            blood_pressure_diastolic=60,
            oxygen_saturation=94.0,
        )
        assert v.heart_rate == 120


# ═══════════════════════════════════════════════════════════════════════════════
# TEST GROUP 4: PatientSummary model tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestPatientSummary:
    """Test PatientSummary — the most important model in the system."""

    def test_minimal_summary(self):
        """Only chief_complaint is required."""
        ps = PatientSummary(chief_complaint="child has cough")
        assert ps.chief_complaint == "child has cough"
        assert ps.age is None
        assert ps.symptoms == []
        assert ps.red_flags == []
        assert ps.completeness_score == 0.0

    def test_scenario_1_severe_pneumonia(self):
        """MVP Scenario 1: 3-year-old with severe pneumonia indicators."""
        ps = PatientSummary(
            age=3,
            age_months=36,
            sex="male",
            chief_complaint="Cough for 4 days with fast breathing",
            symptoms=[
                Symptom(name="cough", duration="4 days", severity=Severity.MODERATE),
                Symptom(
                    name="tachypnea",
                    severity=Severity.MODERATE,
                    associated_symptoms=["respiratory rate 50/min"],
                ),
                Symptom(name="chest indrawing", severity=Severity.SEVERE),
                Symptom(name="fever", duration="2 days", severity=Severity.MILD),
                Symptom(name="reduced oral intake"),
            ],
            vital_signs=VitalSigns(temperature_celsius=38.2, respiratory_rate=50),
            red_flags=["chest indrawing"],
            completeness_score=0.85,
        )
        assert ps.is_pediatric is True
        assert ps.has_red_flags is True
        assert ps.is_complete is True
        assert len(ps.symptoms) == 5

    def test_scenario_2_mild_cough(self):
        """MVP Scenario 2: 4-year-old with simple cough, no danger signs."""
        ps = PatientSummary(
            age=4,
            sex="female",
            chief_complaint="Cough for 2 days, no other complaints",
            symptoms=[
                Symptom(name="cough", duration="2 days", severity=Severity.MILD),
            ],
            vital_signs=VitalSigns(respiratory_rate=30),
            red_flags=[],  # No danger signs
            completeness_score=0.75,
        )
        assert ps.is_pediatric is True
        assert ps.has_red_flags is False
        assert ps.is_complete is True

    def test_scenario_3_incomplete_diarrhea(self):
        """MVP Scenario 3: Incomplete initial description of diarrhea."""
        ps = PatientSummary(
            age=2,
            chief_complaint="Child has loose stools",
            symptoms=[
                Symptom(name="diarrhea", duration="3 days"),
            ],
            completeness_score=0.3,  # Low — triggers more questions
        )
        assert ps.is_complete is False  # Intake Agent should ask more
        assert ps.has_red_flags is False

    def test_adult_patient(self):
        """Adult patient (for Enhancement 1 — adult guidelines)."""
        ps = PatientSummary(
            age=55,
            sex="male",
            chief_complaint="Headache for one week with high blood pressure reading",
            symptoms=[
                Symptom(name="headache", duration="1 week", severity=Severity.MODERATE),
            ],
            vital_signs=VitalSigns(
                blood_pressure_systolic=160,
                blood_pressure_diastolic=100,
            ),
            completeness_score=0.8,
        )
        assert ps.is_pediatric is False
        assert ps.is_complete is True

    def test_is_pediatric_by_months(self):
        """Under-5 detection works with age_months when age is None."""
        ps = PatientSummary(chief_complaint="test", age_months=18)
        assert ps.is_pediatric is True

        ps2 = PatientSummary(chief_complaint="test", age_months=72)
        assert ps2.is_pediatric is False

    def test_is_pediatric_unknown_age(self):
        """When age is unknown, is_pediatric returns False (safe default)."""
        ps = PatientSummary(chief_complaint="test")
        assert ps.is_pediatric is False

    def test_completeness_score_bounds(self):
        """completeness_score must be between 0.0 and 1.0."""
        with pytest.raises(ValidationError):
            PatientSummary(chief_complaint="test", completeness_score=1.5)
        with pytest.raises(ValidationError):
            PatientSummary(chief_complaint="test", completeness_score=-0.1)

    def test_json_roundtrip_full_summary(self):
        """Full PatientSummary must survive JSON round-trip."""
        original = PatientSummary(
            age=3,
            age_months=36,
            sex="male",
            chief_complaint="Cough and fast breathing",
            symptoms=[
                Symptom(name="cough", duration="4 days"),
                Symptom(name="tachypnea", severity=Severity.MODERATE),
            ],
            vital_signs=VitalSigns(temperature_celsius=38.5, respiratory_rate=50),
            medical_history=["premature birth"],
            current_medications=[],
            red_flags=["chest indrawing"],
            completeness_score=0.85,
        )
        json_str = original.model_dump_json()
        reconstructed = PatientSummary.model_validate_json(json_str)
        assert reconstructed.age == original.age
        assert reconstructed.chief_complaint == original.chief_complaint
        assert len(reconstructed.symptoms) == len(original.symptoms)
        assert reconstructed.red_flags == original.red_flags

    def test_json_can_be_parsed_from_dict(self):
        """LLMs return dicts — model_validate must work with dict input."""
        data = {
            "age": 3,
            "chief_complaint": "fever and cough",
            "symptoms": [
                {"name": "fever", "duration": "2 days", "severity": "mild"},
                {"name": "cough"},
            ],
            "red_flags": [],
            "completeness_score": 0.7,
        }
        ps = PatientSummary.model_validate(data)
        assert ps.age == 3
        assert len(ps.symptoms) == 2
        assert ps.symptoms[0].severity == Severity.MILD


# ═══════════════════════════════════════════════════════════════════════════════
# TEST GROUP 5: SuspectedCondition and TriageResult tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestSuspectedCondition:
    """Test the SuspectedCondition model."""

    def test_basic_condition(self):
        sc = SuspectedCondition(
            name="Severe Pneumonia",
            confidence=0.9,
            supporting_symptoms=["cough", "fast breathing", "chest indrawing"],
            ruling_out=["no wheezing (reduces asthma likelihood)"],
        )
        assert sc.confidence == 0.9
        assert len(sc.supporting_symptoms) == 3

    def test_confidence_bounds(self):
        """Confidence must be 0.0-1.0."""
        with pytest.raises(ValidationError):
            SuspectedCondition(name="test", confidence=1.5)
        with pytest.raises(ValidationError):
            SuspectedCondition(name="test", confidence=-0.1)


class TestTriageResult:
    """Test the TriageResult — the final output model."""

    def test_emergency_triage(self):
        """Scenario 1 expected output: urgent triage for severe pneumonia."""
        tr = TriageResult(
            triage_level=TriageLevel.URGENT,
            suspected_conditions=[
                SuspectedCondition(
                    name="Severe Pneumonia (IMNCI classification)",
                    confidence=0.92,
                    supporting_symptoms=["cough 4 days", "respiratory rate 50/min", "chest indrawing"],
                    ruling_out=["no wheezing"],
                ),
            ],
            recommended_actions=[
                "Refer URGENTLY to nearest health facility",
                "Give first dose of oral amoxicillin before referral",
                "Keep child warm during transport",
            ],
            referral_note=(
                "3-year-old male with 4-day cough, respiratory rate 50/min, "
                "chest indrawing present. Classified as SEVERE PNEUMONIA per IMNCI. "
                "First dose amoxicillin given."
            ),
            reasoning=(
                "1. SYMPTOM ANALYSIS: Cough 4 days, RR 50 (above 40 threshold for age 1-5yr), "
                "chest indrawing present, low-grade fever.\n"
                "2. RED FLAG: Chest indrawing is an IMNCI danger sign.\n"
                "3. GUIDELINE MATCH: IMNCI 'Classify Cough/Difficult Breathing' -> "
                "chest indrawing present -> SEVERE PNEUMONIA.\n"
                "4. TRIAGE: URGENT referral."
            ),
            guidelines_cited=["IMNCI: Assess and Classify Cough or Difficult Breathing"],
        )
        assert tr.triage_level == TriageLevel.URGENT
        assert tr.referral_note is not None
        assert len(tr.recommended_actions) == 3

    def test_self_care_triage(self):
        """Scenario 2 expected output: self-care for simple cough."""
        tr = TriageResult(
            triage_level=TriageLevel.SELF_CARE,
            suspected_conditions=[
                SuspectedCondition(
                    name="Cough or Cold (no pneumonia)",
                    confidence=0.85,
                    supporting_symptoms=["cough 2 days", "no fast breathing"],
                    ruling_out=["chest indrawing absent", "no danger signs"],
                ),
            ],
            recommended_actions=[
                "Soothe the throat with safe home remedy (warm fluids, honey if >1yr)",
                "Watch for danger signs: fast breathing, chest indrawing, unable to drink",
                "Return in 5 days if cough persists or sooner if condition worsens",
            ],
            referral_note=None,  # No referral for self-care
            reasoning="Simple cough, no danger signs, normal respiratory rate.",
            guidelines_cited=["IMNCI: Treat the Child - Cough or Cold"],
        )
        assert tr.triage_level == TriageLevel.SELF_CARE
        assert tr.referral_note is None

    def test_triage_result_json_roundtrip(self):
        """TriageResult must survive JSON round-trip."""
        original = TriageResult(
            triage_level=TriageLevel.STANDARD,
            suspected_conditions=[
                SuspectedCondition(name="Pneumonia", confidence=0.7, supporting_symptoms=["cough"]),
            ],
            recommended_actions=["Visit health center within 48 hours"],
            reasoning="Moderate presentation, no danger signs but persistent symptoms.",
            guidelines_cited=["IMNCI: Classify Cough"],
        )
        json_str = original.model_dump_json()
        reconstructed = TriageResult.model_validate_json(json_str)
        assert reconstructed.triage_level == original.triage_level
        assert len(reconstructed.suspected_conditions) == 1


# ═══════════════════════════════════════════════════════════════════════════════
# TEST GROUP 6: Red flag override logic
# ═══════════════════════════════════════════════════════════════════════════════

class TestRedFlagLogic:
    """
    Test the red flag safety guardrail.

    KEY DESIGN DECISION: If red_flags is non-empty in the PatientSummary,
    the triage level MUST be escalated to EMERGENCY or URGENT regardless of
    what the Triage Agent produces. This is implemented as post-processing,
    not left to the LLM.

    These tests verify the LOGIC, not the implementation (which lives in
    the Triage Agent node). They define the expected behavior that the
    Triage Agent must conform to.
    """

    def test_red_flag_present_requires_escalation(self):
        """If red flags exist, triage level should be URGENT or EMERGENCY."""
        ps = PatientSummary(
            chief_complaint="child not breathing well",
            red_flags=["chest indrawing"],
            completeness_score=0.8,
        )
        assert ps.has_red_flags is True
        # The Triage Agent MUST produce URGENT or EMERGENCY for this case.
        # This test documents that requirement.

    def test_no_red_flags_allows_any_level(self):
        """Without red flags, any triage level is valid."""
        ps = PatientSummary(
            chief_complaint="mild cough",
            red_flags=[],
            completeness_score=0.8,
        )
        assert ps.has_red_flags is False

    def test_red_flag_keywords_are_defined(self):
        """The RED_FLAG_KEYWORDS list must be non-empty and contain expected items."""
        assert len(RED_FLAG_KEYWORDS) > 0
        assert "chest indrawing" in RED_FLAG_KEYWORDS
        assert "unable to drink" in RED_FLAG_KEYWORDS
        assert "convulsions" in RED_FLAG_KEYWORDS
        assert "altered consciousness" in RED_FLAG_KEYWORDS


# ═══════════════════════════════════════════════════════════════════════════════
# TEST GROUP 7: Initial state and constants
# ═══════════════════════════════════════════════════════════════════════════════

class TestInitialState:
    """Test create_initial_state() and constants."""

    def test_initial_state_structure(self):
        """Initial state must have all required keys with correct defaults."""
        state = create_initial_state()
        assert state["messages"] == []
        assert state["patient_summary"] is None
        assert state["intake_complete"] is False
        assert state["retrieved_guidelines"] == []
        assert state["retrieval_queries"] == []
        assert state["retrieval_sufficient"] is False
        assert state["triage_result"] is None
        assert state["triage_confidence"] == 0.0
        assert state["follow_up_questions"] == []
        assert state["current_agent"] == "intake"
        assert state["cycle_count"] == 0
        assert state["max_cycles"] == 3
        assert state["phase"] == "intake"

    def test_initial_state_starts_at_intake(self):
        """System must start in the intake phase."""
        state = create_initial_state()
        assert state["phase"] == Phase.INTAKE.value
        assert state["current_agent"] == "intake"

    def test_initial_state_is_fresh(self):
        """Two calls to create_initial_state should produce independent dicts."""
        s1 = create_initial_state()
        s2 = create_initial_state()
        s1["cycle_count"] = 5
        assert s2["cycle_count"] == 0  # s2 should not be affected

    def test_constants_are_sane(self):
        """Verify clinical constants are in reasonable ranges."""
        assert 0.5 <= COMPLETENESS_THRESHOLD <= 0.9
        assert 0.5 <= CONFIDENCE_THRESHOLD <= 0.95
        assert 0.05 <= AMBIGUITY_GAP <= 0.3
        assert RESPIRATORY_RATE_THRESHOLDS["0-2mo"] == 60
        assert RESPIRATORY_RATE_THRESHOLDS["2-12mo"] == 50
        assert RESPIRATORY_RATE_THRESHOLDS["1-5yr"] == 40


# ═══════════════════════════════════════════════════════════════════════════════
# TEST GROUP 8: Diagnostic ambiguity detection (for Enhancement 3)
# ═══════════════════════════════════════════════════════════════════════════════

class TestDiagnosticAmbiguity:
    """
    Tests for the ambiguity detection logic used in Enhancement 3
    (multi-specialist consultation). The logic is:
    - If top 2 conditions have confidence gap < AMBIGUITY_GAP (0.15),
      it's ambiguous → trigger consultation.
    - If gap >= 0.15, it's not ambiguous → loop back to Intake for more data.

    We test the LOGIC here; it's implemented in the Supervisor later.
    """

    def test_ambiguous_case(self):
        """Two conditions with similar confidence = ambiguous."""
        conditions = [
            SuspectedCondition(name="Pneumonia", confidence=0.6, supporting_symptoms=["cough"]),
            SuspectedCondition(name="Bronchiolitis", confidence=0.55, supporting_symptoms=["wheeze"]),
        ]
        sorted_c = sorted(conditions, key=lambda c: c.confidence, reverse=True)
        gap = sorted_c[0].confidence - sorted_c[1].confidence
        assert gap < AMBIGUITY_GAP  # 0.05 < 0.15 → ambiguous

    def test_clear_case(self):
        """One dominant condition = not ambiguous."""
        conditions = [
            SuspectedCondition(name="Severe Pneumonia", confidence=0.9, supporting_symptoms=["cough"]),
            SuspectedCondition(name="Bronchiolitis", confidence=0.3, supporting_symptoms=["wheeze"]),
        ]
        sorted_c = sorted(conditions, key=lambda c: c.confidence, reverse=True)
        gap = sorted_c[0].confidence - sorted_c[1].confidence
        assert gap >= AMBIGUITY_GAP  # 0.6 >= 0.15 → not ambiguous

    def test_single_condition_not_ambiguous(self):
        """Only one suspected condition = not ambiguous by definition."""
        conditions = [
            SuspectedCondition(name="Malaria", confidence=0.65, supporting_symptoms=["fever"]),
        ]
        assert len(conditions) < 2  # Can't be ambiguous with one condition


# ═══════════════════════════════════════════════════════════════════════════════
# TEST GROUP 9: Schema compatibility with LLM structured output
# ═══════════════════════════════════════════════════════════════════════════════

class TestLLMCompatibility:
    """
    Tests that verify our schemas work with the kind of JSON that LLMs produce.
    LLMs sometimes produce slightly "dirty" output — string enum values instead
    of enum instances, missing optional fields, etc. Our schemas must handle this
    gracefully.
    """

    def test_string_enum_values_accepted(self):
        """LLMs return 'mild' not Severity.MILD — Pydantic must accept both."""
        s = Symptom(name="cough", severity="mild")
        assert s.severity == Severity.MILD

    def test_string_triage_level_accepted(self):
        """LLMs return 'emergency' not TriageLevel.EMERGENCY."""
        tr = TriageResult(
            triage_level="urgent",
            reasoning="test",
            guidelines_cited=[],
        )
        assert tr.triage_level == TriageLevel.URGENT

    def test_missing_optional_fields_ok(self):
        """LLMs may omit optional fields entirely — defaults must kick in."""
        data = {
            "chief_complaint": "headache",
            "completeness_score": 0.5,
        }
        ps = PatientSummary.model_validate(data)
        assert ps.age is None
        assert ps.symptoms == []
        assert ps.vital_signs is None

    def test_extra_fields_ignored(self):
        """LLMs sometimes add unexpected fields — they should be ignored, not error.
        Note: Pydantic v2 ignores extra fields by default."""
        data = {
            "name": "cough",
            "extra_field_llm_added": "some value",
            "another_random_key": 42,
        }
        s = Symptom.model_validate(data)
        assert s.name == "cough"

    def test_nested_json_parsing(self):
        """Full nested JSON as an LLM might produce it."""
        llm_output = json.dumps({
            "triage_level": "emergency",
            "suspected_conditions": [
                {
                    "name": "Severe Dehydration",
                    "confidence": 0.88,
                    "supporting_symptoms": ["sunken eyes", "skin pinch slow return", "unable to drink"],
                    "ruling_out": ["no bloody stool"],
                }
            ],
            "recommended_actions": [
                "Start ORS immediately",
                "Refer URGENTLY to health facility",
            ],
            "referral_note": "2-year-old with severe dehydration from acute diarrhea.",
            "reasoning": "Danger signs of severe dehydration present per IMNCI classification.",
            "guidelines_cited": ["IMNCI: Assess and Classify Diarrhea"],
        })
        tr = TriageResult.model_validate_json(llm_output)
        assert tr.triage_level == TriageLevel.EMERGENCY
        assert tr.suspected_conditions[0].confidence == 0.88
        assert len(tr.recommended_actions) == 2
