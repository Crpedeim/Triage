"""
state.py — Shared state schemas for the Multi-Agent Clinical Triage System.

This is the SINGLE SOURCE OF TRUTH for the entire system. Every agent reads from
and writes to these schemas. If you change something here, it affects everything.

Three layers:
1. Data models (Pydantic) — Symptom, VitalSigns, PatientSummary, SuspectedCondition, TriageResult
2. Graph state (TypedDict) — TriageState, the shared state that flows through LangGraph
3. Enums/constants — TriageLevel, Phase, severity thresholds

WHY PYDANTIC AND NOT PLAIN DICTS:
- Validation: if the Intake Agent produces an age of "three" instead of 3, Pydantic catches it
- Type safety: downstream agents know exactly what fields exist and their types
- Serialization: .model_dump_json() gives you clean JSON for logging/debugging
- Structured output: LLMs can produce Pydantic objects directly via .with_structured_output()
"""

from __future__ import annotations

from enum import Enum
from typing import Annotated

from langchain_core.messages import BaseMessage
from pydantic import BaseModel, Field
from typing_extensions import TypedDict

# langgraph uses this to know how to MERGE message lists (append, not replace)
# when a node returns a partial state update with new messages
from langgraph.graph.message import add_messages


# ═══════════════════════════════════════════════════════════════════════════════
# ENUMS — Fixed categories used across the system
# ═══════════════════════════════════════════════════════════════════════════════

class TriageLevel(str, Enum):
    """
    The four triage levels, ordered by severity.
    These map directly to IMNCI classification actions:
    - EMERGENCY: Life-threatening. Immediate referral to hospital. (IMNCI "pink" zone)
    - URGENT: Serious but not immediately life-threatening. Same-day referral. (IMNCI "yellow" zone)
    - STANDARD: Needs medical attention but can wait 48-72 hours. Schedule visit.
    - SELF_CARE: Can be managed at home with advice. Follow-up in 5 days. (IMNCI "green" zone)
    """
    EMERGENCY = "emergency"
    URGENT = "urgent"
    STANDARD = "standard"
    SELF_CARE = "self_care"


class Phase(str, Enum):
    """
    Which phase the system is currently in.
    The Supervisor reads this to decide routing.
    """
    INTAKE = "intake"
    RETRIEVAL = "retrieval"
    TRIAGE = "triage"
    CONSULTATION = "consultation"   # Enhancement 3 — multi-specialist debate
    DONE = "done"


class Severity(str, Enum):
    """Symptom or overall severity level."""
    MILD = "mild"
    MODERATE = "moderate"
    SEVERE = "severe"
    UNKNOWN = "unknown"


# ═══════════════════════════════════════════════════════════════════════════════
# DATA MODELS — Structured clinical data extracted by agents
# ═══════════════════════════════════════════════════════════════════════════════

class Symptom(BaseModel):
    """
    A single symptom reported by or observed in the patient.

    The Intake Agent extracts these from the conversation. Each symptom
    has optional detail fields — the completeness of these fields is what
    determines whether the Intake Agent needs to ask more questions.

    Example:
        Symptom(
            name="cough",
            duration="4 days",
            severity=Severity.MODERATE,
            associated_symptoms=["fast breathing", "chest indrawing"]
        )
    """
    name: str = Field(
        description="Symptom name in clinical terminology (e.g., 'cough', 'tachypnea', 'diarrhea')"
    )
    duration: str | None = Field(
        default=None,
        description="How long the symptom has been present (e.g., '4 days', '2 weeks', 'since yesterday')"
    )
    severity: Severity = Field(
        default=Severity.UNKNOWN,
        description="Severity of this specific symptom"
    )
    associated_symptoms: list[str] = Field(
        default_factory=list,
        description="Other symptoms that accompany this one (e.g., cough + fast breathing)"
    )


class VitalSigns(BaseModel):
    """
    Vital signs if available. ASHA workers may have a thermometer and can
    count respiratory rate, but likely don't have BP equipment or pulse oximeter.
    All fields optional because community-level assessment has limited tools.
    """
    temperature_celsius: float | None = Field(
        default=None,
        description="Body temperature in Celsius. Fever threshold: >= 38.0°C"
    )
    respiratory_rate: int | None = Field(
        default=None,
        description="Breaths per minute. IMNCI thresholds: <2mo: >=60, 2-12mo: >=50, 1-5yr: >=40"
    )
    heart_rate: int | None = Field(
        default=None,
        description="Beats per minute if available"
    )
    blood_pressure_systolic: int | None = Field(
        default=None,
        description="Systolic BP in mmHg (usually not available at ASHA level)"
    )
    blood_pressure_diastolic: int | None = Field(
        default=None,
        description="Diastolic BP in mmHg"
    )
    oxygen_saturation: float | None = Field(
        default=None,
        description="SpO2 percentage (rarely available at community level)"
    )


class PatientSummary(BaseModel):
    """
    Structured extraction of patient information from the intake interview.

    This is the OUTPUT of the Intake Agent and the INPUT to the Retrieval
    and Triage Agents. It transforms free-form conversation into validated,
    typed clinical data.

    The completeness_score (0.0-1.0) is self-assessed by the Intake Agent.
    When it reaches >= 0.7, the Intake Agent signals completion.

    KEY DESIGN DECISION: All field values are in ENGLISH regardless of what
    language the conversation happened in. The Intake Agent translates during
    extraction. This ensures the Retrieval Agent can search English guidelines
    and the Triage Agent reasons in English.
    """
    age: int | None = Field(
        default=None,
        description="Patient age in years. For infants, use 0."
    )
    age_months: int | None = Field(
        default=None,
        description="Patient age in months (for children under 5, more precise than years)"
    )
    sex: str | None = Field(
        default=None,
        description="'male', 'female', or 'other'"
    )
    chief_complaint: str = Field(
        description="Primary reason for the visit in 1-2 sentences"
    )
    symptoms: list[Symptom] = Field(
        default_factory=list,
        description="All symptoms collected during the interview"
    )
    duration: str | None = Field(
        default=None,
        description="Overall duration of the illness"
    )
    severity: Severity = Field(
        default=Severity.UNKNOWN,
        description="Overall assessed severity based on all symptoms"
    )
    vital_signs: VitalSigns | None = Field(
        default=None,
        description="Vital signs if the health worker could measure any"
    )
    medical_history: list[str] = Field(
        default_factory=list,
        description="Known pre-existing conditions (e.g., 'asthma', 'diabetes', 'HIV')"
    )
    current_medications: list[str] = Field(
        default_factory=list,
        description="Medications the patient is currently taking"
    )
    red_flags: list[str] = Field(
        default_factory=list,
        description=(
            "DANGER SIGNS detected. If ANY item is present here, triage level is "
            "automatically escalated. Examples: 'chest indrawing', 'unable to drink', "
            "'convulsions', 'severe dehydration', 'altered consciousness'"
        )
    )
    completeness_score: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description=(
            "Self-assessed by the Intake Agent. 0.0 = no info, 1.0 = comprehensive. "
            "Threshold for completion: >= 0.7"
        )
    )

    @property
    def is_pediatric(self) -> bool:
        """Check if patient is under 5 (IMNCI target population)."""
        if self.age is not None:
            return self.age < 5
        if self.age_months is not None:
            return self.age_months < 60
        return False

    @property
    def has_red_flags(self) -> bool:
        """Check if any danger signs are present."""
        return len(self.red_flags) > 0

    @property
    def is_complete(self) -> bool:
        """Check if enough information has been collected."""
        return self.completeness_score >= 0.7


class SuspectedCondition(BaseModel):
    """
    A single suspected condition from the Triage Agent's differential diagnosis.

    The Triage Agent produces a list of these, ordered by confidence.
    Each condition includes supporting evidence AND what would rule it out —
    this makes the reasoning auditable and helps the health worker understand
    the assessment.
    """
    name: str = Field(
        description="Condition name (e.g., 'Severe Pneumonia', 'Acute Watery Diarrhea')"
    )
    confidence: float = Field(
        ge=0.0,
        le=1.0,
        description="How confident the agent is in this condition (0.0-1.0)"
    )
    supporting_symptoms: list[str] = Field(
        default_factory=list,
        description="Which patient symptoms support this condition"
    )
    ruling_out: list[str] = Field(
        default_factory=list,
        description="Symptoms that, if present, would make this condition less likely"
    )


class TriageResult(BaseModel):
    """
    The FINAL OUTPUT of the system — the triage card that the health worker sees.

    This is produced by the Triage Agent (or the Consultation node in Enhancement 3).
    It contains everything the health worker needs: what to do, why, and how urgently.

    The reasoning field contains the full chain-of-thought — this is crucial for
    auditability. If a supervisor reviews this case later, they can see exactly
    how the system reached its conclusion.
    """
    triage_level: TriageLevel = Field(
        description="Classification: emergency / urgent / standard / self_care"
    )
    suspected_conditions: list[SuspectedCondition] = Field(
        default_factory=list,
        description="Differential diagnosis, ordered by confidence"
    )
    recommended_actions: list[str] = Field(
        default_factory=list,
        description=(
            "What the health worker should do RIGHT NOW. "
            "Examples: 'Refer urgently to nearest health facility', "
            "'Give first dose of amoxicillin', 'Advise home fluid management'"
        )
    )
    referral_note: str | None = Field(
        default=None,
        description=(
            "If triage_level is emergency or urgent, a brief note for the receiving "
            "facility summarizing the case. Not generated for standard/self_care."
        )
    )
    reasoning: str = Field(
        description=(
            "Full chain-of-thought reasoning: symptom analysis -> red flag check -> "
            "guideline matching -> differential -> classification. This is the "
            "audit trail."
        )
    )
    guidelines_cited: list[str] = Field(
        default_factory=list,
        description=(
            "Which guideline sections informed this decision. "
            "Example: 'IMNCI: Assess and Classify Cough or Difficult Breathing'"
        )
    )


# ═══════════════════════════════════════════════════════════════════════════════
# GRAPH STATE — The shared state that flows through LangGraph
# ═══════════════════════════════════════════════════════════════════════════════

class TriageState(TypedDict):
    """
    The shared state for the LangGraph StateGraph.

    WHY TypedDict AND NOT Pydantic:
    LangGraph requires TypedDict for its state schema. This is because LangGraph
    needs to know how to MERGE partial state updates from each node. The
    `Annotated[..., add_messages]` syntax tells LangGraph that the `messages`
    field should be appended to (not replaced) when a node returns new messages.

    Every node (agent) receives this full state as input and returns a PARTIAL
    dict with only the fields it wants to update. LangGraph handles the merge.

    Example — if the Intake Agent returns:
        {"patient_summary": summary, "intake_complete": True}
    LangGraph merges this into the existing state, leaving all other fields unchanged.
    """

    # ─── Conversation History ───
    # The full message list (human + AI). Uses add_messages reducer so that
    # when a node returns new messages, they get APPENDED to the existing list
    # instead of replacing it.
    messages: Annotated[list[BaseMessage], add_messages]

    # ─── Intake Agent Output ───
    patient_summary: PatientSummary | None      # Structured extraction from interview
    intake_complete: bool                        # Has intake collected enough info?

    # ─── Retrieval Agent Output ───
    retrieved_guidelines: list[dict]             # Relevant guideline chunks (serialized)
    retrieval_queries: list[str]                 # Queries used (for debugging/display)
    retrieval_sufficient: bool                   # Did retrieval find relevant content?

    # ─── Triage Agent Output ───
    triage_result: TriageResult | None           # Final triage decision
    triage_confidence: float                     # 0.0 - 1.0
    follow_up_questions: list[str]               # If confidence low, what to ask next

    # ─── Control Flow ───
    current_agent: str                           # Which agent is active
    cycle_count: int                             # How many intake->triage cycles so far
    max_cycles: int                              # Hard limit on cycles (default: 3)
    phase: str                                   # Current phase (Phase enum value)


def create_initial_state() -> dict:
    """
    Create the initial state for a new triage session.

    Call this when a new user session starts. It sets all fields to their
    default values and sets the phase to INTAKE so the Supervisor routes
    to the Intake Agent first.

    Returns a plain dict (not TriageState) because LangGraph expects
    the initial state as a dict that it validates against the TypedDict schema.
    """
    return {
        "messages": [],
        "patient_summary": None,
        "intake_complete": False,
        "retrieved_guidelines": [],
        "retrieval_queries": [],
        "retrieval_sufficient": False,
        "triage_result": None,
        "triage_confidence": 0.0,
        "follow_up_questions": [],
        "current_agent": "intake",
        "cycle_count": 0,
        "max_cycles": 3,
        "phase": Phase.INTAKE.value,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# CONSTANTS — Clinical thresholds used by agents
# ═══════════════════════════════════════════════════════════════════════════════

# Completeness threshold — Intake Agent signals done when score >= this
COMPLETENESS_THRESHOLD = 0.7

# Confidence threshold — Triage Agent's result is accepted when >= this
CONFIDENCE_THRESHOLD = 0.75

# Diagnostic ambiguity gap — if top 2 conditions are within this gap,
# it's considered ambiguous (used in Enhancement 3: multi-specialist consultation)
AMBIGUITY_GAP = 0.15

# IMNCI respiratory rate thresholds (breaths per minute)
# If respiratory rate exceeds these for the given age, it's "fast breathing"
RESPIRATORY_RATE_THRESHOLDS = {
    "0-2mo": 60,    # Under 2 months: >= 60 is fast
    "2-12mo": 50,   # 2-12 months: >= 50 is fast
    "1-5yr": 40,    # 1-5 years: >= 40 is fast
}

# Red flag symptoms — if ANY of these appear, auto-escalate triage level
# These are the IMNCI "general danger signs" plus adult emergency signs
RED_FLAG_KEYWORDS = [
    "chest indrawing",
    "unable to drink",
    "unable to breastfeed",
    "vomits everything",
    "convulsions",
    "seizures",
    "lethargic",
    "unconscious",
    "altered consciousness",
    "severe dehydration",
    "stridor when calm",
    "severe malnutrition",
    "chest pain",
    "difficulty breathing at rest",
    "signs of stroke",
    "severe bleeding",
]
