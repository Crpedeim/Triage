"""
safety.py — Deterministic danger-sign checklist and triage floor.

Everything here is plain Python: no LLM decides whether a danger sign forces
escalation. The LLM only EXTRACTS findings (present / absent / unknown) from the
conversation; these rules turn findings into a minimum triage level.

Sources for every rule (all inside this repository):
  * rag/ingest.py sample IMNCI text: general danger signs, cough / difficult
    breathing classification, fast-breathing thresholds, chest indrawing,
    diarrhoea (severe / some dehydration), fever (very severe febrile disease).
  * rag/ingest.py sample WHO PEN text: hypertensive crisis.
  * state.py: RED_FLAG_KEYWORDS (the project's own danger-sign list) and
    RESPIRATORY_RATE_THRESHOLDS, plus the level mapping
    (IMNCI pink → EMERGENCY, yellow → URGENT, green → SELF_CARE).

Main entry points:
  applicable_groups(summary)          checklist groups intake must cover
  next_unasked_group(summary, asked)  the next group intake should ask about
  checklist_coverage(summary)         fraction of applicable items with a known answer
  derive_red_flags(summary)           canonical red flags (RED_FLAG_KEYWORDS) from findings + text
  triage_floor(summary)               (minimum TriageLevel | None, reasons)
"""

from __future__ import annotations

from dataclasses import dataclass

from state import (
    RED_FLAG_KEYWORDS,
    RESPIRATORY_RATE_THRESHOLDS,
    DangerSign,
    FindingStatus,
    PatientSummary,
    TriageLevel,
)

LEVEL_ORDER = {TriageLevel.SELF_CARE: 0, TriageLevel.STANDARD: 1, TriageLevel.URGENT: 2, TriageLevel.EMERGENCY: 3}


def max_level(a: TriageLevel | None, b: TriageLevel | None) -> TriageLevel | None:
    if a is None:
        return b
    if b is None:
        return a
    return a if LEVEL_ORDER[a] >= LEVEL_ORDER[b] else b


# ═══════════════════════════════════════════════════════════════════════════════
# CHECKLIST
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class ChecklistGroup:
    id: str
    question: str
    signs: tuple[DangerSign, ...]
    needs_respiratory_rate: bool = False


GENERAL_DANGER = ChecklistGroup(
    id="general_danger_signs",
    question=("I need to check for danger signs. Is the child able to drink or breastfeed? "
              "Does the child vomit everything? Has the child had any fits (convulsions) in this illness? "
              "Is the child unusually sleepy or hard to wake?"),
    signs=(DangerSign.UNABLE_TO_DRINK_OR_BREASTFEED, DangerSign.VOMITS_EVERYTHING,
           DangerSign.CONVULSIONS, DangerSign.LETHARGIC_OR_UNCONSCIOUS),
)
BREATHING = ChecklistGroup(
    id="breathing",
    question=("Please count the child's breaths for one full minute while the child is calm: how many breaths? "
              "Does the lower chest pull in when the child breathes in? "
              "Is there a harsh noise when the child breathes in while calm?"),
    signs=(DangerSign.CHEST_INDRAWING, DangerSign.STRIDOR_WHEN_CALM),
    needs_respiratory_rate=True,
)
DEHYDRATION = ChecklistGroup(
    id="dehydration",
    question=("Are the child's eyes sunken? When you pinch the skin of the belly, does it go back very slowly, "
              "slowly, or straight away? Is the child drinking poorly, or drinking eagerly as if thirsty? "
              "Is the child restless or irritable?"),
    signs=(DangerSign.SUNKEN_EYES, DangerSign.SKIN_PINCH_VERY_SLOW, DangerSign.SKIN_PINCH_SLOW,
           DangerSign.DRINKING_POORLY, DangerSign.DRINKS_EAGERLY, DangerSign.RESTLESS_IRRITABLE),
)
FEVER = ChecklistGroup(
    id="fever",
    question="Does the child have a stiff neck?",
    signs=(DangerSign.STIFF_NECK,),
)
ADULT_DANGER = ChecklistGroup(
    id="adult_danger_signs",
    question=("I need to check for danger signs. Does the patient have any chest pain, difficulty breathing "
              "while resting, sudden weakness or numbness on one side, trouble speaking, heavy bleeding, "
              "confusion, fits (seizures), or problems with their vision?"),
    signs=(DangerSign.CHEST_PAIN, DangerSign.DIFFICULTY_BREATHING_AT_REST, DangerSign.SIGNS_OF_STROKE,
           DangerSign.SEVERE_BLEEDING, DangerSign.CONFUSION_OR_UNCONSCIOUS, DangerSign.SEIZURE,
           DangerSign.VISUAL_DISTURBANCE),
)

ALL_GROUPS = (GENERAL_DANGER, BREATHING, DEHYDRATION, FEVER, ADULT_DANGER)

RESPIRATORY_TERMS = ("cough", "breath", "tachypn", "wheez", "pneumon", "chest", "respirat")
DIARRHOEA_TERMS = ("diarrh", "loose stool", "watery stool", "loose motion", "dysentery")
FEVER_TERMS = ("fever", "febrile", "temperature", "hot to touch", "malaria", "pyrexia")


def age_in_months(summary: PatientSummary) -> int | None:
    if summary.age_months is not None:
        return summary.age_months
    if summary.age is not None:
        return summary.age * 12
    return None


def is_under_five(summary: PatientSummary) -> bool | None:
    m = age_in_months(summary)
    return None if m is None else m < 60


def _complaint_text(summary: PatientSummary) -> str:
    parts = [summary.chief_complaint or ""]
    for s in summary.symptoms:
        parts.append(s.name)
        parts.extend(s.associated_symptoms)
    return " ".join(parts).lower()


def _mentions(summary: PatientSummary, terms: tuple[str, ...]) -> bool:
    text = _complaint_text(summary)
    return any(t in text for t in terms)


def applicable_groups(summary: PatientSummary) -> list[ChecklistGroup]:
    """Checklist groups that must be covered for this patient. Empty if age is unknown."""
    under5 = is_under_five(summary)
    if under5 is None:
        return []
    if not under5:
        return [ADULT_DANGER]
    groups = [GENERAL_DANGER]
    if _mentions(summary, RESPIRATORY_TERMS):
        groups.append(BREATHING)
    if _mentions(summary, DIARRHOEA_TERMS):
        groups.append(DEHYDRATION)
    if _mentions(summary, FEVER_TERMS) or (
            summary.vital_signs and summary.vital_signs.temperature_celsius is not None
            and summary.vital_signs.temperature_celsius >= 38.0):
        groups.append(FEVER)
    return groups


def finding(summary: PatientSummary, sign: DangerSign) -> FindingStatus:
    for f in summary.danger_sign_findings:
        if f.sign == sign:
            return f.status
    return FindingStatus.UNKNOWN


def _group_items_known(summary: PatientSummary, group: ChecklistGroup) -> tuple[int, int]:
    known = sum(1 for s in group.signs if finding(summary, s) != FindingStatus.UNKNOWN)
    total = len(group.signs)
    if group.needs_respiratory_rate:
        total += 1
        if summary.vital_signs and summary.vital_signs.respiratory_rate is not None:
            known += 1
    return known, total


def group_is_answered(summary: PatientSummary, group: ChecklistGroup) -> bool:
    known, total = _group_items_known(summary, group)
    return known == total


def next_unasked_group(summary: PatientSummary, asked: list[str]) -> ChecklistGroup | None:
    """First applicable group that has neither been asked nor fully answered already."""
    for g in applicable_groups(summary):
        if g.id not in asked and not group_is_answered(summary, g):
            return g
    return None


def checklist_coverage(summary: PatientSummary) -> float:
    """Fraction of applicable checklist items with a known answer (1.0 when nothing applies)."""
    known = total = 0
    for g in applicable_groups(summary):
        k, t = _group_items_known(summary, g)
        known += k
        total += t
    return 1.0 if total == 0 else known / total


def unassessed_items(summary: PatientSummary) -> list[str]:
    out = []
    for g in applicable_groups(summary):
        for s in g.signs:
            if finding(summary, s) == FindingStatus.UNKNOWN:
                out.append(s.value)
        if g.needs_respiratory_rate and not (summary.vital_signs and summary.vital_signs.respiratory_rate is not None):
            out.append("respiratory_rate")
    return out


# ═══════════════════════════════════════════════════════════════════════════════
# RED FLAGS (RED_FLAG_KEYWORDS) — canonicalisation
# ═══════════════════════════════════════════════════════════════════════════════

# Lay / alternative phrasings → canonical RED_FLAG_KEYWORDS entry
RED_FLAG_SYNONYMS: dict[str, tuple[str, ...]] = {
    "chest indrawing": ("chest indrawing", "lower chest wall indrawing", "chest pulls in", "chest pulling in", "subcostal retraction"),
    "unable to drink": ("unable to drink", "not able to drink", "cannot drink", "can't drink"),
    "unable to breastfeed": ("unable to breastfeed", "not able to breastfeed", "cannot breastfeed", "can't breastfeed"),
    "vomits everything": ("vomits everything", "vomiting everything", "vomits all"),
    "convulsions": ("convulsion", "fits", "fitting"),
    "seizures": ("seizure",),
    "lethargic": ("lethargic", "lethargy", "floppy", "hard to wake", "difficult to wake"),
    "unconscious": ("unconscious", "unresponsive", "not responding"),
    "altered consciousness": ("altered consciousness", "confusion", "confused", "altered mental"),
    "severe dehydration": ("severe dehydration",),
    "stridor when calm": ("stridor",),
    "severe malnutrition": ("severe malnutrition", "severe wasting"),
    "chest pain": ("chest pain",),
    "difficulty breathing at rest": ("difficulty breathing at rest", "breathless at rest", "short of breath at rest"),
    "signs of stroke": ("signs of stroke", "stroke", "facial droop", "one-sided weakness", "slurred speech", "hemiparesis"),
    "severe bleeding": ("severe bleeding", "heavy bleeding", "haemorrhage", "hemorrhage"),
}
assert set(RED_FLAG_SYNONYMS) == set(RED_FLAG_KEYWORDS), "RED_FLAG_SYNONYMS must cover RED_FLAG_KEYWORDS exactly"

# Danger-sign findings → canonical red flag
SIGN_TO_RED_FLAG: dict[DangerSign, str] = {
    DangerSign.UNABLE_TO_DRINK_OR_BREASTFEED: "unable to drink",
    DangerSign.VOMITS_EVERYTHING: "vomits everything",
    DangerSign.CONVULSIONS: "convulsions",
    DangerSign.LETHARGIC_OR_UNCONSCIOUS: "lethargic",
    DangerSign.CHEST_INDRAWING: "chest indrawing",
    DangerSign.STRIDOR_WHEN_CALM: "stridor when calm",
    DangerSign.CHEST_PAIN: "chest pain",
    DangerSign.DIFFICULTY_BREATHING_AT_REST: "difficulty breathing at rest",
    DangerSign.SIGNS_OF_STROKE: "signs of stroke",
    DangerSign.SEVERE_BLEEDING: "severe bleeding",
    DangerSign.CONFUSION_OR_UNCONSCIOUS: "altered consciousness",
    DangerSign.SEIZURE: "seizures",
}

_NEGATIONS = ("no ", "not ", "without ", "denies ", "absent", "never ", "none")


def _negated(text: str) -> bool:
    t = text.strip().lower()
    return t.startswith(_NEGATIONS) or " absent" in t or "not present" in t


def canonical_red_flags_in(texts: list[str]) -> set[str]:
    """Map free-text findings (already affirmative LLM extractions) to RED_FLAG_KEYWORDS."""
    found: set[str] = set()
    for raw in texts:
        if not raw or _negated(raw):
            continue
        t = raw.lower()
        for keyword, synonyms in RED_FLAG_SYNONYMS.items():
            if any(s in t for s in synonyms):
                found.add(keyword)
    return found


def derive_red_flags(summary: PatientSummary) -> list[str]:
    """
    Canonical red flags from (a) danger-sign findings marked PRESENT and
    (b) RED_FLAG_KEYWORDS matched in the LLM's affirmative fields (red_flags,
    symptom names, associated symptoms). Findings marked ABSENT never add a flag.
    """
    flags: set[str] = set()
    for f in summary.danger_sign_findings:
        if f.status == FindingStatus.PRESENT and f.sign in SIGN_TO_RED_FLAG:
            flags.add(SIGN_TO_RED_FLAG[f.sign])
    texts = list(summary.red_flags)
    for s in summary.symptoms:
        texts.append(s.name)
        texts.extend(s.associated_symptoms)
    flags |= canonical_red_flags_in(texts)
    # An explicit ABSENT finding overrides a free-text match for the same sign
    for f in summary.danger_sign_findings:
        if f.status == FindingStatus.ABSENT and f.sign in SIGN_TO_RED_FLAG:
            flags.discard(SIGN_TO_RED_FLAG[f.sign])
    return sorted(flags)


# ═══════════════════════════════════════════════════════════════════════════════
# TRIAGE FLOOR
# ═══════════════════════════════════════════════════════════════════════════════

def fast_breathing_threshold(months: int) -> int:
    """IMNCI fast-breathing threshold (breaths/min) from RESPIRATORY_RATE_THRESHOLDS."""
    if months < 2:
        return RESPIRATORY_RATE_THRESHOLDS["0-2mo"]
    if months < 12:
        return RESPIRATORY_RATE_THRESHOLDS["2-12mo"]
    return RESPIRATORY_RATE_THRESHOLDS["1-5yr"]


def has_fast_breathing(summary: PatientSummary) -> bool:
    months = age_in_months(summary)
    rr = summary.vital_signs.respiratory_rate if summary.vital_signs else None
    if months is None or rr is None or months >= 60:
        return False
    return rr >= fast_breathing_threshold(months)


def _present(summary: PatientSummary, *signs: DangerSign) -> list[DangerSign]:
    return [s for s in signs if finding(summary, s) == FindingStatus.PRESENT]


def triage_floor(summary: PatientSummary) -> tuple[TriageLevel | None, list[str]]:
    """
    Minimum triage level implied by danger signs, with human-readable reasons.
    Returns (None, []) when no rule applies. The final level is
    max(LLM level, floor): this can only escalate, never downgrade.
    """
    emergency: list[str] = []
    urgent: list[str] = []
    under5 = is_under_five(summary)

    # IMNCI general danger signs → refer urgently to hospital (pink)
    gds = _present(summary, DangerSign.UNABLE_TO_DRINK_OR_BREASTFEED, DangerSign.VOMITS_EVERYTHING,
                   DangerSign.CONVULSIONS, DangerSign.LETHARGIC_OR_UNCONSCIOUS)
    if gds:
        emergency.append("IMNCI general danger sign: " + ", ".join(s.value for s in gds))

    # Severe pneumonia or very severe disease: chest indrawing or stridor in a calm child (pink)
    resp = _present(summary, DangerSign.CHEST_INDRAWING, DangerSign.STRIDOR_WHEN_CALM)
    if resp:
        emergency.append("IMNCI severe pneumonia or very severe disease: " + ", ".join(s.value for s in resp))

    # Diarrhoea: SEVERE dehydration = two of lethargic/unconscious, sunken eyes,
    # not able to drink / drinking poorly, skin pinch goes back very slowly (pink)
    severe_dehyd = _present(summary, DangerSign.LETHARGIC_OR_UNCONSCIOUS, DangerSign.SUNKEN_EYES,
                            DangerSign.SKIN_PINCH_VERY_SLOW)
    if _present(summary, DangerSign.DRINKING_POORLY, DangerSign.UNABLE_TO_DRINK_OR_BREASTFEED):
        severe_dehyd.append(DangerSign.DRINKING_POORLY)
    if len(severe_dehyd) >= 2:
        emergency.append("IMNCI severe dehydration: " + ", ".join(s.value for s in severe_dehyd))
    else:
        # SOME dehydration = two of restless/irritable, sunken eyes, drinks eagerly, skin pinch slowly (yellow)
        some = _present(summary, DangerSign.RESTLESS_IRRITABLE, DangerSign.SUNKEN_EYES,
                        DangerSign.DRINKS_EAGERLY, DangerSign.SKIN_PINCH_SLOW, DangerSign.SKIN_PINCH_VERY_SLOW)
        if len(some) >= 2:
            urgent.append("IMNCI some dehydration: " + ", ".join(s.value for s in some))

    # Fever + stiff neck → very severe febrile disease (pink)
    if _present(summary, DangerSign.STIFF_NECK):
        emergency.append("IMNCI very severe febrile disease: stiff neck")

    # Adult danger signs (project RED_FLAG_KEYWORDS)
    adult = _present(summary, DangerSign.CHEST_PAIN, DangerSign.DIFFICULTY_BREATHING_AT_REST,
                     DangerSign.SIGNS_OF_STROKE, DangerSign.SEVERE_BLEEDING,
                     DangerSign.CONFUSION_OR_UNCONSCIOUS, DangerSign.SEIZURE)
    if adult:
        emergency.append("Danger sign: " + ", ".join(s.value for s in adult))

    # Any other canonical red flag from free text (RED_FLAG_KEYWORDS are all danger signs)
    covered = {SIGN_TO_RED_FLAG[s] for s in gds + resp + adult if s in SIGN_TO_RED_FLAG}
    extra = [f for f in derive_red_flags(summary) if f not in covered]
    if extra:
        emergency.append("Red flag: " + ", ".join(extra))

    # WHO PEN hypertensive crisis: systolic > 180 and/or diastolic > 120 WITH symptoms
    v = summary.vital_signs
    if v and ((v.blood_pressure_systolic or 0) > 180 or (v.blood_pressure_diastolic or 0) > 120):
        crisis_symptoms = _present(summary, DangerSign.CHEST_PAIN, DangerSign.VISUAL_DISTURBANCE, DangerSign.SEIZURE)
        if crisis_symptoms or "headache" in _complaint_text(summary):
            emergency.append(f"WHO PEN hypertensive crisis: BP {v.blood_pressure_systolic}/{v.blood_pressure_diastolic} with symptoms")

    # IMNCI pneumonia: fast breathing for age (yellow)
    if under5 and has_fast_breathing(summary):
        months = age_in_months(summary)
        urgent.append(f"IMNCI pneumonia: respiratory rate {summary.vital_signs.respiratory_rate}/min "
                      f">= {fast_breathing_threshold(months)} for age {months} months")

    if emergency:
        return TriageLevel.EMERGENCY, emergency + urgent
    if urgent:
        return TriageLevel.URGENT, urgent
    return None, []
