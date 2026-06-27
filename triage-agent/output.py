"""
output.py — Triage Card Formatter

Converts a completed TriageState into a human-readable triage card
for the health worker. The card is designed to be:
- Scannable at a glance (triage level is the first thing visible)
- Actionable (specific steps, not vague advice)
- Auditable (shows reasoning and cited guidelines)
- Printable (plain text, no markdown that won't render on a phone)

Also exposes format_for_gradio() which returns Markdown for the Gradio UI.
"""

from __future__ import annotations

from state import TriageLevel, TriageResult, TriageState

# Visual indicators per triage level
LEVEL_SYMBOLS = {
    TriageLevel.EMERGENCY: "🔴 EMERGENCY",
    TriageLevel.URGENT:    "🟠 URGENT",
    TriageLevel.STANDARD:  "🟡 STANDARD",
    TriageLevel.SELF_CARE: "🟢 SELF CARE",
}

LEVEL_DESCRIPTIONS = {
    TriageLevel.EMERGENCY: "Life-threatening. Refer to hospital IMMEDIATELY.",
    TriageLevel.URGENT:    "Serious condition. Refer to health center TODAY.",
    TriageLevel.STANDARD:  "Needs medical attention. Visit health center within 48-72 hours.",
    TriageLevel.SELF_CARE: "Manage at home. Return if condition worsens.",
}


def format_triage_card(state: TriageState) -> str:
    """
    Format the triage result as plain text for terminal/SMS output.
    Used in testing and as the base for Gradio formatting.
    """
    result: TriageResult | None = state.get("triage_result")
    confidence: float = state.get("triage_confidence", 0.0)
    summary = state.get("patient_summary")

    if result is None:
        return "TRIAGE INCOMPLETE — No result available."

    level = result.triage_level
    lines = [
        "=" * 50,
        f"  TRIAGE ASSESSMENT",
        "=" * 50,
    ]

    # ─── Patient ─────────────────────────────────────────
    if summary:
        age_str = "Unknown age"
        if summary.age_months and summary.age_months < 24:
            age_str = f"{summary.age_months} months"
        elif summary.age:
            age_str = f"{summary.age} years"
        lines.append(f"Patient: {age_str}, {summary.chief_complaint}")
        lines.append("")

    # ─── Triage Level ────────────────────────────────────
    lines.append(f"TRIAGE LEVEL: {LEVEL_SYMBOLS[level]}")
    lines.append(f"{LEVEL_DESCRIPTIONS[level]}")
    lines.append("")

    # ─── Suspected Conditions ────────────────────────────
    if result.suspected_conditions:
        lines.append("SUSPECTED CONDITIONS:")
        for cond in result.suspected_conditions[:3]:
            pct = int(cond.confidence * 100)
            lines.append(f"  • {cond.name} ({pct}% confidence)")
        lines.append("")

    # ─── Recommended Actions ─────────────────────────────
    lines.append("RECOMMENDED ACTIONS:")
    for i, action in enumerate(result.recommended_actions, 1):
        lines.append(f"  {i}. {action}")
    lines.append("")

    # ─── Referral Note ───────────────────────────────────
    if result.referral_note:
        lines.append("REFERRAL NOTE (for health facility):")
        lines.append(f"  {result.referral_note}")
        lines.append("")

    # ─── Guidelines Cited ────────────────────────────────
    if result.guidelines_cited:
        lines.append("BASED ON:")
        for guideline in result.guidelines_cited:
            lines.append(f"  • {guideline}")
        lines.append("")

    # ─── Confidence ──────────────────────────────────────
    confidence_label = "High" if confidence >= 0.75 else "Moderate" if confidence >= 0.5 else "Low"
    lines.append(f"Assessment confidence: {confidence_label} ({confidence:.0%})")
    if confidence < 0.75:
        lines.append("⚠️  Low confidence — consider seeking medical officer review.")

    lines.append("=" * 50)
    return "\n".join(lines)


def format_for_gradio(state: TriageState) -> str:
    """
    Format the triage result as Markdown for the Gradio UI triage card panel.
    Uses colored headers and bold text that render in Gradio's Markdown component.
    """
    result: TriageResult | None = state.get("triage_result")
    confidence: float = state.get("triage_confidence", 0.0)
    summary = state.get("patient_summary")

    if result is None:
        return "*Triage assessment not yet complete.*"

    level = result.triage_level

    # Color coding via emoji
    level_header = LEVEL_SYMBOLS[level]
    level_desc = LEVEL_DESCRIPTIONS[level]

    lines = [f"## {level_header}"]
    lines.append(f"*{level_desc}*")
    lines.append("")

    if summary and summary.chief_complaint:
        age_str = ""
        if summary.age_months and summary.age_months < 24:
            age_str = f"{summary.age_months}-month-old"
        elif summary.age:
            age_str = f"{summary.age}-year-old"
        lines.append(f"**Patient:** {age_str} — {summary.chief_complaint}")
        lines.append("")

    if result.suspected_conditions:
        lines.append("**Suspected Conditions:**")
        for cond in result.suspected_conditions[:3]:
            pct = int(cond.confidence * 100)
            lines.append(f"- {cond.name} ({pct}%)")
        lines.append("")

    lines.append("**Recommended Actions:**")
    for i, action in enumerate(result.recommended_actions, 1):
        lines.append(f"{i}. {action}")
    lines.append("")

    if result.referral_note:
        lines.append("**Referral Note:**")
        lines.append(f"> {result.referral_note}")
        lines.append("")

    if result.guidelines_cited:
        lines.append("**Based on:**")
        for g in result.guidelines_cited:
            lines.append(f"- {g}")
        lines.append("")

    confidence_label = "High" if confidence >= 0.75 else "Moderate" if confidence >= 0.5 else "Low"
    lines.append(f"---")
    lines.append(f"*Confidence: {confidence_label} ({confidence:.0%})*")
    if confidence < 0.75:
        lines.append("*⚠️ Low confidence — consider medical officer review.*")

    return "\n".join(lines)


def format_agent_status(state: TriageState) -> str:
    """
    One-line status string showing which agent is active.
    Used in the Gradio UI status panel.
    """
    phase = state.get("phase", "intake")
    cycle = state.get("cycle_count", 0)
    confidence = state.get("triage_confidence", 0.0)

    status_map = {
        "intake":    "🎤 Collecting patient information...",
        "retrieval": "🔍 Searching clinical guidelines...",
        "triage":    "🧠 Analyzing case...",
        "done":      f"✅ Assessment complete (confidence: {confidence:.0%})",
    }

    status = status_map.get(phase, "⏳ Processing...")
    if cycle > 0 and phase == "intake":
        status = f"🔄 Gathering additional information (round {cycle + 1})..."
    return status
