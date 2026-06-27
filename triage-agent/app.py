"""
app.py — Gradio Frontend

Run with:
    cd triage-agent
    python app.py

Layout:
  Left panel:  Chat interface (health worker ↔ intake agent)
  Right panel: Status indicator + triage card output

The app manages one LangGraph session per browser tab via session_state.
Each tab gets a unique thread_id so conversations don't cross-contaminate.

FIRST RUN:
  Before launching, you need a populated vector store:
    python -m rag.ingest --sample     # Quick start with sample data
    python -m rag.ingest              # Full ingestion (needs PDFs in rag/guidelines/)

  Set your LLM API key in .env:
    ANTHROPIC_API_KEY=your-key-here
"""

from __future__ import annotations

import uuid
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

import gradio as gr
from graph import create_graph, run_turn
from rag.store import count as store_count

# ─── Initialise the graph once at startup ────────────────────────────────────
# The graph is shared across all sessions. Each session has its own thread_id
# which keys into the MemorySaver checkpointer.
print("[app] Initialising TriageAI graph...")
APP = create_graph()
print("[app] Graph ready.")

# ─── Check knowledge base ─────────────────────────────────────────────────────
KB_COUNT = store_count()
if KB_COUNT == 0:
    print("[app] ⚠️  Vector store is empty. Run: python -m rag.ingest --sample")
else:
    print(f"[app] Knowledge base: {KB_COUNT} guideline chunks loaded.")


# ─── Chat handler ─────────────────────────────────────────────────────────────

def chat(
    user_message: str,
    history: list[tuple[str, str]],
    thread_id: str,
) -> tuple[list, str, str, str]:
    """
    Handle one chat turn.

    Args:
        user_message: What the health worker typed.
        history:      Gradio chat history [(user_msg, bot_msg), ...]
        thread_id:    Unique session ID (stored in gr.State)

    Returns:
        (updated_history, cleared_input, status_text, triage_card_markdown)
    """
    if not user_message.strip():
        return history, "", _status_text("idle"), ""

    if KB_COUNT == 0:
        warning = (
            "⚠️ **Knowledge base is empty.**\n\n"
            "Run this command first:\n```\npython -m rag.ingest --sample\n```\n"
            "Then restart the app."
        )
        history.append((user_message, warning))
        return history, "", _status_text("idle"), ""

    # Process the turn through LangGraph
    result = run_turn(APP, thread_id, user_message)

    # Update chat history
    bot_message = result["message"]
    history.append((user_message, bot_message))

    # Build outputs
    status = result["status"]
    triage_card = result["triage_card"] if result["is_complete"] else ""

    return history, "", status, triage_card


def new_session() -> tuple[list, str, str, str, str]:
    """Reset everything for a new patient."""
    fresh_thread_id = str(uuid.uuid4())
    welcome = (
        "TriageAI is ready. Please describe the patient — "
        "their age, main complaint, and how long they've been unwell."
    )
    return [], fresh_thread_id, _status_text("ready"), "", welcome


def _status_text(state: str) -> str:
    labels = {
        "idle":  "💤 Waiting for patient description",
        "ready": "✅ Ready — describe the patient to begin",
    }
    return labels.get(state, state)


# ─── Build the Gradio UI ──────────────────────────────────────────────────────

with gr.Blocks(
    title="TriageAI — Clinical Triage System",
    theme=gr.themes.Soft(primary_hue="teal"),
    css="""
    .triage-card { background: #f0fdf4; border-left: 4px solid #14b8a6; padding: 1rem; border-radius: 4px; }
    .status-bar  { font-size: 0.9rem; color: #64748b; padding: 0.25rem 0; }
    """,
) as demo:

    # ─── Header ──────────────────────────────────────────────────────────
    gr.Markdown(
        "# 🏥 TriageAI\n"
        "**Multi-Agent Clinical Triage for Frontline Health Workers**  \n"
        "*Powered by WHO/IMNCI guidelines · LangGraph · Claude*"
    )

    # ─── Session state (invisible, per-tab) ──────────────────────────────
    thread_id_state = gr.State(value=str(uuid.uuid4()))

    # ─── Main layout ─────────────────────────────────────────────────────
    with gr.Row():

        # Left: Chat
        with gr.Column(scale=3):
            chatbot = gr.Chatbot(
                label="Clinical Intake",
                height=460,
                bubble_full_width=False,
                avatar_images=(None, "🏥"),
            )
            with gr.Row():
                msg_input = gr.Textbox(
                    placeholder="Describe the patient (age, symptoms, how long)...",
                    show_label=False,
                    scale=5,
                    container=False,
                )
                send_btn = gr.Button("Send", variant="primary", scale=1)

            status_display = gr.Markdown(
                value=_status_text("ready"),
                elem_classes=["status-bar"],
            )
            new_btn = gr.Button("🔄 New Patient", variant="secondary")

        # Right: Triage card
        with gr.Column(scale=2):
            gr.Markdown("### Triage Card")
            triage_output = gr.Markdown(
                value="*Assessment will appear here when complete.*",
                elem_classes=["triage-card"],
                height=460,
            )

    # ─── Example inputs ──────────────────────────────────────────────────
    gr.Examples(
        examples=[
            "3 year old child, cough for 4 days, breathing seems fast",
            "4 year old girl, mild cough for 2 days, no fever, eating fine",
            "2 year old boy, diarrhea for 3 days, not sure if he can drink",
            "55 year old man, headache for one week, measured blood pressure 160/100",
        ],
        inputs=msg_input,
        label="Example presentations",
    )

    # ─── Info footer ─────────────────────────────────────────────────────
    with gr.Accordion("ℹ️ About this system", open=False):
        gr.Markdown(
            "**TriageAI** is a multi-agent AI system built for the WitchHunt Hackathon "
            "(Health & WellBeing theme). It is a **prototype** — not a medical device. "
            "All triage outputs should be reviewed by a medical professional.\n\n"
            "**Architecture:** LangGraph · Intake Agent · Retrieval Agent (RAG) · "
            "Triage Agent · Supervisor · WHO/IMNCI Guidelines\n\n"
            "**Safety:** Red flag danger signs (chest indrawing, unable to drink, "
            "convulsions etc.) always trigger URGENT/EMERGENCY classification, "
            "regardless of other findings."
        )

    # ─── Event handlers ──────────────────────────────────────────────────

    def on_send(msg, hist, tid):
        return chat(msg, hist, tid)

    send_btn.click(
        fn=on_send,
        inputs=[msg_input, chatbot, thread_id_state],
        outputs=[chatbot, msg_input, status_display, triage_output],
    )
    msg_input.submit(
        fn=on_send,
        inputs=[msg_input, chatbot, thread_id_state],
        outputs=[chatbot, msg_input, status_display, triage_output],
    )
    new_btn.click(
        fn=new_session,
        inputs=[],
        outputs=[chatbot, thread_id_state, status_display, triage_output, msg_input],
    )


if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=int(os.getenv("PORT", "7860")),
        share=False,
        show_error=True,
    )
