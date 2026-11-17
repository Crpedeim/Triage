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
    python -m rag.fetch_guidelines && python -m rag.build_kb   # WHO guidelines (the Docker build does this)
    python -m rag.ingest --sample     # quick start: the 8 curated excerpts only

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
from graph import _get_current_state, create_graph, run_turn
from output import format_agent_status, format_for_gradio
from rag.embeddings import warm_up as warm_up_embeddings
from rag.store import count as store_count
from ratelimit import RateLimiter, client_ip

# Per-session / per-IP limits for the public URL (env: RATE_LIMIT_SESSION,
# RATE_LIMIT_IP, MAX_MESSAGE_CHARS). See ratelimit.py.
RATE_LIMITER = RateLimiter()
print(f"[app] Rate limits: session {RATE_LIMITER.session.limit}/{RATE_LIMITER.session.window_s}s, "
      f"IP {RATE_LIMITER.ip.limit}/{RATE_LIMITER.ip.window_s}s, max {RATE_LIMITER.max_chars} chars")

# ─── Initialise the graph once at startup ────────────────────────────────────
# The graph is shared across all sessions. Each session has its own thread_id
# which keys into the MemorySaver checkpointer.
print("[app] Initialising TriageAI graph...")
APP = create_graph()
print("[app] Graph ready.")

# Load the embedding model before serving, so the first users do not pay the
# cold model load inside their request (Phase 5).
warm_up_embeddings()
print("[app] Embedding model loaded.")

BROWSER_STATE_SECRET = os.getenv("BROWSER_STATE_SECRET") or None
if BROWSER_STATE_SECRET is None:
    print("[app] ⚠️  BROWSER_STATE_SECRET not set: open conversations cannot be resumed after a restart.")

# ─── Check knowledge base ─────────────────────────────────────────────────────
KB_COUNT = store_count()
if KB_COUNT == 0:
    print("[app] ⚠️  Vector store is empty. Run: python -m rag.fetch_guidelines && python -m rag.build_kb")
else:
    print(f"[app] Knowledge base: {KB_COUNT} guideline chunks loaded.")


# ─── Guideline attribution ────────────────────────────────────────────────────

def _attribution_markdown() -> str:
    """Footer text built from rag/guidelines/sources.json (titles, years, licences, links)."""
    import json

    manifest = Path(__file__).parent / "rag" / "guidelines" / "sources.json"
    try:
        docs = json.loads(manifest.read_text(encoding="utf-8"))["documents"]
    except Exception:
        return ""
    items = [
        f"[{d['title']}]({d['iris_handle']}). {d['publisher']}, {d['year']}. {d['licence']}"
        for d in docs
    ]
    return (
        "**Guideline sources.** Knowledge base built from: "
        + " · ".join(items)
        + ". Text is extracted and summarised automatically and may contain errors. "
        "This prototype is not affiliated with or endorsed by WHO."
    )


# ─── Chat handler ─────────────────────────────────────────────────────────────

def chat(
    user_message: str,
    history: list[dict],
    thread_id: str,
) -> tuple[list, str, str, str]:
    """
    Handle one chat turn.

    Args:
        user_message: What the health worker typed.
        history:      Gradio chat history in "messages" format
                      [{"role": "user"|"assistant", "content": str}, ...]
        thread_id:    Unique session ID (kept in the browser, gr.BrowserState)

    Returns:
        (updated_history, cleared_input, status_text, triage_card_markdown)
    """
    if not user_message.strip():
        return history, "", _status_text("idle"), ""

    if KB_COUNT == 0:
        warning = (
            "⚠️ **Knowledge base is empty.**\n\n"
            "Run this command first:\n```\npython -m rag.fetch_guidelines && python -m rag.build_kb\n```\n"
            "Then restart the app."
        )
        history = history + [{"role": "user", "content": user_message},
                             {"role": "assistant", "content": warning}]
        return history, "", _status_text("idle"), ""

    # Process the turn through LangGraph
    result = run_turn(APP, thread_id, user_message)

    # Update chat history
    bot_message = result["message"]
    history = history + [{"role": "user", "content": user_message},
                         {"role": "assistant", "content": bot_message}]

    # Build outputs
    status = result["status"]
    triage_card = result["triage_card"] if result["is_complete"] else ""

    return history, "", status, triage_card


def restore_session(thread_id: str | None):
    """
    On page load: if the browser holds a thread_id whose state is in the
    checkpointer (Postgres in production), rebuild the chat so the
    conversation continues, including after a server restart or a reload.
    """
    if not thread_id:
        return [], str(uuid.uuid4()), _status_text("ready"), ""
    state = _get_current_state(APP, {"configurable": {"thread_id": thread_id}})
    if not state:
        return [], thread_id, _status_text("ready"), ""
    history = []
    for m in state.get("messages", []):
        role = {"human": "user", "ai": "assistant"}.get(m.type)
        if role and isinstance(m.content, str):
            history.append({"role": role, "content": m.content})
    done = state.get("phase") == "done"
    return history, thread_id, format_agent_status(state), format_for_gradio(state) if done else ""


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
        "*Retrieval over WHO IMCI and WHO PEN guideline text · LangGraph*"
    )

    # ─── Session state (invisible, per-tab) ──────────────────────────────
    # The session id lives in the browser (localStorage), not in server memory,
    # so a conversation survives a server restart when CHECKPOINT_DB_URL is set.
    # Gradio encrypts BrowserState with `secret`; without a stable secret it
    # generates a random one per launch and stored ids are unreadable after a
    # restart (verified in a real browser). Set BROWSER_STATE_SECRET in production.
    thread_id_state = gr.BrowserState(default_value=None, storage_key="triage_thread_id",
                                      secret=BROWSER_STATE_SECRET)

    # ─── Main layout ─────────────────────────────────────────────────────
    with gr.Row():

        # Left: Chat
        with gr.Column(scale=3):
            chatbot = gr.Chatbot(
                label="Clinical Intake",
                height=460,
                type="messages",
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

    # ─── Guideline attribution (always visible) ──────────────────────────
    gr.Markdown(_attribution_markdown(), elem_classes=["status-bar"])

    # ─── Event handlers ──────────────────────────────────────────────────

    def on_send(msg, hist, tid, request: gr.Request):
        tid = tid or str(uuid.uuid4())
        if msg and msg.strip():
            blocked = RATE_LIMITER.check(msg, tid, client_ip(request))
            if blocked:
                # Not processed: no LLM call, nothing written to the checkpoint.
                hist = hist + [{"role": "user", "content": msg}, {"role": "assistant", "content": blocked}]
                return hist, msg, "⏳ Rate limit reached", gr.update(), tid
        return (*chat(msg, hist, tid), tid)

    demo.load(
        fn=restore_session,
        inputs=[thread_id_state],
        outputs=[chatbot, thread_id_state, status_display, triage_output],
    )

    send_btn.click(
        fn=on_send,
        inputs=[msg_input, chatbot, thread_id_state],
        outputs=[chatbot, msg_input, status_display, triage_output, thread_id_state],
    )
    msg_input.submit(
        fn=on_send,
        inputs=[msg_input, chatbot, thread_id_state],
        outputs=[chatbot, msg_input, status_display, triage_output, thread_id_state],
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
