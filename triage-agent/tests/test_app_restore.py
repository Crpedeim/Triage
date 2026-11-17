"""tests/test_app_restore.py — the UI restores a saved conversation by thread_id (page reload / restart)."""

from __future__ import annotations

import sys
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from tests.stub_llm import StubLLM  # noqa: E402
from tests.test_multiturn import stubbed, summary, triage  # noqa: E402


def test_restore_session_rebuilds_chat_from_checkpoint():
    import app
    from graph import run_turn

    llm = StubLLM(structured={"PatientSummary": [summary(0.3)]}, questions=["How many breaths per minute?"])
    tid = str(uuid.uuid4())
    with stubbed(llm):
        run_turn(app.APP, tid, "3 year old with cough")
    history, restored_tid, status, card = app.restore_session(tid)
    assert restored_tid == tid
    assert history[0] == {"role": "user", "content": "3 year old with cough"}
    assert history[-1]["role"] == "assistant" and history[-1]["content"]
    assert card == ""


def test_restore_session_unknown_or_missing_id_starts_fresh():
    import app
    h, tid, _, _ = app.restore_session(None)
    assert h == [] and tid
    h2, tid2, _, _ = app.restore_session("no-such-thread")
    assert h2 == [] and tid2 == "no-such-thread"


def test_restore_session_shows_card_when_done():
    import app
    from graph import run_turn

    llm = StubLLM(structured={"PatientSummary": [summary(0.9)], "TriageResult": [triage(conf=0.9)]})
    tid = str(uuid.uuid4())
    with stubbed(llm):
        r = run_turn(app.APP, tid, "3 year old, cough, 36/min, drinking well, alert")
    assert r["is_complete"]
    _, _, _, card = app.restore_session(tid)
    assert "URGENT" in card
