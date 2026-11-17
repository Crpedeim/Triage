"""
tests/test_postgres_checkpoint.py — sessions persist in Postgres across restarts.

Requires a reachable PostgreSQL. Skipped unless TEST_POSTGRES_URL is set, e.g.
    TEST_POSTGRES_URL=postgresql://postgres@127.0.0.1:54329/triage pytest tests/test_postgres_checkpoint.py

Each turn of the restart test runs in its own OS process (tests/pg_turn_helper.py),
so the only way turn 2 can see turn 1 is through the database.
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
import uuid
from pathlib import Path

import pytest

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

PG_URL = os.getenv("TEST_POSTGRES_URL")
needs_pg = pytest.mark.skipif(not PG_URL, reason="TEST_POSTGRES_URL not set (needs a PostgreSQL server)")


def _turn(thread_id: str, turn: int, message: str) -> dict:
    out = subprocess.run(
        [sys.executable, str(ROOT / "tests" / "pg_turn_helper.py"), PG_URL, thread_id, str(turn), message],
        capture_output=True, text=True, timeout=120, cwd=str(ROOT),
    )
    assert out.returncode == 0, out.stderr[-2000:]
    return json.loads(out.stdout.strip().splitlines()[-1])


@needs_pg
def test_session_survives_process_restart():
    tid = f"pg-restart-{uuid.uuid4()}"
    r1 = _turn(tid, 0, "My 3 year old has had a cough for 4 days.")
    assert r1["message"] == "question after turn 0?" and not r1["is_complete"]

    # New process: state must come back from Postgres
    r2 = _turn(tid, 1, "About 36 breaths a minute.")
    assert r2["human_messages"] == ["My 3 year old has had a cough for 4 days.", "About 36 breaths a minute."]
    assert r2["message"] == "question after turn 1?"

    r3 = _turn(tid, 2, "She is drinking normally.")
    assert r3["is_complete"] and r3["phase"] == "done" and r3["triage_level"] == "urgent"
    assert len(r3["human_messages"]) == 3
    # Each process did exactly one extraction: nothing was re-run on resume
    assert r1["extractions_this_process"] == r2["extractions_this_process"] == r3["extractions_this_process"] == 1


@needs_pg
def test_postgres_sessions_are_isolated_and_persist_in_tables():
    import psycopg

    a, b = f"pg-a-{uuid.uuid4()}", f"pg-b-{uuid.uuid4()}"
    _turn(a, 0, "child A message")
    rb = _turn(b, 0, "child B message")
    assert rb["human_messages"] == ["child B message"]
    with psycopg.connect(PG_URL) as conn:
        n = conn.execute("SELECT count(*) FROM checkpoints WHERE thread_id = ANY(%s)", ([a, b],)).fetchone()[0]
    assert n >= 2


def test_default_is_in_memory_without_db_url(monkeypatch):
    from langgraph.checkpoint.memory import InMemorySaver

    from checkpointing import make_checkpointer
    monkeypatch.delenv("CHECKPOINT_DB_URL", raising=False)
    assert isinstance(make_checkpointer(), InMemorySaver)


def test_serializer_round_trips_state_types_without_unregistered_warning(caplog):
    from checkpointing import make_serializer
    from state import (DangerSign, DangerSignFinding, FindingStatus, PatientSummary, SuspectedCondition,
                       Symptom, TriageLevel, TriageResult, VitalSigns)

    obj = {
        "patient_summary": PatientSummary(
            age=3, chief_complaint="cough", symptoms=[Symptom(name="cough")], vital_signs=VitalSigns(respiratory_rate=50),
            danger_sign_findings=[DangerSignFinding(sign=DangerSign.CHEST_INDRAWING, status=FindingStatus.PRESENT)]),
        "triage_result": TriageResult(triage_level=TriageLevel.EMERGENCY, reasoning="r",
                                      suspected_conditions=[SuspectedCondition(name="Severe pneumonia", confidence=0.9)]),
    }
    serde = make_serializer()
    with caplog.at_level(logging.WARNING):
        back = serde.loads_typed(serde.dumps_typed(obj))
    assert back == obj
    assert not [r for r in caplog.records if "unregistered" in r.getMessage().lower()]
