"""
checkpointing.py — Checkpointer factory (session persistence).

  CHECKPOINT_DB_URL set   → LangGraph PostgresSaver on a psycopg connection pool.
                            Sessions survive process restarts and are shared by
                            every app instance pointing at the same database.
  CHECKPOINT_DB_URL unset → in-memory saver (development and tests only;
                            sessions are lost on restart).

Both use a serializer with an explicit allowlist of the project's state types.
LangGraph otherwise warns that deserializing unregistered types from a
checkpoint "will be blocked in a future version", which with a persistent
store would make saved sessions unreadable after an upgrade.

Example:
    CHECKPOINT_DB_URL=postgresql://user:pass@host:5432/triage
"""

from __future__ import annotations

import atexit
import os

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer

# Every project type that can appear in TriageState
STATE_TYPES: list[tuple[str, str]] = [
    ("state", name) for name in (
        "PatientSummary", "Symptom", "VitalSigns", "DangerSignFinding", "DangerSign", "FindingStatus",
        "Severity", "SuspectedCondition", "TriageResult", "TriageLevel", "Phase",
    )
]


def make_serializer() -> JsonPlusSerializer:
    return JsonPlusSerializer(allowed_msgpack_modules=STATE_TYPES)


_POOLS: list = []


def make_checkpointer(db_url: str | None = None):
    """Return a PostgresSaver when a database URL is configured, else an in-memory saver."""
    url = db_url if db_url is not None else os.getenv("CHECKPOINT_DB_URL")
    if not url:
        return InMemorySaver(serde=make_serializer())

    from langgraph.checkpoint.postgres import PostgresSaver
    from psycopg.rows import dict_row
    from psycopg_pool import ConnectionPool

    pool = ConnectionPool(
        conninfo=url,
        min_size=1,
        max_size=int(os.getenv("CHECKPOINT_DB_POOL_SIZE", "10")),
        # Settings PostgresSaver requires: autocommit, no server-side prepared
        # statements (safe behind pgbouncer), dict rows.
        kwargs={"autocommit": True, "prepare_threshold": 0, "row_factory": dict_row},
        open=True,
    )
    _POOLS.append(pool)
    saver = PostgresSaver(pool, serde=make_serializer())
    saver.setup()  # idempotent: creates/migrates the checkpoint tables
    return saver


@atexit.register
def _close_pools() -> None:
    for pool in _POOLS:
        try:
            pool.close()
        except Exception:
            pass
