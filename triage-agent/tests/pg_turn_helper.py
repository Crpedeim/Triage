"""
tests/pg_turn_helper.py — run ONE run_turn() in a fresh OS process against Postgres.

Invoked by tests/test_postgres_checkpoint.py as a subprocess so each turn
happens in a separate process (a real restart: nothing survives in memory).

  python tests/pg_turn_helper.py <db_url> <thread_id> <turn_index> <message>

The stub LLM is scripted per turn index (0, 1, 2). Prints one JSON line.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from tests.stub_llm import StubLLM  # noqa: E402
from tests.test_multiturn import stubbed, summary, triage  # noqa: E402


def main() -> None:
    db_url, thread_id, turn, message = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4]
    from checkpointing import make_checkpointer
    from graph import create_graph, run_turn

    # Turn 0 and 1 ask questions; turn 2 completes.
    completeness = [0.3, 0.5, 0.9][turn]
    llm = StubLLM(structured={"PatientSummary": [summary(completeness)], "TriageResult": [triage(conf=0.9)]},
                  questions=[f"question after turn {turn}?"])
    app = create_graph(checkpointer=make_checkpointer(db_url))
    with stubbed(llm):
        r = run_turn(app, thread_id, message)
    st = r["state"]
    print(json.dumps({
        "message": r["message"], "phase": r["phase"], "is_complete": r["is_complete"],
        "human_messages": [m.content for m in st["messages"] if m.type == "human"],
        "n_messages": len(st["messages"]),
        "triage_level": st["triage_result"].triage_level.value if st.get("triage_result") else None,
        "extractions_this_process": llm.count("structured:PatientSummary"),
    }))


if __name__ == "__main__":
    main()
