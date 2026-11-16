# RESUME_FACTS — TriageAI (`triage-agent/`)

Everything here was either read from the code or produced by a command run on
2026-11-16 (Python 3.13.18, 4-CPU Linux container). Anything that was not
measured is marked **NOT MEASURED**.

> **Key caveat:** no LLM API key was available, so **no real LLM call was made**.
> Every LLM-dependent number (triage latency, tokens, cost, accuracy) is
> **NOT MEASURED**. LLM-free parts (embeddings, vector search, graph wiring,
> safety override, output card, Gradio launch) were run for real. Where an
> agent's LLM was swapped for a scripted stub to exercise the graph, that is
> stated.

---

## 1. What it does and who it's for

- A multi-agent triage assistant. A health worker types a free-text
  description of a patient. The system interviews them one question at a
  time, retrieves clinical guideline text from a vector store, and returns a
  triage card with one of four levels: `emergency`, `urgent`, `standard`,
  `self_care` (`state.py: TriageLevel`).
- **Intended users:** frontline / community health workers. The code
  repeatedly refers to ASHA workers (`state.py: VitalSigns` docstring) and
  IMNCI (WHO Integrated Management of Neonatal and Childhood Illness)
  guidance. Adults are partly supported through WHO PEN sample text.
- `app.py` describes it as a hackathon prototype ("WitchHunt Hackathon,
  Health & WellBeing theme"), "not a medical device".

## 2. Agents and hand-offs

Hub-and-spoke LangGraph `StateGraph` (`graph.py`). Entry point is
`supervisor`. `intake`, `retrieval` and `triage` each have an unconditional
edge back to `supervisor`. `supervisor` routes through
`add_conditional_edges(route_next)`, and `output → END`. The graph compiles
with `MemorySaver` and `interrupt_before=["intake"]`.

| Node | File | LLM? | What it does |
|---|---|---|---|
| Supervisor | `agents/supervisor.py` | **No** (pure Python) | Reads state flags (`intake_complete`, `retrieved_guidelines`, `triage_result`, `triage_confidence`, `cycle_count`) and returns the next node name. |
| Intake | `agents/intake.py` | Fast model | Call 1: `with_structured_output(PatientSummary)` over the whole conversation. If `completeness_score ≥ 0.7`, it makes call 2 to write a completion message and sets `intake_complete=True`. Otherwise call 2 asks the next single question, using queued Triage follow-up questions first. |
| Retrieval | `agents/retrieval.py` + `tools/vector_search.py` | Fast model | Up to `MAX_RETRIEVAL_ATTEMPTS=3` loops. Each loop: the LLM writes search queries as a JSON array (falling back to rule-based queries), runs a multi-query ChromaDB search (4 results per query, age-based filter `imnci` for under 5 and `who_pen` for 18+, retried without the filter if empty), then an **LLM-as-judge** scores relevance 1–5. It stops when the score is ≥ 3 and otherwise reformulates, skipping queries it has already tried. It keeps the best-scoring set. |
| Triage | `agents/triage.py` | Main model | `with_structured_output(TriageResult)`, followed by Python post-processing (below). |
| Output | `graph.py: _format_output` | No | No-op terminal node. Cards are formatted by `output.py`. |

**Feedback / self-correction loops in the code:**
1. **Retrieval self-correction:** retrieve → LLM judge → reformulate, at most 3 attempts.
2. **Triage → Intake feedback loop:** if `triage_confidence < 0.75` and
   `cycle_count < max_cycles - 1` (with `max_cycles=3`), the supervisor routes
   back to intake, resets `intake_complete`, and increments `cycle_count`.
   Triage puts up to 3 follow-up questions in state for intake to ask.
3. **Red-flag safety override (Python, not LLM):** if `PatientSummary.red_flags`
   is non-empty and the LLM's level is `standard` or `self_care`, the level is
   forced to `urgent`. An "URGENT: Refer…" action is prepended and a referral
   note is generated.

**Confidence is computed in Python** (`_compute_confidence`). It starts from
the top suspected condition's confidence, adds 0.15 if red flags are present
and 0.05 if guidelines were cited, and subtracts 0.2 if completeness is below
0.4. It returns 0.4 if no conditions were given.

## 3. Tech stack (from code and `requirements.txt`; versions are what pip installed in this run)

- **Orchestration:** LangGraph (1.2.22), LangChain core (1.8.1), `MemorySaver` checkpointer keyed by `thread_id`.
- **LLM:** provider abstraction in `llm.py` with `anthropic` (default),
  `openai`, `groq`, `together` and `ollama`, chosen by the `LLM_PROVIDER` env
  var. Default models: main `claude-sonnet-4-5` (Triage), fast
  `claude-haiku-4-5-20251001` (Intake, Retrieval query generation, relevance
  judge). Temperature 0.0, `max_tokens=4096` for Anthropic.
- **Data models:** Pydantic v2: `Symptom`, `VitalSigns`, `PatientSummary`, `SuspectedCondition`, `TriageResult`. Graph state is the `TriageState` TypedDict, with an `add_messages` reducer.
- **RAG:** ChromaDB `PersistentClient`, cosine space, one collection
  `clinical_guidelines`. Embeddings come from sentence-transformers
  `all-MiniLM-L6-v2` (384-dim, local), with optional OpenAI
  `text-embedding-3-small`. PDF ingestion uses PyMuPDF and
  `RecursiveCharacterTextSplitter` (chunk size 500, overlap 50).
- **Frontend:** Gradio (`app.py`).
- **Tests:** pytest.
- **External APIs:** only the chosen LLM provider's chat API (and OpenAI embeddings, if selected). No other APIs.

## 4. Measured numbers

### Test suite
- **Command:** `cd triage-agent && python -m pytest -q`
- **Result:** **151 passed, 1 skipped, 0 failed, out of 152 collected.** Of the tests that ran, 151/151 passed.
  - Per file (from `--collect-only`): `test_state` 41, `test_rag` 43, `test_agents` 38, `test_graph` 30.
  - The one skip is `tests/test_rag.py:591`, "No API key available for LLM instantiation test".
- Run twice: with real MiniLM embeddings (22.82 s) and with
  `USE_MOCK_EMBEDDINGS=1` hash embeddings (3.81 s). Both gave 151 passed, 1 skipped.
- **Every LLM call in the suite is mocked** (`@patch("agents.*.get_llm" / "get_fast_llm")`).
  The `run_turn()` tests use a mocked app, not the compiled graph.

### Ingestion
- `python -m rag.ingest --sample` stored **8 chunks**: 6 IMNCI (`sample_imnci.txt`) and 2 WHO PEN (`sample_who_pen.txt`).
- The repo contains **no guideline PDFs** (`rag/guidelines/` does not exist), so these 8 sample chunks are the whole knowledge base.

### Retrieval latency (real MiniLM + Chroma, no LLM)
Measured by timing `multi_query_search()` 20 times per case on the 4
example inputs in `app.py`. Queries came from the project's rule-based
`_fallback_queries()`, using hand-filled `PatientSummary` objects. Median
wall time on a 4-CPU container:

| Case | Filter | # queries | Median latency | Top retrieved section (cosine distance) |
|---|---|---|---|---|
| 3 yo, cough 4 d, fast breathing | imnci | 4 | 67.1 ms | ASSESS AND CLASSIFY COUGH OR DIFFICULT BREATHING (0.251) |
| 4 yo, mild cough | imnci | 3 | 48.0 ms | GENERAL DANGER SIGNS (0.386) |
| 2 yo, diarrhoea 3 d | imnci | 3 | 46.5 ms | GENERAL DANGER SIGNS (0.386), then ASSESS AND CLASSIFY DIARRHOEA (0.389) |
| 55 yo, headache, BP 160/100 | who_pen | 4 | 59.4 ms | CARDIOVASCULAR RISK MANAGEMENT (0.422) |

### LLM calls per triage (counted, not timed)
With a stub LLM recording each call through the real compiled graph, a case
where intake finishes on its first pass and retrieval passes on its first
attempt made **5 LLM calls**:
1. Intake extraction
2. Intake completion message
3. Retrieval query generation
4. Relevance judge
5. Triage structured output

From the code, each extra interview turn adds 2 intake calls, and each extra
retrieval attempt adds 2 more calls (at most 3 attempts).

### NOT MEASURED
- **End-to-end latency per triage with a real LLM:** NOT MEASURED (no API key).
- **Tokens and cost per run:** NOT MEASURED (no API key; the code does not log usage).
- **Triage accuracy:** NOT MEASURED. The repo has **no labelled evaluation set**, and no real LLM was run.
- **Retrieval accuracy (precision/recall):** NOT MEASURED. No labelled relevance data exists; the table above shows observed rankings only.

## 5. What was verified to work, and what wasn't

Verified by running it:
- The Gradio app **launches and serves HTTP 200** on port 7860 with **Gradio 5.50.0**.
- The graph compiles. When driven past the interrupt with `app.invoke(None, cfg)` (stub LLM, real retrieval), it goes supervisor → intake → retrieval → triage → output and reaches `phase="done"`.
- The red-flag override works. A stub `standard` result with `red_flags=["chest indrawing"]` came out as **`urgent`**, with the referral note and the prepended "URGENT: Refer…" action. Confidence was 1.0 (0.8 + 0.15 + 0.05).
- Low confidence (forced to 0.3) routes back to intake with `cycle_count=1`.

## 6. Limitations (each one observed in a run or read in the code)

1. **The multi-turn chat flow does not reach the intake agent.** `run_turn()`
   (used by `app.py`) calls `app.invoke({...new message...})` on every turn.
   With `interrupt_before=["intake"]`, each call restarts at the supervisor and
   pauses before intake. Over 3 turns through `run_turn()` on the real compiled
   graph, **the LLM was called 0 times**, and every reply was the fallback
   string "I'm ready to help. Please describe the patient's condition." The
   pipeline only progressed when resumed manually with `invoke(None)`.
2. **The feedback loop never re-runs triage.** After looping back to intake,
   `route_next` keeps returning `"intake"`: case 5 matches before case 6, and
   the supervisor only increments `cycle_count` when the phase is `triage`. In
   the run, 5 more resumes made only intake calls; triage was never called
   again and `cycle_count` stayed at 1.
3. **`app.py` crashes on Gradio 6.x.** `requirements.txt` says `gradio>=5.0.0`,
   which installed 6.32.0. On that version,
   `gr.Chatbot(bubble_full_width=...)` raises `TypeError`. The app works on
   Gradio 5.50.0, which also emits deprecation warnings for the tuple chat
   format.
4. **Tiny knowledge base.** Only 8 built-in sample chunks; no real IMNCI or WHO PEN PDFs are included.
5. **Red flags depend entirely on LLM extraction.** `RED_FLAG_KEYWORDS` and
   `RESPIRATORY_RATE_THRESHOLDS` are defined in `state.py` but not used by any
   non-test code. The override only escalates to `urgent`, never to `emergency`.
6. **The triage prompt asks for `follow_up_questions`, but `TriageResult` has
   no such field.** Follow-up questions are actually generated by Python rules
   (`_extract_follow_up_questions`).
7. **Declared but not implemented:** `Phase.CONSULTATION` and
   `has_diagnostic_ambiguity()` ("Enhancement 3") exist, but the graph has no
   consultation node.
8. **No usage or latency logging**, so cost and latency can't be read from the app itself.
9. **Unit tests mock every LLM call** and test `run_turn()` against a mocked
   app. That is why limitations 1 and 2 are not caught (151/151 of the tests
   that ran passed).
10. In-memory `MemorySaver` only; sessions are lost on restart.
11. The project describes itself as a prototype, not a medical device.
