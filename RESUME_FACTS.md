# RESUME_FACTS — TriageAI (`triage-agent/`)

Everything here was either read from the code or produced by a command run on
2026-11-16 (Python 3.13.18, 4-CPU Linux container). Anything that was not
measured is marked **NOT MEASURED**.

> **Which LLM was measured:** the code's default provider is Anthropic
> (`claude-sonnet-4-5` / `claude-haiku-4-5-20251001`), but no Anthropic key
> was available. Real LLM runs used **Google Gemini**, reached **with no code
> changes** through the project's existing `openai` provider pointed at
> Google's OpenAI-compatible endpoint:
>
> ```
> LLM_PROVIDER=openai
> OPENAI_BASE_URL=https://generativelanguage.googleapis.com/v1beta/openai/
> MODEL_NAME=gemini-3.1-pro-preview      # main model -> Triage Agent
> FAST_MODEL_NAME=gemini-3.8-flash       # fast model -> Intake, Retrieval, relevance judge
> ```
>
> Every LLM number below is for that Gemini configuration. **The default
> Claude configuration was NOT MEASURED.**

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
  judge). Temperature 0.0. Overridable with `MODEL_NAME` / `FAST_MODEL_NAME`.
  Measured here on Gemini through the `openai` provider (see top).
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

### 4.1 Test suite
- **Command:** `cd triage-agent && python -m pytest -q`
- **Result:** **151 passed, 1 skipped, 0 failed, out of 152 collected.** Of the tests that ran, 151/151 passed.
  - Per file (from `--collect-only`): `test_state` 41, `test_rag` 43, `test_agents` 38, `test_graph` 30.
  - The one skip is `tests/test_rag.py:591`, "No API key available for LLM instantiation test".
- Run twice: with real MiniLM embeddings (22.82 s) and with
  `USE_MOCK_EMBEDDINGS=1` hash embeddings (3.81 s). Both gave 151 passed, 1 skipped.
- **Every LLM call in the suite is mocked.** The `run_turn()` tests use a mocked app, not the compiled graph.

### 4.2 Ingestion
- `python -m rag.ingest --sample` stored **8 chunks**: 6 IMNCI and 2 WHO PEN.
  The repo contains **no guideline PDFs**, so these 8 sample chunks are the whole knowledge base.

### 4.3 End-to-end Gemini run: 6 cases
**How it was run:**
- The real compiled graph from `create_graph()` was used unchanged. It was
  driven by resuming past the interrupt (`update_state` with the caregiver's
  reply, then `invoke(None)`), because `run_turn()` does not work (see §6.1).
- A **simulated caregiver** (a separate `gemini-3.8-flash` call, not part of
  the system) answered each intake question from a hidden patient vignette.
  Its time and tokens are **excluded** from all system figures below.
- A LangChain callback recorded every system LLM call's model, latency and token usage.
- One run per case, run sequentially.

**Labels.** I wrote the 6 vignettes. Each expected level comes from the
project's own sample guideline text plus the level mapping in `state.py`
(IMNCI pink → emergency, yellow → urgent, green → self_care). This is a
small, self-made sanity set, **not a validated clinical benchmark**.

| Case | Expected (basis) | Predicted | Match | Intake Qs | LLM calls | System wall time | System tokens |
|---|---|---|---|---|---|---|---|
| A: 3 yo, cough, RR 50, chest indrawing | emergency (severe pneumonia) | **standard** | ✗ | 2 | 9 | 98.48 s | 13,499 |
| B: 3 yo, cough, RR 46, no danger signs | urgent (pneumonia, yellow) | **standard** | ✗ | 2 | 9 | 67.91 s | 12,953 |
| C: 4 yo, mild cough, RR 28 | self_care (cough or cold) | self_care | ✓ | 1 | 7 | 51.34 s | 10,260 |
| D: 2 yo, diarrhoea, lethargic, sunken eyes, drinks poorly | emergency (severe dehydration) | emergency | ✓ | 2 | 9 | 64.75 s | 12,566 |
| E: 18 mo, fever, 2 convulsions today | emergency (very severe febrile disease) | emergency | ✓ | 3 | 11 | 90.51 s | 17,439 |
| F: 55 yo, headache, BP 160/100, no emergency features | standard (WHO PEN grade 2, refer within 24–48 h) | standard | ✓ | 1 | 7 | 57.87 s | 10,209 |

**Accuracy on these 6 labelled cases: 4/6 exact match.** It is still 4/6
even if `urgent` is accepted for the danger-sign cases. Both misses were
**under-triage** to `standard`.

**Latency per triage** (system wall time per case, caregiver excluded): min
**51.34 s**, median **66.33 s**, mean **71.81 s**, max **98.48 s** (n = 6).
Almost all of it is LLM response time:
- Per node: intake median 14.14 s (n = 17), retrieval median 12.93 s (n = 6), triage median 12.14 s (n = 6).
- Per call: `gemini-3.1-pro-preview` median 12.14 s (6 calls); `gemini-3.8-flash` median 6.54 s (46 calls, range 3.74–23.22 s).
- Non-LLM pieces: vector search took 46–67 ms per case (median of 20 runs,
  §4.4), and the one-time cold load of the embedding model took 5.5 s.
- An earlier trial of case A took 159.6 s and then 113.6 s, so latency varies run to run.

**Tokens per triage** (from the API's usage fields): min **10,209**, median
**12,760**, max **17,439**; 76,926 in total across the 6 cases. Gemini's
`total_tokens` was larger than input + output on every call. I recorded the
difference as reasoning ("thinking") tokens; that label is my inference, the
API did not name it.

| Case | Pro: input / output / reasoning | Flash: input / output / reasoning |
|---|---|---|
| A | 2307 / 728 / 663 | 4036 / 1004 / 4761 |
| B | 2308 / 717 / 1119 | 3989 / 888 / 3932 |
| C | 2281 / 666 / 597 | 2807 / 477 / 3432 |
| D | 2343 / 762 / 643 | 4024 / 923 / 3871 |
| E | 2107 / 841 / 664 | 5540 / 1067 / 7220 |
| F | 1464 / 940 / 1806 | 2465 / 540 / 2994 |

**Cost per run: only partly computed.**
- **Pro model (Triage Agent):** using $2 input / $12 output per 1M tokens for
  `gemini-3.1-pro-preview` (prompts ≤200k). I took that price from search
  results quoting Google's pricing page; I could not open the page itself.
  The cost is **$0.0126–$0.0143 per run** if reasoning tokens are not billed,
  or **$0.0197–$0.0359 per run** if they are billed at the output rate. The
  medians are $0.0136 and $0.0219.
- **Flash model:** NOT MEASURED. I found no published price for `gemini-3.8-flash`.
- **Total cost per run: NOT MEASURED.**

**Loops observed in these runs:**
- Retrieval passed on the first attempt in all 6 cases: one query-generation
  call and one judge call each. The call counts equal 2 × (intake turns) + 3
  in every case. The reformulation loop was therefore never triggered.
- Triage confidence was 1.0 in all 6 cases, so the Triage → Intake feedback
  loop was never triggered by a real LLM.
- The red-flag override did not fire. In D and E the LLM itself chose
  `emergency` (red flags: `lethargic` and `altered consciousness` in D,
  `convulsions` in E). In A, no red flag was extracted.

### 4.4 Retrieval latency (real MiniLM + Chroma, no LLM)
I timed `multi_query_search()` 20 times per case on `app.py`'s 4 example
inputs, using queries from the project's rule-based `_fallback_queries()`.

| Case | Filter | Median latency | Top section (cosine distance) |
|---|---|---|---|
| 3 yo, cough, fast breathing | imnci | 67.1 ms | ASSESS AND CLASSIFY COUGH OR DIFFICULT BREATHING (0.251) |
| 4 yo, mild cough | imnci | 48.0 ms | GENERAL DANGER SIGNS (0.386) |
| 2 yo, diarrhoea | imnci | 46.5 ms | GENERAL DANGER SIGNS (0.386), then DIARRHOEA (0.389) |
| 55 yo, BP 160/100 | who_pen | 59.4 ms | CARDIOVASCULAR RISK MANAGEMENT (0.422) |

### 4.5 NOT MEASURED
- Anything for the default Claude configuration (latency, tokens, cost, accuracy).
- Total cost per run (the flash-model price is unknown).
- Retrieval precision/recall (no labelled relevance data).
- Accuracy on any validated or external clinical dataset (the repo has none; §4.3 uses 6 vignettes I wrote).
- Variance across repeated runs of the same case (each case was run once in the final run).

## 5. Verified working

- The Gradio app **launches and serves HTTP 200** on port 7860 with **Gradio 5.50.0**.
- All 6 Gemini cases ran supervisor → intake → retrieval → triage → output and reached `phase="done"` with **0 LLM errors**.
- Structured output (`PatientSummary`, `TriageResult`) parsed with no errors on both Gemini models.
- A stub-LLM run confirmed that the red-flag override escalates `standard` to
  `urgent` and adds a referral note, and that low confidence (0.3) routes back
  to intake.

## 6. Limitations (each one observed in a run or read in the code)

1. **The multi-turn chat flow does not reach the intake agent.** `run_turn()`
   (used by `app.py`) calls `app.invoke({...new message...})` on every turn.
   With `interrupt_before=["intake"]`, each call restarts at the supervisor and
   pauses before intake. Over 3 turns through `run_turn()`, the LLM was called
   **0 times**, and every reply was the fallback string "I'm ready to help.
   Please describe the patient's condition."
2. **Under-triage of a danger sign (case A).** Intake marked the interview
   complete (completeness 0.8) after asking only about breathing rate,
   drinking and alertness. It never asked about chest indrawing, so no red
   flag was recorded, the override could not fire, and a severe-pneumonia
   case was classified `standard` with confidence 1.0. Case B (pneumonia) was
   also classified `standard` against an expected `urgent`.
3. **Confidence saturates at 1.0.** In all 6 Gemini cases the top condition
   came back at 0.95 and citing guidelines adds 0.05. So confidence was 1.0
   even for the wrong answers, and the feedback loop never ran.
4. **The feedback loop never re-runs triage** (stub-LLM run). After looping
   back to intake, `route_next` keeps returning `"intake"`: case 5 matches
   before case 6. Over 5 more resumes, triage was never called again.
5. **Intake leaked raw JSON to the user (case E).** One "question" shown to
   the caregiver was a JSON block of the patient summary, not a question. The
   question call reuses `EXTRACTION_SYSTEM` as its system prompt
   (`agents/intake.py`).
6. **`app.py` crashes on Gradio 6.x.** `requirements.txt` says `gradio>=5.0.0`,
   which installed 6.32.0, where `gr.Chatbot(bubble_full_width=...)` raises
   `TypeError`. The app works on 5.50.0, with deprecation warnings.
7. **Tiny knowledge base.** Only 8 built-in sample chunks; no real IMNCI or WHO PEN PDFs are included.
8. **Red flags depend entirely on LLM extraction.** `RED_FLAG_KEYWORDS` and
   `RESPIRATORY_RATE_THRESHOLDS` in `state.py` are not used by any non-test
   code. The override only escalates to `urgent`, never to `emergency`.
9. **The triage prompt asks for `follow_up_questions`, but `TriageResult` has
   no such field.** Follow-up questions actually come from Python rules.
10. **Declared but not implemented:** `Phase.CONSULTATION` and `has_diagnostic_ambiguity()` ("Enhancement 3"); the graph has no consultation node.
11. **Slow per triage:** a median of 66 s across 6 Gemini runs, almost all of it sequential LLM calls (7–11 per case).
12. **No usage or latency logging** in the app; the figures above came from an external callback.
13. **Unit tests mock every LLM call** and test `run_turn()` against a mocked app, which is why limitations 1 and 4 are not caught.
14. **LangGraph warns** that deserializing the project's types
    (`PatientSummary`, `TriageResult`, enums) from checkpoints "will be blocked
    in a future version".
15. In-memory `MemorySaver` only; sessions are lost on restart. The project describes itself as a prototype, not a medical device.
