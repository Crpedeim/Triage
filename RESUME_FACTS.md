# RESUME_FACTS — TriageAI (`triage-agent/`)

Only facts verified from the code or from commands that were run (2026-11-16 to
2026-11-17, Python 3.13.18, 4-CPU Linux container). Every number traces to a
committed result file under `triage-agent/eval/results/` or to an entry in
[`PHASES.md`](PHASES.md), which records how each was measured. Anything not
measured says **NOT MEASURED**.

> **LLM used for every measurement:** Google Gemini, through the project's
> `openai` provider pointed at Google's OpenAI-compatible endpoint:
> `gemini-3.1-pro-preview` for triage, `gemini-3.8-flash` for intake,
> retrieval and the relevance judge, temperature 0. The code's default
> provider (Anthropic Claude) has **NOT been measured**.
>
> **The evaluation set is self-written:** 31 vignettes whose expected levels
> I derived from the guideline text in the repo. It is a regression and
> safety suite, **not a clinical benchmark**, and no clinician reviewed it.

---

## 1. What it is and who it's for
- A multi-agent triage assistant for **frontline / community health workers**
  (the code targets ASHA workers in India and IMNCI guidance). A health
  worker or caregiver describes a patient in free text. The system:
  1. interviews them one question at a time, including a fixed danger-sign
     checklist
  2. retrieves clinical guideline text from a vector store
  3. returns a triage card with one of four levels (`emergency`, `urgent`,
     `standard`, `self_care`), suspected conditions, actions, a referral
     note and the cited guideline sections
- It started as a hackathon prototype ("WitchHunt Hackathon", per `app.py`)
  and is still described as "not a medical device".

## 2. Agents and hand-offs (final architecture)
A LangGraph `StateGraph` in hub-and-spoke form. **Each user message is one
graph run** on the session's checkpointed state.

| Node | LLM | Role |
|---|---|---|
| Supervisor | none (pure Python) | Routes on state flags. Ends the run when intake has asked a question (`awaiting_user`). |
| Intake | `gemini-3.8-flash` | Structured extraction of a `PatientSummary` (including danger-sign findings and search queries). Asks fixed checklist questions (no LLM) until every applicable group has been asked, then LLM-written questions until complete. The completion message is fixed text. |
| Retrieval | none on attempt 1 | ChromaDB search with the extraction's queries and an age-based filter. Later attempts ask the LLM for reformulated queries. |
| Relevance check | `gemini-3.8-flash` | LLM-as-judge score 1–5 for the retrieved chunks. Runs **in parallel with triage**. |
| Triage | `gemini-3.1-pro-preview` | Structured `TriageResult`, then the deterministic safety floor and the confidence computation. |
| Output | none | Terminal node; `output.py` formats the card. |

**Feedback and self-correction loops (both verified by tests on the real compiled graph):**
1. **Retrieval:** the judge and triage run in parallel. If the judge scores
   < 3, the supervisor discards that triage result and retrieves again with
   reformulated queries. This happens at most 3 times, and stops early when
   a retry finds no new chunk.
2. **Low confidence:** confidence < 0.75 sends the case back to intake. The
   stale triage and retrieval results are cleared, intake asks triage's
   follow-up questions, and retrieval and triage re-run. At most 3 cycles.

**Safety layer (`safety.py`, deterministic):**
- **Checklist groups:** IMNCI general danger signs; breathing (breath
  count, chest indrawing, stridor); dehydration signs; stiff neck; adult
  danger signs. Intake cannot finish until every applicable group has been
  asked.
- **Red flags** come from the findings and from `RED_FLAG_KEYWORDS` (with
  synonyms and negation handling); fast breathing comes from
  `RESPIRATORY_RATE_THRESHOLDS`.
- **Final level = max(LLM level, rule floor).**
  - Danger signs, severe dehydration, stiff neck, adult red flags and
    hypertensive crisis → EMERGENCY.
  - Fast breathing for age and some dehydration → URGENT.
  - The rules never downgrade.
- **Confidence:** the top condition's confidence, adjusted by penalties for
  an ambiguous differential, unanswered checklist items, no citation and low
  completeness. There are no boosts, and it is capped at 0.95.

## 3. Tech stack (pinned in `triage-agent/requirements.txt`)
- **Agents and state:** LangGraph 1.2.22 and LangChain core 1.8.1. Pydantic
  2.12.3 models: `PatientSummary`, `DangerSignFinding`, `VitalSigns`,
  `Symptom`, `SuspectedCondition`, `TriageResult`.
- **Persistence:** `langgraph-checkpoint-postgres` 3.2.1 (`PostgresSaver` on
  a psycopg pool) when `CHECKPOINT_DB_URL` is set, in-memory otherwise;
  explicit serializer allowlist for the state types.
- **LLM providers (`llm.py`):** anthropic (default; Claude Sonnet 4.5 and
  Haiku 4.5), openai or any OpenAI-compatible endpoint, groq, together,
  ollama, and **vertexai** (Gemini on Vertex AI with Application Default
  Credentials).
- **RAG:** ChromaDB 1.5.9 (cosine), sentence-transformers 6.3.0
  `all-MiniLM-L6-v2` (384-dim), PyMuPDF ingestion.
- **UI:** Gradio 5.50.0, with the session id held in the browser
  (`gr.BrowserState`).
- **Tracing:** one trace per turn, `POST /api/v1/traces`, with spans per
  agent and per LLM call (stdlib HTTP, background sender).
- **Container:** `python:3.13-slim` with CPU torch 2.15.0.
- **Deployment target:** Cloud Run (script provided; **not deployed**).
- **Optional:** Google ADK 2.0.0 ReAct retrieval agent with a custom
  `BaseLlm` adapter (`requirements-adk.txt`).
- **Tests:** pytest 9.2.0.

## 4. Before / after (measured)

### 4.1 Test suite
| | Before | After |
|---|---|---|
| Tests collected | 152 | 235 |
| Result | 151 passed, 1 skipped | **234 passed, 1 skipped** with Postgres and the ADK venv (`TEST_POSTGRES_URL` set); 228 passed, 4 skipped in the main env without Postgres |
| Real compiled graph exercised | no: `run_turn()` was only tested against a mocked app, and every LLM call was mocked | yes: multi-turn, feedback loop, parallel fan-out, Postgres across OS processes, tracing against a mock server |

The one remaining skip is the original test that needs a real LLM API key.

### 4.2 Original 6 cases (Phase 0 baseline on the original code)
| | Original code | Final code (same 6 cases, inside the 31-case run) |
|---|---|---|
| Accuracy | 4/6 | 5/6 |
| Under-triage | 2/6 (A severe pneumonia → standard; B pneumonia → standard) | **0/6** (A → emergency, B → urgent) |
| Remaining miss | — | F (grade-2 hypertension) → urgent (over-triage) |

Phase 0 latency was median 75.81 s and p95 153.37 s, but it was measured
sequentially, while every 31-case run used 4 cases in parallel, so I don't
compare those latencies. On the original code the chat flow through
`run_turn()` (what the UI calls) made **0 LLM calls** over 3 turns: intake
never ran.

### 4.3 31-case suite: before (Phase 1 code) vs final
"Before" is the earliest code that can run the suite end to end: the
original code plus the Phase 1 plumbing fixes (multi-turn flow, feedback
loop, intake prompt, Gradio pin) and no clinical-logic changes.

| Metric | Before | **Final** |
|---|---|---|
| **Under-triage (safety metric)** | **10/31** | **0/31** |
| Accuracy (exact level) | 18/31 | **29/31** |
| Over-triage | 3/31 | 2/31 (both grade-2/3 hypertension → urgent) |
| Latency per triage, median (all turns, simulated caregiver excluded) | 63.65 s | **39.92 s** |
| Latency per triage, p95 (nearest rank) | 81.79 s | **49.08 s** |
| Completing turn (intake done → triage card), median / p95 | not recorded | 22.77 s / 38.29 s |
| Median latency of a single user turn (all 90 turns) | not recorded | 9.84 s |
| LLM calls per triage, median | 9 | 5 |
| Tokens per triage, median | 12,534 | 11,189 (p95 13,729) |
| Priced cost per triage, median (triage model only) | $0.0131–$0.0241 | $0.0135–$0.0219 |
| **Total cost per triage** | **NOT MEASURED** | **NOT MEASURED** |
| Confidence | 1.0 in 23/31 cases | 0.782–0.95, never 1.0 (19/31 at the 0.95 cap) |
| LLM errors | 0 | 0 |

In the final run every emergency (15/15), urgent (5/5) and self_care (7/7)
case was correct, and 2 of the 4 standard cases were called urgent.

How each was measured:
- **Latency:** summed system wall time of every `run_turn()` call in a case.
- **Tokens:** API `total_tokens`, including Gemini reasoning tokens.
- **Cost:** only `gemini-3.1-pro-preview` is priced ($2 input / $12 output
  per 1M tokens). That price comes from search results quoting Google's
  pricing page, which I couldn't open. The range is "reasoning tokens not
  billed" to "billed".
- **Total cost** is unknown because I found no published price for
  `gemini-3.8-flash`.

Run-to-run variation: two runs with identical latency-relevant code
differed by about 3 s in median latency (36.95 s vs 39.92 s). Grade-2
hypertension cases F and W flip between standard and urgent across runs.

### 4.4 What each phase contributed (31 cases)
| Change | Effect measured |
|---|---|
| Phase 2: danger-sign checklist, rule floor, EMERGENCY escalation, confidence | under-triage 10/31 → 0/31; accuracy 18/31 → 30/31. The post-hoc override fired 0/31 times: the fixes came from the LLM getting the checklist answers and the rule floor in its prompt. |
| Phase 5: fewer round trips (queries from extraction, fixed completion text), judge ∥ triage, cold-start lock, tail fixes | median 53.95 s → 36.95 s; p95 117.96 s → 52.56 s; LLM calls 7 → 5; under-triage stayed 0/31 |
| Phase 5: flash model for triage (tested, **rejected**) | triage call median 14.21 s (flash) vs 10.88 s (pro); 1 more over-triage |
| Phase 7: ADK ReAct retrieval (tested, kept optional) | accuracy 28/31 vs 29/31; under-triage 0/31 both; median 53.24 s vs 39.92 s; tokens 20,003 vs 11,189 |

### 4.5 Profile (where the time goes)
- On the Phase 1 code, LLM calls were **99.5%** of system wall time.
  Embeddings and vector search: 46–67 ms per case.
- The completing turn made **5 sequential LLM calls** before Phase 5. After
  it, there are 2 sequential calls (extraction, then triage) plus the judge
  in parallel.
- The median flash call produced 470 reasoning tokens against 64 visible
  output tokens (Phase 1 code).

### 4.6 Container (built and run locally; not on Cloud Run)
| Check | Result |
|---|---|
| Clean build / rebuild after an `app.py` change | 283 s / 27 s |
| Image size | 3,642,898,126 bytes (3.64 GB) |
| torch | 2.15.0+cpu, 0 NVIDIA packages |
| Runs as | uid 10001 |
| Secrets or proxy settings in the image | none found |
| Start → HTTP 200 | 8.3 s |
| Memory: idle / after one triage | 464.3 MiB / 497.2 MiB |
| Case A through the container's Gradio API | EMERGENCY in 3 turns (12.6 s, 12.0 s, 25.1 s) |

### 4.7 Sessions and tracing (verified)
- **Postgres** (PostgreSQL 16.15, local): a 3-turn real-Gemini session ran
  every turn in a **new OS process**, and each one restored the
  conversation from Postgres. It finished `urgent`, as expected.
- **Real browser** (Playwright and Chromium): after the container was
  replaced and the page reloaded, the conversation was restored and
  continued. Without a stable `BROWSER_STATE_SECRET`, 0 messages were
  restored.
- **Tracing:** 3/3 traces delivered to a local mock `/api/v1/traces` with
  per-agent spans and real latency and tokens. For example, the triage span
  took 14,955.5 ms, used 4,666 tokens and cost $0.031092. The trace total
  cost was `null`, because flash is unpriced.

## 5. What is deployed where
- **Nothing is deployed.** There is **no live URL**. The build environment
  had no Google Cloud credentials.
- **Ready for you to run:**
  - `triage-agent/Dockerfile` (built and run locally, §4.6)
  - `triage-agent/deploy/deploy_cloud_run.sh` and `deploy/CLOUD_RUN.md`:
    Cloud Build → Cloud Run, Gemini via Vertex AI, optional Cloud SQL
    Postgres, tracing and Secret Manager
- **Not run against Google Cloud:** the deploy script and the Vertex AI
  code path. Vertex is covered by 4 offline unit tests only, and no Vertex
  call has been made.
- **Not connected:** the tracing client has only been tested against a
  local mock server, because the platform's URL and schema weren't
  available. The payload field names are an assumed shape, documented in
  `tracing.py`.
- **Code:** branch `claude/adoring-goldberg-d2ncfb`, one commit per phase,
  with measurements in `PHASES.md`.

## 6. Remaining limitations
1. **The evaluation is self-written and small** (31 vignettes, one run per
   configuration). It is not a clinical benchmark, and no clinician
   reviewed the labels. Differences of 1–2 cases are within observed
   run-to-run variation.
2. **Over-triage of hypertension.** Grade-2/3 hypertension without emergency
   features is labelled `standard`, and the LLM calls it `urgent` in 2–3 of
   the 4 standard cases depending on the run.
3. **Tiny knowledge base:** 8 sample chunks (6 IMNCI, 2 WHO PEN) and no
   real guideline PDFs. Adult chest pain and stroke have no guideline chunk;
   they are handled only by the red-flag rules.
4. **Total cost per triage is unknown** (no `gemini-3.8-flash` price), and
   the pro price is from search results, not the pricing page itself.
5. **Claude, the code's default provider, is NOT MEASURED.** Neither is
   Vertex AI.
6. **Not deployed; no live URL.** Cloud Run cold start, autoscaling
   behaviour and multi-instance sessions are NOT MEASURED, and concurrency
   20 per instance is not load-tested.
7. **Tracing payload shape is assumed** and unconnected to the real
   platform.
8. **The safety floor depends on extraction.** The rules are
   deterministic, but whether a finding is "present" is still read from the
   conversation by the LLM. The low-confidence loop-back path was fixed
   (HTTP 400 when the request ended with a model turn) but was not
   exercised by real LLM calls in the final run; only tests cover it.
9. **Latency:** about 40 s median per triage across 2–4 user turns; the
   final turn waits on 2 sequential LLM calls (median 22.77 s). Gemini
   response time is the floor.
10. **Choices that differ from the source text:** the rules treat any
    convulsion as a danger sign (the sample IMNCI text says "more than one
    convulsion"), and young infants under 2 months use child thresholds
    only. Neither is in the eval.
11. **Upgrade blockers:** Gradio 5.50.0 caps pydantic at 2.12.3, which
    forces older `google-genai` and `google-adk` releases, and `app.py`
    emits Gradio 6 deprecation warnings.
12. Still a prototype, **not a medical device**.
