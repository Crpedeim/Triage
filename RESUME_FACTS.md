# RESUME_FACTS — TriageAI (`triage-agent/`)

Only facts verified from the code or from commands that were run (2026-11-16 to
2026-11-17; Phases 0–7 on Python 3.13.18 in a 4-CPU Linux container; Rounds 2
and 3 on a Windows laptop with Python 3.14.0, on Google Cloud Run and on the
live eval platform). Every number traces to a
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
  `all-MiniLM-L6-v2` (384-dim). Knowledge base: **874 chunks from 3 WHO
  documents** (IMCI chart booklet 2014, IMCI handbook 2005, WHO PEN 2020),
  built at image build time with table-aware PyMuPDF 2.1 extraction
  (glyph-font repair, IMCI classification rows as sign → classification →
  treatment, `find_tables` rows, quality gate).
- **UI:** Gradio 5.50.0, with the session id held in the browser
  (`gr.BrowserState`).
- **Tracing:** connected to the live LLM eval platform
  (https://eval-web-827630707333.us-central1.run.app). One trace per user turn
  in the platform's documented `POST /api/v1/traces` format: a span tree of
  turn → agent → LLM call, with model, tokens, latency and status. Cost is
  computed by the platform. Sent by a stdlib-HTTP background sender.
- **Container:** `python:3.13-slim` with CPU torch 2.15.0.
- **Deployment:** **live on Cloud Run** (§5): max 2 instances, session
  affinity, scale to zero, sessions in Neon Postgres, per-session / per-IP
  rate limits (`ratelimit.py`), secrets in Secret Manager.
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

### 4.8 Round 2: real WHO guidelines, persistence, public deployment
"Before" is the end of Phase 7 (`main` at `e6230ed`); "after" is Round 2
(`claude/who-guidelines-hardening`). Details: `PHASES.md`, R2 Phases 0–5.

| Metric | Before | After |
|---|---|---|
| Knowledge base: documents | 0 (8 hand-written excerpts) | **3 WHO documents** |
| Knowledge base: pages | 0 | **338 pages: 248 used, 0 replaced by curated excerpts, 90 skipped** (front matter, forms, charts, 2 garbled) |
| Knowledge base: chunks | 8 (6 IMNCI, 2 WHO PEN) | **874** (788 IMCI, 86 WHO PEN); 900/120 characters; 8 of 874 over the embedding model's 256 tokens |
| Extraction quality (20 random chunks per document) | — | 0/60 garbled in the final sample (first pass: 6/60, all fixed) |
| Accuracy (31 cases) | 29/31 | **30/31** |
| **Under-triage** | **0/31** | **0/31** |
| Over-triage | 2/31 (F, X) | 1/31 (X) |
| Latency per triage, median / p95 | 39.92 / 49.08 s | 43.74 / 71.24 s |
| Completing turn, median / p95 | 22.77 / 38.29 s | 24.40 / 60.74 s |
| Tokens per triage, median (p95) | 11,189 (13,729) | 12,262 (20,529) |
| LLM calls per triage, median | 5 | 5 |
| Priced cost per triage, median (triage model only) | $0.0135–$0.0219 | $0.0149–$0.0240 |
| Total cost per triage | NOT MEASURED | NOT MEASURED |
| Failed triage LLM call → result | STANDARD, safety floor skipped | at least URGENT; floor applied (danger signs → EMERGENCY) |
| Image build on Cloud Build | fails ("--mount option requires BuildKit") | builds (`deploy/cloudbuild.yaml`) |
| Cloud Run cold start (start → startup probe), 3 deploys each | 47.68 s median (35.50–52.17) | 45.36 s median (36.28–50.31); no measurable difference |
| Cold start on the live public URL after scale to zero | NOT MEASURED (not deployed) | 50.35 s first request; 49.40 s instance start → probe (R3: once 84.38 s / 83.38 s) |
| Live per-turn latency (8 turns, 3 cases) | NOT MEASURED | median 12.97 s (4.97–28.32 s) |
| Live cases | NOT MEASURED | 3/3 correct (A EMERGENCY, B URGENT, C SELF CARE) |
| Session survives scale to zero | NOT MEASURED on Cloud Run | yes: 4/4 messages restored on a new instance, conversation completed |
| One conversation across instances | NOT MEASURED on Cloud Run | yes: turns on 2 and on 3 different instances, 6/6 messages restored |
| "Deserializing unregistered type" warnings | 0 with the allowlist (2 with LangGraph's default serializer) | 0; new test guards the allowlist |
| Rate limits | none | 7/7 live checks passed, including a spoofed `X-Forwarded-For` |
| Tests | 228 passed, 4 skipped without Postgres (Phase 7, Linux container) | 262 passed, 4 skipped, 1 failed on the Windows laptop without Postgres (the failure, a `test_tracing` platform-down test, also fails on untouched `main` there); `test_postgres_checkpoint.py` against Neon: 5 passed |

### 4.9 Round 3: tracing to the live eval platform
"Before" is Round 2; "after" is Round 3 (`PHASES.md`, R3 Phases 1–3).

| Metric | Before | After |
|---|---|---|
| Trace payload | assumed field names, never sent to a real platform | the platform's documented contract. A test parses `TRACING_API.md` and checks its examples, its 422 error text and every response code; the platform's own zod schema accepts the payloads |
| Traces delivered (31-case run) | NOT MEASURED | **95 sent, 95 stored**, same trace ids; 0 failed, dropped or retried; 765 spans |
| Production traces (3 live cases) | none | **8 turns → 8 traces** (3 final), 61 spans |
| **Total cost per triage** (all turns, both models; computed by the platform) | **NOT MEASURED** | **median $0.042758**, p95 $0.100079, range $0.033199–$0.150813 (31 cases); live: median $0.047302 (3 cases) |
| Split by model, median per triage | — | `gemini-3.1-pro-preview` $0.023510, `gemini-3.8-flash` $0.017507 |
| Platform judges' own cost per triage | — | median $0.010870 (separate from the above) |
| Accuracy / under-triage / over-triage (31 cases) | 30/31 / 0/31 / 1/31 (X) | 30/31 / **0/31** / 1/31 (X) |
| Latency per triage, median / p95 | 43.74 s / 71.24 s | 45.99 s / 89.58 s |
| Per-agent latency, median (platform spans) | NOT MEASURED | intake 10.65 s · triage (pro) 11.97 s · relevance judge 4.99 s · retrieval 38 ms, or 9.15 s when it rewrites queries (7 of 40) · supervisor 0 ms |
| Safety judge (LLM, on final turns) | — | **31/31 pass**; agrees with the suite's under-triage flag on 31/31, no disagreements (the run had no under-triage) |
| Safety judge, negative control (24 final traces re-sent with the level one step below expected) | — | first rubric: **caught 17/20** (missed A, K, V: no mapping of IMCI's "urgent referral" to EMERGENCY). **Fixed rubric** (maps guideline wording onto the app's levels; the more severe edition wins): **caught 20/20**, and passed **31/31** original cards re-sent unchanged (no false alarms); 4/4 standard → self_care passed (outside the rubric's scope). One run each |
| Grounding judge (actions supported by the cited text) | — | **21/31 pass**. Failures: Y, Z, U (0.0), X (0.33), N, P, V (0.5), AE, F, W (0.67): drugs, doses or referral timing that are not in the excerpts, and the PEN chest-pain and stroke gaps |
| Triage-level format assertion | — | 31/31 |
| Tests (Windows laptop, no Postgres) | 262 passed, 4 skipped, 1 failed | **278 passed, 4 skipped, 0 failed**, including the contract tests against `TRACING_API.md` and the platform's own zod schema; the Round 2 failure (a too-short wait in a tracing test) is fixed |

## 5. What is deployed where
- **Live URL:** https://triage-ai-827630707333.us-central1.run.app
  - Cloud Run service `triage-ai`, project `gen-lang-client-0005933941`,
    region `us-central1`, revision `triage-ai-00006-bgc` (100% of traffic),
    image `us-central1-docker.pkg.dev/gen-lang-client-0005933941/triage/triage-ai:ff3f310`.
  - **Tracing on:** `TRACE_API_URL` points at the eval platform;
    `TRACE_API_KEY` comes from secret `triage-trace-api-key`.
  - Public (`allUsers`); max 2 instances, min 0; session affinity; 1 vCPU,
    2 GiB, CPU boost, 300 s timeout; runtime service account `triage-ai-run`.
  - LLM: Gemini through the OpenAI-compatible endpoint
    (`gemini-3.1-pro-preview`, `gemini-3.8-flash`), key in Secret Manager.
- **Sessions:** Neon Postgres (PostgreSQL 18.7, AWS us-east-1), direct
  connection string in secret `triage-checkpoint-db-url`;
  `BROWSER_STATE_SECRET` in `triage-browser-state-secret`.
- **Also in the project:** `triage-ai-bench` (private, min 0 / max 1),
  used for the cold-start and multi-instance tests; it costs nothing while
  idle. Artifact Registry repo `triage`.
- **Deploy:** `triage-agent/deploy/deploy_cloud_run.sh` (ran end to end) and
  `deploy/CLOUD_RUN.md`. Checks: `deploy/coldstart_from_logs.py`,
  `deploy/multiinstance_check.py`, `deploy/ratelimit_check.py`,
  `deploy/live_cases.py`.
- **Not run against Google Cloud:** the Vertex AI code path (4 offline unit
  tests only).
- **Eval platform:** project `TriageAI` (org `acme`, id
  `cmwkx3qg60000b8mkzgvjofno`) on
  https://eval-web-827630707333.us-central1.run.app. Three evaluators are
  scoped to `triage.final` traces: `triage-safety-judge`,
  `triage-grounding-judge` (LLM judges, `gemini-3.8-flash`, threshold 0.7)
  and `triage-level-valid` (regex assertion). The seed script is
  `triage-agent/deploy/eval-platform/add-triage-evaluators.ts`.
- **Code:** `main` holds Phases 0–7, Round 2 and Round 3, merged from branch
  `claude/who-guidelines-hardening` (one or more commits per phase,
  measurements in `PHASES.md`).

## 6. Remaining limitations
1. **The evaluation is self-written and small** (31 vignettes, one run per
   configuration). It is not a clinical benchmark, and no clinician
   reviewed the labels. Differences of 1–2 cases are within observed
   run-to-run variation.
2. **Over-triage of hypertension.** Grade-2/3 hypertension without emergency
   features is labelled `standard`, and the LLM calls it `urgent` in 2–3 of
   the 4 standard cases depending on the run.
3. **Knowledge-base gaps and licences.** WHO PEN 2020 has no section on
   acute chest pain, stroke or hypertensive crisis. Those cases rely on the
   red-flag rules, and the relevance judge's rejections add retrieval rounds
   (V, Y, Z: 70–93 s per case). Young-infant (under 2 months) chunks are
   tagged `imnci` like the 2–59-month charts, so an under-5 search can
   return them. Both IMCI documents are "all rights reserved": they are not
   in the repo, and the deployed image holds text derived from them for this
   non-commercial prototype. The handbook is from 2005.
4. **Cost is the platform's computation.** Total cost per triage is now
   measured (§4.9). It comes from the price table in the eval platform's
   `src/lib/pricing.ts`, which its docs say was checked against Google's
   pricing page on 2026-11-17; that page was not re-checked here. The flash
   price doubles on 2027-01-01 in that table, and the figures here use
   today's rates.
5. **Claude, the code's default provider, is NOT MEASURED.** Neither is
   Vertex AI.
6. **Deployment limits.**
   - Cold start: 49.40 s on the public URL in Round 2 (instance start →
     startup probe). It was 83.38 s once in Round 3 (84.38 s to the first
     page), measured from the same log lines. Round 2 traced most of the
     startup time to Python imports.
   - Session affinity is required with 2 instances: without it a turn
     hung for the full 300 s timeout.
   - Rate limits live in memory per instance: a client spread over both
     instances can get up to twice the limit, a restart resets them, and
     clients behind one NAT share the IP limit.
   - Neon (AWS us-east-1) is in a different region from Cloud Run
     (us-central1); the per-turn cost of that is NOT MEASURED separately.
   - The Gemini key is Tier 1: 250 `gemini-3.1-pro` requests per day.
     A 31-case run used about 37 and hit the limit after earlier runs.
   - No load test.
   - When the intake LLM call fails, the conversation produces no triage
     card.
7. **Tracing and the platform judges.**
   - **Safety judge limits.** With the fixed rubric it caught 20/20
     deliberately lowered levels, with no false alarm on 31 unchanged
     cards. Caveats:
     - The lowered cards keep actions written for the higher level, which
       makes them easier to catch than a real, self-consistent under-triage.
     - "The more severe edition wins" decides chest indrawing (2005 handbook
       vs 2014 booklet) the same way the suite does.
     - The rubric ignores under-triage that involves no danger sign
       (standard → self_care).
     - One run each, so judge variation is not measured.
     - The control traces stay in the production project, tagged
       `negativeControl` / `positiveControl`.
   - The judges see only the final trace's input (conversation + guideline
     excerpts) and output (card).
   - Traces carry the conversation text to the platform.
   - The judges use the same Gemini key as TriageAI.
   - Production delivery was 8/8 traces with no load test; Cloud Run
     throttles CPU outside requests, and the background sender has not been
     measured under load.
8. **The safety floor depends on extraction.** The rules are
   deterministic, but whether a finding is "present" is still read from the
   conversation by the LLM. The low-confidence loop-back path was fixed
   (HTTP 400 when the request ended with a model turn) but was not
   exercised by real LLM calls in the final run; only tests cover it.
9. **Latency:** 43.74 s median per triage across 2–4 user turns (p95
   71.24 s) on the WHO knowledge base; the completing turn waits on at least
   2 sequential LLM calls (median 24.40 s). Live, a single turn took 12.97 s
   median. Gemini response time is the floor.
10. **Choices that differ from the source text:** the rules treat any
    convulsion as a danger sign (the sample IMNCI text says "more than one
    convulsion"), and young infants under 2 months use child thresholds
    only. Neither is in the eval.
11. **Upgrade blockers:** Gradio 5.50.0 caps pydantic at 2.12.3, which
    forces older `google-genai` and `google-adk` releases, and `app.py`
    emits Gradio 6 deprecation warnings.
12. Still a prototype, **not a medical device**.
