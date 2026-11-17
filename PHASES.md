# Production-hardening log

One entry per phase: what changed, how it was measured, and the measured
numbers. Every number below comes from a command that was run. Raw per-case
results (transcripts, per-call latency and tokens) are committed under
`triage-agent/eval/results/<label>.json`.

**Measurement setup (all phases unless noted)**
- Eval runner: `triage-agent/eval/run_eval.py`. It drives the real compiled
  graph. A simulated caregiver (a separate `gemini-3.8-flash` call, not part
  of the system) answers intake questions from a hidden vignette; its tokens
  and time are excluded from system numbers.
- LLM: Gemini through the project's `openai` provider and Google's
  OpenAI-compatible endpoint. Main model `gemini-3.1-pro-preview` (triage),
  fast model `gemini-3.8-flash`. Temperature 0.
- Latency: system wall time per case (caregiver excluded). p95 is by nearest
  rank, so with n < 20 it equals the maximum.
- Tokens: sum of the API `total_tokens` over system LLM calls. Gemini's
  total exceeds input + output; the difference is recorded as "reasoning".
- Cost: only `gemini-3.1-pro-preview` is priced ($2 input / $12 output per 1M
  tokens, from search results quoting Google's pricing page; the page itself
  could not be opened). `gemini-3.8-flash` has no published price I could
  find, so **total cost per triage is unknown**. The "priced" figures cover
  the triage model only: low = reasoning tokens not billed, high = billed.
- **The vignettes are self-written** from the project's own guideline text.
  They are a regression and safety suite, **not a clinical benchmark**.
- Each eval runs from a git worktree of that phase's commit, so in-progress
  edits cannot leak into a measurement.

---

## Phase 0: baseline (original code)

**Changed:** added the eval harness (`eval/run_eval.py`, `eval/cases.json`
with the 6 vignettes from the earlier run). No application code changed.

**Run:** `python -m eval.run_eval --label phase0_baseline --driver manual --concurrency 1`
on commit `1bd4fff` code. The `manual` driver resumes the graph with
`update_state` + `invoke(None)`, because `run_turn()` was broken.

| Metric | Phase 0 |
|---|---|
| Accuracy (exact level) | **4/6** |
| Under-triage | **2/6**: A severe pneumonia → standard (expected emergency); B pneumonia → standard (expected urgent) |
| Over-triage | 0/6 |
| Latency per triage, median | **75.81 s** |
| Latency per triage, p95 (nearest rank = max at n=6) | **153.37 s** |
| Latency min / max | 50.04 s / 153.37 s |
| Tokens per triage, median / p95 | **12,875.5** / 22,750 |
| LLM calls per triage, median | 9 |
| Priced cost per triage, median (triage model only) | $0.0139 (reasoning unbilled) – $0.0237 (billed) |
| Total cost per triage | unknown (`gemini-3.8-flash` unpriced) |
| LLM errors | 0 |
| Tests | 151 passed, 1 skipped |

Notes:
- Case D's 153.37 s includes one `gemini-3.8-flash` call that took 71.9 s
  (API-side variance). In the earlier run the same case took 64.75 s.
- An earlier run of the same 6 cases (2026-11-16, recorded in
  `RESUME_FACTS.md`) gave the same 4/6 and the same two under-triaged cases,
  with median latency 66.33 s.

---

## Phase 1: fix what's broken (commit `11aa2e1`)

**Changed:**
1. **`run_turn()` / multi-turn flow** (`graph.py`, `agents/supervisor.py`,
   `agents/intake.py`, `state.py`). Removed `interrupt_before=["intake"]`.
   Each user message is now one graph run on the thread's checkpointed state.
   When intake asks a question it sets `awaiting_user=True` and the
   supervisor routes to END. The next message merges into the saved state and
   starts a new run. No resume step, and no LLM call is ever re-executed.
2. **Feedback loop** (`agents/supervisor.py`, `agents/intake.py`). On a
   low-confidence loop-back the supervisor clears the stale `triage_result`
   and retrieval output, so after the follow-up questions retrieval → triage
   re-run. `route_next` was reordered and the unreachable "case 6" removed.
   Intake now asks queued follow-up questions before it may complete.
3. **Intake JSON leak** (`prompts/intake_prompt.py`, `agents/intake.py`). The
   question call has its own `QUESTION_SYSTEM` prompt instead of reusing
   `EXTRACTION_SYSTEM`. `clean_question()` replaces JSON or code output with
   a plain fallback question.
4. **Gradio** pinned to `5.50.0` (all direct dependencies pinned to the tested
   versions). `app.py` moved to the `messages` history format and dropped the
   removed `bubble_full_width` argument.

**Tests:** 161 passed, 1 skipped (was 151 passed, 1 skipped). 10 new tests in
`tests/test_multiturn.py` run the REAL compiled graph and checkpointer with
a scripted stub LLM:
- a 3-turn `run_turn()` conversation (intake runs every turn; turn 3 completes)
- thread isolation
- the low-confidence loop re-running triage, and terminating at max cycles
- supervisor reset semantics, and routing to END while awaiting the user
- the question prompt not being the extraction prompt
- JSON never being shown, and normal questions being kept
- follow-ups asked before completion

Against the original code the regression tests fail: the 3-turn test, both
loop tests, the supervisor reset test and the prompt tests (run by stashing
the fixes).

**Eval:** `python -m eval.run_eval --label phase1_fixes --driver run_turn --concurrency 1`
(production `run_turn()` path; Phase 0 had to use the manual driver).

| Metric | Phase 0 | Phase 1 |
|---|---|---|
| Driven through `run_turn()` | not possible (0 LLM calls over 3 turns) | **yes, 6/6 reached `done`** |
| Accuracy | 4/6 | 4/6 |
| Under-triage | 2/6 (A, B) | 2/6 (A, B) |
| Over-triage | 0/6 | 0/6 |
| Latency median / p95 | 75.81 s / 153.37 s | **66.07 s / 87.59 s** |
| Tokens per triage, median | 12,875.5 | 11,433.5 |
| LLM calls per triage, median | 9 | 8 |
| Priced cost per triage, median (triage model only) | $0.0139–$0.0237 | $0.0137–$0.0225 |
| JSON shown to the user (intake turns containing `{` or a code fence) | 0 in this run (1 in the earlier run, case E) | 0 |
| Confidence | 1.0 in 6/6 | 1.0 in 6/6 |

Accuracy was not expected to change: Phase 1 fixes plumbing, not clinical
logic. The latency difference is within the run-to-run variance seen so far
(the Phase 0 p95 is one 71.9 s API call), so I don't claim a latency
improvement from Phase 1.

**UI check:** `app.py` on Gradio 5.50.0, driven with `gradio_client` through
its `/on_send` endpoint with real Gemini calls (4-year-old, mild cough).
Turn 1 returned an intake question in 13.3 s. Turn 2 returned the
completion message and a SELF CARE card in 105.5 s.

---

## Phase 3: 31-vignette eval suite (commit `d124426`)

Built **before** the Phase 2 commit was measured, so the safety changes
could be judged on 31 cases rather than 6. The "before" run uses Phase 1
code (`11aa2e1`), the "after" run Phase 2 code (`d124426`).

**Changed:**
- `eval/cases.json`: the 6 original cases plus 25 new ones, 31 in total.
  - Labels: emergency 15, self_care 7, urgent 5, standard 4.
  - Each case has its expected level, the guideline rule it comes from
    (`basis`), the opening message, and the hidden facts the simulated
    caregiver answers from.
  - **Self-written from the repo's guideline text; not a clinical
    benchmark** (see `eval/README.md` for the exclusions and the two
    inferred labels).
- `eval/compare.py` (before/after diff), `eval/README.md`.
- `tests/test_eval_metrics.py` (5 tests): under-triage counting (no result
  counts as under-triage), nearest-rank p95, cost reported as unknown while
  any model is unpriced, and a well-formed cases file.

**Under-triage** = predicted level less severe than expected, or no result.
It is the headline safety metric from here on.

---

## Phase 2: safety layer (commit `423ddbf`)

**Changed:**
- **`safety.py` (new): a fixed danger-sign checklist.** Groups:
  - IMNCI general danger signs (every child under 5)
  - breathing: breath count, chest indrawing, stridor (children with cough or breathing complaints)
  - dehydration signs (children with diarrhoea)
  - stiff neck (children with fever)
  - adult danger signs (age 5 and over)

  Intake must ask every applicable group before it may complete. Each group
  is asked as one fixed question, with no LLM call. Answers are extracted
  into `PatientSummary.danger_sign_findings` as present, absent or unknown.
  An item still unknown after its group was asked is not re-asked; it lowers
  confidence and is listed on the triage card as "not assessed".
- **`RED_FLAG_KEYWORDS` and `RESPIRATORY_RATE_THRESHOLDS` used in real code.**
  Red flags are derived deterministically from the findings and from the
  keywords (with synonyms; negation handled; an explicit "absent" wins).
  Fast breathing uses the age thresholds.
- **Override escalates to EMERGENCY.** Final level = max(LLM level, rule floor).
  - EMERGENCY: any general danger sign, chest indrawing or stridor, severe
    dehydration (two signs), stiff neck, an adult red flag, or a WHO PEN
    hypertensive crisis.
  - URGENT: fast breathing for age, or some dehydration.
  - It never downgrades.
- **Confidence no longer saturates.** The +0.15 red-flag and +0.05 citation
  boosts are gone. Confidence is the top condition's confidence, minus an
  ambiguity penalty, times a checklist-coverage factor (0.6 + 0.4 ×
  coverage), with penalties for no citation and low completeness, capped
  at 0.95.
- The triage prompt shows the checklist findings and the rule-based minimum
  level. Its level definitions now match `state.py`: IMNCI yellow (e.g.
  pneumonia, some dehydration) → urgent, pink → emergency.

**Tests:** 196 passed, 1 skipped at `423ddbf`; 201 passed, 1 skipped at
`d124426` (adds the Phase 3 metric tests). `tests/test_safety.py` has 35
tests: applicability, checklist gating, red-flag synonyms and negation,
every threshold boundary, every floor rule, never downgrading, confidence,
and a real-graph run reproducing Phase 0 case A, which now ends EMERGENCY.

Three existing tests were updated because they encoded the replaced
behaviour: chest indrawing → URGENT, the +0.15 red-flag confidence boost,
and 0.4 confidence for an empty differential (now 0.3 with nothing cited).

**Eval:** 31 cases, `--driver run_turn --concurrency 4`.

| Metric | Before (Phase 1 code) | After (Phase 2 code) |
|---|---|---|
| Accuracy | 18/31 | **30/31** |
| **Under-triage** | **10/31** (A, B, G, I, K, P, T, AB, AC, AD) | **0/31** |
| Over-triage | 3/31 (F, W, X: hypertension called urgent) | 1/31 (X, grade-3 BP without symptoms, called urgent by the LLM) |
| Safety override fired | n/a (old override) | 0/31: every fix came from the LLM, given the checklist answers and the rule floor in its prompt |
| Confidence values | 1.0 in 23/31 | never 1.0; range 0.80–0.95; 18/31 at the 0.95 cap |
| Intake questions per case | 1: 6, 2: 19, 3: 6 | 1: 7, 2: 21, 3: 3 |
| Latency median / p95, all 31 | 63.65 s / 81.79 s | 53.95 s / 117.96 s |
| Latency median / p95, cases 5–31 | 64.74 s / 81.79 s | 51.28 s / 68.03 s |
| Tokens per triage, median | 12,534 | 12,198 |
| LLM calls per triage, median | 9 | 7 |
| Priced cost per triage, median (triage model only) | $0.0131–$0.0241 | $0.0127–$0.0217 |
| LLM errors | 0 | 0 |

The Phase 2 p95 (117.96 s) is a **cold-start artifact**. The run started
right after a container restart, and the first 4 parallel cases (A–D)
spent 31–72 s each in the retrieval node outside the LLM call. That is the
embedding model and torch loading from a cold disk in 4 threads at once:
the lazy loader had no lock. Without those 4 cases the p95 is 68.03 s.
This is a real production cold-start bug, fixed in Phase 5.

Fewer LLM calls per case (9 → 7) is expected: checklist questions are
fixed text, which replaces the LLM question call on those turns.

---

## Phase 4: session state and tracing (commit `9669308`)

**Changed:**
- **`checkpointing.make_checkpointer()`.** With `CHECKPOINT_DB_URL` set it
  returns LangGraph's `PostgresSaver` (`langgraph-checkpoint-postgres==3.2.1`)
  on a psycopg connection pool; `setup()` creates or migrates the tables.
  Otherwise it returns the in-memory saver. Both use a serializer with an
  explicit allowlist of the project's state types: without it LangGraph logs
  "Deserializing unregistered type ... will be blocked in a future version"
  for `PatientSummary`, `TriageResult` and friends (verified), which with a
  persistent store would leave saved sessions unreadable after an upgrade.
- **`tracing.py`.** One trace per `run_turn()`, `POST {TRACE_API_URL}/api/v1/traces`.
  - The `spans` array has one span per agent node and a child span per LLM
    call, each with latency, tokens (input, output, reasoning, total) and
    cost. Cost is `null` when a model has no known price: a partial sum is
    never reported as the total.
  - Key from `TRACE_API_KEY`, sent as Bearer (or in `TRACE_API_KEY_HEADER`).
  - A background sender with a bounded queue makes 3 attempts on
    5xx/429/network errors and none on other 4xx. It never raises into, or
    waits on, a triage.
  - **The payload field names are an assumed shape**: the platform's schema
    wasn't available, so they're documented in the module docstring and need
    checking against the platform before you connect it.
- `pricing.py`: one price table shared by the eval and tracing.

**Tests:** at `9669308`, 213 passed, 1 skipped with `TEST_POSTGRES_URL` set
(local PostgreSQL 16.x server), and 211 passed, 3 skipped without it (the 2
Postgres tests skip).
- `tests/test_postgres_checkpoint.py`: each turn of a 3-turn session runs in
  a **separate OS process** against Postgres (state restored every turn,
  exactly one extraction per process), thread isolation checked in the
  `checkpoints` table, and the serializer round-trip with no
  "unregistered" warning.
- `tests/test_tracing.py` (8 tests) runs against a local mock platform:
  - path, Bearer key, custom key header
  - one trace per turn, agent and LLM spans, per-agent rollups, cost null when unpriced
  - 5xx retried 3 times and 4xx not retried
  - platform down never breaking a triage
  - LLM errors recorded as error spans

**End-to-end check with real Gemini calls** (`eval/results/phase4_e2e_traces.json`):
one case (B, pneumonia) driven through `run_turn()`. **Every turn ran in a
new OS process** with `CHECKPOINT_DB_URL` pointing at Postgres, and tracing
went to a local mock platform.

| Turn | Process | Turn time | Human messages restored from Postgres | Trace delivered |
|---|---|---|---|---|
| 1 | new | 8.13 s | 1 | yes |
| 2 | new | 9.98 s | 2 | yes |
| 3 (completes) | new | 48.59 s | 3 | yes |

Final level `urgent` (expected `urgent`). 3/3 traces received at
`/api/v1/traces` with the right key, 0 failed.

In the completing turn's trace, the agent spans were:
- intake 16,172.6 ms (2 LLM calls)
- retrieval 17,376.0 ms (2 calls: query generation and the judge)
- triage 14,955.5 ms (1 call, 4,666 tokens, $0.031092 with reasoning billed)

The trace total cost was `null`, because `gemini-3.8-flash` is unpriced.

---

## Phase 5: latency (commits `cbcee90`, `c16837b`, `d466cad`)

### Profile: where the time went
From the recorded per-node and per-call timings (`python -m eval.profile`):

| | Phase 1 code (31 cases) | Phase 2 code (31 cases) |
|---|---|---|
| LLM calls as % of system wall time | 99.5% | 88.2% (the remainder is the cold-start bug below) |
| intake node | 58.6% (2 sequential flash calls per turn, ~5.8 s each) | 47.4% |
| retrieval node (query generation → search → judge, sequential) | 21.2% | 33.9% |
| triage node (1 pro call, median ~11–12 s) | 20.1% | 18.6% |
| median reasoning tokens per flash call | 470 (vs 64 visible output) | 462–559 |

The completing turn made **5 sequential LLM calls**: extraction, completion
message, query generation, relevance judge, triage. Embeddings plus vector
search took 46–67 ms per case (Phase 0 measurement), which is negligible.

### What changed
1. **Fewer round trips.** The intake extraction call also emits
   `PatientSummary.search_queries`; retrieval attempt 1 uses them, so the
   separate query-generation call is gone. The completion message is fixed
   text (it was an LLM call writing one courtesy sentence).
2. **Parallel independent calls.** `relevance_check` (the LLM judge) is its
   own node and runs **in parallel with triage** after retrieval (a LangGraph
   fan-out). If the judge rejects the guidelines, the supervisor discards
   that speculative triage and re-runs retrieval with reformulated queries.
   The self-correction loop is preserved at graph level. A timed test
   (0.4 s + 0.4 s stubs finish in < 0.75 s) proves the overlap.
3. **Cold start.** The embedding model load is locked: parallel first
   requests had each loaded their own copy, 31–72 s per case for the first
   4 cases after a restart. The app also warms the model before serving.
4. **Tail fixes found by the Phase 5 eval (5b, 5c):**
   - Stop retrying retrieval when a reformulated search returns no chunk the
     judge hasn't already seen. The adult chest-pain and stroke cases
     retried 3× against the same 2 chunks.
   - Fixed an HTTP 400 on **every** low-confidence loop-back. The extraction
     request ended with the AI's own message, and Gemini rejects requests
     ending with a model turn. The cause was found by recording API error
     bodies: 4/4 errored cases were loop-back cases and every loop-back hit
     it. The stub LLM now mimics that rejection, so the tests catch it.
   - The extraction is retried once before falling back.
5. **Model choice, measured:** flash for triage (variant B) vs pro (variant A).
6. **Not adopted:** `reasoning_effort` on `gemini-3.8-flash`. `"minimal"`
   returns HTTP 400. `"none"` and `"low"` cut reasoning to 0 on some calls
   and not on others (n=3 each), so latency differences were within noise.

### Results (31 cases, `--driver run_turn --concurrency 4`)

| Metric | Before: Phase 2 | A: Phase 5, pro triage (`cbcee90`) | B: Phase 5, flash triage (`cbcee90`) | **Final: 5b, pro triage (`c16837b`)** |
|---|---|---|---|---|
| **Latency median, all 31** | 53.95 s | 35.98 s | 39.93 s | **36.95 s** |
| **Latency p95, all 31** | 117.96 s | 85.84 s | 110.99 s | **52.56 s** |
| Latency median / p95, cases 5–31 (no cold start) | 51.28 / 68.03 s | 35.44 / 85.84 s | 39.18 / 110.99 s | **36.95 / 49.15 s** |
| Completing turn median / p95 | not recorded | 20.67 / 63.03 s | 24.10 / 76.29 s | **22.03 / 41.57 s** |
| Triage LLM call, median | 10.94 s | 10.88 s | **14.21 s** (flash) | 11.59 s |
| LLM calls per triage, median | 7 | 5 | 5 | 5 |
| Tokens per triage, median | 12,198 | 10,869 | 11,178 | 10,891 |
| Priced cost per triage, median (triage model only) | $0.0127–$0.0217 | $0.0133–$0.0219 | unknown (no priced model used) | $0.0133–$0.0216 |
| **Accuracy** | 30/31 | 29/31 | 28/31 | **28/31** |
| **Under-triage** | 0/31 | 0/31 | 0/31 | **0/31** |
| Over-triage | 1/31 (X) | 2/31 (F, X) | 3/31 (F, W, X) | 3/31 (F, W, X) |
| Triage calls (31 cases) | 31 | 37 (6 discarded after the judge rejected) | 36 | 34 |
| LLM errors | 0 | 2 | 1 | 1 (all loop-back 400s; fixed in 5c) |

**Decision:** keep `gemini-3.1-pro-preview` for triage. The flash model was
**slower** on the triage call (14.21 s vs 10.88 s median) and less accurate
(one more over-triage), so "fast model where accuracy holds" did not hold.

**Accuracy caveat:** 30/31 → 28/31 comes entirely from over-triaging
grade-2 hypertension (F, W: standard → urgent). Both were also called
urgent in the Phase 1 31-case run, and they have flipped between standard
and urgent across runs. One run per variant can't separate a Phase 5
effect from run-to-run LLM variation, so I don't claim either way.
Under-triage, the safety metric, stayed at 0/31 in every Phase 5 run.

The 5c loop-back fix landed after the final Phase 5 run. It is measured in
the final full run (Phase 6).

**Tests at `d466cad`** (clean worktree): 221 passed, 3 skipped. With
`TEST_POSTGRES_URL` set, 223 passed, 1 skipped (the original API-key test).

---

## Phase 6: container, Vertex AI and Cloud Run (commit `cadbfe1`)

**Changed:**
- **Vertex AI provider** (`LLM_PROVIDER=vertexai`, `llm.py`):
  `ChatGoogleGenerativeAI(vertexai=True, project=GOOGLE_CLOUD_PROJECT, location=GOOGLE_CLOUD_LOCATION or "global")`,
  authenticated by Application Default Credentials (the Cloud Run service
  account). Pinned to `langchain-google-genai==4.3.0` and
  `google-genai==2.8.0`, because every `google-genai` release from 2.9
  needs pydantic ≥ 2.12.5 and Gradio 5.50.0 needs ≤ 2.12.3.
- **`Dockerfile`:**
  - `python:3.13-slim`, CPU-only torch 2.15.0
  - embedding model and sample knowledge base baked in; Hugging Face
    offline at runtime
  - runs as uid 10001
  - an optional build-only CA secret that never lands in a layer
- **`deploy/deploy_cloud_run.sh` + `deploy/CLOUD_RUN.md`.** The script is
  idempotent. It enables the APIs, creates the Artifact Registry repo and a
  runtime service account (`roles/aiplatform.user`), builds with Cloud
  Build, and runs `gcloud run deploy`. Optional: Cloud SQL Postgres,
  tracing, Secret Manager secrets.
- **UI sessions survive restarts.** The session id moved from `gr.State`
  (server memory, lost on restart even with Postgres) to `gr.BrowserState`,
  encrypted with a stable `BROWSER_STATE_SECRET`, and the chat is rebuilt
  from the checkpoint on page load.
- Requirements split into runtime (`requirements.txt`) and
  `requirements-dev.txt`.

**Deployment status: NOT deployed.** The build environment had no Google
Cloud credentials, so there is **no live URL**. Neither the deploy script
nor the Vertex AI code path has been run against Google Cloud. The Vertex
path is covered by 4 offline unit tests only.

**Measured locally** (Docker 29.9.2 in the build environment):

| Check | Result |
|---|---|
| Clean build / rebuild after an `app.py` change | 283 s / 27 s |
| Image size | 3,642,898,126 bytes (3.64 GB); site-packages 2.3 GB |
| torch in the image | `2.15.0+cpu`, CUDA `None`, 0 NVIDIA packages |
| Secrets or proxy settings in the image or its history | none found |
| Start → HTTP 200 (model loaded before serving) | 8.3 s |
| Memory: idle / after one complete triage | 464.3 MiB / 497.2 MiB |
| Case A through the container's Gradio API (real Gemini, `openai` provider) | 3 turns: 12.6 s, 12.0 s, 25.1 s → **EMERGENCY** (Phase 0: STANDARD) |
| Browser restart test (Playwright and Chromium, Postgres-backed) | after replacing the container (9.0 s) and reloading, 1/1 user and 1/1 bot messages restored; the next answer advanced the interview |
| Same restart test **without** `BROWSER_STATE_SECRET` | 0 messages restored; new interview (Gradio's per-launch random key) |

**Tests:** 228 passed, 3 skipped. With `TEST_POSTGRES_URL` set: 230 passed,
1 skipped. New: `tests/test_llm_providers.py` (4), `tests/test_app_restore.py` (3).

**Final full eval on this commit** (31 cases, the same settings as every
31-case run; this run includes the Phase 5c loop-back fix):

| Metric | Phase 1 code | Phase 2 code | Phase 5b | **Final (`cadbfe1`)** |
|---|---|---|---|---|
| Accuracy | 18/31 | 30/31 | 28/31 | **29/31** |
| **Under-triage** | **10/31** | 0/31 | 0/31 | **0/31** |
| Over-triage | 3/31 | 1/31 | 3/31 | 2/31 (F, X) |
| Latency median / p95 | 63.65 / 81.79 s | 53.95 / 117.96 s | 36.95 / 52.56 s | **39.92 / 49.08 s** |
| Completing turn median / p95 | not recorded | not recorded | 22.03 / 41.57 s | 22.77 / 38.29 s |
| Median latency per user turn (all 90 turns) | not recorded | not recorded | not computed | 9.84 s |
| Tokens per triage, median | 12,534 | 12,198 | 10,891 | 11,189 |
| LLM calls per triage, median | 9 | 7 | 5 | 5 |
| Priced cost per triage, median (triage model only) | $0.0131–$0.0241 | $0.0127–$0.0217 | $0.0133–$0.0216 | $0.0135–$0.0219 |
| LLM errors | 0 | 0 | 1 | **0** |
| Confidence range | 1.0 in 23/31 | 0.80–0.95 | 0.75–0.95 | 0.782–0.95 (19/31 at the cap) |

There were no loop-backs in the final run, so the 5c fix was not exercised
by real LLM calls here; it is covered by its tests. Between 5b and final the
latency-relevant code is identical apart from 5c, and the median moved
36.95 s → 39.92 s. About 3 s of run-to-run variation is the noise floor
for these comparisons.

---

## Phase 7 (optional): ADK ReAct retrieval agent (commit `b0dbc9e`)

**Changed:** `agents/adk_retrieval.py`, enabled with `RETRIEVAL_BACKEND=adk`.
- A Google ADK (`google-adk==2.0.0`) `LlmAgent` with one tool,
  `search_guidelines(query)`. It searches, reads the results, refines and
  searches again (at most 4 searches), then returns the ids of the chunks
  relevant for classifying the case.
- It plugs into the existing retrieval node, so the rest of the graph
  (parallel judge and triage, supervisor, safety floor) is identical for
  both backends.
- **Model adapter.** ADK's built-in Gemini client uses the native API
  (`x-goog-api-key`), which rejected the Bearer key used here (HTTP 401).
  ADK's LiteLLM route pins `openai==2.24.0` and `pydantic==2.12.5`, which
  conflict with `langchain-openai` (`openai>=2.45`) and Gradio 5.50.0
  (`pydantic<=2.12.3`). So `OpenAICompatLlm` implements ADK's `BaseLlm`
  over the openai SDK and the Gemini OpenAI-compatible endpoint.
  - It round-trips **Gemini 3 thought signatures**: without them the second
    tool-use turn failed with HTTP 400 ("Function call is missing a
    thought_signature").
  - It fires LangChain callbacks, so the eval and tracing record its calls.
- **Dependencies:** `requirements-adk.txt` installs into a separate venv.
  ADK 2.0.0 needs `google-genai<2` and `opentelemetry<=1.41.1`, so
  ChromaDB's OTLP exporter has to be pinned to 1.41.1 there too.
- Tests: `tests/test_adk_retrieval.py` has 4 offline adapter tests
  (messages, tool calls, thought-signature round trip, tool schema, id
  parsing). They pass in the ADK venv and skip without `google-adk`.

**Eval** (31 cases, same settings; ADK venv, `RETRIEVAL_BACKEND=adk`)
compared with the final default backend:

| Metric | Default LangGraph retrieval (`suite31_final`) | ADK ReAct retrieval (`suite31_adk`) |
|---|---|---|
| Accuracy | 29/31 | 28/31 |
| **Under-triage** | **0/31** | **0/31** |
| Over-triage | 2/31 (F, X) | 3/31 (F, W, X) |
| Latency median / p95 | **39.92 / 49.08 s** | 53.24 / 70.31 s |
| Completing turn median / p95 | **22.77 / 38.29 s** | 37.02 / 58.05 s |
| Retrieval node, median | 0.06 s (attempt 1 uses the extraction's queries; no LLM call) | 13.51 s (145 agent model calls over 33 retrieval runs) |
| Tokens per triage, median | **11,189** | 20,003 |
| LLM calls per triage, median | **5** | 9 |
| Guideline sections handed to triage, median | 5 | 2 |
| Priced cost per triage, median (triage model only) | $0.0135–$0.0219 | $0.0126–$0.0213 |
| LLM errors | 0 | 0 |

**Verdict:** on this suite the ADK agent did not improve accuracy (one case
lower; W is a hypertension case that has flipped across runs before) and
under-triage stayed at 0/31. It added 13.3 s to the median latency and 79%
more tokens. **The default LangGraph retrieval stays the default; ADK is an
optional backend.** The triage model's priced cost was slightly lower with
ADK, because it gets fewer chunks (2 vs 5), but the extra flash-model calls
are unpriced, so the total cost comparison is unknown. The knowledge base
has only 8 chunks, which leaves a search-and-refine agent little to find. A
real guideline corpus might change the trade-off; that is untested.

---

# Round 2: real guidelines, persistence, public deployment

Same rules as above: every number comes from a command that was run, and
anything not measured is marked NOT MEASURED. Work branch:
`claude/who-guidelines-hardening`, cut from `main` at `e6230ed`.

## R2 Phase 0: save the work

**Done:**
- `main` was fast-forwarded from `1c58788` (README only) to `e6230ed` (the
  end of Phases 0–7 above, branch `claude/adoring-goldberg-d2ncfb`) and
  pushed to `origin/main`. No merge commit was needed.
- A separate quick-deploy branch, `claude/google-cloud-deploy-c09829`
  (commit `13ceea9`), was committed and pushed but **not merged**. It
  changed 9 of the same files as Phases 0–7 and is superseded by them. It
  is what currently runs on Cloud Run (see below).
- A secret scan of both branches (Google API keys, Anthropic/OpenAI keys,
  Postgres URLs with passwords, `.env`/key files) found nothing.

**Deployment state at the start of Round 2** (corrects "Nothing is
deployed" in `RESUME_FACTS.md`, which was written before this):
- Cloud Run service `triage-ai`, project `gen-lang-client-0005933941`,
  region `us-central1`, public URL
  `https://triage-ai-827630707333.us-central1.run.app`, revision
  `triage-ai-00003-kgl`.
- It runs the quick-deploy branch, not `main`: the native Gemini provider
  (`gemini-flash-latest`), the 8 sample chunks, in-memory sessions and
  `max-instances 1`. The Gemini key is read from Secret Manager
  (`gemini-api-key`).
- One manual browser check of case A on that revision returned EMERGENCY.
  No other measurements were taken on it.

## R2 Phase 1: knowledge base from the full WHO guidelines (commits `91557e9`, `f15ca9c`)

### Sources (downloaded from WHO IRIS on 2026-11-17)
| Document | IRIS record | PDF | Pages | Licence (read from the PDF's copyright page) |
|---|---|---|---|---|
| IMCI chart booklet, WHO 2014 (ISBN 9789241506823) | `iris.who.int/handle/10665/104772` | 5,304,057 bytes, sha256 `7018498d…` | 80 | © WHO 2014, all rights reserved |
| IMCI handbook, WHO 2005 (ISBN 9241546441) | `iris.who.int/handle/10665/42939` | 1,299,724 bytes, sha256 `d0f9cf6a…` | 173 | © WHO 2005, all rights reserved |
| WHO PEN, 2020 (ISBN 9789240009226) | `iris.who.int/handle/10665/334186` | 2,584,724 bytes, sha256 `a09cc0ea…` | 85 | CC BY-NC-SA 3.0 IGO (also in the IRIS record) |

Full URLs and hashes: `triage-agent/rag/guidelines/sources.json`. "IMNCI" is
India's adaptation; WHO publishes IMCI, so the WHO IMCI originals are used.
Because both IMCI documents are "all rights reserved" and the GitHub repo is
public, **no PDF and no extracted text is committed**. The PDFs are
sha256-checked, and the multi-stage Dockerfile keeps them out of every layer
of the final image. Attribution: README "Guideline sources and licences" and
an always-visible footer in the app, built from `sources.json`.

### Extraction (`rag/pdf_extract.py`, `rag/build_kb.py`)
- **Glyph-id fonts.** Several fonts in the chart booklet and WHO PEN have no
  Unicode map, so PyMuPDF returns glyph ids: glyph = ASCII - 29, space = 0x03
  (`6,&.<281*` = "SICK YOUNG"). Repaired by decoding, plus 16 ligature and
  symbol glyphs audited in the documents (`ȴ`=fi, `Ȳ`=≥, `Ɖ`=µ, …). A
  font-based rule for single shifted words without 0x03 decodes 18 spans in
  the chart booklet, all correctly. Its first version also wrongly decoded 223
  spans in WHO PEN (digits, ISBNs, URLs); the tightened rule decodes 0 there.
- **Classification tables.** `find_tables()` found the IMCI grids but split
  and merged rows and dropped the spaces between words
  ("Givefirstdoseofanappropriateantibiotic"). Every classification row is a
  pink/yellow/green filled cell triple, so rows are taken from those fills and
  each cell is read by clipping. Each row becomes "Signs: … -> Classification:
  … (Pink/Yellow/Green: meaning) -> Treatment: …" under its heading (e.g.
  `chart booklet > THEN ASK ABOUT MAIN SYMPTOMS > Does the child have cough
  or difficult breathing?`). 48 classification rows were extracted.
- **Other tables** (dose tables, care-response tables): `find_tables()` grid,
  each cell read by clipping, each row "header: value; …". Tables where
  find_tables returns one row per text line (a sparse key column such as
  CLASSIFY AS) are read as prose instead. Merging those rows was tried first
  and mixed neighbouring conditions.
- Every chunk starts with its section heading.

### Quality check (20 random chunks per document, seed 7)
| Pass | Auto-flagged | Found by reading all 60 | Action |
|---|---|---|---|
| 1st | 1/60 (PEN p18, number-heavy) | 5 more: 4 table rows cut mid-word (chart p61, p63; handbook p97; PEN p49), 1 heading in a second glyph encoding (PEN p30) | added the `fragmented` check (words that are not on the page), heading clean-up, prose fallback for line-split tables, URL stripping |
| final | 0/60 | 0 garbled. Cosmetic only: a figure label ("Fold") in one handbook heading; handbook chapters include their embedded case-recording-form text | none |

A page is rejected when more than 30% of its chunks are garbled (fragmented
table rows are dropped one by one and do not reject a page); on accepted
pages, flagged chunks are dropped individually.

| Document | Pages | Used | Replaced by curated excerpt | Skipped (reason: pages) |
|---|---|---|---|---|
| IMCI chart booklet | 80 | 58 | 0 | 22 (front matter 4, recording forms 8, growth charts 6, garbled 2: p28 dose table in a non-linear glyph font, p78 scanned mother's card; no text 2) |
| IMCI handbook | 173 | 149 | 0 | 24 (front matter 10, recording forms 4, mother's card 2, no text 8) |
| WHO PEN | 85 | 41 | 0 | 44 (front matter 7, programme introduction 5, CVD risk charts 1, implementation/annexes 18, no text 13) |
| **Total** | **338** | **248** | **0** | **90** |

No covering page was rejected, so **none of the 8 curated excerpts is kept**.
(In an earlier pass PEN p18 was rejected and the curated hypertension excerpt
replaced it; once URL-only text stopped counting as garbled, its text merged
into p17's chunk and the page counts as used.) Chunks dropped on used pages:
4 (chart), 0 (handbook), 1 (PEN).

### Chunk size (retuned)
The embedding model reads at most 256 tokens (all-MiniLM-L6-v2
`max_seq_length`); text beyond that is ignored. Measured with its tokenizer on
the final extractor:

| size/overlap (chars) | chunks | tokens median/p95/max | over 256 tokens | classification rows split | table rows split |
|---|---|---|---|---|---|
| 500/50 (old default) | 1364 | 110/142/259 | 1 (0.1%) | 8/48 | 15/124 |
| 700/100 | 1053 | 144/189/308 | 4 (0.4%) | 2/48 | 9/124 |
| **900/120 (chosen)** | **874** | **170/232/379** | **8 (0.9%)** | **1/48** | **4/124** |
| 1100/150 | 765 | 183/282/556 | 147 (19.2%) | 1/48 | 2/124 |

### Knowledge base
- **874 chunks** (`imnci` 788, `who_pen` 86) from 3 documents and 248 pages.
  Chunk characters: median 757, p95 964, max 1041. Vector store 8.5 MB.
- **Age filter** unchanged and working on the new metadata (`guideline_type`
  comes from the manifest): `PatientSummary(age=3)` -> `imnci`, `age=40` ->
  `who_pen`, `age=10` or unknown -> no filter. Filtered searches returned only
  chunks of that type. Case A's query: top hits were chart booklet p6 (incl.
  the classification row) and handbook 7.1-7.2. Noted: young-infant
  (under 2 months) chunks are also `imnci`, so an under-5 query can retrieve
  them (a diarrhoea query returned young-infant jaundice rows).
- Built at **image build time** in the Dockerfile's `kb` stage: 71.8 s in
  Cloud Build (log 10.3 s -> 82.1 s), with the same report as the local build.
  Locally (this laptop): extraction + checks 31.05 s, embed + store 14.1 s.

### Build problems found and fixed
- Cloud Build's default path cannot build this Dockerfile ("the --mount option
  requires BuildKit"); `deploy/cloudbuild.yaml` sets `DOCKER_BUILDKIT=1`. The
  existing `deploy_cloud_run.sh` (`gcloud builds submit --tag`) has the same
  problem; it had never been run against Google Cloud.
- iris.who.int returns **HTTP 403 to Cloud Build** (from a workstation the
  same URL returns 200/302 for every user agent tried). The deploy uploads the
  locally fetched, hash-checked PDFs in the private build context;
  `fetch_guidelines` downloads only when a file is missing.

### Cold start, before vs after (Cloud Run, measured)
Service `triage-ai-bench` (private), us-central1, 2 GiB, 1 vCPU, CPU boost,
min 0 / max 1. Each deploy starts a new instance. Cold start = Cloud Run's
"Starting new instance" log -> "STARTUP TCP probe succeeded" log
(`deploy/coldstart_from_logs.py`).

| Image | Knowledge base | Samples (s) | Median |
|---|---|---|---|
| before: `main` `e6230ed` | 8 sample chunks, built at image build | 47.68, 52.17, 35.50 | **47.68 s** |
| after: `f15ca9c` | 874 WHO chunks, built at image build | 45.36, 36.28, 50.31 | **45.36 s** |

**No measurable difference:** both sets span about 35-52 s. One startup per
image, broken down by the app's log timestamps:

| Step | before (`bench-00003`) | after (`bench-00006`) |
|---|---|---|
| instance start -> first app log line | 10.40 s | 14.85 s |
| graph ready -> embedding model starts loading (`torch` / `sentence-transformers` import) | 22.63 s | 31.42 s |
| model weights loaded | 0.83 s | 1.30 s |
| open the vector store and count chunks | 0.47 s | 1.09 s |

Startup is dominated by Python imports, not the knowledge base, and neither
image ingests at startup. The quick-deploy image from the start of Round 2
started in 15.06-19.65 s with different dependencies (Gradio 6, unpinned);
that difference is NOT investigated here.

### Tests
`tests/test_pdf_extract.py`: 25 offline tests (glyph repair on strings from
the real PDFs, including the PEN strings that must not be decoded; quality
flags; line-split tables; a synthetic chart page with coloured cells; heading
clean-up; manifest integrity). Suite on this machine (Python 3.14.0, Windows):
252 passed, 4 skipped, 1 failed. The failure,
`test_tracing.py::test_platform_down_never_breaks_or_blocks_triage`, also
fails on untouched `main` (`e6230ed`) on this machine.
