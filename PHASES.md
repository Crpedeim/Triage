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
