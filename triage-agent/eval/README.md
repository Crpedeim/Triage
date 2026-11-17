# TriageAI evaluation suite

> **This is a self-written set of 31 patient vignettes, not a clinical
> benchmark.** Every expected level was derived by hand from the guideline
> text that ships in this repository (`rag/ingest.py` sample IMNCI and WHO PEN
> chunks), the project's own `RED_FLAG_KEYWORDS`, and the level mapping in
> `state.py` (IMNCI pink → emergency, yellow → urgent, green → self_care).
> No clinician reviewed the labels, and the cases have not been validated
> against real patients. Use the suite to catch regressions and compare
> versions of this system, not to claim clinical accuracy.

## Files
- `cases.json`: the vignettes. Each case has:
  - `id`
  - `expected` (one of `emergency`, `urgent`, `standard`, `self_care`)
  - `basis` (the guideline rule the label comes from)
  - `opening` (the first message the caregiver types)
  - `vignette` (the hidden facts the simulated caregiver answers from)
  - `set`: `baseline6` for the original 6 cases, `extended` for the rest
- `run_eval.py`: the runner.
- `compare.py`: side-by-side of two result files.
- `results/<label>.json`: per-run summary plus per-case detail (transcript,
  every LLM call's model, latency and tokens, node timings, triage card).

## Label distribution (31 cases)
emergency 15 · self_care 7 · urgent 5 · standard 4

Deliberately excluded, because the repository's guideline text cannot label
them unambiguously:
- young infants under 2 months
- malaria-positive fever
- grade-1 hypertension
- pre-diabetes
- a single convulsion (the sample text's general danger sign is "more than one convulsion")

Two labels rely on an inference rather than an explicit rule:
- Q and R ("no dehydration"), inferred from the absence of the listed criteria.
- Y and Z (adult chest pain, stroke signs) come from the project's
  pre-existing `RED_FLAG_KEYWORDS` policy, not from the WHO PEN text.

## How a case runs
1. The real compiled graph (`graph.create_graph()`) is driven through
   `graph.run_turn()`, the same entry point as the Gradio UI.
2. A **simulated caregiver** answers each question the system asks. It is a
   separate LLM call that sees only that case's `vignette` and is told to say
   "I don't know" for anything not in it. It is not part of the system: its
   tokens and time are excluded from every system metric.
3. A LangChain callback records every system LLM call.

## Metrics
| Metric | Definition |
|---|---|
| accuracy | predicted level == expected level |
| **under-triage** (safety metric) | predicted level less severe than expected, **or no result** |
| over-triage | predicted level more severe than expected |
| latency | system wall time per case; median and p95 (nearest rank: for n < 20 the p95 is the maximum) |
| tokens | sum of API `total_tokens` over system calls (Gemini's total includes reasoning tokens beyond input + output) |
| cost | only for models in `PRICES_PER_M`; while any model used is unpriced, **total cost is reported as unknown** |

## Run
```bash
cd triage-agent
export LLM_PROVIDER=openai OPENAI_API_KEY=... \
       OPENAI_BASE_URL=https://generativelanguage.googleapis.com/v1beta/openai/ \
       MODEL_NAME=gemini-3.1-pro-preview FAST_MODEL_NAME=gemini-3.8-flash
python -m eval.run_eval --label my_run --concurrency 4          # all 31
python -m eval.run_eval --label quick --set baseline6           # the original 6
python -m eval.compare eval/results/before.json eval/results/my_run.json
```
LLM calls make results vary from run to run. A single run of 31 cases is a
small sample; treat differences of one or two cases as noise unless they
repeat.
