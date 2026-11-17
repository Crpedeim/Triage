# Deploying TriageAI to Google Cloud Run

> **Status:** deployed with `deploy/deploy_cloud_run.sh` (Round 2). The live
> URL, cold start and per-turn latency are recorded in `PHASES.md`
> (R2 Phase 5) and `RESUME_FACTS.md`.

## What gets deployed
- **Image** (`triage-agent/Dockerfile`, multi-stage):
  - `python:3.13-slim`, CPU-only torch 2.15.0, pinned `requirements.txt`
  - the `all-MiniLM-L6-v2` embedding model and the **WHO guideline knowledge
    base** (874 chunks from the IMCI chart booklet 2014, IMCI handbook 2005
    and WHO PEN 2020) built at image build time in the `kb` stage; the
    service never contacts Hugging Face or WHO and never ingests at startup
  - the WHO PDFs are in no layer of the final image
  - runs as a non-root user (uid 10001), serves Gradio on `$PORT` (8080)
- **LLM** (`LLM_MODE`):
  - `gemini-api` (default): Gemini through Google's OpenAI-compatible
    endpoint with the API key in secret `gemini-api-key`;
    `gemini-3.1-pro-preview` (triage) and `gemini-3.8-flash` (intake,
    retrieval). Every eval in `PHASES.md` used this path.
  - `vertexai`: Gemini on Vertex AI with the service account. Covered by
    offline unit tests only; no Vertex call has been made.
- **Sessions**: LangGraph `PostgresSaver` on the database in secret
  `triage-checkpoint-db-url` (Neon: use the **direct**, non-pooled
  connection string). The browser keeps the session id in `gr.BrowserState`,
  encrypted with `BROWSER_STATE_SECRET` (secret
  `triage-browser-state-secret`, 64 hex characters, must be the same on every
  instance and across restarts). Without the database secret the script
  keeps sessions in memory and pins max 1 instance.
- **Rate limits** (`ratelimit.py`): 20 messages / 10 min per conversation,
  60 / hour per client IP, 2000 characters per message (env
  `RATE_LIMIT_SESSION`, `RATE_LIMIT_IP`, `MAX_MESSAGE_CHARS`).

## Prerequisites
- `gcloud` authenticated with Owner (or equivalent) on a project with billing
- Secrets: `gemini-api-key`; for persistent sessions `triage-checkpoint-db-url`
  (the script creates `triage-browser-state-secret` if missing)
- A local Python (only the standard library is used) to fetch the WHO PDFs:
  iris.who.int answers Cloud Build with HTTP 403, so the PDFs are fetched
  locally (sha256-checked) and sent in the private build upload
- Run from the `triage-agent/` directory

## One-command deploy
```bash
cd triage-agent
PROJECT_ID=my-project REGION=us-central1 ./deploy/deploy_cloud_run.sh
```
Idempotent. It enables the APIs, creates the Artifact Registry repo `triage`
and the runtime service account `triage-ai-run`, grants it the secrets it
uses, fetches the PDFs, builds with `deploy/cloudbuild.yaml` (BuildKit is
required by the Dockerfile) and deploys.

## Service settings the script uses, and why
| Flag | Value | Reason |
|---|---|---|
| `--max-instances` | 2 (1 without a database) | caps cost on a public URL; sessions are in Postgres, so 2 instances share them |
| `--session-affinity` | on | **required** with more than one instance: Gradio's job queue is in-process, and a turn whose result stream reached the other instance hung until the 300 s timeout (R2 Phase 3) |
| `--min-instances` | 0 | scale to zero; a cold start costs about 35–52 s (R2 Phase 1) |
| `--cpu-boost` | on | startup is dominated by Python imports |
| `--timeout` | 300 s | the slowest whole case in the R2 eval took 92.7 s over 2 turns |
| `--memory` / `--cpu` | 2Gi / 1 | measured 497.2 MiB after one triage locally (Phase 6) |
| `--allow-unauthenticated` | on by default | public demo, protected by the rate limits; set `ALLOW_UNAUTHENTICATED=false` for a private service |

## Checking a deployment
- `deploy/coldstart_from_logs.py`: instance cold starts from Cloud Run logs
- `deploy/multiinstance_check.py`: one conversation across instances
- `deploy/ratelimit_check.py`: rate limits on a revision deployed with small limits

## Phase 6: verified locally (the earlier build environment, 2026-11-17)
Built with `docker build` (BuildKit) from `triage-agent/`. The sandbox's
TLS proxy CA was passed as the optional `ca` build secret.

| Check | Measured |
|---|---|
| Clean build (empty cache) | 283 s |
| Rebuild after an `app.py` change | 27 s |
| Image size (`docker image inspect` Size) | 3,642,898,126 bytes (3.64 GB); site-packages 2.3 GB |
| torch inside the image | `2.15.0+cpu`, CUDA `None`, 0 NVIDIA packages |
| Runs as | uid 10001 |
| Build secret / proxy settings left in the image or its history | none found |
| Container start → HTTP 200 (embedding model already loaded) | 8.3 s on a warm local disk (Cloud Run cold start not measured) |
| Memory, idle after start | 464.3 MiB |
| Memory after one complete triage | 497.2 MiB |

**End to end through the container's Gradio API.** Case A (3-year-old,
breathing 50/min, chest indrawing) was run with real Gemini calls. The
local run used the `openai` provider against Google's OpenAI-compatible
endpoint, because Vertex AI was not reachable from the build environment.
It took 3 turns (12.6 s, 12.0 s, 25.1 s) and produced an **EMERGENCY**
card citing "ASSESS AND CLASSIFY COUGH OR DIFFICULT BREATHING", "FAST
BREATHING THRESHOLDS" and "CHEST INDRAWING DEFINITION". The same case was
classified STANDARD by the original code (Phase 0).

**Restart in a real browser** (Playwright and Chromium, Postgres-backed
container):
1. Sent the first message.
2. Replaced the container (`docker rm -f` and a fresh `docker run`, 9.0 s).
3. Reloaded the page: 1 user and 1 bot message were restored (identical
   text), and the next answer advanced the interview to the breathing
   checklist question.

Without `BROWSER_STATE_SECRET` the same test restored 0 messages and the
bot started a new interview.
