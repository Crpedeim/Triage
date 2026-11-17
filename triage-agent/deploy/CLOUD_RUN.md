# Deploying TriageAI to Google Cloud Run (Gemini via Vertex AI)

> **Status:** the image was built and run locally (see "Verified locally").
> The Cloud Run deploy itself has **not been run**: the build environment
> had no Google Cloud credentials. There is no live URL yet. After you
> deploy, record the URL in `PHASES.md` (Phase 6) and `RESUME_FACTS.md`.

## What gets deployed
- **Image** (`triage-agent/Dockerfile`):
  - `python:3.13-slim`, CPU-only torch 2.15.0, pinned `requirements.txt`
  - the `all-MiniLM-L6-v2` embedding model and the sample guideline
    knowledge base baked in; the service never contacts Hugging Face
    (`HF_HUB_OFFLINE=1`)
  - runs as a non-root user (uid 10001), serves Gradio on `$PORT` (8080)
- **LLM**: `LLM_PROVIDER=vertexai`, i.e. Gemini through Vertex AI using the
  service account's Application Default Credentials. No API key.
  Defaults: `gemini-3.1-pro-preview` (triage) and `gemini-3.8-flash`
  (intake, retrieval). These are the models the eval suite was run with,
  but through the Gemini API, not Vertex AI. **Check that both are
  available in your project and Vertex location** (preview models are often
  `global`-only) and override with `MODEL_NAME` / `FAST_MODEL_NAME` if not.
  The Vertex code path is covered by offline unit tests only; no Vertex call
  has been made.
- **Session state**:
  - Without a database, sessions are in memory, so the script pins
    `--max-instances 1`.
  - With Cloud SQL for PostgreSQL (`CHECKPOINT_DB_URL`), sessions survive
    restarts and are shared by every instance.
  - The browser keeps the session id (`gr.BrowserState`, localStorage),
    encrypted with `BROWSER_STATE_SECRET`. That secret must be **stable**
    across restarts and identical on every instance. Gradio otherwise
    generates a random one per launch and a reload starts a new
    conversation (seen in testing). The script creates it in Secret Manager
    when `CLOUDSQL_INSTANCE` is set.
- **Tracing** (optional): `TRACE_API_URL` plus a `TRACE_API_KEY` secret.

## Prerequisites
- `gcloud` authenticated (`gcloud auth login`) with Owner or equivalent on the project
- Billing enabled on the project
- Run everything from the `triage-agent/` directory

## One-command deploy
```bash
cd triage-agent
PROJECT_ID=my-project REGION=us-central1 ./deploy/deploy_cloud_run.sh
```
The script is idempotent. It enables the Run, Cloud Build, Artifact
Registry, Vertex AI and Secret Manager APIs. It creates the Artifact Registry
repo `triage` and a runtime service account `triage-ai-run` with
`roles/aiplatform.user`. It builds with Cloud Build and deploys the service.
It prints the service URL at the end.

## With persistent sessions (Cloud SQL Postgres)
```bash
gcloud sql instances create triage-pg --database-version=POSTGRES_16 \
  --region=us-central1 --tier=db-f1-micro
gcloud sql databases create triage --instance=triage-pg
gcloud sql users create triage --instance=triage-pg --password='CHOOSE-A-PASSWORD'
# Unix-socket URL used inside Cloud Run:
printf 'postgresql://triage:CHOOSE-A-PASSWORD@/triage?host=/cloudsql/%s' \
  "$(gcloud sql instances describe triage-pg --format='value(connectionName)')" \
  | gcloud secrets create triage-checkpoint-db-url --data-file=-

PROJECT_ID=my-project REGION=us-central1 \
CLOUDSQL_INSTANCE="$(gcloud sql instances describe triage-pg --format='value(connectionName)')" \
./deploy/deploy_cloud_run.sh
```
The app creates its checkpoint tables on first start (`PostgresSaver.setup()`).

## With tracing
```bash
printf '%s' 'YOUR-PLATFORM-KEY' | gcloud secrets create triage-trace-api-key --data-file=-
PROJECT_ID=my-project REGION=us-central1 TRACE_API_URL=https://your-platform.example \
./deploy/deploy_cloud_run.sh
```
Before enabling this, check the payload shape documented in `tracing.py`
against your platform's `/api/v1/traces` schema.

## Service settings the script uses, and why
| Flag | Value | Reason |
|---|---|---|
| `--timeout` | 600 s | a completing turn makes several LLM calls; the 31-case eval's slowest whole case (all turns) was 123 s |
| `--session-affinity` | on | Gradio keeps per-tab state (`gr.State`) on the instance that served it |
| `--min-instances` | 1 | avoids the image cold start (model load) on the first request |
| `--max-instances` | 1 without Cloud SQL, 10 with | in-memory sessions are per instance |
| `--memory` | 2Gi (override with `MEMORY`) | measured 497.2 MiB after one triage locally (one session); 2Gi is headroom for `--concurrency 20`, which was not load-tested |
| `--no-allow-unauthenticated` | default | it is a prototype handling health information; set `ALLOW_UNAUTHENTICATED=true` deliberately |

## Verified locally (this repository's build environment, 2026-11-17)
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
