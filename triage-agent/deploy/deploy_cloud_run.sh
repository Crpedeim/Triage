#!/usr/bin/env bash
# deploy/deploy_cloud_run.sh — build the TriageAI image with Cloud Build and
# deploy it to Cloud Run, calling Gemini through Vertex AI.
#
# Usage (from triage-agent/):
#   PROJECT_ID=my-project REGION=us-central1 ./deploy/deploy_cloud_run.sh
#
# Optional:
#   SERVICE=triage-ai               Cloud Run service name
#   VERTEX_LOCATION=global          Vertex AI location for Gemini
#   MODEL_NAME / FAST_MODEL_NAME    override the Gemini models
#   CLOUDSQL_INSTANCE=proj:region:inst  + secret "triage-checkpoint-db-url"
#                                   → Postgres session persistence
#   TRACE_API_URL=https://...       + secret "triage-trace-api-key" → tracing
#   ALLOW_UNAUTHENTICATED=false     set true to make the UI public
#
# Every step is idempotent. See deploy/CLOUD_RUN.md for what each step does.
set -euo pipefail

: "${PROJECT_ID:?set PROJECT_ID}"
: "${REGION:?set REGION, e.g. us-central1}"
SERVICE="${SERVICE:-triage-ai}"
REPO="${REPO:-triage}"
VERTEX_LOCATION="${VERTEX_LOCATION:-global}"
TAG="${TAG:-$(git rev-parse --short HEAD 2>/dev/null || date +%Y%m%d%H%M%S)}"
IMAGE="${REGION}-docker.pkg.dev/${PROJECT_ID}/${REPO}/${SERVICE}:${TAG}"
SA_NAME="${SERVICE}-run"
SA="${SA_NAME}@${PROJECT_ID}.iam.gserviceaccount.com"

gcloud config set project "$PROJECT_ID" >/dev/null

echo "==> Enabling APIs"
gcloud services enable run.googleapis.com cloudbuild.googleapis.com \
  artifactregistry.googleapis.com aiplatform.googleapis.com secretmanager.googleapis.com

echo "==> Artifact Registry repository ${REPO}"
gcloud artifacts repositories describe "$REPO" --location "$REGION" >/dev/null 2>&1 || \
  gcloud artifacts repositories create "$REPO" --repository-format=docker --location "$REGION"

echo "==> Runtime service account ${SA}"
gcloud iam service-accounts describe "$SA" >/dev/null 2>&1 || \
  gcloud iam service-accounts create "$SA_NAME" --display-name "TriageAI Cloud Run runtime"
gcloud projects add-iam-policy-binding "$PROJECT_ID" --member "serviceAccount:${SA}" \
  --role roles/aiplatform.user --condition=None >/dev/null

echo "==> Building ${IMAGE} with Cloud Build"
gcloud builds submit . --tag "$IMAGE"

ENV_VARS="LLM_PROVIDER=vertexai,GOOGLE_CLOUD_PROJECT=${PROJECT_ID},GOOGLE_CLOUD_LOCATION=${VERTEX_LOCATION}"
[[ -n "${MODEL_NAME:-}" ]] && ENV_VARS+=",MODEL_NAME=${MODEL_NAME}"
[[ -n "${FAST_MODEL_NAME:-}" ]] && ENV_VARS+=",FAST_MODEL_NAME=${FAST_MODEL_NAME}"

EXTRA=()
SECRETS=()
if [[ -n "${CLOUDSQL_INSTANCE:-}" ]]; then
  gcloud projects add-iam-policy-binding "$PROJECT_ID" --member "serviceAccount:${SA}" \
    --role roles/cloudsql.client --condition=None >/dev/null
  EXTRA+=(--add-cloudsql-instances "$CLOUDSQL_INSTANCE")
  SECRETS+=("CHECKPOINT_DB_URL=triage-checkpoint-db-url:latest")
  # Stable key for the browser-held session id (gr.BrowserState). Without it a
  # restarted or second instance cannot read the id and the chat starts over.
  gcloud secrets describe triage-browser-state-secret >/dev/null 2>&1 || \
    openssl rand -hex 32 | gcloud secrets create triage-browser-state-secret --data-file=-
  SECRETS+=("BROWSER_STATE_SECRET=triage-browser-state-secret:latest")
fi
if [[ -n "${TRACE_API_URL:-}" ]]; then
  ENV_VARS+=",TRACE_API_URL=${TRACE_API_URL}"
  SECRETS+=("TRACE_API_KEY=triage-trace-api-key:latest")
fi
if (( ${#SECRETS[@]} )); then
  gcloud projects add-iam-policy-binding "$PROJECT_ID" --member "serviceAccount:${SA}" \
    --role roles/secretmanager.secretAccessor --condition=None >/dev/null
  EXTRA+=(--set-secrets "$(IFS=,; echo "${SECRETS[*]}")")
fi
if [[ "${ALLOW_UNAUTHENTICATED:-false}" == "true" ]]; then
  EXTRA+=(--allow-unauthenticated)
else
  EXTRA+=(--no-allow-unauthenticated)
fi

echo "==> Deploying ${SERVICE}"
# --timeout 600: a completing turn makes several sequential LLM calls.
# --session-affinity: Gradio keeps a connection per browser tab.
# Without CLOUDSQL_INSTANCE sessions live in memory: keep --max-instances 1.
MAX_INSTANCES=$([[ -n "${CLOUDSQL_INSTANCE:-}" ]] && echo 10 || echo 1)
gcloud run deploy "$SERVICE" \
  --image "$IMAGE" --region "$REGION" --service-account "$SA" \
  --set-env-vars "$ENV_VARS" \
  --cpu 2 --memory "${MEMORY:-2Gi}" --timeout 600 --concurrency 20 \
  --min-instances "${MIN_INSTANCES:-1}" --max-instances "$MAX_INSTANCES" \
  --session-affinity --port 8080 "${EXTRA[@]}"

URL=$(gcloud run services describe "$SERVICE" --region "$REGION" --format 'value(status.url)')
echo "==> Deployed: ${URL}"
echo "Record this URL in RESUME_FACTS.md / PHASES.md once you have checked it in a browser."
