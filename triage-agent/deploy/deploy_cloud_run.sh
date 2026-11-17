#!/usr/bin/env bash
# deploy/deploy_cloud_run.sh — build the TriageAI image with Cloud Build and
# deploy it to Cloud Run.
#
# Usage (from triage-agent/):
#   PROJECT_ID=my-project REGION=us-central1 ./deploy/deploy_cloud_run.sh
#
# LLM (LLM_MODE):
#   gemini-api (default)  Gemini through Google's OpenAI-compatible endpoint,
#                         key from secret "gemini-api-key" (how every eval ran)
#   vertexai              Gemini on Vertex AI with the service account (no key)
#
# Secrets used when they exist (create them first; values are never printed):
#   gemini-api-key                Gemini API key (LLM_MODE=gemini-api)
#   triage-checkpoint-db-url      Postgres URL for sessions, e.g. Neon's DIRECT
#                                 (non-pooled) connection string
#   triage-browser-state-secret   fixed key for Gradio BrowserState (created
#                                 here if missing, 64 hex characters)
#   triage-trace-api-key          eval platform project API key (evp_live_...):
#                                 when it exists, tracing is on and every turn
#                                 is sent to TRACE_API_URL/api/v1/traces
#
# Tracing: TRACE_API_URL (default https://eval-web-827630707333.us-central1.run.app)
#
# Optional:
#   SERVICE=triage-ai  MAX_INSTANCES=2  MIN_INSTANCES=0  MEMORY=2Gi
#   MODEL_NAME / FAST_MODEL_NAME     default gemini-3.1-pro-preview / gemini-3.8-flash
#   RATE_LIMIT_SESSION=20/600  RATE_LIMIT_IP=60/3600  MAX_MESSAGE_CHARS=2000
#   ALLOW_UNAUTHENTICATED=true       public URL (default true)
#
# Build notes (R2 Phase 1): the Dockerfile needs BuildKit, so the build uses
# deploy/cloudbuild.yaml; iris.who.int returns HTTP 403 to Cloud Build, so the
# WHO PDFs are fetched here first (sha256-checked) and go in the private build
# upload. They are in no layer of the final image.

set -euo pipefail
: "${PROJECT_ID:?set PROJECT_ID}"
: "${REGION:?set REGION, e.g. us-central1}"
SERVICE="${SERVICE:-triage-ai}"
REPO="${REPO:-triage}"
LLM_MODE="${LLM_MODE:-gemini-api}"
TAG="${TAG:-$(git rev-parse --short HEAD 2>/dev/null || date +%Y%m%d%H%M%S)}"
IMAGE="${REGION}-docker.pkg.dev/${PROJECT_ID}/${REPO}/${SERVICE}:${TAG}"
SA_NAME="${SERVICE}-run"
SA="${SA_NAME}@${PROJECT_ID}.iam.gserviceaccount.com"
PY="${PYTHON:-python}"

secret_exists() { gcloud secrets describe "$1" --project "$PROJECT_ID" >/dev/null 2>&1; }

gcloud config set project "$PROJECT_ID" >/dev/null

echo "==> Enabling APIs"
gcloud services enable run.googleapis.com cloudbuild.googleapis.com \
  artifactregistry.googleapis.com secretmanager.googleapis.com
[[ "$LLM_MODE" == "vertexai" ]] && gcloud services enable aiplatform.googleapis.com

echo "==> Artifact Registry repository ${REPO}"
gcloud artifacts repositories describe "$REPO" --location "$REGION" >/dev/null 2>&1 || \
  gcloud artifacts repositories create "$REPO" --repository-format=docker --location "$REGION"

echo "==> Runtime service account ${SA}"
gcloud iam service-accounts describe "$SA" >/dev/null 2>&1 || \
  gcloud iam service-accounts create "$SA_NAME" --display-name "TriageAI Cloud Run runtime"
[[ "$LLM_MODE" == "vertexai" ]] && gcloud projects add-iam-policy-binding "$PROJECT_ID" \
  --member "serviceAccount:${SA}" --role roles/aiplatform.user --condition=None >/dev/null

echo "==> WHO guideline PDFs (download from iris.who.int if missing, check sha256)"
"$PY" -m rag.fetch_guidelines

echo "==> Building ${IMAGE} with Cloud Build (BuildKit)"
gcloud builds submit . --config deploy/cloudbuild.yaml --substitutions "_IMAGE=${IMAGE}"

ENV_VARS="RATE_LIMIT_SESSION=${RATE_LIMIT_SESSION:-20/600},RATE_LIMIT_IP=${RATE_LIMIT_IP:-60/3600},MAX_MESSAGE_CHARS=${MAX_MESSAGE_CHARS:-2000}"
SECRETS=()
if [[ "$LLM_MODE" == "vertexai" ]]; then
  ENV_VARS+=",LLM_PROVIDER=vertexai,GOOGLE_CLOUD_PROJECT=${PROJECT_ID},GOOGLE_CLOUD_LOCATION=${VERTEX_LOCATION:-global}"
  [[ -n "${MODEL_NAME:-}" ]] && ENV_VARS+=",MODEL_NAME=${MODEL_NAME}"
  [[ -n "${FAST_MODEL_NAME:-}" ]] && ENV_VARS+=",FAST_MODEL_NAME=${FAST_MODEL_NAME}"
else
  secret_exists gemini-api-key || { echo "secret gemini-api-key is missing"; exit 1; }
  ENV_VARS+=",LLM_PROVIDER=openai,OPENAI_BASE_URL=https://generativelanguage.googleapis.com/v1beta/openai/"
  ENV_VARS+=",MODEL_NAME=${MODEL_NAME:-gemini-3.1-pro-preview},FAST_MODEL_NAME=${FAST_MODEL_NAME:-gemini-3.8-flash}"
  SECRETS+=("OPENAI_API_KEY=gemini-api-key:latest")
fi

if secret_exists triage-checkpoint-db-url; then
  echo "==> Postgres sessions: secret triage-checkpoint-db-url"
  SECRETS+=("CHECKPOINT_DB_URL=triage-checkpoint-db-url:latest")
  if ! secret_exists triage-browser-state-secret; then
    "$PY" -c "import secrets,sys; sys.stdout.write(secrets.token_hex(32))" | \
      gcloud secrets create triage-browser-state-secret --data-file=-
  fi
  SECRETS+=("BROWSER_STATE_SECRET=triage-browser-state-secret:latest")
  MAX_INSTANCES="${MAX_INSTANCES:-2}"
else
  echo "==> No triage-checkpoint-db-url: sessions in memory, so max 1 instance"
  MAX_INSTANCES=1
fi

if secret_exists triage-trace-api-key; then
  TRACE_API_URL="${TRACE_API_URL:-https://eval-web-827630707333.us-central1.run.app}"
  echo "==> Tracing on: ${TRACE_API_URL}/api/v1/traces (key: secret triage-trace-api-key)"
  ENV_VARS+=",TRACE_API_URL=${TRACE_API_URL}"
  SECRETS+=("TRACE_API_KEY=triage-trace-api-key:latest")
else
  echo "==> No triage-trace-api-key secret: tracing off"
fi

for s in ${SECRETS[@]+"${SECRETS[@]}"}; do
  name="${s#*=}"; name="${name%%:*}"
  gcloud secrets add-iam-policy-binding "$name" --member "serviceAccount:${SA}" \
    --role roles/secretmanager.secretAccessor >/dev/null
done

AUTH=$([[ "${ALLOW_UNAUTHENTICATED:-true}" == "true" ]] && echo --allow-unauthenticated || echo --no-allow-unauthenticated)

echo "==> Deploying ${SERVICE} (max ${MAX_INSTANCES} instances)"
# --timeout 300: a completing turn makes several sequential LLM calls.
# --session-affinity: Gradio keeps a connection per browser tab.
# --cpu-boost: faster cold start (startup is dominated by Python imports).
gcloud run deploy "$SERVICE" \
  --image "$IMAGE" --region "$REGION" --service-account "$SA" \
  --set-env-vars "$ENV_VARS" ${SECRETS:+--set-secrets "$(IFS=,; echo "${SECRETS[*]}")"} \
  --cpu 1 --cpu-boost --memory "${MEMORY:-2Gi}" --timeout 300 \
  --min-instances "${MIN_INSTANCES:-0}" --max-instances "$MAX_INSTANCES" \
  --session-affinity --port 8080 "$AUTH"

URL=$(gcloud run services describe "$SERVICE" --region "$REGION" --format 'value(status.url)')
echo "==> Deployed: ${URL}"
