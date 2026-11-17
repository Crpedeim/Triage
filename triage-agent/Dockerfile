# syntax=docker/dockerfile:1.7
#
# TriageAI container image (Cloud Run target). Build context: triage-agent/
#
#   python -m rag.fetch_guidelines          # put the WHO PDFs in rag/guidelines/
#   docker build -t triage-ai .
#   docker run -p 8080:8080 -e LLM_PROVIDER=... triage-ai
#
# The embedding model (all-MiniLM-L6-v2) and the guideline knowledge base are
# built at BUILD time, in the "kb" stage: the WHO PDFs listed in
# rag/guidelines/sources.json are sha256-checked, extracted (table-aware),
# chunked and embedded into ChromaDB. The running service never downloads from
# Hugging Face or WHO and never ingests at startup.
#
# Where the PDFs come from: rag.fetch_guidelines uses the copies in the build
# context when their sha256 matches, and downloads from iris.who.int only when
# a file is missing. (iris.who.int answers Cloud Build's machines with HTTP 403,
# so deploys put the PDFs, fetched from WHO beforehand, in the private build
# upload.) The PDFs are not in git (two are "All rights reserved") and, because
# only /app, /opt/hf and /opt/chroma_db are copied out of the "kb" stage after
# the PDFs are deleted there, they are in no layer of the final image.
#
# Optional build secret "ca": an extra CA bundle used ONLY during the build
# (for TLS-intercepting build networks). It is never written to an image layer.
#   docker build --secret id=ca,src=/path/ca-bundle.crt ...

FROM python:3.13-slim AS deps

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HOME=/opt/hf \
    CHROMA_PERSIST_DIR=/opt/chroma_db \
    GRADIO_ANALYTICS_ENABLED=False \
    ANONYMIZED_TELEMETRY=False \
    PORT=8080

WORKDIR /app

# CPU-only torch first (Cloud Run has no GPU; the default PyPI wheel bundles CUDA),
# then the pinned runtime requirements (torch==2.15.0 is satisfied by 2.15.0+cpu).
COPY requirements.txt .
RUN --mount=type=secret,id=ca,required=false \
    if [ -f /run/secrets/ca ]; then export PIP_CERT=/run/secrets/ca; fi; \
    pip install --index-url https://download.pytorch.org/whl/cpu torch==2.15.0 \
 && pip install -r requirements.txt


# ─── kb: build the knowledge base (the PDFs exist only in this stage) ────────
FROM deps AS kb
COPY . .
# build_kb writes its page/chunk report to /opt/chroma_db/kb_report.json.
RUN --mount=type=secret,id=ca,required=false \
    if [ -f /run/secrets/ca ]; then export REQUESTS_CA_BUNDLE=/run/secrets/ca SSL_CERT_FILE=/run/secrets/ca; fi; \
    python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')" \
 && python -m rag.fetch_guidelines \
 && python -m rag.build_kb \
 && rm -f rag/guidelines/*.pdf rag/guidelines/*.part


# ─── runtime image ───────────────────────────────────────────────────────────
FROM deps
COPY --from=kb /app /app
COPY --from=kb /opt/hf /opt/hf
COPY --from=kb /opt/chroma_db /opt/chroma_db

# No network access to Hugging Face at runtime
ENV HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1

RUN useradd --uid 10001 --create-home app && chown -R app:app /app /opt/hf /opt/chroma_db
USER app

EXPOSE 8080
CMD ["python", "app.py"]
