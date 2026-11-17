# TriageAI — container image for Google Cloud Run
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/app/.hf_cache \
    CHROMA_PERSIST_DIR=/app/chroma_db \
    GRADIO_ANALYTICS_ENABLED=False \
    PORT=8080

WORKDIR /app

# CPU-only PyTorch first, so sentence-transformers doesn't pull ~2 GB of CUDA wheels
RUN pip install --index-url https://download.pytorch.org/whl/cpu torch

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

# Bake the embedding model and the sample knowledge base into the image,
# so containers start ready without downloading anything.
# To use the full WHO PDFs instead, put them in rag/guidelines/ and
# change this to: python -m rag.ingest
RUN python -m rag.ingest --sample

EXPOSE 8080
CMD ["python", "app.py"]
