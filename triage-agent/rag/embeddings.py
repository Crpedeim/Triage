"""
rag/embeddings.py — Embedding Model Abstraction

All embedding operations (for both ingestion and query) go through here.
Switching embedding providers = changing EMBEDDING_PROVIDER in .env.

WHY KEEP EMBEDDINGS SEPARATE FROM LLM:
The embedding model and the chat LLM are independent choices.
You can use Anthropic for chat but local MiniLM for embeddings (recommended —
saves API costs, works offline, fast enough for this use case).

IMPORTANT: The embedding model used during INGESTION must be the SAME
as the one used during SEARCH. If you change providers after ingesting,
you must re-run ingest.py to rebuild the vector store with new embeddings.

Supported providers (set EMBEDDING_PROVIDER in .env):
    local   → all-MiniLM-L6-v2 via sentence-transformers (default)
              - Free, runs on CPU, 384-dimensional embeddings
              - Downloads ~90MB model on first run, cached locally
    openai  → text-embedding-3-small via OpenAI API
              - Better quality, 1536-dimensional embeddings
              - Costs ~$0.02 per 1M tokens (negligible for our use case)
              - Requires OPENAI_API_KEY
"""

from __future__ import annotations

import os
from functools import lru_cache
from typing import Union

from dotenv import load_dotenv

load_dotenv()

EMBEDDING_PROVIDER = os.getenv("EMBEDDING_PROVIDER", "local")


@lru_cache(maxsize=1)
def _get_local_model():
    """
    Load all-MiniLM-L6-v2 from sentence-transformers.

    @lru_cache ensures we load the model ONCE and reuse it.
    Loading a transformer model takes ~2-3 seconds — we don't want to
    do this on every function call.

    First call downloads the model (~90MB) to ~/.cache/huggingface/
    and caches it permanently. Subsequent calls are instant.
    """
    from sentence_transformers import SentenceTransformer
    print("[embeddings] Loading all-MiniLM-L6-v2... (first run downloads ~90MB)")
    model = SentenceTransformer("all-MiniLM-L6-v2")
    print("[embeddings] Model loaded.")
    return model


def embed_texts(texts: list[str]) -> list[list[float]]:
    """
    Embed a list of text chunks.

    Used by the INGESTION PIPELINE (rag/ingest.py).
    Batches the inputs for efficiency — never call embed_query() in a loop.

    Args:
        texts: List of text strings to embed.

    Returns:
        List of embedding vectors. Each vector is a list of floats.
        Length of each vector depends on the model:
        - all-MiniLM-L6-v2: 384 dimensions
        - text-embedding-3-small: 1536 dimensions

    Example:
        embeddings = embed_texts(["cough in children", "fever management"])
        # embeddings[0] is the vector for "cough in children"
    """
    provider = os.getenv("EMBEDDING_PROVIDER", "local")

    if provider == "local":
        model = _get_local_model()
        vectors = model.encode(texts, batch_size=32, show_progress_bar=False)
        return [v.tolist() for v in vectors]

    elif provider == "openai":
        from openai import OpenAI
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise EnvironmentError("OPENAI_API_KEY required for openai embeddings")
        client = OpenAI(api_key=api_key)
        response = client.embeddings.create(
            input=texts,
            model="text-embedding-3-small",
        )
        return [item.embedding for item in response.data]

    else:
        raise ValueError(
            f"EMBEDDING_PROVIDER='{provider}' not supported. "
            "Choose 'local' or 'openai'."
        )


def embed_query(query: str) -> list[float]:
    """
    Embed a single search query.

    Used by the RETRIEVAL AGENT at runtime. Takes a single string
    and returns a single embedding vector.

    Note: Uses the same model as embed_texts() — they MUST match.

    Args:
        query: The search query string.

    Returns:
        A single embedding vector (list of floats).

    Example:
        vector = embed_query("pneumonia classification child fast breathing")
        results = search(vector, n_results=5)
    """
    embeddings = embed_texts([query])
    return embeddings[0]


def get_embedding_dimension() -> int:
    """
    Return the dimension of embeddings produced by the current model.
    Useful for validating stored vectors match the current model.
    """
    provider = os.getenv("EMBEDDING_PROVIDER", "local")
    dimensions = {
        "local": 384,
        "openai": 1536,
    }
    return dimensions.get(provider, 384)
