"""
conftest.py — Mock embedding model for sandboxed/offline test environments.

If all-MiniLM-L6-v2 is cached locally (~/.cache/huggingface), uses real model.
Otherwise uses a deterministic hash-based mock that produces valid 384-dim vectors.

Mock vectors preserve semantic similarity (shared words = lower cosine distance)
which is enough to test retrieval pipeline structure without a real transformer.
"""

from __future__ import annotations

import hashlib
import math
import os
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

EMBED_DIM = 384


def _hash_embed(text: str) -> list[float]:
    vec = [0.0] * EMBED_DIM
    words = text.lower().split()
    for word in words:
        h = int(hashlib.md5(word.encode()).hexdigest(), 16)
        idx = h % EMBED_DIM
        vec[idx] += 1.0
    norm = math.sqrt(sum(v * v for v in vec)) or 1.0
    return [v / norm for v in vec]


class MockSentenceTransformer:
    def __init__(self, model_name: str, *args, **kwargs):
        self.model_name = model_name

    def encode(self, texts, batch_size=32, show_progress_bar=False):
        import numpy as np
        return np.array([_hash_embed(t) for t in texts])


def _model_is_cached() -> bool:
    """Return True if all-MiniLM-L6-v2 model files are already on disk."""
    cache_dirs = [
        os.path.expanduser("~/.cache/huggingface/hub"),
        os.path.expanduser("~/.cache/torch/sentence_transformers"),
    ]
    for cache_dir in cache_dirs:
        if os.path.isdir(cache_dir):
            for entry in os.listdir(cache_dir):
                if "MiniLM" in entry or "minilm" in entry.lower():
                    return True
    return False


@pytest.fixture(autouse=True)
def mock_embedding_model():
    """Auto-patches embedding model when not cached locally."""
    use_mock = not _model_is_cached() or os.environ.get("USE_MOCK_EMBEDDINGS") == "1"

    # Always clear lru_cache to prevent state bleeding between tests
    try:
        from rag.embeddings import _get_local_model
        _get_local_model.cache_clear()
    except Exception:
        pass

    if not use_mock:
        yield
        return

    mock_model = MockSentenceTransformer("all-MiniLM-L6-v2")
    with patch("rag.embeddings._get_local_model", return_value=mock_model):
        yield
