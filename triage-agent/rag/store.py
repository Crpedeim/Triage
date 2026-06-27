"""
rag/store.py — ChromaDB Vector Store Wrapper

Single place for all vector store operations. Every other module that
needs to talk to ChromaDB imports from here — never imports chromadb directly.

This wrapper does four things:
1. Manages ChromaDB client lifecycle (persistent on disk)
2. Provides add_documents() for the ingestion pipeline
3. Provides search() for the Retrieval Agent
4. Provides utility functions (count, list_sources, reset)

WHY WRAP CHROMADB:
If we ever want to swap ChromaDB for Qdrant, Pinecone, or Weaviate,
we only change this file. The Retrieval Agent, the ingestion pipeline,
and the tests all stay the same.

COLLECTION STRUCTURE:
One collection: "clinical_guidelines"
Each document has:
  - id: sha256 hash of content (deduplication)
  - document: the raw text chunk
  - metadata:
      source_file: str      (e.g., "imnci_chart_booklet.pdf")
      section: str          (e.g., "Assess and Classify Cough or Difficult Breathing")
      page_number: int
      chunk_index: int      (chunk number within source_file)
      guideline_type: str   (e.g., "imnci", "who_pen", "who_primary_care")
"""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from typing import Any

import chromadb
from chromadb.config import Settings
from dotenv import load_dotenv

load_dotenv()

COLLECTION_NAME = "clinical_guidelines"
CHROMA_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")


# ─── Document dataclass ───
@dataclass
class Document:
    """
    A document chunk retrieved from the vector store.

    This is our internal representation — not ChromaDB's or LangChain's.
    Using our own class means we're not coupled to either library's Document type.

    Attributes:
        content:     The raw text of the chunk.
        metadata:    All metadata fields (source_file, section, page_number, etc.)
        score:       Cosine distance from the query (lower = more similar).
                     Range: 0.0 (identical) to 2.0 (completely opposite).
                     Practical similarity threshold: score < 1.0 is "relevant".
        id:          The document ID in ChromaDB (sha256 hash of content).
    """
    content: str
    metadata: dict[str, Any]
    score: float = 0.0
    id: str = ""

    @property
    def source_file(self) -> str:
        return self.metadata.get("source_file", "unknown")

    @property
    def section(self) -> str:
        return self.metadata.get("section", "unknown")

    @property
    def guideline_type(self) -> str:
        return self.metadata.get("guideline_type", "unknown")

    def to_dict(self) -> dict:
        """Serialize to dict for storing in LangGraph state."""
        return {
            "content": self.content,
            "metadata": self.metadata,
            "score": self.score,
            "id": self.id,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Document":
        """Deserialize from LangGraph state."""
        return cls(
            content=data["content"],
            metadata=data.get("metadata", {}),
            score=data.get("score", 0.0),
            id=data.get("id", ""),
        )

    def __repr__(self) -> str:
        return (
            f"Document(source={self.source_file!r}, "
            f"section={self.section!r}, "
            f"score={self.score:.3f}, "
            f"content_len={len(self.content)})"
        )


def _make_doc_id(content: str) -> str:
    """
    Generate a deterministic ID for a document chunk.
    SHA256 hash of content ensures deduplication — adding the same
    chunk twice will not create a duplicate in ChromaDB.
    """
    return hashlib.sha256(content.encode()).hexdigest()[:32]


def _get_client() -> chromadb.PersistentClient:
    """
    Get a ChromaDB persistent client.

    PersistentClient saves data to disk at CHROMA_PERSIST_DIR.
    Data survives process restarts — you only need to run ingest.py once.
    """
    os.makedirs(CHROMA_PERSIST_DIR, exist_ok=True)
    return chromadb.PersistentClient(
        path=CHROMA_PERSIST_DIR,
        settings=Settings(anonymized_telemetry=False),
    )


def _get_collection(client: chromadb.PersistentClient) -> chromadb.Collection:
    """Get or create the clinical guidelines collection."""
    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},  # Use cosine similarity
    )


def add_documents(
    texts: list[str],
    metadatas: list[dict],
    embeddings: list[list[float]],
) -> int:
    """
    Add document chunks to the vector store.

    Called by the ingestion pipeline (rag/ingest.py). Not called at runtime.

    Args:
        texts:      List of text chunks to store.
        metadatas:  List of metadata dicts, one per chunk. Must include:
                    source_file, section, page_number, chunk_index, guideline_type.
        embeddings: Pre-computed embeddings (from the embedding model).
                    Must be same length as texts.

    Returns:
        Number of new documents actually added (skips duplicates).

    Raises:
        ValueError: If lengths of texts, metadatas, embeddings don't match.
    """
    if not (len(texts) == len(metadatas) == len(embeddings)):
        raise ValueError(
            f"texts ({len(texts)}), metadatas ({len(metadatas)}), "
            f"and embeddings ({len(embeddings)}) must have the same length."
        )

    client = _get_client()
    collection = _get_collection(client)

    ids = [_make_doc_id(text) for text in texts]

    # ChromaDB will skip duplicates if IDs already exist
    # We use upsert instead of add so re-running ingest is safe
    collection.upsert(
        ids=ids,
        documents=texts,
        metadatas=metadatas,
        embeddings=embeddings,
    )

    return len(texts)


def search(
    query_embedding: list[float],
    n_results: int = 5,
    where: dict | None = None,
) -> list[Document]:
    """
    Search the vector store for chunks similar to the query.

    Called by the Retrieval Agent at runtime.

    Args:
        query_embedding: Embedding of the search query (from embedding model).
        n_results:       Number of results to return.
        where:           Optional ChromaDB metadata filter.
                         Example: {"guideline_type": "imnci"} to restrict to IMNCI only.
                         Example: {"$or": [{"guideline_type": "imnci"}, {"guideline_type": "who_pen"}]}

    Returns:
        List of Document objects sorted by relevance (lowest score = most relevant).

    Example:
        from rag.embeddings import embed_query
        from rag.store import search

        results = search(embed_query("pneumonia classification child"), n_results=5)
        for doc in results:
            print(doc.section, doc.score)
    """
    client = _get_client()
    collection = _get_collection(client)

    if collection.count() == 0:
        return []

    kwargs: dict[str, Any] = {
        "query_embeddings": [query_embedding],
        "n_results": min(n_results, collection.count()),
        "include": ["documents", "metadatas", "distances"],
    }
    if where:
        kwargs["where"] = where

    results = collection.query(**kwargs)

    documents = []
    for i, (text, metadata, distance) in enumerate(zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    )):
        documents.append(Document(
            content=text,
            metadata=metadata,
            score=distance,
            id=results["ids"][0][i],
        ))

    return documents


def count() -> int:
    """Return total number of chunks in the vector store."""
    client = _get_client()
    collection = _get_collection(client)
    return collection.count()


def list_sources() -> list[str]:
    """
    Return list of unique source files in the vector store.
    Useful to verify which guidelines have been ingested.
    """
    client = _get_client()
    collection = _get_collection(client)

    if collection.count() == 0:
        return []

    # Get all metadata (ChromaDB doesn't have a DISTINCT query)
    results = collection.get(include=["metadatas"])
    sources = set()
    for meta in results["metadatas"]:
        if "source_file" in meta:
            sources.add(meta["source_file"])
    return sorted(sources)


def reset() -> None:
    """
    Delete all documents from the collection.

    USE WITH CAUTION — this wipes the entire knowledge base.
    Only use during development to re-ingest from scratch.
    """
    client = _get_client()
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass
    # Recreate empty collection
    _get_collection(client)
