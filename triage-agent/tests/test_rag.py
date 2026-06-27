"""
tests/test_rag.py — Tests for Phase 2: RAG Pipeline and LLM Provider Layer

Run with:
    cd triage-agent
    python -m pytest tests/test_rag.py -v

These tests cover:
1. Embeddings: model loads, produces correct-dimension vectors, cosine similarity works
2. Store: add, search, count, list_sources, deduplication, reset
3. Ingest: sample data ingestion, chunk structure, metadata fields
4. Vector Search: search_guidelines, multi_query_search, format_for_triage_agent
5. LLM Provider: config parsing, model selection, error on missing key
6. End-to-end: ingest sample → search → verify relevant results returned

IMPORTANT: Tests that call embed_texts() will download all-MiniLM-L6-v2
on first run (~90MB). Subsequent runs use the cached model.

Tests that require API keys (evaluate_relevance, structured output) are
marked with @pytest.mark.skipif and skipped when no key is present.
"""

import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# ─── Make sure we can import from the project root ───
sys.path.insert(0, str(Path(__file__).parent.parent))

# ─── Set a test-only ChromaDB directory so tests don't pollute production ───
TEST_CHROMA_DIR = tempfile.mkdtemp(prefix="triage_test_chroma_")
os.environ["CHROMA_PERSIST_DIR"] = TEST_CHROMA_DIR
os.environ.setdefault("LLM_PROVIDER", "anthropic")
os.environ.setdefault("EMBEDDING_PROVIDER", "local")


# ═══════════════════════════════════════════════════════════════════════════════
# FIXTURES
# ═══════════════════════════════════════════════════════════════════════════════

@pytest.fixture(autouse=True)
def reset_store_between_tests():
    """
    Reset the ChromaDB collection before each test.
    Ensures tests are independent and don't interfere with each other.
    """
    from rag.store import reset
    reset()
    yield
    reset()


@pytest.fixture
def sample_texts():
    return [
        "SEVERE PNEUMONIA: chest indrawing present, urgent referral to hospital required.",
        "Fast breathing threshold: age 1-5 years, 40 breaths per minute or more.",
        "General danger signs: unable to drink, convulsions, lethargic or unconscious.",
        "Diarrhea classification: severe dehydration requires Plan C treatment.",
        "Hypertension grade 3: systolic >= 180, refer urgently to hospital.",
    ]


@pytest.fixture
def sample_metadatas():
    return [
        {"source_file": "imnci.pdf", "section": "Cough Classification", "page_number": 1, "chunk_index": 0, "guideline_type": "imnci"},
        {"source_file": "imnci.pdf", "section": "Fast Breathing", "page_number": 2, "chunk_index": 1, "guideline_type": "imnci"},
        {"source_file": "imnci.pdf", "section": "General Danger Signs", "page_number": 3, "chunk_index": 2, "guideline_type": "imnci"},
        {"source_file": "imnci.pdf", "section": "Diarrhea", "page_number": 4, "chunk_index": 3, "guideline_type": "imnci"},
        {"source_file": "who_pen.pdf", "section": "Hypertension", "page_number": 1, "chunk_index": 0, "guideline_type": "who_pen"},
    ]


# ═══════════════════════════════════════════════════════════════════════════════
# TEST GROUP 1: Embeddings
# ═══════════════════════════════════════════════════════════════════════════════

class TestEmbeddings:
    """Test embedding model loading and vector production."""

    def test_embed_single_text(self):
        """Single text produces a 384-dim vector."""
        from rag.embeddings import embed_texts
        result = embed_texts(["pneumonia in children"])
        assert len(result) == 1
        assert len(result[0]) == 384  # all-MiniLM-L6-v2 dimension

    def test_embed_multiple_texts(self):
        """Multiple texts produce multiple vectors."""
        from rag.embeddings import embed_texts
        texts = ["cough", "fever", "diarrhea", "chest indrawing"]
        result = embed_texts(texts)
        assert len(result) == 4
        for vec in result:
            assert len(vec) == 384

    def test_embed_query(self):
        """embed_query returns a single vector."""
        from rag.embeddings import embed_query
        vec = embed_query("pneumonia classification child")
        assert isinstance(vec, list)
        assert len(vec) == 384
        assert all(isinstance(v, float) for v in vec)

    def test_similar_texts_have_lower_distance(self):
        """
        Semantically similar texts should be closer in vector space
        than semantically unrelated texts.
        This validates the embedding model is working correctly.
        """
        import numpy as np
        from rag.embeddings import embed_texts

        vecs = embed_texts([
            "child has cough and fast breathing",   # query
            "pediatric cough with tachypnea",        # similar
            "diabetes mellitus type 2 management",   # unrelated
        ])

        query = np.array(vecs[0])
        similar = np.array(vecs[1])
        unrelated = np.array(vecs[2])

        # Cosine distance: 1 - dot(a,b) / (|a| * |b|)
        def cosine_dist(a, b):
            return 1.0 - float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

        dist_similar = cosine_dist(query, similar)
        dist_unrelated = cosine_dist(query, unrelated)

        # Similar text should be closer
        assert dist_similar < dist_unrelated, (
            f"Similar text (dist={dist_similar:.3f}) should be closer "
            f"than unrelated (dist={dist_unrelated:.3f})"
        )

    def test_embedding_dimension_function(self):
        """get_embedding_dimension returns correct value for local provider."""
        from rag.embeddings import get_embedding_dimension
        dim = get_embedding_dimension()
        assert dim == 384  # local provider


# ═══════════════════════════════════════════════════════════════════════════════
# TEST GROUP 2: Vector Store (ChromaDB Wrapper)
# ═══════════════════════════════════════════════════════════════════════════════

class TestStore:
    """Test ChromaDB operations via our store wrapper."""

    def test_empty_store_count(self):
        """Fresh store has 0 documents."""
        from rag.store import count
        assert count() == 0

    def test_add_and_count(self, sample_texts, sample_metadatas):
        """Adding documents increases count."""
        from rag.embeddings import embed_texts
        from rag.store import add_documents, count

        embeddings = embed_texts(sample_texts)
        added = add_documents(sample_texts, sample_metadatas, embeddings)

        assert added == 5
        assert count() == 5

    def test_list_sources_empty(self):
        """Empty store returns empty list."""
        from rag.store import list_sources
        assert list_sources() == []

    def test_list_sources_after_add(self, sample_texts, sample_metadatas):
        """list_sources returns unique source files."""
        from rag.embeddings import embed_texts
        from rag.store import add_documents, list_sources

        embeddings = embed_texts(sample_texts)
        add_documents(sample_texts, sample_metadatas, embeddings)

        sources = list_sources()
        assert "imnci.pdf" in sources
        assert "who_pen.pdf" in sources
        assert len(sources) == 2

    def test_deduplication(self, sample_texts, sample_metadatas):
        """Adding same documents twice does not create duplicates."""
        from rag.embeddings import embed_texts
        from rag.store import add_documents, count

        embeddings = embed_texts(sample_texts)
        add_documents(sample_texts, sample_metadatas, embeddings)
        add_documents(sample_texts, sample_metadatas, embeddings)  # Second add

        assert count() == 5  # Still 5, not 10

    def test_search_returns_results(self, sample_texts, sample_metadatas):
        """Searching after adding data returns relevant results."""
        from rag.embeddings import embed_query, embed_texts
        from rag.store import add_documents, search

        embeddings = embed_texts(sample_texts)
        add_documents(sample_texts, sample_metadatas, embeddings)

        query_vec = embed_query("pneumonia chest indrawing")
        results = search(query_vec, n_results=3)

        assert len(results) > 0
        assert len(results) <= 3

    def test_search_returns_document_objects(self, sample_texts, sample_metadatas):
        """Search results are Document objects with correct fields."""
        from rag.embeddings import embed_query, embed_texts
        from rag.store import Document, add_documents, search

        embeddings = embed_texts(sample_texts)
        add_documents(sample_texts, sample_metadatas, embeddings)

        query_vec = embed_query("pneumonia")
        results = search(query_vec, n_results=2)

        for doc in results:
            assert isinstance(doc, Document)
            assert isinstance(doc.content, str)
            assert len(doc.content) > 0
            assert isinstance(doc.score, float)
            assert doc.score >= 0.0
            assert isinstance(doc.metadata, dict)
            assert "source_file" in doc.metadata
            assert "section" in doc.metadata

    def test_search_empty_store_returns_empty(self):
        """Searching empty store returns empty list, not error."""
        from rag.embeddings import embed_query
        from rag.store import search

        query_vec = embed_query("pneumonia")
        results = search(query_vec, n_results=5)
        assert results == []

    def test_search_with_guideline_type_filter(self, sample_texts, sample_metadatas):
        """Metadata filter restricts results to specified guideline type."""
        from rag.embeddings import embed_query, embed_texts
        from rag.store import add_documents, search

        embeddings = embed_texts(sample_texts)
        add_documents(sample_texts, sample_metadatas, embeddings)

        query_vec = embed_query("urgent referral hospital")

        # Filter to who_pen only
        results = search(query_vec, n_results=5, where={"guideline_type": "who_pen"})

        # All results should be from who_pen
        for doc in results:
            assert doc.guideline_type == "who_pen", (
                f"Expected 'who_pen', got '{doc.guideline_type}'"
            )

    def test_search_relevance_ordering(self, sample_texts, sample_metadatas):
        """Most relevant result should come first (lower score = more relevant)."""
        from rag.embeddings import embed_query, embed_texts
        from rag.store import add_documents, search

        embeddings = embed_texts(sample_texts)
        add_documents(sample_texts, sample_metadatas, embeddings)

        # Query specifically about breathing thresholds
        query_vec = embed_query("breathing rate threshold per minute child age")
        results = search(query_vec, n_results=5)

        # Results should be ordered by score (ascending = most relevant first)
        for i in range(len(results) - 1):
            assert results[i].score <= results[i + 1].score, (
                f"Results not sorted: {results[i].score} > {results[i+1].score}"
            )

    def test_document_serialization(self):
        """Documents can round-trip through to_dict/from_dict (for LangGraph state)."""
        from rag.store import Document

        original = Document(
            content="Test clinical content",
            metadata={"source_file": "test.pdf", "section": "Test", "page_number": 1, "chunk_index": 0, "guideline_type": "imnci"},
            score=0.25,
            id="abc123",
        )
        as_dict = original.to_dict()
        reconstructed = Document.from_dict(as_dict)

        assert reconstructed.content == original.content
        assert reconstructed.score == original.score
        assert reconstructed.metadata == original.metadata

    def test_reset_clears_all_documents(self, sample_texts, sample_metadatas):
        """reset() removes all documents."""
        from rag.embeddings import embed_texts
        from rag.store import add_documents, count, reset

        embeddings = embed_texts(sample_texts)
        add_documents(sample_texts, sample_metadatas, embeddings)
        assert count() == 5

        reset()
        assert count() == 0

    def test_add_mismatched_lengths_raises(self, sample_texts, sample_metadatas):
        """Mismatched lengths raise ValueError."""
        from rag.embeddings import embed_texts
        from rag.store import add_documents

        embeddings = embed_texts(sample_texts[:3])  # Only 3 embeddings for 5 texts
        with pytest.raises(ValueError, match="same length"):
            add_documents(sample_texts, sample_metadatas, embeddings)


# ═══════════════════════════════════════════════════════════════════════════════
# TEST GROUP 3: Ingestion Pipeline
# ═══════════════════════════════════════════════════════════════════════════════

class TestIngestion:
    """Test the full ingestion pipeline."""

    def test_sample_data_ingestion(self):
        """ingest_sample_data() adds expected chunks to the store."""
        from rag.ingest import ingest_sample_data
        from rag.store import count, list_sources

        added = ingest_sample_data()

        assert added > 0
        assert count() > 0
        # Sample data has both imnci and who_pen content
        sources = list_sources()
        assert any("sample_imnci" in s for s in sources)

    def test_sample_data_has_required_metadata(self):
        """Every ingested chunk has required metadata fields."""
        from rag.ingest import ingest_sample_data
        from rag.store import _get_client, _get_collection

        ingest_sample_data()

        client = _get_client()
        collection = _get_collection(client)
        results = collection.get(include=["metadatas"])

        required_fields = {"source_file", "section", "page_number", "chunk_index", "guideline_type"}
        for meta in results["metadatas"]:
            missing = required_fields - set(meta.keys())
            assert not missing, f"Metadata missing fields: {missing}"

    def test_sample_data_idempotent(self):
        """Running ingest twice doesn't duplicate chunks."""
        from rag.ingest import ingest_sample_data
        from rag.store import count

        ingest_sample_data()
        count_after_first = count()

        ingest_sample_data()  # Second run
        count_after_second = count()

        assert count_after_first == count_after_second

    def test_chunk_pages_produces_chunks(self):
        """chunk_pages returns correctly structured chunks."""
        from rag.ingest import chunk_pages

        pages = [
            {
                "text": "ASSESS AND CLASSIFY THE SICK CHILD\n\n"
                        "Look for chest indrawing. Count breaths per minute. "
                        "Chest indrawing present means severe pneumonia. "
                        "This is a danger sign requiring urgent referral.",
                "page_number": 1,
                "source_file": "test.pdf",
            }
        ]
        chunks = chunk_pages(pages, "imnci")

        assert len(chunks) > 0
        for chunk in chunks:
            assert "text" in chunk
            assert "metadata" in chunk
            assert chunk["metadata"]["guideline_type"] == "imnci"
            assert chunk["metadata"]["source_file"] == "test.pdf"
            assert "section" in chunk["metadata"]
            assert len(chunk["text"]) >= 50  # Min chunk length

    def test_detect_guideline_type(self):
        """Guideline type is correctly inferred from filename."""
        from rag.ingest import _detect_guideline_type

        assert _detect_guideline_type("imnci_chart_booklet.pdf") == "imnci"
        assert _detect_guideline_type("who_pen_guidelines.pdf") == "who_pen"
        assert _detect_guideline_type("who_primary_care.pdf") == "who_primary_care"
        assert _detect_guideline_type("unknown_document.pdf") == "general"

    def test_section_extraction(self):
        """Section headers are detected correctly."""
        from rag.ingest import _extract_section_from_text

        # IMNCI-style ALL CAPS header
        imnci_text = "ASSESS AND CLASSIFY COUGH OR DIFFICULT BREATHING\nLook for chest indrawing."
        section = _extract_section_from_text(imnci_text, "Previous Section")
        assert "COUGH" in section.upper() or "ASSESS" in section.upper()

        # No header — inherits previous
        plain_text = "Continue looking for signs of pneumonia in the child."
        section2 = _extract_section_from_text(plain_text, "Previous Section")
        assert section2 == "Previous Section"

    def test_ingest_nonexistent_pdf_raises(self):
        """Ingesting a non-existent file raises FileNotFoundError."""
        from rag.ingest import ingest_pdf

        with pytest.raises(FileNotFoundError):
            ingest_pdf(Path("/nonexistent/path/file.pdf"))


# ═══════════════════════════════════════════════════════════════════════════════
# TEST GROUP 4: Vector Search Tool
# ═══════════════════════════════════════════════════════════════════════════════

class TestVectorSearchTool:
    """Test the search tools used by the Retrieval Agent."""

    @pytest.fixture(autouse=True)
    def populate_store(self):
        """Ingest sample data before each test in this group."""
        from rag.ingest import ingest_sample_data
        ingest_sample_data()

    def test_search_guidelines_finds_pneumonia_content(self):
        """Searching for pneumonia returns relevant guideline chunks."""
        from tools.vector_search import search_guidelines

        results = search_guidelines("pneumonia classification chest indrawing child")
        assert len(results) > 0

        # At least one result should mention pneumonia or chest indrawing
        combined_text = " ".join([doc.content.lower() for doc in results])
        assert "pneumonia" in combined_text or "chest indrawing" in combined_text

    def test_search_guidelines_empty_query_returns_empty(self):
        """Empty query returns empty list gracefully."""
        from tools.vector_search import search_guidelines

        results = search_guidelines("")
        assert results == []

    def test_search_guidelines_with_filter(self):
        """Filtering by guideline_type restricts results."""
        from tools.vector_search import search_guidelines

        imnci_results = search_guidelines(
            "urgent referral hospital",
            guideline_type="imnci"
        )
        for doc in imnci_results:
            assert doc.guideline_type == "imnci"

    def test_multi_query_search_deduplicates(self):
        """multi_query_search with overlapping queries doesn't return duplicates."""
        from tools.vector_search import multi_query_search

        queries = [
            "pneumonia classification child",
            "IMNCI classify pneumonia",   # Similar query — should not double docs
        ]
        results = multi_query_search(queries, n_results_per_query=3)

        ids = [doc.id for doc in results]
        assert len(ids) == len(set(ids)), "Duplicate documents returned"

    def test_multi_query_search_broader_coverage(self):
        """Multiple diverse queries retrieve more coverage than single query."""
        from tools.vector_search import multi_query_search, search_guidelines

        single_results = search_guidelines("fever management child", n_results=3)
        single_sections = {doc.section for doc in single_results}

        multi_results = multi_query_search([
            "fever management child",
            "diarrhea dehydration classification",
            "general danger signs",
        ], n_results_per_query=2)
        multi_sections = {doc.section for doc in multi_results}

        # Multi-query should cover more unique sections
        assert len(multi_sections) >= len(single_sections)

    def test_format_for_triage_agent_structure(self):
        """Formatted output has expected structure for LLM consumption."""
        from tools.vector_search import format_for_triage_agent, search_guidelines

        results = search_guidelines("pneumonia child", n_results=2)
        formatted = format_for_triage_agent(results)

        assert "=== GUIDELINE 1 ===" in formatted
        assert "Source:" in formatted
        assert "Section:" in formatted
        assert "---" in formatted

    def test_format_empty_docs(self):
        """format_for_triage_agent handles empty list gracefully."""
        from tools.vector_search import format_for_triage_agent

        result = format_for_triage_agent([])
        assert "No relevant guidelines" in result


# ═══════════════════════════════════════════════════════════════════════════════
# TEST GROUP 5: LLM Provider Configuration
# ═══════════════════════════════════════════════════════════════════════════════

class TestLLMProvider:
    """Test LLM provider abstraction layer."""

    def test_get_llm_info(self):
        """get_llm_info returns expected structure."""
        from llm import get_llm_info

        info = get_llm_info()
        assert "provider" in info
        assert "main_model" in info
        assert "fast_model" in info
        assert "temperature" in info
        assert info["temperature"] == 0.0

    def test_current_provider_reads_env(self):
        """current_provider() reads from LLM_PROVIDER env var."""
        from llm import current_provider

        original = os.environ.get("LLM_PROVIDER", "anthropic")
        os.environ["LLM_PROVIDER"] = "openai"

        try:
            assert current_provider() == "openai"
        finally:
            os.environ["LLM_PROVIDER"] = original

    def test_invalid_provider_raises(self):
        """Unknown provider raises ValueError."""
        from llm import get_llm

        original = os.environ.get("LLM_PROVIDER", "anthropic")
        os.environ["LLM_PROVIDER"] = "banana"
        os.environ.pop("ANTHROPIC_API_KEY", None)

        try:
            with pytest.raises(ValueError, match="not supported"):
                get_llm()
        finally:
            os.environ["LLM_PROVIDER"] = original

    def test_anthropic_missing_key_raises(self):
        """Missing ANTHROPIC_API_KEY raises EnvironmentError for anthropic provider."""
        from llm import get_llm

        original_provider = os.environ.get("LLM_PROVIDER")
        original_key = os.environ.get("ANTHROPIC_API_KEY")

        os.environ["LLM_PROVIDER"] = "anthropic"
        os.environ.pop("ANTHROPIC_API_KEY", None)

        try:
            with pytest.raises(EnvironmentError, match="ANTHROPIC_API_KEY"):
                get_llm()
        finally:
            if original_provider:
                os.environ["LLM_PROVIDER"] = original_provider
            if original_key:
                os.environ["ANTHROPIC_API_KEY"] = original_key

    def test_get_llm_returns_base_chat_model(self):
        """
        get_llm() returns a BaseChatModel instance.
        Tests with a real API key if available, otherwise skips.
        """
        from langchain_core.language_models import BaseChatModel
        from llm import get_llm

        api_key = os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("OPENAI_API_KEY")
        if not api_key:
            pytest.skip("No API key available for LLM instantiation test")

        llm = get_llm()
        assert isinstance(llm, BaseChatModel)

    def test_default_models_defined_for_all_providers(self):
        """All providers have default model names defined."""
        from llm import DEFAULT_MODELS, FAST_MODELS

        expected_providers = {"anthropic", "openai", "groq", "together", "ollama"}
        assert set(DEFAULT_MODELS.keys()) == expected_providers
        assert set(FAST_MODELS.keys()) == expected_providers

        for provider, model in DEFAULT_MODELS.items():
            assert isinstance(model, str) and len(model) > 0, (
                f"Default model for {provider} is empty"
            )


# ═══════════════════════════════════════════════════════════════════════════════
# TEST GROUP 6: End-to-End RAG Flow (no LLM required)
# ═══════════════════════════════════════════════════════════════════════════════

class TestEndToEndRAG:
    """
    End-to-end tests of the full RAG pipeline.
    These tests simulate exactly what the Retrieval Agent does.
    No LLM API key required.
    """

    def test_full_pipeline_scenario_1_pneumonia(self):
        """
        SCENARIO 1 (from MVP success criteria):
        3-year-old with cough, fast breathing, chest indrawing.
        RAG should retrieve IMNCI chest indrawing / pneumonia classification.
        """
        from rag.ingest import ingest_sample_data
        from tools.vector_search import format_for_triage_agent, multi_query_search

        ingest_sample_data()

        # These are the kinds of queries the Retrieval Agent would generate
        # for Scenario 1
        queries = [
            "IMNCI classify cough pneumonia child chest indrawing",
            "chest indrawing danger sign classification child",
            "pediatric pneumonia triage respiratory rate age 1-5",
        ]

        results = multi_query_search(queries, n_results_per_query=3)

        assert len(results) > 0

        # Verify relevant content was retrieved
        combined = " ".join([doc.content.lower() for doc in results])
        assert "chest indrawing" in combined, "Chest indrawing content not retrieved"
        assert "pneumonia" in combined, "Pneumonia content not retrieved"

        # Verify it includes the urgency classification
        assert "urgent" in combined or "refer" in combined, "Referral urgency not in results"

        # Format should be usable by Triage Agent
        formatted = format_for_triage_agent(results[:5])
        assert "GUIDELINE" in formatted
        assert len(formatted) > 100

    def test_full_pipeline_scenario_2_mild_cough(self):
        """
        SCENARIO 2: 4-year-old with simple cough, no danger signs.
        RAG should retrieve 'no pneumonia' classification advice.
        """
        from rag.ingest import ingest_sample_data
        from tools.vector_search import search_guidelines

        ingest_sample_data()

        results = search_guidelines(
            "child cough no fast breathing no chest indrawing",
            n_results=5,
        )

        assert len(results) > 0
        # Should retrieve cough classification content
        combined = " ".join([doc.content.lower() for doc in results])
        assert "cough" in combined

    def test_full_pipeline_scenario_3_diarrhea(self):
        """
        SCENARIO 3: 2-year-old with diarrhea (incomplete description).
        RAG should retrieve dehydration classification content.
        """
        from rag.ingest import ingest_sample_data
        from tools.vector_search import search_guidelines

        ingest_sample_data()

        results = search_guidelines(
            "child diarrhea loose stools dehydration signs",
            n_results=5,
        )

        assert len(results) > 0
        combined = " ".join([doc.content.lower() for doc in results])
        assert "diarrhea" in combined or "dehydration" in combined

    def test_imnci_retrieval_only_returns_imnci(self):
        """
        Age-aware retrieval: pediatric cases should filter to IMNCI.
        This tests the metadata filter used for pediatric patients.
        """
        from rag.ingest import ingest_sample_data
        from tools.vector_search import search_guidelines

        ingest_sample_data()

        # Filter to IMNCI only (as the Retrieval Agent would for a child under 5)
        results = search_guidelines(
            "classify fever child",
            n_results=5,
            guideline_type="imnci",
        )

        for doc in results:
            assert doc.guideline_type == "imnci", (
                f"Expected imnci, got {doc.guideline_type}"
            )

    def test_adult_retrieval_returns_who_pen(self):
        """
        Age-aware retrieval: adult cases should get WHO PEN content.
        """
        from rag.ingest import ingest_sample_data
        from tools.vector_search import search_guidelines

        ingest_sample_data()

        # Filter to who_pen only (as agent would for adult patient)
        results = search_guidelines(
            "blood pressure high hypertension classification",
            n_results=5,
            guideline_type="who_pen",
        )

        if results:  # Skip assertion if no who_pen content available
            for doc in results:
                assert doc.guideline_type == "who_pen"
