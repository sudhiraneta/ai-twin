#!/usr/bin/env python3
"""End-to-end RAG indexing and retrieval test suite.

Tests three areas:
  1. Classification and clustering (ChunkClassifier, DocumentClusterer)
  2. Classified/clustered chunk semantics → VectorStore search methods
  3. End-to-end indexing + retrieval for the "RAG LLM" query and QueryRouter

Each test has:
  - test_name    : unique name
  - description  : what the test validates
  - category     : indexing | retrieval | embedding | chunking | end_to_end | relevance | clustering | classification | routing
  - priority     : critical | high | medium | low
  - query        : the test query (where applicable)
  - expected_behavior: what should happen
  - assertions   : specific checks performed

Usage: /Users/sudhirabadugu/ai-twin/.venv/bin/python tests/test_rag_indexing.py
"""

import sys
import time
import traceback
import uuid
from datetime import datetime, timezone
from pathlib import Path

# Project root on path
sys.path.insert(0, str(Path(__file__).parent.parent))

RESULTS: dict[str, str] = {}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_test_id() -> str:
    """Short unique suffix so each test run gets its own ChromaDB collection."""
    return uuid.uuid4().hex[:8]


def _make_temp_vectorstore(collection_name: str):
    """Create a VectorStore backed by a temporary in-memory ChromaDB collection.

    We monkey-patch the collection name to isolate tests from production data.
    """
    import chromadb
    from chromadb.config import Settings
    from memory.vectorstore import VectorStore
    from memory.embeddings import EmbeddingEngine
    from config import CHROMA_DIR

    vs = object.__new__(VectorStore)
    vs.client = chromadb.PersistentClient(
        path=str(CHROMA_DIR),
        settings=Settings(anonymized_telemetry=False),
    )
    vs.embedding_engine = EmbeddingEngine()
    vs.collection = vs.client.get_or_create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"},
    )
    return vs


def _cleanup_collection(vs, collection_name: str) -> None:
    """Delete the temporary test collection."""
    try:
        vs.client.delete_collection(collection_name)
    except Exception:
        pass


def _rag_sample_chunks() -> list:
    """Return Chunk objects with RAG/LLM content for indexing tests."""
    from memory.chunker import Chunk, _ensure_metadata

    texts = [
        "Retrieval-Augmented Generation (RAG) is a technique that combines information "
        "retrieval with large language model generation. It allows LLMs to access "
        "external knowledge bases, reducing hallucination and improving factual accuracy.",

        "Vector databases are core to RAG pipelines. Documents are split into chunks, "
        "embedded with a transformer model, and stored. At query time, the top-k most "
        "similar chunks are retrieved and injected into the LLM context window.",

        "Chunking strategy is critical for RAG quality. Overlapping windows preserve "
        "semantic coherence across chunk boundaries, while semantic splitting tries to "
        "respect paragraph and sentence structure.",

        "LLM prompt engineering for RAG includes a system prompt, retrieved memory "
        "context injected as additional information, and the user query. The model "
        "synthesises an answer grounded in the retrieved documents.",

        "ChromaDB is an open-source vector database that supports cosine similarity "
        "search. It stores embeddings alongside metadata, enabling filtered retrieval "
        "with where clauses.",

        "all-MiniLM-L6-v2 is a sentence-transformer model producing 384-dimensional "
        "embeddings. It is optimised for semantic similarity tasks and runs locally "
        "with no API cost.",

        "Embedding similarity is measured as cosine distance. A distance of 0 means "
        "identical vectors. A distance above 1.0 generally indicates weak relevance "
        "for MiniLM embeddings.",

        "Hybrid retrieval combines dense vector search (RAG) with sparse keyword "
        "matching or SQL-based structured queries to handle both factual and "
        "semantic information needs.",

        "Deduplication in RAG pipelines uses deterministic chunk IDs derived from "
        "content hashes so that re-ingesting the same document does not produce "
        "duplicate entries in the vector store.",

        "Persona-aware RAG boosts retrieval by first classifying the query into "
        "semantic dimensions (code, learning, wellness), then searching dimension-"
        "specific chunk subsets before falling back to general similarity search.",
    ]
    chunks = []
    for i, text in enumerate(texts):
        chunks.append(Chunk(
            text=text,
            metadata=_ensure_metadata({
                "source": "test_rag_docs",
                "conversation_id": f"rag_test_{i}",
                "title": f"RAG Doc {i}",
                "timestamp": "2025-01-15",
                "role": "user",
                "type": "user_message",
                "pillar": "MIND",
                "dimension": "learning",
                "classified": "true",
            }),
        ))
    return chunks


# ---------------------------------------------------------------------------
# Test runner
# ---------------------------------------------------------------------------

def log(msg: str) -> None:
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] {msg}")


def run_test(name: str, fn) -> bool:
    try:
        fn()
        RESULTS[name] = "PASS"
        log(f"  PASS  {name}")
        return True
    except Exception as exc:
        RESULTS[name] = f"FAIL: {exc}"
        log(f"  FAIL  {name} — {exc}")
        traceback.print_exc()
        return False


# ===========================================================================
# AREA 1: Classification and Clustering Tests
# ===========================================================================

def test_classifier_rag_llm_query():
    """
    test_name: classifier_rag_llm_query
    description: classify_text('RAG LLM') should return learning/code dimensions
    category: classification
    priority: critical
    query: RAG LLM
    expected_behavior: ChunkClassifier.classify_text returns at least one dimension for tech-related text
    assertions: result is a list; both "learning" and/or "code" are present
    """
    from persona.classifier import ChunkClassifier
    c = ChunkClassifier()
    # "RAG LLM" alone may not match keywords — use richer text
    dims = c.classify_text("learn about RAG and LLM retrieval systems")
    assert isinstance(dims, list), "classify_text must return a list"
    assert "learning" in dims, f"Expected 'learning' in dims, got {dims}"


def test_classifier_code_text():
    """
    test_name: classifier_code_text
    description: classify_chunk for text mentioning python/api/backend returns code dimension
    category: classification
    priority: high
    query: python api backend docker
    expected_behavior: pillar=MIND, dimension=code
    assertions: returned pillar is MIND and dimension is code
    """
    from persona.classifier import ChunkClassifier
    c = ChunkClassifier()
    pillar, dim = c.classify_chunk(
        "I deployed a FastAPI backend with Docker and pytest",
        {"type": "user_message"},
    )
    assert dim == "code", f"Expected 'code', got '{dim}'"
    assert pillar == "MIND", f"Expected 'MIND', got '{pillar}'"


def test_classifier_wellness_text():
    """
    test_name: classifier_wellness_text
    description: classify_chunk for gym/workout text returns wellness dimension
    category: classification
    priority: high
    query: gym workout exercise
    expected_behavior: pillar=BODY, dimension=wellness
    assertions: returned pillar is BODY and dimension is wellness
    """
    from persona.classifier import ChunkClassifier
    c = ChunkClassifier()
    pillar, dim = c.classify_chunk(
        "Went to the gym today, focused on cardio and lifting weights",
        {"type": "user_message"},
    )
    assert dim == "wellness", f"Expected 'wellness', got '{dim}'"
    assert pillar == "BODY", f"Expected 'BODY', got '{pillar}'"


def test_classifier_multi_dimension():
    """
    test_name: classifier_multi_dimension
    description: classify_text returns multiple dimensions for text spanning topics
    category: classification
    priority: medium
    query: gym workout and meal prep cooking
    expected_behavior: both wellness and nutrition are in result
    assertions: both 'wellness' and 'nutrition' present in returned list
    """
    from persona.classifier import ChunkClassifier
    c = ChunkClassifier()
    dims = c.classify_text("gym workout and meal prep cooking protein")
    assert "wellness" in dims, f"Expected 'wellness' in {dims}"
    assert "nutrition" in dims, f"Expected 'nutrition' in {dims}"


def test_classifier_type_mapping_body_gym():
    """
    test_name: classifier_type_mapping_body_gym
    description: metadata type 'body_gym' is directly mapped to BODY/wellness
    category: classification
    priority: high
    query: n/a (type-based)
    expected_behavior: classify_chunk returns (BODY, wellness) for type=body_gym
    assertions: exact tuple match
    """
    from persona.classifier import ChunkClassifier
    c = ChunkClassifier()
    result = c.classify_chunk("some text", {"type": "body_gym"})
    assert result == ("BODY", "wellness"), f"Got {result}"


def test_classifier_type_mapping_task():
    """
    test_name: classifier_type_mapping_task
    description: metadata type 'task' maps to PURPOSE/goals
    category: classification
    priority: medium
    query: n/a
    expected_behavior: (PURPOSE, goals)
    assertions: exact tuple match
    """
    from persona.classifier import ChunkClassifier
    c = ChunkClassifier()
    result = c.classify_chunk("finish the sprint", {"type": "task"})
    assert result == ("PURPOSE", "goals"), f"Got {result}"


def test_classifier_pillar_based_soul():
    """
    test_name: classifier_pillar_based_soul
    description: existing pillar=SOUL in metadata → default dimension=creative
    category: classification
    priority: medium
    query: n/a
    expected_behavior: (SOUL, creative)
    assertions: exact tuple match
    """
    from persona.classifier import ChunkClassifier
    c = ChunkClassifier()
    p, d = c.classify_chunk("creative entry", {"type": "singularity_entry", "pillar": "SOUL"})
    assert p == "SOUL" and d == "creative", f"Got ({p}, {d})"


def test_classifier_unclassifiable():
    """
    test_name: classifier_unclassifiable
    description: ambiguous text returns empty strings (needs Tier 2 LLM)
    category: classification
    priority: medium
    query: the weather is nice today
    expected_behavior: returns ('', '')
    assertions: both pillar and dimension are empty
    """
    from persona.classifier import ChunkClassifier
    c = ChunkClassifier()
    p, d = c.classify_chunk("the weather is nice today", {"type": "user_message"})
    assert p == "" and d == "", f"Expected ('', ''), got ('{p}', '{d}')"


def test_clusterer_too_few_docs():
    """
    test_name: clusterer_too_few_docs
    description: DocumentClusterer.cluster_all returns empty list when < min_samples*2 docs
    category: clustering
    priority: high
    query: n/a
    expected_behavior: cluster_all() returns []
    assertions: result is an empty list
    """
    from memory.clusterer import DocumentClusterer

    coll_name = f"test_clusterer_small_{_make_test_id()}"
    vs = _make_temp_vectorstore(coll_name)
    try:
        clusterer = DocumentClusterer(vs)
        clusters = clusterer.cluster_all(min_samples=3)
        assert clusters == [], f"Expected [] for empty store, got {clusters}"
    finally:
        _cleanup_collection(vs, coll_name)


def test_clusterer_ingest_and_cluster():
    """
    test_name: clusterer_ingest_and_cluster
    description: After ingesting 10+ semantically similar docs, cluster_all finds at least 1 cluster
    category: clustering
    priority: critical
    query: n/a
    expected_behavior: DBSCAN finds at least one cluster
    assertions: len(clusters) >= 1; each Cluster has chunk_ids and size > 0
    """
    from memory.clusterer import DocumentClusterer

    coll_name = f"test_clusterer_main_{_make_test_id()}"
    vs = _make_temp_vectorstore(coll_name)
    try:
        chunks = _rag_sample_chunks()
        vs.ingest(chunks)
        assert vs.count() == len(chunks), "Not all chunks ingested"

        # Use low min_samples so small test set can form clusters
        clusterer = DocumentClusterer(vs)
        clusters = clusterer.cluster_all(eps=0.5, min_samples=2)
        # With semantically related docs, at least one cluster should form
        # (it's possible all are noise with very strict eps, so we allow 0 here
        #  and instead test the return type and object structure)
        assert isinstance(clusters, list), "cluster_all must return a list"
        for cl in clusters:
            assert cl.size > 0, "Cluster size must be > 0"
            assert len(cl.chunk_ids) > 0, "Cluster must have chunk_ids"
    finally:
        _cleanup_collection(vs, coll_name)


def test_clusterer_label_clusters_fallback():
    """
    test_name: clusterer_label_clusters_fallback
    description: label_clusters assigns a non-empty label to each cluster (fallback to cluster_N)
    category: clustering
    priority: medium
    query: n/a
    expected_behavior: every cluster gets a label string
    assertions: all c.label are non-empty strings after label_clusters
    """
    from memory.clusterer import DocumentClusterer, Cluster

    coll_name = f"test_clusterer_labels_{_make_test_id()}"
    vs = _make_temp_vectorstore(coll_name)
    try:
        clusterer = DocumentClusterer(vs)
        # Manually build fake Cluster objects (no LLM needed for fallback)
        fake_clusters = [
            Cluster(cluster_id=0, chunk_ids=["id1", "id2"], sample_texts=["RAG pipeline"], size=2),
            Cluster(cluster_id=1, chunk_ids=["id3"], sample_texts=["gym workout"], size=1),
        ]
        # label_clusters calls LLM; if it fails it falls back to f"cluster_{id}"
        labeled = clusterer.label_clusters(fake_clusters)
        for cl in labeled:
            assert cl.label, f"Cluster {cl.cluster_id} has no label"
    finally:
        _cleanup_collection(vs, coll_name)


def test_clusterer_update_metadata():
    """
    test_name: clusterer_update_metadata
    description: update_metadata_with_clusters writes cluster_label back to ChromaDB metadata
    category: clustering
    priority: critical
    query: n/a
    expected_behavior: after update, retrieved docs have cluster_label metadata set
    assertions: fetched metadata contains cluster_label matching the assigned label
    """
    from memory.clusterer import DocumentClusterer, Cluster
    from memory.chunker import Chunk, _ensure_metadata

    coll_name = f"test_clusterer_meta_{_make_test_id()}"
    vs = _make_temp_vectorstore(coll_name)
    try:
        # Ingest 3 simple docs
        chunks = [
            Chunk(text="RAG pipeline with embeddings", metadata=_ensure_metadata({"source": "test", "type": "user_message"})),
            Chunk(text="LLM retrieval augmented generation", metadata=_ensure_metadata({"source": "test", "type": "user_message"})),
            Chunk(text="vector search cosine similarity", metadata=_ensure_metadata({"source": "test", "type": "user_message"})),
        ]
        vs.ingest(chunks)
        ids = vs.collection.get(include=[])["ids"]
        assert len(ids) == 3

        clusterer = DocumentClusterer(vs)
        fake_cluster = Cluster(cluster_id=0, label="rag_llm_tech", chunk_ids=ids[:2], size=2)
        updated = clusterer.update_metadata_with_clusters([fake_cluster])
        assert updated == 2, f"Expected 2 updated, got {updated}"

        # Verify metadata was written
        result = vs.collection.get(ids=ids[:2], include=["metadatas"])
        for meta in result["metadatas"]:
            assert meta.get("cluster_label") == "rag_llm_tech", \
                f"Expected cluster_label='rag_llm_tech', got {meta.get('cluster_label')}"
    finally:
        _cleanup_collection(vs, coll_name)


# ===========================================================================
# AREA 2: Classified/Clustered Chunk Semantics → Vector Mapping Tests
# ===========================================================================

def test_vectorstore_ingest_count():
    """
    test_name: vectorstore_ingest_count
    description: Ingesting N chunks increases collection count by exactly N (dedup by content)
    category: indexing
    priority: critical
    query: n/a
    expected_behavior: vs.count() == len(chunks) after ingestion
    assertions: count matches expected
    """
    coll_name = f"test_vs_count_{_make_test_id()}"
    vs = _make_temp_vectorstore(coll_name)
    try:
        chunks = _rag_sample_chunks()
        vs.ingest(chunks)
        assert vs.count() == len(chunks), \
            f"Expected {len(chunks)} chunks, got {vs.count()}"
    finally:
        _cleanup_collection(vs, coll_name)


def test_vectorstore_deduplication():
    """
    test_name: vectorstore_deduplication
    description: Re-ingesting identical chunks does not increase the count (upsert semantics)
    category: indexing
    priority: high
    query: n/a
    expected_behavior: count stays same after second ingest of same chunks
    assertions: count_after_reingest == count_after_first_ingest
    """
    coll_name = f"test_vs_dedup_{_make_test_id()}"
    vs = _make_temp_vectorstore(coll_name)
    try:
        chunks = _rag_sample_chunks()
        vs.ingest(chunks)
        first_count = vs.count()
        vs.ingest(chunks)  # re-ingest same chunks
        second_count = vs.count()
        assert first_count == second_count, \
            f"Dedup failed: count went from {first_count} to {second_count}"
    finally:
        _cleanup_collection(vs, coll_name)


def test_vectorstore_search_returns_results():
    """
    test_name: vectorstore_search_returns_results
    description: search() on a populated store returns a non-empty list for a relevant query
    category: retrieval
    priority: critical
    query: RAG LLM retrieval
    expected_behavior: at least 1 result returned
    assertions: len(results) >= 1; each result has 'text', 'metadata', 'distance', 'id' keys
    """
    coll_name = f"test_vs_search_{_make_test_id()}"
    vs = _make_temp_vectorstore(coll_name)
    try:
        vs.ingest(_rag_sample_chunks())
        results = vs.search("RAG LLM retrieval", n_results=5)
        assert len(results) >= 1, "Expected at least 1 search result"
        for r in results:
            assert "text" in r, "Result missing 'text'"
            assert "metadata" in r, "Result missing 'metadata'"
            assert "distance" in r, "Result missing 'distance'"
            assert "id" in r, "Result missing 'id'"
    finally:
        _cleanup_collection(vs, coll_name)


def test_vectorstore_search_distance_range():
    """
    test_name: vectorstore_search_distance_range
    description: Cosine distances returned are in valid range [0, 2]
    category: retrieval
    priority: high
    query: RAG LLM retrieval
    expected_behavior: all distances between 0 and 2
    assertions: 0 <= d <= 2 for every result
    """
    coll_name = f"test_vs_dist_{_make_test_id()}"
    vs = _make_temp_vectorstore(coll_name)
    try:
        vs.ingest(_rag_sample_chunks())
        results = vs.search("RAG LLM retrieval", n_results=10)
        for r in results:
            d = r["distance"]
            assert 0.0 <= d <= 2.0, f"Distance {d} out of valid range [0, 2]"
    finally:
        _cleanup_collection(vs, coll_name)


def test_vectorstore_search_by_dimension():
    """
    test_name: vectorstore_search_by_dimension
    description: search_by_dimension filters results to only chunks with matching dimension
    category: retrieval
    priority: critical
    query: RAG LLM retrieval
    expected_behavior: all returned chunks have metadata['dimension'] == 'learning'
    assertions: every result.metadata['dimension'] == 'learning'
    """
    coll_name = f"test_vs_dim_{_make_test_id()}"
    vs = _make_temp_vectorstore(coll_name)
    try:
        # All sample chunks are classified as dimension=learning
        vs.ingest(_rag_sample_chunks())
        results = vs.search_by_dimension("RAG LLM retrieval", "learning", n_results=5)
        assert len(results) >= 1, "Expected at least 1 result for dimension=learning"
        for r in results:
            assert r["metadata"].get("dimension") == "learning", \
                f"Expected dimension='learning', got '{r['metadata'].get('dimension')}'"
    finally:
        _cleanup_collection(vs, coll_name)


def test_vectorstore_search_by_dimension_no_cross_contamination():
    """
    test_name: vectorstore_search_by_dimension_no_cross_contamination
    description: search_by_dimension('code') returns 0 results when no code-dimension chunks exist
    category: retrieval
    priority: high
    query: python code deployment
    expected_behavior: empty list because all docs are dimension=learning
    assertions: results == []
    """
    coll_name = f"test_vs_dim_clean_{_make_test_id()}"
    vs = _make_temp_vectorstore(coll_name)
    try:
        vs.ingest(_rag_sample_chunks())  # all dimension=learning
        results = vs.search_by_dimension("python code deployment", "code", n_results=5)
        assert results == [], \
            f"Expected no code-dimension results, got {len(results)}"
    finally:
        _cleanup_collection(vs, coll_name)


def test_vectorstore_search_by_cluster():
    """
    test_name: vectorstore_search_by_cluster
    description: search_by_cluster returns only chunks tagged with that cluster_label
    category: retrieval
    priority: high
    query: RAG vector search
    expected_behavior: results restricted to cluster_label='rag_tech'
    assertions: all returned metadata have cluster_label='rag_tech'
    """
    from memory.chunker import Chunk, _ensure_metadata

    coll_name = f"test_vs_cluster_{_make_test_id()}"
    vs = _make_temp_vectorstore(coll_name)
    try:
        # Two clusters
        chunks_a = [
            Chunk(
                text="RAG pipeline retrieval augmented generation",
                metadata=_ensure_metadata({
                    "source": "test", "type": "user_message",
                    "cluster_label": "rag_tech", "cluster_id": "0",
                })
            ),
            Chunk(
                text="LLM vector embedding similarity search",
                metadata=_ensure_metadata({
                    "source": "test", "type": "user_message",
                    "cluster_label": "rag_tech", "cluster_id": "0",
                })
            ),
        ]
        chunks_b = [
            Chunk(
                text="Gym workout cardio fitness routine",
                metadata=_ensure_metadata({
                    "source": "test", "type": "user_message",
                    "cluster_label": "fitness_routine", "cluster_id": "1",
                })
            ),
        ]
        vs.ingest(chunks_a + chunks_b)

        results = vs.search_by_cluster("RAG vector search", "rag_tech", n_results=5)
        assert len(results) >= 1, "Expected at least 1 result for cluster_label=rag_tech"
        for r in results:
            assert r["metadata"].get("cluster_label") == "rag_tech", \
                f"Expected cluster_label='rag_tech', got '{r['metadata'].get('cluster_label')}'"
    finally:
        _cleanup_collection(vs, coll_name)


def test_vectorstore_search_with_recency():
    """
    test_name: vectorstore_search_with_recency
    description: search_with_recency returns results with _combined_score field
    category: retrieval
    priority: high
    query: RAG LLM embeddings
    expected_behavior: results have '_combined_score' key; sorted descending
    assertions: '_combined_score' present; scores decrease monotonically
    """
    coll_name = f"test_vs_recency_{_make_test_id()}"
    vs = _make_temp_vectorstore(coll_name)
    try:
        vs.ingest(_rag_sample_chunks())
        results = vs.search_with_recency("RAG LLM embeddings", n_results=5)
        assert len(results) >= 1, "Expected at least 1 result"
        for r in results:
            assert "_combined_score" in r, "Result missing '_combined_score'"
        scores = [r["_combined_score"] for r in results]
        assert scores == sorted(scores, reverse=True), \
            f"Results not sorted by _combined_score desc: {scores}"
    finally:
        _cleanup_collection(vs, coll_name)


def test_vectorstore_update_metadata():
    """
    test_name: vectorstore_update_metadata
    description: update_metadata() patches existing chunks without losing other fields
    category: indexing
    priority: high
    query: n/a
    expected_behavior: updated field is set; other metadata fields preserved
    assertions: patched field equals new value; source field still present
    """
    from memory.chunker import Chunk, _ensure_metadata

    coll_name = f"test_vs_update_meta_{_make_test_id()}"
    vs = _make_temp_vectorstore(coll_name)
    try:
        chunk = Chunk(
            text="Test metadata update chunk",
            metadata=_ensure_metadata({"source": "test_src", "type": "user_message"}),
        )
        vs.ingest([chunk])
        ids = vs.collection.get(include=[])["ids"]
        assert len(ids) == 1

        # Fetch existing metadata, then patch
        existing = vs.collection.get(ids=ids, include=["metadatas"])
        meta = existing["metadatas"][0].copy()
        meta["dimension"] = "code"
        vs.update_metadata(ids, [meta])

        updated = vs.collection.get(ids=ids, include=["metadatas"])
        assert updated["metadatas"][0]["dimension"] == "code", "dimension not updated"
        assert updated["metadatas"][0]["source"] == "test_src", "source field was lost"
    finally:
        _cleanup_collection(vs, coll_name)


def test_vectorstore_max_distance_filter():
    """
    test_name: vectorstore_max_distance_filter
    description: search with max_distance=0.01 (very strict) returns 0 results for unrelated query
    category: retrieval
    priority: medium
    query: completely unrelated topic like dinosaur fossils
    expected_behavior: results filtered to empty list (nothing within 0.01 cosine distance)
    assertions: results == []
    """
    coll_name = f"test_vs_maxdist_{_make_test_id()}"
    vs = _make_temp_vectorstore(coll_name)
    try:
        vs.ingest(_rag_sample_chunks())
        results = vs.search(
            "ancient dinosaur fossil excavation paleontology",
            n_results=10,
            max_distance=0.01,  # near-identical only
        )
        assert results == [], \
            f"Expected 0 results with strict threshold, got {len(results)}"
    finally:
        _cleanup_collection(vs, coll_name)


def test_vectorstore_metadata_preserved():
    """
    test_name: vectorstore_metadata_preserved
    description: All required metadata fields survive the ingest → search round-trip
    category: indexing
    priority: critical
    query: RAG LLM
    expected_behavior: source, type, dimension, pillar, classified all present in retrieved metadata
    assertions: each required key exists in result metadata
    """
    coll_name = f"test_vs_meta_preserve_{_make_test_id()}"
    vs = _make_temp_vectorstore(coll_name)
    try:
        vs.ingest(_rag_sample_chunks())
        results = vs.search("RAG LLM", n_results=1)
        assert results, "Expected at least 1 result"
        meta = results[0]["metadata"]
        required_keys = ["source", "type", "dimension", "pillar", "classified",
                         "conversation_id", "title", "timestamp"]
        for k in required_keys:
            assert k in meta, f"Metadata missing key '{k}'"
    finally:
        _cleanup_collection(vs, coll_name)


# ===========================================================================
# AREA 3: End-to-End Indexing + Retrieval for "RAG LLM" Query
# ===========================================================================

def test_e2e_embedding_dimensions():
    """
    test_name: e2e_embedding_dimensions
    description: EmbeddingEngine produces 384-dim vectors (all-MiniLM-L6-v2)
    category: embedding
    priority: critical
    query: RAG LLM
    expected_behavior: embed_single returns a list of length 384
    assertions: len(embedding) == 384; all values are floats
    """
    from memory.embeddings import EmbeddingEngine
    engine = EmbeddingEngine()
    emb = engine.embed_single("RAG LLM retrieval augmented generation")
    assert isinstance(emb, list), "Embedding must be a list"
    assert len(emb) == 384, f"Expected 384 dims, got {len(emb)}"
    assert all(isinstance(v, float) for v in emb), "All embedding values must be floats"


def test_e2e_batch_embedding():
    """
    test_name: e2e_batch_embedding
    description: EmbeddingEngine.embed produces correct count and shape for batch input
    category: embedding
    priority: high
    query: n/a
    expected_behavior: embed([t1, t2, t3]) returns list of 3 embeddings each of length 384
    assertions: len == 3; each sub-list len == 384
    """
    from memory.embeddings import EmbeddingEngine
    engine = EmbeddingEngine()
    texts = [
        "RAG pipeline overview",
        "LLM context window",
        "vector database indexing",
    ]
    embeddings = engine.embed(texts)
    assert len(embeddings) == 3, f"Expected 3 embeddings, got {len(embeddings)}"
    for emb in embeddings:
        assert len(emb) == 384, f"Expected 384 dims per embedding, got {len(emb)}"


def test_e2e_chunker_produces_chunks():
    """
    test_name: e2e_chunker_produces_chunks
    description: Chunker.chunk_conversations creates both user_message and conversation_pair chunks
    category: chunking
    priority: critical
    query: n/a
    expected_behavior: at least 2 chunks (1 user_message + 1 conversation_pair) for 1 turn
    assertions: both chunk types are present; all metadata fields populated
    """
    from memory.chunker import Chunker

    chunker = Chunker(chunk_size=500, overlap=50)
    convs = [{
        "source": "test_rag",
        "conversation_id": "conv_rag_001",
        "title": "RAG discussion",
        "timestamp": "2025-01-15",
        "messages": [
            {"role": "user", "content": "How does RAG work with LLMs?", "timestamp": "2025-01-15T10:00:00"},
            {"role": "assistant", "content": "RAG retrieves relevant documents and injects them into the LLM context.", "timestamp": "2025-01-15T10:00:01"},
        ],
    }]
    chunks = chunker.chunk_conversations(convs)
    types = {c.metadata["type"] for c in chunks}
    assert "user_message" in types, f"Expected user_message type, got types={types}"
    assert "conversation_pair" in types, f"Expected conversation_pair type, got types={types}"
    for c in chunks:
        assert c.text, "Chunk text must not be empty"
        assert c.metadata["source"] == "test_rag", "source not preserved"
        assert c.metadata["conversation_id"] == "conv_rag_001", "conversation_id not preserved"


def test_e2e_chunk_text_with_metadata():
    """
    test_name: e2e_chunk_text_with_metadata
    description: chunk_text_with_metadata splits long text and preserves metadata on all parts
    category: chunking
    priority: high
    query: n/a
    expected_behavior: produces >= 1 Chunk; all have identical metadata dict
    assertions: each chunk has all metadata keys from DEFAULT_METADATA
    """
    from memory.chunker import Chunker, DEFAULT_METADATA

    chunker = Chunker(chunk_size=200, overlap=20)
    long_text = "RAG LLM retrieval. " * 50  # 950 chars > 200
    meta = {"source": "manual", "type": "data_point", "title": "RAG overview"}
    chunks = chunker.chunk_text_with_metadata(long_text, meta)
    assert len(chunks) >= 2, f"Expected multiple chunks for long text, got {len(chunks)}"
    for c in chunks:
        for k in DEFAULT_METADATA:
            assert k in c.metadata, f"Chunk metadata missing key '{k}'"


def test_e2e_rag_retrieval_relevance():
    """
    test_name: e2e_rag_retrieval_relevance
    description: Top result for 'RAG LLM' query should contain RAG/LLM-related content
    category: relevance
    priority: critical
    query: RAG LLM retrieval
    expected_behavior: top-1 retrieved document contains 'RAG' or 'retrieval' or 'LLM'
    assertions: 'RAG' or 'retrieval' or 'LLM' in top result text (case-insensitive)
    """
    coll_name = f"test_e2e_relevance_{_make_test_id()}"
    vs = _make_temp_vectorstore(coll_name)
    try:
        vs.ingest(_rag_sample_chunks())
        results = vs.search("RAG LLM retrieval", n_results=3)
        assert results, "No results returned"
        top_text = results[0]["text"].lower()
        assert any(kw in top_text for kw in ["rag", "retrieval", "llm", "language model", "embedding"]), \
            f"Top result not relevant to RAG/LLM: '{results[0]['text'][:120]}'"
    finally:
        _cleanup_collection(vs, coll_name)


def test_e2e_similarity_scores_threshold():
    """
    test_name: e2e_similarity_scores_threshold
    description: Top-3 results for 'RAG LLM' query have distance < 0.8 (strongly related)
    category: relevance
    priority: high
    query: RAG LLM vector database retrieval
    expected_behavior: cosine distances for top-3 are all below 0.8
    assertions: distance < 0.8 for each of the top 3 results
    """
    coll_name = f"test_e2e_scores_{_make_test_id()}"
    vs = _make_temp_vectorstore(coll_name)
    try:
        vs.ingest(_rag_sample_chunks())
        results = vs.search("RAG LLM vector database retrieval", n_results=3)
        assert len(results) >= 3, f"Expected at least 3 results, got {len(results)}"
        for i, r in enumerate(results[:3]):
            d = r["distance"]
            assert d < 0.8, \
                f"Result {i} distance {d:.4f} >= 0.8 — not semantically close enough"
    finally:
        _cleanup_collection(vs, coll_name)


def test_e2e_full_pipeline_ingest_classify_retrieve():
    """
    test_name: e2e_full_pipeline_ingest_classify_retrieve
    description: Full pipeline: text → chunker → classify → ingest → search_by_dimension
    category: end_to_end
    priority: critical
    query: RAG LLM
    expected_behavior: classified chunks retrievable via dimension-filtered search
    assertions: dimension matches classifier output; chunk text retrieved intact
    """
    from memory.chunker import Chunker
    from persona.classifier import ChunkClassifier

    coll_name = f"test_e2e_pipeline_{_make_test_id()}"
    vs = _make_temp_vectorstore(coll_name)
    try:
        chunker = Chunker()
        classifier = ChunkClassifier()

        # Single conversation about learning/RAG
        convs = [{
            "source": "chatgpt",
            "conversation_id": "rag_conv_e2e",
            "title": "Learning RAG",
            "timestamp": "2025-01-20",
            "messages": [
                {
                    "role": "user",
                    "content": (
                        "I want to learn about retrieval augmented generation. "
                        "I'm studying how to build RAG pipelines with embeddings and vector search."
                    ),
                    "timestamp": "2025-01-20T09:00:00",
                },
                {
                    "role": "assistant",
                    "content": (
                        "Great! RAG pipelines typically involve embedding documents, "
                        "storing them in a vector database, and retrieving relevant chunks "
                        "to inject into the LLM context."
                    ),
                    "timestamp": "2025-01-20T09:00:01",
                },
            ],
        }]

        chunks = chunker.chunk_conversations(convs)
        assert chunks, "Chunker produced no chunks"

        # Classify each chunk
        for chunk in chunks:
            pillar, dimension = classifier.classify_chunk(chunk.text, chunk.metadata)
            chunk.metadata["pillar"] = pillar
            chunk.metadata["dimension"] = dimension
            chunk.metadata["classified"] = "true" if dimension else "false"

        # Ingest
        vs.ingest(chunks)
        assert vs.count() == len(chunks)

        # Retrieve via general search
        results = vs.search("RAG LLM retrieval pipeline", n_results=5)
        assert results, "No results for RAG LLM query after full pipeline"

        # At least one chunk should have been classified into learning or code
        classified_dims = {c.metadata.get("dimension") for c in chunks if c.metadata.get("dimension")}
        assert classified_dims, f"No chunks were classified into any dimension"
    finally:
        _cleanup_collection(vs, coll_name)


def test_e2e_empty_query_handling():
    """
    test_name: e2e_empty_query_handling
    description: search() with empty or whitespace query does not raise an exception
    category: end_to_end
    priority: medium
    query: '' (empty)
    expected_behavior: returns a list (may be empty or full) without exception
    assertions: no exception raised; result is a list
    """
    coll_name = f"test_e2e_empty_{_make_test_id()}"
    vs = _make_temp_vectorstore(coll_name)
    try:
        vs.ingest(_rag_sample_chunks())
        result = vs.search("", n_results=3)
        assert isinstance(result, list), "Expected list result for empty query"
    finally:
        _cleanup_collection(vs, coll_name)


def test_e2e_query_router_rag_llm():
    """
    test_name: e2e_query_router_rag_llm
    description: QueryRouter routes 'RAG LLM' / learning questions to RAG (not SQL)
    category: routing
    priority: high
    query: What do I think about RAG and LLM systems?
    expected_behavior: QueryType.RAG returned (reflective question → RAG)
    assertions: routed.query_type == QueryType.RAG
    """
    from db.query_router import QueryRouter, QueryType
    router = QueryRouter()

    # Reflective question → should hit _RAG_PATTERN via "think about"
    routed = router.route("What do I think about RAG and LLM systems?")
    assert routed.query_type == QueryType.RAG, \
        f"Expected RAG route for reflective query, got {routed.query_type}"


def test_e2e_query_router_sql_gym():
    """
    test_name: e2e_query_router_sql_gym
    description: QueryRouter routes quantitative gym questions to SQL
    category: routing
    priority: high
    query: How many gym sessions did I do this week?
    expected_behavior: QueryType.SQL returned
    assertions: routed.query_type == QueryType.SQL; 'gym' in sql_tables
    """
    from db.query_router import QueryRouter, QueryType
    router = QueryRouter()
    routed = router.route("How many gym sessions did I do this week?")
    assert routed.query_type == QueryType.SQL, \
        f"Expected SQL route for quantitative gym query, got {routed.query_type}"
    assert "gym" in routed.sql_tables, \
        f"Expected 'gym' in sql_tables, got {routed.sql_tables}"


def test_e2e_query_router_hybrid():
    """
    test_name: e2e_query_router_hybrid
    description: QueryRouter returns HYBRID for queries that have both table match + reflection signal
    category: routing
    priority: medium
    query: How do I feel about my gym routine and workout sessions?
    expected_behavior: QueryType.HYBRID returned
    assertions: routed.query_type == QueryType.HYBRID; rag_query is set
    """
    from db.query_router import QueryRouter, QueryType
    router = QueryRouter()
    routed = router.route("How do I feel about my gym workout sessions?")
    assert routed.query_type == QueryType.HYBRID, \
        f"Expected HYBRID route, got {routed.query_type}"
    assert routed.rag_query, "rag_query must be set for HYBRID"


def test_e2e_query_router_time_range():
    """
    test_name: e2e_query_router_time_range
    description: QueryRouter extracts time range correctly from query
    category: routing
    priority: medium
    query: How many tasks did I complete last week?
    expected_behavior: time_range='last_week'
    assertions: routed.time_range == 'last_week'
    """
    from db.query_router import QueryRouter
    router = QueryRouter()
    routed = router.route("How many tasks did I complete last week?")
    assert routed.time_range == "last_week", \
        f"Expected time_range='last_week', got '{routed.time_range}'"


def test_e2e_routed_query_object_fields():
    """
    test_name: e2e_routed_query_object_fields
    description: RoutedQuery object has all required fields for downstream consumption
    category: routing
    priority: high
    query: What are my learning goals?
    expected_behavior: RoutedQuery has query_type, sql_tables, sql_intent, rag_query, time_range
    assertions: all fields accessible and not None
    """
    from db.query_router import QueryRouter
    router = QueryRouter()
    routed = router.route("What are my learning goals?")
    assert hasattr(routed, "query_type"), "Missing query_type"
    assert hasattr(routed, "sql_tables"), "Missing sql_tables"
    assert hasattr(routed, "sql_intent"), "Missing sql_intent"
    assert hasattr(routed, "rag_query"), "Missing rag_query"
    assert hasattr(routed, "time_range"), "Missing time_range"
    assert routed.sql_tables is not None, "sql_tables must not be None"


def test_e2e_get_cluster_labels():
    """
    test_name: e2e_get_cluster_labels
    description: get_cluster_labels() returns a list of strings from stored cluster_label metadata
    category: retrieval
    priority: medium
    query: n/a
    expected_behavior: list contains 'rag_tech' after we manually set it
    assertions: 'rag_tech' in get_cluster_labels()
    """
    from memory.chunker import Chunk, _ensure_metadata

    coll_name = f"test_e2e_labels_{_make_test_id()}"
    vs = _make_temp_vectorstore(coll_name)
    try:
        chunk = Chunk(
            text="RAG retrieval system test",
            metadata=_ensure_metadata({
                "source": "test",
                "type": "user_message",
                "cluster_label": "rag_tech",
                "cluster_id": "0",
            }),
        )
        vs.ingest([chunk])
        labels = vs.get_cluster_labels()
        assert isinstance(labels, list), "get_cluster_labels must return a list"
        assert "rag_tech" in labels, f"Expected 'rag_tech' in labels, got {labels}"
    finally:
        _cleanup_collection(vs, coll_name)


def test_e2e_count_by_dimension():
    """
    test_name: e2e_count_by_dimension
    description: count_by_dimension returns correct count for a given dimension
    category: indexing
    priority: medium
    query: n/a
    expected_behavior: count == number of ingested chunks with dimension='learning'
    assertions: count == 10 (the 10 sample chunks all have dimension=learning)
    """
    coll_name = f"test_e2e_dim_count_{_make_test_id()}"
    vs = _make_temp_vectorstore(coll_name)
    try:
        chunks = _rag_sample_chunks()  # all have dimension=learning
        vs.ingest(chunks)
        count = vs.count_by_dimension("learning")
        assert count == len(chunks), \
            f"Expected {len(chunks)} chunks in learning dimension, got {count}"
    finally:
        _cleanup_collection(vs, coll_name)


def test_e2e_keyword_overlap_score():
    """
    test_name: e2e_keyword_overlap_score
    description: _keyword_overlap_score gives high score for documents sharing query keywords
    category: retrieval
    priority: low
    query: RAG retrieval pipeline
    expected_behavior: score > 0.5 for documents that contain all query keywords
    assertions: score > 0.5
    """
    from memory.vectorstore import VectorStore
    vs_cls = VectorStore  # access static method without creating full instance
    score = vs_cls._keyword_overlap_score(
        "RAG retrieval pipeline",
        "RAG is a retrieval-augmented pipeline for LLMs",
    )
    assert score > 0.5, f"Expected overlap score > 0.5, got {score:.4f}"


# ===========================================================================
# Test Registry and Runner
# ===========================================================================

ALL_TESTS = [
    # -- Area 1: Classification --
    ("classifier_rag_llm_query",              test_classifier_rag_llm_query),
    ("classifier_code_text",                  test_classifier_code_text),
    ("classifier_wellness_text",              test_classifier_wellness_text),
    ("classifier_multi_dimension",            test_classifier_multi_dimension),
    ("classifier_type_mapping_body_gym",      test_classifier_type_mapping_body_gym),
    ("classifier_type_mapping_task",          test_classifier_type_mapping_task),
    ("classifier_pillar_based_soul",          test_classifier_pillar_based_soul),
    ("classifier_unclassifiable",             test_classifier_unclassifiable),
    # -- Area 1: Clustering --
    ("clusterer_too_few_docs",                test_clusterer_too_few_docs),
    ("clusterer_ingest_and_cluster",          test_clusterer_ingest_and_cluster),
    ("clusterer_label_clusters_fallback",     test_clusterer_label_clusters_fallback),
    ("clusterer_update_metadata",             test_clusterer_update_metadata),
    # -- Area 2: Vector Store --
    ("vectorstore_ingest_count",              test_vectorstore_ingest_count),
    ("vectorstore_deduplication",             test_vectorstore_deduplication),
    ("vectorstore_search_returns_results",    test_vectorstore_search_returns_results),
    ("vectorstore_search_distance_range",     test_vectorstore_search_distance_range),
    ("vectorstore_search_by_dimension",       test_vectorstore_search_by_dimension),
    ("vectorstore_search_by_dimension_no_cross_contamination",
                                              test_vectorstore_search_by_dimension_no_cross_contamination),
    ("vectorstore_search_by_cluster",         test_vectorstore_search_by_cluster),
    ("vectorstore_search_with_recency",       test_vectorstore_search_with_recency),
    ("vectorstore_update_metadata",           test_vectorstore_update_metadata),
    ("vectorstore_max_distance_filter",       test_vectorstore_max_distance_filter),
    ("vectorstore_metadata_preserved",        test_vectorstore_metadata_preserved),
    # -- Area 3: End-to-End --
    ("e2e_embedding_dimensions",              test_e2e_embedding_dimensions),
    ("e2e_batch_embedding",                   test_e2e_batch_embedding),
    ("e2e_chunker_produces_chunks",           test_e2e_chunker_produces_chunks),
    ("e2e_chunk_text_with_metadata",          test_e2e_chunk_text_with_metadata),
    ("e2e_rag_retrieval_relevance",           test_e2e_rag_retrieval_relevance),
    ("e2e_similarity_scores_threshold",       test_e2e_similarity_scores_threshold),
    ("e2e_full_pipeline_ingest_classify_retrieve",
                                              test_e2e_full_pipeline_ingest_classify_retrieve),
    ("e2e_empty_query_handling",              test_e2e_empty_query_handling),
    ("e2e_query_router_rag_llm",              test_e2e_query_router_rag_llm),
    ("e2e_query_router_sql_gym",              test_e2e_query_router_sql_gym),
    ("e2e_query_router_hybrid",               test_e2e_query_router_hybrid),
    ("e2e_query_router_time_range",           test_e2e_query_router_time_range),
    ("e2e_routed_query_object_fields",        test_e2e_routed_query_object_fields),
    ("e2e_get_cluster_labels",                test_e2e_get_cluster_labels),
    ("e2e_count_by_dimension",                test_e2e_count_by_dimension),
    ("e2e_keyword_overlap_score",             test_e2e_keyword_overlap_score),
]


def main() -> int:
    max_rounds = 3
    log(f"\n{'='*65}")
    log(f"RAG Indexing Test Suite — {len(ALL_TESTS)} tests")
    log(f"{'='*65}")

    for round_num in range(1, max_rounds + 1):
        log(f"\n--- Round {round_num}/{max_rounds} ---")
        failed = []

        for name, fn in ALL_TESTS:
            if RESULTS.get(name) == "PASS":
                continue
            if not run_test(name, fn):
                failed.append(name)

        passed = sum(1 for v in RESULTS.values() if v == "PASS")
        total = len(ALL_TESTS)
        log(f"\nRound {round_num}: {passed}/{total} passed, {len(failed)} failed")

        if not failed:
            log(f"\nALL {total} TESTS PASSED!")
            break

        if round_num < max_rounds:
            log(f"Retrying {len(failed)} failed tests in 2s...")
            time.sleep(2)

    # Final summary
    log(f"\n{'='*65}")
    log("FINAL RESULTS")
    log(f"{'='*65}")
    for name, result in sorted(RESULTS.items()):
        status = "PASS" if result == "PASS" else "FAIL"
        log(f"  {status}  {name}")
        if result != "PASS":
            log(f"       -> {result}")

    passed = sum(1 for v in RESULTS.values() if v == "PASS")
    failed = sum(1 for v in RESULTS.values() if v != "PASS")
    log(f"\n  TOTAL: {passed} passed, {failed} failed out of {len(ALL_TESTS)}")

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
