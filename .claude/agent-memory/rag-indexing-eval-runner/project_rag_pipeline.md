---
name: rag_pipeline_findings
description: Key facts about the RAG indexing/retrieval pipeline discovered during test authoring
type: project
---

ChromaDB collection name is `ai_twin_memory` (from config.CHROMA_COLLECTION). The persistent DB path is `data/chroma_db`.

Embedding model is `all-MiniLM-L6-v2` (sentence-transformers), producing **384-dimensional** vectors. Model loads lazily on first `embed()` or `embed_single()` call; every new VectorStore instance triggers a fresh model load unless the EmbeddingEngine is shared.

**Why:** Understanding these facts avoids hardcoding wrong dimension counts and explains why tests that each create a fresh VectorStore are slow (model reload per instance).

**How to apply:** When writing assertions on embedding shape, assert `len(emb) == 384`. To speed up test suites that create many VectorStore instances, share one EmbeddingEngine across them.

Cosine distance range is [0, 2] (0 = identical, 2 = opposite). Top-3 results for a relevant "RAG LLM vector database retrieval" query against 10 on-topic documents have distance < 0.8. Threshold `max_distance=0.01` correctly filters out semantically unrelated documents.

ChunkClassifier Tier 1 rules: keyword matching hits `learning` for text containing "learn/course/study/research"; `code` for Python/Docker/FastAPI terms; `wellness` for gym/workout; `nutrition` for meal/food/cooking. Pure "RAG LLM" two-word query does NOT match any rule — needs richer context. The type-map shortcuts `body_gym` → (BODY, wellness) and `task` → (PURPOSE, goals) are deterministic.

QueryRouter rule-based routing: "What do I think about X?" → RAG (via _RAG_PATTERN `think about`). "How many gym sessions...?" → SQL + tables=["gym"] + time_range="this_week/last_week". "How do I feel about my gym workout?" → HYBRID (both _RAG_PATTERN and table match). Time extraction is exact string match — "last week" → "last_week".

DBSCAN clustering: `eps=0.35, min_samples=3` are production defaults. For small test sets (<10 docs) use `eps=0.5, min_samples=2` to ensure at least one cluster forms. With 10 semantically homogeneous RAG docs, clustering found 1 cluster + 1 noise at eps=0.5.

Deduplication: `VectorStore.ingest()` uses `chromadb.collection.upsert()` with deterministic IDs derived from `md5(text + json(metadata))`. Re-ingesting the same chunks leaves count unchanged.

`search_by_dimension` uses `where={"dimension": dim}` — only returns chunks whose metadata field `dimension` exactly matches. There is NO cross-contamination: a query about "code" returns zero results if only "learning" chunks are present.

`search_with_recency` over-fetches 2x, computes `_combined_score = 0.75*similarity + 0.15*recency + 0.10*keyword_overlap`, sorts descending, and truncates to n_results. Results are guaranteed to have `_combined_score` key and be sorted.

`update_metadata` calls `collection.update()` — caller must pass the full merged metadata dict (fetch existing, mutate, pass back) because ChromaDB replaces the entire metadata object.

`get_cluster_labels` uses `where={"cluster_label": {"$ne": ""}}` — only returns chunks that have a non-empty cluster_label set. Must manually ingest chunks with `cluster_label` in metadata or run `update_metadata_with_clusters`.
