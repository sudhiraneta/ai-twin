---
name: test_patterns_rag
description: Reliable test patterns discovered for this codebase's RAG pipeline tests
type: feedback
---

Use `_make_temp_vectorstore(collection_name)` (monkey-patch approach) to create isolated ChromaDB collections per test instead of the real `ai_twin_memory` collection. This avoids polluting production data.

**Why:** The real VectorStore constructor always uses `CHROMA_COLLECTION = "ai_twin_memory"`. Tests that share this collection interfere with each other and with production data.

**How to apply:** Generate a unique collection name per test using `uuid.uuid4().hex[:8]` suffix. Always call `_cleanup_collection(vs, name)` in a `finally` block to delete the temp collection after the test.

Each test function creates its own VectorStore (and therefore its own EmbeddingEngine). This makes tests slow (~2s each) but fully isolated. Acceptable for a test suite of ~40 tests (~90s total).

For DBSCAN clustering tests with small datasets: use `eps=0.5, min_samples=2` instead of production defaults (`eps=0.35, min_samples=3`). With 10 on-topic docs the production settings may classify all as noise.

Assertion threshold for relevance: `distance < 0.8` for top-3 results against 10 thematically matching documents. For strict "near-duplicate" filtering: `max_distance=0.01` reliably returns 0 results for off-topic queries.

Do NOT test LLM-dependent code paths (batch_classify_llm, label_clusters via LLM, TwinEngine.hybrid_context SQL branch) without a live LLM key — those paths require network calls. The label_clusters fallback (`cluster_N`) is reliable and testable without LLM.

The function-based test runner pattern (not pytest/unittest): each test is a plain function with no arguments. A `run_test(name, fn)` wrapper catches exceptions and records PASS/FAIL. A retry loop of 3 rounds retries only failed tests. This is the established pattern across all tests in this repo.
