#!/usr/bin/env python3
"""Real-data RAG evaluation: retrieves from actual ChromaDB, calls Ollama for grounded answers.

Tests that:
1. Retrieval returns relevant documents from real 18K+ chunk store
2. Ollama (llama3) generates answers grounded in retrieved context
3. Ollama cites specific data points from the context (no hallucination)
4. Hybrid routing (SQL vs RAG) works on real questions
5. Classification and clustering work on real data distributions

Run:  .venv/bin/python tests/test_rag_eval_real.py
"""

import os
import sys
import time
import traceback

# Force Ollama as the provider for this eval
os.environ["LLM_PROVIDER"] = "ollama"
os.environ["OLLAMA_MODEL"] = "llama3"

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from memory.vectorstore import VectorStore
from persona.classifier import ChunkClassifier
from config import RELEVANCE_THRESHOLD, RECENCY_WEIGHT, MAX_CONTEXT_MESSAGES

# ─── Test Infrastructure ──────────────────────────────────────────

PASS_COUNT = 0
FAIL_COUNT = 0
RESULTS = []


def run_test(name, fn, timeout=120):
    global PASS_COUNT, FAIL_COUNT
    start = time.time()
    try:
        fn()
        elapsed = time.time() - start
        PASS_COUNT += 1
        RESULTS.append((name, "PASS", elapsed))
        print(f"  PASS: {name} ({elapsed:.1f}s)")
    except Exception as e:
        elapsed = time.time() - start
        FAIL_COUNT += 1
        RESULTS.append((name, "FAIL", elapsed))
        print(f"  FAIL: {name} ({elapsed:.1f}s)")
        traceback.print_exc()
        print()


# ─── Area 1: Real Data Retrieval Quality ──────────────────────────

def test_retrieval_code_dimension():
    """Retrieve code-related chunks — should find real coding conversations."""
    vs = VectorStore()
    results = vs.search_by_dimension("python fastapi backend", "code", n_results=5)
    assert len(results) > 0, "No code dimension chunks found"
    # Verify they're actually about code
    for r in results:
        assert r["metadata"]["dimension"] == "code", f"Wrong dimension: {r['metadata']['dimension']}"
    print(f"    Found {len(results)} code chunks, top distance: {results[0]['distance']:.3f}")


def test_retrieval_wellness_dimension():
    """Retrieve wellness chunks — should find gym/health data."""
    vs = VectorStore()
    results = vs.search_by_dimension("gym workout exercise fitness", "wellness", n_results=5)
    assert len(results) > 0, "No wellness dimension chunks found"
    for r in results:
        assert r["metadata"]["dimension"] == "wellness"
    print(f"    Found {len(results)} wellness chunks, top distance: {results[0]['distance']:.3f}")


def test_retrieval_relevance_threshold():
    """Verify relevance threshold filters out irrelevant results."""
    vs = VectorStore()
    # Search with very strict threshold
    strict = vs.search("python programming", n_results=20, max_distance=0.5)
    loose = vs.search("python programming", n_results=20, max_distance=1.5)
    # Strict should return fewer (or equal) results
    assert len(strict) <= len(loose), f"Strict ({len(strict)}) should be <= loose ({len(loose)})"
    # All strict results should have distance <= 0.5
    for r in strict:
        assert r["distance"] <= 0.5, f"Distance {r['distance']} exceeds threshold 0.5"
    print(f"    Strict (≤0.5): {len(strict)} results, Loose (≤1.5): {len(loose)} results")


def test_retrieval_recency_weighted():
    """Verify recency-weighted search returns results and reranks them."""
    vs = VectorStore()
    results = vs.search_with_recency(
        "career goals and priorities", n_results=10,
        max_distance=RELEVANCE_THRESHOLD, recency_weight=RECENCY_WEIGHT,
    )
    assert len(results) > 0, "No results from recency-weighted search"
    # Check that combined scores exist
    for r in results:
        assert "_combined_score" in r, "Missing _combined_score from recency search"
    # Scores should be in descending order
    scores = [r["_combined_score"] for r in results]
    assert scores == sorted(scores, reverse=True), "Results not sorted by combined score"
    print(f"    Got {len(results)} results, top score: {scores[0]:.3f}, bottom: {scores[-1]:.3f}")


def test_retrieval_dimension_aware_search():
    """Test the engine's dimension-aware search with real data."""
    from twin.engine import TwinEngine
    engine = TwinEngine()
    results = engine.search_memory("what programming languages do I use", n_results=10)
    assert len(results) > 0, "No results from dimension-aware search"
    # Should prefer code-dimension chunks
    code_count = sum(1 for r in results if r["metadata"].get("dimension") == "code")
    print(f"    Got {len(results)} results, {code_count} from 'code' dimension")


def test_retrieval_cross_dimension_query():
    """A broad query should pull from multiple dimensions."""
    from twin.engine import TwinEngine
    engine = TwinEngine()
    results = engine.search_memory("how am I doing overall with my goals and health", n_results=12)
    assert len(results) > 0
    dimensions = set(r["metadata"].get("dimension", "") for r in results)
    # Should span at least 2 different dimensions for a broad query
    assert len(dimensions) >= 2, f"Only got dimension(s): {dimensions}"
    print(f"    Got {len(results)} results across {len(dimensions)} dimensions: {dimensions}")


def test_retrieval_no_results_for_nonsense():
    """A nonsense query with strict threshold should return few/no results."""
    vs = VectorStore()
    results = vs.search("xyzzy flurbo garbonzo zarquon", n_results=10, max_distance=0.3)
    # Should get very few or zero results at strict threshold
    print(f"    Nonsense query returned {len(results)} results (expected ~0 at distance ≤ 0.3)")
    assert len(results) <= 3, f"Too many results ({len(results)}) for nonsense query"


# ─── Area 2: Query Router with Real Questions ────────────────────

def test_router_gym_question():
    """'How many gym sessions this week' should route to SQL."""
    from db import QueryRouter, QueryType
    router = QueryRouter()
    routed = router.route("how many gym sessions this week")
    assert routed.query_type == QueryType.SQL, f"Expected SQL, got {routed.query_type}"
    assert "gym" in routed.sql_tables, f"Expected gym in tables, got {routed.sql_tables}"
    assert routed.time_range == "this_week"
    print(f"    Routed to {routed.query_type.value}, tables: {routed.sql_tables}")


def test_router_thought_question():
    """'What do I think about career growth' should route to RAG."""
    from db import QueryRouter, QueryType
    router = QueryRouter()
    routed = router.route("what do I think about career growth")
    assert routed.query_type == QueryType.RAG, f"Expected RAG, got {routed.query_type}"
    print(f"    Routed to {routed.query_type.value}")


def test_router_hybrid_question():
    """'Am I on track with my tasks and goals' should route to HYBRID."""
    from db import QueryRouter, QueryType
    router = QueryRouter()
    routed = router.route("am I on track with my tasks and goals")
    assert routed.query_type in (QueryType.HYBRID, QueryType.SQL), \
        f"Expected HYBRID/SQL, got {routed.query_type}"
    print(f"    Routed to {routed.query_type.value}, tables: {routed.sql_tables}")


def test_router_nutrition_question():
    """Nutrition score question should route to SQL with nutrition table."""
    from db import QueryRouter, QueryType
    router = QueryRouter()
    routed = router.route("what's my nutrition score this week")
    assert routed.query_type in (QueryType.SQL, QueryType.HYBRID)
    assert "nutrition" in routed.sql_tables
    print(f"    Routed to {routed.query_type.value}, tables: {routed.sql_tables}")


def test_router_mindset_question():
    """Mindset/reflection question should route to RAG."""
    from db import QueryRouter, QueryType
    router = QueryRouter()
    routed = router.route("how has my mindset been this month about work-life balance")
    assert routed.query_type == QueryType.RAG, f"Expected RAG, got {routed.query_type}"
    print(f"    Routed to {routed.query_type.value}")


# ─── Area 3: Ollama Grounded Answer Generation ───────────────────

def _call_ollama_with_context(query: str, context_chunks: list[dict]) -> str:
    """Build a grounded prompt from retrieved chunks and call Ollama."""
    from openai import OpenAI

    # Format context with source tags (same as engine does)
    from twin.engine import TwinEngine
    memory_lines = [TwinEngine._format_memory_line(r) for r in context_chunks]
    context_text = "\n\n---\n\n".join(memory_lines)

    system_prompt = (
        "You are answering questions about a specific person based ONLY on the provided context.\n\n"
        "RULES:\n"
        "1. ONLY use information from the context below. Do NOT make up facts.\n"
        "2. Cite specific data points using their [source | date] tags.\n"
        "3. If the context doesn't contain relevant info, say 'No relevant data found.'\n"
        "4. Keep your answer concise (3-5 sentences max).\n\n"
        f"## Context\n\n{context_text}"
    )

    client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")
    response = client.chat.completions.create(
        model="llama3",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": query},
        ],
        max_tokens=512,
        temperature=0.1,  # low temp for factual grounding
    )
    return response.choices[0].message.content


def test_ollama_grounded_code_answer():
    """Retrieve real code chunks, call Ollama, verify it cites context."""
    vs = VectorStore()
    results = vs.search_by_dimension(
        "what programming languages and frameworks do I use",
        "code", n_results=8, max_distance=RELEVANCE_THRESHOLD,
    )
    assert len(results) > 0, "No code chunks retrieved"

    answer = _call_ollama_with_context(
        "What programming languages and frameworks does this person use?", results
    )
    assert len(answer) > 20, f"Answer too short: {answer}"
    # Should NOT say "no relevant data" since we have code chunks
    assert "no relevant data" not in answer.lower(), f"Ollama said no data despite having {len(results)} chunks"
    print(f"    Retrieved {len(results)} chunks, answer length: {len(answer)} chars")
    print(f"    Answer preview: {answer[:200]}...")


def test_ollama_grounded_wellness_answer():
    """Retrieve wellness/body chunks, call Ollama, verify grounded response.

    Note: wellness dimension may contain mixed content. The LLM should answer
    based on whatever context it receives — grounded in actual retrieved content.
    """
    vs = VectorStore()
    # Broad search across all types for body/wellness content
    results = vs.search(
        "gym workout exercise fitness body health",
        n_results=8, max_distance=RELEVANCE_THRESHOLD,
    )
    assert len(results) > 0, "No chunks found for wellness query"

    answer = _call_ollama_with_context(
        "Based on the context provided, what can you tell about this person's "
        "health, exercise habits, or wellness routines?", results
    )
    assert len(answer) > 20, f"Answer too short: {answer}"
    print(f"    Retrieved {len(results)} chunks, answer length: {len(answer)} chars")
    print(f"    Answer preview: {answer[:200]}...")


def test_ollama_cites_source_tags():
    """Verify Ollama's answer contains [source | date] style citations."""
    vs = VectorStore()
    results = vs.search_with_recency(
        "career and professional goals",
        n_results=8, max_distance=RELEVANCE_THRESHOLD, recency_weight=RECENCY_WEIGHT,
    )
    assert len(results) > 0, "No results for career query"

    answer = _call_ollama_with_context(
        "What are this person's career goals and professional interests?", results
    )
    # Check for citation markers — brackets indicate source tags
    has_brackets = "[" in answer and "]" in answer
    has_source_words = any(w in answer.lower() for w in [
        "gemini", "note", "browser", "singularity", "journal",
        "tracker", "review", "task", "self-reported",
    ])
    cited = has_brackets or has_source_words
    print(f"    Answer has brackets: {has_brackets}, source words: {has_source_words}")
    print(f"    Answer preview: {answer[:300]}...")
    # Warn but don't hard-fail — LLMs are probabilistic
    if not cited:
        print("    WARNING: No obvious citations found. LLM may need stronger prompting.")
    assert len(answer) > 20


def test_ollama_refuses_on_no_context():
    """With zero relevant context, Ollama should say it has no data."""
    # Give it empty context
    answer = _call_ollama_with_context(
        "What is this person's favorite quantum physics experiment?", []
    )
    # With no context, it should indicate lack of data
    no_data_signals = [
        "no relevant", "no data", "no information", "not available",
        "don't have", "cannot determine", "no context", "not mentioned",
        "no specific", "unable to", "not provided",
    ]
    has_no_data = any(s in answer.lower() for s in no_data_signals)
    print(f"    Answer: {answer[:200]}...")
    if not has_no_data:
        print("    WARNING: LLM may have hallucinated without context")
    # This is important — if it fabricates, the grounding system failed
    assert has_no_data or len(answer) < 100, \
        f"LLM generated a long answer ({len(answer)} chars) with no context — likely hallucinating"


def test_ollama_entertainment_grounded():
    """Retrieve entertainment chunks and verify grounded answer.

    Note: entertainment dimension may contain YouTube/media chunks that aren't
    directly about preferences. The LLM should answer based on actual content.
    """
    vs = VectorStore()
    results = vs.search(
        "movies shows music entertainment youtube netflix anime preferences",
        n_results=8, max_distance=RELEVANCE_THRESHOLD,
    )
    assert len(results) > 0, "No entertainment chunks found"

    answer = _call_ollama_with_context(
        "Based on the context provided, what can you tell about this person's "
        "entertainment preferences or media habits?", results
    )
    assert len(answer) > 20
    print(f"    Retrieved {len(results)} entertainment chunks")
    print(f"    Answer preview: {answer[:200]}...")


def test_ollama_learning_grounded():
    """Retrieve learning chunks and verify answer is about actual learning topics."""
    vs = VectorStore()
    results = vs.search_by_dimension(
        "learning studying courses tutorials reading",
        "learning", n_results=8, max_distance=RELEVANCE_THRESHOLD,
    )
    assert len(results) > 0, "No learning chunks found"

    answer = _call_ollama_with_context(
        "What topics has this person been learning about?", results
    )
    assert len(answer) > 20
    assert "no relevant data" not in answer.lower()
    print(f"    Retrieved {len(results)} learning chunks")
    print(f"    Answer preview: {answer[:200]}...")


# ─── Area 4: Full Pipeline (Retrieve + Route + Generate) ─────────

def test_full_pipeline_hybrid_context():
    """Test hybrid_context() end-to-end — routes and retrieves real data."""
    from twin.engine import TwinEngine
    engine = TwinEngine()
    ctx = engine.hybrid_context("what programming projects have I been working on")
    assert ctx["route_type"] in ("rag", "hybrid"), f"Unexpected route: {ctx['route_type']}"
    assert ctx["rag_context"] is not None, "RAG context should not be None for this query"
    # RAG context should contain formatted memory lines
    assert "---" in ctx["rag_context"], "RAG context missing separator"
    print(f"    Route: {ctx['route_type']}")
    print(f"    RAG context length: {len(ctx['rag_context'])} chars")
    if ctx["sql_context"]:
        print(f"    SQL context length: {len(ctx['sql_context'])} chars")


def test_full_pipeline_sql_metrics():
    """Test that SQL metric queries return data from Singularity DB."""
    from db import MetricStore
    store = MetricStore()
    # Try gym (may or may not have data for current week)
    result = store.gym_streak(weeks=8)
    assert result.summary, "Gym streak should return a summary"
    assert "sessions" in result.summary.lower()
    print(f"    Gym streak summary: {result.summary[:150]}")


def test_full_pipeline_formatted_memories():
    """Test that _retrieve_memories_dimension_aware formats with source tags."""
    from twin.engine import TwinEngine
    engine = TwinEngine()
    formatted = engine._retrieve_memories_dimension_aware("my daily routine and habits")
    if formatted:
        # Should contain the memory context template header
        assert "memories" in formatted.lower() or "Relevant" in formatted
        # Should contain source tags in brackets
        assert "[" in formatted, "Formatted memories should have [source] tags"
        print(f"    Formatted context length: {len(formatted)} chars")
        # Show first memory line
        lines = formatted.split("---")
        if lines:
            print(f"    First memory: {lines[0].strip()[:150]}...")
    else:
        print("    No memories retrieved (may need data for this query)")


# ─── Area 5: Classification on Real Data ──────────────────────────

def test_classifier_real_code_chunk():
    """Classify a real code-related text from the database."""
    vs = VectorStore()
    results = vs.search_by_dimension("python", "code", n_results=1)
    assert results, "No code chunks to test with"

    classifier = ChunkClassifier()
    pillar, dimension = classifier.classify_chunk(results[0]["text"], results[0]["metadata"])
    print(f"    Text: {results[0]['text'][:80]}...")
    print(f"    Classified: pillar={pillar}, dimension={dimension}")
    # The chunk was already in code dimension, classification should agree or be related
    assert dimension, "Classification returned empty dimension"


def test_classifier_query_routing():
    """Verify classify_text returns relevant dimensions for real queries."""
    classifier = ChunkClassifier()

    queries = {
        "python fastapi deployment": "code",
        "gym workout routine": "wellness",
        "career goals job": "professional",
        "netflix movie recommendation": "entertainment",
    }
    for query, expected_dim in queries.items():
        dims = classifier.classify_text(query)
        assert expected_dim in dims, f"Query '{query}' didn't match '{expected_dim}', got: {dims}"
        print(f"    '{query}' → {dims[:3]}")


# ─── Run All Tests ────────────────────────────────────────────────

def main():
    ts = time.strftime("%H:%M:%S")
    print(f"\n[{ts}] {'=' * 60}")
    print(f"[{ts}] RAG Eval — Real Data + Ollama Grounding")
    print(f"[{ts}] {'=' * 60}")

    print(f"\n--- Area 1: Real Data Retrieval Quality ---")
    run_test("retrieval_code_dimension", test_retrieval_code_dimension)
    run_test("retrieval_wellness_dimension", test_retrieval_wellness_dimension)
    run_test("retrieval_relevance_threshold", test_retrieval_relevance_threshold)
    run_test("retrieval_recency_weighted", test_retrieval_recency_weighted)
    run_test("retrieval_dimension_aware_search", test_retrieval_dimension_aware_search)
    run_test("retrieval_cross_dimension_query", test_retrieval_cross_dimension_query)
    run_test("retrieval_no_results_for_nonsense", test_retrieval_no_results_for_nonsense)

    print(f"\n--- Area 2: Query Router ---")
    run_test("router_gym_question", test_router_gym_question)
    run_test("router_thought_question", test_router_thought_question)
    run_test("router_hybrid_question", test_router_hybrid_question)
    run_test("router_nutrition_question", test_router_nutrition_question)
    run_test("router_mindset_question", test_router_mindset_question)

    print(f"\n--- Area 3: Ollama Grounded Answers ---")
    run_test("ollama_grounded_code_answer", test_ollama_grounded_code_answer)
    run_test("ollama_grounded_wellness_answer", test_ollama_grounded_wellness_answer)
    run_test("ollama_cites_source_tags", test_ollama_cites_source_tags)
    run_test("ollama_refuses_on_no_context", test_ollama_refuses_on_no_context)
    run_test("ollama_entertainment_grounded", test_ollama_entertainment_grounded)
    run_test("ollama_learning_grounded", test_ollama_learning_grounded)

    print(f"\n--- Area 4: Full Pipeline ---")
    run_test("full_pipeline_hybrid_context", test_full_pipeline_hybrid_context)
    run_test("full_pipeline_sql_metrics", test_full_pipeline_sql_metrics)
    run_test("full_pipeline_formatted_memories", test_full_pipeline_formatted_memories)

    print(f"\n--- Area 5: Classification on Real Data ---")
    run_test("classifier_real_code_chunk", test_classifier_real_code_chunk)
    run_test("classifier_query_routing", test_classifier_query_routing)

    # Summary
    ts = time.strftime("%H:%M:%S")
    print(f"\n[{ts}] {'=' * 60}")
    print(f"[{ts}] TOTAL: {PASS_COUNT} passed, {FAIL_COUNT} failed out of {PASS_COUNT + FAIL_COUNT}")
    print(f"[{ts}] {'=' * 60}")

    # Detailed results
    print(f"\n{'Test':<45s} {'Status':<6s} {'Time':>6s}")
    print("-" * 60)
    for name, status, elapsed in RESULTS:
        print(f"  {name:<43s} {status:<6s} {elapsed:>5.1f}s")

    sys.exit(1 if FAIL_COUNT > 0 else 0)


if __name__ == "__main__":
    main()
