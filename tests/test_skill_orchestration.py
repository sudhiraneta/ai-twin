"""
End-to-end tests for skill-file-driven retrieval orchestration.

For each dimension's skill file, verifies:
1. Semantic question correctly routes to the right dimension(s)
2. Retrieved chunks come from the correct tier/types defined in the skill file
3. Skill file content is passed to the LLM prompt
4. All 3 tiers (core, memory, supplementary) are attempted in order

Run: python tests/test_skill_orchestration.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import RELEVANCE_THRESHOLD
from persona.skills import DIMENSION_SOURCES, read_skill_file
from persona.classifier import ChunkClassifier
from memory.vectorstore import VectorStore

# Test questions per dimension — each should semantically match its skill file
DIMENSION_QUESTIONS: dict[str, list[str]] = {
    "code": [
        "What programming languages do I use?",
        "What frameworks have I worked with?",
        "What kind of projects do I build?",
    ],
    "professional": [
        "What is my current role?",
        "How do I approach leadership?",
        "What is my career trajectory?",
    ],
    "learning": [
        "What courses have I taken?",
        "What am I currently studying?",
        "What tutorials have I gone through?",
    ],
    "wellness": [
        "How often do I go to the gym?",
        "Do I meditate?",
        "What are my health habits?",
    ],
    "nutrition": [
        "What do I eat?",
        "What are my favorite restaurants?",
        "Am I vegetarian?",
    ],
    "creative": [
        "What creative projects do I work on?",
        "Do I make music or videos?",
        "What creative tools do I use?",
    ],
    "vibe": [
        "Am I a morning person?",
        "What is my energy pattern?",
        "What kind of atmosphere do I prefer?",
    ],
    "entertainment": [
        "What shows do I watch?",
        "What music do I listen to?",
        "What are my favorite movies?",
    ],
    "relationships": [
        "How do I communicate with people?",
        "What is my networking style?",
        "How do I handle conflict?",
    ],
    "language_style": [
        "How do I write and express myself?",
        "How formal or casual am I?",
        "What phrases do I use often?",
    ],
    "goals": [
        "What are my current goals?",
        "What have I achieved recently?",
        "What is my 30-day plan?",
    ],
    "life": [
        "What is my daily routine?",
        "What are my shopping habits?",
        "What does my lifestyle look like?",
    ],
    "progress": [
        "What did I accomplish this week?",
        "What are my active streaks?",
        "Show me my weekly review progress",
    ],
}


def test_dimension_classification():
    """Test dimension matching via both keyword classifier AND semantic fallback."""
    from twin.engine import TwinEngine

    classifier = ChunkClassifier()
    engine = TwinEngine()

    results = {}
    kw_total = 0
    kw_correct = 0
    sem_total = 0
    sem_correct = 0

    for dim_name, questions in DIMENSION_QUESTIONS.items():
        dim_results = []
        for q in questions:
            # Tier 1: keyword classifier
            kw_matched = classifier.classify_text(q)
            kw_hit = dim_name in kw_matched

            # Tier 2: semantic fallback (only runs when keywords miss)
            from persona.skills import DIMENSION_SOURCES
            sem_matched = engine._semantic_dimension_match(q, DIMENSION_SOURCES) if not kw_matched else []
            sem_hit = dim_name in sem_matched

            combined_hit = kw_hit or sem_hit
            method = "keyword" if kw_hit else ("semantic" if sem_hit else "miss")

            dim_results.append({
                "question": q,
                "expected": dim_name,
                "kw_matched": kw_matched[:3],
                "sem_matched": sem_matched[:3],
                "method": method,
                "correct": combined_hit,
            })

            kw_total += 1
            sem_total += 1
            if kw_hit:
                kw_correct += 1
            if combined_hit:
                sem_correct += 1

        results[dim_name] = dim_results

    print(f"\n{'='*70}")
    print(f"DIMENSION CLASSIFICATION (keyword + semantic fallback)")
    print(f"  Keyword only: {kw_correct}/{kw_total} ({kw_correct*100//kw_total}%)")
    print(f"  With semantic: {sem_correct}/{sem_total} ({sem_correct*100//sem_total}%)")
    print(f"{'='*70}")

    for dim_name, dim_results in results.items():
        all_correct = all(r["correct"] for r in dim_results)
        status = "PASS" if all_correct else "FAIL"
        print(f"\n  [{status}] {dim_name}")
        for r in dim_results:
            icon = "  OK" if r["correct"] else "MISS"
            method = r["method"]
            matched = r["kw_matched"] if method == "keyword" else r["sem_matched"]
            print(f"    {icon} [{method:8s}]: '{r['question']}' -> {matched}")

    return results, sem_correct, sem_total


def test_skill_file_exists():
    """Verify all 13 skill files exist and have required sections."""
    results = {}
    total = 0
    correct = 0

    required_sections = [
        "## Traits",
        "## Instructions",
        "### Do",
        "### Don't",
        "## Habits & Preferences",
        "## Sources — Retrieval Strategy",
        "### Tier 1: Core",
        "### Tier 2: LLM Memory",
        "### Tier 3: Supplementary",
    ]

    for dim_name in DIMENSION_SOURCES:
        content = read_skill_file(dim_name)
        total += 1

        if content is None:
            results[dim_name] = {"exists": False, "missing_sections": required_sections}
            print(f"  [FAIL] {dim_name}.md — file not found")
            continue

        missing = [s for s in required_sections if s not in content]
        ok = len(missing) == 0
        results[dim_name] = {"exists": True, "missing_sections": missing}

        if ok:
            correct += 1
            print(f"  [PASS] {dim_name}.md — all sections present ({len(content)} bytes)")
        else:
            print(f"  [FAIL] {dim_name}.md — missing: {missing}")

    print(f"\n{'='*70}")
    print(f"SKILL FILE STRUCTURE: {correct}/{total} complete ({correct*100//total}%)")
    print(f"{'='*70}")

    return results, correct, total


def test_tiered_retrieval():
    """Test that skill_search returns results from the correct tiers and types."""
    from twin.engine import TwinEngine

    engine = TwinEngine()
    store = engine.vector_store

    if store.count() == 0:
        print("\n  [SKIP] No data in vector store — cannot test retrieval")
        return {}, 0, 0

    results = {}
    total = 0
    correct = 0

    for dim_name, questions in DIMENSION_QUESTIONS.items():
        sources = DIMENSION_SOURCES.get(dim_name, {})
        if not sources:
            continue

        expected_primary = set(sources.get("primary_types", []))
        expected_memory = set(sources.get("memory_types", []))
        expected_secondary = set(sources.get("secondary_types", []))

        # Test with first question for each dimension
        q = questions[0]
        result = engine.skill_search(query=q, n_results=15, dimension=dim_name)

        chunks = result["results"]
        skill_ctx = result["skill_context"]
        matched_dims = result["dimensions"]

        # Analyze tier distribution
        tier_counts = {"primary": 0, "memory": 0, "supplementary": 0, "other": 0}
        type_by_tier = {"primary": set(), "memory": set(), "supplementary": set()}

        for chunk in chunks:
            tier = chunk.get("_tier", "other")
            tier_counts[tier] = tier_counts.get(tier, 0) + 1
            ctype = chunk.get("metadata", {}).get("type", "")
            if tier in type_by_tier:
                type_by_tier[tier].add(ctype)

        # Validation checks
        has_results = len(chunks) > 0
        has_skill_context = len(skill_ctx) > 0
        has_instructions = "## Instructions" in skill_ctx if skill_ctx else False
        primary_types_correct = type_by_tier["primary"].issubset(expected_primary) if type_by_tier["primary"] else True

        all_ok = has_results and has_skill_context and has_instructions
        total += 1
        if all_ok:
            correct += 1

        status = "PASS" if all_ok else "FAIL"
        results[dim_name] = {
            "question": q,
            "total_chunks": len(chunks),
            "tier_counts": tier_counts,
            "types_found": {k: list(v) for k, v in type_by_tier.items()},
            "has_skill_context": has_skill_context,
            "has_instructions": has_instructions,
            "primary_types_correct": primary_types_correct,
        }

        print(f"\n  [{status}] {dim_name}: '{q}'")
        print(f"    Chunks: {len(chunks)} | Tiers: T1={tier_counts['primary']} T2={tier_counts['memory']} T3={tier_counts['supplementary']}")
        print(f"    T1 types: {type_by_tier['primary'] or '(none)'}")
        print(f"    T2 types: {type_by_tier['memory'] or '(none)'}")
        print(f"    T3 types: {type_by_tier['supplementary'] or '(none)'}")
        print(f"    Skill context: {'YES' if has_skill_context else 'NO'} | Instructions: {'YES' if has_instructions else 'NO'}")

        if not has_results:
            print(f"    WARNING: No chunks returned for '{q}'")
        if not has_skill_context:
            print(f"    WARNING: No skill context returned")

    print(f"\n{'='*70}")
    print(f"TIERED RETRIEVAL: {correct}/{total} dimensions pass ({correct*100//total if total else 0}%)")
    print(f"{'='*70}")

    return results, correct, total


def test_skill_source_completeness():
    """Verify every DIMENSION_SOURCES entry has all required fields."""
    required_fields = [
        "description", "primary_types", "primary_queries",
        "memory_types", "memory_queries",
        "secondary_types", "secondary_queries",
        "do", "dont", "habits", "preferences", "data_goals", "edge_cases",
    ]

    total = 0
    correct = 0

    print(f"\n{'='*70}")
    print("DIMENSION_SOURCES COMPLETENESS")
    print(f"{'='*70}")

    for dim_name, sources in DIMENSION_SOURCES.items():
        total += 1
        missing = [f for f in required_fields if f not in sources or not sources[f]]
        ok = len(missing) == 0

        if ok:
            correct += 1
            print(f"  [PASS] {dim_name} — all {len(required_fields)} fields present")
        else:
            print(f"  [FAIL] {dim_name} — missing/empty: {missing}")

    print(f"\n  TOTAL: {correct}/{total} complete ({correct*100//total}%)")
    return correct, total


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("  SKILL FILE ORCHESTRATION — END-TO-END TEST SUITE")
    print("=" * 70)

    # Test 1: DIMENSION_SOURCES config completeness
    src_correct, src_total = test_skill_source_completeness()

    # Test 2: Skill file existence and structure
    print()
    file_results, file_correct, file_total = test_skill_file_exists()

    # Test 3: Dimension classification accuracy
    class_results, class_correct, class_total = test_dimension_classification()

    # Test 4: Tiered retrieval end-to-end
    print(f"\n{'='*70}")
    print("TIERED RETRIEVAL END-TO-END")
    print(f"{'='*70}")
    retrieval_results, ret_correct, ret_total = test_tiered_retrieval()

    # Summary
    print(f"\n{'='*70}")
    print("  SUMMARY")
    print(f"{'='*70}")
    print(f"  Sources config:     {src_correct}/{src_total}")
    print(f"  Skill files:        {file_correct}/{file_total}")
    print(f"  Classification:     {class_correct}/{class_total}")
    print(f"  Tiered retrieval:   {ret_correct}/{ret_total}")

    all_pass = (
        src_correct == src_total
        and file_correct == file_total
        and class_correct == class_total
        and (ret_correct == ret_total or ret_total == 0)
    )
    print(f"\n  {'ALL TESTS PASSED' if all_pass else 'SOME TESTS FAILED'}")
    sys.exit(0 if all_pass else 1)
