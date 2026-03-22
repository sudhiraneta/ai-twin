#!/usr/bin/env python3
"""
Parallel Persona Dimension Test Suite
======================================
Tests every aspect of the twin's persona against real data:
  - Chunk availability per dimension
  - Trait extraction quality
  - RAG retrieval accuracy (do the right chunks surface?)
  - Twin response attribution (does the "Why I think this" cite real data?)

Usage:
    python tests/test_persona_dimensions.py
    python tests/test_persona_dimensions.py --dimension nutrition
    python tests/test_persona_dimensions.py --rag-only       # skip LLM response tests
"""

import sys
import argparse
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, str(Path(__file__).parent.parent))

from memory.vectorstore import VectorStore
from persona.profile import PersonaProfile
from persona.dimensions import DIMENSIONS
from persona.skills import read_all_skill_files

# Test queries per dimension — what should the twin know?
DIMENSION_TESTS = {
    "nutrition": {
        "queries": ["What food do I like?", "Am I vegetarian?", "What cuisines do I prefer?"],
        "expected_keywords": ["vegetarian", "thai", "indian"],
    },
    "wellness": {
        "queries": ["How often do I go to the gym?", "Do I journal?", "gym target workout"],
        "expected_keywords": ["gym", "journal", "4x", "workout"],
    },
    "professional": {
        "queries": ["Drata career engineering work", "articulation drill influence authority"],
        "expected_keywords": ["drata", "career", "articulation", "influence"],
    },
    "code": {
        "queries": ["python rust programming language", "FastAPI framework"],
        "expected_keywords": ["python", "rust", "fastapi"],
    },
    "learning": {
        "queries": ["deeplearning.ai course learning", "youtube tutorial learning"],
        "expected_keywords": ["deeplearning", "youtube", "learning"],
    },
    "life": {
        "queries": ["zara shopping fashion clothing", "morning routine meditation cooked"],
        "expected_keywords": ["zara", "morning", "cooked"],
    },
    "goals": {
        "queries": ["30 day plan executive communicate", "ship side projects quarter goals"],
        "expected_keywords": ["plan", "executive", "ship", "project"],
    },
    "progress": {
        "queries": ["weekly review wins journaling score", "week progress march"],
        "expected_keywords": ["journaling", "score", "wins", "progress"],
    },
    "relationships": {
        "queries": ["calm quiet atmosphere focus", "networking manager reciprocity"],
        "expected_keywords": ["calm", "quiet", "networking"],
    },
    "language_style": {
        "queries": ["casual direct writing style communication", "makes sense let me think"],
        "expected_keywords": ["casual", "direct"],
    },
    "vibe": {
        "queries": ["energy morning routine meditation moved", "soul checkin signals habits"],
        "expected_keywords": ["morning", "moved", "meditation"],
    },
    "entertainment": {
        "queries": ["best sci fi movies", "youtube music voyage mosaic"],
        "expected_keywords": ["sci fi", "youtube", "voyage"],
    },
    "creative": {
        "queries": ["youtube substack content writing creative"],
        "expected_keywords": ["youtube", "content"],
    },
}


def test_dimension(dim_name: str, store: VectorStore, profile: PersonaProfile, skills: dict, test_llm: bool = False) -> dict:
    """Test a single dimension end-to-end. Returns results dict."""
    result = {
        "dimension": dim_name,
        "display": DIMENSIONS.get(dim_name, {}).get("display", dim_name),
        "checks": {},
    }
    test_config = DIMENSION_TESTS.get(dim_name, {"queries": [], "expected_keywords": []})

    # 1. Chunk availability
    dim_results = store.search_by_dimension("", dim_name, n_results=200)
    chunk_count = len(dim_results)
    result["chunks"] = chunk_count
    result["checks"]["has_chunks"] = chunk_count > 0

    # 2. Trait extraction
    dim_obj = profile.get_dimension(dim_name)
    traits = dim_obj.traits if dim_obj else {}
    non_empty_traits = sum(1 for v in traits.values() if v) if traits else 0
    result["traits"] = non_empty_traits
    result["confidence"] = dim_obj.confidence if dim_obj else 0
    result["checks"]["has_traits"] = non_empty_traits > 0

    # 3. Skill file exists and has content
    skill_content = skills.get(dim_name, "")
    has_skill_data = skill_content and "No data yet" not in skill_content
    result["checks"]["skill_file_populated"] = has_skill_data

    # 4. RAG retrieval — do dimension-relevant queries return the right chunks?
    rag_hits = 0
    rag_total = 0
    expected_kw = test_config["expected_keywords"]

    for query in test_config["queries"]:
        search_results = store.search(query, n_results=5)
        if not search_results:
            rag_total += 1
            continue

        # Check if any result contains expected keywords
        all_text = " ".join(r["text"].lower() for r in search_results)
        for kw in expected_kw:
            rag_total += 1
            if kw.lower() in all_text:
                rag_hits += 1

    result["rag_hits"] = rag_hits
    result["rag_total"] = rag_total
    result["checks"]["rag_retrieval"] = rag_hits > 0 if rag_total > 0 else None

    # 5. Twin response test (optional — requires LLM)
    if test_llm and test_config["queries"]:
        try:
            from twin.llm_client import chat_completion
            from persona.skills import build_persona_from_skills

            persona = build_persona_from_skills()
            query = test_config["queries"][0]

            # Build a simple context
            memories = store.search(query, n_results=5)
            mem_text = "\n".join(r["text"][:200] for r in memories[:3])

            response = chat_completion(
                system=f"{persona}\n\nAlways include a '> **Why I think this:**' block citing specific data.",
                messages=[{"role": "user", "content": query}],
                max_tokens=500,
            )

            has_why = "why i think this" in response.lower() or "because" in response.lower()
            # Check if response references any expected keywords
            resp_lower = response.lower()
            cited_keywords = [kw for kw in expected_kw if kw.lower() in resp_lower]

            result["response_preview"] = response[:200]
            result["checks"]["response_has_why"] = has_why
            result["checks"]["response_cites_data"] = len(cited_keywords) > 0
            result["cited_keywords"] = cited_keywords

        except Exception as e:
            result["checks"]["response_has_why"] = None
            result["checks"]["response_cites_data"] = None
            result["llm_error"] = str(e)

    return result


def print_result(r: dict):
    """Print a single dimension test result."""
    name = r["dimension"]
    display = r["display"]
    chunks = r["chunks"]
    traits = r["traits"]
    confidence = r["confidence"]

    # Status symbols
    def sym(check):
        if check is True:
            return "\033[32mPASS\033[0m"
        elif check is False:
            return "\033[31mFAIL\033[0m"
        return "\033[33mSKIP\033[0m"

    checks = r["checks"]
    rag_pct = f"{r['rag_hits']}/{r['rag_total']}" if r.get("rag_total") else "n/a"

    print(f"\n{'─'*60}")
    print(f"  {display} ({name})")
    print(f"  chunks: {chunks} | traits: {traits} | confidence: {confidence:.0%}")
    print(f"  RAG hits: {rag_pct}")
    print(f"  Checks:")
    print(f"    has_chunks:          {sym(checks.get('has_chunks'))}")
    print(f"    has_traits:          {sym(checks.get('has_traits'))}")
    print(f"    skill_file:          {sym(checks.get('skill_file_populated'))}")
    print(f"    rag_retrieval:       {sym(checks.get('rag_retrieval'))}")

    if "response_has_why" in checks:
        print(f"    response_has_why:    {sym(checks.get('response_has_why'))}")
        print(f"    response_cites_data: {sym(checks.get('response_cites_data'))}")
        if r.get("cited_keywords"):
            print(f"    cited: {r['cited_keywords']}")

    if r.get("response_preview"):
        print(f"  Response: {r['response_preview'][:120]}...")


def run_tests(dimensions: list[str] | None = None, test_llm: bool = False, parallel: bool = True):
    """Run dimension tests, optionally in parallel."""
    store = VectorStore()
    profile = PersonaProfile.load()
    skills = read_all_skill_files()

    dims = dimensions or list(DIMENSIONS.keys())

    print("=" * 60)
    print(f"  Persona Dimension Test Suite")
    print(f"  Testing {len(dims)} dimensions | LLM responses: {'ON' if test_llm else 'OFF'}")
    print("=" * 60)

    start = time.time()
    results = []

    if parallel and len(dims) > 1 and not test_llm:
        # Parallel RAG-only tests
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = {
                executor.submit(test_dimension, dim, store, profile, skills, test_llm): dim
                for dim in dims
            }
            for future in as_completed(futures):
                results.append(future.result())
    else:
        # Sequential (needed for LLM to avoid hammering)
        for dim in dims:
            results.append(test_dimension(dim, store, profile, skills, test_llm))

    # Sort by dimension name
    results.sort(key=lambda r: r["dimension"])

    for r in results:
        print_result(r)

    # Summary
    elapsed = time.time() - start
    total_checks = sum(len(r["checks"]) for r in results)
    passed = sum(sum(1 for v in r["checks"].values() if v is True) for r in results)
    failed = sum(sum(1 for v in r["checks"].values() if v is False) for r in results)
    skipped = total_checks - passed - failed

    print(f"\n{'='*60}")
    print(f"  SUMMARY: {passed} passed / {failed} failed / {skipped} skipped ({elapsed:.1f}s)")
    print(f"  Total chunks: {store.count()}")
    populated = sum(1 for r in results if r["checks"].get("has_traits"))
    print(f"  Dimensions populated: {populated}/{len(dims)}")
    print(f"  RAG coverage: {sum(r.get('rag_hits',0) for r in results)}/{sum(r.get('rag_total',0) for r in results)}")
    print(f"{'='*60}")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test twin persona dimensions")
    parser.add_argument("--dimension", "-d", type=str, help="Test a single dimension")
    parser.add_argument("--with-llm", action="store_true", help="Also test LLM responses (slower)")
    parser.add_argument("--rag-only", action="store_true", help="Only test RAG retrieval (no LLM)")
    args = parser.parse_args()

    dims = [args.dimension] if args.dimension else None
    test_llm = args.with_llm and not args.rag_only

    run_tests(dimensions=dims, test_llm=test_llm)
