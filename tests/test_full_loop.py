#!/usr/bin/env python3
"""
Full Loop Test — runs the entire pipeline and tests every dimension with LLM responses.

1. Syncs all data from Singularity
2. Classifies chunks into dimensions
3. Extracts traits → regenerates skill files
4. For each dimension: asks the twin a question, validates the response
5. Logs everything to data/logs/test_full_loop.md

Usage:
    python tests/test_full_loop.py
"""

import sys
import json
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import LOGS_DIR
from memory.vectorstore import VectorStore
from persona.classifier import ChunkClassifier
from persona.extractor import PersonaExtractor
from persona.profile import PersonaProfile
from persona.dimensions import DIMENSIONS
from persona.skills import write_all_skill_files, build_persona_from_skills
from connectors import ALL_CONNECTORS
from twin.llm_client import chat_completion

LOG_FILE = LOGS_DIR / "test_full_loop.md"

# Questions per dimension — what to ask the twin
DIMENSION_QUESTIONS = {
    "code":          "What programming languages and frameworks do I use most?",
    "professional":  "What's my current career focus and what am I working on at work?",
    "learning":      "What am I currently learning and what resources do I use?",
    "wellness":      "How often do I work out and what are my health habits?",
    "nutrition":     "What are my food preferences and dietary choices?",
    "creative":      "What creative activities or content do I engage with?",
    "vibe":          "What's my typical energy and what environments do I prefer?",
    "entertainment": "What do I watch or listen to for entertainment?",
    "relationships": "How do I communicate with people and what's my social style?",
    "language_style":"How would you describe my writing and communication style?",
    "goals":         "What are my main goals right now — professional and personal?",
    "life":          "What does my daily life look like — routines, shopping, habits?",
    "progress":      "How has my week been? What wins and progress have I made?",
}

# Expected keywords per dimension — at least one should appear in response
EXPECTED_KEYWORDS = {
    "code":          ["python", "rust", "fastapi"],
    "professional":  ["drata", "career", "engineer", "influence", "articulation"],
    "learning":      ["deeplearning", "javascript", "learning", "course"],
    "wellness":      ["gym", "journal", "workout", "meditation", "4x"],
    "nutrition":     ["vegetarian", "thai", "indian", "protein"],
    "creative":      ["youtube", "music", "content", "creative"],
    "vibe":          ["energy", "morning", "calm", "quiet"],
    "entertainment": ["youtube", "sci fi", "music", "voyage", "action"],
    "relationships": ["direct", "selective", "calm", "networking"],
    "language_style":["casual", "direct", "technical", "makes sense"],
    "goals":         ["plan", "executive", "ship", "project", "30"],
    "life":          ["zara", "morning", "cooked", "cafe", "shopping"],
    "progress":      ["journal", "score", "wins", "week", "progress"],
}


def log(msg: str, lines: list):
    """Print and collect log line."""
    print(msg)
    lines.append(msg)


def run():
    start = time.time()
    lines = []
    log(f"# Full Loop Test — {datetime.now().strftime('%Y-%m-%d %H:%M')}", lines)
    log("", lines)

    store = VectorStore()

    # ── Step 1: Sync ─────────────────────────────────────────────────
    log("## Step 1: Sync Data", lines)
    total_new = 0
    for name, cls in ALL_CONNECTORS.items():
        try:
            connector = cls()
            chunks = connector.sync()
            count = store.ingest(chunks) if chunks else 0
            total_new += count
            if count > 0:
                log(f"- {name}: **{count}** new chunks", lines)
        except Exception as e:
            log(f"- {name}: ERROR — {e}", lines)
    log(f"- **Total new: {total_new}** | Store: {store.count()} chunks", lines)
    log("", lines)

    # ── Step 2: Classify ─────────────────────────────────────────────
    log("## Step 2: Classify Chunks", lines)
    classifier = ChunkClassifier()
    unclassified = store.get_unclassified_chunks(limit=1000)
    ids = unclassified.get("ids", [])
    documents = unclassified.get("documents", [])
    metadatas = unclassified.get("metadatas", [])

    classified_count = 0
    if ids:
        update_ids, update_metas = [], []
        for chunk_id, text, meta in zip(ids, documents, metadatas):
            pillar, dimension = classifier.classify_chunk(text, meta)
            if dimension:
                meta_copy = dict(meta)
                meta_copy["pillar"] = pillar
                meta_copy["dimension"] = dimension
                meta_copy["classified"] = "true"
                update_ids.append(chunk_id)
                update_metas.append(meta_copy)
                classified_count += 1
        if update_ids:
            store.update_metadata(update_ids, update_metas)

    log(f"- Classified: **{classified_count}** / {len(ids)} unclassified", lines)
    log("", lines)

    # ── Step 3: Extract & Skill Files ────────────────────────────────
    log("## Step 3: Extract Traits → Skill Files", lines)
    profile = PersonaProfile.load()
    extractor = PersonaExtractor()

    dims_with_data = []
    for dim_name in DIMENSIONS:
        count = store.count_by_dimension(dim_name)
        if count > 0:
            dims_with_data.append(dim_name)

    # Only re-extract dimensions that need it (no traits yet or significant new data)
    needs_extraction = []
    for dim_name in dims_with_data:
        dim = profile.get_dimension(dim_name)
        if not dim or not dim.traits:
            needs_extraction.append(dim_name)
        else:
            current = store.count_by_dimension(dim_name)
            if current - dim.evidence_count >= 5:
                needs_extraction.append(dim_name)

    if needs_extraction:
        log(f"- Extracting: {', '.join(needs_extraction)}", lines)
        from config import MAX_EVIDENCE_PER_DIMENSION
        chunks_by_dim = {}
        for dim_name in needs_extraction:
            results = store.search_by_dimension("", dim_name, n_results=MAX_EVIDENCE_PER_DIMENSION)
            chunks_by_dim[dim_name] = [r["text"] for r in results]
        profile = extractor.incremental_update(needs_extraction, chunks_by_dim, profile)
    else:
        log("- All dimensions up to date, skipping extraction", lines)

    written = write_all_skill_files(profile.dimensions)
    populated = sum(1 for d in profile.dimensions.values() if d.traits)
    log(f"- Skill files: **{len(written)}** written | **{populated}/13** populated", lines)
    log("", lines)

    # ── Step 4: Test Each Dimension ──────────────────────────────────
    log("## Step 4: Dimension Tests (RAG + LLM Response)", lines)
    log("", lines)

    persona_prompt = build_persona_from_skills()
    system = (
        f"{persona_prompt}\n\n"
        "Always include a '> **Why I think this:**' block at the end citing specific data points, "
        "memories, or patterns that informed your answer. Be specific — reference actual data."
    )

    passed = 0
    failed = 0
    total_dims = len(DIMENSION_QUESTIONS)

    for dim_name, question in DIMENSION_QUESTIONS.items():
        log(f"### {dim_name}", lines)

        dim = profile.get_dimension(dim_name)
        traits = sum(1 for v in (dim.traits or {}).values() if v) if dim else 0
        chunks = store.count_by_dimension(dim_name)
        log(f"- Chunks: {chunks} | Traits: {traits} | Confidence: {dim.confidence:.0%}" if dim else f"- No dimension data", lines)

        # RAG check
        rag_results = store.search(question, n_results=5)
        rag_text = " ".join(r["text"].lower() for r in rag_results)
        expected = EXPECTED_KEYWORDS.get(dim_name, [])
        rag_hits = [kw for kw in expected if kw.lower() in rag_text]
        log(f"- RAG hits: {len(rag_hits)}/{len(expected)} ({', '.join(rag_hits) if rag_hits else 'none'})", lines)

        # LLM response
        try:
            # Include relevant memories in the prompt
            mem_context = ""
            if rag_results:
                mem_lines = []
                for r in rag_results[:3]:
                    mem_lines.append(r["text"][:300])
                mem_context = "\n\nRelevant memories:\n" + "\n---\n".join(mem_lines)

            response = chat_completion(
                system=system + mem_context,
                messages=[{"role": "user", "content": question}],
                max_tokens=500,
            )

            # Check for Why block
            has_why = "why i think this" in response.lower() or "why:" in response.lower() or "because" in response.lower()

            # Check for keyword citations
            resp_lower = response.lower()
            cited = [kw for kw in expected if kw.lower() in resp_lower]

            dim_pass = has_why and (len(cited) > 0 or len(expected) == 0)

            if dim_pass:
                passed += 1
                status = "PASS"
            else:
                failed += 1
                status = "FAIL"

            log(f"- Response has reasoning: {'yes' if has_why else 'NO'}", lines)
            log(f"- Keywords cited: {len(cited)}/{len(expected)} ({', '.join(cited) if cited else 'none'})", lines)
            log(f"- **{status}**", lines)
            log(f"- Response preview: _{response[:200].replace(chr(10), ' ')}_", lines)

        except Exception as e:
            failed += 1
            log(f"- LLM ERROR: {e}", lines)
            log(f"- **FAIL**", lines)

        log("", lines)

    # ── Summary ──────────────────────────────────────────────────────
    elapsed = time.time() - start
    log("## Summary", lines)
    log(f"- **{passed}/{total_dims} dimensions passed** | {failed} failed", lines)
    log(f"- Total chunks: {store.count()}", lines)
    log(f"- Populated dimensions: {populated}/13", lines)
    log(f"- Elapsed: {elapsed:.1f}s", lines)
    log("", lines)

    # Write log file
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    LOG_FILE.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nLog saved to: {LOG_FILE}")

    return passed, total_dims


if __name__ == "__main__":
    passed, total = run()
    if passed == total:
        print(f"\nALL {total} DIMENSIONS PASSED")
    else:
        print(f"\n{passed}/{total} passed — {total - passed} need work")
