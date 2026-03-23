"""RAPTOR-style hierarchical indexing for Apple Notes and other documents.

Creates multi-level summaries:
  Level 0: Raw chunks (already in ChromaDB)
  Level 1: Topic summaries (grouped by category/sub_category)
  Level 2: Pillar summaries (MIND, BODY, SOUL, etc.)
  Level 3: Global persona summary

Each level is stored as a chunk with type="raptor_L{N}" so they can be
retrieved alongside raw chunks for richer context.
"""

import json
import sqlite3
from collections import defaultdict
from datetime import datetime, timezone

from config import SINGULARITY_DIR
from memory.chunker import Chunk, _ensure_metadata
from memory.vectorstore import VectorStore
from persona.dimensions import DIMENSIONS

WEEKLY_STATS_DB = SINGULARITY_DIR / "logs" / "weekly_stats.db"


def _get_notes_by_category() -> dict[str, list[dict]]:
    """Read notes from notes_index grouped by category/sub_category."""
    if not WEEKLY_STATS_DB.exists():
        return {}

    conn = sqlite3.connect(str(WEEKLY_STATS_DB))
    conn.row_factory = sqlite3.Row

    rows = conn.execute("""
        SELECT title, snippet, semantic_summary, category, sub_category,
               topic_tags, week_id, modified_at
        FROM notes_index
        WHERE snippet != '' AND length(snippet) > 10
        ORDER BY category, sub_category, modified_at DESC
    """).fetchall()

    conn.close()

    grouped = defaultdict(list)
    for r in rows:
        cat = r["category"] or "other"
        sub = r["sub_category"] or ""
        key = f"{cat}/{sub}" if sub else cat
        grouped[key].append(dict(r))

    return dict(grouped)


def build_level1_summaries(notes_by_topic: dict[str, list[dict]]) -> list[Chunk]:
    """Level 1: Create a summary chunk for each topic (category/sub_category).

    This is a deterministic summary — no LLM call, just aggregation.
    """
    chunks = []

    for topic, notes in notes_by_topic.items():
        # Collect all tags across notes in this topic
        all_tags = set()
        titles = []
        snippets = []

        for n in notes:
            titles.append(n.get("title", ""))
            if n.get("snippet"):
                snippets.append(n["snippet"][:150])
            try:
                tags = json.loads(n.get("topic_tags", "[]"))
                all_tags.update(tags)
            except (json.JSONDecodeError, TypeError):
                pass

        # Build summary text
        text_parts = [
            f"Topic Summary: {topic}",
            f"Notes: {len(notes)}",
            f"Tags: {', '.join(sorted(all_tags)) if all_tags else 'none'}",
            f"Titles: {'; '.join(t for t in titles[:10] if t)}",
            "",
            "Key content:",
        ]
        for s in snippets[:8]:
            text_parts.append(f"  - {s}")

        text = "\n".join(text_parts)

        # Map topic to dimension
        cat = topic.split("/")[0]
        cat_dim_map = {
            "learning": "learning", "learning/ai": "learning",
            "learning/books": "learning", "learning/product": "learning",
            "career": "professional", "journaling": "life",
            "wellness": "wellness", "other": "life",
        }
        dimension = cat_dim_map.get(cat, cat_dim_map.get(topic, "life"))
        pillar = DIMENSIONS.get(dimension, {}).get("pillar", "PURPOSE")

        latest_date = max((n.get("modified_at", "") for n in notes), default="")

        chunks.append(Chunk(
            text=text,
            metadata=_ensure_metadata({
                "source": "raptor",
                "conversation_id": f"raptor_L1_{topic}",
                "title": f"Topic: {topic}",
                "timestamp": f"{latest_date}T00:00:00+00:00" if latest_date else "",
                "msg_timestamp": f"{latest_date}T00:00:00+00:00" if latest_date else "",
                "role": "system",
                "type": "raptor_L1",
                "pillar": pillar,
                "dimension": dimension,
                "classified": "true",
            }),
        ))

    return chunks


def build_level2_summaries(notes_by_topic: dict[str, list[dict]]) -> list[Chunk]:
    """Level 2: Pillar-level summaries aggregating all topics under each pillar."""
    # Group topics by pillar
    pillar_topics = defaultdict(list)

    cat_pillar_map = {
        "learning/ai": "MIND", "learning/books": "MIND", "learning/product": "MIND",
        "career": "MIND", "journaling": "PURPOSE", "wellness": "BODY", "other": "PURPOSE",
    }

    for topic, notes in notes_by_topic.items():
        cat = topic.split("/")[0]
        pillar = cat_pillar_map.get(cat, cat_pillar_map.get(topic, "PURPOSE"))
        pillar_topics[pillar].append((topic, notes))

    chunks = []
    for pillar, topics in pillar_topics.items():
        total_notes = sum(len(n) for _, n in topics)
        topic_list = [f"{t} ({len(n)} notes)" for t, n in topics]

        all_tags = set()
        for _, notes in topics:
            for n in notes:
                try:
                    tags = json.loads(n.get("topic_tags", "[]"))
                    all_tags.update(tags)
                except (json.JSONDecodeError, TypeError):
                    pass

        text_parts = [
            f"Pillar Summary: {pillar}",
            f"Total notes: {total_notes}",
            f"Topics: {'; '.join(topic_list)}",
            f"Key tags: {', '.join(sorted(all_tags)[:20]) if all_tags else 'none'}",
        ]
        text = "\n".join(text_parts)

        # Map pillar to dimension
        pillar_dim = {"MIND": "learning", "BODY": "wellness", "SOUL": "creative",
                      "SOCIAL": "relationships", "PURPOSE": "goals"}
        dimension = pillar_dim.get(pillar, "life")

        chunks.append(Chunk(
            text=text,
            metadata=_ensure_metadata({
                "source": "raptor",
                "conversation_id": f"raptor_L2_{pillar}",
                "title": f"Pillar: {pillar}",
                "timestamp": datetime.now(tz=timezone.utc).isoformat(),
                "msg_timestamp": datetime.now(tz=timezone.utc).isoformat(),
                "role": "system",
                "type": "raptor_L2",
                "pillar": pillar,
                "dimension": dimension,
                "classified": "true",
            }),
        ))

    return chunks


def build_raptor_index() -> dict:
    """Build the full RAPTOR hierarchical index and ingest into ChromaDB.

    Returns summary of what was created.
    """
    notes_by_topic = _get_notes_by_category()
    if not notes_by_topic:
        return {"status": "no_notes", "topics": 0}

    print(f"Building RAPTOR index from {sum(len(v) for v in notes_by_topic.values())} notes across {len(notes_by_topic)} topics...")

    # Level 1: Topic summaries
    l1_chunks = build_level1_summaries(notes_by_topic)
    print(f"  Level 1 (topic summaries): {len(l1_chunks)} chunks")

    # Level 2: Pillar summaries
    l2_chunks = build_level2_summaries(notes_by_topic)
    print(f"  Level 2 (pillar summaries): {len(l2_chunks)} chunks")

    # Ingest all levels
    vs = VectorStore()
    all_chunks = l1_chunks + l2_chunks
    count = vs.ingest(all_chunks)
    print(f"  Ingested {count} RAPTOR chunks")

    # Return hierarchy
    hierarchy = {}
    for topic in sorted(notes_by_topic.keys()):
        parts = topic.split("/")
        if len(parts) >= 2:
            parent = parts[0]
            child = "/".join(parts[1:])
        else:
            parent = topic
            child = None

        if parent not in hierarchy:
            hierarchy[parent] = {"notes": 0, "sub_topics": {}}

        if child:
            hierarchy[parent]["sub_topics"][child] = len(notes_by_topic[topic])
        else:
            hierarchy[parent]["notes"] += len(notes_by_topic[topic])

    return {
        "status": "ok",
        "topics": len(notes_by_topic),
        "level1_chunks": len(l1_chunks),
        "level2_chunks": len(l2_chunks),
        "total_ingested": count,
        "hierarchy": hierarchy,
    }


def print_hierarchy(result: dict) -> None:
    """Pretty-print the RAPTOR hierarchy."""
    hierarchy = result.get("hierarchy", {})
    print("\nRAPTOR Hierarchy:")
    for parent, info in sorted(hierarchy.items()):
        direct = info.get("notes", 0)
        subs = info.get("sub_topics", {})
        total = direct + sum(subs.values())
        print(f"\n  {parent} ({total} notes)")
        if direct > 0:
            print(f"    (direct): {direct}")
        for sub, count in sorted(subs.items()):
            print(f"    {sub}: {count}")


if __name__ == "__main__":
    result = build_raptor_index()
    print(json.dumps({k: v for k, v in result.items() if k != "hierarchy"}, indent=2))
    print_hierarchy(result)
