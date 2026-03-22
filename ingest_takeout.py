#!/usr/bin/env python3
"""Auto-loop ingestion of Google Takeout data (Gemini + YouTube).

Parses, chunks, classifies, ingests, and re-extracts persona dimensions.
Retries on failure until successful.
"""

import json
import sys
import time
import traceback
from pathlib import Path

from config import NORMALIZED_DIR
from memory.chunker import Chunker
from memory.vectorstore import VectorStore
from persona.classifier import ChunkClassifier


def ingest_gemini_html():
    """Parse and ingest Gemini MyActivity.html."""
    from parsers.gemini_html_parser import GeminiHTMLParser

    html_path = Path("data/raw/gemini/MyActivity.html")
    if not html_path.exists():
        print("  Gemini HTML not found, skipping.")
        return 0

    print("  Parsing Gemini MyActivity.html...")
    parser = GeminiHTMLParser()
    conversations = parser.parse(html_path)
    print(f"  Found {len(conversations)} Gemini conversations")

    if not conversations:
        return 0

    # Save normalized
    parser.save_normalized(conversations, NORMALIZED_DIR)
    print(f"  Saved normalized JSON")

    # Chunk
    chunker = Chunker()
    conv_dicts = [c.to_dict() for c in conversations]
    chunks = chunker.chunk_conversations(conv_dicts)
    print(f"  Created {len(chunks)} chunks")

    # Classify chunks before ingestion
    classifier = ChunkClassifier()
    classified_count = 0
    for chunk in chunks:
        pillar, dimension = classifier.classify_chunk(chunk.text, chunk.metadata)
        if dimension:
            chunk.metadata["pillar"] = pillar
            chunk.metadata["dimension"] = dimension
            chunk.metadata["classified"] = "true"
            classified_count += 1

    print(f"  Pre-classified {classified_count}/{len(chunks)} chunks")

    # Ingest
    store = VectorStore()
    count = store.ingest(chunks)
    print(f"  Ingested {count} Gemini chunks")
    return count


def ingest_youtube_html():
    """Parse and ingest YouTube watch/search history."""
    from parsers.youtube_parser import YouTubeParser

    watch_path = Path("data/raw/youtube/watch-history.html")
    search_path = Path("data/raw/youtube/search-history.html")

    total = 0
    parser = YouTubeParser()
    store = VectorStore()
    chunker = Chunker()
    classifier = ChunkClassifier()

    for name, path in [("watch", watch_path), ("search", search_path)]:
        if not path.exists():
            print(f"  YouTube {name} history not found, skipping.")
            continue

        print(f"  Parsing YouTube {name} history...")
        conversations = parser.parse(path)
        print(f"  Found {len(conversations)} {name} entries")

        if not conversations:
            continue

        # Chunk
        conv_dicts = [c.to_dict() for c in conversations]
        chunks = chunker.chunk_conversations(conv_dicts)

        # Classify — YouTube data maps to entertainment/vibe/learning
        for chunk in chunks:
            pillar, dimension = classifier.classify_chunk(chunk.text, chunk.metadata)
            if not dimension:
                # Default YouTube to entertainment
                chunk.metadata["pillar"] = "SOUL"
                chunk.metadata["dimension"] = "entertainment"
                chunk.metadata["classified"] = "true"
            else:
                chunk.metadata["pillar"] = pillar
                chunk.metadata["dimension"] = dimension
                chunk.metadata["classified"] = "true"

        count = store.ingest(chunks)
        print(f"  Ingested {count} YouTube {name} chunks")
        total += count

    return total


def classify_unclassified():
    """Run classification on any remaining unclassified chunks."""
    from daily_loop import DailyLoop
    loop = DailyLoop()
    result = loop._classify_new_chunks()
    return result


def extract_dimensions():
    """Re-extract persona dimensions with the new data."""
    from daily_loop import DailyLoop
    loop = DailyLoop()
    result = loop._full_persona_extraction()
    return result


def save_snapshot():
    """Save evolution snapshot."""
    from daily_loop import DailyLoop
    loop = DailyLoop()
    return loop._save_snapshot()


def main():
    max_retries = 5
    steps = [
        ("Ingest Gemini conversations", ingest_gemini_html),
        ("Ingest YouTube history", ingest_youtube_html),
        ("Classify unclassified chunks", classify_unclassified),
        ("Extract persona dimensions", extract_dimensions),
        ("Save evolution snapshot", save_snapshot),
    ]

    for step_name, step_fn in steps:
        for attempt in range(1, max_retries + 1):
            print(f"\n{'='*60}")
            print(f"[{step_name}] Attempt {attempt}/{max_retries}")
            print(f"{'='*60}")
            try:
                result = step_fn()
                print(f"  SUCCESS: {result}")
                break
            except Exception as e:
                print(f"  FAILED: {e}")
                traceback.print_exc()
                if attempt < max_retries:
                    wait = attempt * 5
                    print(f"  Retrying in {wait}s...")
                    time.sleep(wait)
                else:
                    print(f"  GIVING UP on: {step_name}")

    # Final status
    print(f"\n{'='*60}")
    print("FINAL STATUS")
    print(f"{'='*60}")
    store = VectorStore()
    print(f"Total chunks: {store.count()}")

    from persona.profile import PersonaProfile
    profile = PersonaProfile.load()
    for name, dim in sorted(profile.dimensions.items()):
        status = "OK" if dim.traits else "empty"
        print(f"  {dim.display_name:25s} | {status:5s} | confidence: {dim.confidence:.0%} | evidence: {dim.evidence_count}")


if __name__ == "__main__":
    main()
