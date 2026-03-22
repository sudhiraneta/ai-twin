#!/usr/bin/env python3
"""
sync_watcher.py — Auto-learns from Singularity data on every loop cycle.

The Singularity loop.py runs every 5 minutes and writes to state.json.
This watcher detects changes and runs the FULL learning pipeline:

  1. Sync: pull new chunks from all connectors (notes, browser, tasks, analytics...)
  2. Classify: assign each chunk to a persona dimension (code, wellness, goals...)
  3. Extract: if enough new data, re-extract traits via LLM
  4. Skill files: regenerate the per-dimension .md files
  5. Persona: the twin reads fresh skill files on every query — no restart needed

Usage:
    python sync_watcher.py              # watch and auto-learn (runs forever)
    python sync_watcher.py --once       # run full pipeline once and exit
    python sync_watcher.py --interval 60 # check every 60s (default: 30)
"""

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

from config import SINGULARITY_DIR, INCREMENTAL_DIMENSION_THRESHOLD

SINGULARITY_STATE = SINGULARITY_DIR / "state.json"


def get_singularity_last_run() -> float:
    """Read the Singularity loop's last_run timestamp."""
    if not SINGULARITY_STATE.exists():
        return 0
    try:
        state = json.loads(SINGULARITY_STATE.read_text())
        return state.get("last_run", 0)
    except Exception:
        return 0


def run_full_pipeline() -> dict:
    """Run the complete learning pipeline: sync → classify → extract → skill files."""
    from connectors import ALL_CONNECTORS
    from memory.vectorstore import VectorStore
    from persona.classifier import ChunkClassifier
    from persona.extractor import PersonaExtractor
    from persona.profile import PersonaProfile
    from persona.dimensions import DIMENSIONS
    from persona.skills import write_all_skill_files

    store = VectorStore()
    results = {"timestamp": datetime.now(tz=timezone.utc).isoformat()}

    # ── Step 1: Sync all connectors ──────────────────────────────────
    sync_results = {}
    total_new = 0
    for name, cls in ALL_CONNECTORS.items():
        try:
            connector = cls()
            chunks = connector.sync()
            if chunks:
                count = store.ingest(chunks)
                sync_results[name] = count
                total_new += count
            else:
                sync_results[name] = 0
        except Exception as e:
            sync_results[name] = f"ERROR: {e}"

    results["sync"] = sync_results
    results["new_chunks"] = total_new

    if total_new == 0:
        results["action"] = "no_new_data"
        return results

    # ── Step 2: Classify unclassified chunks ─────────────────────────
    classifier = ChunkClassifier()
    unclassified = store.get_unclassified_chunks(limit=1000)
    ids = unclassified.get("ids", [])
    documents = unclassified.get("documents", [])
    metadatas = unclassified.get("metadatas", [])

    classified_count = 0
    if ids:
        update_ids = []
        update_metas = []
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

    results["classified"] = classified_count

    # ── Step 3: Detect changed dimensions ────────────────────────────
    profile = PersonaProfile.load()
    changed_dims = []
    for dim_name in DIMENSIONS:
        current_count = store.count_by_dimension(dim_name)
        stored_count = 0
        dim = profile.get_dimension(dim_name)
        if dim:
            stored_count = dim.evidence_count
        delta = current_count - stored_count
        if delta >= INCREMENTAL_DIMENSION_THRESHOLD:
            changed_dims.append(dim_name)

    results["changed_dimensions"] = changed_dims

    # ── Step 4: Extract traits for changed dimensions ────────────────
    if changed_dims:
        extractor = PersonaExtractor()
        from config import MAX_EVIDENCE_PER_DIMENSION

        chunks_by_dim = {}
        for dim_name in changed_dims:
            dim_results = store.search_by_dimension("", dim_name, n_results=MAX_EVIDENCE_PER_DIMENSION)
            chunks_by_dim[dim_name] = [r["text"] for r in dim_results]

        profile = extractor.incremental_update(changed_dims, chunks_by_dim, profile)
        results["extracted"] = changed_dims
    else:
        results["extracted"] = []

    # ── Step 5: Regenerate skill files ───────────────────────────────
    written = write_all_skill_files(profile.dimensions)
    results["skill_files_updated"] = len(written)
    results["action"] = "updated" if changed_dims else "synced_only"
    results["total_chunks"] = store.count()

    return results


def print_results(results: dict):
    """Print pipeline results concisely."""
    now = datetime.now().strftime("%H:%M:%S")
    new = results.get("new_chunks", 0)
    classified = results.get("classified", 0)
    changed = results.get("changed_dimensions", [])
    extracted = results.get("extracted", [])
    total = results.get("total_chunks", 0)

    # Sync details
    sync = results.get("sync", {})
    active = [(k, v) for k, v in sync.items() if isinstance(v, int) and v > 0]
    errors = [(k, v) for k, v in sync.items() if isinstance(v, str)]

    print(f"[{now}] Pipeline complete:")
    print(f"  New chunks: {new} | Classified: {classified} | Total: {total}")

    if active:
        print(f"  Sources: {', '.join(f'{k}({v})' for k, v in active)}")
    if errors:
        for k, v in errors:
            print(f"  {k}: {v}")
    if extracted:
        print(f"  Dimensions updated: {', '.join(extracted)}")
        print(f"  Skill files regenerated → persona will reflect changes on next query")
    elif new > 0:
        print(f"  No dimensions crossed threshold ({INCREMENTAL_DIMENSION_THRESHOLD} new chunks needed)")


def watch(interval: int = 30):
    """Watch Singularity state.json and run full pipeline on changes."""
    print("=" * 60)
    print("  Sudhira-twin Auto-Learner")
    print(f"  Watching: {SINGULARITY_STATE}")
    print(f"  Interval: {interval}s | Ctrl+C to stop")
    print("=" * 60)

    last_seen_run = get_singularity_last_run()
    if last_seen_run:
        ts = datetime.fromtimestamp(last_seen_run).strftime("%Y-%m-%d %H:%M:%S")
        print(f"  Singularity last ran: {ts}")
    print()

    # Initial run
    results = run_full_pipeline()
    print_results(results)

    cycle = 0
    while True:
        try:
            time.sleep(interval)
        except KeyboardInterrupt:
            print("\n\nWatcher stopped.")
            break

        cycle += 1
        current_run = get_singularity_last_run()

        if current_run > last_seen_run:
            delta = current_run - last_seen_run
            print(f"\n{'─'*60}")
            print(f"  Singularity loop detected (cycle #{cycle}, {delta:.0f}s since last)")
            results = run_full_pipeline()
            print_results(results)
            last_seen_run = current_run
        else:
            t = datetime.now().strftime("%H:%M:%S")
            print(f"  [{t}] Watching... (cycle #{cycle})", end="\r")


def main():
    parser = argparse.ArgumentParser(description="Auto-learn from Singularity data")
    parser.add_argument("--once", action="store_true", help="Run full pipeline once and exit")
    parser.add_argument("--interval", type=int, default=30, help="Check interval in seconds")
    args = parser.parse_args()

    if args.once:
        results = run_full_pipeline()
        print_results(results)
    else:
        watch(interval=args.interval)


if __name__ == "__main__":
    main()
