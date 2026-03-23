#!/usr/bin/env python3
"""Daily learning loop: sync → classify → cluster → update persona → snapshot.

Usage:
    python daily_loop.py                      # incremental daily loop
    python daily_loop.py --full               # full re-extraction of all dimensions
    python daily_loop.py --sync-only          # just sync connectors, no persona update
    python daily_loop.py --classify-only      # just classify unclassified chunks
    python daily_loop.py --cluster-only       # just run document clustering
    python daily_loop.py --dimension code     # update only one dimension
    python daily_loop.py --status             # show sync state and dimension health
    python daily_loop.py --schedule           # run twice-daily at 8am and 8pm (blocks)
"""

import argparse
import json
import time
from datetime import datetime, timezone

from config import (
    SYNC_STATE_FILE,
    INCREMENTAL_DIMENSION_THRESHOLD,
    MAX_EVIDENCE_PER_DIMENSION,
    LOGS_DIR,
)
from connectors import ALL_CONNECTORS
from memory.vectorstore import VectorStore
from persona.classifier import ChunkClassifier
from persona.dimensions import DIMENSIONS
from persona.extractor import PersonaExtractor
from persona.profile import PersonaProfile


class DailyLoop:
    """Orchestrates the daily learning cycle."""

    def __init__(self):
        self.vector_store = VectorStore()
        self.classifier = ChunkClassifier()
        self.extractor = PersonaExtractor()

    def run(self, full: bool = False, cluster: bool = True) -> dict:
        """Execute the full daily learning loop."""
        start = time.time()
        results = {"timestamp": datetime.now(tz=timezone.utc).isoformat()}

        print("=" * 60)
        print(f"Daily Loop — {results['timestamp']}")
        print("=" * 60)

        # Step 1: Sync all connectors
        print("\n[1/6] Syncing data sources...")
        results["sync"] = self._sync_all_sources()

        # Step 2: Classify unclassified chunks
        print("\n[2/6] Classifying new chunks...")
        results["classify"] = self._classify_new_chunks()

        # Step 3: Cluster documents by context
        if cluster:
            print("\n[3/6] Clustering documents by context...")
            results["cluster"] = self._cluster_documents()
        else:
            print("\n[3/6] Clustering skipped.")
            results["cluster"] = {"status": "skipped"}

        # Step 4: Detect changed dimensions
        print("\n[4/6] Detecting changed dimensions...")
        changed_dims = self._detect_changed_dimensions()
        results["changed_dimensions"] = changed_dims

        # Step 5: Update persona
        if full:
            print("\n[5/6] Full persona extraction (all dimensions)...")
            results["persona"] = self._full_persona_extraction()
        elif changed_dims:
            print(f"\n[5/6] Incremental persona update ({len(changed_dims)} dimensions)...")
            results["persona"] = self._incremental_persona_update(changed_dims)
        else:
            print("\n[5/6] No dimensions changed — skipping persona update.")
            results["persona"] = {"status": "no_changes"}

        # Step 6: Save evolution snapshot
        print("\n[6/6] Saving evolution snapshot...")
        results["snapshot"] = self._save_snapshot()

        elapsed = time.time() - start
        results["elapsed_seconds"] = round(elapsed, 1)
        print(f"\nDone in {elapsed:.1f}s")

        # Log results
        self._log_results(results)

        return results

    # ------------------------------------------------------------------
    # Step 1: Sync
    # ------------------------------------------------------------------

    def _sync_all_sources(self) -> dict:
        """Sync all connectors incrementally."""
        sync_results = {}
        for name, connector_cls in ALL_CONNECTORS.items():
            try:
                connector = connector_cls()
                chunks = connector.sync()
                if chunks:
                    count = self.vector_store.ingest(chunks)
                    sync_results[name] = {"chunks": count, "status": "ok"}
                    print(f"  {name}: {count} new chunks")
                else:
                    sync_results[name] = {"chunks": 0, "status": "ok"}
                    print(f"  {name}: up to date")
            except Exception as e:
                sync_results[name] = {"chunks": 0, "status": "error", "error": str(e)}
                print(f"  {name}: ERROR — {e}")
        return sync_results

    # ------------------------------------------------------------------
    # Step 2: Classify
    # ------------------------------------------------------------------

    def _classify_new_chunks(self) -> dict:
        """Classify unclassified chunks using Tier 1 (rules) then Tier 2 (LLM)."""
        unclassified = self.vector_store.get_unclassified_chunks(limit=1000)

        ids = unclassified.get("ids", [])
        documents = unclassified.get("documents", [])
        metadatas = unclassified.get("metadatas", [])

        if not ids:
            print("  No unclassified chunks.")
            return {"tier1": 0, "tier2": 0, "total": 0}

        print(f"  Found {len(ids)} unclassified chunks.")

        tier1_count = 0
        tier2_candidates = []
        updates_ids = []
        updates_meta = []

        # Tier 1: Rule-based
        for i, (chunk_id, text, meta) in enumerate(zip(ids, documents, metadatas)):
            pillar, dimension = self.classifier.classify_chunk(text, meta)
            if dimension:
                meta_copy = dict(meta)
                meta_copy["pillar"] = pillar
                meta_copy["dimension"] = dimension
                meta_copy["classified"] = "true"
                updates_ids.append(chunk_id)
                updates_meta.append(meta_copy)
                tier1_count += 1
            else:
                tier2_candidates.append({
                    "id": chunk_id,
                    "text": text,
                    "metadata": meta,
                    "index": i,
                })

        # Apply Tier 1 updates
        if updates_ids:
            self.vector_store.update_metadata(updates_ids, updates_meta)
        print(f"  Tier 1 (rules): classified {tier1_count}")

        # Tier 2: LLM batch classify (in batches of 50)
        tier2_count = 0
        for batch_start in range(0, len(tier2_candidates), 50):
            batch = tier2_candidates[batch_start:batch_start + 50]
            results = self.classifier.batch_classify_llm(batch)

            batch_ids = []
            batch_metas = []
            for item, (pillar, dimension) in zip(batch, results):
                if dimension:
                    meta_copy = dict(item["metadata"])
                    meta_copy["pillar"] = pillar
                    meta_copy["dimension"] = dimension
                    meta_copy["classified"] = "true"
                    batch_ids.append(item["id"])
                    batch_metas.append(meta_copy)
                    tier2_count += 1

            if batch_ids:
                self.vector_store.update_metadata(batch_ids, batch_metas)

        print(f"  Tier 2 (LLM):   classified {tier2_count}")

        return {
            "tier1": tier1_count,
            "tier2": tier2_count,
            "total": tier1_count + tier2_count,
            "remaining": len(tier2_candidates) - tier2_count,
        }

    # ------------------------------------------------------------------
    # Step 3: Cluster documents
    # ------------------------------------------------------------------

    def _cluster_documents(self) -> dict:
        """Cluster all documents by embedding similarity and label clusters."""
        from memory.clusterer import DocumentClusterer

        clusterer = DocumentClusterer(self.vector_store)
        total = self.vector_store.count()
        if total < 20:
            print(f"  Too few documents ({total}) for clustering. Skipping.")
            return {"status": "skipped", "reason": "too_few_documents"}

        result = clusterer.run()
        print(f"  Clusters: {result['clusters']}, Updated: {result['updated']} chunks.")
        return result

    # ------------------------------------------------------------------
    # Step 4: Detect changed dimensions
    # ------------------------------------------------------------------

    def _detect_changed_dimensions(self) -> list[str]:
        """Detect dimensions with significant new evidence since last extraction."""
        profile = PersonaProfile.load()
        changed = []

        for dim_name in DIMENSIONS:
            current_count = self.vector_store.count_by_dimension(dim_name)
            stored_count = 0
            dim = profile.get_dimension(dim_name)
            if dim:
                stored_count = dim.evidence_count

            delta = current_count - stored_count
            if delta >= INCREMENTAL_DIMENSION_THRESHOLD:
                changed.append(dim_name)
                print(f"  {dim_name}: {delta} new chunks (threshold: {INCREMENTAL_DIMENSION_THRESHOLD})")

        if not changed:
            print("  No dimensions exceeded the change threshold.")

        return changed

    # ------------------------------------------------------------------
    # Step 4: Persona update
    # ------------------------------------------------------------------

    def _get_chunks_by_dimension(self, dimensions: list[str]) -> dict[str, list[str]]:
        """Fetch chunk texts from ChromaDB grouped by dimension."""
        chunks_by_dim = {}
        for dim_name in dimensions:
            results = self.vector_store.search_by_dimension(
                query="",  # empty query returns by distance
                dimension=dim_name,
                n_results=MAX_EVIDENCE_PER_DIMENSION,
            )
            chunks_by_dim[dim_name] = [r["text"] for r in results]
        return chunks_by_dim

    def _full_persona_extraction(self) -> dict:
        """Full re-extraction of all dimensions."""
        all_dims = list(DIMENSIONS.keys())
        chunks_by_dim = self._get_chunks_by_dimension(all_dims)

        populated = {k: v for k, v in chunks_by_dim.items() if v}
        print(f"  Extracting {len(populated)} dimensions with data...")

        profile = self.extractor.extract_all_dimensions(chunks_by_dim)
        return {
            "status": "full_extraction",
            "dimensions_extracted": len(populated),
        }

    def _incremental_persona_update(self, changed_dims: list[str]) -> dict:
        """Update only changed dimensions."""
        chunks_by_dim = self._get_chunks_by_dimension(changed_dims)

        populated = {k: v for k, v in chunks_by_dim.items() if v}
        print(f"  Updating {len(populated)} dimensions...")

        profile = self.extractor.incremental_update(changed_dims, chunks_by_dim)
        return {
            "status": "incremental",
            "dimensions_updated": list(populated.keys()),
        }

    # ------------------------------------------------------------------
    # Step 5: Snapshot
    # ------------------------------------------------------------------

    def _save_snapshot(self) -> dict:
        """Save daily evolution snapshot and regenerate skill files."""
        from persona.skills import write_all_skill_files

        profile = PersonaProfile.load()
        snapshot = profile.snapshot_all()
        populated = sum(1 for d in snapshot["dimensions"].values() if d.get("traits"))
        print(f"  Snapshot saved: {populated}/{len(DIMENSIONS)} dimensions populated.")

        # Regenerate skill files from latest dimension data
        written = write_all_skill_files(profile.dimensions)
        print(f"  Skill files regenerated: {len(written)} files.")

        return {"date": snapshot["date"], "populated_dimensions": populated}

    # ------------------------------------------------------------------
    # Logging
    # ------------------------------------------------------------------

    def _log_results(self, results: dict) -> None:
        """Append results to daily log file."""
        log_file = LOGS_DIR / "daily_loop.log"
        with open(log_file, "a") as f:
            f.write(json.dumps(results, default=str) + "\n")

    # ------------------------------------------------------------------
    # Status
    # ------------------------------------------------------------------

    @staticmethod
    def show_status() -> None:
        """Print sync state and dimension health."""
        print("=" * 60)
        print("AI Twin — Status")
        print("=" * 60)

        # Sync state
        print("\n--- Connector Sync State ---")
        if SYNC_STATE_FILE.exists():
            state = json.loads(SYNC_STATE_FILE.read_text())
            for name, info in state.items():
                last = info.get("last_sync")
                if last:
                    ts = datetime.fromtimestamp(last, tz=timezone.utc).strftime("%Y-%m-%d %H:%M")
                else:
                    ts = "never"
                total = info.get("chunks_total", 0)
                print(f"  {name:20s} | last sync: {ts} | total chunks: {total}")
        else:
            print("  No sync state found.")

        # Persona dimensions
        print("\n--- Persona Dimensions ---")
        profile = PersonaProfile.load()
        for dim_name, dim in sorted(profile.dimensions.items()):
            status = "populated" if dim.traits else "empty"
            updated = dim.last_updated[:10] if dim.last_updated else "never"
            print(
                f"  {dim.display_name:25s} | {status:10s} | "
                f"confidence: {dim.confidence:.0%} | evidence: {dim.evidence_count} | "
                f"updated: {updated}"
            )

        # Snapshots
        print("\n--- Evolution Snapshots ---")
        snapshots = PersonaProfile.list_snapshots()
        if snapshots:
            print(f"  {len(snapshots)} snapshots: {snapshots[0]} → {snapshots[-1]}")
        else:
            print("  No snapshots yet.")

        # Vector store
        vs = VectorStore()
        print(f"\n--- Vector Store ---")
        print(f"  Total chunks: {vs.count()}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _run_scheduled(loop: DailyLoop):
    """Run the daily loop twice daily at 8am and 8pm.

    Blocks forever. Use Ctrl+C to stop.
    For production, prefer a macOS LaunchAgent or crontab instead.
    """
    import sched
    import calendar

    scheduler = sched.scheduler(time.time, time.sleep)

    def _next_run_time() -> float:
        """Find the next 8:00 or 20:00 in local time."""
        now = datetime.now()
        today_8am = now.replace(hour=8, minute=0, second=0, microsecond=0)
        today_8pm = now.replace(hour=20, minute=0, second=0, microsecond=0)

        candidates = [today_8am, today_8pm]
        # Add tomorrow's times
        from datetime import timedelta
        tomorrow = now + timedelta(days=1)
        candidates.append(tomorrow.replace(hour=8, minute=0, second=0, microsecond=0))
        candidates.append(tomorrow.replace(hour=20, minute=0, second=0, microsecond=0))

        future = [t for t in candidates if t > now]
        next_time = min(future)
        return next_time.timestamp()

    def _run_and_reschedule():
        print(f"\n{'=' * 60}")
        print(f"Scheduled run — {datetime.now().isoformat()}")
        try:
            loop.run(full=False, cluster=True)
        except Exception as e:
            print(f"ERROR in scheduled run: {e}")
        # Schedule next run
        next_ts = _next_run_time()
        next_dt = datetime.fromtimestamp(next_ts)
        print(f"\nNext run scheduled for: {next_dt.strftime('%Y-%m-%d %H:%M')}")
        scheduler.enterabs(next_ts, 1, _run_and_reschedule)

    # Run once immediately
    _run_and_reschedule()
    print("\nScheduler running (8am + 8pm). Press Ctrl+C to stop.")
    try:
        scheduler.run()
    except KeyboardInterrupt:
        print("\nScheduler stopped.")


def main():
    parser = argparse.ArgumentParser(description="AI Twin Daily Learning Loop")
    parser.add_argument("--full", action="store_true", help="Full re-extraction of all dimensions")
    parser.add_argument("--sync-only", action="store_true", help="Only sync connectors")
    parser.add_argument("--classify-only", action="store_true", help="Only classify unclassified chunks")
    parser.add_argument("--dimension", type=str, help="Update a single dimension by name")
    parser.add_argument("--status", action="store_true", help="Show sync and dimension status")
    parser.add_argument("--cluster-only", action="store_true", help="Only run document clustering")
    parser.add_argument("--schedule", action="store_true",
                        help="Run twice-daily at 8am and 8pm (blocks forever)")

    args = parser.parse_args()
    loop = DailyLoop()

    if args.status:
        DailyLoop.show_status()
        return

    if args.schedule:
        _run_scheduled(loop)
        return

    if args.cluster_only:
        print("Cluster-only mode...")
        results = loop._cluster_documents()
        print(json.dumps(results, indent=2, default=str))
        return

    if args.sync_only:
        print("Sync-only mode...")
        results = loop._sync_all_sources()
        print(json.dumps(results, indent=2, default=str))
        return

    if args.classify_only:
        print("Classify-only mode...")
        results = loop._classify_new_chunks()
        print(json.dumps(results, indent=2, default=str))
        return

    if args.dimension:
        dim = args.dimension
        if dim not in DIMENSIONS:
            print(f"Unknown dimension: {dim}")
            print(f"Available: {', '.join(DIMENSIONS.keys())}")
            return
        print(f"Updating single dimension: {dim}")
        chunks_by_dim = loop._get_chunks_by_dimension([dim])
        if chunks_by_dim.get(dim):
            profile = loop.extractor.incremental_update([dim], chunks_by_dim)
            print(f"Done. {dim} updated with {len(chunks_by_dim[dim])} chunks.")
        else:
            print(f"No chunks found for dimension: {dim}")
        return

    # Default: full daily loop
    results = loop.run(full=args.full)
    print("\n" + json.dumps(results, indent=2, default=str))


if __name__ == "__main__":
    main()
