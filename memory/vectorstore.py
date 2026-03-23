import hashlib
import json
import math
from datetime import datetime, timezone

import chromadb
from chromadb.config import Settings

from config import CHROMA_DIR, CHROMA_COLLECTION
from .chunker import Chunk
from .embeddings import EmbeddingEngine


class VectorStore:
    """ChromaDB-backed vector store for conversation memory."""

    def __init__(self):
        self.client = chromadb.PersistentClient(
            path=str(CHROMA_DIR),
            settings=Settings(anonymized_telemetry=False),
        )
        self.embedding_engine = EmbeddingEngine()
        self.collection = self.client.get_or_create_collection(
            name=CHROMA_COLLECTION,
            metadata={"hnsw:space": "cosine"},
        )

    @staticmethod
    def _chunk_id(text: str, metadata: dict) -> str:
        """Generate a deterministic hash-based ID for deduplication."""
        key = text + json.dumps(metadata, sort_keys=True, default=str)
        return f"chunk_{hashlib.md5(key.encode()).hexdigest()[:12]}"

    def ingest(self, chunks: list[Chunk], batch_size: int = 100) -> int:
        """Ingest chunks into the vector store. Returns count of added chunks."""
        total = 0

        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i + batch_size]
            texts = [c.text for c in batch]
            metadatas = [c.metadata for c in batch]
            ids = [self._chunk_id(c.text, c.metadata) for c in batch]

            embeddings = self.embedding_engine.embed(texts)

            self.collection.upsert(
                documents=texts,
                embeddings=embeddings,
                metadatas=metadatas,
                ids=ids,
            )
            total += len(batch)
            print(f"Ingested {total}/{len(chunks)} chunks...")

        return total

    def search(
        self,
        query: str,
        n_results: int = 10,
        where: dict | None = None,
        where_document: dict | None = None,
        max_distance: float | None = None,
    ) -> list[dict]:
        """Search for relevant memory chunks.

        Args:
            max_distance: If set, filter out results with cosine distance above
                          this threshold (0 = identical, 2 = opposite).
        """
        query_embedding = self.embedding_engine.embed_single(query)

        kwargs = {
            "query_embeddings": [query_embedding],
            "n_results": n_results,
        }
        if where:
            kwargs["where"] = where
        if where_document:
            kwargs["where_document"] = where_document

        results = self.collection.query(**kwargs)

        # Flatten results into a list of dicts
        items = []
        if results and results["documents"]:
            for j in range(len(results["documents"][0])):
                items.append({
                    "text": results["documents"][0][j],
                    "metadata": results["metadatas"][0][j] if results["metadatas"] else {},
                    "distance": results["distances"][0][j] if results["distances"] else None,
                    "id": results["ids"][0][j] if results["ids"] else None,
                })

        # Filter by relevance threshold
        if max_distance is not None:
            items = [item for item in items if item["distance"] is not None and item["distance"] <= max_distance]

        return items

    # ------------------------------------------------------------------
    # Typed search helpers
    # ------------------------------------------------------------------

    def search_user_messages(self, query: str, n_results: int = 10, max_distance: float | None = None) -> list[dict]:
        """Search only user messages (for persona-related queries)."""
        return self.search(query, n_results, where={"type": "user_message"}, max_distance=max_distance)

    def search_conversations(self, query: str, n_results: int = 10, max_distance: float | None = None) -> list[dict]:
        """Search conversation pairs (for context retrieval)."""
        return self.search(query, n_results, where={"type": "conversation_pair"}, max_distance=max_distance)

    def search_notes(self, query: str, n_results: int = 10, max_distance: float | None = None) -> list[dict]:
        return self.search(query, n_results, where={"type": "note"}, max_distance=max_distance)

    def search_browser(self, query: str, n_results: int = 10, max_distance: float | None = None) -> list[dict]:
        return self.search(query, n_results, where={"type": "browser_daily"}, max_distance=max_distance)

    def search_tasks(self, query: str, n_results: int = 10, max_distance: float | None = None) -> list[dict]:
        return self.search(query, n_results, where={"type": "task"}, max_distance=max_distance)

    def search_by_pillar(self, query: str, pillar: str, n_results: int = 10, max_distance: float | None = None) -> list[dict]:
        return self.search(query, n_results, where={"pillar": pillar}, max_distance=max_distance)

    def search_by_source(self, query: str, source: str, n_results: int = 10, max_distance: float | None = None) -> list[dict]:
        return self.search(query, n_results, where={"source": source}, max_distance=max_distance)

    # ------------------------------------------------------------------
    # Recency-weighted search with keyword reranking
    # ------------------------------------------------------------------

    @staticmethod
    def _keyword_overlap_score(query: str, document: str) -> float:
        """Simple keyword overlap as a reranking signal. Returns 0.0-1.0."""
        stopwords = {
            "the", "a", "an", "is", "are", "was", "were", "i", "you", "my", "me",
            "do", "does", "did", "have", "has", "had", "what", "when", "where",
            "how", "why", "which", "that", "this", "to", "for", "in", "on", "at",
            "of", "and", "or", "but", "with", "about", "would", "should", "could",
        }
        query_words = set(query.lower().split()) - stopwords
        if not query_words:
            return 0.0
        doc_words = set(document.lower().split())
        overlap = query_words & doc_words
        return len(overlap) / len(query_words)

    def search_with_recency(
        self,
        query: str,
        n_results: int = 10,
        where: dict | None = None,
        max_distance: float | None = None,
        recency_weight: float = 0.15,
        keyword_weight: float = 0.10,
    ) -> list[dict]:
        """Search with combined scoring: similarity + recency + keyword overlap.

        Over-fetches 2x results, then reranks by a blended score:
          (1 - recency_weight - keyword_weight) * similarity
          + recency_weight * recency_score
          + keyword_weight * keyword_overlap
        """
        raw_results = self.search(
            query=query,
            n_results=n_results * 2,
            where=where,
            max_distance=max_distance,
        )
        if not raw_results:
            return []

        now = datetime.now(tz=timezone.utc)
        similarity_weight = 1.0 - recency_weight - keyword_weight

        scored = []
        for r in raw_results:
            # Similarity score: convert cosine distance (0-2) to similarity (1-0)
            distance = r.get("distance") or 1.0
            similarity_score = max(0.0, 1.0 - distance / 2.0)

            # Recency score: exponential decay, half-life ~1 year
            recency_score = 0.0
            ts = r["metadata"].get("msg_timestamp") or r["metadata"].get("timestamp", "")
            if ts:
                try:
                    dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
                    days_ago = max(0, (now - dt).days)
                    recency_score = math.exp(-days_ago / 365.0)
                except (ValueError, TypeError):
                    pass

            # Keyword overlap score
            kw_score = self._keyword_overlap_score(query, r["text"])

            r["_combined_score"] = (
                similarity_weight * similarity_score
                + recency_weight * recency_score
                + keyword_weight * kw_score
            )
            scored.append(r)

        scored.sort(key=lambda x: x["_combined_score"], reverse=True)
        return scored[:n_results]

    # ------------------------------------------------------------------
    # Dimension-aware search (v2)
    # ------------------------------------------------------------------

    def search_by_dimension(self, query: str, dimension: str, n_results: int = 10, max_distance: float | None = None) -> list[dict]:
        """Search chunks classified into a specific persona dimension."""
        return self.search(query, n_results, where={"dimension": dimension}, max_distance=max_distance)

    def search_by_cluster(self, query: str, cluster_label: str, n_results: int = 10, max_distance: float | None = None) -> list[dict]:
        """Search chunks within a specific cluster."""
        return self.search(query, n_results, where={"cluster_label": cluster_label}, max_distance=max_distance)

    def get_cluster_labels(self) -> list[str]:
        """Get all distinct cluster labels in the collection."""
        try:
            result = self.collection.get(
                where={"cluster_label": {"$ne": ""}},
                include=["metadatas"],
                limit=10000,
            )
            labels = set()
            for meta in (result.get("metadatas") or []):
                label = meta.get("cluster_label", "")
                if label:
                    labels.add(label)
            return sorted(labels)
        except Exception:
            return []

    def get_unclassified_chunks(self, limit: int = 500) -> dict:
        """Get chunks that haven't been classified yet."""
        return self.collection.get(
            where={"classified": "false"},
            limit=limit,
            include=["documents", "metadatas"],
        )

    def update_metadata(self, chunk_ids: list[str], new_metadatas: list[dict]) -> None:
        """Batch-update metadata on existing chunks (e.g. after classification)."""
        if not chunk_ids:
            return
        self.collection.update(ids=chunk_ids, metadatas=new_metadatas)

    def count_by_dimension(self, dimension: str) -> int:
        """Count chunks classified into a dimension."""
        try:
            result = self.collection.get(
                where={"dimension": dimension},
                limit=1,
                include=[],
            )
            # ChromaDB doesn't have a direct count with where,
            # so we use a get with include=[] and count the IDs
            result = self.collection.get(
                where={"dimension": dimension},
                include=[],
            )
            return len(result["ids"])
        except Exception:
            return 0

    # ------------------------------------------------------------------
    # General
    # ------------------------------------------------------------------

    def count(self) -> int:
        """Return total number of chunks in the store."""
        return self.collection.count()

    def clear(self):
        """Clear all data from the collection."""
        self.client.delete_collection(CHROMA_COLLECTION)
        self.collection = self.client.get_or_create_collection(
            name=CHROMA_COLLECTION,
            metadata={"hnsw:space": "cosine"},
        )
