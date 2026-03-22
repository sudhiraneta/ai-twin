import hashlib
import json

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
    ) -> list[dict]:
        """Search for relevant memory chunks."""
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

        return items

    # ------------------------------------------------------------------
    # Typed search helpers
    # ------------------------------------------------------------------

    def search_user_messages(self, query: str, n_results: int = 10) -> list[dict]:
        """Search only user messages (for persona-related queries)."""
        return self.search(query, n_results, where={"type": "user_message"})

    def search_conversations(self, query: str, n_results: int = 10) -> list[dict]:
        """Search conversation pairs (for context retrieval)."""
        return self.search(query, n_results, where={"type": "conversation_pair"})

    def search_notes(self, query: str, n_results: int = 10) -> list[dict]:
        return self.search(query, n_results, where={"type": "note"})

    def search_browser(self, query: str, n_results: int = 10) -> list[dict]:
        return self.search(query, n_results, where={"type": "browser_daily"})

    def search_tasks(self, query: str, n_results: int = 10) -> list[dict]:
        return self.search(query, n_results, where={"type": "task"})

    def search_by_pillar(self, query: str, pillar: str, n_results: int = 10) -> list[dict]:
        return self.search(query, n_results, where={"pillar": pillar})

    def search_by_source(self, query: str, source: str, n_results: int = 10) -> list[dict]:
        return self.search(query, n_results, where={"source": source})

    # ------------------------------------------------------------------
    # Dimension-aware search (v2)
    # ------------------------------------------------------------------

    def search_by_dimension(self, query: str, dimension: str, n_results: int = 10) -> list[dict]:
        """Search chunks classified into a specific persona dimension."""
        return self.search(query, n_results, where={"dimension": dimension})

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
