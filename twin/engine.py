from datetime import datetime, timezone

from config import MAX_CONTEXT_MESSAGES, RELEVANCE_THRESHOLD, RECENCY_WEIGHT
from db import QueryRouter, MetricStore, QueryType
from db.sql_prompts import format_metric_results
from memory.vectorstore import VectorStore
from memory.chunker import Chunk, _ensure_metadata
from .prompts import MEMORY_CONTEXT_TEMPLATE


class TwinEngine:
    """Core engine: persona-aware memory search and data ingestion."""

    def __init__(self):
        from persona.profile import PersonaProfile
        from persona.classifier import ChunkClassifier

        self.vector_store = VectorStore()
        self.persona = PersonaProfile.load()
        self.classifier = ChunkClassifier()
        self.query_router = QueryRouter()
        self.metric_store = MetricStore()

    def learn(self, data_point: str) -> dict:
        """Ingest a new user-provided data point into memory."""
        pillar, dimension = self.classifier.classify_chunk(data_point, {"type": "data_point"})

        chunk = Chunk(
            text=data_point,
            metadata=_ensure_metadata({
                "source": "self_reported",
                "conversation_id": "",
                "title": "User data point",
                "timestamp": datetime.now(tz=timezone.utc).isoformat(),
                "msg_timestamp": datetime.now(tz=timezone.utc).isoformat(),
                "role": "user",
                "type": "data_point",
                "pillar": pillar,
                "dimension": dimension,
                "classified": "true" if dimension else "false",
            })
        )
        count = self.vector_store.ingest([chunk])
        return {"status": "learned", "chunks_added": count, "data_point": data_point}

    def search_memory(self, query: str, n_results: int = 10) -> list[dict]:
        """Search the twin's memory with dimension-aware boosting."""
        relevant_dims = self.classifier.classify_text(query)

        seen_ids = set()
        results = []

        # Dimension-targeted results first
        for dim in relevant_dims[:3]:
            dim_results = self.vector_store.search_by_dimension(
                query, dim, n_results=n_results // 2, max_distance=RELEVANCE_THRESHOLD,
            )
            for r in dim_results:
                if r["id"] not in seen_ids:
                    seen_ids.add(r["id"])
                    results.append(r)

        # Fill remaining with recency-weighted general search
        general = self.vector_store.search_with_recency(
            query=query, n_results=n_results, max_distance=RELEVANCE_THRESHOLD,
            recency_weight=RECENCY_WEIGHT,
        )
        for r in general:
            if r["id"] not in seen_ids:
                seen_ids.add(r["id"])
                results.append(r)

        results.sort(key=lambda r: r.get("distance", 1.0))
        return results[:n_results]

    @staticmethod
    def _format_memory_line(r: dict) -> str:
        """Format a single memory result into a prefixed line for prompt injection."""
        source = r["metadata"].get("source", "unknown")
        title = r["metadata"].get("title", "")
        timestamp = r["metadata"].get("timestamp", "")
        mem_type = r["metadata"].get("type", "")
        pillar = r["metadata"].get("pillar", "")

        type_prefix_map = {
            "note": f"[Apple Note - {title} | {timestamp[:10]}]",
            "browser_daily": f"[Browser Activity | {timestamp[:10]}]",
            "browser_domain": f"[Browser/{title} | {timestamp[:10]}]",
            "singularity_entry": f"[Singularity/{pillar} - {title} | {timestamp[:10]}]",
            "task": f"[Task/{pillar} | {timestamp[:10]}]",
            "body_gym": f"[Gym Tracker | {timestamp[:10]}]",
            "body_nutrition": f"[Nutrition | {timestamp[:10]}]",
            "weekly_review": f"[Weekly Review | {timestamp[:10]}]",
            "soul_checkin": f"[Soul Checkin | {timestamp[:10]}]",
            "goals_completed": f"[Goals Completed | {timestamp[:10]}]",
            "card_counters": f"[Rep Milestones | {timestamp[:10]}]",
            "plan_note": f"[30-Day Plan | {timestamp[:10]}]",
            "week_carry": f"[Week Summary | {timestamp[:10]}]",
            "pillar_journal": f"[Journal/{pillar} | {title}]",
            "data_point": f"[self-reported - {title} | {timestamp[:10]}]",
        }

        prefix = type_prefix_map.get(mem_type)
        if not prefix:
            prefix = f"[{source}"
            if title:
                prefix += f" - {title}"
            if timestamp:
                prefix += f" | {timestamp[:10]}"
            prefix += "]"

        return f"{prefix}\n{r['text']}"

    def _retrieve_memories_dimension_aware(self, query: str) -> str | None:
        """Retrieve memories with dimension-weighted boosting, formatted for prompt."""
        if self.vector_store.count() == 0:
            return None

        results = self.search_memory(query, n_results=MAX_CONTEXT_MESSAGES)
        if not results:
            return None

        memory_lines = [self._format_memory_line(r) for r in results]
        memories_text = "\n\n---\n\n".join(memory_lines)
        return MEMORY_CONTEXT_TEMPLATE.format(memories=memories_text)

    def hybrid_context(self, query: str) -> dict:
        """Route query and retrieve from SQL, RAG, or both.

        Returns dict with keys:
            sql_context: str | None — formatted SQL metrics for prompt injection
            rag_context: str | None — formatted RAG memories for prompt injection
            route_type: str — "sql", "rag", or "hybrid"
        """
        routed = self.query_router.route(query)

        sql_context = None
        rag_context = None

        if routed.query_type in (QueryType.SQL, QueryType.HYBRID):
            metric_results = self.metric_store.query_from_intent(
                routed.sql_intent, routed.sql_tables, routed.time_range,
            )
            sql_context = format_metric_results(metric_results)

        if routed.query_type in (QueryType.RAG, QueryType.HYBRID):
            rag_context = self._retrieve_memories_dimension_aware(routed.rag_query)

        return {
            "sql_context": sql_context,
            "rag_context": rag_context,
            "route_type": routed.query_type.value,
        }

    def reload_persona(self):
        """Reload persona profile from disk."""
        from persona.profile import PersonaProfile
        self.persona = PersonaProfile.load()
