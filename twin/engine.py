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

    # ------------------------------------------------------------------
    # Semantic dimension matching — fallback when keyword classifier misses
    # ------------------------------------------------------------------

    def _semantic_dimension_match(self, query: str, dimension_sources: dict) -> list[str]:
        """Find best-matching dimensions by embedding similarity to their queries.

        Embeds the user query and each dimension's primary_queries, then picks
        the dimension(s) whose queries are closest in embedding space.
        """
        from memory.embeddings import EmbeddingEngine

        engine = self.vector_store.embedding_engine
        query_emb = engine.embed_single(query)

        import numpy as np
        query_vec = np.array(query_emb, dtype=np.float32)
        query_norm = np.linalg.norm(query_vec)
        if query_norm > 0:
            query_vec = query_vec / query_norm

        scores: list[tuple[str, float]] = []

        for dim_name, sources in dimension_sources.items():
            # Combine primary + memory queries as dimension signature
            all_queries = sources.get("primary_queries", []) + sources.get("memory_queries", [])
            if not all_queries:
                continue

            combined_text = " ".join(all_queries)
            dim_emb = engine.embed_single(combined_text)
            dim_vec = np.array(dim_emb, dtype=np.float32)
            dim_norm = np.linalg.norm(dim_vec)
            if dim_norm > 0:
                dim_vec = dim_vec / dim_norm

            # Cosine similarity (higher = more similar)
            similarity = float(np.dot(query_vec, dim_vec))
            scores.append((dim_name, similarity))

        # Sort by similarity descending, return top matches above threshold
        scores.sort(key=lambda x: -x[1])
        threshold = 0.3  # minimum similarity to consider a match
        matched = [name for name, sim in scores if sim >= threshold]
        return matched[:3]

    # ------------------------------------------------------------------
    # Skill-aware search — uses DIMENSION_SOURCES for tiered retrieval
    # ------------------------------------------------------------------

    def skill_search(self, query: str, n_results: int = 15, dimension: str | None = None) -> dict:
        """Skill-file-driven retrieval with 3-tier sources.

        Tier 1 (CORE):  Singularity entries + Apple Notes + data_points + journals
        Tier 2 (MEMORY): Imported LLM conversations (ChatGPT/Claude/Gemini)
        Tier 3 (SUPPLEMENTARY): Browser, weekly reviews, photos, etc.

        Returns:
            {
                "dimensions": list[str] — matched dimension names,
                "results": list[dict] — merged retrieval chunks with _tier tags,
                "skill_context": str — skill file content for LLM prompt,
            }
        """
        from persona.skills import DIMENSION_SOURCES, read_skill_file

        # 1. Determine relevant dimensions
        if dimension:
            matched_dims = [dimension]
        else:
            matched_dims = self.classifier.classify_text(query)

        # Fallback: if keyword classifier finds nothing, use semantic matching
        # against each dimension's primary queries to find the best-fit dimension
        if not matched_dims:
            matched_dims = self._semantic_dimension_match(query, DIMENSION_SOURCES)

        if not matched_dims:
            # Last resort: generic search, no skill orchestration
            return {
                "dimensions": [],
                "results": self.search_memory(query, n_results),
                "skill_context": "",
            }

        seen_ids: set[str] = set()
        results: list[dict] = []
        skill_parts: list[str] = []

        # Budget allocation: ~50% primary, ~30% memory, ~20% supplementary
        tier1_budget = max(n_results // 2, 4)
        tier2_budget = max(n_results * 3 // 10, 3)

        for dim_name in matched_dims[:3]:
            sources = DIMENSION_SOURCES.get(dim_name)
            if not sources:
                continue

            # --- Collect skill file content for prompt injection ---
            skill_content = read_skill_file(dim_name)
            if skill_content:
                skill_parts.append(skill_content)

            # =============================================================
            # TIER 1: Core — Singularity + clustered Apple Notes + journals
            # =============================================================
            primary_types = sources.get("primary_types", [])
            primary_queries = sources.get("primary_queries", [])

            per_type = max(tier1_budget // max(len(primary_types), 1), 2)
            for ptype in primary_types:
                type_results = self.vector_store.search(
                    query=query,
                    n_results=per_type,
                    where={"type": ptype},
                    max_distance=RELEVANCE_THRESHOLD,
                )
                for r in type_results:
                    if r["id"] not in seen_ids:
                        seen_ids.add(r["id"])
                        r["_tier"] = "primary"
                        results.append(r)

            # Run skill-defined queries scoped to this dimension for broader recall
            for pq in primary_queries:
                pq_results = self.vector_store.search_by_dimension(
                    query=pq,
                    dimension=dim_name,
                    n_results=3,
                    max_distance=RELEVANCE_THRESHOLD,
                )
                for r in pq_results:
                    if r["id"] not in seen_ids:
                        seen_ids.add(r["id"])
                        r["_tier"] = "primary"
                        results.append(r)

            # =============================================================
            # TIER 2: LLM Memory — ChatGPT/Claude/Gemini conversations
            # =============================================================
            memory_types = sources.get("memory_types", [])
            memory_queries = sources.get("memory_queries", [])

            if memory_types:
                per_mem_type = max(tier2_budget // max(len(memory_types), 1), 2)
                for mtype in memory_types:
                    mem_results = self.vector_store.search(
                        query=query,
                        n_results=per_mem_type,
                        where={"type": mtype},
                        max_distance=RELEVANCE_THRESHOLD,
                    )
                    for r in mem_results:
                        if r["id"] not in seen_ids:
                            seen_ids.add(r["id"])
                            r["_tier"] = "memory"
                            results.append(r)

                # Run memory-specific queries for broader LLM conversation recall
                for mq in memory_queries:
                    mq_results = self.vector_store.search_with_recency(
                        query=mq,
                        n_results=3,
                        where={"type": "user_message"},
                        max_distance=RELEVANCE_THRESHOLD,
                        recency_weight=RECENCY_WEIGHT,
                    )
                    for r in mq_results:
                        if r["id"] not in seen_ids:
                            seen_ids.add(r["id"])
                            r["_tier"] = "memory"
                            results.append(r)

            # =============================================================
            # TIER 3: Supplementary — browser, reviews, photos (fill gaps)
            # =============================================================
            if len(results) < n_results:
                secondary_types = sources.get("secondary_types", [])
                secondary_queries = sources.get("secondary_queries", [])
                remaining = n_results - len(results)

                per_sec_type = max(remaining // max(len(secondary_types), 1), 1)
                for stype in secondary_types:
                    sec_results = self.vector_store.search(
                        query=query,
                        n_results=per_sec_type,
                        where={"type": stype},
                        max_distance=RELEVANCE_THRESHOLD,
                    )
                    for r in sec_results:
                        if r["id"] not in seen_ids:
                            seen_ids.add(r["id"])
                            r["_tier"] = "supplementary"
                            results.append(r)

                for sq in secondary_queries:
                    sq_results = self.vector_store.search_with_recency(
                        query=sq,
                        n_results=3,
                        max_distance=RELEVANCE_THRESHOLD,
                        recency_weight=RECENCY_WEIGHT,
                    )
                    for r in sq_results:
                        if r["id"] not in seen_ids:
                            seen_ids.add(r["id"])
                            r["_tier"] = "supplementary"
                            results.append(r)

        # Sort: tier order (primary > memory > supplementary), then by distance
        tier_order = {"primary": 0, "memory": 1, "supplementary": 2}
        results.sort(key=lambda r: (tier_order.get(r.get("_tier", ""), 3), r.get("distance", 1.0)))

        # Build skill context string for LLM prompt
        skill_context = "\n\n---\n\n".join(skill_parts) if skill_parts else ""

        return {
            "dimensions": matched_dims[:3],
            "results": results[:n_results],
            "skill_context": skill_context,
        }

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
