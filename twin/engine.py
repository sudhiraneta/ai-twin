import json
import re
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone

from config import MAX_CONTEXT_MESSAGES
from memory.vectorstore import VectorStore
from memory.chunker import Chunk, _ensure_metadata
from .llm_client import chat_completion
from .prompts import build_system_prompt, MEMORY_CONTEXT_TEMPLATE


@dataclass
class DecisionResponse:
    """Structured response from the decision mode."""
    your_decision: str = ""
    ideal_decision: str = ""
    reasoning_gap: str = ""
    confidence_score: str = ""
    follow_up_questions: list[str] = field(default_factory=list)
    raw_response: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


class TwinEngine:
    """The core AI twin: persona-prompted LLM with RAG memory retrieval."""

    def __init__(self):
        from persona.profile import PersonaProfile
        from persona.classifier import ChunkClassifier

        self.vector_store = VectorStore()
        self.persona = PersonaProfile.load()
        self.classifier = ChunkClassifier()
        self.conversation_history: list[dict] = []
        self._message_count = 0

    def _get_persona_prompt(self) -> str:
        """Build persona prompt from skill files (data-driven, not hardcoded).
        Falls back to stored system_prompt if no skill files exist yet."""
        from persona.skills import build_persona_from_skills
        skills_prompt = build_persona_from_skills()
        if skills_prompt:
            return skills_prompt
        return self.persona.system_prompt

    def chat(self, user_message: str) -> str:
        """Send a message to the twin and get a response."""
        self._message_count += 1

        # Enable data collection every ~5 messages
        enable_data_collection = (self._message_count % 5 == 0)

        memory_context = self._retrieve_memories(user_message)
        dimension_context = self._get_dimension_context(user_message)

        system_prompt = build_system_prompt(
            persona_prompt=self._get_persona_prompt(),
            memory_context=memory_context,
            decision_mode=False,
            enable_data_collection=enable_data_collection,
            dimension_context=dimension_context,
        )

        self.conversation_history.append({
            "role": "user",
            "content": user_message,
        })

        assistant_message = chat_completion(
            system=system_prompt,
            messages=self.conversation_history,
            max_tokens=4096,
        )

        self.conversation_history.append({
            "role": "assistant",
            "content": assistant_message,
        })

        return assistant_message

    def decide(self, question: str) -> DecisionResponse:
        """Analyze a decision question with dual-lens (your decision vs ideal)."""
        memory_context = self._retrieve_memories(question)

        # Include decision history from persona if available
        decision_history_context = ""
        if self.persona.decision_history:
            recent = self.persona.decision_history[-10:]
            history_lines = []
            for d in recent:
                line = f"- Q: {d['question']} → Decided: {d['decision']}"
                if d.get("outcome"):
                    line += f" → Outcome: {d['outcome']}"
                history_lines.append(line)
            decision_history_context = "\n## Past Decision History\n" + "\n".join(history_lines)

        # Build persona context with extra decision traits
        persona_context = self._get_persona_prompt()
        if self.persona.cognitive_biases:
            persona_context += f"\n\nKnown cognitive biases: {', '.join(self.persona.cognitive_biases)}"
        if self.persona.risk_tolerance:
            persona_context += f"\nRisk tolerance: {self.persona.risk_tolerance}"
        if self.persona.time_preference:
            persona_context += f"\nTime preference: {self.persona.time_preference}"

        full_memory = (memory_context or "") + decision_history_context
        dimension_context = self._get_dimension_context(question)

        system_prompt = build_system_prompt(
            persona_prompt=persona_context,
            memory_context=full_memory if full_memory.strip() else None,
            decision_mode=True,
            dimension_context=dimension_context,
        )

        raw = chat_completion(
            system=system_prompt,
            messages=[{"role": "user", "content": question}],
            max_tokens=4096,
        )
        return self._parse_decision_response(raw)

    def learn(self, data_point: str) -> dict:
        """Ingest a new user-provided data point into memory."""
        # Classify the data point
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

    def _retrieve_memories(self, query: str) -> str | None:
        """Retrieve relevant conversation memories for context."""
        if self.vector_store.count() == 0:
            return None

        results = self.vector_store.search(
            query=query,
            n_results=MAX_CONTEXT_MESSAGES,
        )

        if not results:
            return None

        memory_lines = []
        for r in results:
            source = r["metadata"].get("source", "unknown")
            title = r["metadata"].get("title", "")
            timestamp = r["metadata"].get("timestamp", "")
            mem_type = r["metadata"].get("type", "")
            pillar = r["metadata"].get("pillar", "")

            if mem_type == "note":
                prefix = f"[Apple Note - {title} | {timestamp[:10]}]"
            elif mem_type == "browser_daily":
                prefix = f"[Browser Activity | {timestamp[:10]}]"
            elif mem_type == "browser_domain":
                prefix = f"[Browser/{title} | {timestamp[:10]}]"
            elif mem_type == "singularity_entry":
                prefix = f"[Singularity/{pillar} - {title} | {timestamp[:10]}]"
            elif mem_type == "task":
                prefix = f"[Task/{pillar} | {timestamp[:10]}]"
            elif mem_type == "body_gym":
                prefix = f"[Gym Tracker | {timestamp[:10]}]"
            elif mem_type == "body_nutrition":
                prefix = f"[Nutrition | {timestamp[:10]}]"
            elif mem_type == "weekly_review":
                prefix = f"[Weekly Review | {timestamp[:10]}]"
            elif mem_type == "soul_checkin":
                prefix = f"[Soul Checkin | {timestamp[:10]}]"
            elif mem_type == "goals_completed":
                prefix = f"[Goals Completed | {timestamp[:10]}]"
            elif mem_type == "card_counters":
                prefix = f"[Rep Milestones | {timestamp[:10]}]"
            elif mem_type == "plan_note":
                prefix = f"[30-Day Plan | {timestamp[:10]}]"
            elif mem_type == "week_carry":
                prefix = f"[Week Summary | {timestamp[:10]}]"
            elif mem_type == "pillar_journal":
                prefix = f"[Journal/{pillar} | {title}]"
            elif mem_type == "data_point":
                prefix = f"[self-reported - {title} | {timestamp[:10]}]"
            else:
                prefix = f"[{source}"
                if title:
                    prefix += f" - {title}"
                if timestamp:
                    prefix += f" | {timestamp[:10]}"
                prefix += "]"

            memory_lines.append(f"{prefix}\n{r['text']}")

        memories_text = "\n\n---\n\n".join(memory_lines)
        return MEMORY_CONTEXT_TEMPLATE.format(memories=memories_text)

    def _parse_decision_response(self, raw: str) -> DecisionResponse:
        """Parse the structured decision response from the LLM."""
        response = DecisionResponse(raw_response=raw)

        sections = {
            "your_decision": r"## Your Likely Decision\s*\n(.*?)(?=\n## |\Z)",
            "ideal_decision": r"## Ideal Decision\s*\n(.*?)(?=\n## |\Z)",
            "reasoning_gap": r"## Gap Analysis\s*\n(.*?)(?=\n## |\Z)",
            "confidence_score": r"## Confidence Score\s*\n(.*?)(?=\n## |\Z)",
            "follow_up_raw": r"## Follow-Up Questions\s*\n(.*?)(?=\n## |\Z)",
        }

        for field_name, pattern in sections.items():
            match = re.search(pattern, raw, re.DOTALL)
            if match:
                value = match.group(1).strip()
                if field_name == "follow_up_raw":
                    # Extract individual questions
                    questions = re.findall(r'[-•*]\s*(.+)', value)
                    if not questions:
                        questions = [q.strip() for q in value.split('\n') if q.strip()]
                    response.follow_up_questions = questions
                else:
                    setattr(response, field_name, value)

        # If parsing failed, put everything in your_decision
        if not response.your_decision and not response.ideal_decision:
            response.your_decision = raw

        return response

    def _get_dimension_context(self, query: str) -> str | None:
        """Detect relevant dimensions for a query and return their persona summaries."""
        relevant_dims = self.classifier.classify_text(query)
        if not relevant_dims:
            return None
        summary = self.persona.get_relevant_dimensions(relevant_dims)
        return summary if summary else None

    def _retrieve_memories_dimension_aware(self, query: str) -> str | None:
        """Retrieve memories with dimension-weighted boosting."""
        if self.vector_store.count() == 0:
            return None

        relevant_dims = self.classifier.classify_text(query)

        seen_ids = set()
        results = []

        # Dimension-targeted search (5 per relevant dimension)
        for dim in relevant_dims[:3]:  # limit to top 3 dimensions
            dim_results = self.vector_store.search_by_dimension(query, dim, n_results=5)
            for r in dim_results:
                if r["id"] not in seen_ids:
                    seen_ids.add(r["id"])
                    results.append(r)

        # General search for breadth
        general_results = self.vector_store.search(query, n_results=MAX_CONTEXT_MESSAGES)
        for r in general_results:
            if r["id"] not in seen_ids:
                seen_ids.add(r["id"])
                results.append(r)

        # Limit total results
        results = results[:MAX_CONTEXT_MESSAGES]

        if not results:
            return None

        memory_lines = []
        for r in results:
            source = r["metadata"].get("source", "unknown")
            title = r["metadata"].get("title", "")
            timestamp = r["metadata"].get("timestamp", "")
            mem_type = r["metadata"].get("type", "")
            pillar = r["metadata"].get("pillar", "")

            if mem_type == "note":
                prefix = f"[Apple Note - {title} | {timestamp[:10]}]"
            elif mem_type == "browser_daily":
                prefix = f"[Browser Activity | {timestamp[:10]}]"
            elif mem_type == "browser_domain":
                prefix = f"[Browser/{title} | {timestamp[:10]}]"
            elif mem_type == "singularity_entry":
                prefix = f"[Singularity/{pillar} - {title} | {timestamp[:10]}]"
            elif mem_type == "task":
                prefix = f"[Task/{pillar} | {timestamp[:10]}]"
            elif mem_type == "body_gym":
                prefix = f"[Gym Tracker | {timestamp[:10]}]"
            elif mem_type == "body_nutrition":
                prefix = f"[Nutrition | {timestamp[:10]}]"
            elif mem_type == "weekly_review":
                prefix = f"[Weekly Review | {timestamp[:10]}]"
            elif mem_type == "soul_checkin":
                prefix = f"[Soul Checkin | {timestamp[:10]}]"
            elif mem_type == "goals_completed":
                prefix = f"[Goals Completed | {timestamp[:10]}]"
            elif mem_type == "card_counters":
                prefix = f"[Rep Milestones | {timestamp[:10]}]"
            elif mem_type == "plan_note":
                prefix = f"[30-Day Plan | {timestamp[:10]}]"
            elif mem_type == "week_carry":
                prefix = f"[Week Summary | {timestamp[:10]}]"
            elif mem_type == "pillar_journal":
                prefix = f"[Journal/{pillar} | {title}]"
            elif mem_type == "data_point":
                prefix = f"[self-reported - {title} | {timestamp[:10]}]"
            else:
                prefix = f"[{source}"
                if title:
                    prefix += f" - {title}"
                if timestamp:
                    prefix += f" | {timestamp[:10]}"
                prefix += "]"

            memory_lines.append(f"{prefix}\n{r['text']}")

        memories_text = "\n\n---\n\n".join(memory_lines)
        return MEMORY_CONTEXT_TEMPLATE.format(memories=memories_text)

    def search_memory(self, query: str, n_results: int = 10) -> list[dict]:
        """Search the twin's memory directly."""
        return self.vector_store.search(query=query, n_results=n_results)

    def reset_conversation(self):
        """Clear the current conversation history (not memory)."""
        self.conversation_history = []
        self._message_count = 0

    def reload_persona(self):
        """Reload persona profile from disk."""
        from persona.profile import PersonaProfile
        self.persona = PersonaProfile.load()
