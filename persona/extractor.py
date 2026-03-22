import json
import re

from twin.llm_client import chat_completion
from .profile import PersonaProfile
from .dimensions import (
    DIMENSIONS,
    DIMENSION_SCHEMAS,
    DIMENSION_EXTRACTION_PROMPTS,
)


def _repair_json(raw: str) -> dict | None:
    """Try multiple strategies to extract valid JSON from LLM output."""
    text = raw.strip()

    # Strip markdown fences
    if text.startswith("```"):
        text = re.sub(r"^```\w*\n?", "", text)
        text = re.sub(r"\n?```$", "", text)
        text = text.strip()

    # Try direct parse
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Find JSON object in mixed text
    match = re.search(r"\{[\s\S]*\}", text)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass

    # Try fixing common issues: trailing commas, single quotes
    cleaned = re.sub(r",\s*([}\]])", r"\1", text)  # trailing commas
    cleaned = cleaned.replace("'", '"')  # single quotes
    match = re.search(r"\{[\s\S]*\}", cleaned)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass

    return None


class PersonaExtractor:
    """Extracts a persona profile from user conversation history using the configured LLM.

    Supports both legacy flat extraction and per-dimension v2 extraction.
    """

    def __init__(self):
        pass

    # ------------------------------------------------------------------
    # Legacy flat extraction (backward compat)
    # ------------------------------------------------------------------

    def extract(self, user_messages: list[str], max_sample: int = 500) -> PersonaProfile:
        """Analyze user messages to build a persona profile (v1 flat extraction)."""
        if len(user_messages) > max_sample:
            import random
            sampled = random.sample(user_messages, max_sample)
        else:
            sampled = user_messages

        messages_text = "\n---\n".join(sampled[:max_sample])

        profile_data = self._analyze_messages(messages_text)
        system_prompt = self._generate_system_prompt(profile_data)

        profile = PersonaProfile(
            communication_style=profile_data.get("communication_style", {}),
            knowledge_domains=profile_data.get("knowledge_domains", []),
            decision_patterns=profile_data.get("decision_patterns", []),
            values_and_priorities=profile_data.get("values_and_priorities", []),
            interests=profile_data.get("interests", []),
            cognitive_biases=profile_data.get("cognitive_biases", []),
            risk_tolerance=profile_data.get("risk_tolerance", "moderate"),
            time_preference=profile_data.get("time_preference", "balanced"),
            system_prompt=system_prompt,
        )

        profile.save()
        return profile

    # ------------------------------------------------------------------
    # Per-dimension extraction (v2)
    # ------------------------------------------------------------------

    def extract_dimension(
        self,
        dimension_name: str,
        chunk_texts: list[str],
        max_chunks: int = 200,
    ) -> dict:
        """Extract traits for a single dimension from relevant chunks.

        Returns a traits dict matching the dimension's schema.
        """
        if dimension_name not in DIMENSIONS:
            return {}

        if not chunk_texts:
            return {}

        texts = chunk_texts[:max_chunks]
        combined = "\n---\n".join(texts)

        schema = DIMENSION_SCHEMAS.get(dimension_name, {})
        schema_desc = json.dumps({k: v.__name__ for k, v in schema.items()}, indent=2)
        extraction_guide = DIMENSION_EXTRACTION_PROMPTS.get(dimension_name, "")

        prompt = (
            f"Analyze the following messages/data about a person and extract their "
            f"\"{DIMENSIONS[dimension_name]['display']}\" profile.\n\n"
            f"Questions to answer:\n{extraction_guide}\n\n"
            f"Return a JSON object with exactly these fields:\n{schema_desc}\n\n"
            f"Be specific and evidence-based. Only include traits clearly supported "
            f"by the data. Use empty strings/lists for traits with no evidence.\n\n"
            f"Data:\n{combined}\n\n"
            f"Return ONLY the JSON object, no other text."
        )

        # Limit combined text to avoid overwhelming small models
        if len(combined) > 8000:
            combined = combined[:8000] + "\n...(truncated)"

        for attempt in range(3):
            try:
                if attempt == 0:
                    msg = prompt
                elif attempt == 1:
                    # Simpler prompt on retry
                    msg = (
                        f"From the data below, extract a JSON object with these fields: {schema_desc}\n\n"
                        f"Data (summarize what you find):\n{combined[:4000]}\n\n"
                        f"Return ONLY valid JSON. No explanation."
                    )
                else:
                    # Minimal prompt
                    fields = list(DIMENSION_SCHEMAS.get(dimension_name, {}).keys())
                    msg = (
                        f"Extract these fields about a person: {fields}\n"
                        f"From: {combined[:2000]}\n"
                        f"Reply with ONLY a JSON object."
                    )

                raw = chat_completion(
                    system="You are a personality analysis expert. Return only valid JSON, no markdown.",
                    messages=[{"role": "user", "content": msg}],
                    max_tokens=4096,
                ).strip()

                result = _repair_json(raw)
                if result and isinstance(result, dict):
                    # Verify it has at least one non-empty value
                    has_data = any(
                        v for v in result.values()
                        if (isinstance(v, str) and v) or (isinstance(v, list) and v) or (isinstance(v, dict) and v)
                    )
                    if has_data:
                        return result

                print(f"    Attempt {attempt + 1}: empty result, retrying...")

            except Exception as e:
                print(f"    Attempt {attempt + 1} failed: {e}")

        print(f"    {dimension_name}: extraction failed after 3 attempts")
        return {}

    def extract_all_dimensions(
        self,
        chunks_by_dimension: dict[str, list[str]],
        profile: PersonaProfile | None = None,
    ) -> PersonaProfile:
        """Full extraction across all dimensions.

        chunks_by_dimension: {dimension_name: [chunk_texts]}
        """
        if profile is None:
            profile = PersonaProfile.load()

        from datetime import datetime, timezone

        for dim_name in DIMENSIONS:
            chunk_texts = chunks_by_dimension.get(dim_name, [])
            if not chunk_texts:
                continue

            print(f"  Extracting dimension: {dim_name} ({len(chunk_texts)} chunks)...")
            traits = self.extract_dimension(dim_name, chunk_texts)
            if traits:
                confidence = min(1.0, len(chunk_texts) / 100)
                profile.update_dimension(
                    dim_name,
                    traits=traits,
                    confidence=confidence,
                    evidence_count=len(chunk_texts),
                )

        profile.last_full_extraction = datetime.now(tz=timezone.utc).isoformat()

        # Regenerate system prompt from dimension summaries
        dim_summary = profile.get_all_dimensions_summary()
        if dim_summary:
            system_prompt = self._generate_system_prompt_from_dimensions(dim_summary)
            profile.system_prompt = system_prompt

        profile.save()
        return profile

    def incremental_update(
        self,
        changed_dimensions: list[str],
        chunks_by_dimension: dict[str, list[str]],
        profile: PersonaProfile | None = None,
    ) -> PersonaProfile:
        """Only re-extract dimensions that have new evidence."""
        if profile is None:
            profile = PersonaProfile.load()

        for dim_name in changed_dimensions:
            chunk_texts = chunks_by_dimension.get(dim_name, [])
            if not chunk_texts:
                continue

            print(f"  Updating dimension: {dim_name} ({len(chunk_texts)} chunks)...")
            traits = self.extract_dimension(dim_name, chunk_texts)
            if traits:
                confidence = min(1.0, len(chunk_texts) / 100)
                profile.update_dimension(
                    dim_name,
                    traits=traits,
                    confidence=confidence,
                    evidence_count=len(chunk_texts),
                )

        # Regenerate system prompt
        dim_summary = profile.get_all_dimensions_summary()
        if dim_summary:
            system_prompt = self._generate_system_prompt_from_dimensions(dim_summary)
            profile.system_prompt = system_prompt

        profile.save()
        return profile

    # ------------------------------------------------------------------
    # Internal prompt helpers
    # ------------------------------------------------------------------

    def _analyze_messages(self, messages_text: str) -> dict:
        """Use the configured LLM to analyze user messages and extract persona traits."""
        prompt = f"""Analyze the following messages written by a single person. Extract their complete personality and decision-making profile.

Return a JSON object with these fields:
- "communication_style": {{
    "tone": description of their tone (e.g., "casual and direct", "formal and analytical"),
    "formality": "casual" | "moderate" | "formal",
    "vocabulary_level": "simple" | "moderate" | "advanced" | "technical",
    "sentence_patterns": [list of notable patterns],
    "common_phrases": [phrases they frequently use]
  }}
- "knowledge_domains": [areas they frequently discuss or show expertise in]
- "decision_patterns": [how they approach decisions, e.g., "weighs pros/cons explicitly", "prioritizes speed over perfection"]
- "values_and_priorities": [what they care about most based on their messages]
- "interests": [topics, hobbies, technologies they show interest in]
- "cognitive_biases": [detected cognitive biases visible in their reasoning, e.g., "anchoring to first option presented", "confirmation bias when defending technical choices", "optimism bias on timelines", "loss aversion in career decisions". Only include biases clearly supported by evidence in the messages]
- "risk_tolerance": "low" | "moderate" | "high" (based on how they approach uncertain situations, new technologies, career moves, etc.)
- "time_preference": "short-term" | "balanced" | "long-term" (do they optimize for immediate results or long-term outcomes?)

Be specific and evidence-based. Only include traits clearly supported by the messages.

Messages:
{messages_text}

Return ONLY the JSON object, no other text."""

        text = chat_completion(
            system="You are a personality analysis expert. Return only valid JSON.",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=4096,
        ).strip()

        if text.startswith("```"):
            text = text.split("\n", 1)[1]
            if text.endswith("```"):
                text = text[:-3]

        return json.loads(text)

    def _generate_system_prompt(self, profile_data: dict) -> str:
        """Generate a natural language system prompt from flat profile (v1)."""
        text = chat_completion(
            system="You are a prompt engineering expert.",
            messages=[{
                "role": "user",
                "content": f"""Based on this personality profile, write a system prompt that instructs an AI to respond exactly as this person would. The prompt should capture their voice, style, knowledge, and decision-making patterns.

The system prompt should:
1. Define the persona in first person ("You are...")
2. Describe how to communicate (tone, vocabulary, patterns)
3. List areas of knowledge and expertise
4. Describe decision-making tendencies, including known biases
5. Include specific phrases or patterns to use
6. Note their risk tolerance and time horizon preferences
7. Describe how they weigh trade-offs

Profile:
{json.dumps(profile_data, indent=2)}

Write the system prompt directly, no explanations or meta-commentary."""
            }],
            max_tokens=2048,
        )

        return text.strip()

    def _generate_system_prompt_from_dimensions(self, dimension_summary: str) -> str:
        """Generate a system prompt from multi-dimensional persona data (v2)."""
        text = chat_completion(
            system="You are a prompt engineering expert.",
            messages=[{
                "role": "user",
                "content": (
                    "Based on the following multi-dimensional personality profile, write a "
                    "system prompt that instructs an AI to respond exactly as this person would.\n\n"
                    "The system prompt should:\n"
                    "1. Define the persona in second person ('You are...')\n"
                    "2. Capture their communication style and tone\n"
                    "3. Reflect their professional skills and code preferences\n"
                    "4. Include their health/wellness habits and nutrition preferences\n"
                    "5. Reference their goals, values, and life priorities\n"
                    "6. Capture their vibe, entertainment taste, and creative side\n"
                    "7. Note their decision-making patterns and cognitive biases\n"
                    "8. Be concise but comprehensive — this is injected as a system prompt\n\n"
                    f"Dimension Profile:\n{dimension_summary}\n\n"
                    "Write the system prompt directly, no explanations."
                ),
            }],
            max_tokens=2048,
        )

        return text.strip()
