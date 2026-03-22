import json

import anthropic

from config import ANTHROPIC_API_KEY, CLAUDE_MODEL
from .profile import PersonaProfile


class PersonaExtractor:
    """Extracts a persona profile from user conversation history using Claude."""

    def __init__(self):
        self.client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

    def extract(self, user_messages: list[str], max_sample: int = 500) -> PersonaProfile:
        """Analyze user messages to build a persona profile."""
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

    def _analyze_messages(self, messages_text: str) -> dict:
        """Use Claude to analyze user messages and extract persona traits."""
        response = self.client.messages.create(
            model=CLAUDE_MODEL,
            max_tokens=4096,
            messages=[{
                "role": "user",
                "content": f"""Analyze the following messages written by a single person. Extract their complete personality and decision-making profile.

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
            }]
        )

        text = response.content[0].text.strip()
        if text.startswith("```"):
            text = text.split("\n", 1)[1]
            if text.endswith("```"):
                text = text[:-3]

        return json.loads(text)

    def _generate_system_prompt(self, profile_data: dict) -> str:
        """Generate a natural language system prompt from the profile."""
        response = self.client.messages.create(
            model=CLAUDE_MODEL,
            max_tokens=2048,
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
            }]
        )

        return response.content[0].text.strip()
