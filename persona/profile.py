import json
from dataclasses import dataclass, field, asdict
from pathlib import Path

from config import PERSONA_DIR


@dataclass
class PersonaProfile:
    """Structured representation of the user's personality and style."""

    communication_style: dict = field(default_factory=lambda: {
        "tone": "",
        "formality": "",
        "vocabulary_level": "",
        "sentence_patterns": [],
        "common_phrases": [],
    })
    knowledge_domains: list[str] = field(default_factory=list)
    decision_patterns: list[str] = field(default_factory=list)
    values_and_priorities: list[str] = field(default_factory=list)
    interests: list[str] = field(default_factory=list)
    cognitive_biases: list[str] = field(default_factory=list)
    risk_tolerance: str = "moderate"  # "low" | "moderate" | "high"
    time_preference: str = "balanced"  # "short-term" | "balanced" | "long-term"
    decision_history: list[dict] = field(default_factory=list)  # {"question", "decision", "outcome", "date"}
    system_prompt: str = ""

    def save(self):
        """Save persona profile to disk."""
        PERSONA_DIR.mkdir(parents=True, exist_ok=True)

        # Save structured profile
        profile_path = PERSONA_DIR / "persona.json"
        profile_path.write_text(json.dumps(asdict(self), indent=2))

        # Save system prompt
        prompt_path = PERSONA_DIR / "system_prompt.txt"
        prompt_path.write_text(self.system_prompt)

        print(f"Persona saved to {PERSONA_DIR}")

    @classmethod
    def load(cls) -> "PersonaProfile":
        """Load persona profile from disk."""
        profile_path = PERSONA_DIR / "persona.json"
        if not profile_path.exists():
            return cls()

        data = json.loads(profile_path.read_text())
        # Handle loading profiles saved before new fields were added
        valid_fields = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in data.items() if k in valid_fields}
        return cls(**filtered)

    @classmethod
    def load_system_prompt(cls) -> str:
        """Load just the system prompt."""
        prompt_path = PERSONA_DIR / "system_prompt.txt"
        if prompt_path.exists():
            return prompt_path.read_text()
        return ""

    def add_decision(self, question: str, decision: str, outcome: str = "", date: str = ""):
        """Record a decision for future reference."""
        self.decision_history.append({
            "question": question,
            "decision": decision,
            "outcome": outcome,
            "date": date,
        })
        self.save()
