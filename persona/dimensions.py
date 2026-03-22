"""Dimension registry and PersonaDimension dataclass for multi-dimensional persona modeling.

Maps Singularity's pillar system (MIND, BODY, SOUL, SOCIAL, PURPOSE) to
fine-grained persona dimensions, each with its own trait schema.
"""

import copy
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import Any


# ---------------------------------------------------------------------------
# Pillar → Dimension registry
# ---------------------------------------------------------------------------

DIMENSIONS: dict[str, dict[str, str]] = {
    # Pillar: MIND
    "code":          {"pillar": "MIND",    "display": "Code & Engineering"},
    "professional":  {"pillar": "MIND",    "display": "Professional & Career"},
    "learning":      {"pillar": "MIND",    "display": "Learning & Growth"},

    # Pillar: BODY
    "wellness":      {"pillar": "BODY",    "display": "Wellness & Health"},
    "nutrition":     {"pillar": "BODY",    "display": "Nutrition & Food"},

    # Pillar: SOUL
    "creative":      {"pillar": "SOUL",    "display": "Creative"},
    "vibe":          {"pillar": "SOUL",    "display": "Vibe & Energy"},
    "entertainment": {"pillar": "SOUL",    "display": "Entertainment & Media"},

    # Pillar: SOCIAL
    "relationships": {"pillar": "SOCIAL",  "display": "Relationships & Social"},
    "language_style": {"pillar": "SOCIAL", "display": "Language & Style"},

    # Pillar: PURPOSE
    "goals":         {"pillar": "PURPOSE", "display": "Goals & Aspirations"},
    "life":          {"pillar": "PURPOSE", "display": "Life & Daily Living"},
    "progress":      {"pillar": "PURPOSE", "display": "Progress & Tracking"},
}

# Reverse lookup: pillar → list of dimension names
PILLAR_TO_DIMENSIONS: dict[str, list[str]] = {}
for _dim, _meta in DIMENSIONS.items():
    PILLAR_TO_DIMENSIONS.setdefault(_meta["pillar"], []).append(_dim)


# ---------------------------------------------------------------------------
# Trait schemas — what each dimension extracts
# ---------------------------------------------------------------------------

DIMENSION_SCHEMAS: dict[str, dict[str, type]] = {
    "code": {
        "languages": list,
        "frameworks": list,
        "patterns": list,
        "problem_solving_style": str,
        "code_review_preferences": list,
        "preferred_tools": list,
        "architecture_preferences": list,
    },
    "professional": {
        "current_role": str,
        "career_goals": list,
        "skills": list,
        "industry_knowledge": list,
        "work_style": str,
        "leadership_style": str,
    },
    "learning": {
        "learning_style": str,
        "current_topics": list,
        "preferred_resources": list,
        "certifications": list,
        "growth_areas": list,
    },
    "wellness": {
        "exercise_routine": str,
        "exercise_frequency": str,
        "mental_health_practices": list,
        "sleep_patterns": str,
        "health_goals": list,
    },
    "nutrition": {
        "dietary_preferences": list,
        "cuisine_preferences": list,
        "cooking_habits": str,
        "favorite_restaurants": list,
        "food_values": list,
        "meal_patterns": str,
    },
    "creative": {
        "outlets": list,
        "aesthetic_preferences": list,
        "design_sensibility": str,
        "creative_tools": list,
        "creative_process": str,
    },
    "vibe": {
        "energy_patterns": str,
        "mood_patterns": list,
        "music_taste": dict,
        "aesthetic": str,
        "atmosphere_preferences": list,
    },
    "entertainment": {
        "genres": list,
        "platforms": list,
        "favorite_shows_movies": list,
        "reading_habits": str,
        "gaming": str,
        "podcasts": list,
    },
    "relationships": {
        "communication_styles_by_context": dict,
        "social_preferences": list,
        "relationship_values": list,
        "conflict_resolution_style": str,
        "networking_approach": str,
    },
    "language_style": {
        "writing_style": str,
        "vocabulary_level": str,
        "humor_style": str,
        "tone_by_context": dict,
        "common_phrases": list,
        "sentence_patterns": list,
    },
    "goals": {
        "short_term": list,
        "long_term": list,
        "professional": list,
        "personal": list,
        "progress_notes": list,
    },
    "life": {
        "morning_routine": str,
        "evening_routine": str,
        "habits": list,
        "lifestyle_choices": list,
        "living_situation": str,
        "time_management": str,
    },
    "progress": {
        "active_projects": list,
        "milestones_reached": list,
        "current_streaks": list,
        "blockers": list,
        "weekly_wins": list,
    },
}

# ---------------------------------------------------------------------------
# Extraction prompts — tailored per dimension
# ---------------------------------------------------------------------------

DIMENSION_EXTRACTION_PROMPTS: dict[str, str] = {
    "code": (
        "What programming languages do they use most? What frameworks and libraries? "
        "What coding patterns or paradigms do they prefer? How do they approach debugging "
        "and problem solving? What's their code review style? What tools (IDE, CLI, etc.) "
        "do they prefer? What architecture patterns do they gravitate toward?"
    ),
    "professional": (
        "What is their current role and career stage? What are their career goals? "
        "What professional skills do they have? What industries or domains do they know well? "
        "How do they approach work — collaborative, independent, structured, flexible? "
        "What's their leadership or management style?"
    ),
    "learning": (
        "How do they prefer to learn (reading, video, hands-on, courses)? "
        "What topics are they currently studying or interested in? What resources do they use? "
        "Do they have certifications or formal education goals? What areas are they trying to grow in?"
    ),
    "wellness": (
        "What exercise or fitness routine do they follow? How often do they work out? "
        "What mental health practices do they engage in (meditation, therapy, journaling)? "
        "What are their sleep patterns? What health goals are they working toward?"
    ),
    "nutrition": (
        "What dietary preferences or restrictions do they have? What cuisines do they enjoy most? "
        "Do they cook often — what's their cooking style? What are their favorite restaurants or food spots? "
        "What food values matter to them (organic, local, protein-focused, etc.)? "
        "What are their typical meal patterns (meal prep, eating out, snacking habits)?"
    ),
    "creative": (
        "What creative outlets do they pursue (writing, music, art, design, photography)? "
        "What aesthetic do they prefer? What's their design sensibility? "
        "What tools do they use for creative work? How do they approach the creative process?"
    ),
    "vibe": (
        "What's their typical energy level and how does it vary throughout the day? "
        "What mood patterns are visible? What kind of music do they listen to — genres, artists, moods? "
        "What aesthetic or atmosphere do they prefer (minimalist, cozy, vibrant)? "
        "What environments do they thrive in?"
    ),
    "entertainment": (
        "What genres of entertainment do they enjoy (sci-fi, comedy, thriller, etc.)? "
        "What platforms do they use (Netflix, YouTube, Spotify, etc.)? "
        "What are their favorite shows, movies, or series? Do they read — what kind? "
        "Do they game — what kind? What podcasts do they listen to?"
    ),
    "relationships": (
        "How do they communicate differently in professional vs personal contexts? "
        "What are their social preferences (introvert, extrovert, selective)? "
        "What do they value in relationships? How do they handle conflict? "
        "How do they approach networking and building connections?"
    ),
    "language_style": (
        "What's their writing style (concise, elaborate, technical, casual)? "
        "What vocabulary level do they typically use? What kind of humor do they have? "
        "How does their tone change by context (work vs friends vs strangers)? "
        "What phrases or expressions do they commonly use? What sentence patterns are characteristic?"
    ),
    "goals": (
        "What short-term goals are they working on (this week/month)? "
        "What long-term goals or visions do they have? What professional goals are active? "
        "What personal goals matter to them? Any notes on progress toward these goals?"
    ),
    "life": (
        "What does their morning routine look like? Evening routine? "
        "What daily habits do they maintain? What lifestyle choices define them? "
        "What's their living situation? How do they manage their time?"
    ),
    "progress": (
        "What active projects are they working on? What milestones have they recently reached? "
        "What streaks or consistent habits are they maintaining? "
        "What blockers or challenges are they facing? What wins have they had recently?"
    ),
}


# ---------------------------------------------------------------------------
# PersonaDimension dataclass
# ---------------------------------------------------------------------------

@dataclass
class PersonaDimension:
    """A single persona dimension with traits, confidence, and evolution history."""

    name: str
    pillar: str
    display_name: str = ""
    traits: dict[str, Any] = field(default_factory=dict)
    confidence: float = 0.0           # 0.0 – 1.0
    evidence_count: int = 0
    last_updated: str = ""
    history: list[dict] = field(default_factory=list)  # timestamped snapshots

    def __post_init__(self):
        if not self.display_name and self.name in DIMENSIONS:
            self.display_name = DIMENSIONS[self.name]["display"]

    def snapshot(self) -> dict:
        """Create a timestamped snapshot of current state for history tracking."""
        return {
            "timestamp": datetime.now(tz=timezone.utc).isoformat(),
            "traits": copy.deepcopy(self.traits),
            "confidence": self.confidence,
            "evidence_count": self.evidence_count,
        }

    def update(self, traits: dict[str, Any], confidence: float, evidence_count: int) -> None:
        """Update traits and auto-snapshot the previous state to history."""
        if self.traits:  # only snapshot if there was a previous state
            self.history.append(self.snapshot())
        self.traits = traits
        self.confidence = confidence
        self.evidence_count = evidence_count
        self.last_updated = datetime.now(tz=timezone.utc).isoformat()

    def get_summary(self) -> str:
        """Return a natural language summary of this dimension's traits."""
        if not self.traits:
            return f"{self.display_name}: No data yet."
        parts = [f"### {self.display_name} (confidence: {self.confidence:.0%})"]
        for key, value in self.traits.items():
            label = key.replace("_", " ").title()
            if isinstance(value, list) and value:
                parts.append(f"- **{label}**: {', '.join(str(v) for v in value)}")
            elif isinstance(value, dict) and value:
                items = "; ".join(f"{k}: {v}" for k, v in value.items())
                parts.append(f"- **{label}**: {items}")
            elif value:
                parts.append(f"- **{label}**: {value}")
        return "\n".join(parts)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "PersonaDimension":
        return cls(**{k: v for k, v in data.items()
                      if k in cls.__dataclass_fields__})


def create_empty_dimensions() -> dict[str, PersonaDimension]:
    """Create the full set of empty dimensions from the registry."""
    return {
        name: PersonaDimension(
            name=name,
            pillar=meta["pillar"],
            display_name=meta["display"],
        )
        for name, meta in DIMENSIONS.items()
    }
