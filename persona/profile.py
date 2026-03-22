import json
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path

from config import PERSONA_DIR
from .dimensions import (
    PersonaDimension,
    DIMENSIONS,
    create_empty_dimensions,
)

SNAPSHOTS_DIR = PERSONA_DIR / "snapshots"


@dataclass
class PersonaProfile:
    """Structured representation of the user's personality and style.

    v2 adds multi-dimensional persona on top of the legacy flat fields.
    """

    # Legacy fields (backward compat with v1)
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
    risk_tolerance: str = "moderate"
    time_preference: str = "balanced"
    decision_history: list[dict] = field(default_factory=list)
    system_prompt: str = ""

    # v2 fields
    dimensions: dict[str, PersonaDimension] = field(default_factory=dict)
    version: int = 2
    last_full_extraction: str = ""

    # ------------------------------------------------------------------
    # Dimension helpers
    # ------------------------------------------------------------------

    def get_dimension(self, name: str) -> PersonaDimension | None:
        return self.dimensions.get(name)

    def update_dimension(
        self,
        name: str,
        traits: dict,
        confidence: float,
        evidence_count: int,
    ) -> None:
        """Update a single dimension's traits (auto-snapshots previous state)."""
        if name not in self.dimensions:
            meta = DIMENSIONS.get(name, {"pillar": "", "display": name})
            self.dimensions[name] = PersonaDimension(
                name=name,
                pillar=meta["pillar"],
                display_name=meta["display"],
            )
        self.dimensions[name].update(traits, confidence, evidence_count)
        self.save()

    def get_relevant_dimensions(self, dimension_names: list[str]) -> str:
        """Build a prompt-ready summary of the specified dimensions."""
        summaries = []
        for name in dimension_names:
            dim = self.dimensions.get(name)
            if dim and dim.traits:
                summaries.append(dim.get_summary())
        return "\n\n".join(summaries) if summaries else ""

    def get_all_dimensions_summary(self) -> str:
        """Build a prompt-ready summary of ALL populated dimensions."""
        populated = [n for n, d in self.dimensions.items() if d.traits]
        return self.get_relevant_dimensions(populated)

    # ------------------------------------------------------------------
    # Evolution snapshots
    # ------------------------------------------------------------------

    def snapshot_all(self) -> dict:
        """Save a timestamped snapshot of all dimensions to disk."""
        SNAPSHOTS_DIR.mkdir(parents=True, exist_ok=True)
        today = datetime.now(tz=timezone.utc).strftime("%Y-%m-%d")
        snapshot = {
            "timestamp": datetime.now(tz=timezone.utc).isoformat(),
            "date": today,
            "dimensions": {},
        }
        for name, dim in self.dimensions.items():
            snapshot["dimensions"][name] = {
                "confidence": dim.confidence,
                "evidence_count": dim.evidence_count,
                "traits": dim.traits,
            }
        path = SNAPSHOTS_DIR / f"{today}.json"
        path.write_text(json.dumps(snapshot, indent=2, default=str))
        return snapshot

    @staticmethod
    def list_snapshots() -> list[str]:
        """List available snapshot dates."""
        if not SNAPSHOTS_DIR.exists():
            return []
        return sorted(p.stem for p in SNAPSHOTS_DIR.glob("*.json"))

    @staticmethod
    def load_snapshot(date: str) -> dict | None:
        path = SNAPSHOTS_DIR / f"{date}.json"
        if path.exists():
            return json.loads(path.read_text())
        return None

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self) -> None:
        """Save persona profile to disk (v2 + legacy v1 + skill files)."""
        PERSONA_DIR.mkdir(parents=True, exist_ok=True)

        # v2 with dimensions
        v2_data = self._to_v2_dict()
        (PERSONA_DIR / "persona_v2.json").write_text(
            json.dumps(v2_data, indent=2, default=str)
        )

        # Legacy v1 (flat, no dimensions) for backward compat
        v1_data = {k: v for k, v in asdict(self).items()
                   if k not in ("dimensions", "version", "last_full_extraction")}
        (PERSONA_DIR / "persona.json").write_text(
            json.dumps(v1_data, indent=2, default=str)
        )

        # System prompt as plain text
        (PERSONA_DIR / "system_prompt.txt").write_text(self.system_prompt)

        # Regenerate individual skill files from dimensions
        from .skills import write_all_skill_files
        write_all_skill_files(self.dimensions)

    def _to_v2_dict(self) -> dict:
        """Serialize the full profile including dimensions."""
        data = {}
        # Legacy flat fields
        for f in self.__dataclass_fields__:
            if f == "dimensions":
                continue
            data[f] = getattr(self, f)
        # Dimensions as nested dicts
        data["dimensions"] = {
            name: dim.to_dict() for name, dim in self.dimensions.items()
        }
        return data

    @classmethod
    def load(cls) -> "PersonaProfile":
        """Load persona profile from disk. Prefers v2, falls back to v1."""
        v2_path = PERSONA_DIR / "persona_v2.json"
        v1_path = PERSONA_DIR / "persona.json"

        if v2_path.exists():
            return cls._load_v2(v2_path)
        if v1_path.exists():
            return cls._load_v1(v1_path)
        return cls()

    @classmethod
    def _load_v2(cls, path: Path) -> "PersonaProfile":
        data = json.loads(path.read_text())
        dims_raw = data.pop("dimensions", {})

        valid_fields = {f for f in cls.__dataclass_fields__ if f != "dimensions"}
        filtered = {k: v for k, v in data.items() if k in valid_fields}
        profile = cls(**filtered)

        # Reconstruct PersonaDimension objects
        for name, dim_data in dims_raw.items():
            profile.dimensions[name] = PersonaDimension.from_dict(dim_data)

        # Ensure all registry dimensions exist
        for name, meta in DIMENSIONS.items():
            if name not in profile.dimensions:
                profile.dimensions[name] = PersonaDimension(
                    name=name,
                    pillar=meta["pillar"],
                    display_name=meta["display"],
                )

        # If all dimensions are empty but we have legacy data, migrate
        all_empty = all(not d.traits for d in profile.dimensions.values())
        has_legacy = bool(profile.communication_style.get("tone") or profile.knowledge_domains)
        if all_empty and has_legacy:
            profile.migrate_from_v1()

        return profile

    @classmethod
    def _load_v1(cls, path: Path) -> "PersonaProfile":
        data = json.loads(path.read_text())
        valid_fields = {f for f in cls.__dataclass_fields__
                        if f not in ("dimensions", "version", "last_full_extraction")}
        filtered = {k: v for k, v in data.items() if k in valid_fields}
        profile = cls(**filtered)
        profile.version = 2  # upgrade
        profile.dimensions = create_empty_dimensions()
        profile.migrate_from_v1()
        return profile

    def migrate_from_v1(self) -> None:
        """Map legacy flat fields into the new dimension system."""
        # Language & Style from communication_style
        if self.communication_style:
            cs = self.communication_style
            self.dimensions["language_style"].traits = {
                "writing_style": cs.get("tone", ""),
                "vocabulary_level": cs.get("vocabulary_level", ""),
                "humor_style": "",
                "tone_by_context": {"general": cs.get("formality", "")},
                "common_phrases": cs.get("common_phrases", []),
                "sentence_patterns": cs.get("sentence_patterns", []),
            }
            self.dimensions["language_style"].confidence = 0.5

        # Professional from knowledge_domains
        if self.knowledge_domains:
            self.dimensions["professional"].traits = {
                "current_role": "",
                "career_goals": [],
                "skills": self.knowledge_domains[:],
                "industry_knowledge": [],
                "work_style": "",
                "leadership_style": "",
            }
            self.dimensions["professional"].confidence = 0.3

        # Goals from values_and_priorities
        if self.values_and_priorities:
            self.dimensions["goals"].traits = {
                "short_term": [],
                "long_term": [],
                "professional": [],
                "personal": self.values_and_priorities[:],
                "progress_notes": [],
            }
            self.dimensions["goals"].confidence = 0.3

        # Spread interests across relevant dimensions
        if self.interests:
            self.dimensions["entertainment"].traits = {
                "genres": [],
                "platforms": [],
                "favorite_shows_movies": [],
                "reading_habits": "",
                "gaming": "",
                "podcasts": [],
            }
            self.dimensions["vibe"].traits = {
                "energy_patterns": "",
                "mood_patterns": [],
                "music_taste": {},
                "aesthetic": "",
                "atmosphere_preferences": [],
            }

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
