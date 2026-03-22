"""Two-tier chunk classifier: rule-based (free) + LLM batch (for ambiguous chunks).

Assigns a `pillar` and `dimension` to every chunk before it enters ChromaDB.
"""

import json
import re

from .dimensions import DIMENSIONS, PILLAR_TO_DIMENSIONS


# ---------------------------------------------------------------------------
# Tier 1 — Rule-based classification (zero cost, instant)
# ---------------------------------------------------------------------------

# Metadata type → (pillar, dimension)
TYPE_MAP: dict[str, tuple[str, str]] = {
    "body_gym":          ("BODY",    "wellness"),
    "body_nutrition":    ("BODY",    "nutrition"),
    "browser_daily":     ("MIND",    "learning"),     # default; refined by keywords
    "browser_domain":    ("MIND",    "learning"),
    "task":              ("PURPOSE", "goals"),         # refined by pillar if present
    "note":              ("",        ""),              # too varied; needs Tier 2
    "singularity_entry": ("",        ""),              # use pillar from metadata
    "data_point":        ("",        ""),              # classify by content
    "user_message":      ("",        ""),              # classify by content
    "conversation_pair": ("",        ""),              # classify by content
}

# Singularity pillar → default dimension
PILLAR_DEFAULT_DIMENSION: dict[str, str] = {
    "MIND":    "professional",
    "BODY":    "wellness",
    "SOUL":    "creative",
    "SOCIAL":  "relationships",
    "PURPOSE": "goals",
    "WEALTH":  "professional",
    "SPACE":   "life",
}

# Keyword patterns → dimension (checked in order; first match wins)
KEYWORD_RULES: list[tuple[str, str]] = [
    # Code & Engineering
    (r"\b(python|javascript|typescript|java|rust|golang|react|fastapi|django|flask|"
     r"docker|kubernetes|git|github|deploy|cicd|ci/cd|backend|frontend|fullstack|"
     r"api|endpoint|database|sql|nosql|mongodb|postgres|redis|aws|gcp|azure|"
     r"terraform|ansible|nginx|linux|bash|shell|debug|refactor|PR|pull request|"
     r"commit|branch|merge|test|unittest|pytest|npm|pip|conda|venv)\b", "code"),

    # Wellness
    (r"\b(gym|workout|exercise|fitness|cardio|lifting|weights|yoga|meditation|"
     r"mindfulness|therapy|mental health|sleep|rest|recovery|stretch|run|jog|"
     r"walk|steps|heart rate|blood pressure|health check)\b", "wellness"),

    # Nutrition
    (r"\b(breakfast|lunch|dinner|meal prep|recipe|cook|calorie|protein|carb|fat|"
     r"vegetable|fruit|supplement|vitamin|diet|vegan|vegetarian|keto|fasting|"
     r"hydration|water intake|restaurant|cafe|coffee shop|food)\b", "nutrition"),

    # Entertainment
    (r"\b(movie|film|show|series|netflix|spotify|youtube|podcast|album|song|"
     r"playlist|anime|manga|book|novel|game|gaming|xbox|playstation|steam|"
     r"twitch|concert|festival|theater|theatre|tame impala|bonobo|"
     r"listening to|watching|binge|streaming)\b", "entertainment"),

    # Creative
    (r"\b(design|art|photo|photography|music production|beat|synth|draw|paint|"
     r"illustration|typography|ui design|ux design|figma|sketch|creative|"
     r"writing|blog|content creation|video edit)\b", "creative"),

    # Vibe
    (r"\b(vibe|mood|energy|aesthetic|atmosphere|ambiance|chill|hype|calm|focus|"
     r"lofi|lo-fi|playlist mood|morning energy|night owl|music|tempo|beats)\b", "vibe"),

    # Relationships
    (r"\b(friend|family|partner|relationship|social|networking|community|"
     r"team|collaboration|mentoring|mentor|dating|communication style)\b", "relationships"),

    # Goals
    (r"\b(goal|target|milestone|deadline|resolution|objective|plan|roadmap|"
     r"quarter|okr|kpi|track progress|accountability|habit track)\b", "goals"),

    # Life
    (r"\b(morning routine|evening routine|daily routine|habit|lifestyle|"
     r"apartment|house|move|relocate|commute|time management|schedule|"
     r"productivity system|todo|planner)\b", "life"),

    # Progress
    (r"\b(progress|streak|wins|milestone reached|shipped|launched|completed|"
     r"achievement|retrospective|weekly review|daily standup)\b", "progress"),

    # Professional
    (r"\b(career|job|salary|promotion|interview|resume|linkedin|startup|"
     r"company|business|entrepreneur|freelance|consulting|client|manager|"
     r"lead|senior|principal|staff|director)\b", "professional"),

    # Learning
    (r"\b(learn|course|tutorial|documentation|study|research|paper|article|"
     r"conference|workshop|certification|bootcamp|mooc|udemy|coursera)\b", "learning"),
]

# Browser site_classifier category → dimension
BROWSER_CATEGORY_MAP: dict[str, str] = {
    "work":         "professional",
    "learning":     "learning",
    "google_tools": "professional",
    "gmail":        "professional",
    "linkedin":     "professional",
    "news":         "learning",
    "entertainment":"entertainment",
    "social":       "relationships",
    "shopping":     "life",
    "lifestyle":    "life",
    "other":        "",
}


class ChunkClassifier:
    """Classifies chunks into pillar + dimension using rules and optionally LLM."""

    def __init__(self):
        self._compiled_rules: list[tuple[re.Pattern, str]] = [
            (re.compile(pattern, re.IGNORECASE), dim)
            for pattern, dim in KEYWORD_RULES
        ]

    def classify_chunk(self, text: str, metadata: dict) -> tuple[str, str]:
        """Return (pillar, dimension) for a chunk. Empty strings if ambiguous.

        Tries, in order:
        1. Metadata type mapping
        2. Existing pillar in metadata → default dimension
        3. Keyword matching on text
        """
        chunk_type = metadata.get("type", "")
        existing_pillar = metadata.get("pillar", "")

        # 1. Type-based mapping
        if chunk_type in TYPE_MAP:
            pillar, dimension = TYPE_MAP[chunk_type]
            if pillar and dimension:
                return pillar, dimension

        # 2. Existing pillar from Singularity data → default dimension
        if existing_pillar and existing_pillar in PILLAR_DEFAULT_DIMENSION:
            dimension = PILLAR_DEFAULT_DIMENSION[existing_pillar]
            return existing_pillar, dimension

        # 3. Browser category mapping
        if chunk_type in ("browser_daily", "browser_domain"):
            # Try to detect from text content
            dim = self._keyword_classify(text)
            if dim:
                pillar = DIMENSIONS[dim]["pillar"]
                return pillar, dim
            return "MIND", "learning"  # fallback for browser

        # 4. Keyword-based classification on text content
        dim = self._keyword_classify(text)
        if dim:
            pillar = DIMENSIONS[dim]["pillar"]
            return pillar, dim

        return "", ""  # unclassified — needs Tier 2

    def _keyword_classify(self, text: str) -> str:
        """Match text against keyword rules. Return dimension name or empty string."""
        for pattern, dimension in self._compiled_rules:
            if pattern.search(text):
                return dimension
        return ""

    def classify_text(self, text: str) -> list[str]:
        """Return ALL matching dimensions for a text (for query routing)."""
        matches = []
        for pattern, dimension in self._compiled_rules:
            if pattern.search(text):
                matches.append(dimension)
        return list(dict.fromkeys(matches))  # deduplicate, preserve order

    # ------------------------------------------------------------------
    # Tier 2 — LLM batch classification
    # ------------------------------------------------------------------

    def batch_classify_llm(self, chunks: list[dict]) -> list[tuple[str, str]]:
        """Classify a batch of chunks using the configured LLM. Returns list of (pillar, dimension).

        Each item in chunks should have 'text' and 'id' keys.
        """
        if not chunks:
            return []

        from twin.llm_client import chat_completion

        dim_list = ", ".join(DIMENSIONS.keys())

        # Build numbered list for the prompt
        numbered = []
        for i, chunk in enumerate(chunks):
            text = chunk["text"][:300]  # truncate for efficiency
            numbered.append(f"{i+1}. {text}")
        numbered_text = "\n".join(numbered)

        raw = chat_completion(
            system="You are a text classifier. Return only valid JSON.",
            messages=[{
                "role": "user",
                "content": (
                    f"Classify each numbered text snippet into exactly one dimension.\n"
                    f"Available dimensions: {dim_list}\n\n"
                    f"Texts:\n{numbered_text}\n\n"
                    f"Return a JSON array of objects, one per text, each with "
                    f"\"index\" (1-based) and \"dimension\" fields. "
                    f"Return ONLY the JSON array."
                ),
            }],
            max_tokens=4096,
        ).strip()
        if raw.startswith("```"):
            raw = raw.split("\n", 1)[1]
            if raw.endswith("```"):
                raw = raw[:-3]

        try:
            results = json.loads(raw)
        except json.JSONDecodeError:
            return [("", "")] * len(chunks)

        # Map results back
        classified = [("", "")] * len(chunks)
        for item in results:
            idx = item.get("index", 0) - 1
            dim = item.get("dimension", "")
            if 0 <= idx < len(chunks) and dim in DIMENSIONS:
                pillar = DIMENSIONS[dim]["pillar"]
                classified[idx] = (pillar, dim)

        return classified
