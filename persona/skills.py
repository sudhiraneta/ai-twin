"""
Skill Files — per-dimension markdown files that represent who you are.

Each skill file has two sections:
  1. **Traits** — what the twin knows about you (auto-extracted, editable)
  2. **Sources** — where to pull data for this dimension (retrieval orchestration)

The Sources section tells the twin which data sources and chunk types to query
when a question maps to this dimension. This is the workflow orchestration layer.
"""

import json
from datetime import datetime, timezone
from pathlib import Path

from config import PERSONA_DIR
from .dimensions import DIMENSIONS, PersonaDimension

SKILLS_DIR = PERSONA_DIR / "skills"

# Default retrieval sources per dimension — what to search when this dimension is relevant
DIMENSION_SOURCES: dict[str, dict] = {
    "code": {
        "description": "Programming languages, frameworks, coding patterns",
        "search_types": ["data_point", "singularity_entry", "pillar_journal"],
        "search_queries": ["python rust code programming framework deploy"],
        "data_sources": ["singularity_db (career entries)", "self_reported data points"],
    },
    "professional": {
        "description": "Career, work, Drata, leadership, influence",
        "search_types": ["singularity_entry", "pillar_journal", "browser_daily", "data_point"],
        "search_queries": ["career work drata engineer leadership influence articulation"],
        "data_sources": ["singularity_db (career pillar)", "browser (work sites)", "weekly reviews"],
    },
    "learning": {
        "description": "Courses, books, tutorials, growth areas",
        "search_types": ["browser_domain", "browser_daily", "singularity_entry", "plan_note"],
        "search_queries": ["learning course tutorial book deeplearning youtube education"],
        "data_sources": ["browser (learning sites, YouTube)", "plan notes (30-day plan)", "singularity_db"],
    },
    "wellness": {
        "description": "Gym frequency, workouts, meditation, sleep, health goals",
        "search_types": ["body_gym", "body_nutrition", "soul_checkin", "singularity_entry", "pillar_journal"],
        "search_queries": ["gym workout meditation sleep health journal body wellness"],
        "data_sources": ["body_tracker (gym/nutrition)", "soul_checkin (habits)", "singularity_db (body pillar)"],
    },
    "nutrition": {
        "description": "Dietary preferences, cuisines, cooking, restaurants",
        "search_types": ["body_nutrition", "data_point", "singularity_entry"],
        "search_queries": ["vegetarian food thai indian protein restaurant cooking meal"],
        "data_sources": ["body_tracker (nutrition analysis)", "self_reported preferences", "notes"],
    },
    "creative": {
        "description": "Content creation, music, writing, creative outlets",
        "search_types": ["data_point", "browser_domain", "singularity_entry"],
        "search_queries": ["youtube music content creative writing substack"],
        "data_sources": ["browser (YouTube, creative sites)", "self_reported", "singularity_db (create pillar)"],
    },
    "vibe": {
        "description": "Energy patterns, mood, atmosphere, aesthetic preferences",
        "search_types": ["soul_checkin", "data_point", "singularity_entry"],
        "search_queries": ["energy morning calm quiet vibe atmosphere mood"],
        "data_sources": ["soul_checkin (daily signals)", "self_reported preferences"],
    },
    "entertainment": {
        "description": "Movies, shows, music, podcasts, gaming",
        "search_types": ["browser_domain", "data_point"],
        "search_queries": ["netflix youtube movies shows music podcast sci fi entertainment"],
        "data_sources": ["browser (entertainment sites, YouTube)", "self_reported"],
    },
    "relationships": {
        "description": "Communication style, social preferences, networking",
        "search_types": ["data_point", "singularity_entry", "pillar_journal"],
        "search_queries": ["communication networking social relationship direct selective"],
        "data_sources": ["singularity_db (voice pillar)", "self_reported", "card_counters (networking reps)"],
    },
    "language_style": {
        "description": "Writing tone, vocabulary, common phrases, formality",
        "search_types": ["data_point", "user_message", "conversation_pair"],
        "search_queries": ["casual direct technical writing style communication"],
        "data_sources": ["conversation history", "self_reported", "AI chat exports"],
    },
    "goals": {
        "description": "Short/long-term goals, professional targets, personal aspirations",
        "search_types": ["plan_note", "weekly_review", "goals_completed", "singularity_entry", "task"],
        "search_queries": ["goal plan target ship project executive quarter objective"],
        "data_sources": ["plan notes (30-day plan)", "weekly reviews", "tasks", "goals archive"],
    },
    "life": {
        "description": "Daily routines, shopping, living situation, habits",
        "search_types": ["browser_daily", "browser_domain", "soul_checkin", "data_point"],
        "search_queries": ["morning routine shopping zara daily life cafe cooked"],
        "data_sources": ["browser (shopping, daily sites)", "soul_checkin (habits)", "self_reported"],
    },
    "progress": {
        "description": "Weekly wins, milestones, streaks, active projects",
        "search_types": ["weekly_review", "week_carry", "card_counters", "goals_completed", "singularity_entry"],
        "search_queries": ["progress week wins score milestone streak journal"],
        "data_sources": ["weekly reviews", "week carries", "card counters (rep milestones)", "singularity_db"],
    },
}


def generate_skill_file(dim: PersonaDimension) -> str:
    """Generate a markdown skill file with traits + retrieval sources."""
    lines = []
    lines.append(f"# {dim.display_name}")
    lines.append(f"<!-- pillar: {dim.pillar} | dimension: {dim.name} -->")
    lines.append(f"<!-- confidence: {dim.confidence:.0%} | evidence: {dim.evidence_count} chunks -->")
    if dim.last_updated:
        lines.append(f"<!-- last updated: {dim.last_updated[:10]} -->")
    lines.append("")

    # Traits section
    lines.append("## Traits")
    lines.append("")

    if not dim.traits:
        lines.append("*No data yet. This dimension will populate as more data flows in.*")
    else:
        for key, value in dim.traits.items():
            label = key.replace("_", " ").title()

            if isinstance(value, list) and value:
                lines.append(f"### {label}")
                for item in value:
                    if isinstance(item, dict):
                        lines.append(f"- {json.dumps(item)}")
                    else:
                        lines.append(f"- {item}")
                lines.append("")

            elif isinstance(value, dict) and value:
                lines.append(f"### {label}")
                for k, v in value.items():
                    lines.append(f"- **{k}**: {v}")
                lines.append("")

            elif isinstance(value, str) and value:
                lines.append(f"### {label}")
                lines.append(value)
                lines.append("")

    # Sources section — retrieval orchestration
    lines.append("")
    lines.append("## Sources")
    lines.append("")

    sources = DIMENSION_SOURCES.get(dim.name, {})
    if sources:
        lines.append(f"**What this covers:** {sources.get('description', '')}")
        lines.append("")
        lines.append("**Where to pull data:**")
        for ds in sources.get("data_sources", []):
            lines.append(f"- {ds}")
        lines.append("")
        lines.append("**Chunk types to search:**")
        lines.append(f"`{', '.join(sources.get('search_types', []))}`")
        lines.append("")
        lines.append("**Default search queries:**")
        for q in sources.get("search_queries", []):
            lines.append(f"- `{q}`")
    else:
        lines.append("*No source configuration yet.*")

    lines.append("")
    return "\n".join(lines)


def write_all_skill_files(dimensions: dict[str, PersonaDimension]) -> list[Path]:
    """Write all dimension skill files to disk. Returns list of written paths."""
    SKILLS_DIR.mkdir(parents=True, exist_ok=True)
    written = []
    for name, dim in dimensions.items():
        path = SKILLS_DIR / f"{name}.md"
        # Only overwrite the Traits section if file exists with user edits
        if path.exists():
            existing = path.read_text(encoding="utf-8")
            # Check if user added custom content (not auto-generated)
            if "<!-- user-edited -->" in existing:
                # Only update the Traits section, preserve Sources and custom content
                new_content = generate_skill_file(dim)
                # Keep user's Sources section if they edited it
                if "## Sources" in existing:
                    user_sources = existing[existing.index("## Sources"):]
                    new_traits = new_content[:new_content.index("## Sources")]
                    content = new_traits + user_sources
                else:
                    content = new_content
            else:
                content = generate_skill_file(dim)
        else:
            content = generate_skill_file(dim)

        path.write_text(content, encoding="utf-8")
        written.append(path)
    return written


def read_skill_file(dimension_name: str) -> str | None:
    """Read a single skill file. Returns content or None."""
    path = SKILLS_DIR / f"{dimension_name}.md"
    if path.exists():
        return path.read_text(encoding="utf-8")
    return None


def write_skill_file(dimension_name: str, content: str) -> bool:
    """Write a single skill file (user edit). Marks as user-edited."""
    SKILLS_DIR.mkdir(parents=True, exist_ok=True)
    path = SKILLS_DIR / f"{dimension_name}.md"

    # Add user-edited marker if not present
    if "<!-- user-edited -->" not in content:
        # Insert after the first HTML comment block
        lines = content.split("\n")
        insert_at = 0
        for i, line in enumerate(lines):
            if line.startswith("<!--"):
                insert_at = i + 1
        lines.insert(insert_at, "<!-- user-edited -->")
        content = "\n".join(lines)

    path.write_text(content, encoding="utf-8")
    return True


def read_all_skill_files() -> dict[str, str]:
    """Read all skill files from disk. Returns {dimension_name: content}."""
    if not SKILLS_DIR.exists():
        return {}
    skills = {}
    for path in sorted(SKILLS_DIR.glob("*.md")):
        skills[path.stem] = path.read_text(encoding="utf-8")
    return skills


def build_persona_from_skills() -> str:
    """Compose a system prompt from all populated skill files."""
    skills = read_all_skill_files()
    if not skills:
        return ""

    parts = []
    parts.append("You are a digital twin of a real person. Everything below describes who they are,")
    parts.append("extracted from their real data — notes, browsing, tasks, habits, health, goals.")
    parts.append("This evolves daily as new data arrives. Use it to respond exactly as they would.")
    parts.append("Each dimension also lists WHERE to find supporting data — use the Sources section")
    parts.append("to know which memory types to cite in your responses.\n")

    pillar_groups: dict[str, list[tuple[str, str]]] = {}
    for dim_name, content in skills.items():
        if "*No data yet" in content and "## Sources" not in content:
            continue
        meta = DIMENSIONS.get(dim_name, {})
        pillar = meta.get("pillar", "OTHER")
        pillar_groups.setdefault(pillar, []).append((dim_name, content))

    pillar_order = ["MIND", "BODY", "SOUL", "SOCIAL", "PURPOSE"]
    pillar_labels = {
        "MIND": "Mind & Career",
        "BODY": "Body & Health",
        "SOUL": "Soul & Creativity",
        "SOCIAL": "Social & Communication",
        "PURPOSE": "Purpose & Goals",
    }

    for pillar in pillar_order:
        group = pillar_groups.get(pillar)
        if not group:
            continue
        parts.append(f"\n---\n## {pillar_labels.get(pillar, pillar)}\n")
        for dim_name, content in group:
            lines = content.split("\n")
            clean_lines = [l for l in lines if not l.startswith("<!--")]
            parts.append("\n".join(clean_lines))

    return "\n".join(parts)
