"""
Skill Files — per-dimension markdown files that represent who you are.

Each dimension of the persona gets its own SKILL.md file under
data/persona/skills/. These files:
  1. Auto-generate from PersonaDimension traits (populated by the daily loop)
  2. Are human-readable — you can open and edit them
  3. Get composed into the system prompt automatically
  4. Evolve every time the daily loop runs with new data

This replaces any hardcoded persona prompt. The twin learns who you are
entirely from your real data — notes, browsing, tasks, habits, goals.
"""

import json
from datetime import datetime, timezone
from pathlib import Path

from config import PERSONA_DIR
from .dimensions import DIMENSIONS, PersonaDimension

SKILLS_DIR = PERSONA_DIR / "skills"


def generate_skill_file(dim: PersonaDimension) -> str:
    """Generate a markdown skill file from a PersonaDimension."""
    lines = []
    lines.append(f"# {dim.display_name}")
    lines.append(f"<!-- pillar: {dim.pillar} | dimension: {dim.name} -->")
    lines.append(f"<!-- confidence: {dim.confidence:.0%} | evidence: {dim.evidence_count} chunks -->")
    if dim.last_updated:
        lines.append(f"<!-- last updated: {dim.last_updated[:10]} -->")
    lines.append("")

    if not dim.traits:
        lines.append("*No data yet. This dimension will populate as more data flows in.*")
        return "\n".join(lines)

    for key, value in dim.traits.items():
        label = key.replace("_", " ").title()

        if isinstance(value, list) and value:
            lines.append(f"## {label}")
            for item in value:
                if isinstance(item, dict):
                    lines.append(f"- {json.dumps(item)}")
                else:
                    lines.append(f"- {item}")
            lines.append("")

        elif isinstance(value, dict) and value:
            lines.append(f"## {label}")
            for k, v in value.items():
                lines.append(f"- **{k}**: {v}")
            lines.append("")

        elif isinstance(value, str) and value:
            lines.append(f"## {label}")
            lines.append(value)
            lines.append("")

    return "\n".join(lines)


def write_all_skill_files(dimensions: dict[str, PersonaDimension]) -> list[Path]:
    """Write all dimension skill files to disk. Returns list of written paths."""
    SKILLS_DIR.mkdir(parents=True, exist_ok=True)
    written = []
    for name, dim in dimensions.items():
        content = generate_skill_file(dim)
        path = SKILLS_DIR / f"{name}.md"
        path.write_text(content, encoding="utf-8")
        written.append(path)
    return written


def read_all_skill_files() -> dict[str, str]:
    """Read all skill files from disk. Returns {dimension_name: content}."""
    if not SKILLS_DIR.exists():
        return {}
    skills = {}
    for path in sorted(SKILLS_DIR.glob("*.md")):
        skills[path.stem] = path.read_text(encoding="utf-8")
    return skills


def build_persona_from_skills() -> str:
    """Compose a system prompt from all populated skill files.

    This replaces any hardcoded persona description. The twin's understanding
    of who you are comes entirely from these evolving skill files.
    """
    skills = read_all_skill_files()
    if not skills:
        return ""

    parts = []
    parts.append("You are a digital twin of a real person. Everything below describes who they are,")
    parts.append("extracted from their real data — notes, browsing, tasks, habits, health, goals.")
    parts.append("This evolves daily as new data arrives. Use it to respond exactly as they would.\n")

    # Group by pillar for cleaner presentation
    pillar_groups: dict[str, list[tuple[str, str]]] = {}
    for dim_name, content in skills.items():
        if "*No data yet" in content:
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
            # Strip the HTML comments and title for a cleaner prompt
            lines = content.split("\n")
            clean_lines = [l for l in lines if not l.startswith("<!--")]
            parts.append("\n".join(clean_lines))

    return "\n".join(parts)
