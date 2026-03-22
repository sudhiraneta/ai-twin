"""
Analytics connector — imports Singularity's weekly reviews, soul checkin signals,
goals archive, card counters, plan notes, and week carries into the AI Twin's
RAG pipeline so the twin knows about wins, habits, progress, and growth patterns.
"""

import json
import re
from datetime import datetime, timezone
from pathlib import Path

from config import SINGULARITY_DIR
from memory.chunker import Chunk, _ensure_metadata
from .base import BaseConnector

LOGS_DIR = SINGULARITY_DIR / "logs"


class AnalyticsConnector(BaseConnector):
    """Connector for Singularity analytics: reviews, checkins, goals, plan notes."""

    source_name = "analytics"

    def fetch(self, since: float | None = None, **kwargs) -> list[Chunk]:
        chunks = []
        chunks.extend(self._fetch_weekly_reviews(since))
        chunks.extend(self._fetch_soul_checkin(since))
        chunks.extend(self._fetch_goals_archive(since))
        chunks.extend(self._fetch_card_counters())
        chunks.extend(self._fetch_plan_notes())
        chunks.extend(self._fetch_week_carry())
        chunks.extend(self._fetch_pillar_journals(since))
        return chunks

    # ------------------------------------------------------------------
    # Weekly Reviews (logs/weekly/*.md)
    # ------------------------------------------------------------------
    def _fetch_weekly_reviews(self, since: float | None) -> list[Chunk]:
        weekly_dir = LOGS_DIR / "weekly"
        if not weekly_dir.exists():
            return []

        chunks = []
        for md_file in sorted(weekly_dir.glob("*.md")):
            # Parse week from filename like 2026-W10.md
            week_str = md_file.stem  # e.g. "2026-W10"
            content = md_file.read_text(encoding="utf-8")

            # Extract key sections for a dense chunk
            summary = self._extract_review_summary(content, week_str)

            chunks.append(Chunk(
                text=summary,
                metadata=_ensure_metadata({
                    "source": self.source_name,
                    "conversation_id": f"weekly_review_{week_str}",
                    "title": f"Weekly Review {week_str}",
                    "timestamp": self._week_to_iso(week_str),
                    "msg_timestamp": self._week_to_iso(week_str),
                    "role": "user",
                    "type": "weekly_review",
                    "pillar": "PURPOSE",
                    "dimension": "progress",
                    "classified": "true",
                }),
            ))

        return chunks

    def _extract_review_summary(self, content: str, week_str: str) -> str:
        """Extract the most useful parts of a weekly review for RAG."""
        lines = [f"Weekly Review: {week_str}"]

        # Week at a glance metrics
        labels_patterns = [
            ("Notes logged", r"Total notes logged\s*\|\s*(\d+)"),
            ("High-signal entries", r"High-signal entries.*?\|\s*(\d+)"),
            ("Average val_signal", r"Average val_signal\s*\|\s*([\d.]+)"),
            ("Most active pillar", r"Most active pillar\s*\|\s*(\w+)"),
        ]
        for label, pattern in labels_patterns:
            m = re.search(pattern, content)
            if m:
                lines.append(f"{label}: {m.group(1)}")

        # Pillar pulse (compact)
        pulse_lines = re.findall(r"\*\*(\w+)\*\*:.*?(🔥.*?|🟩.*?|🟨.*?|⬜.*?)$", content, re.MULTILINE)
        if pulse_lines:
            lines.append("\nPillar pulse:")
            for pillar, status in pulse_lines:
                lines.append(f"  {pillar}: {status.strip()}")

        # High-signal entries
        high_entries = re.findall(r"\*\*GEN_\d+\*\*.*?score ([\d.]+).*?_(.*?)_", content)
        if high_entries:
            lines.append("\nTop wins:")
            for score, title in high_entries[:5]:
                lines.append(f"  - {title.strip()} (score {score})")

        # Reflection prompts
        reflections = re.findall(r"Reflection prompt:\*\*\s*_(.*?)_", content)
        if reflections:
            lines.append("\nReflection prompts:")
            for r in reflections:
                lines.append(f"  - {r}")

        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Soul Checkin Signals (logs/soul_checkin.md)
    # ------------------------------------------------------------------
    def _fetch_soul_checkin(self, since: float | None) -> list[Chunk]:
        checkin_file = LOGS_DIR / "soul_checkin.md"
        if not checkin_file.exists():
            return []

        content = checkin_file.read_text(encoding="utf-8")
        entries = {}
        for line in content.split("\n"):
            m = re.match(r"(\d{4}-\d{2}-\d{2}):\s*(.+)", line.strip())
            if m:
                date_str = m.group(1)
                signals = [s.strip() for s in m.group(2).split() if s.strip()]
                entries[date_str] = signals

        if not entries:
            return []

        # Create weekly summary chunks (more useful than daily)
        chunks = []
        sorted_dates = sorted(entries.keys())

        # Group by week
        weeks = {}
        for date_str in sorted_dates:
            dt = datetime.strptime(date_str, "%Y-%m-%d")
            week_key = dt.strftime("%Y-W%W")
            if week_key not in weeks:
                weeks[week_key] = {"dates": [], "signals": {}}
            weeks[week_key]["dates"].append(date_str)
            for sig in entries[date_str]:
                weeks[week_key]["signals"][sig] = weeks[week_key]["signals"].get(sig, 0) + 1

        for week_key, data in weeks.items():
            all_signals = [f"{sig} ({count}x)" for sig, count in
                          sorted(data["signals"].items(), key=lambda x: -x[1])]
            days_active = len(data["dates"])
            total_signals = sum(data["signals"].values())

            text_lines = [
                f"Soul Checkin: {week_key}",
                f"Active days: {days_active}/7",
                f"Total signals: {total_signals}",
                f"Habits tracked: {', '.join(all_signals)}",
            ]
            # Per-day detail
            for d in data["dates"]:
                text_lines.append(f"  {d}: {', '.join(entries[d])}")

            chunks.append(Chunk(
                text="\n".join(text_lines),
                metadata=_ensure_metadata({
                    "source": self.source_name,
                    "conversation_id": f"soul_checkin_{week_key}",
                    "title": f"Soul checkin {week_key}",
                    "timestamp": f"{data['dates'][0]}T00:00:00+00:00",
                    "msg_timestamp": f"{data['dates'][-1]}T00:00:00+00:00",
                    "role": "user",
                    "type": "soul_checkin",
                    "pillar": "SOUL",
                    "dimension": "wellness",
                    "classified": "true",
                }),
            ))

        return chunks

    # ------------------------------------------------------------------
    # Goals Archive (logs/goals_archive.md)
    # ------------------------------------------------------------------
    def _fetch_goals_archive(self, since: float | None) -> list[Chunk]:
        archive_file = LOGS_DIR / "goals_archive.md"
        if not archive_file.exists():
            return []

        content = archive_file.read_text(encoding="utf-8")
        if not content.strip():
            return []

        # Parse sections by week
        chunks = []
        current_section = ""
        current_tasks = []

        for line in content.split("\n"):
            if line.startswith("## "):
                if current_section and current_tasks:
                    chunks.append(self._goals_chunk(current_section, current_tasks))
                current_section = line[3:].strip()
                current_tasks = []
            elif line.startswith("- "):
                current_tasks.append(line[2:].strip())

        if current_section and current_tasks:
            chunks.append(self._goals_chunk(current_section, current_tasks))

        return chunks

    def _goals_chunk(self, section_header: str, tasks: list[str]) -> Chunk:
        text_lines = [f"Completed Goals: {section_header}", f"Tasks completed: {len(tasks)}"]
        for t in tasks[:20]:
            text_lines.append(f"  - {t}")
        # Extract date from section if possible
        date_match = re.search(r"(\d{4}-\d{2}-\d{2})", section_header)
        date_iso = f"{date_match.group(1)}T00:00:00+00:00" if date_match else ""

        return Chunk(
            text="\n".join(text_lines),
            metadata=_ensure_metadata({
                "source": self.source_name,
                "conversation_id": f"goals_{section_header[:30]}",
                "title": f"Goals completed {section_header}",
                "timestamp": date_iso,
                "msg_timestamp": date_iso,
                "role": "user",
                "type": "goals_completed",
                "pillar": "PURPOSE",
                "dimension": "goals",
                "classified": "true",
            }),
        )

    # ------------------------------------------------------------------
    # Card Counters / Rep Milestones (logs/card_counters.json)
    # ------------------------------------------------------------------
    def _fetch_card_counters(self) -> list[Chunk]:
        counters_file = LOGS_DIR / "card_counters.json"
        if not counters_file.exists():
            return []

        counters = json.loads(counters_file.read_text(encoding="utf-8"))
        if not counters:
            return []

        lines = ["Rep Milestones & Card Progress"]
        total_reps = sum(counters.values())
        lines.append(f"Total reps across all cards: {total_reps}")
        lines.append("")

        for card, count in sorted(counters.items(), key=lambda x: -x[1]):
            milestone = "none"
            for m in [500, 200, 100, 50, 10]:
                if count >= m:
                    milestone = f"passed {m}"
                    break
            next_m = 10
            for m in [10, 50, 100, 200, 500]:
                if count < m:
                    next_m = m
                    break
            lines.append(f"  {card}: {count} reps (milestone: {milestone}, next: {next_m})")

        now = datetime.now(tz=timezone.utc).isoformat()
        return [Chunk(
            text="\n".join(lines),
            metadata=_ensure_metadata({
                "source": self.source_name,
                "conversation_id": "card_counters_latest",
                "title": "Rep milestones",
                "timestamp": now,
                "msg_timestamp": now,
                "role": "user",
                "type": "card_counters",
                "pillar": "PURPOSE",
                "dimension": "progress",
                "classified": "true",
            }),
        )]

    # ------------------------------------------------------------------
    # Plan Notes (logs/plan_note.md) — 30-day coaching journey
    # ------------------------------------------------------------------
    def _fetch_plan_notes(self) -> list[Chunk]:
        plan_file = LOGS_DIR / "plan_note.md"
        if not plan_file.exists():
            return []

        content = plan_file.read_text(encoding="utf-8").strip()
        if not content:
            return []

        # Extract metadata from comment
        day_match = re.search(r"day (\d+)", content)
        date_match = re.search(r"generated (\d{4}-\d{2}-\d{2})", content)
        day_num = day_match.group(1) if day_match else "?"
        date_str = date_match.group(1) if date_match else datetime.now().strftime("%Y-%m-%d")

        # Strip HTML comment for clean text
        clean = re.sub(r"<!--.*?-->", "", content).strip()

        text = f"30-Day Plan: Day {day_num} ({date_str})\n{clean}"

        return [Chunk(
            text=text,
            metadata=_ensure_metadata({
                "source": self.source_name,
                "conversation_id": f"plan_day_{day_num}",
                "title": f"Plan Day {day_num}",
                "timestamp": f"{date_str}T00:00:00+00:00",
                "msg_timestamp": f"{date_str}T00:00:00+00:00",
                "role": "user",
                "type": "plan_note",
                "pillar": "PURPOSE",
                "dimension": "goals",
                "classified": "true",
            }),
        )]

    # ------------------------------------------------------------------
    # Week Carry (logs/week_carry.md) — last week's summary
    # ------------------------------------------------------------------
    def _fetch_week_carry(self) -> list[Chunk]:
        carry_file = LOGS_DIR / "week_carry.md"
        if not carry_file.exists():
            return []

        content = carry_file.read_text(encoding="utf-8").strip()
        if not content:
            return []

        # Extract week from header
        week_match = re.search(r"(\d{4}-W\d+)", content)
        week_str = week_match.group(1) if week_match else "unknown"

        text = f"Week Summary Carry: {week_str}\n{content}"

        return [Chunk(
            text=text,
            metadata=_ensure_metadata({
                "source": self.source_name,
                "conversation_id": f"week_carry_{week_str}",
                "title": f"Week carry {week_str}",
                "timestamp": self._week_to_iso(week_str) if week_str != "unknown" else "",
                "msg_timestamp": self._week_to_iso(week_str) if week_str != "unknown" else "",
                "role": "user",
                "type": "week_carry",
                "pillar": "PURPOSE",
                "dimension": "progress",
                "classified": "true",
            }),
        )]

    # ------------------------------------------------------------------
    # Pillar Journals (logs/voice.md, career.md, body.md, etc.)
    # ------------------------------------------------------------------
    def _fetch_pillar_journals(self, since: float | None) -> list[Chunk]:
        """Import GEN entries from pillar journal markdown files."""
        pillar_files = {
            "voice": LOGS_DIR / "voice.md",
            "career": LOGS_DIR / "career.md",
            "body": LOGS_DIR / "body.md",
            "create": LOGS_DIR / "create.md",
            "soul": LOGS_DIR / "soul.md",
        }
        pillar_to_dim = {
            "voice": ("MIND", "professional"),
            "career": ("MIND", "professional"),
            "body": ("BODY", "wellness"),
            "create": ("SOUL", "creative"),
            "soul": ("SOUL", "vibe"),
        }

        chunks = []
        seen_gen_ids = set()

        for pillar_name, filepath in pillar_files.items():
            if not filepath.exists():
                continue

            content = filepath.read_text(encoding="utf-8")
            # Split by GEN entries (## GEN_XXXX)
            entries = re.split(r"(?=## GEN_\d+)", content)

            mapped_pillar, dimension = pillar_to_dim.get(pillar_name, ("", ""))

            for entry in entries:
                entry = entry.strip()
                if not entry.startswith("## GEN_"):
                    continue

                # Extract GEN ID
                gen_match = re.match(r"## (GEN_\d+)", entry)
                if not gen_match:
                    continue
                gen_id = gen_match.group(1)

                # Deduplicate across pillar files
                if gen_id in seen_gen_ids:
                    continue
                seen_gen_ids.add(gen_id)

                # Extract score if present
                score_match = re.search(r"val[=:]\s*([\d.]+)", entry)
                score = score_match.group(1) if score_match else ""

                text = f"[{pillar_name}] {entry[:800]}"

                chunks.append(Chunk(
                    text=text,
                    metadata=_ensure_metadata({
                        "source": self.source_name,
                        "conversation_id": f"journal_{gen_id}",
                        "title": f"{pillar_name} journal {gen_id}",
                        "timestamp": "",
                        "msg_timestamp": "",
                        "role": "user",
                        "type": "pillar_journal",
                        "pillar": mapped_pillar,
                        "dimension": dimension,
                        "classified": "true" if dimension else "false",
                    }),
                ))

        return chunks

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _week_to_iso(week_str: str) -> str:
        """Convert 2026-W10 to an ISO date string (Monday of that week)."""
        try:
            m = re.match(r"(\d{4})-W(\d+)", week_str)
            if m:
                year, week = int(m.group(1)), int(m.group(2))
                dt = datetime.strptime(f"{year}-W{week:02d}-1", "%Y-W%W-%w")
                return dt.strftime("%Y-%m-%dT00:00:00+00:00")
        except Exception:
            pass
        return ""
