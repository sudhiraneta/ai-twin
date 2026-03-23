import sqlite3
from datetime import datetime, timezone

from config import SINGULARITY_DIR
from memory.chunker import Chunk, _ensure_metadata
from .base import BaseConnector

WEEKLY_STATS_DB = SINGULARITY_DIR / "logs" / "weekly_stats.db"


class BodyConnector(BaseConnector):
    """Connector for gym, nutrition, and wellness data from Singularity's weekly_stats.db."""

    source_name = "body"

    def fetch(self, since: float | None = None, days_back: int = 30) -> list[Chunk]:
        if not WEEKLY_STATS_DB.exists():
            print(f"  weekly_stats.db not found at {WEEKLY_STATS_DB}")
            return []

        conn = sqlite3.connect(str(WEEKLY_STATS_DB))
        conn.row_factory = sqlite3.Row
        chunks = []

        # --- Gym sessions ---
        if since:
            since_str = datetime.fromtimestamp(since, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%S")
            gym_rows = conn.execute(
                "SELECT * FROM gym WHERE logged_at > ? ORDER BY logged_at DESC", (since_str,)
            ).fetchall()
        else:
            gym_rows = conn.execute(
                "SELECT * FROM gym ORDER BY logged_at DESC LIMIT 50"
            ).fetchall()

        # Group gym sessions by week
        gym_by_week = {}
        for row in gym_rows:
            week = row["week_id"]
            gym_by_week.setdefault(week, []).append(dict(row))

        for week_id, sessions in gym_by_week.items():
            days = [s["week_day"] for s in sessions]
            count = len(sessions)
            notes = [s["notes"] for s in sessions if s.get("notes")]

            text_lines = [
                f"Gym Tracker: {week_id}",
                f"Sessions: {count}/4 target",
                f"Days: {', '.join(days)}",
            ]
            if count >= 4:
                text_lines.append("Hit 4x gym target this week!")
            else:
                text_lines.append(f"{4 - count} sessions left to hit 4x.")
            if notes:
                text_lines.append(f"Notes: {'; '.join(notes[:3])}")

            logged_at = sessions[0].get("logged_at", "")

            chunks.append(Chunk(
                text="\n".join(text_lines),
                metadata=_ensure_metadata({
                    "source": self.source_name,
                    "conversation_id": f"gym_{week_id}",
                    "title": f"Gym {week_id}",
                    "timestamp": logged_at,
                    "msg_timestamp": logged_at,
                    "role": "user",
                    "type": "body_gym",
                    "pillar": "BODY",
                    "dimension": "wellness",
                    "classified": "true",
                }),
            ))

        # --- Nutrition ---
        if since:
            nut_rows = conn.execute(
                "SELECT * FROM nutrition WHERE logged_at > ? ORDER BY logged_at DESC", (since_str,)
            ).fetchall()
        else:
            nut_rows = conn.execute(
                "SELECT * FROM nutrition ORDER BY logged_at DESC LIMIT 20"
            ).fetchall()

        nut_by_week = {}
        for row in nut_rows:
            week = row["week_id"]
            nut_by_week.setdefault(week, []).append(dict(row))

        for week_id, entries in nut_by_week.items():
            text_lines = [f"Nutrition: {week_id}"]
            for entry in entries:
                day = entry.get("week_day", "?")
                source = entry.get("meal_source", "?")
                deficit = entry.get("calorie_deficit", "?")
                notes_raw = entry.get("notes", "")

                text_lines.append(f"  {day}: {source} meal, deficit={deficit}")

                # Parse JSON notes if present
                if notes_raw and notes_raw.startswith("{"):
                    try:
                        import json
                        data = json.loads(notes_raw)
                        for key in ["veggies", "protein", "carbs", "cheats"]:
                            items = data.get(key, [])
                            if items:
                                text_lines.append(f"    {key.title()}: {', '.join(items)}")
                    except Exception:
                        text_lines.append(f"    {notes_raw[:100]}")

            logged_at = entries[0].get("logged_at", "")

            chunks.append(Chunk(
                text="\n".join(text_lines),
                metadata=_ensure_metadata({
                    "source": self.source_name,
                    "conversation_id": f"nutrition_{week_id}",
                    "title": f"Nutrition {week_id}",
                    "timestamp": logged_at,
                    "msg_timestamp": logged_at,
                    "role": "user",
                    "type": "body_nutrition",
                    "pillar": "BODY",
                    "dimension": "nutrition",
                    "classified": "true",
                }),
            ))

        # --- Wellness (journaling, sleep, phone off, etc.) ---
        if since:
            well_rows = conn.execute(
                "SELECT * FROM wellness WHERE logged_at > ? ORDER BY logged_at DESC", (since_str,)
            ).fetchall()
        else:
            well_rows = conn.execute(
                "SELECT * FROM wellness ORDER BY logged_at DESC LIMIT 30"
            ).fetchall()

        well_by_week = {}
        for row in well_rows:
            week = row["week_id"]
            well_by_week.setdefault(week, []).append(dict(row))

        for week_id, entries in well_by_week.items():
            text_lines = [f"Wellness Habits: {week_id}"]
            for entry in entries:
                task = entry.get("task_id", "?")
                days_done = entry.get("days_done", 0)
                notes = entry.get("notes", "")
                text_lines.append(f"  {task}: {days_done} days ({notes})")

            logged_at = entries[0].get("logged_at", "")

            chunks.append(Chunk(
                text="\n".join(text_lines),
                metadata=_ensure_metadata({
                    "source": self.source_name,
                    "conversation_id": f"wellness_{week_id}",
                    "title": f"Wellness {week_id}",
                    "timestamp": logged_at,
                    "msg_timestamp": logged_at,
                    "role": "user",
                    "type": "body_wellness",
                    "pillar": "BODY",
                    "dimension": "wellness",
                    "classified": "true",
                }),
            ))

        conn.close()
        return chunks
