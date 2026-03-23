"""Direct SQL access to Singularity's structured metric databases.

Reads from weekly_stats.db and singularity.db — never writes.
All queries are parameterized; no LLM-generated SQL.
"""

import json
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime, timedelta

from config import SINGULARITY_DIR

WEEKLY_DB = SINGULARITY_DIR / "logs" / "weekly_stats.db"
SINGULARITY_DB = SINGULARITY_DIR / "logs" / "singularity.db"


@dataclass
class MetricResult:
    table: str
    description: str
    data: list[dict] = field(default_factory=list)
    summary: str = ""
    time_range: str = ""


def _current_week_id() -> str:
    """ISO week ID for the current week, e.g. '2026-W13'."""
    now = datetime.now()
    return f"{now.isocalendar()[0]}-W{now.isocalendar()[1]:02d}"


def _previous_week_id(offset: int = 1) -> str:
    d = datetime.now() - timedelta(weeks=offset)
    return f"{d.isocalendar()[0]}-W{d.isocalendar()[1]:02d}"


def _week_ids_range(weeks: int) -> list[str]:
    """Return last N week IDs including current."""
    return [_previous_week_id(i) for i in range(weeks - 1, -1, -1)]


def _query(db_path, sql: str, params: tuple = ()) -> list[dict]:
    """Run a read-only query and return rows as dicts."""
    try:
        conn = sqlite3.connect(str(db_path))
        conn.row_factory = sqlite3.Row
        rows = conn.execute(sql, params).fetchall()
        conn.close()
        return [dict(r) for r in rows]
    except (sqlite3.OperationalError, sqlite3.DatabaseError):
        return []


class MetricStore:
    """Query Singularity's SQL databases for structured metrics."""

    # ── Gym ──────────────────────────────────────────────────────

    def gym_this_week(self, week_id: str | None = None) -> MetricResult:
        wk = week_id or _current_week_id()
        rows = _query(WEEKLY_DB,
            "SELECT week_day, workout_type, intensity, notes FROM gym WHERE week_id = ? ORDER BY id",
            (wk,))
        count = len(rows)
        types = [r["workout_type"] for r in rows]
        days = [r["week_day"] for r in rows]
        summary = f"Week {wk}: {count} workout(s) on {', '.join(days)}."
        if types:
            summary += f" Types: {', '.join(types)}."
        return MetricResult(table="gym", description="Gym sessions this week",
                            data=rows, summary=summary, time_range=wk)

    def gym_streak(self, weeks: int = 8) -> MetricResult:
        wk_ids = _week_ids_range(weeks)
        rows = _query(WEEKLY_DB,
            f"SELECT week_id, COUNT(*) as sessions FROM gym "
            f"WHERE week_id IN ({','.join('?' * len(wk_ids))}) "
            f"GROUP BY week_id ORDER BY week_id",
            tuple(wk_ids))
        week_map = {r["week_id"]: r["sessions"] for r in rows}
        streak_lines = [f"  {wk}: {week_map.get(wk, 0)} sessions" for wk in wk_ids]
        summary = f"Gym sessions (last {weeks} weeks):\n" + "\n".join(streak_lines)
        return MetricResult(table="gym", description="Gym streak",
                            data=rows, summary=summary, time_range=f"last {weeks} weeks")

    # ── Nutrition ────────────────────────────────────────────────

    def nutrition_this_week(self, week_id: str | None = None) -> MetricResult:
        wk = week_id or _current_week_id()
        rows = _query(WEEKLY_DB,
            "SELECT week_day, meal_source, calorie_deficit, notes FROM nutrition WHERE week_id = ? ORDER BY id",
            (wk,))
        home = sum(1 for r in rows if r["meal_source"] == "home")
        deficit_on = sum(1 for r in rows if r["calorie_deficit"] == "on")
        summary = f"Week {wk}: {len(rows)} days logged. {home} home-cooked, {deficit_on} on calorie deficit."
        # Parse JSON notes for details
        for r in rows:
            if r["notes"]:
                try:
                    r["parsed_notes"] = json.loads(r["notes"])
                except (json.JSONDecodeError, TypeError):
                    pass
        return MetricResult(table="nutrition", description="Nutrition this week",
                            data=rows, summary=summary, time_range=wk)

    # ── Communications ───────────────────────────────────────────

    def comms_this_week(self, week_id: str | None = None) -> MetricResult:
        wk = week_id or _current_week_id()
        rows = _query(WEEKLY_DB,
            "SELECT week_day, manager, co_worker, skip_level_manager, "
            "non_tech_co_worker, networking_event, new_person_at_networking, notes "
            "FROM communications WHERE week_id = ? ORDER BY rowid",
            (wk,))
        total_interactions = sum(
            r["manager"] + r["co_worker"] + r["skip_level_manager"]
            + r["non_tech_co_worker"] + r["networking_event"] + r["new_person_at_networking"]
            for r in rows
        )
        days_logged = len(rows)
        summary = f"Week {wk}: {total_interactions} total interactions across {days_logged} days."
        # Break down by type
        if rows:
            mgr = sum(r["manager"] for r in rows)
            cw = sum(r["co_worker"] for r in rows)
            skip = sum(r["skip_level_manager"] for r in rows)
            nontech = sum(r["non_tech_co_worker"] for r in rows)
            net_event = sum(r["networking_event"] for r in rows)
            new_ppl = sum(r["new_person_at_networking"] for r in rows)
            summary += (f"\n  Manager: {mgr}, Co-workers: {cw}, Skip-level: {skip}, "
                        f"Non-tech: {nontech}, Networking events: {net_event}, New people: {new_ppl}.")
        return MetricResult(table="communications", description="Communications this week",
                            data=rows, summary=summary, time_range=wk)

    def comms_trend(self, weeks: int = 4) -> MetricResult:
        wk_ids = _week_ids_range(weeks)
        rows = _query(WEEKLY_DB,
            f"SELECT week_id, "
            f"SUM(manager + co_worker + skip_level_manager + non_tech_co_worker "
            f"    + networking_event + new_person_at_networking) as total "
            f"FROM communications WHERE week_id IN ({','.join('?' * len(wk_ids))}) "
            f"GROUP BY week_id ORDER BY week_id",
            tuple(wk_ids))
        lines = [f"  {r['week_id']}: {r['total']} interactions" for r in rows]
        summary = f"Communication trend (last {weeks} weeks):\n" + "\n".join(lines) if lines else "No communication data."
        return MetricResult(table="communications", description="Communications trend",
                            data=rows, summary=summary, time_range=f"last {weeks} weeks")

    # ── Tasks ────────────────────────────────────────────────────

    def tasks_this_week(self, week_id: str | None = None) -> MetricResult:
        wk = week_id or _current_week_id()
        rows = _query(WEEKLY_DB,
            "SELECT * FROM tasks WHERE week_id = ?", (wk,))
        if rows:
            r = rows[0]
            summary = (f"Week {wk}: {r['total_entries']} entries, "
                       f"{r['high_signal_entries']} high-signal (avg {r['avg_score']:.2f}). "
                       f"Completed tasks: {r['completed_tasks']}.\n"
                       f"  By pillar — Voice: {r['voice_entries']}, Career: {r['career_entries']}, "
                       f"Body: {r['body_entries']}, Create: {r['create_entries']}, Soul: {r['soul_entries']}.")
        else:
            summary = f"Week {wk}: No task data yet."
        return MetricResult(table="tasks", description="Tasks this week",
                            data=rows, summary=summary, time_range=wk)

    def tasks_completion_trend(self, weeks: int = 4) -> MetricResult:
        wk_ids = _week_ids_range(weeks)
        rows = _query(WEEKLY_DB,
            f"SELECT week_id, total_entries, high_signal_entries, completed_tasks, avg_score "
            f"FROM tasks WHERE week_id IN ({','.join('?' * len(wk_ids))}) ORDER BY week_id",
            tuple(wk_ids))
        lines = [f"  {r['week_id']}: {r['completed_tasks']} completed, {r['total_entries']} entries (avg {r['avg_score']:.2f})" for r in rows]
        summary = f"Task trend (last {weeks} weeks):\n" + "\n".join(lines) if lines else "No task data."
        return MetricResult(table="tasks", description="Task completion trend",
                            data=rows, summary=summary, time_range=f"last {weeks} weeks")

    # ── Wellness Habits ──────────────────────────────────────────

    def wellness_this_week(self, week_id: str | None = None) -> MetricResult:
        wk = week_id or _current_week_id()
        rows = _query(WEEKLY_DB,
            "SELECT task_id, days_done, notes FROM wellness WHERE week_id = ? ORDER BY task_id",
            (wk,))
        lines = [f"  {r['task_id']}: {r['days_done']}/7 days" for r in rows]
        summary = f"Week {wk} wellness habits:\n" + "\n".join(lines) if lines else f"Week {wk}: No wellness data."
        return MetricResult(table="wellness", description="Wellness habits this week",
                            data=rows, summary=summary, time_range=wk)

    # ── Browser ──────────────────────────────────────────────────

    def browser_this_week(self, week_id: str | None = None) -> MetricResult:
        wk = week_id or _current_week_id()
        rows = _query(WEEKLY_DB,
            "SELECT category, SUM(visit_count) as visits FROM browser "
            "WHERE week_id = ? GROUP BY category ORDER BY visits DESC",
            (wk,))
        total = sum(r["visits"] for r in rows)
        lines = [f"  {r['category']}: {r['visits']} visits" for r in rows]
        summary = f"Week {wk} browsing ({total} total visits):\n" + "\n".join(lines) if lines else f"Week {wk}: No browser data."
        return MetricResult(table="browser", description="Browser activity this week",
                            data=rows, summary=summary, time_range=wk)

    # ── Weekly Overview ──────────────────────────────────────────

    def weekly_summary(self, week_id: str | None = None) -> MetricResult:
        wk = week_id or _current_week_id()
        dim = _query(WEEKLY_DB, "SELECT * FROM week_dim WHERE week_id = ?", (wk,))
        review = _query(WEEKLY_DB, "SELECT notes, raw_summary FROM weekly_review WHERE week_id = ?", (wk,))

        parts = [f"Week {wk} Overview:"]
        if dim:
            d = dim[0]
            parts.append(f"  Notes logged: {d['notes_logged']}, High-signal: {d['high_signal_count']}, "
                         f"Avg signal: {d['avg_signal']:.2f}, Most active: {d['most_active_pillar']}")
        if review:
            r = review[0]
            if r.get("notes"):
                try:
                    notes = json.loads(r["notes"])
                    if notes.get("weekly_wins"):
                        parts.append("  Wins: " + "; ".join(notes["weekly_wins"][:5]))
                    if notes.get("feedback"):
                        parts.append("  Feedback: " + "; ".join(notes["feedback"][:3]))
                except (json.JSONDecodeError, TypeError):
                    pass

        summary = "\n".join(parts) if len(parts) > 1 else f"Week {wk}: No summary data."
        return MetricResult(table="week_dim", description="Weekly overview",
                            data=dim + review, summary=summary, time_range=wk)

    # ── Singularity Entries ──────────────────────────────────────

    def entries_by_pillar(self, pillar: str, days: int = 7) -> MetricResult:
        since = (datetime.now() - timedelta(days=days)).isoformat()
        rows = _query(SINGULARITY_DB,
            "SELECT gen_id, pillar, score, label, title, snippet, tags, logged_at "
            "FROM entries WHERE pillar = ? AND logged_at >= ? ORDER BY logged_at DESC",
            (pillar, since))
        summary = f"{len(rows)} {pillar} entries in last {days} days."
        if rows:
            avg_score = sum(r["score"] for r in rows) / len(rows)
            summary += f" Avg score: {avg_score:.2f}."
        return MetricResult(table="entries", description=f"{pillar} entries",
                            data=rows, summary=summary, time_range=f"last {days} days")

    def daily_activity(self, days: int = 7) -> MetricResult:
        since = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
        rows = _query(SINGULARITY_DB,
            "SELECT date, pillar, entry_count, avg_score, high_count "
            "FROM daily_stats WHERE date >= ? ORDER BY date DESC, pillar",
            (since,))
        summary = f"Daily activity (last {days} days): {len(rows)} pillar-day records."
        return MetricResult(table="daily_stats", description="Daily activity",
                            data=rows, summary=summary, time_range=f"last {days} days")

    # ── Notes Index ──────────────────────────────────────────────

    def notes_by_category(self, category: str | None = None, limit: int = 20) -> MetricResult:
        if category:
            rows = _query(WEEKLY_DB,
                "SELECT title, category, sub_category, topic_tags, semantic_summary, modified_at "
                "FROM notes_index WHERE category = ? ORDER BY modified_at DESC LIMIT ?",
                (category, limit))
            summary = f"{len(rows)} notes in category '{category}'."
        else:
            rows = _query(WEEKLY_DB,
                "SELECT category, COUNT(*) as count FROM notes_index GROUP BY category ORDER BY count DESC", ())
            lines = [f"  {r['category']}: {r['count']} notes" for r in rows]
            summary = "Notes by category:\n" + "\n".join(lines) if lines else "No indexed notes."
        return MetricResult(table="notes_index", description="Notes index",
                            data=rows, summary=summary, time_range="all")

    # ── Intent Dispatcher ────────────────────────────────────────

    def query_from_intent(self, intent: str, tables: list[str], time_range: str) -> list[MetricResult]:
        """Dispatch to the right query methods based on router output."""
        results = []
        week_id = self._resolve_time_range(time_range)

        for table in tables:
            if table == "gym":
                if "streak" in intent or "trend" in intent:
                    results.append(self.gym_streak())
                else:
                    results.append(self.gym_this_week(week_id))
            elif table == "nutrition":
                results.append(self.nutrition_this_week(week_id))
            elif table == "communications":
                if "trend" in intent:
                    results.append(self.comms_trend())
                else:
                    results.append(self.comms_this_week(week_id))
            elif table == "tasks":
                if "trend" in intent or "completion" in intent:
                    results.append(self.tasks_completion_trend())
                else:
                    results.append(self.tasks_this_week(week_id))
            elif table == "wellness":
                results.append(self.wellness_this_week(week_id))
            elif table == "browser":
                results.append(self.browser_this_week(week_id))
            elif table == "weekly_summary":
                results.append(self.weekly_summary(week_id))
            elif table == "entries":
                # Extract pillar from intent if possible
                for p in ["voice", "career", "body", "create", "soul"]:
                    if p in intent.lower():
                        results.append(self.entries_by_pillar(p))
                        break
                else:
                    results.append(self.daily_activity())
            elif table == "notes_index":
                results.append(self.notes_by_category())

        return results if results else [MetricResult(
            table="none", description="No matching metrics",
            summary="No structured data found for this query.")]

    def _resolve_time_range(self, time_range: str) -> str | None:
        """Convert time_range string to a week_id. Returns None for current week."""
        if not time_range or time_range in ("this_week", "current"):
            return None  # methods default to current week
        if time_range == "last_week":
            return _previous_week_id(1)
        if time_range.startswith("20") and "-W" in time_range:
            return time_range  # already a week_id
        return None
