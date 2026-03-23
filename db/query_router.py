"""Two-tier query router: classifies questions as SQL, RAG, or HYBRID.

Tier 1: Rule-based regex patterns (instant, zero cost)
Tier 2: LLM fallback for ambiguous queries
"""

import re
from dataclasses import dataclass, field
from enum import Enum


class QueryType(Enum):
    SQL = "sql"
    RAG = "rag"
    HYBRID = "hybrid"


@dataclass
class RoutedQuery:
    query_type: QueryType
    sql_tables: list[str] = field(default_factory=list)
    sql_intent: str = ""
    rag_query: str = ""
    time_range: str = ""


# ── Tier 1 Patterns ─────────────────────────────────────────────

# SQL table triggers: (compiled_regex, tables_to_query)
_SQL_TABLE_PATTERNS: list[tuple[re.Pattern, list[str]]] = [
    (re.compile(r"\bgym\b|\bworkout\b|\bexercis\w*\b|\bsession\w?\b.*\bgym\b", re.I), ["gym"]),
    (re.compile(r"\bnutrition\b|\bmeal\b|\bfood\b|\bdiet\b|\bcalori\w*\b|\bprotein\b|\bvegg\w*\b", re.I), ["nutrition"]),
    (re.compile(r"\bcommunicat\w*\b|\bnetwork\w*\b|\bmanager\b|\bco.?worker\b|\binteraction\b|\bskip.?level\b", re.I), ["communications"]),
    (re.compile(r"\btask\w?\b|\bcomplete\w*\b|\bcheckbox\w*\b|\btodo\b", re.I), ["tasks"]),
    (re.compile(r"\bwellness\b|\bmeditat\w*\b|\bjournal\w*\b.*\bhabit\b|\bmorning.?routine\b|\bsleep\b.*\bquality\b", re.I), ["wellness"]),
    (re.compile(r"\bbrows\w*\b|\bsite\w?\b|\bvisit\w*\b|\bfocus.?score\b|\bscreen.?time\b", re.I), ["browser"]),
    (re.compile(r"\bweekly.?summary\b|\bweek.?overview\b|\bhow.?was.?my.?week\b|\bweekly.?review\b", re.I), ["weekly_summary"]),
    (re.compile(r"\bnote\w?.?index\b|\bnote\w?.?categor\w*\b|\bapple.?note\w?\b.*\bcategor\w*\b", re.I), ["notes_index"]),
    (re.compile(r"\bentr(?:y|ies)\b.*\bpillar\b|\bpillar\b.*\bentr(?:y|ies)\b|\bdaily.?stats\b", re.I), ["entries"]),
]

# Quantitative signal words — if present with table match, route to SQL
_QUANT_PATTERN = re.compile(
    r"\bhow many\b|\bhow often\b|\bcount\b|\btotal\b|\baverage\b"
    r"|\bstreak\b|\btrend\b|\bscore\b|\brate\b|\bpercentage\b"
    r"|\bnumber of\b|\bstats?\b|\bmetric\w*\b|\bdata\b",
    re.I,
)

# Pure RAG triggers — thought/reflection questions
_RAG_PATTERN = re.compile(
    r"\bthink about\b|\bfeel about\b|\bthoughts on\b|\bopinion on\b"
    r"|\bmindset\b|\breflect\w*\b|\bjournal\b(?!.*\bhabit\b)"
    r"|\bwhat have I been\b|\bmy approach to\b|\bhow do I feel\b"
    r"|\bwhat do I\b.*\babout\b|\bbeliev\w*\b|\bvalue\w?\b",
    re.I,
)

# Time range extraction
_TIME_PATTERNS: list[tuple[re.Pattern, str]] = [
    (re.compile(r"\bthis week\b|\bcurrent week\b", re.I), "this_week"),
    (re.compile(r"\blast week\b|\bprevious week\b", re.I), "last_week"),
    (re.compile(r"\blast month\b|\bthis month\b|\blast 30 days\b", re.I), "last_30_days"),
    (re.compile(r"\btoday\b", re.I), "today"),
    (re.compile(r"\byesterday\b", re.I), "yesterday"),
]


class QueryRouter:
    """Classifies questions into SQL, RAG, or hybrid retrieval paths."""

    def route(self, question: str) -> RoutedQuery:
        """Route a question. Tries rules first, falls back to LLM."""
        result = self._rule_based_route(question)
        if result:
            return result
        return self._llm_route(question)

    def _rule_based_route(self, question: str) -> RoutedQuery | None:
        """Tier 1: regex pattern matching."""
        # Extract time range
        time_range = ""
        for pattern, tr in _TIME_PATTERNS:
            if pattern.search(question):
                time_range = tr
                break

        # Check for table matches
        matched_tables = []
        for pattern, tables in _SQL_TABLE_PATTERNS:
            if pattern.search(question):
                matched_tables.extend(tables)

        # Deduplicate
        matched_tables = list(dict.fromkeys(matched_tables))

        has_quant = bool(_QUANT_PATTERN.search(question))
        has_rag = bool(_RAG_PATTERN.search(question))

        # Decision logic
        if matched_tables and has_quant and not has_rag:
            # Clear quantitative question about specific tables
            return RoutedQuery(
                query_type=QueryType.SQL,
                sql_tables=matched_tables,
                sql_intent=question,
                rag_query=question,
                time_range=time_range or "this_week",
            )
        elif matched_tables and has_rag:
            # Both signals — hybrid
            return RoutedQuery(
                query_type=QueryType.HYBRID,
                sql_tables=matched_tables,
                sql_intent=question,
                rag_query=question,
                time_range=time_range or "this_week",
            )
        elif has_rag and not matched_tables:
            # Pure thought/reflection question
            return RoutedQuery(
                query_type=QueryType.RAG,
                rag_query=question,
            )
        elif matched_tables:
            # Table match but no clear quant/rag signal — default to hybrid
            return RoutedQuery(
                query_type=QueryType.HYBRID,
                sql_tables=matched_tables,
                sql_intent=question,
                rag_query=question,
                time_range=time_range or "this_week",
            )

        # No match — fall through to Tier 2
        return None

    def _llm_route(self, question: str) -> RoutedQuery:
        """Tier 2: LLM-based routing for ambiguous questions.
        Falls back to RAG if LLM routing fails."""
        try:
            from twin.llm_client import chat_completion
            import json as _json

            system = (
                "You are a query router. Classify the user's question into one of:\n"
                "- SQL: needs exact numbers from databases (gym, nutrition, communications, tasks, wellness, browser)\n"
                "- RAG: needs journal entries, thoughts, opinions, memories\n"
                "- HYBRID: needs both metrics AND context\n\n"
                "Available SQL tables: gym, nutrition, communications, tasks, wellness, browser, weekly_summary, entries, notes_index\n\n"
                'Respond ONLY with JSON: {"type": "sql"|"rag"|"hybrid", "tables": [...], "time_range": "this_week"|"last_week"|""}'
            )
            raw = chat_completion(system=system, messages=[{"role": "user", "content": question}], max_tokens=150)

            # Parse JSON from response
            # Strip markdown code fences if present
            cleaned = raw.strip().strip("`").strip()
            if cleaned.startswith("json"):
                cleaned = cleaned[4:].strip()
            parsed = _json.loads(cleaned)

            qtype = {"sql": QueryType.SQL, "rag": QueryType.RAG, "hybrid": QueryType.HYBRID}.get(
                parsed.get("type", "rag"), QueryType.RAG
            )
            return RoutedQuery(
                query_type=qtype,
                sql_tables=parsed.get("tables", []),
                sql_intent=question,
                rag_query=question,
                time_range=parsed.get("time_range", ""),
            )
        except Exception:
            # On any failure, default to RAG
            return RoutedQuery(query_type=QueryType.RAG, rag_query=question)
