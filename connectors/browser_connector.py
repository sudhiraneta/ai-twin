import sys
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from urllib.parse import urlparse

from config import SINGULARITY_AGENT_DIR
from memory.chunker import Chunk, _ensure_metadata
from persona.classifier import BROWSER_CATEGORY_MAP
from persona.dimensions import DIMENSIONS
from .base import BaseConnector


class BrowserConnector(BaseConnector):
    """Connector for Chrome/Safari history via Singularity's browser_reader."""

    source_name = "browser"

    def _import_modules(self):
        if str(SINGULARITY_AGENT_DIR) not in sys.path:
            sys.path.insert(0, str(SINGULARITY_AGENT_DIR))
        import browser_reader
        import site_classifier
        return browser_reader, site_classifier

    def fetch(self, since: float | None = None, days_back: int = 30) -> list[Chunk]:
        reader, classifier = self._import_modules()

        if since:
            start_date = datetime.fromtimestamp(since)
        else:
            start_date = datetime.now() - timedelta(days=days_back)

        chunks = []
        current = start_date.date()
        today = datetime.now().date()

        while current <= today:
            date_str = current.strftime("%Y-%m-%d")
            try:
                visits = reader.get_date_history(date_str)
            except Exception:
                visits = []

            if visits:
                daily_chunks = self._build_daily_chunks(
                    visits, date_str, current, classifier
                )
                chunks.extend(daily_chunks)

            current += timedelta(days=1)

        return chunks

    def _build_daily_chunks(
        self, visits: list, date_str: str, date_obj, classifier
    ) -> list[Chunk]:
        chunks = []

        # Categorize visits
        by_category = defaultdict(int)
        by_domain = defaultdict(list)
        for v in visits:
            cat = classifier.classify_url(v["url"])
            by_category[cat] += v.get("visit_count", 1)
            domain = self._extract_domain(v["url"])
            by_domain[domain].append(v)

        total = sum(by_category.values())
        productive = sum(
            by_category.get(c, 0) for c in ["work", "learning", "google_tools"]
        )
        focus_score = round((productive / total) * 100) if total > 0 else 0

        # Determine dominant dimension from categories
        dominant_cat = max(by_category, key=by_category.get) if by_category else "other"
        dimension = BROWSER_CATEGORY_MAP.get(dominant_cat, "learning")
        pillar = DIMENSIONS.get(dimension, {}).get("pillar", "MIND") if dimension else "MIND"
        if not dimension:
            dimension = "learning"
            pillar = "MIND"

        # Top domains by visit count
        domain_counts = sorted(
            [(d, sum(v.get("visit_count", 1) for v in vlist)) for d, vlist in by_domain.items()],
            key=lambda x: -x[1],
        )[:10]
        top_str = ", ".join(f"{d} ({c})" for d, c in domain_counts)

        # Notable visits
        notable = []
        seen_titles = set()
        for v in sorted(visits, key=lambda x: -x.get("visit_count", 1))[:15]:
            title = v.get("title", "").strip()
            if title and title not in seen_titles:
                seen_titles.add(title)
                notable.append(f"- {title}")
                if len(notable) >= 8:
                    break

        # Category breakdown
        cat_parts = []
        for cat in ["work", "learning", "google_tools", "gmail", "linkedin",
                     "news", "entertainment", "social", "shopping", "lifestyle", "other"]:
            if by_category.get(cat, 0) > 0:
                cat_parts.append(f"{cat} ({by_category[cat]})")
        cat_str = ", ".join(cat_parts)

        day_name = date_obj.strftime("%A")
        text_lines = [
            f"Browser Activity: {date_str} ({day_name})",
            f"Visited {len(visits)} unique URLs. Focus score: {focus_score}%",
            f"Categories: {cat_str}",
            f"Top sites: {top_str}",
        ]
        if notable:
            text_lines.append("Notable visits:")
            text_lines.extend(notable)

        text = "\n".join(text_lines)

        chunks.append(Chunk(
            text=text,
            metadata=_ensure_metadata({
                "source": self.source_name,
                "conversation_id": f"browser_{date_str}",
                "title": f"Browser activity {date_str}",
                "timestamp": f"{date_str}T00:00:00+00:00",
                "msg_timestamp": f"{date_str}T00:00:00+00:00",
                "role": "user",
                "type": "browser_daily",
                "pillar": pillar,
                "dimension": dimension,
                "classified": "true",
                "focus_score": str(focus_score),
                "total_visits": str(len(visits)),
            }),
        ))

        # Domain-level chunks for heavily visited domains (5+ visits)
        for domain, vlist in by_domain.items():
            total_visits = sum(v.get("visit_count", 1) for v in vlist)
            if total_visits >= 5:
                titles = []
                for v in vlist:
                    t = v.get("title", "").strip()
                    if t:
                        titles.append(f"- {t}")
                domain_text = f"Browser: {domain} visits on {date_str}\n{total_visits} page visits:\n" + "\n".join(
                    titles[:15]
                )
                chunks.append(Chunk(
                    text=domain_text,
                    metadata=_ensure_metadata({
                        "source": self.source_name,
                        "conversation_id": f"browser_{date_str}_{domain}",
                        "title": f"{domain} on {date_str}",
                        "timestamp": f"{date_str}T00:00:00+00:00",
                        "msg_timestamp": f"{date_str}T00:00:00+00:00",
                        "role": "user",
                        "type": "browser_domain",
                        "pillar": pillar,
                        "dimension": dimension,
                        "classified": "true",
                        "total_visits": str(total_visits),
                    }),
                ))

        return chunks

    @staticmethod
    def _extract_domain(url: str) -> str:
        try:
            domain = urlparse(url).netloc.lower()
            if domain.startswith("www."):
                domain = domain[4:]
            return domain
        except Exception:
            return ""
