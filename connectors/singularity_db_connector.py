import sqlite3
from datetime import datetime, timezone

from config import SINGULARITY_DIR
from memory.chunker import Chunk, _ensure_metadata
from persona.classifier import PILLAR_DEFAULT_DIMENSION
from .base import BaseConnector

DB_PATH = SINGULARITY_DIR / "logs" / "singularity.db"


class SingularityDBConnector(BaseConnector):
    """Connector for Singularity's classified GEN entries."""

    source_name = "singularity_db"

    def fetch(self, since: float | None = None) -> list[Chunk]:
        if not DB_PATH.exists():
            print(f"  Singularity DB not found at {DB_PATH}")
            return []

        conn = sqlite3.connect(str(DB_PATH))
        conn.row_factory = sqlite3.Row

        if since:
            since_str = datetime.fromtimestamp(since, tz=timezone.utc).strftime(
                "%Y-%m-%d %H:%M:%S"
            )
            rows = conn.execute(
                "SELECT * FROM entries WHERE logged_at > ? ORDER BY logged_at DESC",
                (since_str,),
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT * FROM entries ORDER BY logged_at DESC"
            ).fetchall()

        conn.close()

        chunks = []
        for row in rows:
            row = dict(row)
            title = row.get("title", "")
            pillar = row.get("pillar", "")
            score = row.get("score", 0)
            label = row.get("label", "")
            snippet = row.get("snippet", "")
            tags = row.get("tags", "")
            celebration = row.get("celebration", "")
            gen_id = row.get("gen_id", "")
            logged_at = row.get("logged_at", "")

            text_parts = [f"[{pillar}] {title}"]
            text_parts.append(f"Score: {score} | Label: {label}")
            if tags:
                text_parts.append(f"Tags: {tags}")
            if snippet:
                text_parts.append(f"\n{snippet}")
            if celebration:
                text_parts.append(f"\n{celebration}")
            text = "\n".join(text_parts)

            # Map pillar to dimension
            dimension = PILLAR_DEFAULT_DIMENSION.get(pillar, "")

            metadata = _ensure_metadata({
                "source": self.source_name,
                "conversation_id": gen_id,
                "title": title,
                "timestamp": logged_at,
                "msg_timestamp": row.get("note_date", logged_at),
                "role": "user",
                "type": "singularity_entry",
                "pillar": pillar,
                "dimension": dimension,
                "classified": "true" if dimension else "false",
                "score": str(score),
                "label": label,
                "tags": tags,
            })

            chunks.append(Chunk(text=text, metadata=metadata))

        return chunks
