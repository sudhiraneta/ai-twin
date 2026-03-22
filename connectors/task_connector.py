import hashlib
import sys

from config import SINGULARITY_AGENT_DIR
from memory.chunker import Chunk, _ensure_metadata
from persona.classifier import PILLAR_DEFAULT_DIMENSION
from .base import BaseConnector


class TaskConnector(BaseConnector):
    """Connector for extracted tasks via Singularity's task_extractor."""

    source_name = "tasks"

    def _import_extractor(self):
        if str(SINGULARITY_AGENT_DIR) not in sys.path:
            sys.path.insert(0, str(SINGULARITY_AGENT_DIR))
        import task_extractor
        return task_extractor

    def fetch(self, since: float | None = None, days_back: int = 14) -> list[Chunk]:
        extractor = self._import_extractor()
        tasks = extractor.extract_from_notes(days=days_back)

        chunks = []
        for t in tasks:
            task_text = t.get("task", "")
            pillar = t.get("pillar", "")
            source_note = t.get("source", "")
            date = t.get("date", "")
            card = t.get("card", "")

            text = f"Task [{pillar}]: {task_text}\nSource: {source_note} | Date: {date}"
            if card:
                text += f"\nCard: {card}"

            # Deterministic ID from task content
            task_hash = hashlib.md5(f"{task_text}_{date}".encode()).hexdigest()[:8]

            # Map pillar to dimension
            dimension = PILLAR_DEFAULT_DIMENSION.get(pillar, "goals")

            metadata = _ensure_metadata({
                "source": "singularity_tasks",
                "conversation_id": f"task_{task_hash}",
                "title": task_text[:80],
                "timestamp": f"{date}T00:00:00+00:00" if date else "",
                "msg_timestamp": f"{date}T00:00:00+00:00" if date else "",
                "role": "user",
                "type": "task",
                "pillar": pillar,
                "dimension": dimension,
                "classified": "true" if dimension else "false",
                "card": card or "",
            })

            chunks.append(Chunk(text=text, metadata=metadata))

        return chunks
