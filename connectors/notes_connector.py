import sys
from datetime import datetime, timezone

from config import SINGULARITY_AGENT_DIR, CHUNK_SIZE, CHUNK_OVERLAP
from memory.chunker import Chunk, Chunker, _ensure_metadata
from .base import BaseConnector


class NotesConnector(BaseConnector):
    """Connector for Apple Notes via Singularity's notes_reader."""

    source_name = "apple_notes"

    def __init__(self):
        self._chunker = Chunker(chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP)

    def _import_reader(self):
        if str(SINGULARITY_AGENT_DIR) not in sys.path:
            sys.path.insert(0, str(SINGULARITY_AGENT_DIR))
        import notes_reader
        return notes_reader

    def fetch(self, since: float | None = None, days_back: int = 30) -> list[Chunk]:
        reader = self._import_reader()

        if since:
            notes = reader.get_new_notes(since=since, limit=500)
        else:
            notes = reader.get_all_notes(limit=500, days_back=days_back)

        chunks = []
        for note in notes:
            title = note.get("title", "(untitled)")
            created = datetime.fromtimestamp(
                note.get("created_at", 0), tz=timezone.utc
            ).isoformat()
            modified = datetime.fromtimestamp(
                note.get("modified_at", 0), tz=timezone.utc
            ).isoformat()

            text = f"Note: {title}\nWritten: {created[:10]} | Modified: {modified[:10]}\n\n{note.get('full_text', '')}"

            metadata = _ensure_metadata({
                "source": self.source_name,
                "conversation_id": f"note_{note.get('id', '')}",
                "title": title,
                "timestamp": modified,
                "msg_timestamp": created,
                "role": "user",
                "type": "note",
                "note_id": str(note.get("id", "")),
                # Notes are too varied — leave unclassified for Tier 2
                "pillar": "",
                "dimension": "",
                "classified": "false",
            })

            chunks.extend(self._chunker.chunk_text_with_metadata(text, metadata))

        return chunks
