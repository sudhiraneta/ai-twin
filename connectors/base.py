import json
import time
from abc import ABC, abstractmethod

from config import SYNC_STATE_FILE
from memory.chunker import Chunk


class BaseConnector(ABC):
    """Base class for Singularity data source connectors.

    Each connector reads from a live data source (SQLite DB, files, etc.)
    and produces Chunk objects ready for vector store ingestion.
    Supports incremental sync via a shared sync_state.json file.
    """

    source_name: str = ""

    @abstractmethod
    def fetch(self, since: float | None = None) -> list[Chunk]:
        """Fetch data from the source. If since is provided, only fetch newer data."""
        ...

    def get_last_sync(self) -> float | None:
        """Read last sync timestamp for this connector from sync state file."""
        if not SYNC_STATE_FILE.exists():
            return None
        state = json.loads(SYNC_STATE_FILE.read_text())
        entry = state.get(self.source_name)
        if entry:
            return entry.get("last_sync")
        return None

    def set_last_sync(self, ts: float, chunks_total: int = 0) -> None:
        """Write last sync timestamp for this connector to sync state file."""
        state = {}
        if SYNC_STATE_FILE.exists():
            state = json.loads(SYNC_STATE_FILE.read_text())
        prev_total = state.get(self.source_name, {}).get("chunks_total", 0)
        state[self.source_name] = {
            "last_sync": ts,
            "chunks_total": prev_total + chunks_total,
        }
        SYNC_STATE_FILE.write_text(json.dumps(state, indent=2))

    def sync(self) -> list[Chunk]:
        """Fetch new data since last sync, then update sync timestamp."""
        last = self.get_last_sync()
        now = time.time()
        chunks = self.fetch(since=last)
        self.set_last_sync(now, chunks_total=len(chunks))
        return chunks

    def get_status(self) -> dict:
        """Return sync status for this connector."""
        if not SYNC_STATE_FILE.exists():
            return {"source": self.source_name, "last_sync": None, "chunks_total": 0}
        state = json.loads(SYNC_STATE_FILE.read_text())
        entry = state.get(self.source_name, {})
        return {
            "source": self.source_name,
            "last_sync": entry.get("last_sync"),
            "chunks_total": entry.get("chunks_total", 0),
        }
