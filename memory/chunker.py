from dataclasses import dataclass


@dataclass
class Chunk:
    text: str
    metadata: dict  # source, conversation_id, timestamp, role, pillar, dimension, etc.


# Default metadata fields for every chunk
DEFAULT_METADATA = {
    "source": "",
    "conversation_id": "",
    "title": "",
    "timestamp": "",
    "msg_timestamp": "",
    "role": "",
    "type": "",
    "pillar": "",
    "dimension": "",
    "classified": "false",
}


def _ensure_metadata(metadata: dict) -> dict:
    """Ensure all canonical metadata fields exist and values are strings."""
    result = {**DEFAULT_METADATA, **metadata}
    # ChromaDB requires all metadata values to be str, int, float, or bool
    for key, value in result.items():
        if value is None:
            result[key] = ""
        elif isinstance(value, (list, dict)):
            import json
            result[key] = json.dumps(value)
    return result


class Chunker:
    """Splits conversation messages into semantic chunks for embedding."""

    def __init__(self, chunk_size: int = 1200, overlap: int = 150):
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk_conversations(self, conversations: list[dict]) -> list[Chunk]:
        """Chunk all conversations into embeddable segments."""
        chunks = []

        for conv in conversations:
            source = conv["source"]
            conv_id = conv["conversation_id"]
            title = conv.get("title", "")
            timestamp = conv["timestamp"]

            # Strategy 1: Chunk individual user messages (for persona/memory)
            for msg in conv["messages"]:
                if msg["role"] == "user":
                    text_chunks = self._split_text(msg["content"])
                    for text in text_chunks:
                        chunks.append(Chunk(
                            text=text,
                            metadata=_ensure_metadata({
                                "source": source,
                                "conversation_id": conv_id,
                                "title": title or "",
                                "timestamp": timestamp,
                                "msg_timestamp": msg.get("timestamp", ""),
                                "role": "user",
                                "type": "user_message",
                            })
                        ))

            # Strategy 2: Chunk user-assistant pairs (for context retrieval)
            messages = conv["messages"]
            for i in range(0, len(messages) - 1, 2):
                if i + 1 < len(messages):
                    user_msg = messages[i]
                    asst_msg = messages[i + 1]
                    if user_msg["role"] == "user" and asst_msg["role"] == "assistant":
                        pair_text = f"User: {user_msg['content']}\nAssistant: {asst_msg['content']}"
                        text_chunks = self._split_text(pair_text)
                        for text in text_chunks:
                            chunks.append(Chunk(
                                text=text,
                                metadata=_ensure_metadata({
                                    "source": source,
                                    "conversation_id": conv_id,
                                    "title": title or "",
                                    "timestamp": timestamp,
                                    "msg_timestamp": user_msg.get("timestamp", ""),
                                    "role": "pair",
                                    "type": "conversation_pair",
                                })
                            ))

        return chunks

    def _split_text(self, text: str) -> list[str]:
        """Split text into chunks with overlap."""
        if len(text) <= self.chunk_size:
            return [text]

        chunks = []
        start = 0
        while start < len(text):
            end = start + self.chunk_size

            # Try to break at sentence boundary
            if end < len(text):
                for sep in ["\n\n", ". ", "\n", "! ", "? ", ", "]:
                    last_sep = text[start:end].rfind(sep)
                    if last_sep > self.chunk_size // 2:
                        end = start + last_sep + len(sep)
                        break

            chunks.append(text[start:end].strip())
            start = end - self.overlap

        return [c for c in chunks if c]

    def chunk_text_with_metadata(self, text: str, metadata: dict) -> list[Chunk]:
        """Split a single text into chunks, preserving metadata on each."""
        metadata = _ensure_metadata(metadata)
        segments = self._split_text(text)
        return [Chunk(text=seg, metadata=metadata.copy()) for seg in segments]
