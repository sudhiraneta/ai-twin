import json
from datetime import datetime, timezone
from pathlib import Path

from .base import BaseParser, Conversation, Message


class GeminiParser(BaseParser):
    """Parser for Google Gemini data exports (via Google Takeout).

    Gemini Takeout exports contain conversation files in JSON format
    under a 'Gemini Apps' directory.
    """

    def parse(self, path: Path) -> list[Conversation]:
        if path.is_dir():
            # Look for JSON files in the Gemini export directory
            json_files = list(path.glob("**/*.json"))
            if not json_files:
                raise FileNotFoundError(f"No JSON files found in {path}")
        else:
            json_files = [path]

        conversations = []

        for json_file in json_files:
            try:
                raw = json.loads(json_file.read_text())
            except (json.JSONDecodeError, UnicodeDecodeError):
                continue

            # Handle both single conversation and list of conversations
            if isinstance(raw, list):
                for conv in raw:
                    parsed = self._parse_conversation(conv)
                    if parsed:
                        conversations.append(parsed)
            elif isinstance(raw, dict):
                parsed = self._parse_conversation(raw)
                if parsed:
                    conversations.append(parsed)

        return conversations

    def _parse_conversation(self, conv: dict) -> Conversation | None:
        """Parse a single Gemini conversation."""
        messages = self._extract_messages(conv)
        if not messages:
            return None

        # Gemini uses various timestamp formats
        timestamp = self._get_timestamp(conv)
        if not timestamp:
            timestamp = messages[0].timestamp or datetime.now(tz=timezone.utc).isoformat()

        return Conversation(
            source="gemini",
            conversation_id=conv.get("conversationId", conv.get("id", "")),
            timestamp=timestamp,
            messages=messages,
            title=conv.get("title"),
        )

    def _extract_messages(self, conv: dict) -> list[Message]:
        """Extract messages from Gemini conversation format."""
        messages = []

        # Gemini Takeout format uses "turns" or "messages"
        turns = conv.get("turns", conv.get("messages", []))

        for turn in turns:
            role = turn.get("role", "").lower()
            if role in ("model", "assistant", "1"):
                role = "assistant"
            elif role in ("user", "human", "0"):
                role = "user"
            else:
                continue

            # Content extraction - handles multiple formats
            content = ""
            parts = turn.get("parts", [])
            if parts:
                content = " ".join(
                    p.get("text", "") if isinstance(p, dict) else str(p)
                    for p in parts
                ).strip()

            if not content:
                content = turn.get("text", turn.get("content", ""))
                if isinstance(content, list):
                    content = " ".join(str(c) for c in content)

            if not content or not content.strip():
                continue

            timestamp = self._get_timestamp(turn)

            messages.append(Message(
                role=role,
                content=content.strip(),
                timestamp=timestamp,
            ))

        return messages

    def _get_timestamp(self, obj: dict) -> str | None:
        """Extract timestamp from various Gemini format fields."""
        for field in ("createTime", "created_at", "timestamp", "updateTime"):
            val = obj.get(field)
            if val:
                if isinstance(val, (int, float)):
                    return datetime.fromtimestamp(val, tz=timezone.utc).isoformat()
                return str(val)
        return None
