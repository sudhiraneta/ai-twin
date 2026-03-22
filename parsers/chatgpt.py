import json
from datetime import datetime, timezone
from pathlib import Path

from .base import BaseParser, Conversation, Message


class ChatGPTParser(BaseParser):
    """Parser for ChatGPT data exports.

    Expects the `conversations.json` file from ChatGPT's export.
    Format: list of conversation objects with nested message mapping.
    """

    def parse(self, path: Path) -> list[Conversation]:
        # Accept either a directory or the JSON file directly
        if path.is_dir():
            json_file = path / "conversations.json"
        else:
            json_file = path

        if not json_file.exists():
            raise FileNotFoundError(f"ChatGPT export not found at {json_file}")

        raw = json.loads(json_file.read_text())
        conversations = []

        for conv in raw:
            messages = self._extract_messages(conv.get("mapping", {}))
            if not messages:
                continue

            timestamp = conv.get("create_time")
            if timestamp:
                timestamp = datetime.fromtimestamp(timestamp, tz=timezone.utc).isoformat()
            else:
                timestamp = messages[0].timestamp or datetime.now(tz=timezone.utc).isoformat()

            conversations.append(Conversation(
                source="chatgpt",
                conversation_id=conv.get("id", ""),
                timestamp=timestamp,
                messages=messages,
                title=conv.get("title"),
            ))

        return conversations

    def _extract_messages(self, mapping: dict) -> list[Message]:
        """Extract ordered messages from ChatGPT's nested mapping structure."""
        messages = []
        nodes = []

        for node_id, node in mapping.items():
            msg = node.get("message")
            if not msg:
                continue

            role = msg.get("author", {}).get("role", "")
            if role not in ("user", "assistant"):
                continue

            content_parts = msg.get("content", {}).get("parts", [])
            content = " ".join(
                str(p) for p in content_parts if isinstance(p, str)
            ).strip()

            if not content:
                continue

            ts = msg.get("create_time")
            timestamp = None
            if ts:
                timestamp = datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()

            nodes.append((ts or 0, Message(
                role=role,
                content=content,
                timestamp=timestamp,
            )))

        # Sort by timestamp to get correct order
        nodes.sort(key=lambda x: x[0])
        return [msg for _, msg in nodes]
