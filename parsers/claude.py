import json
from datetime import datetime, timezone
from pathlib import Path

from .base import BaseParser, Conversation, Message


class ClaudeParser(BaseParser):
    """Parser for Claude (claude.ai) data exports.

    Claude exports contain a list of conversations, each with chat_messages.
    """

    def parse(self, path: Path) -> list[Conversation]:
        # Accept directory or file
        if path.is_dir():
            # Claude export may have multiple JSON files
            json_files = list(path.glob("*.json"))
            if not json_files:
                raise FileNotFoundError(f"No JSON files found in {path}")
            raw = []
            for f in json_files:
                data = json.loads(f.read_text())
                if isinstance(data, list):
                    raw.extend(data)
                else:
                    raw.append(data)
        else:
            raw = json.loads(path.read_text())
            if not isinstance(raw, list):
                raw = [raw]

        conversations = []

        for conv in raw:
            messages = self._extract_messages(conv)
            if not messages:
                continue

            timestamp = conv.get("created_at") or conv.get("updated_at")
            if not timestamp:
                timestamp = messages[0].timestamp or datetime.now(tz=timezone.utc).isoformat()

            conversations.append(Conversation(
                source="claude",
                conversation_id=conv.get("uuid", conv.get("id", "")),
                timestamp=timestamp,
                messages=messages,
                title=conv.get("name") or conv.get("title"),
            ))

        return conversations

    def _extract_messages(self, conv: dict) -> list[Message]:
        """Extract messages from Claude's conversation format."""
        messages = []
        chat_messages = conv.get("chat_messages", [])

        for msg in chat_messages:
            role = msg.get("sender", "")
            # Claude uses "human" and "assistant"
            if role == "human":
                role = "user"
            elif role != "assistant":
                continue

            # Content can be a string or a list of content blocks
            content = msg.get("text", "")
            if not content:
                content_blocks = msg.get("content", [])
                if isinstance(content_blocks, list):
                    content = " ".join(
                        b.get("text", "") for b in content_blocks
                        if isinstance(b, dict) and b.get("type") == "text"
                    ).strip()
                elif isinstance(content_blocks, str):
                    content = content_blocks

            if not content:
                continue

            timestamp = msg.get("created_at") or msg.get("updated_at")

            messages.append(Message(
                role=role,
                content=content,
                timestamp=timestamp,
            ))

        return messages
