from abc import ABC, abstractmethod
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
import json


@dataclass
class Message:
    role: str  # "user" or "assistant"
    content: str
    timestamp: str | None = None  # ISO format


@dataclass
class Conversation:
    source: str  # "chatgpt", "claude", "gemini"
    conversation_id: str
    timestamp: str  # ISO format
    messages: list[Message] = field(default_factory=list)
    title: str | None = None
    topics: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)

    def user_messages(self) -> list[str]:
        return [m.content for m in self.messages if m.role == "user"]


class BaseParser(ABC):
    """Base class for platform-specific conversation parsers."""

    @abstractmethod
    def parse(self, path: Path) -> list[Conversation]:
        """Parse export files and return normalized conversations."""
        ...

    def save_normalized(self, conversations: list[Conversation], output_dir: Path) -> Path:
        """Save normalized conversations to JSON."""
        output_dir.mkdir(parents=True, exist_ok=True)
        output_file = output_dir / f"{conversations[0].source}_normalized.json"
        data = [c.to_dict() for c in conversations]
        output_file.write_text(json.dumps(data, indent=2, default=str))
        print(f"Saved {len(conversations)} conversations to {output_file}")
        return output_file
