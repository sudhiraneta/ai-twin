"""Parser for YouTube watch history and search history from Google Takeout.

Extracts watch history and search history as conversation-like entries
for ingestion into the AI Twin's memory.
"""

import re
from datetime import datetime

from bs4 import BeautifulSoup

from .base import Message, Conversation, BaseParser


class YouTubeParser(BaseParser):
    """Parses YouTube watch/search history from Google Takeout HTML files."""

    source = "youtube"

    def parse(self, file_path) -> list[Conversation]:
        with open(file_path, "r", encoding="utf-8") as f:
            soup = BeautifulSoup(f.read(), "html.parser")

        outer_cells = soup.find_all("div", class_="outer-cell")
        conversations = []

        for cell in outer_cells:
            content_cells = cell.find_all("div", class_="content-cell")
            if not content_cells:
                continue

            main_cell = content_cells[0]
            text = main_cell.get_text(separator="\n", strip=True)
            if not text:
                continue

            lines = text.split("\n")
            lines = [l.strip() for l in lines if l.strip()]

            # Detect type: "Watched" or "Searched for"
            if not lines:
                continue

            first = lines[0]
            entry_type = ""
            content = ""

            if first.startswith("Watched"):
                entry_type = "watch"
                # Title is usually the next non-empty line (the link text)
                content = " ".join(lines[1:3]) if len(lines) > 1 else first
            elif first.startswith("Searched for"):
                entry_type = "search"
                content = first.replace("Searched for", "").strip()
            else:
                continue

            # Extract timestamp
            ts = ""
            timestamp_pattern = re.compile(
                r"(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+\d{1,2},\s+\d{4},\s+\d{1,2}:\d{2}:\d{2}\s+(AM|PM)\s+\w+"
            )
            for line in lines:
                match = timestamp_pattern.search(line)
                if match:
                    ts_clean = re.sub(r"\s+\w+$", "", match.group())
                    try:
                        dt = datetime.strptime(ts_clean, "%b %d, %Y, %I:%M:%S %p")
                        ts = dt.isoformat()
                    except ValueError:
                        ts = match.group()
                    break

            if not content or len(content) < 3:
                continue

            # Extract channel name if present (usually after the title)
            channel = ""
            for line in lines:
                if line and not line.startswith("Watched") and not line.startswith("Searched") and "youtube.com" not in line.lower():
                    if timestamp_pattern.search(line):
                        continue
                    if len(line) < 50 and line != content:
                        channel = line
                        break

            msg_text = f"YouTube {entry_type}: {content}"
            if channel:
                msg_text += f" (by {channel})"

            messages = [Message(role="user", content=msg_text, timestamp=ts)]

            conv = Conversation(
                source=self.source,
                conversation_id=f"yt_{entry_type}_{len(conversations)}",
                timestamp=ts,
                messages=messages,
                title=content[:80],
            )
            conversations.append(conv)

        return conversations
