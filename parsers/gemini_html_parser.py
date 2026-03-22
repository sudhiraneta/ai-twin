"""Parser for Google Takeout Gemini MyActivity.html exports.

Google exports Gemini conversations as HTML (not JSON). Each conversation
is an outer-cell div containing the user prompt and Gemini response.
"""

import re
from datetime import datetime

from bs4 import BeautifulSoup

from .base import Message, Conversation, BaseParser


class GeminiHTMLParser(BaseParser):
    """Parses Gemini conversations from Google Takeout MyActivity.html."""

    source = "gemini"

    def parse(self, file_path) -> list[Conversation]:
        with open(file_path, "r", encoding="utf-8") as f:
            soup = BeautifulSoup(f.read(), "html.parser")

        outer_cells = soup.find_all("div", class_="outer-cell")
        conversations = []

        for cell in outer_cells:
            content_cells = cell.find_all("div", class_="content-cell")
            if not content_cells:
                continue

            # First content-cell has the prompt + response + timestamp
            main_cell = content_cells[0]
            text = main_cell.get_text(separator="\n", strip=True)

            if not text:
                continue

            # Parse the structure: "Prompted <user text>\n<timestamp>\n<response>"
            lines = text.split("\n")
            if not lines:
                continue

            # Extract user prompt
            user_text = ""
            response_text = ""
            timestamp_str = ""

            # Find "Prompted" prefix
            first_line = lines[0]
            if first_line.startswith("Prompted"):
                user_text = first_line.replace("Prompted", "", 1).strip()
                # Sometimes the prompt has a colon prefix
                if user_text.startswith(":"):
                    user_text = user_text[1:].strip()
            else:
                user_text = first_line

            # Find timestamp — format like "Mar 21, 2026, 3:38:11 PM PDT"
            timestamp_pattern = re.compile(
                r"(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+\d{1,2},\s+\d{4},\s+\d{1,2}:\d{2}:\d{2}\s+(AM|PM)\s+\w+"
            )

            ts_idx = None
            for i, line in enumerate(lines):
                if timestamp_pattern.search(line):
                    timestamp_str = line.strip()
                    ts_idx = i
                    break

            # Everything after the timestamp is the response
            if ts_idx is not None and ts_idx + 1 < len(lines):
                response_lines = lines[ts_idx + 1:]
                # Filter out boilerplate
                response_lines = [
                    l for l in response_lines
                    if not l.startswith("Products:")
                    and not l.startswith("Why is this here")
                    and not l.startswith("This activity was saved")
                    and l.strip() != "Gemini Apps"
                    and "settings were on" not in l
                    and "control these settings" not in l
                    and l.strip() != "here"
                    and l.strip() != "."
                ]
                response_text = "\n".join(response_lines).strip()

            # Parse timestamp
            ts = ""
            if timestamp_str:
                # Remove timezone abbrev for parsing
                ts_clean = re.sub(r"\s+\w+$", "", timestamp_str)
                try:
                    dt = datetime.strptime(ts_clean, "%b %d, %Y, %I:%M:%S %p")
                    ts = dt.isoformat()
                except ValueError:
                    ts = timestamp_str

            if not user_text or len(user_text) < 3:
                continue

            messages = [Message(role="user", content=user_text, timestamp=ts)]
            if response_text and len(response_text) > 5:
                messages.append(Message(role="assistant", content=response_text, timestamp=ts))

            conv = Conversation(
                source=self.source,
                conversation_id=f"gemini_html_{len(conversations)}",
                timestamp=ts,
                messages=messages,
                title=user_text[:80],
            )
            conversations.append(conv)

        return conversations
