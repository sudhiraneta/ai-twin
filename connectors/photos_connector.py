"""
Google Photos connector via Google Takeout.

Google locked down the Photos Library API for new projects in 2025.
Instead, we parse Google Takeout exports which contain full photo metadata.

Setup:
  1. Go to takeout.google.com
  2. Deselect all → select only "Google Photos"
  3. Export → download ZIP
  4. Put the ZIP (or extracted folder) at: data/takeout/
  5. Run sync — we parse all JSON metadata automatically

The metadata includes: filename, date, location (GPS), description, people, albums.
No images are stored — only metadata goes into the twin's memory.
"""

import json
import os
import zipfile
from datetime import datetime, timezone
from pathlib import Path

from config import DATA_DIR
from memory.chunker import Chunk, _ensure_metadata
from .base import BaseConnector

TAKEOUT_DIR = DATA_DIR / "takeout"


class PhotosConnector(BaseConnector):
    """Connector for Google Photos via Takeout export."""

    source_name = "google_photos"

    def fetch(self, since: float | None = None, **kwargs) -> list[Chunk]:
        """Parse Google Takeout photo metadata from data/takeout/."""
        TAKEOUT_DIR.mkdir(parents=True, exist_ok=True)

        # Auto-extract any ZIP files
        for zip_file in TAKEOUT_DIR.glob("*.zip"):
            extract_dir = TAKEOUT_DIR / zip_file.stem
            if not extract_dir.exists():
                print(f"  Extracting {zip_file.name}...")
                with zipfile.ZipFile(zip_file, "r") as zf:
                    zf.extractall(extract_dir)

        # Find all photo metadata JSON files
        json_files = []
        for root, dirs, files in os.walk(TAKEOUT_DIR):
            for f in files:
                if f.endswith(".json") and not f.startswith("."):
                    json_files.append(Path(root) / f)

        if not json_files:
            print(f"  No takeout data found in {TAKEOUT_DIR}")
            print(f"  Export from takeout.google.com → drop ZIP in {TAKEOUT_DIR}")
            return []

        print(f"  Found {len(json_files)} metadata files")

        # Parse all photo metadata
        photos = []
        for jf in json_files:
            try:
                data = json.loads(jf.read_text(encoding="utf-8"))
                photo = self._parse_metadata(data, jf)
                if photo:
                    photos.append(photo)
            except (json.JSONDecodeError, KeyError):
                continue

        print(f"  Parsed {len(photos)} photos with metadata")

        # Filter by date if since is set
        if since:
            photos = [p for p in photos if p.get("timestamp", 0) > since]

        # Group by date and create chunks
        chunks = []
        by_date: dict[str, list] = {}
        for photo in photos:
            date_str = photo.get("date", "unknown")
            by_date.setdefault(date_str, []).append(photo)

        for date_str, day_photos in sorted(by_date.items()):
            chunks.extend(self._create_day_chunks(date_str, day_photos))

        return chunks

    def _parse_metadata(self, data: dict, file_path: Path) -> dict | None:
        """Parse a single Google Takeout photo metadata JSON."""
        # Takeout JSON structure varies — handle common formats
        title = data.get("title", "")
        description = data.get("description", "")

        # Timestamp
        ts_data = data.get("photoTakenTime", data.get("creationTime", {}))
        timestamp = 0
        date_str = "unknown"
        if isinstance(ts_data, dict) and ts_data.get("timestamp"):
            timestamp = int(ts_data["timestamp"])
            date_str = datetime.fromtimestamp(timestamp, tz=timezone.utc).strftime("%Y-%m-%d")
        elif isinstance(ts_data, str):
            try:
                dt = datetime.fromisoformat(ts_data.replace("Z", "+00:00"))
                timestamp = int(dt.timestamp())
                date_str = dt.strftime("%Y-%m-%d")
            except ValueError:
                pass

        # Location
        geo = data.get("geoData", data.get("geoDataExif", {}))
        lat = geo.get("latitude", 0)
        lng = geo.get("longitude", 0)
        altitude = geo.get("altitude", 0)
        has_location = lat != 0 and lng != 0

        # People
        people = [p.get("name", "") for p in data.get("people", []) if p.get("name")]

        # Google's auto-labels (if present)
        labels = data.get("labels", [])

        # Determine category from filename, description, and labels
        all_text = f"{title} {description} {' '.join(labels)}".lower()
        category = "general"
        if any(w in all_text for w in ["outfit", "ootd", "mirror", "selfie", "dress", "shirt", "jacket", "fashion"]):
            category = "outfit"
        elif any(w in all_text for w in ["food", "meal", "dinner", "lunch", "restaurant", "cafe", "coffee", "cook"]):
            category = "food"
        elif any(w in all_text for w in ["travel", "trip", "flight", "hotel", "beach", "mountain", "hike", "vacation"]):
            category = "travel"
        elif any(w in all_text for w in ["gym", "workout", "run", "yoga", "fitness"]):
            category = "fitness"
        elif has_location:
            category = "places"

        if not title and not description and not has_location and not people:
            return None

        return {
            "title": title,
            "description": description,
            "date": date_str,
            "timestamp": timestamp,
            "lat": lat,
            "lng": lng,
            "has_location": has_location,
            "people": people,
            "labels": labels,
            "category": category,
            "file_path": str(file_path),
        }

    def _create_day_chunks(self, date_str: str, photos: list) -> list[Chunk]:
        """Create chunks from a day's worth of photos."""
        chunks = []

        # Daily summary
        categories = {}
        locations = set()
        all_people = set()
        descriptions = []

        for p in photos:
            cat = p.get("category", "general")
            categories[cat] = categories.get(cat, 0) + 1
            if p.get("has_location"):
                locations.add(f"{p['lat']:.3f},{p['lng']:.3f}")
            all_people.update(p.get("people", []))
            if p.get("description"):
                descriptions.append(p["description"])

        text_lines = [f"Photos: {date_str} ({len(photos)} photos)"]

        cat_str = ", ".join(f"{cat} ({n})" for cat, n in sorted(categories.items(), key=lambda x: -x[1]))
        text_lines.append(f"Categories: {cat_str}")

        if descriptions:
            text_lines.append(f"Descriptions: {'; '.join(descriptions[:5])}")
        if all_people:
            text_lines.append(f"People: {', '.join(sorted(all_people))}")
        if locations:
            text_lines.append(f"Locations: {len(locations)} unique spots")

        # List some filenames
        filenames = [p["title"] for p in photos if p.get("title")][:10]
        if filenames:
            text_lines.append(f"Files: {', '.join(filenames)}")

        # Determine dimension
        dominant = max(categories, key=categories.get) if categories else "general"
        dim_map = {
            "outfit": ("SOCIAL", "wardrobe"),
            "food": ("BODY", "nutrition"),
            "travel": ("PURPOSE", "travel"),
            "fitness": ("BODY", "wellness"),
            "places": ("PURPOSE", "life"),
            "general": ("PURPOSE", "life"),
        }
        pillar, dimension = dim_map.get(dominant, ("PURPOSE", "life"))

        chunks.append(Chunk(
            text="\n".join(text_lines),
            metadata=_ensure_metadata({
                "source": self.source_name,
                "conversation_id": f"photos_{date_str}",
                "title": f"Photos {date_str}",
                "timestamp": f"{date_str}T00:00:00+00:00" if date_str != "unknown" else "",
                "msg_timestamp": f"{date_str}T00:00:00+00:00" if date_str != "unknown" else "",
                "role": "user",
                "type": "photo_daily",
                "pillar": pillar,
                "dimension": dimension,
                "classified": "true",
                "photo_count": str(len(photos)),
            }),
        ))

        # Create per-category chunks for outfit/travel
        for cat in ["outfit", "travel", "food"]:
            cat_photos = [p for p in photos if p.get("category") == cat]
            if len(cat_photos) >= 1:
                cat_lines = [f"{cat.title()}: {date_str} ({len(cat_photos)} photos)"]
                for p in cat_photos[:10]:
                    parts = []
                    if p.get("title"):
                        parts.append(p["title"])
                    if p.get("description"):
                        parts.append(p["description"])
                    if p.get("people"):
                        parts.append(f"with {', '.join(p['people'])}")
                    cat_lines.append(f"- {' | '.join(parts) if parts else '(no details)'}")

                cat_pillar, cat_dim = dim_map.get(cat, ("PURPOSE", "life"))
                chunks.append(Chunk(
                    text="\n".join(cat_lines),
                    metadata=_ensure_metadata({
                        "source": self.source_name,
                        "conversation_id": f"photos_{cat}_{date_str}",
                        "title": f"{cat.title()} {date_str}",
                        "timestamp": f"{date_str}T00:00:00+00:00" if date_str != "unknown" else "",
                        "msg_timestamp": f"{date_str}T00:00:00+00:00" if date_str != "unknown" else "",
                        "role": "user",
                        "type": f"photo_{cat}",
                        "pillar": cat_pillar,
                        "dimension": cat_dim,
                        "classified": "true",
                    }),
                ))

        return chunks

    def get_albums(self) -> list[dict]:
        """List albums found in takeout data."""
        if not TAKEOUT_DIR.exists():
            return []

        albums = []
        for d in TAKEOUT_DIR.rglob("*"):
            if d.is_dir() and any(d.glob("*.json")):
                photo_count = len(list(d.glob("*.json")))
                albums.append({
                    "id": d.name,
                    "title": d.name,
                    "count": str(photo_count),
                })
        return albums
