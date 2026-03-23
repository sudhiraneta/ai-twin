"""
Google Photos connector — syncs photo metadata into ChromaDB for wardrobe + travel.

Syncs every 2 days. Only pulls metadata (dates, locations, descriptions, categories).
Does NOT download images — zero storage cost.

First-time setup:
  1. Go to console.cloud.google.com → create project
  2. Enable "Photos Library API"
  3. Create OAuth 2.0 credentials (Desktop app)
  4. Download client_secret.json → save to data/google_photos_client.json
  5. Run: python -c "from connectors.photos_connector import PhotosConnector; PhotosConnector().authenticate()"
  6. This opens a browser for one-time OAuth consent
"""

import json
import os
from datetime import datetime, timezone
from pathlib import Path

from config import DATA_DIR
from memory.chunker import Chunk, _ensure_metadata
from .base import BaseConnector

CREDENTIALS_FILE = DATA_DIR / "google_photos_client.json"
TOKEN_FILE = DATA_DIR / "google_photos_token.json"
SCOPES = ["https://www.googleapis.com/auth/photoslibrary.readonly"]


class PhotosConnector(BaseConnector):
    """Connector for Google Photos metadata — wardrobe, travel, food, events."""

    source_name = "google_photos"

    def authenticate(self):
        """One-time OAuth flow — opens browser for consent."""
        from google_auth_oauthlib.flow import InstalledAppFlow

        if not CREDENTIALS_FILE.exists():
            print(f"Missing: {CREDENTIALS_FILE}")
            print("Download OAuth client JSON from Google Cloud Console and save it there.")
            return None

        flow = InstalledAppFlow.from_client_secrets_file(str(CREDENTIALS_FILE), SCOPES)
        creds = flow.run_local_server(port=0)

        TOKEN_FILE.write_text(creds.to_json())
        print(f"Authenticated! Token saved to {TOKEN_FILE}")
        return creds

    def _get_credentials(self):
        """Load saved credentials, refreshing if expired."""
        if not TOKEN_FILE.exists():
            return None

        from google.oauth2.credentials import Credentials
        from google.auth.transport.requests import Request

        creds = Credentials.from_authorized_user_file(str(TOKEN_FILE), SCOPES)
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
            TOKEN_FILE.write_text(creds.to_json())
        return creds if creds and creds.valid else None

    def _get_service(self):
        """Build the Google Photos API service."""
        from googleapiclient.discovery import build

        creds = self._get_credentials()
        if not creds:
            print("Not authenticated. Run: PhotosConnector().authenticate()")
            return None
        return build("photoslibrary", "v1", credentials=creds, static_discovery=False)

    def fetch(self, since: float | None = None, days_back: int = 60) -> list[Chunk]:
        """Fetch photo metadata and create chunks for wardrobe/travel/food."""
        service = self._get_service()
        if not service:
            return []

        # Build date filter
        if since:
            start_dt = datetime.fromtimestamp(since, tz=timezone.utc)
        else:
            from datetime import timedelta
            start_dt = datetime.now(tz=timezone.utc) - timedelta(days=days_back)

        date_filter = {
            "dateFilter": {
                "ranges": [{
                    "startDate": {
                        "year": start_dt.year,
                        "month": start_dt.month,
                        "day": start_dt.day,
                    },
                    "endDate": {
                        "year": datetime.now().year,
                        "month": datetime.now().month,
                        "day": datetime.now().day,
                    },
                }]
            }
        }

        # Fetch photos
        all_items = []
        page_token = None
        max_pages = 10  # limit to avoid excessive API calls

        for _ in range(max_pages):
            body = {"filters": date_filter, "pageSize": 100}
            if page_token:
                body["pageToken"] = page_token

            try:
                results = service.mediaItems().search(body=body).execute()
            except Exception as e:
                print(f"  Photos API error: {e}")
                break

            items = results.get("mediaItems", [])
            all_items.extend(items)
            page_token = results.get("nextPageToken")
            if not page_token:
                break

        print(f"  Fetched {len(all_items)} photos metadata")

        # Group by date and create chunks
        chunks = []
        by_date: dict[str, list] = {}

        for item in all_items:
            metadata = item.get("mediaMetadata", {})
            creation_time = metadata.get("creationTime", "")
            date_str = creation_time[:10] if creation_time else "unknown"

            photo_info = {
                "id": item.get("id", ""),
                "filename": item.get("filename", ""),
                "description": item.get("description", ""),
                "mime_type": item.get("mimeType", ""),
                "creation_time": creation_time,
                "width": metadata.get("width", ""),
                "height": metadata.get("height", ""),
            }

            # Extract location if available (from photo metadata)
            if "photo" in metadata:
                photo_meta = metadata["photo"]
                photo_info["camera"] = photo_meta.get("cameraMake", "") + " " + photo_meta.get("cameraModel", "")

            by_date.setdefault(date_str, []).append(photo_info)

        # Create daily photo summary chunks
        for date_str, photos in sorted(by_date.items()):
            descriptions = [p["description"] for p in photos if p.get("description")]
            filenames = [p["filename"] for p in photos]

            text_lines = [
                f"Photos: {date_str} ({len(photos)} photos)",
            ]

            if descriptions:
                text_lines.append(f"Descriptions: {'; '.join(descriptions[:10])}")

            # Infer categories from filenames and descriptions
            all_text = " ".join(descriptions + filenames).lower()
            categories = []
            if any(w in all_text for w in ["outfit", "ootd", "mirror", "selfie", "dress", "shirt", "jacket"]):
                categories.append("outfit")
            if any(w in all_text for w in ["food", "meal", "dinner", "lunch", "restaurant", "cafe", "coffee"]):
                categories.append("food")
            if any(w in all_text for w in ["travel", "trip", "flight", "hotel", "beach", "mountain", "hike"]):
                categories.append("travel")
            if any(w in all_text for w in ["gym", "workout", "run", "yoga"]):
                categories.append("fitness")

            if categories:
                text_lines.append(f"Categories: {', '.join(categories)}")

            text_lines.append(f"Files: {', '.join(filenames[:10])}")

            # Determine dimension
            if "outfit" in categories:
                dimension = "wardrobe"
                pillar = "SOCIAL"
            elif "travel" in categories:
                dimension = "travel"
                pillar = "PURPOSE"
            elif "food" in categories:
                dimension = "nutrition"
                pillar = "BODY"
            else:
                dimension = "life"
                pillar = "PURPOSE"

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

        # Also create per-category summary chunks for wardrobe analysis
        outfit_photos = [p for photos in by_date.values() for p in photos
                        if any(w in (p.get("description", "") + p.get("filename", "")).lower()
                              for w in ["outfit", "ootd", "mirror", "selfie"])]
        if outfit_photos:
            text = f"Wardrobe: {len(outfit_photos)} outfit photos found\n"
            text += "Dates: " + ", ".join(sorted(set(p["creation_time"][:10] for p in outfit_photos)))
            chunks.append(Chunk(
                text=text,
                metadata=_ensure_metadata({
                    "source": self.source_name,
                    "conversation_id": "wardrobe_summary",
                    "title": "Wardrobe photo summary",
                    "timestamp": datetime.now(tz=timezone.utc).isoformat(),
                    "msg_timestamp": datetime.now(tz=timezone.utc).isoformat(),
                    "role": "user",
                    "type": "wardrobe_summary",
                    "pillar": "SOCIAL",
                    "dimension": "wardrobe",
                    "classified": "true",
                }),
            ))

        return chunks

    def get_albums(self) -> list[dict]:
        """List all albums (useful for finding outfit/travel albums)."""
        service = self._get_service()
        if not service:
            return []

        albums = []
        page_token = None
        for _ in range(5):
            result = service.albums().list(pageSize=50, pageToken=page_token).execute()
            albums.extend(result.get("albums", []))
            page_token = result.get("nextPageToken")
            if not page_token:
                break

        return [{"id": a.get("id"), "title": a.get("title"), "count": a.get("mediaItemsCount")}
                for a in albums]
