import os
from pathlib import Path

# Paths
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
NORMALIZED_DIR = DATA_DIR / "normalized"
CHROMA_DIR = DATA_DIR / "chroma_db"
PERSONA_DIR = DATA_DIR / "persona"

# Ensure directories exist
for d in [RAW_DIR, NORMALIZED_DIR, CHROMA_DIR, PERSONA_DIR,
          RAW_DIR / "chatgpt", RAW_DIR / "claude", RAW_DIR / "gemini"]:
    d.mkdir(parents=True, exist_ok=True)

# API Keys
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")

# LLM Settings
CLAUDE_MODEL = "claude-sonnet-4-20250514"
MAX_CONTEXT_MESSAGES = 20  # max memory chunks to include in context

# Embedding Settings
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
CHUNK_SIZE = 500  # characters per chunk
CHUNK_OVERLAP = 50

# ChromaDB
CHROMA_COLLECTION = "ai_twin_memory"
