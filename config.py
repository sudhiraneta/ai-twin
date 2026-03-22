import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# Twin Identity
TWIN_NAME = "Sudhira-twin"

# Paths
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
NORMALIZED_DIR = DATA_DIR / "normalized"
CHROMA_DIR = DATA_DIR / "chroma_db"
PERSONA_DIR = DATA_DIR / "persona"

# Singularity Integration
SINGULARITY_DIR = Path(os.environ.get(
    "SINGULARITY_DIR",
    os.path.expanduser("~/PycharmProjects/Singularity"),
))
SINGULARITY_AGENT_DIR = SINGULARITY_DIR / "agent"
SINGULARITY_DATA_DIR = SINGULARITY_DIR / "data"
SYNC_STATE_FILE = DATA_DIR / "sync_state.json"

# Ensure directories exist
for d in [RAW_DIR, NORMALIZED_DIR, CHROMA_DIR, PERSONA_DIR,
          RAW_DIR / "chatgpt", RAW_DIR / "claude", RAW_DIR / "gemini"]:
    d.mkdir(parents=True, exist_ok=True)

# LLM Provider — supports "anthropic", "ollama", "openrouter", "openai_compat"
# Ollama is FREE (runs locally). OpenRouter gives access to cheap models.
LLM_PROVIDER = os.environ.get("LLM_PROVIDER", "ollama")

# API Keys (only needed for paid providers)
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")

# LLM Settings per provider
LLM_MODELS = {
    "ollama": os.environ.get("OLLAMA_MODEL", "llama3"),             # FREE, local
    "anthropic": os.environ.get("CLAUDE_MODEL", "claude-sonnet-4-20250514"),
    "openrouter": os.environ.get("OPENROUTER_MODEL", "meta-llama/llama-3.1-8b-instruct:free"),  # FREE tier
    "openai_compat": os.environ.get("OPENAI_MODEL", "gpt-4o-mini"),
}

LLM_MODEL = LLM_MODELS.get(LLM_PROVIDER, LLM_MODELS["ollama"])

# Base URLs per provider
LLM_BASE_URLS = {
    "ollama": os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434/v1"),
    "openrouter": "https://openrouter.ai/api/v1",
    "openai_compat": os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1"),
}

MAX_CONTEXT_MESSAGES = 20  # max memory chunks to include in context

# Embedding Settings
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
CHUNK_SIZE = 500  # characters per chunk
CHUNK_OVERLAP = 50

# ChromaDB
CHROMA_COLLECTION = "ai_twin_memory"

# Persona Dimensions (v2)
PERSONA_VERSION = 2
SNAPSHOTS_DIR = PERSONA_DIR / "snapshots"
LOGS_DIR = DATA_DIR / "logs"
INCREMENTAL_DIMENSION_THRESHOLD = 10  # re-extract dimension if 10+ new chunks
MAX_EVIDENCE_PER_DIMENSION = 200      # max chunks to send to LLM per dimension

# Ensure v2 directories exist
for d in [SNAPSHOTS_DIR, LOGS_DIR]:
    d.mkdir(parents=True, exist_ok=True)
