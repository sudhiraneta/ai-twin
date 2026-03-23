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

# LLM Provider — "groq" (fast+free), "ollama" (local), "anthropic", "openrouter", "openai_compat"
# Groq: free tier ~500 tok/sec. Ollama: free, local ~30 tok/sec.
LLM_PROVIDER = os.environ.get("LLM_PROVIDER", "groq")

# API Keys
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")

# LLM Settings per provider
LLM_MODELS = {
    "groq": os.environ.get("GROQ_MODEL", "llama-3.1-8b-instant"),  # FREE, fast
    "ollama": os.environ.get("OLLAMA_MODEL", "llama3"),             # FREE, local
    "anthropic": os.environ.get("CLAUDE_MODEL", "claude-sonnet-4-20250514"),
    "openrouter": os.environ.get("OPENROUTER_MODEL", "meta-llama/llama-3.1-8b-instruct:free"),
    "openai_compat": os.environ.get("OPENAI_MODEL", "gpt-4o-mini"),
}

LLM_MODEL = LLM_MODELS.get(LLM_PROVIDER, LLM_MODELS["groq"])

# Base URLs per provider
LLM_BASE_URLS = {
    "groq": "https://api.groq.com/openai/v1",
    "ollama": os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434/v1"),
    "openrouter": "https://openrouter.ai/api/v1",
    "openai_compat": os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1"),
}

MAX_CONTEXT_MESSAGES = 12  # fewer high-quality chunks > many low-quality

# RAG Retrieval Quality
RELEVANCE_THRESHOLD = 1.2   # cosine distance cutoff (0=identical, 2=opposite); lower = stricter
RECENCY_WEIGHT = 0.15       # 0.0 = pure similarity, 1.0 = pure recency

# Embedding Settings
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
CHUNK_SIZE = 1200   # characters per chunk (matches MiniLM ~256 token window)
CHUNK_OVERLAP = 150

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
