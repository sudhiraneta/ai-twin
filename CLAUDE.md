# AI Twin

Personal AI assistant that creates a digital replica of the user by learning from past conversations across ChatGPT, Claude, and Gemini. Uses RAG, vector embeddings, and persona modeling to respond as the user would.

## Tech Stack

- **Python 3.13+** with FastAPI backend and Streamlit frontend
- **Anthropic Claude API** (claude-sonnet-4-20250514) for generation
- **ChromaDB** for vector storage (cosine similarity)
- **sentence-transformers** (all-MiniLM-L6-v2) for local embeddings
- **Pydantic** for data validation

## Project Structure

```
api/routes.py        # FastAPI endpoints (chat, decide, ingest, memory, persona)
twin/engine.py       # Core TwinEngine: chat, decide, learn, memory retrieval
twin/prompts.py      # Modular prompt composition (base, decision, data collection)
memory/vectorstore.py # ChromaDB wrapper (ingest, search, filter by type)
memory/chunker.py    # Text chunking (user messages + user-assistant pairs)
memory/embeddings.py # Lazy-loaded sentence-transformer embeddings
persona/profile.py   # PersonaProfile dataclass (style, domains, biases, values)
persona/extractor.py # Claude-powered persona extraction from user messages
parsers/             # Platform-specific parsers (chatgpt, claude, gemini)
ui/app.py            # Streamlit UI (chat, decide, memory search, import)
tests/               # Tests
data/                # Raw exports, normalized JSON, ChromaDB, persona profiles
config.py            # Configuration, paths, API keys
main.py              # FastAPI entry point (uvicorn, port 8000)
```

## Running

```bash
# API server
python main.py  # http://localhost:8000, docs at /docs

# Web UI
streamlit run ui/app.py

# Tests
python tests/test_coffee_decision.py
```

## Key Patterns

- **RAG Architecture**: Conversations -> chunks -> embeddings -> ChromaDB -> semantic retrieval -> prompt injection
- **Dual-Lens Decision Mode**: Predicts user's likely decision + objectively ideal decision with gap analysis
- **Lazy Initialization**: Singleton TwinEngine and VectorStore, models loaded on first use
- **Modular Prompts**: Composable system prompts built from base + persona + memory + mode-specific parts
- **Dual Chunking**: Individual user messages (persona analysis) + user-assistant pairs (context retrieval)

## Environment

- Requires `ANTHROPIC_API_KEY` environment variable (or in `.env`)
- Data stored in `data/` directory (raw, normalized, chroma_db, persona)
