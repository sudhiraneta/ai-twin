import json
from pathlib import Path

from fastapi import APIRouter, UploadFile, File, Form
from pydantic import BaseModel

from config import RAW_DIR, NORMALIZED_DIR
from parsers import ChatGPTParser, ClaudeParser, GeminiParser
from memory.chunker import Chunker
from memory.vectorstore import VectorStore
from persona.extractor import PersonaExtractor
from persona.profile import PersonaProfile
from twin.engine import TwinEngine

router = APIRouter()

# Singleton instances
_twin_engine: TwinEngine | None = None
_vector_store: VectorStore | None = None


def get_twin() -> TwinEngine:
    global _twin_engine
    if _twin_engine is None:
        _twin_engine = TwinEngine()
    return _twin_engine


def get_vector_store() -> VectorStore:
    global _vector_store
    if _vector_store is None:
        _vector_store = VectorStore()
    return _vector_store


# --- Chat ---

class ChatRequest(BaseModel):
    message: str


class ChatResponse(BaseModel):
    response: str


@router.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    twin = get_twin()
    response = twin.chat(req.message)
    return ChatResponse(response=response)


@router.post("/chat/reset")
def reset_chat():
    twin = get_twin()
    twin.reset_conversation()
    return {"status": "conversation reset"}


# --- Decision Mode ---

class DecideRequest(BaseModel):
    question: str


@router.post("/decide")
def decide(req: DecideRequest):
    """Get a dual-lens decision analysis: your likely decision vs ideal decision."""
    twin = get_twin()
    result = twin.decide(req.question)
    return result.to_dict()


# --- Learn ---

class LearnRequest(BaseModel):
    data_point: str


@router.post("/learn")
def learn(req: LearnRequest):
    """Ingest a new data point about the user into memory."""
    twin = get_twin()
    return twin.learn(req.data_point)


# --- Memory ---

class SearchRequest(BaseModel):
    query: str
    n_results: int = 10


@router.post("/memory/search")
def search_memory(req: SearchRequest):
    twin = get_twin()
    results = twin.search_memory(req.query, n_results=req.n_results)
    return {"results": results}


@router.get("/memory/stats")
def memory_stats():
    store = get_vector_store()
    return {"total_chunks": store.count()}


# --- Persona ---

@router.get("/persona")
def get_persona():
    profile = PersonaProfile.load()
    return {
        "communication_style": profile.communication_style,
        "knowledge_domains": profile.knowledge_domains,
        "decision_patterns": profile.decision_patterns,
        "values_and_priorities": profile.values_and_priorities,
        "interests": profile.interests,
        "cognitive_biases": profile.cognitive_biases,
        "risk_tolerance": profile.risk_tolerance,
        "time_preference": profile.time_preference,
        "decision_history": profile.decision_history,
        "system_prompt": profile.system_prompt,
    }


# --- Ingest ---

PARSERS = {
    "chatgpt": ChatGPTParser(),
    "claude": ClaudeParser(),
    "gemini": GeminiParser(),
}


@router.post("/ingest")
async def ingest(
    file: UploadFile = File(...),
    source: str = Form(...),
):
    """Upload and ingest a conversation export file."""
    if source not in PARSERS:
        return {"error": f"Unknown source: {source}. Use: chatgpt, claude, gemini"}

    raw_dir = RAW_DIR / source
    raw_dir.mkdir(parents=True, exist_ok=True)
    file_path = raw_dir / file.filename
    content = await file.read()
    file_path.write_bytes(content)

    parser = PARSERS[source]
    conversations = parser.parse(file_path)

    if not conversations:
        return {"error": "No conversations found in the uploaded file"}

    parser.save_normalized(conversations, NORMALIZED_DIR)

    chunker = Chunker()
    conv_dicts = [c.to_dict() for c in conversations]
    chunks = chunker.chunk_conversations(conv_dicts)

    store = get_vector_store()
    count = store.ingest(chunks)

    return {
        "status": "success",
        "conversations_parsed": len(conversations),
        "chunks_ingested": count,
    }


@router.post("/persona/extract")
def extract_persona():
    """Extract/re-extract persona from all normalized conversations."""
    all_user_messages = []

    for f in NORMALIZED_DIR.glob("*_normalized.json"):
        conversations = json.loads(f.read_text())
        for conv in conversations:
            for msg in conv["messages"]:
                if msg["role"] == "user":
                    all_user_messages.append(msg["content"])

    if not all_user_messages:
        return {"error": "No user messages found. Ingest conversations first."}

    extractor = PersonaExtractor()
    profile = extractor.extract(all_user_messages)

    twin = get_twin()
    twin.reload_persona()

    return {
        "status": "success",
        "messages_analyzed": len(all_user_messages),
        "knowledge_domains": profile.knowledge_domains,
        "interests": profile.interests,
        "cognitive_biases": profile.cognitive_biases,
        "risk_tolerance": profile.risk_tolerance,
    }
