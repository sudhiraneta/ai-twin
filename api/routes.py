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


# --- Singularity Integration ---

class SingularitySyncRequest(BaseModel):
    sources: list[str] | None = None

class SingularityImportRequest(BaseModel):
    sources: list[str] | None = None
    days_back: int = 30


@router.post("/singularity/sync")
def sync_singularity(req: SingularitySyncRequest = SingularitySyncRequest()):
    """Incremental sync from Singularity data sources."""
    from connectors import ALL_CONNECTORS

    requested = req.sources or list(ALL_CONNECTORS.keys())
    store = get_vector_store()
    results = {}

    for name in requested:
        if name not in ALL_CONNECTORS:
            results[name] = {"error": f"Unknown source: {name}"}
            continue
        try:
            connector = ALL_CONNECTORS[name]()
            chunks = connector.sync()
            if chunks:
                count = store.ingest(chunks)
            else:
                count = 0
            results[name] = {"status": "ok", "chunks_ingested": count}
        except Exception as e:
            results[name] = {"error": str(e)}

    return {"status": "success", "sources": results}


@router.post("/singularity/import")
def import_singularity(req: SingularityImportRequest = SingularityImportRequest()):
    """Full re-import from Singularity data sources (resets sync state)."""
    from connectors import ALL_CONNECTORS

    requested = req.sources or list(ALL_CONNECTORS.keys())
    store = get_vector_store()
    results = {}

    for name in requested:
        if name not in ALL_CONNECTORS:
            results[name] = {"error": f"Unknown source: {name}"}
            continue
        try:
            connector = ALL_CONNECTORS[name]()
            try:
                chunks = connector.fetch(since=None, days_back=req.days_back)
            except TypeError:
                chunks = connector.fetch(since=None)
            if chunks:
                count = store.ingest(chunks)
                connector.set_last_sync(
                    __import__("time").time(), chunks_total=count
                )
            else:
                count = 0
            results[name] = {"status": "ok", "chunks_ingested": count}
        except Exception as e:
            results[name] = {"error": str(e)}

    return {"status": "success", "sources": results}


@router.get("/singularity/status")
def singularity_status():
    """Return sync state for all Singularity connectors."""
    from connectors import ALL_CONNECTORS

    statuses = {}
    for name, cls in ALL_CONNECTORS.items():
        connector = cls()
        statuses[name] = connector.get_status()
    return {"sources": statuses}


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


# --- Daily Loop / Sync ---

@router.post("/sync/run")
def run_daily_loop():
    """Trigger the daily learning loop (incremental)."""
    from daily_loop import DailyLoop
    loop = DailyLoop()
    results = loop.run(full=False)
    twin = get_twin()
    twin.reload_persona()
    return results


@router.post("/sync/run-full")
def run_daily_loop_full():
    """Trigger the daily learning loop with full persona re-extraction."""
    from daily_loop import DailyLoop
    loop = DailyLoop()
    results = loop.run(full=True)
    twin = get_twin()
    twin.reload_persona()
    return results


@router.get("/sync/status")
def sync_status():
    """Return sync state for all connectors and dimension health."""
    from connectors import ALL_CONNECTORS
    from persona.dimensions import DIMENSIONS

    # Connector sync state
    connector_status = {}
    for name, cls in ALL_CONNECTORS.items():
        connector = cls()
        connector_status[name] = connector.get_status()

    # Dimension health
    profile = PersonaProfile.load()
    dimension_status = {}
    for dim_name, dim in profile.dimensions.items():
        dimension_status[dim_name] = {
            "display_name": dim.display_name,
            "pillar": dim.pillar,
            "confidence": dim.confidence,
            "evidence_count": dim.evidence_count,
            "last_updated": dim.last_updated,
            "has_traits": bool(dim.traits),
        }

    store = get_vector_store()
    return {
        "connectors": connector_status,
        "dimensions": dimension_status,
        "total_chunks": store.count(),
        "last_full_extraction": profile.last_full_extraction,
    }


# --- Persona Dimensions (v2) ---

@router.get("/persona/dimensions")
def list_dimensions():
    """List all persona dimensions with confidence and evidence counts."""
    profile = PersonaProfile.load()
    dimensions = {}
    for dim_name, dim in profile.dimensions.items():
        dimensions[dim_name] = {
            "display_name": dim.display_name,
            "pillar": dim.pillar,
            "confidence": dim.confidence,
            "evidence_count": dim.evidence_count,
            "last_updated": dim.last_updated,
            "has_traits": bool(dim.traits),
            "summary": dim.get_summary() if dim.traits else "",
        }
    return {"dimensions": dimensions}


@router.get("/persona/dimensions/{name}")
def get_dimension(name: str):
    """Get detailed data for a single persona dimension."""
    profile = PersonaProfile.load()
    dim = profile.get_dimension(name)
    if not dim:
        return {"error": f"Unknown dimension: {name}"}
    return dim.to_dict()


@router.get("/persona/evolution")
def persona_evolution():
    """Get time series of persona evolution snapshots."""
    snapshots = PersonaProfile.list_snapshots()
    data = []
    for date in snapshots:
        snapshot = PersonaProfile.load_snapshot(date)
        if snapshot:
            data.append(snapshot)
    return {"snapshots": data}


class ExtractDimensionRequest(BaseModel):
    dimension: str


@router.post("/persona/extract-dimension")
def extract_single_dimension(req: ExtractDimensionRequest):
    """Extract/re-extract a single persona dimension."""
    from persona.dimensions import DIMENSIONS as DIM_REGISTRY
    if req.dimension not in DIM_REGISTRY:
        return {"error": f"Unknown dimension: {req.dimension}. Available: {list(DIM_REGISTRY.keys())}"}

    store = get_vector_store()
    results = store.search_by_dimension(query="", dimension=req.dimension, n_results=200)
    chunk_texts = [r["text"] for r in results]

    if not chunk_texts:
        return {"error": f"No chunks found for dimension: {req.dimension}"}

    extractor = PersonaExtractor()
    profile = extractor.incremental_update(
        changed_dimensions=[req.dimension],
        chunks_by_dimension={req.dimension: chunk_texts},
    )

    twin = get_twin()
    twin.reload_persona()

    dim = profile.get_dimension(req.dimension)
    return {
        "status": "success",
        "dimension": req.dimension,
        "confidence": dim.confidence if dim else 0,
        "evidence_count": dim.evidence_count if dim else 0,
        "traits": dim.traits if dim else {},
    }
