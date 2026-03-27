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
    dimension: str | None = None


@router.post("/memory/search")
def search_memory(req: SearchRequest):
    """Skill-aware memory search — uses DIMENSION_SOURCES for tiered retrieval."""
    twin = get_twin()
    skill_result = twin.skill_search(
        query=req.query,
        n_results=req.n_results,
        dimension=req.dimension,
    )
    return {
        "results": skill_result["results"],
        "dimensions": skill_result["dimensions"],
        "skill_context": skill_result["skill_context"],
    }


@router.get("/memory/stats")
def memory_stats():
    store = get_vector_store()

    # Count by dimension
    from persona.dimensions import DIMENSIONS
    dim_counts = {}
    for dim_name in DIMENSIONS:
        dim_counts[dim_name] = store.count_by_dimension(dim_name)

    return {
        "total_chunks": store.count(),
        "by_dimension": dim_counts,
    }


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


# --- Ask Persona (context + LLM answer) ---

class AskPersonaRequest(BaseModel):
    query: str
    context: str
    skill_context: str = ""


@router.post("/persona/ask")
def ask_persona(req: AskPersonaRequest):
    """Generate an LLM answer using skill-file instructions + tiered context."""
    from twin.llm_client import chat_completion
    from persona.profile import PersonaProfile

    profile = PersonaProfile.load()

    # Build system prompt with skill file instructions layered in
    parts = [
        "You are analyzing a person's data to answer questions about them.",
        "Use ONLY the context provided below — do not fabricate information.",
        "If the context doesn't contain relevant data, say so clearly.",
    ]

    # Inject skill file content (traits + do/don't instructions)
    if req.skill_context:
        parts.append("")
        parts.append("=== SKILL FILE (dimension knowledge + instructions) ===")
        parts.append(req.skill_context)
        parts.append("=== END SKILL FILE ===")
        parts.append("")
        parts.append(
            "IMPORTANT: Follow the Instructions section above. "
            "The 'Do' list tells you what to include. "
            "The 'Don't' list tells you what to avoid. "
            "The Traits section contains pre-extracted knowledge — use it as your baseline, "
            "then supplement with the retrieved context below."
        )
    else:
        parts.append(f"\nPersona summary: {profile.system_prompt[:500]}")

    parts.append(f"\n=== RETRIEVED CONTEXT ===\n{req.context}\n=== END CONTEXT ===")

    system = "\n".join(parts)

    answer = chat_completion(
        system=system,
        messages=[{"role": "user", "content": req.query}],
        max_tokens=2048,
    )
    return {"answer": answer}


# --- Skill Files (read/write) ---

@router.get("/persona/skills")
def list_skill_files():
    """List all skill files with their content."""
    from persona.skills import read_all_skill_files, DIMENSION_SOURCES
    from persona.dimensions import DIMENSIONS as DIM_REG

    skills = read_all_skill_files()
    result = {}
    for name, content in skills.items():
        meta = DIM_REG.get(name, {})
        sources = DIMENSION_SOURCES.get(name, {})
        result[name] = {
            "content": content,
            "pillar": meta.get("pillar", ""),
            "display": meta.get("display", name),
            "description": sources.get("description", ""),
            "user_edited": "<!-- user-edited -->" in content,
        }
    return {"skills": result}


@router.get("/persona/skills/{name}")
def get_skill_file(name: str):
    """Read a single skill file."""
    from persona.skills import read_skill_file
    content = read_skill_file(name)
    if content is None:
        return {"error": f"Skill file not found: {name}"}
    return {"name": name, "content": content}


class SkillFileUpdate(BaseModel):
    content: str


@router.put("/persona/skills/{name}")
def update_skill_file(name: str, req: SkillFileUpdate):
    """Update a skill file (user edit). Marks as user-edited so auto-regen preserves changes."""
    from persona.skills import write_skill_file
    success = write_skill_file(name, req.content)
    if success:
        twin = get_twin()
        twin.reload_persona()
        return {"status": "saved", "name": name}
    return {"error": "Failed to save"}


# --- Wardrobe (Google Photos) ---

@router.post("/wardrobe/sync")
def sync_wardrobe(days_back: int = 60):
    """Sync Google Photos metadata for wardrobe/travel analysis."""
    from connectors.photos_connector import PhotosConnector

    connector = PhotosConnector()
    try:
        chunks = connector.fetch(days_back=days_back)
    except Exception as e:
        return {"error": str(e)}

    if chunks:
        store = get_vector_store()
        count = store.ingest(chunks)
        connector.set_last_sync(__import__("time").time(), chunks_total=count)
        return {"status": "success", "photos_synced": count, "total_chunks": len(chunks)}
    return {"status": "no_photos", "photos_synced": 0}


@router.get("/wardrobe")
def get_wardrobe():
    """Get wardrobe summary — outfits, styles, and travel from Google Photos."""
    store = get_vector_store()

    # Search for outfit and wardrobe data
    outfit_chunks = store.search("outfit clothing style wardrobe", n_results=20)
    travel_chunks = store.search("travel trip visit city", n_results=20)
    food_chunks = store.search("restaurant cafe food meal", n_results=10)

    return {
        "outfits": [{"text": c["text"][:300], "date": c["metadata"].get("timestamp", "")[:10]}
                    for c in outfit_chunks if c["metadata"].get("type") in ("photo_daily", "wardrobe_summary")],
        "travel": [{"text": c["text"][:300], "date": c["metadata"].get("timestamp", "")[:10]}
                   for c in travel_chunks if c["metadata"].get("type") == "photo_daily"],
        "food": [{"text": c["text"][:300], "date": c["metadata"].get("timestamp", "")[:10]}
                 for c in food_chunks if c["metadata"].get("source") == "google_photos"],
        "total_photos_indexed": sum(1 for c in outfit_chunks + travel_chunks
                                    if c["metadata"].get("source") == "google_photos"),
    }


# --- Metrics (SQL) ---

@router.get("/metrics/gym")
def gym_metrics(week_id: str | None = None):
    """Get gym session data for a given week."""
    from db import MetricStore
    store = MetricStore()
    result = store.gym_this_week(week_id)
    return {"data": result.data, "summary": result.summary, "time_range": result.time_range}


@router.get("/metrics/gym/streak")
def gym_streak():
    """Get gym session trend over last 8 weeks."""
    from db import MetricStore
    store = MetricStore()
    result = store.gym_streak()
    return {"data": result.data, "summary": result.summary, "time_range": result.time_range}


@router.get("/metrics/nutrition")
def nutrition_metrics(week_id: str | None = None):
    """Get nutrition data for a given week."""
    from db import MetricStore
    store = MetricStore()
    result = store.nutrition_this_week(week_id)
    return {"data": result.data, "summary": result.summary, "time_range": result.time_range}


@router.get("/metrics/communications")
def comms_metrics(week_id: str | None = None):
    """Get communication interaction data for a given week."""
    from db import MetricStore
    store = MetricStore()
    result = store.comms_this_week(week_id)
    return {"data": result.data, "summary": result.summary, "time_range": result.time_range}


@router.get("/metrics/tasks")
def task_metrics(week_id: str | None = None):
    """Get task completion data for a given week."""
    from db import MetricStore
    store = MetricStore()
    result = store.tasks_this_week(week_id)
    return {"data": result.data, "summary": result.summary, "time_range": result.time_range}


@router.get("/metrics/wellness")
def wellness_metrics(week_id: str | None = None):
    """Get wellness habit data for a given week."""
    from db import MetricStore
    store = MetricStore()
    result = store.wellness_this_week(week_id)
    return {"data": result.data, "summary": result.summary, "time_range": result.time_range}


@router.get("/metrics/weekly-summary")
def weekly_summary_metrics(week_id: str | None = None):
    """Get full weekly overview with all metrics."""
    from db import MetricStore
    store = MetricStore()
    result = store.weekly_summary(week_id)
    return {"data": result.data, "summary": result.summary, "time_range": result.time_range}


@router.post("/metrics/query")
def hybrid_query(req: SearchRequest):
    """Smart hybrid query — routes to SQL, RAG, or both based on question."""
    twin = get_twin()
    return twin.hybrid_context(req.query)


@router.get("/wardrobe/albums")
def list_albums():
    """List Google Photos albums."""
    from connectors.photos_connector import PhotosConnector
    connector = PhotosConnector()
    albums = connector.get_albums()
    return {"albums": albums}
