#!/usr/bin/env python3
"""Comprehensive test suite for AI Twin — runs all tests in a loop until all pass.

Usage: python tests/test_all.py
"""

import json
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

RESULTS = {}
LOG_FILE = "data/logs/test_results.log"


def log(msg):
    ts = datetime.now().strftime("%H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line)
    with open(LOG_FILE, "a") as f:
        f.write(line + "\n")


def run_test(name, fn):
    """Run a test function, return True if passed."""
    try:
        fn()
        RESULTS[name] = "PASS"
        log(f"  PASS: {name}")
        return True
    except Exception as e:
        RESULTS[name] = f"FAIL: {e}"
        log(f"  FAIL: {name} — {e}")
        traceback.print_exc()
        return False


# ======================================================================
# Test functions
# ======================================================================

def test_dimensions_registry():
    from persona.dimensions import DIMENSIONS, DIMENSION_SCHEMAS, DIMENSION_EXTRACTION_PROMPTS, PILLAR_TO_DIMENSIONS, create_empty_dimensions
    assert len(DIMENSIONS) == 13
    assert len(DIMENSION_SCHEMAS) == 13
    assert len(DIMENSION_EXTRACTION_PROMPTS) == 13
    assert len(PILLAR_TO_DIMENSIONS) == 5
    dims = create_empty_dimensions()
    assert len(dims) == 13


def test_persona_dimension_dataclass():
    from persona.dimensions import PersonaDimension
    dim = PersonaDimension(name="code", pillar="MIND")
    assert dim.display_name == "Code & Engineering"
    snap = dim.snapshot()
    assert "timestamp" in snap
    dim.update(traits={"languages": ["python"]}, confidence=0.8, evidence_count=50)
    assert dim.confidence == 0.8
    dim.update(traits={"languages": ["python", "rust"]}, confidence=0.9, evidence_count=75)
    assert len(dim.history) == 1
    d = dim.to_dict()
    dim2 = PersonaDimension.from_dict(d)
    assert dim2.confidence == 0.9
    summary = dim.get_summary()
    assert "Code & Engineering" in summary


def test_classifier_type_based():
    from persona.classifier import ChunkClassifier
    c = ChunkClassifier()
    assert c.classify_chunk("x", {"type": "body_gym"}) == ("BODY", "wellness")
    assert c.classify_chunk("x", {"type": "body_nutrition"}) == ("BODY", "nutrition")


def test_classifier_pillar_based():
    from persona.classifier import ChunkClassifier
    c = ChunkClassifier()
    p, d = c.classify_chunk("entry", {"type": "singularity_entry", "pillar": "SOUL"})
    assert p == "SOUL" and d == "creative"


def test_classifier_keyword_based():
    from persona.classifier import ChunkClassifier
    c = ChunkClassifier()
    _, d = c.classify_chunk("deployed React with Docker", {"type": "user_message"})
    assert d == "code"
    _, d = c.classify_chunk("gym squats workout", {"type": "user_message"})
    assert d == "wellness"
    _, d = c.classify_chunk("netflix binge watching", {"type": "user_message"})
    assert d == "entertainment"


def test_classifier_multi_match():
    from persona.classifier import ChunkClassifier
    c = ChunkClassifier()
    dims = c.classify_text("gym workout and meal prep")
    assert "wellness" in dims
    assert "nutrition" in dims


def test_classifier_unclassifiable():
    from persona.classifier import ChunkClassifier
    c = ChunkClassifier()
    _, d = c.classify_chunk("the weather is nice", {"type": "user_message"})
    assert d == ""


def test_chunker_metadata():
    from memory.chunker import _ensure_metadata, DEFAULT_METADATA
    assert "pillar" in DEFAULT_METADATA
    assert "dimension" in DEFAULT_METADATA
    meta = _ensure_metadata({"source": "test"})
    assert meta["pillar"] == ""
    assert meta["classified"] == "false"


def test_chunker_conversations():
    from memory.chunker import Chunker
    chunker = Chunker(chunk_size=500, overlap=50)
    convs = [{
        "source": "test", "conversation_id": "c1", "title": "Test",
        "timestamp": "2024-01-01",
        "messages": [
            {"role": "user", "content": "Hello", "timestamp": ""},
            {"role": "assistant", "content": "Hi", "timestamp": ""},
        ],
    }]
    chunks = chunker.chunk_conversations(convs)
    assert len(chunks) >= 2
    for c in chunks:
        assert "pillar" in c.metadata
        assert "dimension" in c.metadata


def test_vectorstore_search_by_dimension():
    from memory.vectorstore import VectorStore
    vs = VectorStore()
    results = vs.search_by_dimension("code", "code", n_results=5)
    assert isinstance(results, list)


def test_vectorstore_get_unclassified():
    from memory.vectorstore import VectorStore
    vs = VectorStore()
    result = vs.get_unclassified_chunks(limit=10)
    assert "ids" in result


def test_vectorstore_count_by_dimension():
    from memory.vectorstore import VectorStore
    vs = VectorStore()
    count = vs.count_by_dimension("code")
    assert isinstance(count, int)


def test_profile_load():
    from persona.profile import PersonaProfile
    p = PersonaProfile.load()
    assert p.version == 2
    assert len(p.dimensions) == 13


def test_profile_dimensions():
    from persona.profile import PersonaProfile
    p = PersonaProfile.load()
    dim = p.get_dimension("code")
    assert dim is not None
    assert dim.pillar == "MIND"
    assert p.get_dimension("nonexistent") is None


def test_profile_save_load_roundtrip():
    from persona.profile import PersonaProfile
    p = PersonaProfile.load()
    p.save()
    p2 = PersonaProfile.load()
    assert p2.version == 2
    assert len(p2.dimensions) == 13


def test_profile_snapshot():
    from persona.profile import PersonaProfile
    p = PersonaProfile.load()
    snap = p.snapshot_all()
    assert "dimensions" in snap
    snapshots = PersonaProfile.list_snapshots()
    assert len(snapshots) >= 1


def test_profile_migration():
    from persona.profile import PersonaProfile
    from persona.dimensions import create_empty_dimensions
    fresh = PersonaProfile(
        communication_style={"tone": "casual", "formality": "casual", "vocabulary_level": "technical", "sentence_patterns": [], "common_phrases": ["cool"]},
        knowledge_domains=["python"],
    )
    fresh.dimensions = create_empty_dimensions()
    fresh.migrate_from_v1()
    lang = fresh.get_dimension("language_style")
    assert lang.traits["writing_style"] == "casual"


def test_engine_init():
    from twin.engine import TwinEngine
    engine = TwinEngine()
    assert engine.vector_store is not None
    assert engine.persona is not None
    assert engine.classifier is not None


def test_engine_learn():
    from twin.engine import TwinEngine
    engine = TwinEngine()
    result = engine.learn("Test data point for testing")
    assert result["status"] == "learned"


def test_engine_search_memory():
    from twin.engine import TwinEngine
    engine = TwinEngine()
    results = engine.search_memory("python", n_results=3)
    assert isinstance(results, list)


def test_api_persona_ask():
    """Verify the /persona/ask endpoint exists and accepts requests."""
    from api.routes import router
    routes = [r.path for r in router.routes]
    assert "/persona/ask" in routes


def test_prompts_templates():
    from twin.prompts import MEMORY_CONTEXT_TEMPLATE, DIMENSION_CONTEXT_TEMPLATE
    assert "{memories}" in MEMORY_CONTEXT_TEMPLATE
    assert "{dimension_summaries}" in DIMENSION_CONTEXT_TEMPLATE


def test_config_constants():
    from config import PERSONA_VERSION, SNAPSHOTS_DIR, LOGS_DIR, INCREMENTAL_DIMENSION_THRESHOLD
    assert PERSONA_VERSION == 2
    assert SNAPSHOTS_DIR.exists()
    assert LOGS_DIR.exists()
    assert INCREMENTAL_DIMENSION_THRESHOLD == 10


def test_daily_loop_init():
    from daily_loop import DailyLoop
    loop = DailyLoop()
    assert loop.vector_store is not None
    assert loop.classifier is not None
    assert loop.extractor is not None


def test_daily_loop_status():
    import subprocess
    result = subprocess.run(
        [sys.executable, "daily_loop.py", "--status"],
        capture_output=True, text=True, timeout=30
    )
    assert result.returncode == 0
    assert "AI Twin" in result.stdout


def test_api_routes_registered():
    from api.routes import router
    routes = [r.path for r in router.routes]
    required = ["/learn", "/memory/search", "/persona",
                "/sync/run", "/persona/dimensions", "/persona/evolution",
                "/persona/ask", "/wardrobe/sync", "/wardrobe"]
    for path in required:
        assert path in routes, f"Missing route: {path}"


def test_api_memory_search_fields():
    """Verify memory search returns fields the UI expects."""
    from fastapi.testclient import TestClient
    from main import app
    client = TestClient(app)
    r = client.post("/api/memory/search", json={"query": "test", "n_results": 3})
    assert r.status_code == 200
    data = r.json()
    assert "results" in data
    if data["results"]:
        r0 = data["results"][0]
        assert "text" in r0
        assert "metadata" in r0
        meta = r0["metadata"]
        assert "source" in meta
        assert "title" in meta


def test_api_memory_stats():
    from fastapi.testclient import TestClient
    from main import app
    client = TestClient(app)
    r = client.get("/api/memory/stats")
    assert r.status_code == 200
    assert "total_chunks" in r.json()


def test_api_persona_dimensions():
    from fastapi.testclient import TestClient
    from main import app
    client = TestClient(app)
    r = client.get("/api/persona/dimensions")
    assert r.status_code == 200
    assert len(r.json()["dimensions"]) == 13


def test_api_sync_status():
    from fastapi.testclient import TestClient
    from main import app
    client = TestClient(app)
    r = client.get("/api/sync/status")
    assert r.status_code == 200
    assert "connectors" in r.json()
    assert "dimensions" in r.json()


def test_api_persona_evolution():
    from fastapi.testclient import TestClient
    from main import app
    client = TestClient(app)
    r = client.get("/api/persona/evolution")
    assert r.status_code == 200
    assert "snapshots" in r.json()


def test_parsers_gemini_html():
    from parsers.gemini_html_parser import GeminiHTMLParser
    p = GeminiHTMLParser()
    assert p.source == "gemini"


def test_parsers_youtube():
    from parsers.youtube_parser import YouTubeParser
    p = YouTubeParser()
    assert p.source == "youtube"


def test_connectors_all():
    from connectors import ALL_CONNECTORS
    assert len(ALL_CONNECTORS) >= 5
    for name, cls in ALL_CONNECTORS.items():
        c = cls()
        assert hasattr(c, "source_name")
        assert hasattr(c, "fetch")
        assert hasattr(c, "sync")


# --- UI-specific tests ---

def test_ui_syntax():
    import ast
    with open("ui/app.py") as f:
        ast.parse(f.read())


def test_ui_persona_dimensions_fields():
    """Verify /persona/dimensions returns all fields the Persona UI page reads."""
    from fastapi.testclient import TestClient
    from main import app
    client = TestClient(app)
    r = client.get("/api/persona/dimensions")
    assert r.status_code == 200
    dims = r.json()["dimensions"]
    assert len(dims) == 13
    for dim_name, info in dims.items():
        for field in ["display_name", "pillar", "confidence", "evidence_count", "has_traits", "last_updated", "summary"]:
            assert field in info, f"{dim_name} missing {field}"


def test_ui_sync_status_fields():
    """Verify /sync/status returns all fields the Data Sources UI page reads."""
    from fastapi.testclient import TestClient
    from main import app
    client = TestClient(app)
    r = client.get("/api/sync/status")
    assert r.status_code == 200
    data = r.json()
    assert "connectors" in data
    assert "dimensions" in data
    assert "total_chunks" in data
    for name, info in data["connectors"].items():
        assert "last_sync" in info, f"{name} missing last_sync"
        assert "chunks_total" in info, f"{name} missing chunks_total"


def test_ui_singularity_status_fields():
    """Verify /singularity/status returns all fields the Data Sources UI page reads."""
    from fastapi.testclient import TestClient
    from main import app
    client = TestClient(app)
    r = client.get("/api/singularity/status")
    assert r.status_code == 200
    data = r.json()
    assert "sources" in data
    for name, info in data["sources"].items():
        assert "last_sync" in info
        assert "chunks_total" in info


def test_ui_evolution_fields():
    """Verify /persona/evolution returns correct structure for UI."""
    from fastapi.testclient import TestClient
    from main import app
    client = TestClient(app)
    r = client.get("/api/persona/evolution")
    assert r.status_code == 200
    data = r.json()
    assert "snapshots" in data
    if data["snapshots"]:
        snap = data["snapshots"][0]
        assert "date" in snap
        assert "dimensions" in snap


def test_ui_pillar_grouping():
    """Verify all 5 pillar groups are present for the Persona UI."""
    from fastapi.testclient import TestClient
    from main import app
    client = TestClient(app)
    r = client.get("/api/persona/dimensions")
    dims = r.json()["dimensions"]
    pillars_found = set()
    for info in dims.values():
        pillars_found.add(info["pillar"])
    for pillar in ["MIND", "BODY", "SOUL", "SOCIAL", "PURPOSE"]:
        assert pillar in pillars_found, f"Missing pillar: {pillar}"


def test_ui_normalized_files():
    """Check that normalized files exist for Data Sources page."""
    from config import NORMALIZED_DIR
    import json as json_lib
    files = list(NORMALIZED_DIR.glob("*_normalized.json"))
    assert len(files) >= 1, "No normalized files found"
    for f in files:
        data = json_lib.loads(f.read_text())
        assert isinstance(data, list)


def test_ui_learn_endpoint():
    """Verify the Learn page's teach functionality works."""
    from fastapi.testclient import TestClient
    from main import app
    client = TestClient(app)
    r = client.post("/api/learn", json={"data_point": "UI test: prefer dark mode"})
    assert r.status_code == 200
    assert r.json()["status"] == "learned"


def test_ui_chat_reset():
    """Verify chat reset works (Chat page uses this)."""
    from fastapi.testclient import TestClient
    from main import app
    client = TestClient(app)
    r = client.post("/api/chat/reset")
    assert r.status_code == 200


# ======================================================================
# Test runner
# ======================================================================

ALL_TESTS = [
    # Core modules
    ("dimensions_registry", test_dimensions_registry),
    ("persona_dimension_dataclass", test_persona_dimension_dataclass),
    ("classifier_type_based", test_classifier_type_based),
    ("classifier_pillar_based", test_classifier_pillar_based),
    ("classifier_keyword_based", test_classifier_keyword_based),
    ("classifier_multi_match", test_classifier_multi_match),
    ("classifier_unclassifiable", test_classifier_unclassifiable),
    ("chunker_metadata", test_chunker_metadata),
    ("chunker_conversations", test_chunker_conversations),
    ("vectorstore_search_by_dimension", test_vectorstore_search_by_dimension),
    ("vectorstore_get_unclassified", test_vectorstore_get_unclassified),
    ("vectorstore_count_by_dimension", test_vectorstore_count_by_dimension),
    ("profile_load", test_profile_load),
    ("profile_dimensions", test_profile_dimensions),
    ("profile_save_load_roundtrip", test_profile_save_load_roundtrip),
    ("profile_snapshot", test_profile_snapshot),
    ("profile_migration", test_profile_migration),
    ("engine_init", test_engine_init),
    ("engine_learn", test_engine_learn),
    ("engine_search_memory", test_engine_search_memory),
    ("prompts_templates", test_prompts_templates),
    ("config_constants", test_config_constants),
    ("daily_loop_init", test_daily_loop_init),
    ("daily_loop_status", test_daily_loop_status),
    # API endpoints
    ("api_routes_registered", test_api_routes_registered),
    ("api_memory_stats", test_api_memory_stats),
    ("api_memory_search_fields", test_api_memory_search_fields),
    ("api_persona_dimensions", test_api_persona_dimensions),
    ("api_sync_status", test_api_sync_status),
    ("api_persona_evolution", test_api_persona_evolution),
    ("api_persona_ask", test_api_persona_ask),
    ("parsers_gemini_html", test_parsers_gemini_html),
    ("parsers_youtube", test_parsers_youtube),
    ("connectors_all", test_connectors_all),
    # UI-specific tests
    ("ui_syntax", test_ui_syntax),
    ("ui_persona_dimensions_fields", test_ui_persona_dimensions_fields),
    ("ui_sync_status_fields", test_ui_sync_status_fields),
    ("ui_singularity_status_fields", test_ui_singularity_status_fields),
    ("ui_evolution_fields", test_ui_evolution_fields),
    ("ui_pillar_grouping", test_ui_pillar_grouping),
    ("ui_normalized_files", test_ui_normalized_files),
    ("ui_learn_endpoint", test_ui_learn_endpoint),
]


def main():
    max_rounds = 3
    log(f"\n{'='*60}")
    log(f"AI Twin Test Suite — {len(ALL_TESTS)} tests")
    log(f"{'='*60}")

    for round_num in range(1, max_rounds + 1):
        log(f"\n--- Round {round_num}/{max_rounds} ---")
        failed = []

        for name, fn in ALL_TESTS:
            if name in RESULTS and RESULTS[name] == "PASS":
                continue  # already passed
            if not run_test(name, fn):
                failed.append(name)

        passed = sum(1 for v in RESULTS.values() if v == "PASS")
        total = len(ALL_TESTS)
        log(f"\nRound {round_num}: {passed}/{total} passed, {len(failed)} failed")

        if not failed:
            log(f"\nALL {total} TESTS PASSED!")
            break

        if round_num < max_rounds:
            log(f"Retrying {len(failed)} failed tests...")
            time.sleep(2)

    # Final summary
    log(f"\n{'='*60}")
    log("FINAL RESULTS")
    log(f"{'='*60}")
    for name, result in sorted(RESULTS.items()):
        log(f"  {name:40s} {result}")

    passed = sum(1 for v in RESULTS.values() if v == "PASS")
    failed = sum(1 for v in RESULTS.values() if v != "PASS")
    log(f"\n  TOTAL: {passed} passed, {failed} failed out of {len(ALL_TESTS)}")

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
