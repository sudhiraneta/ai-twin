import sys
from datetime import datetime as dt
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import streamlit as st
import requests

from config import TWIN_NAME

API_BASE = "http://localhost:8000/api"

st.set_page_config(page_title=TWIN_NAME, page_icon="🧠", layout="wide")


def api_call(method: str, endpoint: str, **kwargs) -> dict | None:
    try:
        url = f"{API_BASE}{endpoint}"
        resp = getattr(requests, method)(url, **kwargs)
        resp.raise_for_status()
        return resp.json()
    except requests.exceptions.ConnectionError:
        st.error("Cannot connect to API. Run: `python main.py` first.")
        return None
    except Exception as e:
        st.error(f"API error: {e}")
        return None


# --- Sidebar ---
st.sidebar.title(TWIN_NAME)
page = st.sidebar.radio("Navigate", ["Ask Persona", "Persona", "Wardrobe", "Data", "Learn"])

stats = api_call("get", "/memory/stats")
if stats:
    st.sidebar.metric("Memory Chunks", stats.get("total_chunks", 0))
    by_dim = stats.get("by_dimension", {})
    if by_dim:
        top = sorted(by_dim.items(), key=lambda x: -x[1])[:5]
        st.sidebar.caption("Top dimensions: " + ", ".join(f"{d} ({c})" for d, c in top))

st.sidebar.markdown("---")
st.sidebar.subheader("Quick Teach")
learn_input = st.sidebar.text_area("Tell me something...", placeholder="e.g., 'I prefer async communication'", height=60, key="learn_input")
if st.sidebar.button("Save") and learn_input:
    result = api_call("post", "/learn", json={"data_point": learn_input})
    if result and result.get("status") == "learned":
        st.sidebar.success("Learned!")


# ======================================================================
# Ask Persona — Shows fetched context (top-k) + generated LLM answer
# ======================================================================
if page == "Ask Persona":
    st.title("Ask About Me")
    st.caption("Ask anything about yourself. See the raw context fetched from memory (top-k chunks) and the generated answer side by side.")

    query = st.text_input("What do you want to know?", placeholder="e.g., 'What are my coding habits?' or 'What do I eat?'")

    col_filter1, col_filter2 = st.columns(2)
    with col_filter1:
        dim_options = ["All dimensions"]
        dims_data = api_call("get", "/persona/dimensions")
        if dims_data:
            dim_options += sorted(dims_data["dimensions"].keys())
        dim_filter = st.selectbox("Filter by dimension", dim_options, key="ask_dim")
    with col_filter2:
        top_k = st.slider("Top-K context chunks", 3, 30, 10, key="ask_topk")

    if query and st.button("Ask", type="primary"):
        # Fetch raw context chunks
        search_body = {"query": query, "n_results": top_k}
        if dim_filter != "All dimensions":
            search_body["dimension"] = dim_filter

        with st.spinner("Fetching context..."):
            search_results = api_call("post", "/memory/search", json=search_body)

        context_chunks = []
        if search_results and search_results.get("results"):
            context_chunks = search_results["results"]

        # Build context text for LLM
        context_text = ""
        for i, r in enumerate(context_chunks):
            meta = r.get("metadata", {})
            source = meta.get("source", "?")
            dim = meta.get("dimension", "?")
            ts = meta.get("timestamp", "")[:10]
            title = meta.get("title", "")
            context_text += f"[{source}/{dim} | {title} | {ts}]\n{r['text']}\n\n---\n\n"

        # Generate LLM answer using the context
        llm_answer = ""
        if context_text:
            with st.spinner("Generating answer from context..."):
                answer_result = api_call("post", "/persona/ask", json={
                    "query": query,
                    "context": context_text,
                })
                if answer_result:
                    llm_answer = answer_result.get("answer", "")

        # Display side by side
        st.markdown("---")

        left, right = st.columns([1, 1])

        with left:
            st.subheader(f"Fetched Context ({len(context_chunks)} chunks)")
            if context_chunks:
                for i, r in enumerate(context_chunks):
                    meta = r.get("metadata", {})
                    source = meta.get("source", "?")
                    dim = meta.get("dimension", "?")
                    mem_type = meta.get("type", "?")
                    title = meta.get("title", "")
                    ts = meta.get("timestamp", "")[:10]
                    distance = r.get("distance", 0)
                    relevance = max(0, round((1 - distance) * 100))

                    TYPE_ICONS = {
                        "user_message": "💬", "conversation_pair": "💬",
                        "data_point": "📝", "singularity_entry": "⚡",
                        "note": "📒", "browser_daily": "🌐", "browser_domain": "🔗",
                        "task": "📋", "body_gym": "💪", "body_nutrition": "🥗",
                        "weekly_review": "📊", "soul_checkin": "🌙",
                        "goals_completed": "🏆", "plan_note": "📅",
                        "pillar_journal": "📓", "photo_daily": "📸",
                    }
                    icon = TYPE_ICONS.get(mem_type, "📄")

                    with st.expander(f"{icon} #{i+1} [{dim}] {relevance}% — {title[:40] or r['text'][:40]}"):
                        st.caption(f"Source: {source} | Type: {mem_type} | Dimension: {dim} | Date: {ts}")
                        st.write(r["text"])
            else:
                st.info("No context found for this query.")

        with right:
            st.subheader("Generated Answer")
            if llm_answer:
                st.markdown(llm_answer)
            elif not context_chunks:
                st.info("No data to generate an answer from.")
            else:
                st.warning("LLM answer not available. Check API connection.")


# ======================================================================
# Persona Page — Editable Skill Files
# ======================================================================
elif page == "Persona":
    st.title("Persona Skill Files")
    st.caption(f"Each dimension is a live .md file defining who {TWIN_NAME} thinks you are + where to find data. You can edit them.")

    # Fetch skill files
    skills_data = api_call("get", "/persona/skills")

    if skills_data and skills_data.get("skills"):
        skills = skills_data["skills"]

        # Summary metrics
        populated = sum(1 for s in skills.values() if "No data yet" not in s.get("content", ""))
        user_edited = sum(1 for s in skills.values() if s.get("user_edited"))

        col1, col2, col3 = st.columns(3)
        col1.metric("Dimensions", f"{populated}/13 populated")
        col2.metric("User Edited", user_edited)
        col3.metric("Total Skills", len(skills))

        st.markdown("---")

        # Group by pillar
        PILLAR_ORDER = ["MIND", "BODY", "SOUL", "SOCIAL", "PURPOSE"]
        PILLAR_ICONS = {"MIND": "🧠", "BODY": "💪", "SOUL": "🎨", "SOCIAL": "🤝", "PURPOSE": "🎯"}

        pillar_groups = {}
        for name, info in skills.items():
            pillar = info.get("pillar", "OTHER")
            pillar_groups.setdefault(pillar, []).append((name, info))

        for pillar in PILLAR_ORDER:
            if pillar not in pillar_groups:
                continue

            icon = PILLAR_ICONS.get(pillar, "")
            st.subheader(f"{icon} {pillar}")

            for dim_name, info in pillar_groups[pillar]:
                display = info.get("display", dim_name)
                content = info.get("content", "")
                description = info.get("description", "")
                is_edited = info.get("user_edited", False)

                badge = " (edited)" if is_edited else ""
                has_data = "No data yet" not in content

                if has_data:
                    status = "🟢"
                else:
                    status = "🔴"

                with st.expander(f"{status} {display}{badge} — {description}"):
                    # Editable text area
                    edited = st.text_area(
                        f"Edit {dim_name}.md",
                        value=content,
                        height=400,
                        key=f"skill_{dim_name}",
                    )

                    if edited != content:
                        if st.button(f"Save {dim_name}", key=f"save_{dim_name}"):
                            result = api_call("put", f"/persona/skills/{dim_name}", json={"content": edited})
                            if result and result.get("status") == "saved":
                                st.success(f"Saved! Twin persona updated.")
                                st.rerun()

        st.markdown("---")

        # System prompt preview
        with st.expander("Composed System Prompt (auto-built from all skill files)"):
            persona = api_call("get", "/persona")
            if persona:
                st.code(persona.get("system_prompt", "Not generated yet"), language=None)


# ======================================================================
# Wardrobe Page
# ======================================================================
elif page == "Wardrobe":
    st.title("Digital Wardrobe & Travel")
    st.caption("Your style, outfits, and places visited — powered by Google Photos.")

    col1, col2 = st.columns(2)
    with col1:
        if st.button("Sync Google Photos", type="primary"):
            with st.spinner("Syncing photo metadata..."):
                result = api_call("post", "/wardrobe/sync")
            if result and result.get("status") == "success":
                st.success(f"Synced {result['photos_synced']} photo chunks!")
            elif result and result.get("error"):
                st.error(result["error"])
    with col2:
        st.caption("Only pulls metadata (dates, locations, tags) — no image downloads.")

    st.markdown("---")

    wardrobe = api_call("get", "/wardrobe")
    if wardrobe:
        col1, col2, col3 = st.columns(3)
        col1.metric("Photos Indexed", wardrobe.get("total_photos_indexed", 0))
        col2.metric("Outfits", len(wardrobe.get("outfits", [])))
        col3.metric("Travel Spots", len(wardrobe.get("travel", [])))

        st.markdown("---")

        if wardrobe.get("outfits"):
            st.subheader("Outfit History")
            for item in wardrobe["outfits"]:
                with st.expander(f"{item.get('date', '?')}"):
                    st.write(item["text"])

        if wardrobe.get("travel"):
            st.subheader("Places Visited")
            for item in wardrobe["travel"]:
                with st.expander(f"{item.get('date', '?')}"):
                    st.write(item["text"])

        if wardrobe.get("food"):
            st.subheader("Food & Dining")
            for item in wardrobe["food"]:
                with st.expander(f"{item.get('date', '?')}"):
                    st.write(item["text"])

        if not wardrobe.get("outfits") and not wardrobe.get("travel"):
            st.info("No photos synced yet. Click 'Sync Google Photos' above.")

    with st.expander("Google Photos Albums"):
        albums = api_call("get", "/wardrobe/albums")
        if albums and albums.get("albums"):
            for album in albums["albums"]:
                st.write(f"**{album.get('title', 'Untitled')}** — {album.get('count', 0)} photos")
        else:
            st.caption("No albums found or not authenticated.")


# ======================================================================
# Data Page — Sources dashboard + search
# ======================================================================
elif page == "Data":
    st.title("Data Sources")
    st.caption("What data feeds your twin and when.")

    status = api_call("get", "/sync/status")
    singularity_status = api_call("get", "/singularity/status")

    if status:
        st.metric("Total Memory Chunks", status.get("total_chunks", 0))
        st.markdown("---")

    # Singularity sources
    st.subheader("Live Sources (Singularity)")
    if singularity_status and singularity_status.get("sources"):
        SOURCE_ICONS = {"apple_notes": "📒", "singularity_db": "⚡", "browser": "🌐", "tasks": "📋", "body": "💪", "analytics": "📊"}
        cols = st.columns(3)
        for i, (name, info) in enumerate(singularity_status["sources"].items()):
            with cols[i % 3]:
                icon = SOURCE_ICONS.get(name, "📦")
                last = info.get("last_sync")
                total = info.get("chunks_total", 0)
                last_str = dt.fromtimestamp(last).strftime("%Y-%m-%d %H:%M") if last else "Never"
                st.markdown(f"**{icon} {name}**")
                st.write(f"Chunks: **{total}** | Last: {last_str}")

    st.markdown("---")

    # Conversation imports
    st.subheader("Conversation Imports")
    from config import NORMALIZED_DIR
    import json as json_lib
    IMPORT_ICONS = {"chatgpt": "🤖", "claude": "🟣", "gemini": "🔵", "youtube": "🎬"}
    normalized_files = list(NORMALIZED_DIR.glob("*_normalized.json"))
    if normalized_files:
        cols = st.columns(min(len(normalized_files), 4))
        for i, f in enumerate(sorted(normalized_files)):
            with cols[i % len(cols)]:
                platform = f.stem.replace("_normalized", "")
                icon = IMPORT_ICONS.get(platform, "📄")
                try:
                    data = json_lib.loads(f.read_text())
                    count = len(data) if isinstance(data, list) else 0
                    modified = dt.fromtimestamp(f.stat().st_mtime).strftime("%Y-%m-%d %H:%M")
                except Exception:
                    count = 0
                    modified = "?"
                st.markdown(f"**{icon} {platform.title()}**")
                st.write(f"Conversations: **{count}** | Imported: {modified}")
    else:
        st.info("No imports yet. Go to Learn to import data.")



# ======================================================================
# Learn Page — Sync + Import + Teach
# ======================================================================
elif page == "Learn":
    st.title(f"Teach {TWIN_NAME}")
    st.caption("Sync data, import conversations, or tell your twin something.")

    st.subheader("Sync My Data")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Sync Everything", type="primary"):
            with st.spinner("Syncing..."):
                result = api_call("post", "/singularity/sync")
            if result and result.get("sources"):
                total = sum(v.get("chunks_ingested", 0) for v in result["sources"].values() if "error" not in v)
                st.success(f"Synced {total} new data points.")
    with col2:
        if st.button("Run Full Learning Loop"):
            with st.spinner("Running daily loop..."):
                result = api_call("post", "/sync/run")
            if result:
                st.success("Learning loop complete!")

    st.markdown("---")

    st.subheader("Import Conversations")
    source = st.selectbox("Platform", ["chatgpt", "claude", "gemini"])
    uploaded_file = st.file_uploader("Upload export file (JSON)", type=["json"])
    if uploaded_file and st.button("Import"):
        with st.spinner(f"Parsing {source} data..."):
            files = {"file": (uploaded_file.name, uploaded_file.getvalue(), "application/json")}
            data = {"source": source}
            result = api_call("post", "/ingest", files=files, data=data)
            if result and result.get("status") == "success":
                st.success(f"Imported {result['conversations_parsed']} conversations ({result['chunks_ingested']} chunks)")

    st.markdown("---")

    st.subheader("Tell Your Twin Something")
    teach_input = st.text_area("What should your twin know?", placeholder="e.g., 'I prefer cortados over lattes'", height=80)
    if st.button("Save Data Point") and teach_input:
        result = api_call("post", "/learn", json={"data_point": teach_input})
        if result and result.get("status") == "learned":
            st.success("Got it!")
