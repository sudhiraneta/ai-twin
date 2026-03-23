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
page = st.sidebar.radio("Navigate", ["Chat", "Decide", "Persona", "Data Sources", "Learn"])

stats = api_call("get", "/memory/stats")
if stats:
    st.sidebar.metric("Memory Chunks", stats.get("total_chunks", 0))

st.sidebar.markdown("---")

# Quick teach in sidebar
st.sidebar.subheader(f"Quick Teach")
learn_input = st.sidebar.text_area(
    "Tell me something...",
    placeholder="e.g., 'I prefer async communication'",
    height=60,
    key="learn_input",
)
if st.sidebar.button("Save") and learn_input:
    result = api_call("post", "/learn", json={"data_point": learn_input})
    if result and result.get("status") == "learned":
        st.sidebar.success("Learned!")


# ======================================================================
# Chat Page
# ======================================================================
if page == "Chat":
    st.title(f"Chat with {TWIN_NAME}")
    st.caption("Your twin remembers everything — conversations, notes, browsing, tasks, and more.")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

    if prompt := st.chat_input("Ask your twin anything..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                result = api_call("post", "/chat", json={"message": prompt})
                if result:
                    response = result["response"]
                    st.write(response)
                    st.session_state.messages.append({"role": "assistant", "content": response})

    if st.sidebar.button("Reset Conversation"):
        api_call("post", "/chat/reset")
        st.session_state.messages = []
        st.rerun()


# ======================================================================
# Decide Page
# ======================================================================
elif page == "Decide":
    st.title("Decision Mode")
    st.caption("Ask a decision question. Your twin shows what YOU'd decide vs what's IDEAL.")

    question = st.text_area(
        "What decision are you facing?",
        placeholder="e.g., 'Should I switch from Python to Rust for my backend?'",
        height=100,
    )

    if st.button("Analyze Decision", type="primary") and question:
        with st.spinner("Analyzing through your lens and the ideal lens..."):
            result = api_call("post", "/decide", json={"question": question})

        if result:
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("### Your Likely Decision")
                st.info(result.get("your_decision", "No prediction available"))
            with col2:
                st.markdown("### Ideal Decision")
                st.success(result.get("ideal_decision", "No ideal available"))

            gap = result.get("reasoning_gap", "")
            if gap:
                st.markdown("### Gap Analysis")
                st.warning(gap)

            confidence = result.get("confidence_score", "")
            if confidence:
                st.markdown("### Confidence")
                st.caption(confidence)

            follow_ups = result.get("follow_up_questions", [])
            if follow_ups:
                st.markdown("### Help me understand you better")
                for i, q in enumerate(follow_ups):
                    answer = st.text_input(q, key=f"followup_{i}")
                    if answer:
                        if st.button(f"Submit", key=f"submit_{i}"):
                            api_call("post", "/learn", json={"data_point": f"Q: {q} A: {answer}"})
                            st.success("Learned!")


# ======================================================================
# Persona Page — Multi-Dimensional Skill Files (read-only)
# ======================================================================
elif page == "Persona":
    st.title("Your Persona Profile")
    st.caption(f"How {TWIN_NAME} understands you across 13 life dimensions. Read-only — updated by the daily learning loop.")

    # Fetch dimensions
    dims_data = api_call("get", "/persona/dimensions")
    persona = api_call("get", "/persona")

    if dims_data and dims_data.get("dimensions"):
        dimensions = dims_data["dimensions"]

        # Summary metrics
        populated = sum(1 for d in dimensions.values() if d.get("has_traits"))
        total_evidence = sum(d.get("evidence_count", 0) for d in dimensions.values())
        avg_confidence = sum(d.get("confidence", 0) for d in dimensions.values() if d.get("has_traits"))
        if populated > 0:
            avg_confidence /= populated

        col1, col2, col3 = st.columns(3)
        col1.metric("Dimensions Populated", f"{populated}/13")
        col2.metric("Total Evidence", f"{total_evidence} chunks")
        col3.metric("Avg Confidence", f"{avg_confidence:.0%}")

        st.markdown("---")

        # Group by pillar
        PILLAR_ORDER = ["MIND", "BODY", "SOUL", "SOCIAL", "PURPOSE"]
        PILLAR_ICONS = {"MIND": "🧠", "BODY": "💪", "SOUL": "🎨", "SOCIAL": "🤝", "PURPOSE": "🎯"}

        # Build pillar groups
        pillar_groups = {}
        for dim_name, dim_info in dimensions.items():
            pillar = dim_info.get("pillar", "OTHER")
            pillar_groups.setdefault(pillar, []).append((dim_name, dim_info))

        # Render each pillar as a section
        for pillar in PILLAR_ORDER:
            if pillar not in pillar_groups:
                continue

            icon = PILLAR_ICONS.get(pillar, "")
            st.subheader(f"{icon} {pillar}")

            dims_in_pillar = pillar_groups[pillar]
            cols = st.columns(len(dims_in_pillar))

            for col, (dim_name, dim_info) in zip(cols, dims_in_pillar):
                with col:
                    display = dim_info.get("display_name", dim_name)
                    confidence = dim_info.get("confidence", 0)
                    evidence = dim_info.get("evidence_count", 0)
                    has_traits = dim_info.get("has_traits", False)
                    updated = dim_info.get("last_updated", "")

                    # Status indicator
                    if has_traits and confidence >= 0.5:
                        status_color = "🟢"
                    elif has_traits:
                        status_color = "🟡"
                    else:
                        status_color = "🔴"

                    st.markdown(f"**{status_color} {display}**")
                    st.caption(f"Confidence: {confidence:.0%} | Evidence: {evidence}")
                    if updated:
                        st.caption(f"Updated: {updated[:10]}")

                    # Show traits in expander (read-only)
                    if has_traits:
                        summary = dim_info.get("summary", "")
                        if summary:
                            with st.expander("View traits"):
                                st.markdown(summary)

        # System prompt (read-only)
        if persona:
            st.markdown("---")
            with st.expander("System Prompt (generated from all dimensions)"):
                st.code(persona.get("system_prompt", "Not generated yet"), language=None)

        # Evolution
        st.markdown("---")
        st.subheader("Persona Evolution")
        evolution = api_call("get", "/persona/evolution")
        if evolution and evolution.get("snapshots"):
            snapshots = evolution["snapshots"]
            st.caption(f"{len(snapshots)} snapshots recorded")
            for snap in reversed(snapshots[-5:]):
                date = snap.get("date", "")
                dims = snap.get("dimensions", {})
                populated = sum(1 for d in dims.values() if d.get("traits"))
                st.write(f"**{date}**: {populated}/13 dimensions populated")
        else:
            st.caption("No evolution snapshots yet. Run the daily loop to start tracking.")


# ======================================================================
# Data Sources Page — Last synced / imported (read-only dashboard)
# ======================================================================
elif page == "Data Sources":
    st.title("Data Sources")
    st.caption("Track what data has been fed to your twin and when. Read-only overview.")

    status = api_call("get", "/sync/status")
    singularity_status = api_call("get", "/singularity/status")

    if status:
        total = status.get("total_chunks", 0)
        st.metric("Total Memory Chunks", total)
        st.markdown("---")

    # --- Singularity Connectors ---
    st.subheader("Singularity Sources (live sync)")
    st.caption("These sync incrementally from your Singularity project.")

    if singularity_status and singularity_status.get("sources"):
        sources = singularity_status["sources"]

        SOURCE_ICONS = {
            "apple_notes": "📒",
            "singularity_db": "⚡",
            "browser": "🌐",
            "tasks": "📋",
            "body": "💪",
            "analytics": "📊",
        }

        cols = st.columns(3)
        for i, (name, info) in enumerate(sources.items()):
            with cols[i % 3]:
                icon = SOURCE_ICONS.get(name, "📦")
                last = info.get("last_sync")
                total = info.get("chunks_total", 0)

                if last:
                    last_str = dt.fromtimestamp(last).strftime("%Y-%m-%d %H:%M")
                    age_hours = (dt.now().timestamp() - last) / 3600
                    if age_hours < 1:
                        freshness = "🟢 Just synced"
                    elif age_hours < 24:
                        freshness = "🟡 Today"
                    else:
                        freshness = f"🔴 {int(age_hours / 24)}d ago"
                else:
                    last_str = "Never"
                    freshness = "⚪ Not synced"

                st.markdown(f"**{icon} {name}**")
                st.write(f"Chunks: **{total}**")
                st.write(f"Last sync: {last_str}")
                st.caption(freshness)

    st.markdown("---")

    # --- Conversation Imports ---
    st.subheader("Conversation Imports (file uploads)")
    st.caption("ChatGPT, Claude, and Gemini conversation exports.")

    # Check what normalized files exist
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
                    conv_count = len(data) if isinstance(data, list) else 0
                    modified = dt.fromtimestamp(f.stat().st_mtime).strftime("%Y-%m-%d %H:%M")
                except Exception:
                    conv_count = 0
                    modified = "unknown"

                st.markdown(f"**{icon} {platform.title()}**")
                st.write(f"Conversations: **{conv_count}**")
                st.write(f"Imported: {modified}")
    else:
        st.info("No conversation imports yet. Go to **Learn** to import data.")

    st.markdown("---")

    # --- Dimension Coverage ---
    st.subheader("Dimension Coverage")
    st.caption("How well each persona dimension is covered by data.")

    if status and status.get("dimensions"):
        dims = status["dimensions"]
        for dim_name, info in sorted(dims.items(), key=lambda x: -x[1].get("evidence_count", 0)):
            display = info.get("display_name", dim_name)
            evidence = info.get("evidence_count", 0)
            confidence = info.get("confidence", 0)
            has_traits = info.get("has_traits", False)

            # Progress bar
            bar_value = min(1.0, evidence / 100)  # 100 chunks = full bar
            status_icon = "🟢" if has_traits and confidence >= 0.5 else "🟡" if has_traits else "🔴"

            col1, col2 = st.columns([3, 1])
            with col1:
                st.progress(bar_value, text=f"{status_icon} {display}: {evidence} chunks")
            with col2:
                st.caption(f"{confidence:.0%} confidence")

    # --- Search Memory ---
    st.markdown("---")
    st.subheader("Search Memory")

    search_col1, search_col2, search_col3 = st.columns([3, 1, 1])
    with search_col1:
        query = st.text_input("Search...", placeholder="e.g., 'Python projects' or 'gym this week'")
    with search_col2:
        dim_options = ["All dimensions"] + sorted(dims.keys()) if status and status.get("dimensions") else ["All dimensions"]
        dim_filter = st.selectbox("Filter by dimension", dim_options, key="search_dim_filter")
    with search_col3:
        n_results = st.slider("Results", 5, 30, 10, key="search_n")

    if query:
        search_body = {"query": query, "n_results": n_results}
        if dim_filter != "All dimensions":
            search_body["dimension"] = dim_filter

        with st.spinner("Searching..."):
            results = api_call("post", "/memory/search", json=search_body)

        if results and results.get("results"):
            for r in results["results"]:
                meta = r.get("metadata", {})
                source = meta.get("source", "?")
                dim = meta.get("dimension", "?")
                title = meta.get("title", "")
                ts = meta.get("timestamp", "")
                ts_short = ts[:10] if ts else ""
                mem_type = meta.get("type", "")
                distance = r.get("distance", 0)
                relevance = max(0, round((1 - distance) * 100))

                TYPE_ICONS = {
                    "user_message": "💬", "conversation_pair": "💬",
                    "data_point": "📝", "singularity_entry": "⚡",
                    "note": "📒", "browser_daily": "🌐", "browser_domain": "🔗",
                    "task": "📋", "body_gym": "💪", "body_nutrition": "🥗",
                    "weekly_review": "📊", "soul_checkin": "🌙",
                    "goals_completed": "🏆", "plan_note": "📅",
                    "pillar_journal": "📓",
                }
                icon = TYPE_ICONS.get(mem_type, "📄")

                with st.expander(f"{icon} [{dim}] {title or r['text'][:50]} — {relevance}% match"):
                    st.caption(f"Source: {source} | Type: {mem_type} | Dimension: {dim} | Date: {ts_short}")
                    st.write(r["text"])
        else:
            st.info("No results found. Try a different search.")


# ======================================================================
# Learn Page — Sync + Import + Teach
# ======================================================================
elif page == "Learn":
    st.title(f"Teach {TWIN_NAME}")
    st.caption("Sync data, import conversations, or tell your twin something. Everything builds who your twin is.")

    # --- Quick Sync ---
    st.subheader("Sync My Data")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Sync Everything", type="primary"):
            with st.spinner("Syncing all data sources..."):
                result = api_call("post", "/singularity/sync")
            if result and result.get("sources"):
                total = sum(v.get("chunks_ingested", 0) for v in result["sources"].values() if "error" not in v)
                st.success(f"Synced {total} new data points.")

    with col2:
        if st.button("Run Full Learning Loop"):
            with st.spinner("Running daily loop (sync + classify + persona)..."):
                result = api_call("post", "/sync/run")
            if result:
                st.success("Learning loop complete!")

    st.markdown("---")

    # --- Import Conversations ---
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

    # --- Teach directly ---
    st.subheader("Tell Your Twin Something")
    teach_input = st.text_area(
        "What should your twin know?",
        placeholder="e.g., 'I switched to a standing desk' or 'I prefer cortados over lattes'",
        height=80,
    )
    if st.button("Save Data Point") and teach_input:
        result = api_call("post", "/learn", json={"data_point": teach_input})
        if result and result.get("status") == "learned":
            st.success("Got it!")
