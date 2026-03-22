import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import streamlit as st
import requests

API_BASE = "http://localhost:8000/api"

st.set_page_config(page_title="AI Twin", page_icon="🧠", layout="wide")


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
st.sidebar.title("AI Twin")
page = st.sidebar.radio("Navigate", ["Chat", "Decide", "Memory", "Persona", "Import Data"])

stats = api_call("get", "/memory/stats")
if stats:
    st.sidebar.metric("Memory Chunks", stats.get("total_chunks", 0))

# --- Learn About Me (sidebar) ---
st.sidebar.markdown("---")
st.sidebar.subheader("Teach Your Twin")
learn_input = st.sidebar.text_area(
    "Tell me something about yourself...",
    placeholder="e.g., 'I prefer async communication over meetings' or 'I always choose Python for data projects'",
    height=80,
    key="learn_input",
)
if st.sidebar.button("Save Data Point") and learn_input:
    result = api_call("post", "/learn", json={"data_point": learn_input})
    if result and result.get("status") == "learned":
        st.sidebar.success("Learned!")


# --- Chat Page ---
if page == "Chat":
    st.title("Chat with your AI Twin")
    st.caption("Your twin remembers everything you've discussed across ChatGPT, Claude, and Gemini.")

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


# --- Decide Page ---
elif page == "Decide":
    st.title("Decision Mode")
    st.caption("Ask a decision question. Your twin will show what YOU would decide vs what's IDEAL.")

    question = st.text_area(
        "What decision are you facing?",
        placeholder="e.g., 'Should I switch from Python to Rust for my backend?' or 'Should I take the remote job or the in-office one?'",
        height=100,
    )

    if st.button("Analyze Decision", type="primary") and question:
        with st.spinner("Analyzing through your lens and the ideal lens..."):
            result = api_call("post", "/decide", json={"question": question})

        if result:
            # Side-by-side decisions
            col1, col2 = st.columns(2)

            with col1:
                st.markdown("### 🎯 Your Likely Decision")
                st.info(result.get("your_decision", "No prediction available"))

            with col2:
                st.markdown("### ⭐ Ideal Decision")
                st.success(result.get("ideal_decision", "No ideal available"))

            # Gap analysis
            gap = result.get("reasoning_gap", "")
            if gap:
                st.markdown("### 🔍 Gap Analysis")
                st.warning(gap)

            # Confidence
            confidence = result.get("confidence_score", "")
            if confidence:
                st.markdown("### 📊 Confidence")
                st.caption(confidence)

            # Follow-up questions
            follow_ups = result.get("follow_up_questions", [])
            if follow_ups:
                st.markdown("### 💡 Help me understand you better")
                st.caption("Answer these to improve future predictions:")

                for i, q in enumerate(follow_ups):
                    answer = st.text_input(q, key=f"followup_{i}")
                    if answer:
                        if st.button(f"Submit answer #{i+1}", key=f"submit_{i}"):
                            data_point = f"Q: {q} A: {answer}"
                            learn_result = api_call("post", "/learn", json={"data_point": data_point})
                            if learn_result and learn_result.get("status") == "learned":
                                st.success("Learned! This will improve future decisions.")


# --- Memory Page ---
elif page == "Memory":
    st.title("Memory Browser")
    st.caption("Search through everything you've ever discussed with AI.")

    query = st.text_input("Search your memory...", placeholder="e.g., 'Python project ideas' or 'that bug I fixed last month'")
    n_results = st.slider("Results", 5, 50, 10)

    if query:
        with st.spinner("Searching memories..."):
            results = api_call("post", "/memory/search", json={"query": query, "n_results": n_results})

        if results and results.get("results"):
            for i, r in enumerate(results["results"]):
                meta = r.get("metadata", {})
                source = meta.get("source", "unknown")
                title = meta.get("title", "Untitled")
                timestamp = meta.get("timestamp", "")
                distance = r.get("distance", 0)
                relevance = max(0, round((1 - distance) * 100))
                mem_type = meta.get("type", "")
                badge = "📝" if mem_type == "data_point" else "💬"

                with st.expander(f"{badge} [{source}] {title} — {relevance}% relevant"):
                    st.caption(f"Source: {source} | Type: {mem_type} | Date: {timestamp}")
                    st.write(r["text"])
        else:
            st.info("No memories found. Try a different search or ingest more data.")


# --- Persona Page ---
elif page == "Persona":
    st.title("Your Persona Profile")
    st.caption("This is how your AI twin understands you.")

    persona = api_call("get", "/persona")

    if persona:
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Communication Style")
            style = persona.get("communication_style", {})
            if style:
                st.write(f"**Tone:** {style.get('tone', 'Not analyzed yet')}")
                st.write(f"**Formality:** {style.get('formality', 'N/A')}")
                st.write(f"**Vocabulary:** {style.get('vocabulary_level', 'N/A')}")
                if style.get("common_phrases"):
                    st.write("**Common phrases:**")
                    for phrase in style["common_phrases"]:
                        st.write(f"- _{phrase}_")

            st.subheader("Knowledge Domains")
            for domain in persona.get("knowledge_domains", []):
                st.write(f"- {domain}")

            st.subheader("Interests")
            for interest in persona.get("interests", []):
                st.write(f"- {interest}")

        with col2:
            st.subheader("Decision Patterns")
            for pattern in persona.get("decision_patterns", []):
                st.write(f"- {pattern}")

            st.subheader("Cognitive Biases")
            biases = persona.get("cognitive_biases", [])
            if biases:
                for bias in biases:
                    st.write(f"- {bias}")
            else:
                st.caption("Not yet analyzed. Ingest data and extract persona.")

            st.subheader("Risk & Time Profile")
            st.write(f"**Risk tolerance:** {persona.get('risk_tolerance', 'N/A')}")
            st.write(f"**Time preference:** {persona.get('time_preference', 'N/A')}")

            st.subheader("Values & Priorities")
            for value in persona.get("values_and_priorities", []):
                st.write(f"- {value}")

        # Decision history
        history = persona.get("decision_history", [])
        if history:
            st.markdown("---")
            st.subheader("Decision History")
            for d in history[-10:]:
                st.write(f"**Q:** {d['question']}")
                st.write(f"**Decided:** {d['decision']}")
                if d.get("outcome"):
                    st.write(f"**Outcome:** {d['outcome']}")
                st.markdown("---")

        with st.expander("System Prompt (used by your twin)"):
            st.code(persona.get("system_prompt", "Not generated yet"), language=None)

        if st.button("Re-extract Persona"):
            with st.spinner("Analyzing your messages..."):
                result = api_call("post", "/persona/extract")
                if result and result.get("status") == "success":
                    st.success(f"Persona updated! Analyzed {result['messages_analyzed']} messages.")
                    st.rerun()


# --- Import Page ---
elif page == "Import Data":
    st.title("Import Conversation Data")
    st.caption("Upload your exports from ChatGPT, Claude, or Gemini.")

    st.markdown("""
    ### How to export your data

    **ChatGPT:**
    1. Go to chatgpt.com → Settings → Data Controls → Export Data
    2. Click Export → confirm via email → download ZIP
    3. Extract and upload `conversations.json` below

    **Claude:**
    1. Go to claude.ai → Settings → Account → Export Data
    2. Download and upload the JSON file below

    **Gemini:**
    1. Go to takeout.google.com
    2. Select only "Gemini Apps" → Export
    3. Extract and upload the JSON files below
    """)

    source = st.selectbox("Platform", ["chatgpt", "claude", "gemini"])
    uploaded_file = st.file_uploader("Upload export file (JSON)", type=["json"])

    if uploaded_file and st.button("Ingest Data"):
        with st.spinner(f"Parsing and ingesting {source} data..."):
            files = {"file": (uploaded_file.name, uploaded_file.getvalue(), "application/json")}
            data = {"source": source}
            result = api_call("post", "/ingest", files=files, data=data)

            if result and result.get("status") == "success":
                st.success(
                    f"Ingested {result['conversations_parsed']} conversations "
                    f"({result['chunks_ingested']} memory chunks)"
                )
                st.info("Now go to the Persona page and click 'Re-extract Persona' to update your twin's personality.")
            elif result:
                st.error(result.get("error", "Unknown error"))
