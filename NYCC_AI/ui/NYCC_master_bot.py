"""
NYCC Master Bot — polished, council-staff-facing UI for the multi-tool agent.

Wraps the exact same agent as agent_chatbot.py (MCPToolClient, answer_query_agentic,
_format_tool_catalog) — no changes to that file. Unlike agent_chatbot_ui.py, this
shows ONLY the conversation (no tool-call badges/expanders), with an inline feedback
row (thumbs up/down + optional comment) attached under every response — deliberately
not a modal, so it never covers the response itself. Each feedback submission is
saved as its own JSON file in feedback/. Run with: streamlit run "ui/NYCC_master_bot.py"
"""

import asyncio
import json
import os
import tempfile
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

import requests
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "orchestrators"))
from agent_chatbot import MCP_SERVER_SCRIPT, MCPToolClient, _format_tool_catalog, answer_query_agentic

# Overridable, and never fatal: on a hosted container the app directory is often
# not writable by the runtime user, and losing feedback storage must not take the
# whole assistant down with it.
FEEDBACK_DIR = Path(os.getenv("FEEDBACK_DIR", Path(__file__).resolve().parent.parent / "feedback"))
try:
    FEEDBACK_DIR.mkdir(parents=True, exist_ok=True)
except OSError:
    FEEDBACK_DIR = Path(tempfile.gettempdir()) / "nycc_feedback"
    FEEDBACK_DIR.mkdir(parents=True, exist_ok=True)


def _friendly_error(e: BaseException) -> str:
    """MCP wraps errors in nested TaskGroups (anyio), so a plain 'connection
    refused' from local_ai.py not running surfaces as an opaque ExceptionGroup."""
    root = e
    while isinstance(root, BaseExceptionGroup) and root.exceptions:
        root = root.exceptions[0]

    if isinstance(root, requests.exceptions.ConnectionError):
        return "The assistant is temporarily unavailable. Please try again in a moment."

    return "Something went wrong while processing your request. Please try again."


def _save_feedback(query: str, response: str, rating: str, comment: str) -> None:
    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "query": query,
        "response": response,
        "rating": rating,
        "comment": comment,
    }
    filename = f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%f')}_{uuid.uuid4().hex[:8]}.json"
    with open(FEEDBACK_DIR / filename, "w") as f:
        json.dump(record, f, indent=2)


st.set_page_config(
    page_title="NYCC Master Bot",
    page_icon="🏛️",
    layout="centered",
)

st.markdown(
    """
    <style>
    .stApp {
        background: linear-gradient(180deg, #f7f8fb 0%, #eef1f6 100%);
    }
    .master-header {
        background: linear-gradient(135deg, #0c2340 0%, #16345e 55%, #1c4270 100%);
        padding: 2.25rem 2.5rem;
        border-radius: 18px;
        margin-bottom: 1.75rem;
        box-shadow: 0 10px 30px rgba(12, 35, 64, 0.28);
        position: relative;
        overflow: hidden;
    }
    .master-header::after {
        content: "";
        position: absolute;
        top: -40%;
        right: -8%;
        width: 260px;
        height: 260px;
        border-radius: 50%;
        background: radial-gradient(circle, rgba(201,162,39,0.25) 0%, rgba(201,162,39,0) 70%);
    }
    .master-header h1 {
        color: #ffffff;
        margin: 0;
        font-size: 2rem;
        font-weight: 800;
        letter-spacing: -0.01em;
    }
    .master-header p {
        color: #cfe0ff;
        margin: 0.5rem 0 0 0;
        font-size: 1rem;
        max-width: 32rem;
        line-height: 1.5;
    }
    .master-badge {
        display: inline-block;
        background: rgba(201, 162, 39, 0.18);
        color: #e9c46a;
        border: 1px solid rgba(233, 196, 106, 0.4);
        border-radius: 999px;
        padding: 0.2rem 0.75rem;
        font-size: 0.72rem;
        font-weight: 700;
        letter-spacing: 0.05em;
        text-transform: uppercase;
        margin-bottom: 0.85rem;
    }
    [data-testid="stChatMessage"] {
        border-radius: 16px;
    }
    .master-footer {
        text-align: center;
        color: #8a93a6;
        font-size: 0.78rem;
        margin-top: 2rem;
        padding-top: 1rem;
        border-top: 1px solid #e2e6ee;
    }
    .feedback-row [data-testid="stHorizontalBlock"] {
        gap: 0.25rem;
        align-items: center;
    }
    .feedback-thanks {
        color: #6b9b6f;
        font-size: 0.82rem;
        margin-top: 0.25rem;
    }
    </style>
    <div class="master-header">
        <span class="master-badge">NYC Council &middot; AI Assistant</span>
        <h1>NYCC Master Bot</h1>
        <p>Ask about legislation, council meetings, internal resources, or general
        questions — I'll do the research and give you a clear answer.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

if "master_messages" not in st.session_state:
    st.session_state.master_messages = []
if "feedback_given" not in st.session_state:
    st.session_state.feedback_given = set()


async def _run_query(query: str) -> dict:
    async with MCPToolClient(MCP_SERVER_SCRIPT) as client:
        tools_result = await client.session.list_tools()
        catalog = _format_tool_catalog(tools_result.tools)
        return await answer_query_agentic(client, catalog, query)


def _render_feedback_row(msg_id: str, query: str, response: str) -> None:
    """Inline feedback attached under a response — never a modal, so it can
    never cover the response the way st.dialog did. Both thumbs up and down
    reveal an optional comment box before saving, so either can capture detail."""
    if msg_id in st.session_state.feedback_given:
        st.markdown('<div class="feedback-thanks">✅ Thanks for your feedback!</div>', unsafe_allow_html=True)
        return

    rating_key = f"rating_{msg_id}"

    if rating_key not in st.session_state:
        with st.container():
            st.markdown('<div class="feedback-row">', unsafe_allow_html=True)
            col1, col2, col3 = st.columns([1, 1, 10])
            with col1:
                if st.button("👍", key=f"up_{msg_id}", help="Helpful"):
                    st.session_state[rating_key] = "👍 Helpful"
                    st.rerun()
            with col2:
                if st.button("👎", key=f"down_{msg_id}", help="Not helpful"):
                    st.session_state[rating_key] = "👎 Not helpful"
                    st.rerun()
            st.markdown("</div>", unsafe_allow_html=True)
        return

    rating = st.session_state[rating_key]
    st.caption(f"You selected: {rating}")
    comment = st.text_input("Add a comment (optional)", key=f"comment_{msg_id}")
    col1, col2, col3 = st.columns([1, 1, 8])
    with col1:
        if st.button("Submit", key=f"submit_{msg_id}", type="primary"):
            _save_feedback(query, response, rating, comment)
            st.session_state.feedback_given.add(msg_id)
            del st.session_state[rating_key]
            st.rerun()
    with col2:
        if st.button("Cancel", key=f"cancel_{msg_id}"):
            del st.session_state[rating_key]
            st.rerun()


for message in st.session_state.master_messages:
    avatar = "🏛️" if message["role"] == "assistant" else None
    with st.chat_message(message["role"], avatar=avatar):
        st.markdown(message["content"])
        if message["role"] == "assistant":
            _render_feedback_row(message["id"], message["query"], message["content"])

prompt = st.chat_input("Ask NYCC Master Bot...")

if prompt:
    st.session_state.master_messages.append({"role": "user", "content": prompt, "id": None, "query": None})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant", avatar="🏛️"):
        with st.spinner("Thinking..."):
            try:
                result = asyncio.run(_run_query(prompt))
                answer = result["final_answer"]
            except Exception as e:
                answer = _friendly_error(e)
        st.markdown(answer)

        msg_id = uuid.uuid4().hex
        _render_feedback_row(msg_id, prompt, answer)

    st.session_state.master_messages.append(
        {"role": "assistant", "content": answer, "id": msg_id, "query": prompt}
    )

st.markdown(
    '<div class="master-footer">NYCC Master Bot is AI-generated and may make mistakes. '
    "For official guidance, please consult your council office.</div>",
    unsafe_allow_html=True,
)
