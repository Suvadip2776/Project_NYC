"""
Streamlit UI for the multi-tool agent pipeline (agent_chatbot.py).

Wraps agent_chatbot.py's existing agent loop (MCPToolClient, answer_query_agentic,
_format_tool_catalog) as-is — no changes to that file, so `python agent_chatbot.py`
still works as a standalone CLI. Run this with: streamlit run agent_chatbot_ui.py
"""

import asyncio
import sys
from pathlib import Path

import requests
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "orchestrators"))
from agent_chatbot import MCP_SERVER_SCRIPT, MCPToolClient, _format_tool_catalog, answer_query_agentic


def _friendly_error(e: BaseException) -> str:
    """MCP wraps errors in nested TaskGroups (anyio), so a plain 'connection
    refused' surfaces as an opaque ExceptionGroup. Unwrap to the real cause and
    give an actionable message."""
    root = e
    while isinstance(root, BaseExceptionGroup) and root.exceptions:
        root = root.exceptions[0]

    if isinstance(root, requests.exceptions.ConnectionError):
        return (
            "Could not reach the model. Check your connection — or, if API_URL is set "
            "to a local_ai.py route, make sure `uvicorn local_ai:app` is running."
        )

    return f"Something went wrong: {root}"

st.set_page_config(
    page_title="NYC Council Internal AI — Agent",
    page_icon="🏛️",
    layout="centered",
)

st.markdown(
    """
    <style>
    .stApp { background-color: #f4f6f9; }
    .nycc-header {
        background: linear-gradient(135deg, #0b3d91 0%, #14509e 100%);
        padding: 1.75rem 2rem;
        border-radius: 12px;
        margin-bottom: 1.5rem;
        box-shadow: 0 4px 14px rgba(11, 61, 145, 0.25);
    }
    .nycc-header h1 {
        color: #ffffff;
        margin: 0;
        font-size: 1.8rem;
        font-weight: 700;
    }
    .nycc-header p {
        color: #cfe0ff;
        margin: 0.35rem 0 0 0;
        font-size: 0.95rem;
    }
    .step-badge {
        display: inline-block;
        background-color: #eef3ff;
        color: #0b3d91;
        border: 1px solid #b9cdf5;
        border-radius: 999px;
        padding: 0.15rem 0.7rem;
        font-size: 0.75rem;
        font-weight: 600;
        margin: 0.2rem 0.3rem 0.2rem 0;
    }
    </style>
    <div class="nycc-header">
        <h1>NYC Council Internal AI — Agent Mode</h1>
        <p>Calls as many tools as it needs, in whatever order the question requires.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.subheader("How this differs")
    st.markdown(
        """
The standard assistant picks **one** tool per question.

**Agent mode** can chain multiple tools together — e.g. look up a
council district's population, then calculate a percentage of it —
deciding for itself how many steps a question needs (up to 5).
        """
    )
    st.divider()
    st.caption("Pipeline: query → agent loop (model picks tools) → MCP tool calls → final answer")

if "agent_messages" not in st.session_state:
    st.session_state.agent_messages = []


async def _run_query(query: str) -> dict:
    async with MCPToolClient(MCP_SERVER_SCRIPT) as client:
        tools_result = await client.session.list_tools()
        catalog = _format_tool_catalog(tools_result.tools)
        return await answer_query_agentic(client, catalog, query)


def _render_steps(steps: list):
    if not steps:
        st.markdown('<span class="step-badge">no tools used</span>', unsafe_allow_html=True)
        return

    badges = "".join(f'<span class="step-badge">{i + 1}. {s["action"]}</span>' for i, s in enumerate(steps))
    st.markdown(badges, unsafe_allow_html=True)
    with st.expander("Tool call details"):
        for i, s in enumerate(steps):
            st.markdown(f"**Step {i + 1}: `{s['action']}`**")
            st.json(s["action_input"])
            st.text(s["observation"])
            if i < len(steps) - 1:
                st.divider()


for message in st.session_state.agent_messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message.get("steps") is not None:
            _render_steps(message["steps"])

prompt = st.chat_input("Ask anything — the agent will use whatever tools it needs...")

if prompt:
    st.session_state.agent_messages.append({"role": "user", "content": prompt, "steps": None})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Working through the request..."):
            try:
                result = asyncio.run(_run_query(prompt))
                answer = result["final_answer"]
                steps = result["steps"]
            except Exception as e:
                answer = _friendly_error(e)
                steps = []

        st.markdown(answer)
        _render_steps(steps)

    st.session_state.agent_messages.append({"role": "assistant", "content": answer, "steps": steps})
