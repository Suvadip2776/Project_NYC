"""
Shared MCP client infrastructure used by every orchestrator: the path to
mcp/server.py, the stdio session wrapper, and the HTTP call to the model.

There are two ways to reach a model, and call_model picks between them:

  direct (the default) — POST straight to OpenRouter with OPENROUTER_API_KEY.
      OpenRouter is already a hosted HTTP endpoint, so proxying it through
      local_ai.py would add a process without adding anything else. Nothing
      to start: run the CLI or a UI on its own.

  proxy — set API_URL to a running local_ai.py route. This is what the council
      VPN path needs (API_URL=http://localhost:8000/chat/qwen), since the
      internal Open WebUI deployment is what local_ai.py was written to front. Split out from final_master_chatbot.py (removed
along with the rest of the single-route pipeline - see the
legacy-single-route-pipeline branch) since agent_chatbot.py depends on this
part but not on that file's single-route-specific routing logic.
"""

import os
import sys
from contextlib import AsyncExitStack
from pathlib import Path

import requests
from dotenv import load_dotenv
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

BASE_DIR = Path(__file__).resolve().parent
REPO_ROOT = BASE_DIR.parent
MCP_SERVER_SCRIPT = REPO_ROOT / "mcp" / "server.py"

# .env in the working directory, then NYCC_AI/, then the project root above it.
# load_dotenv never overwrites an already-set variable, so the environment wins.
load_dotenv()
load_dotenv(REPO_ROOT / ".env")
load_dotenv(REPO_ROOT.parent / ".env")

# Unset (the default) means direct mode. Set it to a local_ai.py route to proxy.
API_URL = os.getenv("API_URL")

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
OPENROUTER_MODEL = os.getenv("OPENROUTER_MODEL", "qwen/qwen3.8-27b")

# (connect, read): a generous read timeout for long answers, a shorter connect
# so an unreachable host fails quickly instead of stalling the agent loop.
MODEL_REQUEST_TIMEOUT = (10, 120)


class MCPToolClient:
    """Long-lived MCP client session backed by an mcp/server.py subprocess."""

    def __init__(self, server_script: Path):
        # Pass our environment through explicitly. The MCP SDK otherwise starts the
        # server with a filtered, "safe" subset of variables, so anything the tools
        # read (PUBLIC_DEPLOYMENT, SOCRATA_APP_TOKEN, the endpoint overrides) would
        # silently never arrive in the subprocess.
        self._server_params = StdioServerParameters(
            command=sys.executable,
            args=[str(server_script)],
            env=os.environ.copy(),
        )
        self._stack = AsyncExitStack()
        self.session: ClientSession | None = None

    async def __aenter__(self) -> "MCPToolClient":
        await self._stack.__aenter__()
        # The MCP server's own stderr (its "MCP server running" / request logging)
        # gets forwarded to our stderr by default — discard it to keep output clean.
        devnull = self._stack.enter_context(open(os.devnull, "w"))
        read, write = await self._stack.enter_async_context(
            stdio_client(self._server_params, errlog=devnull)
        )
        self.session = await self._stack.enter_async_context(ClientSession(read, write))
        await self.session.initialize()
        return self

    async def __aexit__(self, exc_type, exc, tb):
        await self._stack.__aexit__(exc_type, exc, tb)

    async def call_tool(self, name: str, arguments: dict) -> str:
        result = await self.session.call_tool(name, arguments=arguments)

        text_parts = [block.text for block in result.content if getattr(block, "type", None) == "text"]
        text = "\n".join(text_parts) if text_parts else "(tool returned no text content)"

        if result.isError:
            return f"MCP tool '{name}' reported an error: {text}"

        return text


def call_model(messages: list, temperature: float = 0.3, max_tokens: int = 800) -> str:
    payload = {"messages": messages, "temperature": temperature, "max_tokens": max_tokens}
    headers = {}

    if API_URL:
        url = API_URL
    else:
        if not OPENROUTER_API_KEY:
            raise RuntimeError(
                "OPENROUTER_API_KEY is not set. Put it in NYCC_AI/.env (or the project "
                "root .env), or set API_URL to a running local_ai.py route instead."
            )
        url = OPENROUTER_URL
        payload["model"] = OPENROUTER_MODEL
        headers = {
            "Authorization": f"Bearer {OPENROUTER_API_KEY}",
            "X-Title": "NYCC AI Assistant",
        }

    response = requests.post(url, json=payload, headers=headers, timeout=MODEL_REQUEST_TIMEOUT)

    try:
        response.raise_for_status()
    except requests.exceptions.HTTPError as e:
        # Surface what the provider actually said (bad key, unknown model id,
        # out of credit) instead of a bare "400 Client Error".
        raise RuntimeError(
            f"Model request failed ({response.status_code}): {response.text[:300]}"
        ) from e

    data = response.json()
    return data["choices"][0]["message"]["content"].strip()
