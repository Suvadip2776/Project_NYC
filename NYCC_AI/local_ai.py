import os
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import httpx
import traceback

BASE_DIR = Path(__file__).resolve().parent

# Look for .env in the working directory, next to this file, then one level up
# at the project root — so a single OPENROUTER_API_KEY at the top of the project
# serves every entry point without copying the secret into each subfolder.
# load_dotenv never overwrites a variable that is already set, so this layering
# is safe: the environment always wins, then the nearest .env.
load_dotenv()
load_dotenv(BASE_DIR / ".env")
load_dotenv(BASE_DIR.parent / ".env")

app = FastAPI()

# "OWUI" = Open WebUI. The council moved qwen behind an Open WebUI deployment
# (vLLM-backed) instead of the old bare model server. Direct API passthrough
# had to be enabled by the admin (ENABLE_OPENAI_API_PASSTHROUGH); confirmed
# working path is /openai/chat/completions (no /v1) with a Bearer API key.
OWUI_API_KEY = os.getenv("OWUI_API_KEY", "")

# OpenRouter is the off-network route: a hosted, OpenAI-compatible endpoint that
# fronts hundreds of models behind one key. It is what makes this clone runnable
# outside the council VPN - the response shape is identical to Open WebUI's, so
# nothing downstream (call_model, the agent loop, the UIs) changes at all.
# Any catalogue id works; OPENROUTER_MODEL sets the default, and a request may
# override it per call (see ChatRequest.model).
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
OPENROUTER_MODEL = os.getenv("OPENROUTER_MODEL", "qwen/qwen3.8-27b")

# "gemma" is not currently available via the new deployment (only qwen3.8-27b
# is listed in Open WebUI's /api/models) - left pointing at the old address
# so a request fails with a clear connection error rather than silently.
#
# "key_env" names the variable a route authenticates with; None means the route
# sends no Authorization header at all.
MODELS = {
    "openrouter": {
        "url": "https://openrouter.ai/api/v1/chat/completions",
        "model": OPENROUTER_MODEL,
        "key": OPENROUTER_API_KEY,
        "key_env": "OPENROUTER_API_KEY",
    },
    "qwen": {
        "url": "http://10.250.7.47:8086/openai/chat/completions",
        "model": "qwen3.8-27b",
        "key": OWUI_API_KEY,
        "key_env": "OWUI_API_KEY",
    },
    "gemma": {
        "url": "http://10.250.7.47:8085/v1/chat/completions",
        "model": None,
        "key": None,
        "key_env": None,
    },
}


class ChatRequest(BaseModel):
    messages: list
    temperature: float = 0.7
    max_tokens: int = 512
    # Optional per-request override of the route's configured model, so a caller
    # can reach any OpenRouter model ("anthropic/claude-sonnet-4.5", ...) without
    # restarting the server. Ignored by routes that pin their own model (gemma).
    model: str | None = None


@app.get("/")
async def root():
    return {
        "status": "NYCC local AI router is running",
        "routes": list(MODELS),
        "openrouter_model": OPENROUTER_MODEL,
    }


@app.post("/chat/{model}")
async def chat(model: str, req: ChatRequest):
    if model not in MODELS:
        raise HTTPException(status_code=404, detail="Model not found")

    target = MODELS[model]

    payload = {
        "messages": req.messages,
        "temperature": req.temperature,
        "max_tokens": req.max_tokens,
    }
    model_id = req.model or target["model"]
    if model_id:
        payload["model"] = model_id

    headers = {}
    if target["key_env"]:
        if not target["key"]:
            raise HTTPException(
                status_code=500,
                detail=f"{target['key_env']} is not configured",
            )
        headers["Authorization"] = f"Bearer {target['key']}"
        # Optional OpenRouter attribution headers; harmless everywhere else.
        headers["X-Title"] = "NYCC AI Assistant"

    try:
        # Long read timeout (a big answer legitimately takes a while) but a short
        # connect timeout, so an unreachable internal host fails in seconds rather
        # than leaving the caller hanging for two minutes.
        timeout = httpx.Timeout(120.0, connect=3.05)
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.post(
                target["url"],
                json=payload,
                headers=headers,
            )

        print("MODEL:", model, "->", model_id)
        print("MODEL STATUS:", response.status_code)
        print("MODEL RAW RESPONSE:", response.text)

        if response.status_code != 200:
            raise HTTPException(
                status_code=response.status_code,
                detail=response.text,
            )

        try:
            return response.json()
        except Exception:
            raise HTTPException(
                status_code=500,
                detail=f"Model returned non-JSON response: {response.text}",
            )

    except HTTPException:
        raise

    except Exception as e:
        traceback.print_exc()
        raise HTTPException(
            status_code=500,
            # Some httpx timeout errors stringify to "" — fall back to the class name
            # so the message never reads "Backend connection failed: " with nothing after it.
            detail=f"Backend connection failed: {str(e) or type(e).__name__}",
        )









