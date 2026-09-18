"""Text embeddings via the council's internal octen-8b model.

NOT REGISTERED in mcp/server.py on this clone. The model lives on the council
network (10.250.7.47:8081) and there is no public, keyless embeddings endpoint
to proxy it with — OpenRouter serves chat completions only. The tool also
returned a raw vector preview, which the agent had no way to use in an answer.

Kept because it still works verbatim on the VPN: re-register it by restoring
the import and the @mcp.tool() wrapper in mcp/server.py.
"""

import json
import os

import requests

EMBEDDING_API_URL = os.getenv("EMBEDDING_API_URL", "http://10.250.7.47:8081/v1/embeddings")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "octen-8b")
# (connect, read) timeouts. The short connect timeout matters off the council
# network: an unreachable internal host now reports a clear error in seconds
# instead of stalling the agent loop for the full read timeout. On the VPN the
# connect is instant, so real behaviour is unchanged.
REQUEST_TIMEOUT = (3.05, 30)


def get_embedding(text: str) -> str:
    """Generate an embedding vector for text via the local embedding model."""
    try:
        response = requests.post(
            EMBEDDING_API_URL,
            json={"input": [text], "model": EMBEDDING_MODEL},
            timeout=REQUEST_TIMEOUT,
        )
        response.raise_for_status()
        data = response.json()
    except Exception as e:
        return f"Embedding error: {e}"

    try:
        vector = data["data"][0]["embedding"]
    except (KeyError, IndexError, TypeError):
        return f"Embedding error: unexpected response format: {json.dumps(data)[:500]}"

    preview = ", ".join(f"{v:.4f}" for v in vector[:8])
    return f"Embedding generated: {len(vector)} dimensions.\nFirst 8 values: [{preview}, ...]"
