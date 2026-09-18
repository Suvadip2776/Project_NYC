"""
Run queries from an exported query log (e.g. ObiWomm_Export.json) through the
real agent and save a JSON file with both the query and the full response -
readable directly (real newlines, no CSV quoting/escaping), and easy to build
a review tool on top of later if a spreadsheet turns out to be too cramped
for long multi-paragraph answers.

Duplicate queries (same text, asked more than once) are only run once - the
original IDs that asked it are kept together in the "ids" field so nothing
is lost, you're just not paying for the same live model call twice.

Prerequisites:
  - The model server must be running: `uvicorn local_ai:app --reload` (from
    the NYCC_AI/ dir, in another terminal) - this script checks for it and
    tells you clearly if it's not up yet.
  - The venv must be active: `source nyc_project/bin/activate`

Usage (from anywhere, run from inside the NYCC_AI venv):
    python results/run_obiwomm_batch.py <path-to-export.json> [N]

    <path-to-export.json>  Path to the query export, e.g.
                            /Users/SSana/Desktop/playground/ObiWomm_Export.json
    [N]                    How many unique queries to run (default: 20, for a
                            quick sanity check). Pass a large number (or the
                            true unique count, printed on startup) to run
                            everything.

Output:
    results/testing_file_<timestamp>.json - already gitignored, never gets
    committed. A list of objects, each with: ids, query, response,
    tools_used, valid (null, for reviewers to fill in - true/false/null),
    comments ("").
"""

import asyncio
import json
import sys
from collections import OrderedDict
from datetime import datetime, timezone
from pathlib import Path

import requests

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "orchestrators"))

from agent_chatbot import _COULD_NOT_ANSWER, _format_tool_catalog, answer_query_agentic  # noqa: E402
from mcp_client import API_URL, MCP_SERVER_SCRIPT, MCPToolClient  # noqa: E402

RESULTS_DIR = REPO_ROOT / "results"


def _check_model_server() -> None:
    # Direct mode (no API_URL) talks to OpenRouter over the internet — there is
    # no local server to wait on, and a bad key surfaces per-request anyway.
    if not API_URL:
        return

    base_url = API_URL.split("/chat/")[0]
    try:
        requests.get(base_url, timeout=5).raise_for_status()
    except requests.exceptions.RequestException:
        sys.exit(
            f"Could not reach the model server at {base_url}.\n"
            "Start it first, in another terminal: uvicorn local_ai:app --reload"
        )


def _load_unique_queries(source_file: Path) -> "OrderedDict[str, dict]":
    with open(source_file) as f:
        raw = json.load(f)

    by_query: "OrderedDict[str, dict]" = OrderedDict()
    for item in raw:
        key = item["User Prompt"].strip().lower()
        if key not in by_query:
            by_query[key] = {"query": item["User Prompt"].strip(), "ids": []}
        by_query[key]["ids"].append(item["ID"])

    print(f"Loaded {len(raw)} total queries, {len(by_query)} unique.")
    return by_query


async def main():
    if len(sys.argv) < 2:
        sys.exit(__doc__)

    source_file = Path(sys.argv[1])
    limit = int(sys.argv[2]) if len(sys.argv) > 2 else 20

    _check_model_server()
    by_query = _load_unique_queries(source_file)
    batch = list(by_query.values())[:limit]
    print(f"Running {len(batch)} now.\n")

    rows = []
    async with MCPToolClient(MCP_SERVER_SCRIPT) as client:
        tools = await client.session.list_tools()
        catalog = _format_tool_catalog(tools.tools)

        for i, item in enumerate(batch, 1):
            query = item["query"]
            print(f"[{i}/{len(batch)}] {query[:80]}")
            try:
                result = await answer_query_agentic(client, catalog, query)
                tools_used = [s["action"] for s in result["steps"]]
                answer = result["final_answer"]
                hit_fallback = answer == _COULD_NOT_ANSWER
                print(f"   tools: {', '.join(tools_used) or '(none)'}{'  [FALLBACK]' if hit_fallback else ''}")
            except Exception as e:
                tools_used = []
                answer = f"ERROR: {e}"
                print(f"   ERROR: {e}")

            rows.append(
                {
                    "ids": item["ids"],
                    "query": query,
                    "response": answer,
                    "tools_used": tools_used,
                    "valid": None,
                    "comments": "",
                }
            )

    RESULTS_DIR.mkdir(exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    out_path = RESULTS_DIR / f"testing_file_{timestamp}.json"
    with open(out_path, "w") as f:
        json.dump(rows, f, indent=2)

    print(f"\nSaved {len(rows)} entries to {out_path}")


if __name__ == "__main__":
    asyncio.run(main())
