#!/usr/bin/env bash
# Publish this project to a Hugging Face Space, and re-publish it after edits.
#
#   ./deploy/hf_space.sh <hf-username>/<space-name>
#
# Run it as many times as you like: the first run clones (or initialises) the
# Space repo under deploy/.space/, every later run syncs your current files into
# that clone and pushes the difference. Hugging Face rebuilds the image on push.
#
# The Space clone is kept out of this project's own git history (see .gitignore)
# so the two repositories never nest or fight over the same files.
set -euo pipefail

TARGET="${1:-}"
if [[ ! "$TARGET" =~ ^[^/]+/[^/]+$ ]]; then
    echo "Usage: $0 <hf-username>/<space-name>" >&2
    echo "Example: $0 sdananya/nycc-ai-assistant" >&2
    exit 1
fi

APP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SPACE_NAME="${TARGET#*/}"
WORK="$APP_DIR/deploy/.space/$SPACE_NAME"
REMOTE="https://huggingface.co/spaces/$TARGET"

# Files that make up the deployed app. Anything not listed here never leaves
# your machine — notably .env, the virtualenv, and the feedback folder.
PAYLOAD=(Dockerfile requirements.txt local_ai.py url_resources.yaml
         orchestrators ui prompts mcp)

if [[ ! -d "$WORK/.git" ]]; then
    mkdir -p "$(dirname "$WORK")"
    echo "First run — fetching $REMOTE"
    if ! git clone "$REMOTE" "$WORK" 2>/dev/null; then
        echo "Could not clone it. Create the Space first (SDK: Docker) at"
        echo "  https://huggingface.co/new-space"
        echo "then run this again."
        exit 1
    fi
fi

echo "Syncing files into $WORK"
for item in "${PAYLOAD[@]}"; do
    if [[ -d "$APP_DIR/$item" ]]; then
        rsync -a --delete \
              --exclude='__pycache__' --exclude='*.pyc' \
              "$APP_DIR/$item/" "$WORK/$item/"
    else
        cp "$APP_DIR/$item" "$WORK/$item"
    fi
done

# Written once. Later edits to the Space's description survive re-runs.
if [[ ! -f "$WORK/README.md" ]]; then
    cat > "$WORK/README.md" <<'MD'
---
title: NYCC AI Assistant
emoji: 🏛️
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 8501
pinned: false
---

# NYCC AI Assistant

Ask a question about New York City Council business in plain English. The agent
decides which tools it needs, calls them over MCP, and answers from the data
that comes back — not from what the model remembers.

Try:

- *What's 15% of the population of council district 31?*
- *Are there any Council hearings in the next 7 days?*
- *What does NYC Council legislation say about street vendor enforcement?*
- *What bills address lithium-ion battery safety?*

Every source is public: NYC Open Data for legislation, the Council's public
Legistar calendar for meetings, and the Council's own `councilcount` package for
census demographics.
MD
    echo "Wrote the Space README (front-matter tells HF to build the Dockerfile)"
fi

# Hugging Face reads the port from the README's front-matter and defaults it to
# 7860, but this app serves Streamlit on 8501 — a mismatch shows up as a Space
# that builds cleanly and then reports no running app. Creating a Space through
# the web form writes that README for you, so correct it here rather than
# leaving it as a step to remember.
python3 - "$WORK/README.md" <<'PYEOF'
import re, sys, pathlib

readme = pathlib.Path(sys.argv[1])
text = readme.read_text()
match = re.match(r"^---\n(.*?)\n---\n", text, re.S)
if not match:
    sys.exit(0)  # no front-matter: the block above wrote a correct one

front, rest = match.group(1), text[match.end():]
original, notes = front, []

# The app serves Streamlit on 8501; Hugging Face otherwise assumes 7860 and the
# Space builds cleanly but reports no running app.
if not re.search(r"^app_port:\s*8501\s*$", front, re.M):
    if re.search(r"^app_port:", front, re.M):
        front = re.sub(r"^app_port:.*$", "app_port: 8501", front, flags=re.M)
    else:
        front += "\napp_port: 8501"
    notes.append("app_port: 8501")

if not re.search(r"^sdk:\s*docker\s*$", front, re.M):
    if re.search(r"^sdk:", front, re.M):
        front = re.sub(r"^sdk:.*$", "sdk: docker", front, flags=re.M)
    else:
        front += "\nsdk: docker"
    notes.append("sdk: docker")

# Creating a Space through the web form writes Gradio-only keys. They are
# invalid under sdk: docker and the Space refuses to start with a CONFIG_ERROR,
# so drop them rather than leaving a confusing failure behind.
dropped = [k for k in ("sdk_version", "app_file") if re.search(rf"^{k}:", front, re.M)]
for key in dropped:
    front = re.sub(rf"^{key}:.*$\n?", "", front, flags=re.M)
if dropped:
    notes.append("dropped " + ", ".join(dropped))

if front != original:
    readme.write_text(f"---\n{front}\n---\n{rest}")
    print("  README front-matter: " + "; ".join(notes))
PYEOF

cd "$WORK"
git add -A
if git diff --cached --quiet; then
    echo "No changes since the last publish — nothing to push."
    exit 0
fi

echo
echo "Changes to publish:"
git diff --cached --stat | sed 's/^/  /'
git commit -qm "Update NYCC AI Assistant ($(date -u +%Y-%m-%dT%H:%MZ))"

if git push origin HEAD 2>&1 | sed 's/^/  /'; then
    echo
    echo "Published. HF is rebuilding now (about 2-4 minutes):"
    echo "  $REMOTE"
else
    echo
    echo "Push failed — usually authentication. Use an access token from"
    echo "https://huggingface.co/settings/tokens (write scope) as the password,"
    echo "or run: pip install huggingface_hub && huggingface-cli login"
    exit 1
fi
