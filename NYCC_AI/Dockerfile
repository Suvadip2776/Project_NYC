FROM python:3.12-slim

# Hugging Face Spaces (and most managed container hosts) run the container as a
# non-root user with uid 1000. Anything the app writes at runtime — Streamlit's
# config dir, the feedback folder — has to belong to that user or the app dies
# on startup with a PermissionError.
RUN useradd -m -u 1000 user

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY local_ai.py .
COPY orchestrators ./orchestrators
COPY ui ./ui
COPY prompts ./prompts
COPY url_resources.yaml .
COPY mcp ./mcp

RUN chown -R user:user /app

USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    FEEDBACK_DIR=/home/user/feedback

EXPOSE 8000
EXPOSE 8501

# Default to the staff-facing UI: the agent calls OpenRouter directly, so no
# model server needs to run. docker-compose overrides this per service, and the
# optional proxy is still available via:
#   uvicorn local_ai:app --host 0.0.0.0 --port 8000
CMD ["streamlit", "run", "ui/NYCC_master_bot.py", \
     "--server.address", "0.0.0.0", "--server.port", "8501", "--server.headless", "true"]
