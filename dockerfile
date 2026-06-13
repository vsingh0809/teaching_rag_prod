FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    # WHY: tells uv where to install — we control the path
    UV_PROJECT_ENVIRONMENT=/app/.venv \
    # WHY: disables uv cache inside container — we don't need it after install
    UV_NO_CACHE=1

# WHY --no-install-recommends: skip optional packages debian pulls in
# WHY combine apt commands in one RUN: each RUN = one layer
# WHY rm -rf /var/lib/apt/lists: delete apt cache immediately in same layer
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# WHY install uv via script not COPY --from:
# COPY --from pulls the entire uv Docker image (58MB of bloat)
# This installs only the uv binary (~10MB)
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:$PATH"

# WHY create user BEFORE copying files:
# So we can set ownership at COPY time, not via chown layer after
RUN adduser --disabled-password --gecos "" appuser

WORKDIR /app

COPY pyproject.toml uv.lock ./

# WHY --frozen: use exact uv.lock versions, no resolution
# WHY --no-dev: skip pytest/ruff in production
# WHY && find ... -delete: remove compiled .pyc cache uv leaves behind
RUN uv sync --frozen --no-dev && \
    find /app/.venv -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true

# WHY --chown here instead of chown layer:
# COPY --chown sets ownership inline = zero extra layer = no size duplication
COPY --chown=appuser:appuser api/ ./api/
COPY --chown=appuser:appuser clients/ ./clients/
COPY --chown=appuser:appuser ingestion/ ./ingestion/
COPY --chown=appuser:appuser retrieval/ ./retrieval/
COPY --chown=appuser:appuser utils/ ./utils/
COPY --chown=appuser:appuser models/ ./models/

USER appuser

EXPOSE 8000

CMD ["uv", "run", "uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]