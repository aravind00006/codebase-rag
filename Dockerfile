# ──────────────────────────────────────────────────────────────────────────────
# Dockerfile — RAG Codebase Q&A
# ──────────────────────────────────────────────────────────────────────────────

FROM python:3.11-slim AS base

# System dependencies for gitpython + sentence-transformers + healthcheck
RUN apt-get update && apt-get install -y --no-install-recommends \
        git \
        build-essential \
        curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies before copying source so
# Docker layer cache is reused on source-only changes.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source
COPY . .

# Create runtime directories that are typically volume-mounted in production
RUN mkdir -p chroma_db bm25_indexes repos evaluation/results

# ── HF Spaces requires a non-root user (uid 1000) ─────────────────────────────
RUN useradd -m -u 1000 user && chown -R user:user /app
USER user

# ── Default command ───────────────────────────────────────────────────────────
# Local dev (docker-compose):
#   docker run ... rag-codebase-qa api    → starts FastAPI only
#   docker run ... rag-codebase-qa ui     → starts Streamlit only
# HF Spaces deployment:
#   MODE=combined runs both API and UI in one container
ARG MODE=combined
ENV MODE=${MODE}
ENV API_BASE_URL=http://localhost:8000

EXPOSE 8000 7860

RUN chmod +x start.sh

CMD if [ "$MODE" = "api" ]; then \
        uvicorn api.main:app --host 0.0.0.0 --port 8000; \
    elif [ "$MODE" = "ui" ]; then \
        streamlit run frontend/app.py --server.port=7860 --server.address=0.0.0.0; \
    elif [ "$MODE" = "combined" ]; then \
        ./start.sh; \
    else \
        echo "Unknown MODE: $MODE. Use 'api', 'ui', or 'combined'."; exit 1; \
    fi