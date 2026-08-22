FROM python:3.11-slim

# ponytail: build deps only for wheels that occasionally need compiling; slim base keeps it small
ENV PYTHONUNBUFFERED=1 \
    UV_NO_CACHE=1 \
    UV_PROJECT_ENVIRONMENT=/usr/local

WORKDIR /app

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Deps first for layer caching
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --no-install-project --no-dev

COPY zotero_rag/ ./zotero_rag/
COPY example_configs/ ./example_configs/

EXPOSE 8501

CMD ["streamlit", "run", "zotero_rag/app.py", \
     "--server.address=0.0.0.0", "--server.port=8501", \
     "--server.headless=true"]
