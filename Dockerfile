FROM python:3.11-slim

# ponytail: build deps only for wheels that occasionally need compiling; slim base keeps it small
ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    POETRY_VIRTUALENVS_CREATE=false

WORKDIR /app

RUN pip install --no-cache-dir poetry

# Deps first for layer caching
COPY pyproject.toml poetry.lock README.md ./
RUN poetry install --no-root --no-interaction

# nltk punkt at build time so no network is needed at runtime
RUN python -c "import nltk; nltk.download('punkt', quiet=True); nltk.download('punkt_tab', quiet=True)"

COPY zotero_rag/ ./zotero_rag/
COPY example_configs/ ./example_configs/

EXPOSE 8501

CMD ["streamlit", "run", "zotero_rag/app.py", \
     "--server.address=0.0.0.0", "--server.port=8501", \
     "--server.headless=true"]
