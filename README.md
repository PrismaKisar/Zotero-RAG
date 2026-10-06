# Zotero RAG Navigator

A question-answering system for your uploaded PDFs or Zotero library featuring a multi-stage RAG pipeline with GROBID parsing, Qdrant hybrid search, contextualization, reranking, and extractive QA. Generate precise answers from your research papers with automatic highlighting.

## Features

**3-Stage RAG Pipeline**: Qdrant Retrieval → CrossEncoder reranking → Extractive QA

- **Flexible PDF sources**: PDF upload or Zotero library selection
- GROBID sentence-level parsing with coordinates for precise PDF highlighting
- Dense + Sparse retrieval with Qdrant hybrid search
- Chunk contextualization via Ollama before embedding (optional)
- Fixed pipeline settings, measured on benchmarks rather than tuned per question
- Sliding window QA for answers spanning chunk boundaries
- Multi-color highlighting for multiple queries
- Web interface (FastAPI + a single HTML page) with real-time progress tracking

## Requirements

- Python 3.11+
- Zotero (with local PDF storage) **OR** PDF files on your device
- GROBID server (Docker)
- Qdrant server (Docker)
- Ollama server (Docker), only for the optional chunk contextualization
- PyTorch with MPS/CUDA support (optional, CPU works too)

## Project Structure

```
.
├── zotero_rag/                 # Main package
│   ├── contextualizer.py       # Ollama context line per chunk, before embedding
│   ├── embedding_manager.py    # Dense and sparse embeddings
│   ├── generative_reader.py    # Benchmark reader: answer + cited passages (Ollama)
│   ├── highlighter.py          # PDF annotation using coordinates
│   ├── models.py               # Data classes
│   ├── pdf_cache_manager.py    # Local PDF cache management
│   ├── pdf_processor.py        # GROBID client and TEI parsing
│   ├── pdf_utils.py            # PDF utilities
│   ├── hybrid_reader.py        # Benchmark reader: generative choice, extractive span
│   ├── pipeline.py             # Main orchestration class
│   ├── qa_engine.py            # Extractive QA, the delivered reader
│   ├── qdrant_manager.py       # Qdrant hybrid indexing and search
│   ├── reranker.py             # CrossEncoder reranking
│   ├── run_from_config.py      # Programmatic YAML config runner
│   ├── server.py               # Web API, serves web/index.html
│   ├── selection.py            # Rerank gate and margin, shared with the benchmarks
│   ├── settings.py             # Fixed, measured pipeline settings
│   ├── zotero_db.py            # Zotero SQLite database interface
│   └── web/                    # Chat-style page (HTML, CSS, JS; no build step), favicon, brand/ logos
│
├── example_configs/         # Example YAML configurations
│   ├── basic.yaml           # Zotero source, every option documented
│   ├── advanced.yaml        # Zotero collection, device, a colour per question
│   ├── folder.yaml          # Folder source, for PDF upload workflows
│   └── highlight_colors.html # Swatch reference for the highlight_color key
│
├── pyproject.toml           # Poetry dependencies
├── README.md                # This file
└── LICENSE                  # GPL v3.0 license

output/                     # App run artefacts (benchmarks: data/ and runs/, see benchmark/README.md)
├── highlighted_results
│   └──{title}.pdf          #Highlighted PDFs
|
├── pdf_cache
│   └──{hash}.pdf           #Indexed PDFs cache
|
└── tei_cache/
    └── {hash}.tei.xml      #GROBID output cache

logs/                       # app.log, zotero_rag.log, run_from_config.log
```

## Installation

### 1. Clone & Setup Environment

```bash
git clone https://github.com/eliroc98/zoteroRAG.git
cd zoteroRAG
uv sync
```

### 2. Start GROBID Service

```bash
docker run -d -p 8070:8070 grobid/grobid:latest
```

Verify it's running:
```bash
curl http://localhost:8070/api/isalive
```

### 3. Start Qdrant Service

```bash
docker run -d -p 6333:6333 qdrant/qdrant
```

Verify it's running:
```bash
curl http://localhost:6333/
```

### 4. Start Ollama Service (Optional)

```bash
docker run -d -v ollama:/root/.ollama -p 11434:11434 --name ollama ollama/ollama

docker exec -it ollama ollama pull llama3.2:3b
```

Verify it's running and contains the model:
```bash
docker exec -it ollama ollama list
```

### 5. Run the App

```bash
uv run uvicorn server:app --app-dir zotero_rag --port 8501
```

Then navigate to `http://localhost:8501`

## Usage

### Settings

Settings, at the bottom of the sidebar, opens the services: GROBID, Qdrant and Ollama URLs, the output
directory (PDF cache, TEI cache, highlighted PDFs), the device (auto picks CUDA,
then MPS, then CPU) and the optional chunk contextualization (off by default:
unmeasured, and indexing fails if Ollama does not answer). The models are fixed
(`google/embeddinggemma-300m` for embeddings): the settings are calibrated on them.
The defaults come from `GROBID_URL`, `QDRANT_URL`, `OLLAMA_URL` and `OUTPUT_DIR`.

### Library

- **Upload**: drop PDFs anywhere on the page, or use the clip next to the question box
- **Zotero**: open Library in the sidebar, pick a collection (or the whole library) and index it
- Library also filters the indexed PDFs, removes the selected ones, or clears the index

### Ask

1. Type a question, e.g. "What methods are used for circuit interpretability?"
2. Every answer cites its paper, page and section, and the link opens the PDF at
   that page; "Passage" shows the chunk with the answer marked and the retrieval,
   rerank and QA scores
3. Select answers, pick a colour and click "Highlight in PDF": the annotated PDFs
   appear in the chat for download. Highlights accumulate in the same file across
   questions, so different colours can mark different aspects of a paper
4. Chats are kept in the browser and listed in the sidebar

## Architecture

### Data Flow

```
PDF Selection (PDF upload OR Zotero Library)
    ↓
GROBID Processing (sentence segmentation + coordinates)
    ↓
TEI Cache (mtime-keyed, persistent)
    ↓
Paragraph Extraction (section classification)
    ↓
Chunk Contextualization (Ollama, optional)
    ↓
Dense + Sparse Encoding (auto batch-size)
    ↓
Qdrant Hybrid Index
```

### Query Pipeline

```
User Question
    ↓
Qdrant Hybrid Retrieval (dense + sparse, fused client-side)
    ↓
CrossEncoder Reranking (gate on the best candidate, margin below it)
    ↓
Extractive QA (overlapping 512-token windows, joint span search)
    ↓
Answer Deduplication & Scoring
    ↓
PDF Highlighting (TEI coordinate mapping)
```

## Configuration

### Environment Variables (Optional)

```bash
# Default Zotero directory (auto-detected from ~/Zotero, ~/Documents/Zotero, ~/.zotero)
export ZOTERO_DATA_DIR=/path/to/zotero

# GROBID timeout (seconds)
export GROBID_TIMEOUT=180

# Disable progress bars (useful in headless environments)
export TQDM_DISABLE=1

# Tokenizer parallelism
export TOKENIZERS_PARALLELISM=false
```

### Pipeline Settings

There are no question types and no parameters to adjust. The retrieval, reranking
and answering thresholds are fixed values measured on the benchmarks, including on
questions the library cannot answer, for which the right output is nothing. They
are documented with their measurements in `zotero_rag/settings.py`.

## Citation

If you use this system in your research, please cite:

```bibtex
@software{zoterorag2026,
  author = {Elisabetta Rocchetti},
  title = {Zotero RAG Navigator: Multi-Stage Question Answering for Research Libraries},
  year = {2026},
  url = {https://github.com/eliroc98/zoteroRAG}
}
```

## License

This project is licensed under the GNU General Public License v3.0. See [LICENSE](LICENSE) for details.
