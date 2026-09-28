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
- Streamlit web interface with real-time progress tracking

## Requirements

- Python 3.11+
- Zotero (with local PDF storage) **OR** PDF files on your device
- GROBID server (Docker)
- Qdrant server (Docker)
- Ollama server (Docker)
- PyTorch with MPS/CUDA support (optional, CPU works too)

## Project Structure

```
.
├── zotero_rag/                 # Main package
│   ├── app.py                  # Streamlit web interface
│   ├── contextualizer.py       # Ollama context line per chunk, before embedding
│   ├── embedding_manager.py    # Dense and sparse embeddings
│   ├── highlighter.py          # PDF annotation using coordinates
│   ├── models.py               # Data classes
│   ├── pdf_cache_manager.py    # Local PDF cache management
│   ├── pdf_processor.py        # GROBID client and TEI parsing
│   ├── pdf_utils.py            # PDF utilities
│   ├── pipeline.py             # Main orchestration class
│   ├── qa_engine.py            # Extractive QA
│   ├── qdrant_manager.py       # Qdrant hybrid indexing and search
│   ├── reranker.py             # CrossEncoder reranking
│   ├── run_from_config.py      # Programmatic YAML config runner
│   ├── selection.py            # Rerank gate and margin, shared with the benchmarks
│   ├── settings.py             # Fixed, measured pipeline settings
│   └── zotero_db.py            # Zotero SQLite database interface
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

output/                     # App run artefacts; each benchmark run gets output_<corpus>/
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
uv run streamlit run zotero_rag/app.py
```

Then navigate to `http://localhost:8501`

## Usage

### Initial Setup

1. **Output Directory** (first page)
   - Specify where to store PDF cache, TEI cache, and highlighted PDFs
   - Default: `./literature_output`

2. **GROBID Configuration**
   - Service URL (default: `http://localhost:8070`)
   - Leave default if running locally
   - Required for new PDFs; cached TEI files are reused

3. **Qdrant Configuration**
    - Service URL (default: `http://localhost:6333`)
    - Leave default if running locally
    - Used for hybrid indexing and retrieval

4. **Ollama Configuration (Optional)**
    - Service URL (default: `http://localhost:11434`)
    - Leave default if running locally
    - Used for chunk contextualization before embedding

5. **Select Embedding Model**
   - Model name: Any HuggingFace SentenceTransformer (e.g., `google/embeddinggemma-300m`)
   - Device: auto/cpu/mps/cuda
   - Batch sizes: encoding and reranking (default 32)
   - Click "Load Model"

6. **Add PDFs to Index**

    Choose between two ways of selecting PDFs to index:

    - **Upload PDFs**: Add one or more PDF files directly from your device

    - **Zotero Collection**: Choose a collection or "All Library", all PDFs in the selected Zotero scope will be indexed

### Question Answering

1. **Enter a Question**
   - Natural language queries supported
   - Example: "What methods are used for circuit interpretability?"

2. **Search & Navigate**
   - View answers with context and scores
   - Navigate results with Previous/Next buttons
   - See PDF source, page number, and section
   - Click "Open PDF" to view in default viewer

3. **Highlight Color**
   - Pick any color with the color picker (default yellow)

4. **Highlight PDF**
   - Select only the answers you like to highlight
   - Click "Highlight PDF" to add colored annotations

5. **Multi-Query Highlighting**
   - Run multiple queries with different colors
   - All highlights accumulate in the same PDF
   - Perfect for exploring different aspects of a paper

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
Qdrant Hybrid Retrieval (dense + sparse)
    ↓
CrossEncoder Reranking (probability threshold)
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
