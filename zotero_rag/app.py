"""
app.py - Streamlit web interface for Zotero RAG System

Run with: streamlit run app.py
"""

import os
import sqlite3
import subprocess
import sys
import time

import streamlit as st
from pdf_utils import sanitize_filename
from question_presets import PRESETS, resolve
from zotero_db import ZoteroDatabase

from zotero_rag import ZoteroRAG


def rgb_to_hex(rgb):
    """Convert RGB tuple (0-1) to hex color"""
    r, g, b = [int(x * 255) for x in rgb]
    return f'#{r:02x}{g:02x}{b:02x}'

def _load_zotero_collections():
    try:
        with st.spinner("Loading Zotero collections..."):
            st.session_state.zotero_collections = ZoteroDatabase(None).list_collections()
            st.session_state.zotero_collections_loaded = True
    except sqlite3.OperationalError as e:
        if "locked" in str(e):
            st.error("Zotero database is locked")
            st.warning("""
            **The database is currently locked by Zotero.**

            You have two options:

            1. **Close Zotero** (Recommended)
            - Close the Zotero application completely
            - Then refresh this page

            2. **Keep Zotero open** (Advanced)
            - The app will try to read the database in read-only mode
            - Click the button below to retry
            """)

            if st.button("Retry Connection"):
                st.rerun()
        else:
            st.error(f"Database error: {e}")
        st.stop()
    except Exception as e:  # noqa: BLE001 - UI boundary: report, never crash the app
        st.error(f"Error loading Zotero: {e}")
        st.info("Make sure Zotero is installed and the database is accessible")


def _run_ingest_and_index(result):
    if result is None:
        st.warning("No ingest result returned.")
        return

    if result.failed_pdfs:
        st.warning(f"{len(result.failed_pdfs)} PDF could not be uploaded to cache.")
        with st.expander("Show upload issues"):
            for item in result.failed_pdfs:
                st.write(f"- {item.get('title', 'Unknown file')}: {item.get('error', 'Unknown error')}")

    ingested_pdfs = getattr(result, "ingested_pdfs", [])

    if not ingested_pdfs:
        st.info("No new PDF to index.")
        return

    newly_cached_count = len([item for item in ingested_pdfs if getattr(item, "newly_cached", False)])
    if newly_cached_count < len(ingested_pdfs):
        st.info(
            f"Using cache for {len(ingested_pdfs) - newly_cached_count} PDF(s) already stored by hash."
        )

    stage_status = st.empty()
    pdf_progress_bar = st.progress(0, text="PDF analysis pending...")
    contextualization_progress_bar = st.progress(0, text="Contextualization pending...")
    encoding_progress_bar = st.progress(0, text="Embedding pending...")
    upsert_progress_bar = st.progress(0, text="Qdrant upsert pending...")

    def progress_callback(stage, current, total, message):
        progress = current / total if total > 0 else 0
        stage_status.info(message)
        if stage == 'pdf':
            pdf_progress_bar.progress(progress, text=f"PDF analysis: {message}")
        elif stage == 'contextualization':
            contextualization_progress_bar.progress(progress, text=f"Contextualization: {message}")
        elif stage == 'encoding':
            encoding_progress_bar.progress(progress, text=f"Embedding: {message}")
        elif stage == 'upserting':
            upsert_progress_bar.progress(progress, text=f"Qdrant upsert: {message}")

    try:
        upsert_report = st.session_state.rag.upsert_pdfs(
            target_pdfs=ingested_pdfs,
            progress_callback=progress_callback
        )

        indexed_chunks = int(getattr(upsert_report, "indexed_chunks", 0))
        processed_pdfs = int(getattr(upsert_report, "processed_pdfs", 0))
        failed_pdfs = getattr(upsert_report, "failed_pdfs", [])
        already_indexed = getattr(upsert_report, "already_indexed", [])
        title_overrides = getattr(upsert_report, "title_overrides", [])
        duplicate_titles = getattr(upsert_report, "duplicate_titles", [])

        if already_indexed:
            st.info(
                f"{len(already_indexed)} PDF(s) skipped because already indexed in Qdrant."
            )
            with st.expander("Show Qdrant duplicates"):
                for item in already_indexed:
                    input_title = item.get("input_title", "")
                    indexed_title = item.get("indexed_title", "")
                    if indexed_title and indexed_title != input_title:
                        st.write(f"- {input_title} -> {indexed_title}")
                    else:
                        st.write(f"- {input_title or indexed_title}")

        if duplicate_titles:
            st.warning(
                f"{len(duplicate_titles)} PDF(s) skipped because the title is already in use."
            )
            with st.expander("Show duplicate titles"):
                for title in duplicate_titles:
                    st.write(f"- {title}")

        if title_overrides:
            st.info(
                f"{len(title_overrides)} PDF(s) use existing registry titles."
            )
            with st.expander("Show registry title overrides"):
                for item in title_overrides:
                    input_title = item.get("input_title", "")
                    indexed_title = item.get("indexed_title", "")
                    if indexed_title and indexed_title != input_title:
                        st.write(f"- {input_title} -> {indexed_title}")
                    else:
                        st.write(f"- {input_title or indexed_title}")

        if failed_pdfs:
            st.warning(f"{len(failed_pdfs)} PDF failed during processing.")
            grobid_errors = [
                item for item in failed_pdfs
                if "GROBID service not reachable" in (item.get("error") or "")
            ]
            if grobid_errors:
                st.error("GROBID is not running. Start the service and retry indexing.")
            with st.expander("Show processing errors"):
                for item in failed_pdfs:
                    st.write(f"- {item.get('title', 'Unknown PDF')}: {item.get('error', 'Unknown error')}")

        stage_status.success("Indexing workflow completed.")
        pdf_progress_bar.progress(1.0, text="PDF analysis completed")
        contextualization_progress_bar.progress(1.0, text="Contextualization completed")
        encoding_progress_bar.progress(1.0, text="Embedding completed")
        upsert_progress_bar.progress(1.0, text="Qdrant upsert completed")

        if indexed_chunks > 0:
            st.session_state.indexed = True
            st.success(
                f"Indexing complete: {processed_pdfs} new PDFs processed, {indexed_chunks} chunks upserted."
            )
        else:
            st.info("Indexing finished but no new chunks were upserted.")

        time.sleep(5.0)
        st.rerun()
    except ConnectionError as e:
        if "Qdrant" in str(e):
            st.error("Qdrant is not running. Start the service and retry indexing.")
        elif "GROBID" in str(e):
            st.error("GROBID is not running. Start the service and retry indexing.")
        else:
            st.error(f"Connection error: {e}")
    except Exception as e:  # noqa: BLE001 - UI boundary: report, never crash the app
        stage_status.error("Indexing failed.")
        with st.expander("Show full error"):
            st.exception(e)

def main():
    st.set_page_config(
        page_title="Zotero RAG Navigator",
        page_icon="",
        layout="wide"
    )

    st.title("Zotero RAG Navigator")
    st.markdown("Query your local Zotero library with natural language")

    # Output directory (first thing asked)
    st.subheader("Output Directory")
    if 'output_dir' not in st.session_state:
        st.session_state.output_dir = os.environ.get("OUTPUT_DIR", "./output")
    output_dir = st.text_input(
        "Base output directory (TEI cache, highlights)",
        value=st.session_state.output_dir,
        help="Choose where to store cached TEI files, and highlighted PDFs."
    )
    st.session_state.output_dir = output_dir or "./output"

    # Initialize session state
    if 'rag' not in st.session_state:
        st.session_state.rag = None
    if 'search_results' not in st.session_state:
        st.session_state.search_results = []
    if 'search_candidates' not in st.session_state:
        st.session_state.search_candidates = []
    if 'current_answer' not in st.session_state:
        st.session_state.current_answer = 0
    if 'indexed' not in st.session_state:
        st.session_state.indexed = False
    if 'zotero_collections_loaded' not in st.session_state:
        st.session_state.zotero_collections_loaded = False
    if 'zotero_collections' not in st.session_state:
        st.session_state.zotero_collections = []
    if 'zotero_collection' not in st.session_state:
        st.session_state.zotero_collection = None
    if 'dense_model_name' not in st.session_state:
        st.session_state.dense_model_name = "BAAI/bge-base-en-v1.5"
    if 'model_loaded' not in st.session_state:
        st.session_state.model_loaded = False
    if 'model_device' not in st.session_state:
        st.session_state.model_device = None  # auto-select

    # Main tabs - only show after model is loaded and indexed
    if st.session_state.model_loaded and st.session_state.indexed:
        tab1, tab2 = st.tabs(["Setup", "Search & Highlight"])

        with tab1:
            show_setup_tab()

        with tab2:
            show_search_tab()
    else:
        # Show setup only
        show_setup_tab()

def show_setup_tab():
    """Setup tab for collection selection, model loading, and indexing."""

    st.header("Setup Configuration")

    # Services configuration
    st.subheader("1⃣ Service Endpoints")
    st.text("Configure the local services used for PDF parsing and vector search.")


    col_required_services, col_optional_services = st.columns(2, gap="large")

    with col_required_services:
        st.markdown("##### Required Services")

        grobid_url = st.text_input(
            "GROBID Service URL",
            value=os.environ.get("GROBID_URL", "http://localhost:8070"),
            key="grobid_url_input",
            help="URL of GROBID service. Start with: docker run -p 8070:8070 grobid/grobid"
        )

        qdrant_url = st.text_input(
            "Qdrant Service URL",
            value=os.environ.get("QDRANT_URL", "http://localhost:6333"),
            key="qdrant_url_input",
            help="URL of Qdrant service. Start with: docker run -p 6333:6333 qdrant/qdrant"
        )

    with col_optional_services:
        st.markdown("##### Optional Contextualization")
        use_chunk_contextualization = st.checkbox(
            "Enable chunk contextualization before upsert",
            value=st.session_state.get("use_chunk_contextualization", True),
            key="use_chunk_contextualization_chk",
            help="When enabled, each chunk is enriched with document-level context before vectorization."
        )

        st.session_state.use_chunk_contextualization = use_chunk_contextualization

        st.info(
            "Chunk contextualization can improve retrieval precision by enriching each chunk with global document context. "
            "However, it may significantly increase indexing time and computational cost, especially for long PDFs."
        )

        ollama_url = st.text_input(
            "Ollama Service URL",
            value=os.environ.get("OLLAMA_URL", "http://localhost:11434"),
            key="ollama_url_input",
            disabled=not use_chunk_contextualization,
            help="URL of Ollama service. Start with: docker run -d -v ollama:/root/.ollama -p 11434:11434 ollama/ollama"
        )

    st.markdown("---")

    # Model Selection
    st.subheader("2⃣ Select Embedding Model")

    st.markdown(
        """
        <style>
        .st-key-load_model_btn button {
            min-height: 108px;
            font-size: 1.05rem;
            font-weight: 600;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    model_col_left, model_col_right = st.columns([4, 1], gap="large")

    with model_col_left:
        model_input = st.text_input(
            "FastEmbed dense model name",
            value=st.session_state.dense_model_name,
            placeholder="e.g., BAAI/bge-base-en-v1.5",
            help="Enter a FastEmbed-supported dense model (for example 'BAAI/bge-base-en-v1.5' or 'sentence-transformers/all-MiniLM-L6-v2')"
        )

        col_device, col_encode_batch, col_rerank_batch = st.columns(3)

        with col_device:
            device_options = ["auto", "cpu", "cuda"]
            current_device = "auto" if st.session_state.model_device is None else st.session_state.model_device
            if current_device not in device_options:
                current_device = "auto"

            device_choice = st.selectbox(
                "Device",
                options=device_options,
                index=device_options.index(current_device),
                help="Select compute device. 'auto' uses CUDA when available, otherwise CPU."
            )

        with col_encode_batch:
            encode_batch_auto = st.checkbox(
                "Auto-detect encoding batch",
                value=st.session_state.get('encode_batch_auto', True),
                help="Auto-detect safe batch size (targets 75% memory usage)",
                key="encode_batch_auto_chk"
            )
            if not encode_batch_auto:
                encode_batch_size = st.number_input(
                    "Encoding batch size",
                    min_value=1, max_value=256, value=st.session_state.get('encode_batch_size', 8),
                    help="Manual batch size for encoding"
                )
            else:
                encode_batch_size = None
                st.session_state.encode_batch_auto = True

        with col_rerank_batch:
            rerank_batch_auto = st.checkbox(
                "Auto-detect rerank batch",
                value=st.session_state.get('rerank_batch_auto', True),
                help="Auto-detect safe batch size (targets 75% memory usage)",
                key="rerank_batch_auto_chk"
            )
            if not rerank_batch_auto:
                rerank_batch_size = st.number_input(
                    "Reranking batch size",
                    min_value=1, max_value=256, value=st.session_state.get('rerank_batch_size', 8),
                    help="Manual batch size for reranking"
                )
            else:
                rerank_batch_size = None
                st.session_state.rerank_batch_auto = True

    with model_col_right:
        st.write("")
        st.write("")
        load_model_clicked = st.button(
            "Load Model",
            type="primary",
            width="stretch",
            key="load_model_btn"
        )

    if load_model_clicked:
        if model_input:
            with st.spinner(f"Loading model: {model_input}..."):
                try:
                    st.session_state.dense_model_name = model_input
                    st.session_state.model_device = None if device_choice == "auto" else device_choice
                    st.session_state.encode_batch_size = encode_batch_size
                    st.session_state.rerank_batch_size = rerank_batch_size

                    st.session_state.rag = ZoteroRAG(
                        dense_model_name=model_input,
                        grobid_url=grobid_url,
                        qdrant_url=qdrant_url,
                        ollama_url=ollama_url,
                        output_base_dir=st.session_state.output_dir,
                        model_device=st.session_state.model_device,
                        encode_batch_size=encode_batch_size,
                        rerank_batch_size=rerank_batch_size,
                        use_chunk_contextualization=st.session_state.use_chunk_contextualization
                    )

                    st.session_state.model_loaded = True
                    st.session_state.indexed = False  # Reset indexed status

                    # Show configuration summary
                    batch_info = []
                    if encode_batch_size is None:
                        batch_info.append("Encoding: Auto-detect")
                    else:
                        batch_info.append(f"Encoding: {encode_batch_size}")
                    if rerank_batch_size is None:
                        batch_info.append("Reranking: Auto-detect")
                    else:
                        batch_info.append(f"Reranking: {rerank_batch_size}")

                    st.success(f"Model loaded: {model_input}\n\n**Batch sizes:** {' | '.join(batch_info)}")
                    st.rerun()
                except Exception as e:  # noqa: BLE001 - UI boundary: report, never crash the app
                    st.error(f"Error loading model: {e}")
                    st.info("Make sure the model name is correct and supported by FastEmbed")
        else:
            st.error("Please enter a model name")

    if st.session_state.model_loaded:
        st.info(f"Current model: **{st.session_state.dense_model_name}**")

    st.markdown("---")

    # Indexing Section - only show if model is loaded
    if st.session_state.model_loaded:
        st.header("Indexed PDFs Manager")
        col_left, col_right = st.columns([3, 1], gap="large")

        with col_left:
            st.markdown("### Indexed PDFs")

            try:
                indexed_titles = st.session_state.rag.get_indexed_pdfs() if st.session_state.rag else []
            except Exception as e:  # noqa: BLE001 - UI boundary: report, never crash the app
                if isinstance(e, ConnectionError) and "Qdrant" in str(e):
                    st.error("Qdrant is not running. Start the service to view indexed PDFs.")
                else:
                    st.error(f"Error fetching indexed PDFs: {e}")
                indexed_titles = []

            new_indexed = len(indexed_titles) > 0
            if st.session_state.indexed != new_indexed:
                st.session_state.indexed = new_indexed
                st.rerun()

            total_pdfs = len(indexed_titles)

            if not st.session_state.rag.consistency_check(indexed_titles):
                st.warning("Consistency check failed: some PDFs in the index may be missing from the cache or have changed. Consider re-indexing to ensure data integrity.")

            filter_text = st.text_input(
                "Search indexed PDFs",
                value="",
                placeholder="Type part of the PDF name...",
                help="Filter list by filename. Useful when index contains many PDFs.",
                key="indexed_pdf_filter"
            )

            def _get_pdf_title(item):
                if isinstance(item, dict):
                    return item.get("title", "")
                return str(item)

            if filter_text:
                filtered_pdfs = [
                    item for item in indexed_titles
                    if filter_text.lower() in _get_pdf_title(item).lower()
                ]
            else:
                filtered_pdfs = indexed_titles

            if filtered_pdfs:
                total_filtered = len(filtered_pdfs)
                page_size = 10
                items_per_page = page_size * 2
                max_page = max((total_filtered - 1) // items_per_page + 1, 1)

                col_count, col_page = st.columns(2)
                with col_count:
                    st.metric("Total indexed", total_pdfs)
                with col_page:
                    current_page = st.number_input(
                        "Page",
                        min_value=1,
                        max_value=max_page,
                        value=1,
                        step=1,
                        key="indexed_pdf_page"
                    )

                start = (current_page - 1) * items_per_page
                end = start + items_per_page
                page_items = filtered_pdfs[start:end]

                left_column_items = page_items[::2]
                right_column_items = page_items[1::2]
                max_rows = max(len(left_column_items), len(right_column_items))

                two_col_rows = []
                for row_idx in range(max_rows):
                    left_value = left_column_items[row_idx] if row_idx < len(left_column_items) else None
                    right_value = right_column_items[row_idx] if row_idx < len(right_column_items) else None
                    two_col_rows.append({
                        "PDF Name (1)": _get_pdf_title(left_value) if left_value else "",
                        "PDF Name (2)": _get_pdf_title(right_value) if right_value else "",
                    })

                st.caption(f"Showing {start + 1}-{min(end, total_filtered)} of {total_filtered} PDFs")
                st.dataframe(
                    two_col_rows,
                    width="stretch",
                    height=420,
                    column_config={
                        "PDF Name (1)": st.column_config.TextColumn(width="large"),
                        "PDF Name (2)": st.column_config.TextColumn(width="large"),
                    },
                    hide_index=True
                )
            else:
                if total_pdfs == 0:
                    st.info("No indexed PDFs yet. Build your index by adding PDFs from the action panel.")
                else:
                    st.warning("No PDF matches your search filter.")

        with col_right:
            st.markdown("### Actions")

            with st.expander("Add PDFs", expanded=True):
                uploaded_pdfs = st.file_uploader(
                    "Select PDF files to index",
                    type=["pdf"],
                    accept_multiple_files=True,
                    key="add_pdfs_uploader"
                )

                if uploaded_pdfs:
                    st.caption(f"Selected {len(uploaded_pdfs)} PDF file(s).")

                if st.button("Index selected PDFs", width="stretch", key="btn_add_selected_pdfs"):
                    if st.session_state.rag is None:
                        st.error("Load a model first.")
                    elif not uploaded_pdfs:
                        st.warning("Select at least one PDF file.")
                    else:
                        result = st.session_state.rag.ingest_pdfs_from_upload(
                            uploaded_pdfs
                        )
                        _run_ingest_and_index(result)

            with st.expander("Add PDFs from Zotero", expanded=False):
                _load_zotero_collections()
                if st.session_state.zotero_collections_loaded and st.session_state.zotero_collections:
                    zotero_collection_options = ["All Library"]
                    for coll in st.session_state.zotero_collections:
                        name = coll['name']
                        if coll['parent_id']:
                            parent_name = next((c['name'] for c in st.session_state.zotero_collections
                                            if c['id'] == coll['parent_id']), "Unknown")
                            name = f"{parent_name} > {name}"
                        zotero_collection_options.append(name)

                    selected_zotero_collection = st.selectbox(
                        "Choose which Zotero collection to search",
                        zotero_collection_options,
                        key="zotero_collection_selector"
                    )

                    st.session_state.zotero_collection = None if selected_zotero_collection == "All Library" else selected_zotero_collection.split(" > ")[-1].strip()
                else:
                    st.session_state.zotero_collection = None

                if st.button("Index PDFs from selected collection", width="stretch", key="btn_index_zotero_collection"):
                    if st.session_state.rag is None:
                        st.error("Load a model first.")
                    else:
                        result = st.session_state.rag.ingest_pdfs_from_zotero(
                                st.session_state.zotero_collection
                            )
                        _run_ingest_and_index(result)

            with st.expander("Remove one PDF", expanded=False):
                pdf_name_to_delete = st.text_input(
                    "PDF name to delete, must be in the given PDF source",
                    placeholder="example_paper.pdf",
                    key="pdf_name_delete"
                )
                if st.button("Delete PDF", width="stretch", key="btn_delete_pdf"):
                    if not pdf_name_to_delete.strip():
                        st.warning("Insert the exact PDF name.")
                    elif st.session_state.rag is None:
                        st.error("Load a model first.")
                    else:
                        deleted = st.session_state.rag.delete_pdf_by_title(pdf_name_to_delete.strip())
                        if deleted:
                            st.success(f"Deleted PDF '{pdf_name_to_delete}' from index.")
                            time.sleep(2.0)
                            st.rerun()
                        else:
                            st.warning(f"No PDF named '{pdf_name_to_delete}' found in index.")

            with st.expander("Clear full index", expanded=False):
                st.warning("This operation removes all indexed PDFs and vectors.")
                if st.button("Clear all PDFs", width="stretch", key="btn_clear_all_pdfs"):
                    if st.session_state.rag is None:
                        st.error("Load a model first.")
                    else:
                        st.session_state.rag.clear_index()
                        st.session_state.indexed = False
                        st.session_state.model_loaded = False
                        st.session_state.rag = None
                        st.success("Full index clear requested.")
                        st.rerun()

    st.markdown("---")

    # Reset button
    if st.button("Start Over", width="stretch"):
        zotero_collections = st.session_state.get('zotero_collections', [])
        zotero_collections_loaded = st.session_state.get('zotero_collections_loaded', False)
        st.session_state.clear()
        st.session_state.zotero_collections_loaded = zotero_collections_loaded
        st.session_state.zotero_collections = zotero_collections
        st.rerun()

def _format_time(seconds: float) -> str:
    """Format time in seconds to human-readable string."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:  # less than 1 hour
        minutes = int(seconds // 60)
        secs = seconds % 60
        return f"{minutes}m {secs:.0f}s"
    elif seconds < 86400:  # less than 1 day
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        return f"{hours}h {minutes}m"
    else:  # 1 day or more
        days = int(seconds // 86400)
        hours = int((seconds % 86400) // 3600)
        return f"{days}d {hours}h"


def show_search_tab():
    """Search and highlight tab."""

    st.header("Search Your Library")

    # UI-only copy for the type selector. The hyperparameters themselves live in
    # question_presets, so a preset change is reflected here without editing app.py.
    QUESTION_TYPE_DESCRIPTIONS = {
        'factoid': 'Specific facts or entities',
        'methodology': 'Processes, methods, algorithms',
        'explanation': 'How/why something works',
        'comparison': 'Contrasting different concepts',
        'definition': 'What something is',
        'general': 'General questions',
        'custom': 'Custom settings (fully configurable)',
    }

    # Initialize session state for presets if not exists
    if 'selected_question_type' not in st.session_state:
        st.session_state.selected_question_type = 'general'

    # Query input (moved to top)
    query = st.text_input(
        "Enter your question",
        placeholder="What are the main findings about...",
        key="search_input"
    )

    st.markdown("---")

    col_question_type, col_adjust = st.columns([1, 2], gap="large")

    with col_question_type:
        # Question type selector
        st.subheader("Question Type")

        question_type_options = list(PRESETS.keys())
        question_type_labels = [
            f"{qt.title()} - {QUESTION_TYPE_DESCRIPTIONS[qt]}"
            for qt in question_type_options
        ]

        selected_idx = question_type_options.index(st.session_state.selected_question_type)
        selected_label = st.selectbox(
            "Select question type to apply preset parameters",
            question_type_labels,
            index=selected_idx,
            key="question_type_selector"
        )

        # Extract question type from label
        selected_question_type = question_type_options[question_type_labels.index(selected_label)]
        st.session_state.selected_question_type = selected_question_type

    preset = resolve(selected_question_type)

    with col_adjust:
        # Always show configurable parameters (pre-filled with preset values)
        st.subheader("Adjust Parameters (optional)")
        col_retrieval, col_rerank, col_qa = st.columns(3)

        with col_retrieval:
            retrieval_threshold = st.number_input(
                "1. Retrieval Threshold",
                min_value=0.0, max_value=1.0,
                value=preset['retrieval_threshold'],
                step=0.05,
                help="Stage 1 (Cosine Similarity): Lower = more chunks retrieved (-1.0, 1.0)."
            )

        with col_rerank:
            rerank_threshold = st.number_input(
                "2. Rerank Threshold",
                min_value=0.0, max_value=1.0,
                value=preset['rerank_threshold'],
                step=0.05,
                help="Stage 2 (CrossEncoder): Minimum semantic similarity score (0.0, 1.0)."
            )

        with col_qa:
            qa_score_threshold = st.number_input(
                "3. QA Confidence",
                min_value=0.0, max_value=1.0,
                value=preset['qa_score_threshold'],
                step=0.05,
                help="Stage 3 (QA Model): Confidence threshold."
            )

        col_min_words, col_paraphrases = st.columns(2)
        with col_min_words:
            min_answer_words = st.number_input(
                "Min Answer Words",
                min_value=1, max_value=20,
                value=preset['min_answer_words'],
                step=1,
                help="Minimum words in an answer."
            )

        with col_paraphrases:
            num_paraphrases = st.number_input(
                "Question Paraphrases",
                min_value=0, max_value=10,
                value=2,
                step=1,
                help="Number of question paraphrases to generate (0 = disabled, uses only original question)."
            )

    # User overrides applied on top of the question-type preset by the resolver.
    # The number inputs are pre-filled with preset values, so unchanged fields
    # resolve back to the preset default.
    overrides = {
        'retrieval_threshold': retrieval_threshold,
        'rerank_threshold': rerank_threshold,
        'qa_score_threshold': qa_score_threshold,
        'min_answer_words': min_answer_words,
    }

    # Predefined color presets (defined here for use in search)
    COLOR_PRESETS = {
        'Yellow': (1.0, 1.0, 0.0),
        'Cyan': (0.0, 1.0, 1.0),
        'Orange': (1.0, 0.5, 0.0),
        'Green': (0.5, 1.0, 0.5),
        'Pink': (1.0, 0.7, 0.8),
        'Purple': (0.7, 0.5, 1.0),
        'Custom': None  # Will be set via RGB picker
    }

    # Initialize session state for color
    if 'highlight_color_preset' not in st.session_state:
        st.session_state.highlight_color_preset = 'Yellow'
    if 'highlight_color_custom' not in st.session_state:
        st.session_state.highlight_color_custom = (1.0, 1.0, 0.0)

    def _get_selected_highlight_color():
        color_preset = st.session_state.get('highlight_color_preset', 'Yellow')
        if color_preset == 'Custom':
            return st.session_state.get('highlight_color_custom', (1.0, 1.0, 0.0))
        return COLOR_PRESETS.get(color_preset, (1.0, 1.0, 0.0))

    st.markdown("---")

    # Paraphrase management UI
    if num_paraphrases > 0 and query:
        st.subheader("Question Paraphrases")

        # Initialize session state for paraphrases
        if 'paraphrase_candidates' not in st.session_state:
            st.session_state.paraphrase_candidates = []
        if 'paraphrase_selected' not in st.session_state:
            st.session_state.paraphrase_selected = set()
        if 'paraphrase_query' not in st.session_state:
            st.session_state.paraphrase_query = None

        # Generate paraphrases button
        col_gen, col_more = st.columns([1, 1])

        with col_gen:
            if st.button("Generate Paraphrases", width="stretch"):
                if st.session_state.rag and st.session_state.rag.qa_engine.paraphraser:
                    with st.spinner("Generating paraphrases..."):
                        variations = st.session_state.rag.qa_engine.expand_question(
                            query,
                            num_variations=num_paraphrases * 2  # Generate more for selection
                        )
                        # Store candidates (excluding original)
                        st.session_state.paraphrase_candidates = variations[1:]
                        st.session_state.paraphrase_query = query
                        # Auto-select first N paraphrases
                        st.session_state.paraphrase_selected = set(range(min(num_paraphrases, len(variations) - 1)))
                        st.rerun()
                else:
                    st.error("Paraphraser not available. Question expansion is disabled.")

        with col_more:
            if (st.session_state.paraphrase_candidates and st.button("Generate More", width="stretch")
                    and st.session_state.rag and st.session_state.rag.qa_engine.paraphraser):
                    with st.spinner("Generating more paraphrases..."):
                        # Generate additional paraphrases
                        more_variations = st.session_state.rag.qa_engine.expand_question(
                            query,
                            num_variations=num_paraphrases
                        )
                        # Add new ones to existing (avoid duplicates)
                        existing_set = set(st.session_state.paraphrase_candidates)
                        for var in more_variations[1:]:
                            if var not in existing_set:
                                st.session_state.paraphrase_candidates.append(var)
                                existing_set.add(var)
                        st.rerun()

        # Show original question
        st.markdown("**Original Question (always included):**")
        st.info(f"{query}")

        # Show and allow selection/editing of paraphrases
        if st.session_state.paraphrase_candidates and st.session_state.paraphrase_query == query:
            st.markdown(f"**Generated Paraphrases** ({len(st.session_state.paraphrase_candidates)} available):")
            st.markdown("*Select which paraphrases to use and edit them if needed:*")

            # Track which ones to keep
            selected_indices = set()
            edited_paraphrases = {}

            for i, paraphrase in enumerate(st.session_state.paraphrase_candidates):
                col_check, col_edit = st.columns([0.1, 0.9])

                with col_check:
                    is_selected = st.checkbox(
                        "",
                        value=i in st.session_state.paraphrase_selected,
                        key=f"para_check_{i}",
                        label_visibility="collapsed"
                    )
                    if is_selected:
                        selected_indices.add(i)

                with col_edit:
                    edited = st.text_input(
                        f"Paraphrase {i+1}",
                        value=paraphrase,
                        key=f"para_edit_{i}",
                        label_visibility="collapsed",
                        disabled=not is_selected
                    )
                    if is_selected and edited != paraphrase:
                        edited_paraphrases[i] = edited

            # Update session state
            st.session_state.paraphrase_selected = selected_indices

            # Apply edits
            for i, new_text in edited_paraphrases.items():
                st.session_state.paraphrase_candidates[i] = new_text

            st.markdown(f"**Selected:** {len(selected_indices)} paraphrase(s) + original question")
        elif not st.session_state.paraphrase_candidates:
            st.info("Click 'Generate Paraphrases' to create question variations")
        elif st.session_state.paraphrase_query != query:
            st.warning("Question changed. Click 'Generate Paraphrases' to update.")

    st.markdown("---")
    col_search, col_clear = st.columns([1, 4])
    with col_search:
        search_clicked = st.button("Search", type="primary", width="stretch")
    with col_clear:
        if st.button("Clear Results", width="stretch"):
            st.session_state.search_results = []
            st.session_state.current_answer = 0
            st.rerun()

    if search_clicked and query:
        import time

        # Create separate progress tracking for each stage
        st.markdown("#### Processing Pipeline")

        rerank_status = st.empty()
        rerank_progress_bar = st.progress(0)

        qa_status = st.empty()
        qa_progress_bar = st.progress(0)

        # Track timing for each stage
        rerank_start_time = [None]  # Use list for mutability in nested function
        qa_start_time = [None]

        def rerank_callback(current, total, message):
            if rerank_start_time[0] is None:
                rerank_start_time[0] = time.time()

            if total > 0:
                progress = current / total
                rerank_progress_bar.progress(progress)

                # Calculate time estimates
                elapsed = time.time() - rerank_start_time[0]
                if current > 0 and progress < 1.0:
                    estimated_total = elapsed / progress
                    remaining = estimated_total - elapsed
                    time_info = f"Elapsed: {_format_time(elapsed)} | Remaining: ~{_format_time(remaining)}"
                elif progress >= 1.0:
                    time_info = f"Completed in {_format_time(elapsed)}"
                else:
                    time_info = ""
            else:
                rerank_progress_bar.progress(0)
                time_info = ""

            rerank_status.text(f"Stage 1 - Reranking (merged candidates): {message} {time_info}")

        def qa_callback(current, total, message):
            if qa_start_time[0] is None:
                qa_start_time[0] = time.time()

            if total > 0:
                progress = current / total
                qa_progress_bar.progress(progress)

                # Calculate time estimates
                elapsed = time.time() - qa_start_time[0]
                if current > 0 and progress < 1.0:
                    estimated_total = elapsed / progress
                    remaining = estimated_total - elapsed
                    time_info = f"Elapsed: {_format_time(elapsed)} | Remaining: ~{_format_time(remaining)}"
                elif progress >= 1.0:
                    time_info = f"Completed in {_format_time(elapsed)}"
                else:
                    time_info = ""
            else:
                qa_progress_bar.progress(0)
                time_info = ""

            qa_status.text(f"Stage 2 - QA Extraction: {message} {time_info}")

        try:
            # Prepare selected paraphrases (if any)
            selected_paraphrases = None
            if (num_paraphrases > 0 and
                st.session_state.get('paraphrase_candidates') and
                st.session_state.get('paraphrase_query') == query):
                # Get selected paraphrases
                selected_paraphrases = [query]  # Always include original
                for i in sorted(st.session_state.paraphrase_selected):
                    if i < len(st.session_state.paraphrase_candidates):
                        selected_paraphrases.append(st.session_state.paraphrase_candidates[i])

            # Pass the selected question type, overrides, and paraphrases
            st.session_state.search_results = st.session_state.rag.answer_question(
                question=query,
                question_type=selected_question_type,
                overrides=overrides,
                progress_callback=qa_callback,
                rerank_callback=rerank_callback,
                num_paraphrases=num_paraphrases,
                question_variations=selected_paraphrases
            )

            st.session_state.search_run_id = st.session_state.get('search_run_id', 0) + 1
            st.session_state.highlight_selected = {
                i: True for i in range(len(st.session_state.search_results))
            }

            # Mark completion
            rerank_progress_bar.progress(1.0)
            qa_progress_bar.progress(1.0)
            rerank_status.text("Stage 1 - Reranking (merged candidates): Complete!")
            qa_status.text("Stage 2 - QA Extraction: Complete!")

            st.session_state.search_candidates = getattr(st.session_state.rag, "last_candidates", [])
            st.session_state.current_answer = 0
            st.session_state.current_query = query
        except Exception as e:  # noqa: BLE001 - UI boundary: report, never crash the app
            st.error(f"Search failed: {e}")
            st.session_state.search_results = []
            with st.expander("Show full error"):
                st.exception(e)
        finally:
            time.sleep(1.0)
            rerank_status.empty()
            rerank_progress_bar.empty()
            qa_status.empty()
            qa_progress_bar.empty()

    # Display results
    if st.session_state.search_results:
        if 'search_run_id' not in st.session_state:
            st.session_state.search_run_id = 1
        if ('highlight_selected' not in st.session_state or
            len(st.session_state.highlight_selected) != len(st.session_state.search_results)):
            st.session_state.highlight_selected = {
                i: True for i in range(len(st.session_state.search_results))
            }

        st.success(f"Found {len(st.session_state.search_results)} answers for: *{st.session_state.current_query}*")

        # Navigation and actions
        st.write("")
        st.write("")
        st.write("")
        col1, col2, col3, spaceCol, col4 = st.columns([1, 1, 1, 3, 1], gap="large")

        with col1:
            if st.button("Previous"):
                # Wrap around: if at first result, go to last
                st.session_state.current_answer = (st.session_state.current_answer - 1) % len(st.session_state.search_results)
                st.rerun()

        with col2:
            st.markdown(f"**Answer {st.session_state.current_answer + 1} / {len(st.session_state.search_results)}**")

        with col3:
            if st.button("Next "):
                # Wrap around: if at last result, go to first
                st.session_state.current_answer = (st.session_state.current_answer + 1) % len(st.session_state.search_results)
                st.rerun()

        with spaceCol:
            pass # Empty column for spacing

        with col4:
            if st.button("Open PDF"):
                answer = st.session_state.search_results[st.session_state.current_answer]
                pdf_path = answer.pdf_path
                try:
                    if sys.platform == 'darwin':  # macOS
                        subprocess.run(['open', pdf_path], check=False)
                    elif sys.platform == 'win32':  # Windows
                        os.startfile(pdf_path)
                    else:  # Linux
                        subprocess.run(['xdg-open', pdf_path], check=False)
                    st.success(f"Opened PDF at page {answer.page_number + 1}")
                except OSError as e:
                    st.error(f"Could not open PDF: {e}")

        # Highlight color selection for Highlight Selected
        st.write("")
        st.write("")
        st.subheader("Highlight Color")

        col_preset, col_rgb, col_highlight = st.columns([1, 2, 1], gap="large")

        with col_preset:
            color_preset = st.selectbox(
                "Color Preset",
                options=list(COLOR_PRESETS.keys()),
                index=list(COLOR_PRESETS.keys()).index(st.session_state.highlight_color_preset),
                help="Choose a predefined color or select 'Custom' to use RGB picker",
                key="color_preset_results"
            )
            st.session_state.highlight_color_preset = color_preset

        with col_rgb:
            if color_preset == 'Custom':
                # Show RGB sliders for custom color
                col_r, col_g, col_b = st.columns(3)
                with col_r:
                    r = st.slider("R", 0.0, 1.0, st.session_state.highlight_color_custom[0], 0.05, key="r_results")
                with col_g:
                    g = st.slider("G", 0.0, 1.0, st.session_state.highlight_color_custom[1], 0.05, key="g_results")
                with col_b:
                    b = st.slider("B", 0.0, 1.0, st.session_state.highlight_color_custom[2], 0.05, key="b_results")
                highlight_color_next = (r, g, b)
                st.session_state.highlight_color_custom = highlight_color_next
            else:
                highlight_color_next = COLOR_PRESETS[color_preset]

            # Show color preview
            color_hex_next = rgb_to_hex(highlight_color_next)
            st.markdown(
                f"**Preview:** <span style='background-color: {color_hex_next}; padding: 5px 20px; border: 1px solid #ccc;'>&nbsp;&nbsp;&nbsp;&nbsp;</span>",
                unsafe_allow_html=True
            )

        with col_highlight:
            if st.button("Highlight Selected Articles", width="stretch"):
                highlight_color_override = _get_selected_highlight_color()
                selected_indices = [
                    i for i, selected in st.session_state.highlight_selected.items() if selected
                ]
                selected_answers = [
                    ans for i, ans in enumerate(st.session_state.search_results)
                    if i in selected_indices
                ]

                if not selected_answers:
                    st.warning("Select at least one answer to highlight.")
                else:
                    output_dir = os.path.join(st.session_state.output_dir, "highlighted_results")
                    os.makedirs(output_dir, exist_ok=True)

                    # Group answers by PDF
                    pdfs_answers = {}
                    for answer in selected_answers:
                        answer.color = highlight_color_override
                        if answer.pdf_path not in pdfs_answers:
                            pdfs_answers[answer.pdf_path] = []
                        pdfs_answers[answer.pdf_path].append(answer)

                    highlighted_paths = []
                    failed_highlights = []
                    progress_bar = st.progress(0)

                    for idx, (pdf_path, answers) in enumerate(pdfs_answers.items()):
                        display_title = answers[0].title or os.path.splitext(os.path.basename(pdf_path))[0]
                        safe_title = sanitize_filename(display_title)
                        output_filename = f"{safe_title}_highlighted.pdf"
                        output_path = os.path.join(output_dir, output_filename)

                        result_path = st.session_state.rag.highlight_pdf(answers, output_path)
                        if result_path:
                            highlighted_paths.append(result_path)
                        else:
                            failed_highlights.append(display_title)

                        progress_bar.progress((idx + 1) / len(pdfs_answers))

                    progress_bar.empty()

                    if failed_highlights:
                        st.error(f"Failed to highlight {len(failed_highlights)} PDF(s)")
                        with st.expander("Show failed PDFs"):
                            for pdf_name in failed_highlights:
                                st.text(f"{pdf_name}")

                    if highlighted_paths:
                        st.success(f"Successfully highlighted {len(highlighted_paths)} PDF(s)")
                        with st.expander("Show highlighted files"):
                            for path in highlighted_paths:
                                st.text(f"{os.path.basename(path)}")
                            st.text(f"Location: {output_dir}")

        # Current result display
        st.markdown("---")
        answer = st.session_state.search_results[st.session_state.current_answer]

        st.subheader(f"{answer.title}")
        info_col, select_col = st.columns([5, 1], gap="large")

        with info_col:
            st.markdown(
                f"""**PDF**: {answer.title}<br>
                **Page**: {answer.page_number + 1} |
                **Section**: {answer.section or 'Unknown'} |
                **Retrieval Score**: {answer.retrieval_score:.4f} |
                **Rerank Score**: {answer.rerank_score:.4f} |
                **QA Score**: {answer.score:.4f}""",
                unsafe_allow_html=True
            )

        with select_col:
            selection_key = f"highlight_select_{st.session_state.search_run_id}_{st.session_state.current_answer}"
            highlight_this = st.checkbox(
                "Highlight this answer",
                value=st.session_state.highlight_selected.get(st.session_state.current_answer, True),
                key=selection_key
            )
            st.session_state.highlight_selected[st.session_state.current_answer] = highlight_this

        # Answer display
        st.subheader("Answer")
        st.info(answer.text)

        # Context
        st.subheader("Context (Full Chunk)")
        st.text_area(
            "Full chunk containing the answer",
            value=answer.context,
            height=250,
            disabled=True,
            label_visibility="collapsed",
            key=f"context_{st.session_state.current_answer}"
        )

    elif query and search_clicked:
        st.warning("No results found")
        st.info("""
        **Possible reasons:**
        - QA score threshold too high (try 0.0)
        - Retrieval threshold too low (try increasing to 3.0-5.0)
        - No semantically similar chunks found
        - Question format doesn't match extractive QA style

        **Try:**
        - Lower the QA Score Threshold to 0.0
        - Increase Retrieval Threshold to 3.0 or higher
        - Rephrase as a specific question (e.g., "What is X?" instead of "Tell me about X")
        """)
    elif not query and search_clicked:
        st.info("Enter a question above and click Search to get started")

    # Footer
    st.markdown("---")
    st.markdown(
        "<p style='text-align: center; color: gray;'>Zotero RAG Navigator • Built with Streamlit</p>",
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    main()